"""Fed: A Flower / PyTorch app."""
import random
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from flwr.common import NDArrays
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from fed.partitioning import load_partitioned_dataset


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Global cache for FederatedDataset
_fds_cache = {}  # (dataset_name, strategy, num_partitions) -> FederatedDataset


def load_data(
    partition_id: int,
    num_partitions: int,
    dataset_name: str,
    partitioning_strategy: str,
    seed: Optional[int] = None,
    test_size: float = 0.2,
    validate_size: float = 0.0,
    batch_size=20,
    **partitioning_kwargs,
) -> tuple[DataLoader, DataLoader | None, DataLoader]:
    # Set partition_by for label-based strategies if not provided
    if (
        partitioning_strategy in ["dirichlet", "pathological", "shard"]
        and "partition_by" not in partitioning_kwargs
    ):
        partitioning_kwargs["partition_by"] = "label"

    # Create a cache key based on the dataset and partitioning configuration
    cache_key = (
        dataset_name,
        partitioning_strategy,
        num_partitions,
        str(sorted(partitioning_kwargs.items())),
    )

    # Only initialize FederatedDataset once for a given configuration
    global _fds_cache
    if cache_key not in _fds_cache:
        _fds_cache[cache_key] = load_partitioned_dataset(
            dataset_name=dataset_name,
            strategy=partitioning_strategy,
            num_partitions=num_partitions,
            **partitioning_kwargs,
        )

    fds = _fds_cache[cache_key]

    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    # Load the specific partition
    partition = fds.load_partition(partition_id)

    # Divide data on each node: 80% train, 20% test
    train_test_partition = partition.train_test_split(test_size=test_size, seed=seed)
    train_partition = train_test_partition["train"]
    test_partition = train_test_partition["test"].with_transform(
        apply_transforms
    )

    if validate_size > 0.0:
        train_val_partition = train_partition.train_test_split(
            test_size=validate_size, seed=seed
        )
        train_partition = train_val_partition["train"]
        val_partition = train_val_partition["test"].with_transform(
            apply_transforms
        )
        val_loader = DataLoader(
            val_partition, batch_size=batch_size
        )
    else:
        val_loader = None

    train_partition = train_partition.with_transform(apply_transforms)
    trainloader = DataLoader(
        train_partition, batch_size=batch_size, shuffle=True
    )

    testloader = DataLoader(test_partition, batch_size=batch_size)

    return trainloader, val_loader, testloader

def test(
    net: torch.nn.Module,
    testloader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0

    net.to(device)
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()

    accuracy = correct / len(testloader.dataset)
    avg_loss = loss / len(testloader)
    return avg_loss, accuracy

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def should_stop(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

def train(
    net: torch.nn.Module,
    trainloader: DataLoader,
    valloader: DataLoader | None,
    epochs: int,
    device: torch.device,
    log_to_wandb: bool = False,
    early_stopping: bool = False,
    hyperparameters: dict[str, float | str] | None = None,
) -> tuple[list[float], list[float], list[float]]:
    """Train the model and return per-epoch metrics."""
    if hyperparameters is None:
        hyperparameters = {"optimizer": "adam", "learning_rate": 0.01}

    criterion = torch.nn.CrossEntropyLoss().to(device)

    optimizer_type = hyperparameters["optimizer"].lower()
    lr = float(hyperparameters["learning_rate"])

    match optimizer_type:
        case "sgd":
            optimizer = torch.optim.SGD(net.parameters(), lr=lr)
        case "adam":
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        case _:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    net.to(device)
    avg_loss_per_epoch = []
    avg_val_loss_per_epoch = []
    accuracy_per_epoch = []

    if early_stopping:
        early_stopper = EarlyStopper(patience=5, min_delta=0.01)
    for epoch_index in range(epochs):
        net.train()
        running_loss = 0.0
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(trainloader)

        if valloader:
            avg_val_loss, accuracy = test(net, valloader, device)
        else:
            avg_val_loss = accuracy = float("nan")

        if log_to_wandb:
            print(f"Epoch {epoch_index}/{epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Validation Loss: {avg_val_loss:.4f}, "
                  f"Accuracy: {accuracy:.4f}")
            wandb.log({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_accuracy": accuracy,
            }, step=epoch_index)

        avg_loss_per_epoch.append(avg_train_loss)
        avg_val_loss_per_epoch.append(avg_val_loss)
        accuracy_per_epoch.append(accuracy)

        if early_stopping and early_stopper.should_stop(avg_val_loss):
            if log_to_wandb:
                print(f"Early stopping at epoch {epoch_index}")
            break

    return avg_loss_per_epoch, avg_val_loss_per_epoch, accuracy_per_epoch



def get_weights(net) -> NDArrays:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)