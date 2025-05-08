"""Fed: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common import NDArrays
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from fed.partitioning import load_partitioned_dataset


class Net(nn.Module):
    """Strict VGG-7 model for CIFAR-10"""

    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Global cache for FederatedDataset
_fds_cache = {}  # (dataset_name, strategy, num_partitions) -> FederatedDataset

def load_data(partition_id: int, num_partitions: int, dataset_name: str = "uoft-cs/cifar10",
              partitioning_strategy: str = "iid", test_size: float = 0.2, seed: float = 42, **partitioning_kwargs):
    """Load partition data with configurable partitioning strategy.

    Args:
        partition_id: ID of the partition to load
        num_partitions: Total number of partitions
        dataset_name: Name of the dataset to load
        partitioning_strategy: Strategy to use for partitioning
        **partitioning_kwargs: Additional kwargs for the partitioner

    Returns:
        A tuple of (trainloader, testloader)
    """
    # Set partition_by for label-based strategies if not provided
    if partitioning_strategy in ["dirichlet", "pathological", "shard"] and "partition_by" not in partitioning_kwargs:
        partitioning_kwargs["partition_by"] = "label"

    # Create a cache key based on the dataset and partitioning configuration
    cache_key = (dataset_name, partitioning_strategy, num_partitions,
                 str(sorted(partitioning_kwargs.items())))

    # Only initialize FederatedDataset once for a given configuration
    global _fds_cache
    if cache_key not in _fds_cache:
        print(f"Creating new FederatedDataset with strategy: {partitioning_strategy}")
        _fds_cache[cache_key] = load_partitioned_dataset(
            dataset_name=dataset_name,
            strategy=partitioning_strategy,
            num_partitions=num_partitions,
            **partitioning_kwargs
        )

    fds = _fds_cache[cache_key]

    # Load the specific partition
    partition = fds.load_partition(partition_id)

    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=test_size, seed=seed)
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(net) -> NDArrays:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)