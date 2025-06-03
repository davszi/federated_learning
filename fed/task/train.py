import torch
from torch.utils.data import DataLoader

from fed.logger import Logger

from .test import test

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def should_stop(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = min(val_loss, self.best_loss)
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
    logger: Logger | None = None,
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

        if logger is not None:
            print(f"Epoch {epoch_index}/{epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Validation Loss: {avg_val_loss:.4f}, "
                  f"Accuracy: {accuracy:.4f}")
            logger.log({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_accuracy": accuracy,
            }, step=epoch_index)

        avg_loss_per_epoch.append(avg_train_loss)
        avg_val_loss_per_epoch.append(avg_val_loss)
        accuracy_per_epoch.append(accuracy)

        if early_stopping and early_stopper.should_stop(avg_val_loss):
            if logger is not None:
                print(f"Early stopping at epoch {epoch_index}")
            break

    return avg_loss_per_epoch, avg_val_loss_per_epoch, accuracy_per_epoch