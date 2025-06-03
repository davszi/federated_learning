import torch
from torch.utils.data import DataLoader


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
