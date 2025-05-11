import torch
import wandb
from fed.task import Net, load_data, test, train

def main():
    config = {
            "epochs": 50,
            "batch_size": 64,
            "lr": 0.001,
            "optimizer": "adam",
            "architecture": "VGG7-mod",
    }

    trainloader, validateloader, testloader = load_data(
        partition_id=0,
        num_partitions=1,
        dataset_name="uoft-cs/cifar10",
        partitioning_strategy="iid",
        batch_size=config["batch_size"],
        test_size=0.2,
        validate_size=0.1,
        partitioning_kwargs={},
    )

    wandb.init(
        entity="federated-flower-wanb",
        project="AD-Project",
        name="centralized-baseline",
        config=config
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print(f"Dataset sizes: train: {len(trainloader.dataset)}, val: {len(validateloader.dataset)}, test: {len(testloader.dataset)}")
    net = Net()

    train(
        net,
        trainloader,
        validateloader,
        config["epochs"],
        device,
        log_to_wandb=True,
        early_stopping=True,
        hyperparameters={
            "optimizer": config["optimizer"],
            "learning_rate": config["lr"],
        },
    )

    avg_test_loss, test_accuracy = test(net, testloader, device)
    print(f"Test Loss: {avg_test_loss}, Test Accuracy: {test_accuracy}")

    wandb.log(
        {
            "centralized/test_loss": avg_test_loss,
            "centralized/test_accuracy": test_accuracy,
        }
    )

if __name__ == "__main__":
    main()