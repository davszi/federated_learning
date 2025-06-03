import torch

from fed.logger.factory import LoggerFactory
from fed.task import Net, load_data, test, train

def main():
    run_name = "centralized-baseline"
    logger_type = "file"

    config = {
        "epochs": 50,
        "batch_size": 2048,
        "lr": 0.001,
        "optimizer": "adam",
        "architecture": "VGG11",
    }

    trainloader, validateloader, testloader = load_data(
        partition_id=0,
        num_partitions=1,
        dataset_name="uoft-cs/cifar10",
        partitioning_strategy="iid",
        batch_size=config["batch_size"],
        test_size=0.1,
        validate_size=0.2,
        partitioning_kwargs={},
    )

    logger = LoggerFactory.create(logger_type, run_name=run_name, config=config)

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
        logger=logger,
        early_stopping=True,
        hyperparameters={
            "optimizer": config["optimizer"],
            "learning_rate": config["lr"],
        },
    )

    avg_test_loss, test_accuracy = test(net, testloader, device)
    print(f"Test Loss: {avg_test_loss}, Test Accuracy: {test_accuracy}")

    logger.log({
        "test_loss": avg_test_loss,
        "test_accuracy": test_accuracy,
    })

if __name__ == "__main__":
    main()