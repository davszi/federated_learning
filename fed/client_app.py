"""Fed: A Flower / PyTorch app."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, NDArrays, Scalar

from fed.task import Net, get_weights, load_data, set_weights, test, train


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, testloader, id):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.id = id
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config) -> tuple[NDArrays, int, dict[str, Scalar]]:
        # config should have fit configuration from the strategy like LR, batch size, etc. if provided
        set_weights(self.net, parameters)
        avg_loss_per_epoch, avg_val_loss_per_epoch, accuracy_per_epoch = train(
            self.net,
            self.trainloader,
            self.valloader,
            config["local-epochs"],
            self.device,
            early_stopping=config["early-stopping"],
            hyperparameters={
                "optimizer": config["client-optimizer"],
                "learning_rate": config["client-learning-rate"],
            },
        )
        print(f"Client {self.id} finished training with final train loss: {avg_loss_per_epoch[-1]} and val loss: {avg_val_loss_per_epoch[-1]} and val accuracy: {accuracy_per_epoch[-1]}")
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {
                "client_id": self.id,
                "round": config["current_round"],
                "train_loss": avg_loss_per_epoch[-1],
                "val_loss": avg_val_loss_per_epoch[-1],
                "val_accuracy": accuracy_per_epoch[-1],
            },
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        avg_loss, accuracy = test(self.net, self.testloader, self.device)
        return (
            avg_loss,
            len(self.testloader.dataset),
            {
                "client_id": self.id,
                "round": config["current_round"],
                "test_loss": avg_loss,
                "test_accuracy": accuracy,
            },
        )


def client_fn(context: Context):
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    partitioning_strategy = context.run_config.get("partitioning-strategy", "iid")
    partitioning_kwargs = {}

    # Always set partition_by to "label" for strategies that need it
    if partitioning_strategy in ["dirichlet", "pathological", "shard"]:
        partitioning_kwargs["partition_by"] = "label"

    if "dirichlet-alpha" in context.run_config:
        partitioning_kwargs["alpha"] = float(context.run_config["dirichlet-alpha"])

    if "classes-per-partition" in context.run_config:
        partitioning_kwargs["classes_per_partition"] = int(
            context.run_config["classes-per-partition"]
        )

    dataset_name = context.run_config.get("dataset", "uoft-cs/cifar10")

    seed = context.run_config.get("seed", -1)

    trainloader, valloader, testloader = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        dataset_name=dataset_name,
        partitioning_strategy=partitioning_strategy,
        test_size=0.2,
        validate_size=0.1,
        seed=seed if seed != -1 else None,
        batch_size=context.run_config["batch-size"],
        **partitioning_kwargs
    )

    return FlowerClient(net, trainloader, valloader, testloader, partition_id).to_client()


app = ClientApp(
    client_fn,
)