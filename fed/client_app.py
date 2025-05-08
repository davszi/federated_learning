"""Fed: A Flower / PyTorch app."""

import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from fed.task import Net, get_weights, load_data, set_weights, test, train


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, partition_id):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.partition_id = partition_id
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Client {self.partition_id} using device {self.device}")
        self.net.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        print(f"Client {self.partition_id} evaluation loss {loss} / accuracy {accuracy}")
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


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
        partitioning_kwargs["classes_per_partition"] = int(context.run_config["classes-per-partition"])

    dataset_name = context.run_config.get("dataset", "uoft-cs/cifar10")

    print(
        f"Client {partition_id}: Loading data with {partitioning_strategy} partitioning"
        f" and parameters {partitioning_kwargs}"
    )

    trainloader, valloader = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        dataset_name=dataset_name,
        partitioning_strategy=partitioning_strategy,
        **partitioning_kwargs
    )

    local_epochs = context.run_config["local-epochs"]

    return FlowerClient(net, trainloader, valloader, local_epochs, partition_id).to_client()


app = ClientApp(
    client_fn,
)