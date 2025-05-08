"""Fed: A Flower / PyTorch app."""
from typing import Optional

from flwr.common import Context, ndarrays_to_parameters, Scalar, NDArrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from fed.task import Net, get_weights, load_data, test, set_weights
import torch


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_fn=lambda round, parameters, config: evaluate(
            round, parameters, config, context
        ),
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# The `evaluate` function will be called by Flower after every round
def evaluate(
    server_round: int,
    parameters: NDArrays,
    config: dict[str, Scalar],
    context: Context,
) -> Optional[tuple[float, dict[str, Scalar]]]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)

    partitioning_strategy = context.run_config.get("partitioning-strategy", "iid")
    dataset_name = context.run_config.get("dataset", "uoft-cs/cifar10")

    # Get optional partitioning parameters
    partitioning_kwargs = {}
    if "dirichlet-alpha" in context.run_config:
        partitioning_kwargs["alpha"] = float(context.run_config["dirichlet-alpha"])
    if "classes-per-partition" in context.run_config:
        partitioning_kwargs["classes_per_partition"] = int(context.run_config["classes-per-partition"])

    print(f"Server-side evaluation for round {server_round} using partitioning strategy {partitioning_strategy} on {dataset_name}")

    # Load test data from partition 0
    _, testloader = load_data(
        partition_id=0,
        num_partitions=1,  # Server uses centralized evaluation
        dataset_name=dataset_name,
        partitioning_strategy=partitioning_strategy,
    )

    set_weights(net, parameters)  # Update model with the latest parameters
    loss, accuracy = test(net, testloader, device)
    print(f"Server-side round {server_round} evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}

app = ServerApp(server_fn=server_fn)