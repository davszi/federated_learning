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
        evaluate_fn=evaluate,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# The `evaluate` function will be called by Flower after every round
def evaluate(
    server_round: int,
    parameters: NDArrays,
    config: dict[str, Scalar],
) -> Optional[tuple[float, dict[str, Scalar]]]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)
    _, testloader = load_data(0, 1)
    set_weights(net, parameters)  # Update model with the latest parameters
    loss, accuracy = test(net, testloader, device)
    print(f"Server-side round {server_round} evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}

# Create ServerApp
app = ServerApp(server_fn=server_fn)