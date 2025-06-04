"""Fed: A Flower / PyTorch app."""
from pathlib import Path
from typing import Optional, Dict, Union, Callable, List, Tuple

import torch
from flwr.common import Context, ndarrays_to_parameters, Scalar, NDArrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from fed.strategies import AdaptiveFederatedOptimization, WeightedFedAvg
from fed.logger import LoggerFactory, Logger
from fed.task import Net, get_weights, load_data, test, set_weights

client_registry = {}


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    run_name = context.run_config["run-name"]
    logger_type = context.run_config["logger-type"]
    logger = LoggerFactory.create(
        logger_type, run_name=run_name, config=context.run_config
    )

    best_model_data = {
        "loss": float("inf"),
        "path": Path('models') / f'{run_name}.pth'
    }

    # Define strategy
    common_strategy_props = {
        "fraction_fit": fraction_fit,
        "fraction_evaluate": 1.0,
        "min_available_clients": 2,
        "initial_parameters": parameters,
        "evaluate_fn": lambda round, parameters, config: evaluate_server_side(
            round, parameters, config, context, logger=logger, best_model=best_model_data
        ),
        "on_fit_config_fn": get_on_fit_config_fn(context.run_config),
        "on_evaluate_config_fn": get_on_evaluate_config_fn(context.run_config),
        "fit_metrics_aggregation_fn": get_fit_metrics_aggregation_fn(logger=logger),
        "evaluate_metrics_aggregation_fn": get_evaluate_metrics_aggregation_fn(logger=logger),
        "accept_failures": False,
    }
    strategy_name = context.run_config["strategy-name"]
    if strategy_name == 'fedavg':
        strategy = FedAvg(**common_strategy_props)
    elif strategy_name == 'adaptive-fed-optimization':
        strategy = AdaptiveFederatedOptimization(
            **common_strategy_props,
            # Server-side hyperparameters
            optimizer_name=context.run_config.get("optimizer", "adam"),
            server_learning_rate=float(
                context.run_config.get("server-learning-rate", 0.01)
            ),
        )
    elif strategy_name == 'fedavg-weighted':
        strategy = WeightedFedAvg(**common_strategy_props)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


def get_on_fit_config_fn(config: Dict[str, Union[int, float, str]]) -> Callable:
    def fit_config_fn(server_round: int) -> Dict[str, Union[int, float, str]]:
        # pass the config to client as metric
        return {**config, "current_round": server_round}

    return fit_config_fn


def get_on_evaluate_config_fn(
    config: Dict[str, Union[int, float, str]] = None,
) -> Callable:
    def evaluate_config_fn(server_round: int) -> Dict[str, Union[int, float, str]]:
        # pass the config to client as metric
        return {**config, "current_round": server_round}

    return evaluate_config_fn


def get_fit_metrics_aggregation_fn(logger: Logger = None):
    def fit_metrics_aggregation_fn(
        fit_metrics: List[Tuple[int, Dict[str, float]]],
    ) -> Dict[str, float]:
        """Aggregate training metrics from clients."""
        if not fit_metrics:
            return {}
        # Process each client's metrics
        for idx, (examples, client_metrics) in enumerate(fit_metrics):
            # Use index as client ID if not provided in metrics
            client_id = f"client_{idx}"
            # Register client if not seen before
            if client_id not in client_registry:
                client_registry[client_id] = {
                    "id": client_id,
                    "rounds_participated": 0,
                    "metrics_history": [],
                }

            client_registry[client_id]["rounds_participated"] += 1
            client_registry[client_id]["metrics_history"].append(
                {**client_metrics, "examples": examples}
            )

            logger.log(
                 {
                    "train_loss": client_metrics.get("train_loss", -1.0),
                    "val_loss": client_metrics.get(
                        "val_loss", -1.0
                    ),
                    f"val_accuracy": client_metrics.get(
                        "val_accuracy", -1.0
                    ),
                },
                name=client_id,
                step=int(client_metrics.get("round", -1)),
            )
        return {"test": 100}

    return fit_metrics_aggregation_fn


def get_evaluate_metrics_aggregation_fn(logger: Logger = None):
    """Return function that aggregates evaluation metrics from your client implementation."""

    def evaluate_metrics_aggregation_fn(
        eval_metrics: List[Tuple[int, Dict[str, float]]],
    ) -> Dict[str, float]:
        """Aggregate evaluation metrics from clients."""
        if not eval_metrics:
            return {}
        for idx, (examples, client_metrics) in enumerate(eval_metrics):
            client_id = f"client_{idx}"

            print(f"Logging steps for {client_id}, step: {int(client_metrics.get('round', -1))}")
            logger.log(
                {
                    "test_loss": client_metrics.get(
                        "test_loss", -1.0
                    ),
                    "test_accuracy": client_metrics.get(
                        "test_accuracy", -1.0
                    ),
                },
                name=client_id,
                step=int(client_metrics.get("round", -1)),
            )

        return {"test1": 100} #TODO implement aggregation
    return evaluate_metrics_aggregation_fn


def evaluate_server_side(
    server_round: int,
    parameters: NDArrays,
    config: dict[str, Scalar],
    context: Context,
    logger: Logger = None,
    best_model: dict = None,
) -> Optional[tuple[float, dict[str, Scalar]]]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)

    dataset_name = context.run_config.get("dataset", "uoft-cs/cifar10")
    seed = context.run_config.get("seed", -1)

    # Load test data from partition 0
    _, _, testloader = load_data(
        partition_id=0,
        num_partitions=1,  # Server uses centralized evaluation
        dataset_name=dataset_name,
        partitioning_strategy="iid",
        test_size=0.99, # 0.0 and 1.0 is not allowed
        validate_size=0.0,
        seed=seed if seed != -1 else None,
        batch_size=context.run_config["batch-size"],
    )

    set_weights(net, parameters)  # Update model with the latest parameters
    avg_loss, accuracy = test(net, testloader, device)

    logger.log(
        {
            "test_loss": avg_loss,
            "test_accuracy": accuracy,
            "fed-round": server_round,
        },
        name='centralized'
    )
    if avg_loss < best_model["loss"]:
        best_model["loss"] = avg_loss
        best_model["path"].parent.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), best_model["path"])
        print(f"[Round {server_round}] New best model saved with test_loss={avg_loss:.4f}")

    return avg_loss, {"test_accuracy": accuracy}


app = ServerApp(server_fn=server_fn)