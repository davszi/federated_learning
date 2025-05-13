"""Fed: A Flower / PyTorch app."""

from typing import Optional, Dict, Union, Callable, List, Tuple

from flwr.common import Context, ndarrays_to_parameters, Scalar, NDArrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from fed.adaptive_federated_optimization import AdaptiveFederatedOptimization
from fed.task import Net, get_weights, load_data, test, set_weights
import torch
import wandb

client_registry = {}


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = AdaptiveFederatedOptimization(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_fn=lambda round, parameters, config: evaluate_server_side(
            round, parameters, config, context
        ),
        on_fit_config_fn=get_on_fit_config_fn(context.run_config),
        on_evaluate_config_fn=get_on_evaluate_config_fn(context.run_config),
        fit_metrics_aggregation_fn=get_fit_metrics_aggregation_fn(),
        evaluate_metrics_aggregation_fn=get_evaluate_metrics_aggregation_fn(),
        accept_failures=False,
        # Server-side hyperparameters
        optimizer_name=context.run_config.get("optimizer", "adam"),
        server_learning_rate=float(
            context.run_config.get("server-learning-rate", 0.01)
        ),
    )
    config = ServerConfig(num_rounds=num_rounds)

    run_name = context.run_config["run-name"]

    wandb.init(
        entity="federated-flower-wanb",
        project="AD-Project",
        name=run_name,
        reinit=True,
        resume="allow",
        config=context.run_config,
    )

    wandb.define_metric("centralized/test_loss", step_metric="centralized/fed-round")
    wandb.define_metric(
        "centralized/test_accuracy", step_metric="centralized/fed-round"
    )
    wandb.define_metric(
        "centralized/client_avg_train_loss", step_metric="centralized/fed-round"
    )
    wandb.define_metric(
        "centralized/client_avg_val_loss", step_metric="centralized/fed-round"
    )
    wandb.define_metric(
        "centralized/client_avg_val_accuracy", step_metric="centralized/fed-round"
    )
    wandb.define_metric(
        "centralized/client_avg_test_loss", step_metric="centralized/fed-round"
    )
    wandb.define_metric(
        "centralized/client_avg_test_accuracy", step_metric="centralized/fed-round"
    )

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


def get_fit_metrics_aggregation_fn():
    def fit_metrics_aggregation_fn(
        fit_metrics: List[Tuple[int, Dict[str, float]]],
    ) -> Dict[str, float]:
        """Aggregate training metrics from clients."""
        if not fit_metrics:
            return {}

        aggregated_metrics = {
            "train_loss": 0.0,
            "val_loss": 0.0,
            "val_accuracy": 0.0,
            "total_examples": 0,
            "participating_clients": 0,
            "total_registered_clients": 0
        }

        for idx, (examples, client_metrics) in enumerate(fit_metrics):
            client_id = client_metrics.get("client_id", -1)
            client_name = f"client_{idx}"
            print(f"Logging steps for {client_id}, step: {int(client_metrics.get('round', -1))}")

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

            wandb.log(
                {
                    f"{client_name}/train_loss": client_metrics.get("train_loss", -1.0),
                    f"{client_name}/val_loss": client_metrics.get("val_loss", -1.0),
                    f"{client_name}/val_accuracy": client_metrics.get(
                        "val_accuracy", -1.0
                    ),
                    f"centralized/fed-round": client_metrics.get("round", -1),
                },
            )

            aggregated_metrics["total_examples"] += examples
            aggregated_metrics["train_loss"] += examples * client_metrics.get("train_loss", 0.0)
            aggregated_metrics["val_loss"] += examples * client_metrics.get("val_loss", 0.0)
            aggregated_metrics["val_accuracy"] += examples * client_metrics.get("val_accuracy", 0.0)

        if aggregated_metrics["total_examples"] > 0:
            aggregated_metrics["train_loss"] /= aggregated_metrics["total_examples"]
            aggregated_metrics["val_loss"] /= aggregated_metrics["total_examples"]
            aggregated_metrics["val_accuracy"] /= aggregated_metrics["total_examples"]

        aggregated_metrics["participating_clients"] = len(fit_metrics)
        aggregated_metrics["total_registered_clients"] = len(client_registry)

        current_round = fit_metrics[0][1].get("round", -1) if fit_metrics else -1

        wandb.log(
            {
                "centralized/client_avg_train_loss": aggregated_metrics["train_loss"],
                "centralized/client_avg_val_loss": aggregated_metrics["val_loss"],
                "centralized/client_avg_val_accuracy": aggregated_metrics["val_accuracy"],
                "centralized/fed-round": current_round,
            }
        )

        return aggregated_metrics

    return fit_metrics_aggregation_fn


def get_evaluate_metrics_aggregation_fn():
    """Return function that aggregates evaluation metrics from your client implementation."""

    def evaluate_metrics_aggregation_fn(
        eval_metrics: List[Tuple[int, Dict[str, float]]],
    ) -> Dict[str, float]:
        """Aggregate evaluation metrics from clients."""
        if not eval_metrics:
            return {}

        aggregated_metrics = {
            "test_loss": 0.0,
            "test_accuracy": 0.0,
            "total_examples": 0,
        }


        for idx, (examples, client_metrics) in enumerate(eval_metrics):
            client_id = client_metrics.get("client_id", -1)
            client_name = f"client_{idx}"
            print(f"Logging steps for {client_id}, step: {int(client_metrics.get('round', -1))}")

            print(f"Logging steps for {client_id}, step: {int(client_metrics.get('round', -1))}")
            wandb.log(
                {
                    f"{client_name}/test_loss": client_metrics.get(
                        "test_loss", -1.0
                    ),
                    f"{client_name}/test_accuracy": client_metrics.get(
                        "test_accuracy", -1.0
                    ),
                },
                step=int(client_metrics.get("round", -1)),
            )

            aggregated_metrics["total_examples"] += examples
            aggregated_metrics["test_loss"] += examples * client_metrics.get(
                "test_loss", 0.0
            )
            aggregated_metrics["test_accuracy"] += examples * client_metrics.get(
                "test_accuracy", 0.0
            )

        if aggregated_metrics["total_examples"] > 0:
            aggregated_metrics["test_loss"] /= aggregated_metrics["total_examples"]
            aggregated_metrics["test_accuracy"] /= aggregated_metrics["total_examples"]

        current_round = eval_metrics[0][1].get("round", -1) if eval_metrics else -1

        wandb.log(
            {
                "centralized/client_avg_test_loss": aggregated_metrics["test_loss"],
                "centralized/client_avg_test_accuracy": aggregated_metrics["test_accuracy"],
                "centralized/fed-round": current_round,
            }
        )

        return aggregated_metrics

    return evaluate_metrics_aggregation_fn


def evaluate_server_side(
    server_round: int,
    parameters: NDArrays,
    config: dict[str, Scalar],
    context: Context,
) -> Optional[tuple[float, dict[str, Scalar]]]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)

    partitioning_strategy = context.run_config.get("partitioning-strategy", "iid")
    dataset_name = context.run_config.get("dataset", "uoft-cs/cifar10")

    partitioning_kwargs = {}
    if "dirichlet-alpha" in context.run_config:
        partitioning_kwargs["alpha"] = float(context.run_config["dirichlet-alpha"])
    if "classes-per-partition" in context.run_config:
        partitioning_kwargs["classes_per_partition"] = int(
            context.run_config["classes-per-partition"]
        )

    seed = context.run_config.get("seed", -1)

    # Load test data from partition 0
    _, _, testloader = load_data(
        partition_id=0,
        num_partitions=1,  # Server uses centralized evaluation
        dataset_name=dataset_name,
        partitioning_strategy=partitioning_strategy,
        test_size=0.99,  # 0.0 and 1.0 is not allowed
        validate_size=0.0,
        seed=seed if seed != -1 else None,
        batch_size=context.run_config["batch-size"],
        partitioning_kwargs=partitioning_kwargs,
    )

    set_weights(net, parameters)  # Update model with the latest parameters
    avg_loss, accuracy = test(net, testloader, device)

    wandb.log(
        {
            "centralized/test_loss": avg_loss,
            "centralized/test_accuracy": accuracy,
            "centralized/fed-round": server_round,
        }
    )
    return avg_loss, {"test_accuracy": accuracy}

import os
os.environ["RAY_DEDUP_LOGS"] = "0"
app = ServerApp(server_fn=server_fn)