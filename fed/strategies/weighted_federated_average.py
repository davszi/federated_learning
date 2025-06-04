from typing import Union, Optional, Tuple, Dict, List, Callable

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    NDArrays,
    MetricsAggregationFn,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from fed.task import Net, get_weights


class WeightedFedAvg(FedAvg):
    """Federated Averaging strategy that weights updates by both num_epochs and num_examples."""

    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        # Extract weights and number of epochs from client results
        weights_and_factors = []
        for client_proxy, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            num_examples = fit_res.num_examples
            num_epochs = fit_res.metrics.get("num_epochs", 1)

            factor = num_examples * num_epochs
            weights_and_factors.append((weights, factor))

        # Compute total number of epochs
        total_factor = sum(factor for _, factor in weights_and_factors)
        if total_factor == 0:
            print("Warning: Total `factor` is zero. Defaulting to uniform averaging.")
            total_factor = len(weights_and_factors)
            weights_and_factors = [(weights, 1) for weights, _ in weights_and_factors]

        # Perform weighted average by num_epochs
        n_layers = len(weights_and_factors[0][0])
        averaged_weights = []
        for i in range(n_layers):
            weighted_sum = sum(weights[i] * (factor / total_factor)
                               for weights, factor in weights_and_factors)
            averaged_weights.append(weighted_sum)

        # Return new model parameters
        aggregated_parameters = ndarrays_to_parameters(averaged_weights)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}

        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            print("WARNING: No fit_metrics_aggregation_fn provided")

        return aggregated_parameters, metrics_aggregated
