from typing import Union, Optional, Tuple, Dict, List, Callable

import numpy as np
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    NDArrays,
    MetricsAggregationFn,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from fed.task import Net, get_weights


class AdaptiveFederatedOptimization(FedAvg):
    """Implements Adaptive Federated Optimization Strategy.

    This strategy extends FedAvg by incorporating adaptive optimization techniques
    at the server level, such as AdaGrad, Adam, and Yogi, to improve convergence.

    Paper: Reddi et al. "Adaptive Federated Optimization" (https://arxiv.org/abs/2003.00295)
    """

    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, dict[str, Scalar]],
                Optional[tuple[float, dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        optimizer_name: str = "adam",  # Options: "adagrad", "adam", "yogi"
        server_learning_rate: float = 0.01,
        epsilon: float = 1e-3,
        tau: float = 0.001,
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

        self.optimizer_name = optimizer_name.lower()
        self.server_learning_rate = server_learning_rate
        self.epsilon = epsilon
        self.tau = tau
        match self.optimizer_name:
            case "adagrad":
                self.beta1 = 0.0
                self.beta2 = np.inf  # not used for Adagrad
            case "adam":
                self.beta1 = 0.9
                self.beta2 = 0.99
            case "yogi":
                self.beta1 = 0.9
                self.beta2 = 0.99
            case _:
                raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        self.current_weights: Optional[NDArrays] = None
        self.m_t: Optional[List[np.ndarray]] = None  # First moment
        self.v_t: Optional[List[np.ndarray]] = None  # Second moment

    def __repr__(self) -> str:
        return f"AdaptiveFedOpt(optimizer={self.optimizer_name})"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        if self.initial_parameters is not None:
            ndarrays = parameters_to_ndarrays(self.initial_parameters)
        else:
            raise ValueError(
                "Initial parameters must be provided for Adaptive Federated Optimization."
            )

        self.current_weights = ndarrays
        self.m_t = [np.zeros_like(w) for w in ndarrays]
        self.v_t = [np.zeros_like(w) for w in ndarrays]

        return ndarrays_to_parameters(ndarrays)

    def _compute_pseudo_gradient(self, avg_weights: NDArrays) -> NDArrays:
        """Compute the pseudo-gradient from client updates."""
        if self.current_weights is None:  # First round
            self.current_weights = avg_weights
            return [np.zeros_like(w) for w in avg_weights]  # No update on first round

        # Compute pseudo-gradient (negative of model update)
        # g_t = x_t - weighted_avg_updates
        pseudo_gradient = []
        for current, avg in zip(self.current_weights, avg_weights):
            pseudo_gradient.append(current - avg)

        return pseudo_gradient

    def _apply_adaptive_optimization(
        self, pseudo_gradient: List[np.ndarray]
    ) -> Parameters:
        """Apply the chosen adaptive optimization method."""
        new_weights = []

        for i, (w, g, m, v) in enumerate(
            zip(self.current_weights, pseudo_gradient, self.m_t, self.v_t)
        ):
            # Update first moment estimate (momentum)
            self.m_t[i] = self.beta1 * m + (1 - self.beta1) * g

            # Update second moment based on optimizer
            g_squared = g * g
            match self.optimizer_name.lower():
                case "adagrad":
                    self.v_t[i] = v + g_squared
                case "adam":
                    self.v_t[i] = self.beta2 * v + (1 - self.beta2) * g_squared
                case "yogi":
                    self.v_t[i] = v - (1 - self.beta2) * g_squared * np.sign(
                        v - g_squared
                    )
                case _:
                    raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

            step_size = self.server_learning_rate / (
                np.sqrt(self.v_t[i]) + self.epsilon
            )
            new_w = w - step_size * self.m_t[i]
            new_weights.append(new_w)

        self.current_weights = new_weights

        return ndarrays_to_parameters(new_weights)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        fed_avg_parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round=server_round,
            results=results,
            failures=failures,
        )
        if fed_avg_parameters_aggregated is None:
            print("FedAvg aggregation failed, failures:", failures)
            return None, {}

        fed_avg_weights_aggregated = parameters_to_ndarrays(
            fed_avg_parameters_aggregated
        )

        pseudo_gradient = self._compute_pseudo_gradient(fed_avg_weights_aggregated)

        parameters_aggregated = self._apply_adaptive_optimization(pseudo_gradient)

        return parameters_aggregated, metrics_aggregated