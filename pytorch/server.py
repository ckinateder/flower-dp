import flwr as fl

from typing import List, Tuple, Optional
from io import BytesIO

import numpy as np
from typing import cast
import privacy


class ServerSideNoiseStrategy(fl.server.strategy.FedAvg):
    """Federated average strategy with server side gaussian noise"""

    def __init__(self, target_epsilon: float = 10.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.target_epsilon = target_epsilon

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[fl.common.Weights]:
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        sigma_d = privacy.calculate_sigma_d(
            epsilon=self.target_epsilon, N=self.min_available_clients
        )
        # add noise
        if aggregated_weights is not None:
            noised_weights = list(aggregated_weights)
            for i in range(len(aggregated_weights)):
                if type(aggregated_weights[i]) == fl.common.typing.Parameters:
                    weights = fl.common.parameters_to_weights(aggregated_weights[i])
                    weights = privacy.noise_weights(weights, sigma_d)
                    noised_parameters = fl.common.weights_to_parameters(weights)
                    noised_weights[i] = noised_parameters

            aggregated_weights = tuple(noised_weights)  # convert back
        return aggregated_weights


def main(
    min_clients: int = 3, num_rounds: int = 3, target_epsilon: float = 19.74
) -> None:
    """Run the server
    Args:
        min_clients (int, optional): minimum number of clients to train. Defaults to 3.
        num_rounds (int, optional): number of rounds to run. Defaults to 3.
        target_epsilon (float, optional): epsilon target privacy budget. Defaults to 19.74.
    """
    strategy = ServerSideNoiseStrategy(
        target_epsilon=target_epsilon,
        min_available_clients=min_clients,
        min_fit_clients=min_clients,
    )
    fl.server.start_server(config={"num_rounds": num_rounds}, strategy=strategy)


if __name__ == "__main__":
    main()
