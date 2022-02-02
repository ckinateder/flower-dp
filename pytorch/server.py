import flwr as fl

from typing import List, Tuple, Optional
from io import BytesIO

import numpy as np
from typing import cast
import privacy


class ServerSideNoiseStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[fl.common.Weights]:
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        sigma_d = privacy.calculate_sigma_d(epsilon=1.5, N=self.min_available_clients)
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


if __name__ == "__main__":
    strategy = ServerSideNoiseStrategy(min_available_clients=3)
    fl.server.start_server(config={"num_rounds": 3}, strategy=strategy)
