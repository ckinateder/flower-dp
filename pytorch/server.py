from typing import List, Optional, Tuple

import flwr as fl

import privacy


class ServerSideNoiseStrategy(fl.server.strategy.FedAvg):
    """Federated average strategy with server side gaussian noise"""

    def __init__(self, target_epsilon: float = 10.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.target_epsilon = target_epsilon
        # calculate std of the samples from the normal
        self.sigma_d = privacy.calculate_sigma_d(
            epsilon=self.target_epsilon, N=self.min_available_clients
        )

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[fl.common.Weights]:
        """Call the superclass strategy aggregate_fit and then noise the weights.

        Args:
            rnd (int): current round number
            results (List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]]): results
            failures (List[BaseException]): failure list

        Returns:
            Optional[fl.common.Weights]: computed weights
        """
        # call the superclass method
        aggregated_weights = super().aggregate_fit(rnd, results, failures)

        # add noise
        if aggregated_weights is not None:
            noised_weights = list(aggregated_weights)  # make into list so assignable
            for i in range(len(aggregated_weights)):
                if type(aggregated_weights[i]) == fl.common.typing.Parameters:
                    weights = fl.common.parameters_to_weights(aggregated_weights[i])
                    weights = privacy.noise_weights(
                        weights, self.sigma_d
                    )  # noise weights
                    noised_parameters = fl.common.weights_to_parameters(weights)
                    noised_weights[i] = noised_parameters  # reassign parameters
        return tuple(noised_weights)


def main(
    clients_per_round: int = 3, num_rounds: int = 3, target_epsilon: float = 10.0
) -> None:
    """Run the server
    Args:
        clients_per_round (int, optional): minimum number of clients to train. Defaults to 3.
        num_rounds (int, optional): number of rounds to run. Defaults to 3.
        target_epsilon (float, optional): epsilon target privacy budget. Defaults to 10.
    """
    strategy = ServerSideNoiseStrategy(
        target_epsilon=target_epsilon,
        min_available_clients=clients_per_round,
        min_fit_clients=clients_per_round,
    )
    fl.server.start_server(config={"num_rounds": num_rounds}, strategy=strategy)


if __name__ == "__main__":
    main()
