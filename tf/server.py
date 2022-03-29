from typing import Dict, List, Optional, Tuple

import flwr as fl
from flwr.common.typing import Parameters, Scalar, Weights

import privacy


class PrivateServer(fl.server.strategy.FedAvg):
    """Federated average strategy with server side gaussian noise"""

    def __init__(
        self,
        epsilon: float,
        delta: float = 1 / 2e5,
        l2_norm_clip: float = 1.5,
        num_rounds: int = None,
        min_dataset_size: int = 1e5,
        *args,
        **kwargs
    ) -> None:
        """Init function
        Args:
            epsilon (float): measures the strength of the privacy guarantee by
                bounding how much the probability of a particular model output
                can vary by including (or excluding) a single training point.
            delta (float, optional): Bounds the probability of the privacy guarantee
                not holding. A rule of thumb is to set it to be less than the
                inverse of the size of the training dataset. Defaults to 1 / 2e5.
            l2_norm_clip (int, optional): l2_norm_clip - clipping threshold for gradients. Defaults to 1.5.
            num_rounds (int, optional): num rounds - number of aggregation times. Must be
                greater than or equal to L. Defaults to None, which is
                then set to value of L if given, or 3.
            num_clients (int, optional): number of clients. Defaults to 3.
            min_dataset_size (int, optional): minimum size of local datasets. Defaults to 1e5.
        """
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.l2_norm_clip = l2_norm_clip
        self.sigma_d = privacy.calculate_sigma_d(
            epsilon=epsilon,
            delta=delta,
            l2_norm_clip=l2_norm_clip,
            num_exposures=num_rounds,
            num_rounds=num_rounds,
            num_clients=self.min_fit_clients,
            min_dataset_size=min_dataset_size,
        )

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Call the superclass strategy aggregate_fit and then noise the weights.
        Args:
            rnd (int): current round number
            results (List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]]): results
            failures (List[BaseException]): failure list
        Returns:
            Tuple[Optional[Parameters], Dict[str, Scalar]]: computed weights
        """
        # call the superclass method
        response = super().aggregate_fit(rnd, results, failures)
        aggregated_params = response[0]

        # add noise
        noised_parameters = privacy.server_side_noise(aggregated_params, self.sigma_d)
        return noised_parameters, response[1]


def main(
    epsilon: float = 10,
    delta: float = 1 / 2e5,
    l2_norm_clip: float = 1.5,
    num_rounds: int = 3,
    min_available_clients: int = 2,
    clients_per_round: int = 2,
    min_dataset_size: int = 1e5,
    host: str = "[::]:8080",
) -> None:
    """Run the server
    Args:
        epsilon (float, optional): measures the strength of the privacy guarantee by
            bounding how much the probability of a particular model output
            can vary by including (or excluding) a single training point. Defaults to 10.
        delta (float, optional): Bounds the probability of the privacy guarantee
            not holding. A rule of thumb is to set it to be less than the
            inverse of the size of the training dataset. Defaults to 1 / 2e5.
        l2_norm_clip (int, optional): l2_norm_clip - clipping threshold for gradients. Defaults to 1.5.
        num_rounds (int, optional): num rounds - number of aggregation times. Must be
            greater than or equal to L. Defaults to 3.
        min_available_clients (int, optional): number of clients. Defaults to 3.
        clients_per_round (int, optional):  number of clients to train per round. Defaults to 3.
        min_dataset_size (int, optional): minimum size of local datasets. Defaults to 1e5.
        host (str, optional): hostname and port to connect to. Defaults to "[::]:8080".
    """
    strategy = PrivateServer(
        min_available_clients=min_available_clients,
        min_fit_clients=clients_per_round,
        epsilon=epsilon,
        delta=delta,
        l2_norm_clip=l2_norm_clip,
        num_rounds=num_rounds,
        min_dataset_size=min_dataset_size,
    )
    fl.server.start_server(host, config={"num_rounds": num_rounds}, strategy=strategy)


if __name__ == "__main__":
    main()
