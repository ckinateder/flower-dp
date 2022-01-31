import flwr as fl

from typing import List, Tuple, Optional


class ServerSideNoiseStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[fl.common.Weights]:
        aggregated_weights = super().aggregate_fit(rnd, results, failures)

        if aggregated_weights is not None:
            # add noise to the aggregated_weights here
            pass
        return aggregated_weights


# Create strategy and run server
strategy = ServerSideNoiseStrategy(
    # (same arguments as FedAvg here)
)
fl.server.start_server(strategy=strategy)
if __name__ == "__main__":
    strategy = fl.server.strategy.FedAvg()
    fl.server.start_server(config={"num_rounds": 3}, strategy=strategy)
