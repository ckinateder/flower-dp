import flwr as fl

strategy = fl.server.strategy.FedAvg(min_available_clients=3)
fl.server.start_server(config={"num_rounds": 4}, strategy=strategy)
