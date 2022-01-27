import flwr as fl

if __name__ == "__main__":
    strategy = fl.server.strategy.FedAvg()
    fl.server.start_server(config={"num_rounds": 3}, strategy=strategy)
