from multiprocessing import Process
from typing import List

import client
import server

# convert to cli args
if __name__ == "__main__":
    # global variables
    num_clients = 6

    # client variables
    epochs = 3
    batch_size = 32
    l2_norm_clip = 1.5
    noise_multiplier = 0.3
    learning_rate = 0.001

    # server variables
    min_available_clients = 3
    num_rounds = 10
    target_epsilon = 19.74

    # create server process
    server_process = Process(
        target=server.main,
        args=(
            min_available_clients,
            num_rounds,
            target_epsilon,
        ),
    )
    server_process.start()

    # create client processes
    client_processes: List[Process] = []
    for c in range(num_clients):
        client_processes.append(
            Process(
                target=client.main,
                args=(
                    epochs,
                    batch_size,
                    l2_norm_clip,
                    noise_multiplier,
                    learning_rate,
                ),
            )
        )
    # start client processes
    for c in client_processes:
        c.start()

    # finally join
    for c in client_processes:
        c.join()
    server_process.join()
