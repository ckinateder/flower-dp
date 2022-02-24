from multiprocessing import Process
from typing import List

import client
import server

if __name__ == "__main__":
    # global variables
    num_clients = 3  # total number of clients

    # client variables
    epochs = 1  # how many epochs to go through
    batch_size = 256  # batch size for training
    l2_norm_clip = 1.5  # max euclidian norm of the weight gradients
    noise_multiplier = 1.0  # how much noise to add in
    learning_rate = 0.001  # how quickly the model learns

    # server variables
    min_available_clients = 3  # minimum number of clients to train/val
    num_rounds = 3  # number of train/val rounds to go through
    target_epsilon = 19.74  # target privacy guarantee
    # delta is assumed to be `1/num_training_examples`

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
