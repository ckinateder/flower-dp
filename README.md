# flower-dp

Implementing custom differential privacy into the flower.dev federated learning framework using pytorch.

## Setup

To setup with a virtual environment, use `virtualenv`.

```bash
virtualenv env
source env/bin/activate
pip3 install -r requirements.txt
```

To build and run the docker image

```bash
docker build -t flower-dp:latest .
docker run --rm -it -v `pwd`:`pwd` -w `pwd` --gpus all flower-dp:latest bash
```

Alternatively, the simulation can be run directly, without interactively entering the container.

```bash
docker run --rm -v `pwd`:`pwd` -w `pwd` --gpus all flower-dp:latest python pytorch/simulation.py
```

## Execution

**Make sure you execute through the container described above.**  

Trying the demo is as simple as running

```bash
python pytorch/simulation.py
```

Experiment with the following variables (in the main function) to learn how each affects the system.

```python
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
```
