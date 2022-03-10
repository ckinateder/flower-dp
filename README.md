# flower-dp

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white) ![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white) ![nVIDIA](https://img.shields.io/badge/nVIDIA-%2376B900.svg?style=for-the-badge&logo=nVIDIA&logoColor=white)

A custom *(ε, δ)*-DP implementation into the [flower.dev](https://flower.dev/) federated learning framework. `flower-dp` utilizes both the noising before model aggregation FL (NbAFL) method, as well as noising during model aggregation.[^dpfl] All the noising is implemented and shown within the code, rather than relying on an outside source. This decision was made considering the values of transparency, practical functionality, and abilty to adapt to other machine learning frameworks. While researching other frameworks, I found that most all of them were based around passing a "noise multipler" as a parameter and calculating *ε* (the privacy budget) using that multiplier and other parameters. One of the features that I wanted to center for this custom implementation was the ability to pass *ε* as a parameter. I think that being able to ensure *ε* up front rather than an arbitrary "noise multiplier" is very important. From a practical standpoint, it makes much more sense to be able to **pre-emptively ensure a metric of privacy with real meaning.**

`flower-dp` is currently just designed for pytorch, but will be expanded to include tensorflow as well.  
Project based on the paper [Federated Learning with Differential Privacy: Algorithms and Performance Analysis](https://doi.org/10.48550/arXiv.1911.00222).

## A Quick Overview of Differential Privacy for Federated Learning

Imagine that you have two neighboring datasets *x* and *y* and randomisation mechanism *M*. Since they're neighboring, *x* and *y* differ by one value. We can say that *M* is *ε*-differentially private if that, for every run of randomisation mechanism *M(x)*, it's about equally likely to see the same output for every neighboring dataset *y*, and this probabilty is set by *ε*. [^dpfl2]

Assume that

<!-- $$S\subseteq \mathrm{Range}(\mathcal M)$$ -->
<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math={\color{white}\large S\subseteq \mathrm{Range}(\mathcal M)}#gh-dark-mode-only"></div>
<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math={\color{black}\large S\subseteq \mathrm{Range}(\mathcal M)}#gh-light-mode-only"></div>

In other words, *M* preserves *ε*-DP if

<!-- $$ P[\mathcal M (x) \in S] \le \exp(\epsilon) P[\mathcal M (y) \in S] $$ -->
<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math={\color{white}\large P[\mathcal M (x) \in S] \le \exp(\epsilon) P[\mathcal M (y) \in S]}#gh-dark-mode-only"></div>
<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math={\color{black}\large P[\mathcal M (x) \in S] \le \exp(\epsilon) P[\mathcal M (y) \in S]}#gh-light-mode-only"></div>

In our scenario, the "datasets" would be the weights of the model. So, we add a certain amount of noise to each gradient during gradient descent to ensure that specific users data cannot be extracted but the model can still learn. Because we're adding to the gradients, we must bound them. We do this by clipping using the [Euclidian (*L<sup>2</sup>*) norm](https://mathworld.wolfram.com/L2-Norm.html). This is controlled by the parameter *C* or `l2_norm_clip`.  

*δ* is the probability of information being accidentially leaked (*0 ≤ δ ≤ 1*). This value is proportional to the size of the dataset. Typically we'd like to see values of *δ* that are less than the inverse of the size of the dataset. For example, if the training dataset was *20000* rows, *δ ≤ 1 / 20000*. To include this in the general formula,

<!-- $$ P[\mathcal M (x) \in S] \le \exp(\epsilon) P[\mathcal M (y) \in S] + \delta $$ -->
<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math={\color{white}\large P[\mathcal M (x) \in S] \le \exp(\epsilon) P[\mathcal M (y) \in S] %2b \delta}#gh-dark-mode-only"></div>
<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math={\color{black}\large P[\mathcal M (x) \in S] \le \exp(\epsilon) P[\mathcal M (y) \in S] %2b \delta}#gh-light-mode-only"></div>

## Getting Started

To install clone the repo and `cd` into the directory.

```bash
git clone https://github.com/ckinateder/flower-dp.git
cd flower-dp
```

To install the packages, you can use virtualenv (for a lighter setup) or Docker (recommended).

### virtualenv

To setup with a virtual environment, use `virtualenv`.

```bash
virtualenv env
source env/bin/activate
pip3 install -r requirements.txt
```

### Docker

To build and run the docker image

```bash
docker build -t flower-dp:latest .
docker run --rm -it -v `pwd`:`pwd` -w `pwd` --gpus all --network host flower-dp:latest bash
```

Alternatively, the simulation can be run directly, without interactively entering the container.

```bash
docker run --rm -v `pwd`:`pwd` -w `pwd` --gpus all --network host flower-dp:latest python3 pytorch/simulation.py
```

## Usage

The demo is setup with a simple [CIFAR10](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) model. The demo can be executed with

```bash
python3 pytorch/simulation.py
```

### Tweaking Parameters

```python
...
# privacy guarantees for (epsilon, delta)-DP
epsilon = 0.8  # lower is better
delta = 1 / 2e5
l2_norm_clip = 1.5  # max euclidian norm of the weight gradients

# cuda device if available
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# client variables
epochs = 1  # how many epochs to go through
batch_size = 256  # batch size for training
learning_rate = 0.001  # how quickly the model learns
min_dataset_size = 1e5  # minimum training set size

# server variables
num_rounds = 3  # number of train/val rounds to go through
min_available_clients = 3  # minimum number of clients to train/val - `N``
clients_per_round = 3  # number of clients to be selected for each round - `K`
...
```

(from [pytorch/simulation.py](pytorch/simulation.py))  

### PrivateClient Walkthrough

(from [pytorch/client.py](pytorch/client.py))

### PrivateServer Walkthrough

`PrivateServer` is pretty simple. It's a subclass of [`fl.server.strategy.FedAvg`](https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedavg.py), with modifications to the constructor and `aggregate_fit` function. The following arguments have been added:

```python
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
        super().__init__(*args, **kwargs)
        self.sigma_d = privacy.calculate_sigma_d(
            epsilon=epsilon,
            delta=delta,
            l2_norm_clip=l2_norm_clip,
            num_exposures=num_rounds,
            num_rounds=num_rounds,
            num_clients=self.min_fit_clients,
            min_dataset_size=min_dataset_size,
        )
```

These extra arguments (`epsilon`, `delta`, `l2_norm_clip`, `num_rounds`, `min_dataset_size`) are used to compute `self.sigma_d`.[^dpfl]

The `aggregate_fit` function is expanded to include noising. The function's signature is the same as its superclass.

```python
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[fl.common.Parameters]:
        # call the superclass method
        aggregated_weights = super().aggregate_fit(rnd, results, failures)

        # add noise
        noised_weights = privacy.noise_aggregated_weights(
            aggregated_weights, sigma=self.sigma_d
        )
        return noised_weights
```

Naturally, this can be extended to any other strategy. Make sure to apply the noise as the last step in the `aggregate_fit` function.

(from [pytorch/server.py](pytorch/server.py))

### Model Handling

Using a custom model, optimizer, and loss function with `flower-dp` is simple. When instantiating `client.PrivateClient`, pass the custom model as the `model` parameter, the optimizer you'd like to use as the `optimizer` parameter, and the loss function you'd like to use as the `loss_function` parameter. For example

```python
import client
import torch

class Net(torch.nn.Module):
    ...
    # custom model definition here

trainloader = ... # dataloader with training set
testloader = ... # dataloader with testing set
net = Net()
optimizer = torch.optim.Adam(net.parameters())
loss = torch.nn.CrossEntropyLoss()

client = client.PrivateClient(
    trainloader=trainloader,
    testloader=testloader,
    model=net,
    optimizer=optimizer,
    loss_function=loss,
)
```

## Links

### Frameworks

- [flower.dev](https://flower.dev/)
- [pytorch](https://pytorch.org/)
- [tensorfow_privacy](https://www.tensorflow.org/responsible_ai/privacy/guide)

### Material for Future Reference

- [AdaCliP: Adaptive Clipping for Private SGD](https://doi.org/10.48550/arXiv.1908.07643)
- [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133)
  
[^dpsgd]: [DP-SGD explained](https://medium.com/pytorch/differential-privacy-series-part-1-dp-sgd-algorithm-explained-12512c3959a3)
[^dpfl]: [Federated Learning with Differential Privacy: Algorithms and Performance Analysis](https://doi.org/10.48550/arXiv.1911.00222)
[^dpfl2]: [Federated Learning and Differential Privacy: Software tools analysis, the Sherpa.ai FL framework and methodological guidelines for preserving data privacy](https://doi.org/10.48550/arXiv.2007.00914)

<!-- Latex generated from https://editor.codecogs.com/>