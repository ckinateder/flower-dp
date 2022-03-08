# flower-dp

A custom $(\epsilon$, $\delta)$-DP implementation into the [flower.dev](https://flower.dev/) federated learning framework. `flower-dp` utilizes both the noising before model aggregation FL (NbAFL) method, as well as noising during model aggregation. All the noising is implemented and shown within the code, rather than relying on an outside source. This decision was made around the values of transparency, practical functionality, and abilty to adapt to other machine learning frameworks. One of the features that I wanted to ensure was the ability to pass $\epsilon$ (privacy budget) as a parameter, rather than an arbitrary "noise multiplier" and calculate $\epsilon$ from that. From a practical standpoint, it makes much more sense to be able to pre-emptively ensure a metric of privacy with real meaning.

`flower-dp` is currently just designed for pytorch, but will be expanded to include tensorflow as well.  
Project based on the paper [Federated Learning with Differential Privacy: Algorithms and Performance Analysis](https://doi.org/10.48550/arXiv.1911.00222).

## A Quick Overview of Differential Privacy for Federated Learning

Imagine that you have two datasets $D$ and $D\prime$ that differ in only a single record (e.g., my data) and you interact with the data via a process or mechanism called M (this can be anything, more on this later). We can say that M is $\epsilon$-differentially private if for every possible output x, the probability that this output is observed never differs by more than exp($\epsilon$) between the two scenarios (with and without my data).[^dpsgd]

In our scenerio, the "datasets" would be the weights of the model. So, we add a certain amount of noise to each gradient during gradient descent to ensure that specific users data cannot be extracted but the model can still learn. Because we're adding to the gradients, we must bound them. We do this by clipping using the Euclidian norm. This is controlled by the parameter $C$ or `l2_norm_clip`.  

$\delta$ is the probability of information being accidentially leaked. This value is proportional to the size of the dataset. Typically we'd like to see values of $\delta$ that are less than size of the dataset. For example, if the training dataset was $20000$ rows, $\delta \le 1/20000$.

## Getting Started

Clone the repo at

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

### Running the Demo

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

(from `pytorch/simulation.py`)  

### Using a Different Model




## Links

### Frameworks

- [flower.dev](https://flower.dev/)
- [pytorch](https://pytorch.org/)
- [tensorfow_privacy](https://www.tensorflow.org/responsible_ai/privacy/guide)

### Papers Referenced

- [Federated Learning with Differential Privacy: Algorithms and Performance Analysis](https://doi.org/10.48550/arXiv.1911.00222)
- [Understanding Gradient Clipping in Private SGD: A Geometric Perspective](https://proceedings.neurips.cc/paper/2020/file/9ecff5455677b38d19f49ce658ef0608-Paper.pdf)

### Material for Future Reference

- [AdaCliP: Adaptive Clipping for Private SGD](https://doi.org/10.48550/arXiv.1908.07643)
- [Renyi Differential Privacy](https://arxiv.org/abs/1702.07476v3)
- [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133)
- [Measuring RDP](https://www.tensorflow.org/responsible_ai/privacy/guide/measure_privacy)
- [Compute RDP Parameters](https://www.tensorflow.org/responsible_ai/privacy/tutorials/classification_privacy)
- [rdp_accountant](https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/analysis/rdp_accountant.py)

[^dpsgd]: [DP-SGD explained](https://medium.com/pytorch/differential-privacy-series-part-1-dp-sgd-algorithm-explained-12512c3959a3)
