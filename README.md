# flower-dp

A custom (ǫ, δ)-DP implementation into the [flower.dev](https://flower.dev/) federated learning framework. `flower-dp` utilizes both the noising before model aggregation FL (NbAFL) method, as well as noising during model aggregation. All the noising is implemented and shown within the code, rather than relying on an outside source. This decision was made around the values of transparency, practical functionality, and abilty to adapt to other machine learning frameworks. One of the features that I wanted to ensure was the ability to pass epsilon (privacy budget) as a parameter, rather than an arbitrary "noise multiplier" and calculate epsilon from that. From a practical standpoint, it makes much more sense to be able to pre-emptively ensure a set metric of privacy with real meaning.

`flower-dp` is currently just designed for pytorch, but will be expanded to include tensorflow as well.  
Project based on the paper [Federated Learning with Differential Privacy: Algorithms and Performance Analysis](https://doi.org/10.48550/arXiv.1911.00222).

## Getting Started

Clone the repo at

```bash
git clone https://github.com/ckinateder/flower-dp.git
cd flower-dp
```

To install the packages, you can use virtualenv (for a lightweight setup) or Docker (recommended for continued use).

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

**Make sure you execute through the container described above.**  

Trying the demo is as simple as running

```bash
python3 pytorch/simulation.py
```

Experiment with the following variables (in the main function) to learn how each affects the system.

```python

```

## Explanations

Number of exposures is assumed to equal number of rounds.

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