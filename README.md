# flower-dp

Implementing custom differential privacy into the flower.dev federated learning framework.
**NOTE - this branch is only using pytorch**
## Setup

To setup with a virtual environment, use `virtualenv`.

```bash
virtualenv env
source env/bin/activate
pip3 install -r requirements.txt
```

To build image

```bash
docker build -t flower-dp:latest .
```

To run image

```bash
docker run --rm -it -v `pwd`:`pwd` -w `pwd` --gpus all flower-dp:latest bash
```

## Examples

- pytorch example with `./pytorch.sh`.
- tensorflow example with `./tf.sh`.

## Privacy

Delta is assumed to be `1/num_training_examples`.

## Todo

- Package