# flower-dp

Implementing custom differential privacy into the flower.dev federated learning framework.

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
docker run --rm -it -v `pwd`:`pwd` -w `pwd` flower-dp:latest
```

## Examples

- pytorch example with `./pytorch.sh`.
- tensorflow example with `./tf.sh`.

## Todo

- Implement server-side noising
- Validate correct laplacian noise
- Make docker image