# flower-dp

Implementing custom differential privacy into the flower.dev federated learning framework.

## Setup

To setup with a virtual environment, use `virtualenv`.

```bash
virtualenv env
source env/bin/activate
pip3 install -r requirements.txt
```

## Examples

- pytorch example with `./pytorch.sh`.
- tensorflow example with `./tf.sh`.

## Todo

- Implement server-side noising
- Validate correct laplacian noise
- Make docker image