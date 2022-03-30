# TensorFlow Details

## Usage

The demo is setup with a simple [CIFAR10](https://flower.dev/docs/quickstart_tensorflow.html) model. The demo can be executed with

```bash
python3 tf/simulation.py
```

## Important Parameters

```python
...
# privacy guarantees for (epsilon, delta)-DP
epsilon = 0.8  # lower is better
delta = 1 / 2e5 # probability of exceeding the privacy budget
l2_norm_clip = 1.5  # max euclidean norm of gradients

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

## PrivateServer Walkthrough

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
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # call the superclass method
        response = super().aggregate_fit(rnd, results, failures)
        aggregated_params = response[0]

        # add noise
        noised_parameters = privacy.server_side_noise(aggregated_params, self.sigma_d)
        return noised_parameters, response[1]
```

Naturally, this can be extended to any other strategy. Make sure to apply the noise as the last step in the `aggregate_fit` function.

(from [tf/server.py](tf/server.py))
  
[^dpsgd]: [DP-SGD explained](https://medium.com/pytorch/differential-privacy-series-part-1-dp-sgd-algorithm-explained-12512c3959a3)
[^dpfl]: [Federated Learning with Differential Privacy: Algorithms and Performance Analysis](https://doi.org/10.48550/arXiv.1911.00222)
[^dpfl2]: [Federated Learning and Differential Privacy: Software tools analysis, the Sherpa.ai FL framework and methodological guidelines for preserving data privacy](https://doi.org/10.48550/arXiv.2007.00914)

## PrivateClient Walkthrough

This client is loosely based off of the [TensorFlow Quickstart](https://flower.dev/docs/quickstart_tensorflow.html) from [flower's docs](https://flower.dev/docs/index.html).

The constructor is pretty self explanatory. It consists of variable assignment, and then computation of `sigma_u` for the added noise in training.

```python
    def __init__(
        self,
        trainset: Union[np.ndarray, np.ndarray],
        testset: Union[np.ndarray, np.ndarray],
        net: tf.keras.Model,
        loss_function: tf.keras.losses.Loss,
        epsilon: float = 10,
        delta: float = 1 / 2e5,
        l2_norm_clip: float = 1.5,
        num_rounds: int = 3,
        min_dataset_size: int = 1e5,
        epochs: int = 1,
        batch_size: int = 256,
        *args,
        **kwargs,
    ) -> None:
        ...
        self.sigma_u = privacy.calculate_sigma_u(
            epsilon=epsilon,
            delta=delta,
            l2_norm_clip=l2_norm_clip,
            num_exposures=num_rounds,
            min_dataset_size=min_dataset_size,
        )
```

Instead of using `model.fit()`, `PrivateClient` manually implements the training steps, allowing the use of the `privacy.noise_and_clip_gradients` function. This is defined in [tf/privacy.py](tf/privacy.py) and handles clipping and noising the model's gradients.

```python
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.net(x, training=True)
            loss_value = self.loss_function(y, logits)
        grads = tape.gradient(loss_value, self.net.trainable_weights)
        grads_prime = privacy.noise_and_clip_gradients(
            grads,
            l2_norm_clip=self.l2_norm_clip,
            sigma=self.sigma_u,
        )
        self.optimizer.apply_gradients(zip(grads_prime, self.net.trainable_weights))
        return loss_value
```

The testing step utilizes the existing `model.evaluate()` method.


Lastly, the `get_parameters`, `set_parameters`, `fit`, and `evaluate` functions are what tie it all together. You can read more about them on [flower's docs](https://flower.dev/docs/quickstart_tensorflow.html).

```python
    def get_parameters(self):
        return self.net.get_weights()

    def set_parameters(self, parameters):
        self.net.set_weights(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.train()
        return self.get_parameters(), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.test()
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}
```

(from [tf/client.py](tf/client.py))

## Model Handling

Using a custom model, and loss function with `flower-dp` is simple. When instantiating `client.PrivateClient`, pass the custom model as the `model` parameter and the loss function you'd like to use as the `loss_function` parameter. For example

```python
import client
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

loss = keras.losses.SparseCategoricalCrossentropy()
net = tf.keras.applications.MobileNetV2(
    x_train.shape[1:],
    classes=10,
    weights=None,
)

client = client.PrivateClient(
    trainset=(x_train, y_train),
    testset=(x_test, y_test),
    net=net,
    loss_function=loss,
)
```

Note that the [Adam optimizer](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam) is default in the `client.PrivateClient` class.

## Privacy Endpoints

The [privacy](tf/privacy.py) class contains the most important privacy calculations. The equations used are taken directly from [Federated Learning with Differential Privacy: Algorithms and Performance Analysis](https://doi.org/10.48550/arXiv.1911.00222).

## Calculations

- `calculate_c(delta: float) -> float`
- `calculate_sigma_d(epsilon: float, delta: float, l2_norm_clip: float, num_exposures: int, num_rounds: int, num_clients: int, min_dataset_size: int) -> float`
- `calculate_sigma_u(epsilon: float, delta: float, l2_norm_clip: float, num_exposures: int, min_dataset_size: int) -> float`
  