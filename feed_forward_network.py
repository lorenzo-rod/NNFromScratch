import numpy as np
from abc import ABC, abstractmethod


class Layer(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def forward_no_grad(self, x):
        pass

    @abstractmethod
    def backward(self, y_true, y_pred):
        pass

    @abstractmethod
    def reset_gradients(self):
        pass

    @abstractmethod
    def update_params(self, learning_rate):
        pass


class LinearLayer(Layer):
    def __init__(self, input_dim, output_dim, batch_size=1):
        self._weights = np.random.randn(input_dim, output_dim)
        self._biases = np.random.randn(output_dim, 1)
        self._x = np.zeros((input_dim, batch_size))
        self._grad_weights = np.zeros((input_dim, output_dim))
        self._grad_biases = np.zeros((output_dim, 1))
        self._batch_size = batch_size

    def forward(self, x):
        self._x = x
        return self.forward_no_grad(x)

    def forward_no_grad(self, x):
        return self._weights.T @ x + self._biases

    def backward(self, upstream_gradient):
        self._grad_biases += np.sum(upstream_gradient, axis=1, keepdims=True)
        self._grad_weights += self._x @ upstream_gradient.T
        return self._weights @ upstream_gradient

    def reset_gradients(self):
        self._grad_weights = np.zeros_like(self._grad_weights)
        self._grad_biases = np.zeros_like(self._grad_biases)

    def update_params(self, learning_rate):
        self._weights -= learning_rate * (self._grad_weights / self._batch_size)
        self._biases -= learning_rate * (self._grad_biases / self._batch_size)


class ReLU(Layer):
    def forward(self, z):
        self._z = z
        return self.forward_no_grad(z)

    def forward_no_grad(self, z):
        return np.maximum(0, z)

    def backward(self, upstream_gradient):
        return np.where(self._z > 0, upstream_gradient, 0)

    def reset_gradients(self):
        pass

    def update_params(self, learning_rate):
        pass


class Sigmoid(Layer):
    def forward(self, z):
        self._a = self.forward_no_grad(z)
        return self._a

    def forward_no_grad(self, z):
        return 1 / (1 + np.exp(-z))

    def backward(self, upstream_gradient):
        return upstream_gradient * self._a * (1 - self._a)

    def reset_gradients(self):
        pass

    def update_params(self, learning_rate):
        pass


class Softmax(Layer):
    def forward(self, z):
        return self.forward_no_grad(z)

    def forward_no_grad(self, z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z)

    def backward(self):
        pass

    def reset_gradients(self):
        pass

    def update_params(self, learning_rate):
        pass


class Identity(Layer):
    def forward(self, x):
        return self.forward_no_grad(x)

    def forward_no_grad(self, x):
        return x

    def backward(self):
        pass

    def reset_gradients(self):
        pass

    def update_params(self):
        pass


class FeedForwardNetwork:
    def __init__(self, layer_sizes, activation_functions, batch_size=1):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                LinearLayer(layer_sizes[i], layer_sizes[i + 1], batch_size)
            )
            self.layers.append(activation_functions[i])

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward_no_grad(self, x):
        for layer in self.layers:
            x = layer.forward_no_grad(x)
        return x

    def backward(self, y_true, y_pred, learning_rate):
        upstream_gradient = y_pred - y_true
        for layer in reversed(self.layers[:-1]):
            upstream_gradient = layer.backward(upstream_gradient)
            layer.update_params(learning_rate)
            layer.reset_gradients()
