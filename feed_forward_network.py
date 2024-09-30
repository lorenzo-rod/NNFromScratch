
import numpy as np

class LinearLayer:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim)
        self.biases = np.random.randn(output_dim, 1)
        self.z = np.zeros((output_dim, 1))
        self.x = np.zeros((input_dim, 1))
        self.grad_weights = np.zeros((input_dim, output_dim))
        self.grad_biases = np.zeros((output_dim, 1))

    def forward(self, x):
        self.x = x
        self.z = np.transpose(self.weights) @ x + self.biases
        return self.z
    
    def reset_gradients(self):
        self.grad_weights = np.zeros_like(self.grad_weights)
        self.grad_biases = np.zeros_like(self.grad_biases)
        
class ReLU:
    def forward(self, z):
        self.z = z
        self.a = np.maximum(0, z)
        return self.a
    
class LinearLayerWithReLU:
    def __init__(self, input_dim, output_dim):
        self.linear = LinearLayer(input_dim, output_dim)
        self.relu = ReLU()
        
    def forward(self, x):
        return self.relu.forward(self.linear.forward(x))
    
class Softmax:
    def forward(self, z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z)
    
class Sigmoid:
    def forward(self, z):
        self.z = z
        self.a = 1 / (1 + np.exp(-z))
        return self.a
    
class FeedForwardNetwork:
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim, is_classification=False):
        self.is_classification = is_classification
        self.layers = []
        self.layers.append(LinearLayerWithReLU(input_dim, hidden_dim))
        for _ in range(n_layers - 1):
            self.layers.append(LinearLayerWithReLU(hidden_dim, hidden_dim))
        self.layers.append(LinearLayer(hidden_dim, output_dim))
        if is_classification:
            self.last_training_layer_index = -2
            if output_dim == 1:
                self.layers.append(Sigmoid())
                self.is_binary_classification = True
            else:
                self.layers.append(Softmax())
                self.is_binary_classification = False
        else:
            self.last_training_layer_index = -1
            self.is_binary_classification = False
        self.n_backprop_executions = 0
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, y_true, y_pred):
        error = y_pred - y_true
        self.layers[self.last_training_layer_index].grad_biases += error
        pad_size = self.layers[self.last_training_layer_index].weights.shape[0] - error.shape[0]
        error = np.pad(error, ((0, pad_size), (0, 0)), mode='edge')
        self.layers[self.last_training_layer_index].grad_weights += error * self.layers[self.last_training_layer_index].x
        upstream_gradient = error
        if self.is_binary_classification or not self.is_classification:
            weights_diagonal = self.layers[self.last_training_layer_index].weights
        else:
            weights_diagonal = np.diagonal(self.layers[self.last_training_layer_index].weights).reshape(-1, 1)
        pad_size = upstream_gradient.shape[0] - weights_diagonal.shape[0]
        weights_diagonal = np.pad(weights_diagonal, ((0, pad_size), (0, 0)), mode='edge')
        self.layers[self.last_training_layer_index - 1].linear.grad_biases += np.where(self.layers[self.last_training_layer_index - 1].linear.z > 0, weights_diagonal * upstream_gradient, 0)
        self.layers[self.last_training_layer_index - 1].linear.grad_weights += np.outer(self.layers[self.last_training_layer_index - 1].linear.x, np.where(self.layers[self.last_training_layer_index - 1].linear.z > 0, weights_diagonal * upstream_gradient, 0))
        upstream_gradient = np.where(self.layers[self.last_training_layer_index - 1].linear.z > 0, weights_diagonal * upstream_gradient, 0)
        for layer in reversed(self.layers[:self.last_training_layer_index - 1]):
            weights_diagonal = (np.diagonal(self.layers[self.layers.index(layer) + 1].linear.weights)).reshape(-1, 1)
            layer.linear.grad_biases += (np.where(layer.linear.z > 0, weights_diagonal, 0) * upstream_gradient)
            layer.linear.grad_weights += np.outer(layer.linear.x, np.where(layer.linear.z > 0, weights_diagonal * upstream_gradient, 0))
            upstream_gradient = np.where(layer.linear.z > 0, weights_diagonal * upstream_gradient, 0)
        self.n_backprop_executions += 1
            
    def reset_gradients(self):
        self.layers[self.last_training_layer_index].reset_gradients()
        for layer in self.layers[:self.last_training_layer_index]:
            layer.linear.reset_gradients()
        self.n_backprop_executions = 0
        
    def update_params(self, learning_rate):
        self.layers[self.last_training_layer_index].weights -= learning_rate * (self.layers[self.last_training_layer_index].grad_weights / self.n_backprop_executions)
        self.layers[self.last_training_layer_index].biases -= learning_rate * (self.layers[self.last_training_layer_index].grad_biases / self.n_backprop_executions)
        for layer in self.layers[:self.last_training_layer_index]:
            layer.linear.weights -= learning_rate * (layer.linear.grad_weights / self.n_backprop_executions)
            layer.linear.biases -= learning_rate * (layer.linear.grad_biases / self.n_backprop_executions)
        self.reset_gradients()
        