
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
    
class FeedForwardNetwork:
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim):
        self.layers = []
        self.layers.append(LinearLayerWithReLU(input_dim, hidden_dim))
        for _ in range(n_layers - 1):
            self.layers.append(LinearLayerWithReLU(hidden_dim, hidden_dim))
        self.layers.append(LinearLayer(hidden_dim, output_dim))
        self.n_backprop_executions = 0
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, y_true, y_pred):
        error = - y_true + y_pred
        self.layers[-1].grad_biases += error
        self.layers[-1].grad_weights += error * self.layers[-1].x
        upstream_gradient = error
        self.layers[-2].linear.grad_biases += np.where(self.layers[-2].linear.z > 0, self.layers[-1].weights, 0) * upstream_gradient
        weights_diagonal = self.layers[-1].weights
        self.layers[-2].linear.grad_weights += np.outer(self.layers[-2].linear.x, np.where(self.layers[-2].linear.z > 0, weights_diagonal * upstream_gradient, 0))
        upstream_gradient = np.where(self.layers[-2].linear.z > 0, weights_diagonal * upstream_gradient, 0)
        for layer in reversed(self.layers[:-2]):
            weights_diagonal = (np.diagonal(self.layers[self.layers.index(layer) + 1].linear.weights)).reshape(-1, 1)
            layer.linear.grad_biases += (np.where(layer.linear.z > 0, weights_diagonal, 0) * upstream_gradient)
            layer.linear.grad_weights += np.outer(layer.linear.x, np.where(layer.linear.z > 0, weights_diagonal * upstream_gradient, 0))
            upstream_gradient = np.where(layer.linear.z > 0, weights_diagonal * upstream_gradient, 0)
        self.n_backprop_executions += 1
            
    def reset_gradients(self):
        self.layers[-1].reset_gradients()
        for layer in self.layers[:-1]:
            layer.linear.reset_gradients()
        self.n_backprop_executions = 0
        
    def update_params(self, learning_rate):
        self.layers[-1].weights -= learning_rate * (self.layers[-1].grad_weights / self.n_backprop_executions)
        self.layers[-1].biases -= learning_rate * (self.layers[-1].grad_biases / self.n_backprop_executions)
        for layer in self.layers[:-1]:
            layer.linear.weights -= learning_rate * (layer.linear.grad_weights / self.n_backprop_executions)
            layer.linear.biases -= learning_rate * (layer.linear.grad_biases / self.n_backprop_executions)
        self.reset_gradients()
        