
import numpy as np

class LinearLayer:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = np.random.randn(output_dim, input_dim)
        self.biases = np.random.randn(output_dim)
        self.z = np.zeros(output_dim)
        self.x = np.zeros(input_dim)
        self.grad_weights = np.zeros((output_dim, input_dim))
        self.grad_biases = np.zeros(output_dim)

    def forward(self, x):
        self.x = x
        self.z = np.dot(self.weights, x) + self.biases
        return self.z
    
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
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, x, y_true, y_pred):
        pass
            