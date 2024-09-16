
import numpy as np

class LinearLayer:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = np.random.randn(output_dim, input_dim)
        self.biases = np.random.randn(output_dim)

    def forward(self, x):
        return np.dot(self.weights, x) + self.biases
    
class ReLU:
    def forward(self, x):
        return np.maximum(0, x)
    
class Softmax:
    def forward(self, x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)
    
class FeedForwardNetwork:
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim, is_clasification=False):
        self.layers = []
        self.layers.append(LinearLayer(input_dim, hidden_dim))
        self.layers.append(ReLU())
        for _ in range(n_layers - 1):
            self.layers.append(LinearLayer(hidden_dim, hidden_dim))
            self.layers.append(ReLU())
        self.layers.append(LinearLayer(hidden_dim, output_dim))
        if is_clasification:
            self.layers.append(Softmax())
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
        