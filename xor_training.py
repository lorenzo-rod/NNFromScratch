from feed_forward_network import FeedForwardNetwork
from config_params import INPUT_DIM, HIDDEN_DIM, N_LAYERS, OUTPUT_DIM, LEARNING_RATE, N_EPOCHS, BATCH_SIZE
import numpy as np

def xor_gate(x1, x2):
    return x1 ^ x2

network = FeedForwardNetwork(INPUT_DIM, HIDDEN_DIM, N_LAYERS, OUTPUT_DIM)

# Training data
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([xor_gate(x1, x2) for x1, x2 in x_train])

# Training loop
for epoch in range(N_EPOCHS):
    for _ in range(BATCH_SIZE):
        random_index = np.random.choice(x_train.shape[0])
        input = x_train[random_index]
        y_pred = network.forward(input)
        y_true = y_train[random_index]
        network.backward(y_true, y_pred)
    network.update_weights(LEARNING_RATE)
    
# Testing
for x in x_train:
    print(f'{x[0]} xor {x[1]} = {network.forward(x)[0]}')
    
