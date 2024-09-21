import time
import numpy as np
import matplotlib.pyplot as plt
from feed_forward_network import FeedForwardNetwork
from config_params import INPUT_DIM, HIDDEN_DIM, N_LAYERS, OUTPUT_DIM, LEARNING_RATE, N_EPOCHS, BATCH_SIZE

def xor_gate(x1, x2):
    return x1 ^ x2

network = FeedForwardNetwork(INPUT_DIM, HIDDEN_DIM, N_LAYERS, OUTPUT_DIM)

# Training data
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([xor_gate(x1, x2) for x1, x2 in x_train])

# Initialize loss history and timing variables
loss_history = np.zeros(N_EPOCHS)
start_time = time.time()

# Training loop with detailed information
for epoch in range(N_EPOCHS):
    epoch_start_time = time.time()
    loss = 0

    for _ in range(BATCH_SIZE):
        random_index = np.random.choice(x_train.shape[0])
        input = np.transpose(x_train[random_index]).reshape(INPUT_DIM, 1)
        y_pred = network.forward(input)
        y_true = y_train[random_index]
        network.backward(y_true, y_pred)
        loss += np.sum((y_true - y_pred) ** 2)
    
    network.update_params(LEARNING_RATE)
    loss_history[epoch] = loss / BATCH_SIZE

    # Time calculation
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    total_elapsed_time = epoch_end_time - start_time
    time_per_epoch = total_elapsed_time / (epoch + 1)
    estimated_time_remaining = time_per_epoch * (N_EPOCHS - (epoch + 1))

    # Print detailed training information
    print(f"Epoch {epoch + 1}/{N_EPOCHS} | Loss: {loss_history[epoch]:.6f} | "
          f"Epoch Time: {epoch_duration:.2f}s | "
          f"Total Time: {total_elapsed_time:.2f}s | "
          f"Estimated Time Left: {estimated_time_remaining:.2f}s")

# Plot loss history
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, N_EPOCHS + 1), loss_history, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.legend()
plt.grid(True)
plt.show()

# Testing
for x in x_train:
    input = np.transpose(x).reshape(INPUT_DIM, 1)
    print(f'{x[0]} xor {x[1]} = {network.forward(input)[0]}')
