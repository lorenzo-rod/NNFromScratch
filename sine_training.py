import time
import numpy as np
import matplotlib.pyplot as plt
from feed_forward_network import FeedForwardNetwork

INPUT_DIM = 1
HIDDEN_DIM = 10
N_LAYERS = 2
OUTPUT_DIM = 1
LEARNING_RATE = 1e-3
N_EPOCHS = 100000
BATCH_SIZE = 8

# Generate training data for the sine wave
x_train = np.linspace(0, 2 * np.pi, 100)  # 100 samples between 0 and 2π
y_train = np.sin(x_train)  # Corresponding sine values

# Reshape data to fit the network's input and output dimensions
x_train = x_train.reshape(-1, 1)  # Each input is a single value (reshape to (100, 1))
y_train = y_train.reshape(-1, 1)  # Each output is a single sine value (reshape to (100, 1))

network = FeedForwardNetwork(INPUT_DIM, HIDDEN_DIM, N_LAYERS, OUTPUT_DIM)

# Initialize loss history and timing variables
loss_history = np.zeros(N_EPOCHS)
start_time = time.time()

# Training loop with detailed information
for epoch in range(N_EPOCHS):
    epoch_start_time = time.time()
    loss = 0

    for _ in range(BATCH_SIZE):
        # Randomly sample from the training data
        random_index = np.random.choice(x_train.shape[0])
        input = x_train[random_index].reshape(INPUT_DIM, 1)
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

# Testing: Predict the sine values on the training data
y_pred = np.zeros_like(y_train)
for i, x in enumerate(x_train):
    input = x.reshape(INPUT_DIM, 1)
    y_pred[i] = network.forward(input)

# Plot the true sine wave and the network's predictions
plt.figure(figsize=(10, 6))
plt.plot(x_train, y_train, label="True Sine Wave", color="blue")
plt.plot(x_train, y_pred, label="Network Prediction", color="red", linestyle='--')
plt.xlabel("Input")
plt.ylabel("Sine Value")
plt.title("True Sine Wave vs. Network Prediction")
plt.legend()
plt.grid(True)
plt.show()
