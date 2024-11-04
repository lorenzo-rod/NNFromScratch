from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from feed_forward_network import FeedForwardNetwork, ReLU, Softmax, Sigmoid
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load the MNIST dataset
mnist = fetch_openml("mnist_784", version=1)

# Split the data into features and target
x = mnist.data / 255.0
y = mnist.target.astype(int)
x = x.to_numpy().reshape(-1, 28 * 28)
y = y.to_numpy().astype(int)

# Normalize the data
x = x / 255.0

# Minimum float value in numpy
EPSILON = np.finfo(float).eps

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Convert the labels to one-hot encoding
y_train = np.eye(10)[y_train]  # Shape: (60000, 10)
y_test = np.eye(10)[y_test]  # Shape: (10000, 10)

print("MNIST dataset loaded and normalized.")

# Define the network architecture
LAYER_SIZES = [784, 128, 64, 10]
ACTIVATION_FUNCTIONS = [ReLU(), Sigmoid(), Softmax()]
LEARNING_RATE = 1e-3
N_EPOCHS = 100
BATCH_SIZE = 1
TEST_PERIOD = 1

# Initialize the network
network = FeedForwardNetwork(LAYER_SIZES, ACTIVATION_FUNCTIONS)

# Initialize loss history and timing variables
train_loss_history = np.zeros(N_EPOCHS)
test_loss_history = np.zeros(N_EPOCHS // TEST_PERIOD)
start_time = time.time()

# Initialize acc history
train_acc_history = np.zeros(N_EPOCHS)
test_acc_history = np.zeros(N_EPOCHS // TEST_PERIOD)

# Initilialize max test accuracy
max_test_accuracy = 0

# Training loop
for epoch in range(N_EPOCHS):
    epoch_start_time = time.time()
    loss = 0
    acc = 0

    for i in range(x_train.shape[0] // BATCH_SIZE):
        indices = [i * BATCH_SIZE + j for j in range(BATCH_SIZE)]
        x_batch = x_train[indices].T
        y_batch = y_train[indices].T
        y_pred_batch = network.forward(x_batch)
        network.backward(y_batch, y_pred_batch, LEARNING_RATE)
        loss += -np.sum(y_batch * np.log(y_pred_batch + EPSILON))
        acc += np.sum(np.argmax(y_pred_batch, axis=0) == np.argmax(y_batch, axis=0))
        # Print % of epoch completed
        completition_bar = "" + "=" * int(
            (i + 1) / (x_train.shape[0] // BATCH_SIZE) * 10
        )
        completition_bar += " " * (10 - len(completition_bar))
        print(
            f"[{completition_bar}] {int((i + 1) / (x_train.shape[0] // BATCH_SIZE) * 100)}% of epoch {epoch + 1} completed",
            end="\r",
        )

    train_loss_history[epoch] = loss / x_train.shape[0]
    train_acc_history[epoch] = acc / x_train.shape[0]

    # Time calculation
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    total_elapsed_time = epoch_end_time - start_time
    time_per_epoch = total_elapsed_time / (epoch + 1)
    estimated_time_remaining = time_per_epoch * (N_EPOCHS - (epoch + 1))

    # Print detailed training information
    print(
        f"Epoch {epoch + 1}/{N_EPOCHS} | Loss: {train_loss_history[epoch]:.6f} | "
        f"Accuracy: {train_acc_history[epoch]:.4f} | "
        f"Epoch Time: {epoch_duration:.2f}s | "
        f"Total Time: {total_elapsed_time:.2f}s | "
        f"Estimated Time Left: {estimated_time_remaining:.2f}s"
    )

    # Test the network every TEST_PERIOD epochs
    if (epoch + 1) % TEST_PERIOD == 0:
        y_pred = network.forward_no_grad(x_test.T)
        acc_test = (
            np.sum(np.argmax(y_pred, axis=0) == np.argmax(y_test.T, axis=0))
            / y_test.shape[0]
        )
        print(f"Test Accuracy: {acc_test:.4f}")
        test_acc_history[epoch // TEST_PERIOD] = acc_test
        # Save the highest test accuracy model
        if acc_test > max_test_accuracy:
            max_test_accuracy = acc_test
            model_name = f"mnist_model_{max_test_accuracy:.4f}.pkl"
            with open(model_name, "wb") as f:
                pickle.dump(network, f)


# Plot loss history for training
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, N_EPOCHS + 1), train_loss_history, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.legend()
plt.grid(True)
plt.show()

# Plot accuracy history for training and testing
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, N_EPOCHS + 1), train_acc_history, label="Train Accuracy")
plt.plot(
    np.arange(TEST_PERIOD, N_EPOCHS + 1, TEST_PERIOD),
    test_acc_history,
    label="Test Accuracy",
)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Testing Accuracy Over Time")
plt.legend()
plt.grid(True)
plt.show()

print(f"Maximum Test Accuracy: {max_test_accuracy:.4f}")
