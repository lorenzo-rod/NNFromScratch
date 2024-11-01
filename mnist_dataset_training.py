from sklearn.datasets import fetch_openml
import numpy as np
from feed_forward_network import FeedForwardNetwork, ReLU, Softmax, Sigmoid
import time
import matplotlib.pyplot as plt

# Fetch MNIST from openml
mnist = fetch_openml("mnist_784")
x = mnist.data / 255.0
y = mnist.target.astype(int)

# Split the dataset
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=101
)

# Flatten the images (28x28) into vectors of length 784 to feed into your network
x_train = x_train.to_numpy().reshape(-1, 28 * 28)  # Shape: (60000, 784)
x_test = x_test.to_numpy().reshape(-1, 28 * 28)  # Shape: (10000, 784)

# Normalize the input data
x_train = (x_train - np.mean(x_train)) / np.std(x_train)
x_test = (x_test - np.mean(x_train)) / np.std(x_train)

# Convert the labels to one-hot encoding
y_train = np.eye(10)[y_train]  # Shape: (60000, 10)
y_test = np.eye(10)[y_test]  # Shape: (10000, 10)

# Define network parameters
LAYER_SIZES = [784, 512, 256, 128, 10]
ACTIVATION_FUNCTIONS = [ReLU(), ReLU(), ReLU(), Softmax()]
LEARNING_RATE = 1e-6
N_EPOCHS = 4000
BATCH_SIZE = 16

# Initialize your neural network
network = FeedForwardNetwork(LAYER_SIZES, ACTIVATION_FUNCTIONS, BATCH_SIZE)

# Initialize loss history and timing variables
loss_history = np.zeros(N_EPOCHS)
accuracy_history = np.zeros(N_EPOCHS)
start_time = time.time()
test_accuracy_history = np.zeros(N_EPOCHS // 10)

max_accuracy = 0

for epoch in range(N_EPOCHS):
    epoch_start_time = time.time()
    loss = 0
    n_correct = 0

    # for _ in range(BATCH_SIZE):
    #     n_index = np.random.choice(x_train.shape[0])
    #     input = x_train[n_index].T.reshape(-1, 1)
    #     y_pred = network.forward(input)
    #     y_true = y_train[n_index].T.reshape(-1, 1)
    #     network.backward(y_true, y_pred, LEARNING_RATE)
    #     loss += np.sum(-y_true * np.log(y_pred + 1e-8))
    #     # Count correct predictions
    #     if np.argmax(y_pred) == np.argmax(y_true):
    #         n_correct += 1

    for _ in range(60000 // BATCH_SIZE):
        batch_indices = np.random.choice(x_train.shape[0], BATCH_SIZE)
        x_batch = x_train[batch_indices].T
        y_batch = y_train[batch_indices].T
        y_pred_batch = network.forward(x_batch)
        network.backward(y_batch, y_pred_batch, LEARNING_RATE)
        loss += np.sum(-y_batch * np.log(y_pred_batch + 1e-8))
        n_correct += np.sum(
            np.argmax(y_pred_batch, axis=0) == np.argmax(y_batch, axis=0)
        )

    loss_history[epoch] = loss / (60000 // BATCH_SIZE)
    accuracy_history[epoch] = n_correct / 60000

    # Time calculation
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    total_elapsed_time = epoch_end_time - start_time
    time_per_epoch = total_elapsed_time / (epoch + 1)
    estimated_time_remaining = time_per_epoch * (N_EPOCHS - (epoch + 1))

    # Print detailed training information
    print(
        f"Epoch {epoch + 1}/{N_EPOCHS} | Loss: {loss_history[epoch]:.6f} | "
        f"Epoch Time: {epoch_duration:.2f}s | "
        f"Total Time: {total_elapsed_time:.2f}s | "
        f"Estimated Time Left: {estimated_time_remaining:.2f}s | "
        f"Training Accuracy: {accuracy_history[epoch] * 100:.2f}%"
    )

    # Evaluate the model on the test set every 10 epochs
    if (epoch + 1) % 10 == 0:
        correct_predictions = 0
        for i in range(x_test.shape[0]):
            input = x_test[i].reshape(LAYER_SIZES[0], 1)
            output = network.forward_no_grad(input)
            predicted_label = np.argmax(output)
            true_label = np.argmax(y_test[i])
            if predicted_label == true_label:
                correct_predictions += 1

        accuracy = correct_predictions / x_test.shape[0]
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        if accuracy > max_accuracy:
            max_accuracy = accuracy
        test_accuracy_history[epoch // 10] = accuracy


# Plot loss history and test history
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, N_EPOCHS + 1), loss_history, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Test Loss Over Time")
plt.legend()
plt.grid(True)
plt.show()

# Plot accuracy history
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, N_EPOCHS + 1), accuracy_history, label="Training Accuracy")
plt.plot(np.arange(1, N_EPOCHS + 1, 10), test_accuracy_history, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Test Accuracy Over Time")
plt.legend()
plt.grid(True)
plt.show()

# Evaluate the model
correct_predictions = 0
for i in range(x_test.shape[0]):
    input = x_test[i].reshape(LAYER_SIZES[0], 1)
    output = network.forward_no_grad(input)
    predicted_label = np.argmax(output)
    true_label = np.argmax(y_test[i])
    if predicted_label == true_label:
        correct_predictions += 1

accuracy = correct_predictions / x_test.shape[0]
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Maximum Test Accuracy: {max_accuracy * 100:.2f}%")
