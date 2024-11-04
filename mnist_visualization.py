import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.datasets import fetch_openml
import os
from sklearn.model_selection import train_test_split

# Load the MNIST dataset
mnist = fetch_openml("mnist_784", version=1)
x = mnist.data / 255.0
y = mnist.target.astype(int)
x = x.to_numpy().reshape(-1, 28 * 28)
y = y.to_numpy().astype(int)

# Normalize the data
x = x / 255.0

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Load pre-trained model
path = os.path.join("TrainedModels", "mnist_model_0.9002.pkl")
with open(path, "rb") as f:
    network = pickle.load(f)

while True:
    # Select a random test image
    index = np.random.randint(0, x_test.shape[0])
    image = x_test[index].reshape(28, 28)
    true_label = y_test[index]

    # Get prediction
    image_input = x_test[index].reshape(-1, 1)
    prediction = network.forward_no_grad(image_input)
    predicted_label = np.argmax(prediction)
    certainty = np.max(prediction) * 100

    # Plot the image with predicted label and certainty
    plt.imshow(image, cmap="gray")
    plt.title(
        f"Predicted: {predicted_label} (Certainty: {certainty:.2f}%)\nTrue Label: {true_label}"
    )
    plt.axis("off")
    plt.show()
