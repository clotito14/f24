from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Fetch the Fashion MNIST dataset
fashion_mnist = fetch_openml(name="Fashion-MNIST", version=1, as_frame=False)

# Extract data and labels
X, y = fashion_mnist.data, fashion_mnist.target

# Convert labels to integers
y = y.astype(int)

# Define class names for Fashion-MNIST
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

print(X[1])

X = X / 255

print(X[1])