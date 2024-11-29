from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np

# Fetch the Fashion MNIST dataset
fashion_mnist = fetch_openml(name="Fashion-MNIST", version=1, as_frame=False)

# Extract data and labels
X, y = fashion_mnist.data, fashion_mnist.target

# Convert labels to integers
y = y.astype(int)

# Normalize pixel values (optional: scales data to 0-1 range)
X = X / 255.0

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verify the shapes
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test labels shape: {y_test.shape}")