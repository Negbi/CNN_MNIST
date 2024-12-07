import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from network import Network

n_samples=3000
radius=0.5
center=(0, 0)
# Generate random (x, y) points
X = np.random.uniform(-1, 1, (n_samples, 2))

# Assign labels based on whether the points are inside the circle
y = np.array([1 if (x[0] - center[0]) ** 2 + (x[1] - center[1]) ** 2 <= radius ** 2 else 0 for x in X])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

training_data = [(x, y) for x, y in zip(X_train, y_train)]
test_data = [(x, y) for x, y in zip(X_test, y_test)]
# Network structure: 2 input neurons, 2 hidden neurons, 1 output neuron
net = Network([2, 2, 1], 500)

net.sgd(training_data, rounds=2, mini_batch_size=20, l_rate=1.0, test_data=test_data)

plt.figure(figsize=(6, 6))
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue')
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', alpha=0.6)
plt.legend()
plt.title("2D Circle Classification Dataset")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
