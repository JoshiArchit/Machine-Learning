import numpy as np
import matplotlib.pyplot as plt

# Set the random seed for reproducibility (optional)
np.random.seed(42)

# Number of samples for each class
n_samples = 200

# Generate random scattered points for class 0
class_0_x = np.random.normal(3, 1, n_samples)
class_0_y = np.random.normal(3, 1, n_samples)

# Generate random scattered points for class 1
class_1_x = np.random.normal(7, 1, n_samples)
class_1_y = np.random.normal(7, 1, n_samples)

# Combine the features and labels for the dataset
X = np.vstack((np.hstack((class_0_x, class_1_x)), np.hstack((class_0_y, class_1_y)))).T
y = np.hstack((np.zeros(n_samples), np.ones(n_samples)))

# Plot the generated dataset
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='red', label='Class 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='blue', label='Class 1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
