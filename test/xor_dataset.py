import numpy as np

def create_xor_dataset(num_samples_per_class=100, noise=0.1):
    """
    Create a larger dataset simulating XOR relationship with two classes and two input variables.

    Parameters:
    - num_samples_per_class (int): Number of samples per class in the dataset.
    - noise (float): Amount of noise to add to the dataset.

    Returns:
    - X (numpy array): Input features.
    - y (numpy array): Output labels.
    """
    np.random.seed(42)

    # Define the four centroids, and associated labels
    centroids = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    labels = [0, 1, 1, 0]
    
    # Generate samples around each centroid
    X = np.zeros((num_samples_per_class * len(centroids), 2))
    y = np.zeros((num_samples_per_class * len(centroids),))

    for i, centroid in enumerate(centroids):
        # Generate samples around the centroid with some noise
        X[i*num_samples_per_class:(i+1)*num_samples_per_class, :] = centroid + noise * np.random.randn(num_samples_per_class, 2)
        y[i*num_samples_per_class:(i+1)*num_samples_per_class] = labels[i]

    # Shuffle the dataset
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    return X, y
