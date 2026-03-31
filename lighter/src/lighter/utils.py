import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_loss(train, val, *, log=False, title=None):
    xAxis = np.arange(1, len(train)+1)
    plt.figure()
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    
    if title:
        plt.title(title)

    plt.plot(
        xAxis,
        train,
        label='Train Losses'
        )
    if val :
        plt.plot(
            xAxis,
            val,
            label='Val Losses'
            )

    if log :
        plt.yscale('log')

    plt.legend()
    # plt.show()

def plot_decision_boundary(model, dataloader, h=0.02):
    """
    Plot the decision boundary of a trained model using a DataLoader.

    Parameters:
    - model: Trained machine learning model with a predict function.
    - dataloader: DataLoader containing input features and true labels.
    - h: Step size for the mesh grid.

    Returns:
    - None (displays the plot).
    """
    plt.figure()
    unique_classes = np.unique([labels.item() for _, labels in dataloader.dataset])

    cmap_classes = ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA'])  # Background colors for classes

    all_points = []
    for inputs, labels in dataloader:
        all_points.append(torch.cat((inputs, labels.view(-1, 1)), dim=1).numpy())

    all_points = np.concatenate(all_points, axis=0)

    # Create a mesh grid
    x_min, x_max = all_points[:, 0].min() - 0.1, all_points[:, 0].max() + 0.1
    y_min, y_max = all_points[:, 1].min() - 0.1, all_points[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the class for each point in the mesh grid
    Z = model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])).detach().numpy()
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)

    # Plot decision boundary & individual points
    plt.contourf(xx, yy, Z, cmap=cmap_classes, alpha=0.3)
    plt.scatter(all_points[:, 0], all_points[:, 1], c=all_points[:, 2], cmap=cmap_classes,
                edgecolors='k', marker='o', s=30)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary and Class Regions')
    plt.show()
