import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

def plot_curve(x, y, z, title):
    """
    Plots a 3D curve.

    Args:
        x: Tensor of x-coordinates.
        y: Tensor of y-coordinates.
        z: Tensor of z-coordinates.
        title: Title of the plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x.detach().numpy(), y.detach().numpy(), z.detach().numpy())
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.show()
