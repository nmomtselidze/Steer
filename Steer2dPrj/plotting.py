
import matplotlib.pyplot as plt
import torch

def plot_trajectory(xs, ys, arg, a0):
    """
    Plots the trajectory using matplotlib.
    Accepts torch tensors, converts to numpy for plotting
    """
    # Convert torch tensors to numpy arrays for plotting
    xs_np = xs.detach().cpu().numpy()
    ys_np = ys.detach().cpu().numpy()

    plt.title(f'Graph Plot f(l) = {arg}; start angle = {a0}', fontsize=15, color='blue', fontfamily='serif')
    plt.plot(xs_np, ys_np)
    plt.show()
