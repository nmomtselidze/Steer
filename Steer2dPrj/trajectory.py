import torch
from itertools import accumulate

def compute_trajectory(f, a0, x0, y0, dl, lmax):
    """
    Computes the trajectory using PyTorch tensors and enables gradient calculation.
    """

    # Create the arc-length parameter values.
    l_values = torch.arange(0, lmax, dl)
    dl_tensor = torch.tensor(dl) # dl as tensor

    # Define the state update step. The state is a tuple: (angle, x, y)
    def step(state, l_val):
        angle, x, y = state
        delta_angle = f(l_val) * dl_tensor  # compute incremental change in angle
        angle = angle + delta_angle
        x = x + dl_tensor * torch.cos(angle) # Use previous angle
        y = y + dl_tensor * torch.sin(angle) # Use previous angle
        return (angle, x, y)

    # Accumulate states starting from the initial state.  Ensure initial state is tensor
    states = list(accumulate(l_values, lambda state, l: step(state, l), initial=(torch.tensor(a0), torch.tensor(x0), torch.tensor(y0))))

    # Unpack the states into separate tensors.
    angles, xs, ys = zip(*states)
    angles = torch.stack(angles)
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    return angles, xs, ys

