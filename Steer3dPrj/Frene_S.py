# cirve_examples.py

import torch

def straight_line(l):
    """Defines curvature and torsion for a straight line."""
    return lambda l: 0.0, lambda l: 0.0

def circle(l):
    """Defines curvature and torsion for a circle."""
    return lambda l: 1.0, lambda l: 0.0


def helix(l):
    """Defines curvature and torsion for a helix."""
    return lambda l: 1.0, lambda l: l * 0.05

def screw(l):
    """Defines curvature and torsion for a screw curve."""
    return lambda l: l * 0.1, lambda l: 0.05

def up_arm(l_values):
    """
    Defines curvature and torsion for an up_arm curve.
    Args: l_values (Tensor): The arc length values.
    """
    return lambda l: 1.0, lambda l: 0.1 * l + 0.1
    #return lambda l: 9.0 * l**8, lambda l: 0.1 * l + 0.1

def beveled_up_arm(l_values):
    """
    Defines curvature and torsion for an beveled_up_arm curve.
    Args: l_values (Tensor): The arc length values.
    """
    return lambda l: 9.0 * l**8, lambda l: 0.1 * l + 0.1

def tourch_cube(l):
   """Defines curvature and torsion 0.1 * l**3 + 2 * l**2 + 0.1"""
   return lambda l: 0.1 * l**3 + 2 * l**2 + 0.1, lambda l: 1.0
    
# frenet-serret.py

import torch

def frenet_serret(l_values, curvature_func, torsion_func):
    """
    Calculates a 3D curve using the Frenet-Serret formulas with PyTorch.

    Args:
        l_values: Tensor of arc length values.
        curvature_func: Function that returns curvature as a function of arc length.
        torsion_func: Function that returns torsion as a function of arc length.

    Returns:
        Tensors of x, y, and z coordinates of the curve.
    """

    T = torch.tensor([1., 0., 0.])
    N = torch.tensor([0., 1., 0.])
    B = torch.linalg.cross(T, N)
    P = torch.tensor([0., 0., 0.])

    x_values = [P[0].item()]
    y_values = [P[1].item()]
    z_values = [P[2].item()]

    for i in range(1, len(l_values)):
        l = l_values[i]
        dl = l - l_values[i - 1]
        kappa = curvature_func(l)
        tau = torsion_func(l)

        dT_dl = kappa * N
        dN_dl = -kappa * T + tau * B
        dB_dl = -tau * N

        T = T + dT_dl * dl
        N = N + dN_dl * dl
        B = B + dB_dl * dl

        # Normalize the vectors
        T = T / torch.linalg.norm(T)
        N = N / torch.linalg.norm(N)
        B = B / torch.linalg.norm(B)

        P = P + T * dl
        x_values.append(P[0].item())
        y_values.append(P[1].item())
        z_values.append(P[2].item())

    return torch.tensor(x_values), torch.tensor(y_values), torch.tensor(z_values)
    
# curve_plotting.py

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
    
# main.py

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from frenet_serret import frenet_serret
import Steer3dPrj.curve_examplesFix as curve_examplesFix
import curve_plotting

if __name__ == "__main__":
    l_values = torch.linspace(0, 10 * torch.pi, 500)  # Arc length values

    # Example Usage
    curvature, torsion = curve_examplesFix.helix(l_values)
    x, y, z = frenet_serret(l_values, curvature, torsion)
    curve_plotting.plot_curve(x, y, z, '3D Curve (Helix Example)')

    curvature, torsion = curve_examplesFix.screw(l_values)
    x, y, z = frenet_serret(l_values, curvature, torsion)
    curve_plotting.plot_curve(x, y, z, '3D Curve (Screw Example)')

    curvature, torsion = curve_examplesFix.circle(l_values)
    x_circle, y_circle, z_circle = frenet_serret(l_values, curvature, torsion)
    curve_plotting.plot_curve(x_circle, y_circle, z_circle, "3D Circle")

    curvature, torsion = curve_examplesFix.straight_line(l_values)
    x_straight, y_straight, z_straight = frenet_serr
# =====================================================
The value in 
def helix(l_values):
    """Defines curvature and torsion for a helix."""
    curvature = lambda l: torch.tensor(1.0)
    torsion = lambda l: torch.tensor(0.05)
    return curvature, torsion
    
are constant, not trigonometric functions.

