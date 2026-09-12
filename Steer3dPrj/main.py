# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from frenet_serret import frenet_serret
import curve_examples
import curve_plotting

if __name__ == "__main__":
    # l_values = torch.linspace(0, 10 * torch.pi, 500)  # Arc length values
    l = torch.linspace(0, 10 * torch.pi, 2500, device='cpu', dtype=torch.float32)

    # Example Usage
    curvature, torsion = curve_examples.straight_line(l)
    x_straight, y_straight, z_straight = frenet_serret(l, curvature, torsion)
    curve_plotting.plot_curve(x_straight, y_straight, z_straight, "Straight Line in 3D")

    curvature, torsion = curve_examples.circle(l)
    x_circle, y_circle, z_circle = frenet_serret(l, curvature, torsion)
    curve_plotting.plot_curve(x_circle, y_circle, z_circle, "3D Circle")

    curvature, torsion = curve_examples.helix(l)
    x, y, z = frenet_serret(l, curvature, torsion)
    curve_plotting.plot_curve(x, y, z, '3D Curve (Helix Example)')

    curvature, torsion = curve_examples.screw(l)
    x, y, z = frenet_serret(l, curvature, torsion)
    curve_plotting.plot_curve(x, y, z, '3D Curve (Screw Example)')

    curvature, torsion = curve_examples.up_arm(l)
    x, y, z = frenet_serret(l, curvature, torsion)
    curve_plotting.plot_curve(x, y, z, '3D Curve (up_arm Example)')

    curvature, torsion = curve_examples.up_arm(l)
    x, y, z = frenet_serret(l, curvature, torsion)
    curve_plotting.plot_curve(x, y, z, '3D Curve (beveled_up_arm Example)')

    curvature, torsion = curve_examples.tourch_cube(l)
    x_straight, y_straight, z_straight = frenet_serret(l, curvature, torsion)
    curve_plotting.plot_curve(x_straight, y_straight, z_straight, '3D Curve (Cubic Parabola)')



