import torch


def straight_line(l):
    """Defines curvature and torsion for a straight line."""
    curvature = lambda l: 0.0       #torch.tensor(0.0)
    torsion = lambda l: 0.0         #torch.tensor(0.0)
    return curvature, torsion

def circle(l):
    """Defines curvature and torsion for a circle."""
    curvature = lambda l: 1.0       #torch.tensor(1.0)
    torsion = lambda l: 0.0         #torch.tensor(0.0)
    return curvature, torsion

def helix(l):
    """Defines curvature and torsion for a helix."""
    curvature = lambda l: 1.0       # torch.tensor(1.0)
    torsion = lambda l: l * 0.05
    #torsion = lambda l: torch.tensor(l * 0.05)
    return curvature, torsion

def screw(l):
    """Defines curvature and torsion for a screw curve."""
    curvature = lambda l: l * 0.1
    torsion = lambda l: 0.05        #torch.tensor(0.05)
    return curvature, torsion

def up_arm(l_values):
    """
    Defines curvature and torsion for an up_arm curve.
    Args: l_values (Tensor): The arc length values.
    """
    curvature = lambda l: 1.0       #torch.tensor(1.0)
    torsion = lambda l: 0.1 * l + 0.1
    #torsion = lambda l: torch.tensor(0.1 * l + 0.1)
    return curvature, torsion

def tourch_cube(l):
   """Defines curvature and torsion 0.1 * l**3 + 2 * l**2 + 0.1"""
   curvature = lambda l: 0.1 * l**3 + 2 * l**2 + 0.1
   torsion = lambda l: 1.0          #torch.tensor(1.0)
   return curvature, torsion
