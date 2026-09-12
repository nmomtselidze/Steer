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

