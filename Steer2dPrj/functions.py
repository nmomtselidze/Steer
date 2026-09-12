import torch

def get_f(arg, R):
    """
    Returns a function f(l) based on the chosen parameter `arg`.
    Now using torch for consistency.
    """
    f_funcs = {
        0: lambda l: pow(torch.exp,(-0.15*l)) * torch.sin(l),
        1: lambda l: 1.0 / R,
        2: lambda l: 2.0 * l + 3.0,
        3: lambda l: 0.5 * l**2 + 2 * l + 1,
        4: lambda l: 0.5 * l**3 + 2 * l**2 + 1,
        5: lambda l: torch.log(l + torch.finfo(torch.float32).eps),  # torch.log
        6: lambda l: torch.sin(l),                                   # torch.sin
        7: lambda l: torch.sin(l) - 0.5,
        8: lambda l: torch.where(l < torch.pi / 2, torch.tensor(torch.pi / 2), torch.tensor(0.0)) # torch.where
    }
    return f_funcs.get(arg, lambda l: 0.5 * l**4 + 2 * l**3 + 4 * l**2  + 1)

'''    
def helix(l_values):
    """Defines curvature and torsion for a helix."""
    curvature = lambda l: torch.tensor(1.0)
    torsion = lambda l: torch.tensor(0.05)
    return curvature, torsion

def screw(l_values):
    """Defines curvature and torsion for a screw curve."""
    curvature = lambda l: l * 0.1
    torsion = lambda l: torch.tensor(0.05)
    return curvature, torsion

def circle(l_values):
    """Defines curvature and torsion for a circle."""
    curvature = lambda l: torch.tensor(1.0 / R)
    return curvature

def straight_line(l_values):
    """Defines curvature and torsion for a straight line."""
    curvature = lambda l: torch.tensor(0.0)
    return curvature

def tourch_cube(l_values):
   """Defines curvature and torsion 0.5 * l**3 + 2 * l**2 + 1"""
   curvature = lambda l: 0.5 * l**3 + 2 * l**2 + 1
   torsion = lambda l: torch.tensor(0.0)
   return curvature, torsion
'''
