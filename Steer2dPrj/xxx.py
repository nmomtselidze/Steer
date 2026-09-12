
import torch

def up_arm(l_values, R=1.0):
    """
    Defines curvature and torsion for an up_arm curve.
    
    Args:
        l_values (Tensor): The arc length values.
        R (float): The radius parameter.
    """
    # Fix 1: Explicitly use R, and ensure it returns a tensor if you are in a PyTorch workflow
    curvature = lambda l: torch.tensor(0.1 / R) if not isinstance(l, torch.Tensor) else 0.1 / R
    
    # Fix 2: Ensure the lambda works seamlessly whether 'l' is a scalar or a tensor
    torsion = lambda l: l * 0.5
    
    return curvature, torsion
