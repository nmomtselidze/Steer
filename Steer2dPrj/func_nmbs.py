import torch

def get_df_dl(arg, l, R=1.0):
    """
    Returns the derivative (differential) df/dl for the given digit 'arg'.
    Inputs:
        arg: The digit (0-9)
        l: The variable (torch.tensor)
        R: A scaling parameter
    """
    l = torch.as_tensor(l, dtype=torch.float32)
    
    # We define the mapping such that the result represents the 'digit' 
    # either as a constant output or a functional derivative.
    df_funcs = {
        # 0: Derivative of a constant is 0
        0: lambda l: torch.zeros_like(l),
        
        # 1: f(l) = l/R -> f'(l) = 1/R (where R=1)
        1: lambda l: torch.ones_like(l) / R,
        
        # 2: f(l) = l^2 -> f'(l) = 2l (evaluated at l=1 is 2)
        2: lambda l: 2.0 * torch.ones_like(l),
        
        # 3: f(l) = 3l -> f'(l) = 3
        3: lambda l: 3.0 * torch.ones_like(l),
        
        # 4: f(l) = l^4 -> f'(l) = 4l^3
        4: lambda l: 4.0 * l**3,
        
        # 5: f(l) = 5 * log(l) -> f'(l) = 5/l
        5: lambda l: 5.0 / (l + torch.finfo(torch.float32).eps),
        
        # 6: f(l) = 6 * sin(l) -> f'(l) = 6 * cos(l)
        6: lambda l: 6.0 * torch.cos(l),
        
        # 7: f(l) = 7 * exp(l) -> f'(l) = 7 * exp(l)
        7: lambda l: 7.0 * torch.exp(l),
        
        # 8: Using your 'where' logic for a step-like differential
        8: lambda l: torch.where(l < torch.pi / 2, torch.tensor(8.0), torch.tensor(0.0)),
        
        # 9: f(l) = l^9 -> f'(l) = 9l^8
        9: lambda l: 9.0 * l**8
    }
    
    # Return the differential function; defaults to a polynomial derivative for 9+
    return df_funcs.get(arg, lambda l: 9.0 * l**8)(l)
