import torch

def get_differential_value(arg, l_val, R=1.0):
    """
    Computes the derivative (differential) of f(l) at l_val 
    using PyTorch Autograd.
    """
    # Create a tensor that tracks gradients
    l = torch.tensor([float(l_val)], requires_grad=True)
    
    # Define the original functions f(l)
    f_funcs = {
        0: lambda l: torch.exp(-0.15 * l) * torch.sin(l),
        1: lambda l: l / R,
        2: lambda l: l**2 + 3.0,
        3: lambda l: 0.5 * l**2 + 2 * l + 1,
        4: lambda l: 0.5 * l**3 + 2 * l**2 + 1,
        5: lambda l: torch.log(l + 1e-7),
        6: lambda l: torch.sin(l),
        7: lambda l: torch.sin(l) - 0.5,
        8: lambda l: torch.where(l < torch.pi / 2, (torch.pi / 2) * l, torch.tensor(0.0)),
        9: lambda l: 0.5 * l**4 + 2 * l**3 + 4 * l**2 + 1
    }
    
    # Get the function based on the digit
    f_l = f_funcs.get(arg, f_funcs[9])(l) 
    print(f_l)
   
    
    # Compute the gradient (the differential df/dl)
    f_l.backward()
    
    return l.grad.item()

# Example: Get the differential for the digit 2 at l=1.0
# f(l) = l^2 + 3 -> f'(l) = 2l. At l=1, result should be 2.0.
print(f"Differential for digit 9: {get_differential_value(9, 1.0)}")

# https://gemini.google.com/app/dffc7a1053d0fbf3?utm_source=app_launcher&utm_medium=owned&utm_campaign=base_all
