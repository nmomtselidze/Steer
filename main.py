import torch

# Set the radius of the sphere
RADIUS = 1.0

# 1. Define the parameterization of the sphere
# Maps 2D coordinates (theta, phi) to 3D space (x, y, z)
def sphere_parameterization(coords: torch.Tensor) -> torch.Tensor:
    """
    Args:
        coords: A tensor of shape (..., 2) where the last dimension
                contains theta (polar angle) and phi (azimuthal angle).
    Returns:
        A tensor of shape (..., 3) for the (x, y, z) coordinates.
    """
    theta = coords[..., 0]
    phi = coords[..., 1]

    x = RADIUS * torch.sin(theta) * torch.cos(phi)
    y = RADIUS * torch.sin(theta) * torch.sin(phi)
    z = RADIUS * torch.cos(theta)
    
    return torch.stack([x, y, z], dim=-1)

# 2. Function to compute the metric tensor using autograd
def metric_tensor(coords: torch.Tensor) -> torch.Tensor:
    """
    Computes the metric tensor for the sphere at given coordinates.
    """
    # Use autograd to get the Jacobian matrix (partial derivatives)
    # The Jacobian's columns are the tangent vectors to the surface
    J = torch.autograd.functional.jacobian(sphere_parameterization, coords)
    
    # The metric tensor g_ij is the dot product of the tangent vectors
    # This is equivalent to J^T @ J
    # J shape: (..., 3, 2), J.transpose shape: (..., 2, 3)
    # Resulting metric tensor shape: (..., 2, 2)
    g = torch.einsum('...ki,...kj->...ij', J, J) # More general than transpose
    
    return g

# 3. Define the curve on the sphere
# A line of longitude from the North Pole (theta=0) to the Equator (theta=pi/2)
# We keep phi constant (e.g., phi=0)
num_steps = 1000
t = torch.linspace(0, torch.pi / 2, num_steps) # Parameter t from 0 to pi/2

# Path in the parameter space (theta(t), phi(t))
theta_path = t  # theta varies with t
phi_path = torch.zeros_like(t) # phi is constant at 0
path = torch.stack([theta_path, phi_path], dim=-1)

# Velocity of the curve in parameter space (d(theta)/dt, d(phi)/dt)
# For our simple path, this is (1, 0)
path_velocity = torch.autograd.functional.jacobian(lambda x: torch.stack([x, torch.zeros_like(x)]), t)
path_velocity = torch.unsqueeze(path_velocity, dim=-1) # Shape (num_steps, 2, 1)

# 4. Numerically integrate the arc length
total_length = 0.0
dt = (torch.pi / 2) / (num_steps - 1)

# Calculate the metric tensor at each point along the path
g_path = metric_tensor(path) # Shape (num_steps, 2, 2)

# Calculate ds^2 = v^T * g * v for each point
# (1, 2) @ (2, 2) -> (1, 2) @ (2, 1) -> (1, 1)
ds_squared = torch.transpose(path_velocity, -1, -2)
ds_squared = torch.squeeze(ds_squared) # Remove extra dimensions

# The total length is the integral of sqrt(ds^2)
total_length = torch.sum(torch.sqrt(ds_squared) * dt)

# --- Output the results ---
print(f"Pytorch calculated arc length: {total_length.item():.5f}")
print(f"Analytical solution (pi*r/2): {torch.pi * RADIUS / 2:.5f}")

# Example of the metric tensor at the equator (theta=pi/2, phi=0)
equator_point = torch.tensor([torch.pi / 2, 0.0])
g_equator = metric_tensor(equator_point)
print(f"\nMetric tensor at the equator (theta=pi/2, phi=0):\n{g_equator}")
# Expected analytical result: [[r^2, 0], [0, r^2*sin(theta)^2]] = [[1, 0], [0, 1]]
