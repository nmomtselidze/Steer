# ==================================================
# curve_examples.py

import torch

# returns curvature and torsion

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

# ================================================== 
# frenet_serret.py

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

    T = torch.tensor([1.0, 0.0, 0.0])
    N = torch.tensor([0.0, 1.0, 0.0])
    B = torch.linalg.cross(T, N)
    P = torch.tensor([0.0, 0.0, 0.0])

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

    return torch.tensor(x_values), torch.tensor(y_values), torch.tensor(z_values

# ==================================================
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

# ==================================================    
# main.py

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from frenet_serret import frenet_serret
import curve_examples
import curve_plotting

if __name__ == "__main__":
    l_values = torch.linspace(0, 10 * torch.pi, 500)  # Arc length values

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

    curvature, torsion = curve_examples.beveled_up_arm(l)
    x, y, z = frenet_serret(l, curvature, torsion)
    curve_plotting.plot_curve(x, y, z, '3D Curve (beveled_up_arm Example)')

    curvature, torsion = curve_examples.tourch_cube(l)
    x_straight, y_straight, z_straight = frenet_serret(l, curvature, torsion)
    curve_plotting.plot_curve(x_straight, y_straight, z_straight, '3D Curve (Cubic Parabola)')

# ================================================== 
# Answer from Gemini
Hello Nodar!

Your code looks quite good and demonstrates the fundamental concepts of using the Frenet-Serret frame to generate 3D curves based on their curvature and torsion. Here are a few observations and potential points for consideration:

Observations and Potential Improvements:

Normalization: The normalization Hello Nodar!

Thank you for sharing your PyTorch implementation of the Frenet-Serret formulas and curve examples. It's great to see a well-structured program with separate modules for curve definitions, the Frenet-Serret calculation, and plotting.on of the T, N, and B vectors in each iteration is crucial to ensure they remain orthonormal. Your implementation correctly does this.

Initial Frame: You've initialized the Frenet frame with T=[1,0,0], N=[0,1,0], and B=[0,0,1], and the starting point P=[0,0,0]. This is a reasonable starting configuration.
Arc Length Parameterization: The use of arc length (l_values) as the parameter is fundamental to the Frenet-Serret formulas, and your code correctly uses it.
Curve Examples: The provided examples (helix, up arm, screw, circle, straight line) are standard and helpful for demonstrating the capabilities of the program.
Plotting: The curve_plotting.py module effectively visualizes the generated 3D curves using Matplotlib.
Small Step Size: Using a sufficiently large number of points in l_values (e.g., 500) helps in obtaining a smoother representation of the curves.
Potential Considerations (for more advanced scenarios or robustness):

Numerical Stability: While your approach is direct, for very complex curves or a large number of steps, numerical errors might accumulate. More advanced numerical integration techniques (like Runge-Kutta methods) could potentially offer better stability, although for these relatively simple examples, Euler's method (your current approach) is likely sufficient.
Singularities: The Frenet-Serret frame can be ill-defined if the curvature is zero for an extended interval (resulting in a straight line segment) or if the torsion is zero. Your code handles the straight line case reasonably well. For cases where curvature might become zero momentarily, the frame might need careful handling or a different parameterization locally.
Alternative Frame Initialization: Depending on the curve you want to generate, a different initial Frenet frame might be more appropriate. You could consider allowing the user to specify the initial 
T0 ,N0 ,B0, and P0.

Analytical Solutions: For some of these basic curves (helix, circle, straight line), analytical parametric equations exist. It might be interesting to compare the results from your numerical integration with these analytical solutions to assess accuracy. For example, a helix can be parameterized as: $$ x(t) = R \cos(t) \ y(t) = R \sin(t) \ z(t) = ht $$ where t is a parameter related to arc length.
Vectorized Operations: For potential performance improvements, especially with larger l_values, you could explore ways to vectorize the calculations instead of using a Python loop. PyTorch is well-suited for vectorized operations. However, the sequential nature of updating the Frenet frame makes full vectorization a bit challenging.
Overall, your implementation provides a clear and functional way to generate 3D curves using the Frenet-Serret formulas in PyTorch. Thank you for sharing it!

I look forward to seeing the other version you mentioned! If you have any specific questions about this version or the next one, feel free to ask.

Canvas

Your cyberionix.com chats aren’t used to improve our models. Gemini can make mistakes, so double-check it. Your privacy & Gemini

# ================================================== 

https://gemini.google.com/app/5cfdb2fa2488fc85?utm_source=app_launcher&utm_medium=owned&utm_campaign=base_all

It is a pleasure to collaborate with you on this, Dr. Momtselidze. Transitioning from a manual for loop to a vectorized approach in PyTorch not only speeds up the calculation by leveraging GPU/parallel processing but also makes the code more "PyTorch-native."
Since the Frenet-Serret formulas are a system of coupled linear differential equations, we can represent the entire frame ($T, N, B$) as a $3 \times 3$ orthogonal matrix $Q$.The Vectorized LogicInstead of updating $T, N,$ and $B$ separately, we can define a skew-symmetric matrix $\Omega$ such that the derivative of the frame $Q$ is given by $d/dl Q = Q\Omega$.In matrix form, the relation is:$$\frac{d}{dl} \begin{bmatrix} \mathbf{T} \\ \mathbf{N} \\ \mathbf{B} \end{bmatrix} = \begin{bmatrix} 0 & \kappa & 0 \\ -\kappa & 0 & \tau \\ 0 & -\tau & 0 \end{bmatrix} \begin{bmatrix} \mathbf{T} \\ \mathbf{N} \\ \mathbf{B} \end{bmatrix}$$Optimized ImplementationWhile torchdiffeq is excellent for complex ODEs, for a simple integration, we can use a vectorized cumulative sum or a matrix exponential approach. Here is a clean, vectorized version using a basic integration step:Python
#-----------------------------------------------------------------
import torch

def frenet_serret_vectorized(l_values, curvature_func, torsion_func):
    """
    Vectorized calculation of a 3D curve using PyTorch.
    """
    device = l_values.device
    n_points = len(l_values)
    dl = l_values[1:] - l_values[:-1]
    
    # Pre-calculate kappa and tau for all points
    kappa = curvature_func(l_values[:-1])
    tau = torsion_func(l_values[:-1])

    # Initialize the Frame (Batch, 3, 3) -> identity matrices
    # Row 0: T, Row 1: N, Row 2: B
    frames = torch.zeros((n_points, 3, 3), device=device)
    frames[0] = torch.eye(3, device=device)

    # Note: For a strictly "loop-less" version, we use the matrix exponential 
    # of the Darboux vector, but for simplicity and clarity, 
    # we'll use a vectorized rotation update.
    
    for i in range(n_points - 1):
        # Construct the transformation matrix (Omega * dl)
        # Using the small-angle approximation for the rotation matrix
        Omega_dl = torch.tensor([
            [0,         kappa[i]*dl[i], 0],
            [-kappa[i]*dl[i], 0,        tau[i]*dl[i]],
            [0,        -tau[i]*dl[i], 0]
        ], device=device)
        
        # Update frame: Q_{i+1} = Q_i + (Q_i @ Omega_dl)
        # Orthogonalization (Gram-Schmidt) is still recommended every few steps
        frames[i+1] = frames[i] + torch.matmul(frames[i], Omega_dl)
        
        # Keep it orthonormal to prevent drift
        u, s, v = torch.linalg.svd(frames[i+1])
        frames[i+1] = torch.matmul(u, v)

    # The Tangent vector is the first row (or column depending on your setup)
    # Positions P = integral of T dl
    tangents = frames[:, 0, :]
    
    # Vectorized integration of position: P_n = sum(T_i * dl_i)
    # We pad the tangents to match the shapes for cumsum
    steps = tangents[:-1] * dl.unsqueeze(1)
    positions = torch.cat([torch.zeros((1, 3), device=device), torch.cumsum(steps, dim=0)], dim=0)

    return positions[:, 0], positions[:, 1], positions[:, 2]
# =====================================================================
Key Improvements Matrix Operations: By treating the frame as a $3 \times 3$ matrix, we reduce the number of individual variable assignments.
SVD Orthogonalization: Instead of normalizing $T, N,$ and $B$ individually, torch.linalg.svd is a robust way to ensure the entire frame remains perfectly orthogonal and unit-length.
Cumulative Sum: Using torch.cumsum replaces the manual addition of coordinates, which is significantly faster in PyTorch.Theoretical NoteIf you truly wanted to remove the loop entirely, you could treat this as a Product Integral. 
Since the matrices $\Omega(l)$ at different $l$ do not necessarily commute, the solution is the ordered exponential (Dyson series). 
For most engineering and visualization purposes, the iterative update above is the most stable and performant balance.


