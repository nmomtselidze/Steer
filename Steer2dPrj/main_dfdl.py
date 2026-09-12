import argparse
import torch
from func_nmbs import get_df_dl
from plotting import plot_trajectory

def step(state, l_val, f, dl_tensor):
    angle, x, y = state
    delta_angle = f(l_val) * dl_tensor		# da = f(l) * dl
    angle = angle + delta_angle		#  a = a + ad
    x = x + dl_tensor * torch.cos(angle)
    y = y + dl_tensor * torch.sin(angle)
    return (angle, x, y)
    
def main(arg, a0):
    # Constants and initial conditions.
    R = 1.0
    x0 = 0.0
    y0 = 0.0
    dl = 0.01
    lmax = 2 * torch.pi * R
    dl_tensor = torch.tensor(dl)	# tensor(0.0100)

    # Make sure a0, x0, y0 are tensors and require grad
    a0_tensor = torch.tensor(a0, requires_grad=True)	# tensor(1., requires_grad=True)
    x0_tensor = torch.tensor(x0, requires_grad=True)	# tensor(0., requires_grad=True)
    y0_tensor = torch.tensor(y0, requires_grad=True)	# tensor(0., requires_grad=True)

    # Retrieve the function
    f = get_df_dl(arg, l, R=1.0)

    # Compute trajectory (manual unrolling for gradients)
    l_values = torch.arange(0, lmax, dl)
    state = (a0_tensor, x0_tensor, y0_tensor)  # Initial state as tensors with grad
    states = [state]				# [(tensor(1., requires_grad=True), tensor(0., requires_grad=True), tensor(0., requires_grad=True))]
    for l_val in l_values:
        state = step(state, l_val, f, dl_tensor)
        states.append(state)

    angles, xs, ys = zip(*states)
    angles = torch.stack(angles)
    xs = torch.stack(xs)
    ys = torch.stack(ys)

    # Compute loss (e.g., final x-coordinate)
    loss = xs[-1]

    # Compute gradients
    loss.backward()

    # Access the gradient of the loss with respect to a0
    print(f"Gradient of loss (final x) w.r.t. a0: {a0_tensor.grad}")	# Gradient of loss (final x) w.r.t. a0: -10.57718276977539

    # Plot
    plot_trajectory(xs.detach().cpu(), ys.detach().cpu(), arg, a0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute trajectory and gradient w.r.t a0."
    )
    parser.add_argument("--arg", type=int, default=1, help="Function parameter")
    parser.add_argument("--a0", type=float, default=0.0, help="Initial angle")
    args = parser.parse_args()
    main(args.arg, args.a0)
