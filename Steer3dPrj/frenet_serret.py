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

    T = torch.tensor([1., 0., 0.])
    N = torch.tensor([0., 1., 0.])
    B = torch.linalg.cross(T, N)
    P = torch.tensor([0., 0., 0.])

    x_values = [P[0].item()]
    y_values = [P[1].item()]
    z_values = [P[2].item()]

    dl = l_values[1] - l_values[0]

    for l in l_values:
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

    return torch.tensor(x_values), torch.tensor(y_values), torch.tensor(z_values)
