"""
ODE solvers for sampling from trained flow models.

We solve: dφ_t/dt = v_t(φ_t), φ_0 = x_0 ~ N(0, I), t ∈ [0, 1]

Four solvers with increasing accuracy:
  - Euler:    1st order, NFE = N
  - Midpoint: 2nd order, NFE = 2N
  - RK4:      4th order, NFE = 4N
  - dopri5:   Adaptive 5th order (from scipy)
"""

import torch
import numpy as np
from scipy.integrate import solve_ivp


@torch.no_grad()
def euler_solve(velocity_fn, x0: torch.Tensor, n_steps: int = 100,
                return_trajectory: bool = False):
    """
    Euler method: x_{k+1} = x_k + h · v(x_k, t_k)

    Args:
        velocity_fn: callable(x, t) -> dx/dt
        x0: (B, 2) initial noise samples
        n_steps: number of Euler steps (NFE = n_steps)
        return_trajectory: if True, return all intermediate states

    Returns:
        x1 or trajectory (n_steps+1, B, 2)
    """
    dt = 1.0 / n_steps
    x = x0.clone()
    trajectory = [x.clone()] if return_trajectory else None

    for k in range(n_steps):
        t = k * dt
        t_tensor = torch.full((x.shape[0], 1), t, device=x.device)
        v = velocity_fn(x, t_tensor)
        x = x + dt * v
        if return_trajectory:
            trajectory.append(x.clone())

    if return_trajectory:
        return torch.stack(trajectory, dim=0)  # (n_steps+1, B, 2)
    return x


@torch.no_grad()
def midpoint_solve(velocity_fn, x0: torch.Tensor, n_steps: int = 100,
                   return_trajectory: bool = False):
    """
    Midpoint method (2nd order):
        k1 = v(x_k, t_k)
        x_{k+1} = x_k + h · v(x_k + h/2 · k1, t_k + h/2)

    NFE = 2 * n_steps
    """
    dt = 1.0 / n_steps
    x = x0.clone()
    trajectory = [x.clone()] if return_trajectory else None

    for k in range(n_steps):
        t = k * dt
        t_tensor = torch.full((x.shape[0], 1), t, device=x.device)
        t_mid = torch.full((x.shape[0], 1), t + dt / 2, device=x.device)

        k1 = velocity_fn(x, t_tensor)
        x_mid = x + (dt / 2) * k1
        k2 = velocity_fn(x_mid, t_mid)
        x = x + dt * k2

        if return_trajectory:
            trajectory.append(x.clone())

    if return_trajectory:
        return torch.stack(trajectory, dim=0)
    return x


@torch.no_grad()
def rk4_solve(velocity_fn, x0: torch.Tensor, n_steps: int = 100,
              return_trajectory: bool = False):
    """
    Classical Runge-Kutta 4th order:
        k1 = v(x_k, t_k)
        k2 = v(x_k + h/2·k1, t_k + h/2)
        k3 = v(x_k + h/2·k2, t_k + h/2)
        k4 = v(x_k + h·k3, t_k + h)
        x_{k+1} = x_k + h/6·(k1 + 2k2 + 2k3 + k4)

    NFE = 4 * n_steps
    """
    dt = 1.0 / n_steps
    x = x0.clone()
    trajectory = [x.clone()] if return_trajectory else None

    for k in range(n_steps):
        t = k * dt
        t0 = torch.full((x.shape[0], 1), t, device=x.device)
        t_mid = torch.full((x.shape[0], 1), t + dt / 2, device=x.device)
        t1 = torch.full((x.shape[0], 1), t + dt, device=x.device)

        k1 = velocity_fn(x, t0)
        k2 = velocity_fn(x + (dt / 2) * k1, t_mid)
        k3 = velocity_fn(x + (dt / 2) * k2, t_mid)
        k4 = velocity_fn(x + dt * k3, t1)

        x = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        if return_trajectory:
            trajectory.append(x.clone())

    if return_trajectory:
        return torch.stack(trajectory, dim=0)
    return x


@torch.no_grad()
def dopri5_solve(velocity_fn, x0: torch.Tensor, atol: float = 1e-5,
                 rtol: float = 1e-5, return_trajectory: bool = False):
    """
    Adaptive Dormand-Prince (dopri5) solver via scipy.

    This is the solver used in the paper for final evaluation.
    Returns (samples, nfe) where nfe is the number of function evaluations.
    """
    device = x0.device
    batch_size, dim = x0.shape
    x0_np = x0.cpu().numpy().flatten()

    nfe_count = [0]

    def rhs(t, x_flat):
        nfe_count[0] += 1
        x_tensor = torch.tensor(x_flat.reshape(batch_size, dim),
                                dtype=torch.float32, device=device)
        t_tensor = torch.full((batch_size, 1), t, device=device)
        v = velocity_fn(x_tensor, t_tensor)
        return v.cpu().numpy().flatten()

    if return_trajectory:
        t_eval = np.linspace(0, 1, 50)
    else:
        t_eval = None

    sol = solve_ivp(rhs, [0, 1], x0_np, method='RK45', atol=atol, rtol=rtol,
                    t_eval=t_eval)

    if return_trajectory:
        traj = sol.y.reshape(dim, batch_size, -1)  # (dim, B, T)
        traj = np.transpose(traj, (2, 1, 0))  # (T, B, dim)
        return torch.tensor(traj, dtype=torch.float32, device=device), nfe_count[0]

    x1_np = sol.y[:, -1].reshape(batch_size, dim)
    return torch.tensor(x1_np, dtype=torch.float32, device=device), nfe_count[0]
