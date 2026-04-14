"""
Visualization functions reproducing figures from the Flow Matching paper.

- Trajectory plots (Figure 4 left)
- Density evolution at t=0, 1/3, 2/3, 1 (Figure 2)
- NFE vs quality comparison (Figure 4 right)
- Training curves
- Final sample scatter plots
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import os


def set_style():
    """Set clean plot style."""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'figure.dpi': 150,
    })


set_style()


# ---------------------------------------------------------------------------
# Figure 4 (left): Sampling trajectories
# ---------------------------------------------------------------------------

def plot_trajectories(trajectories_dict, save_path, n_traj=200, lim=3.5):
    """
    Plot sampling trajectories for each method side by side.

    Args:
        trajectories_dict: {method_name: (T, B, 2) tensor}
        save_path: output file path
        n_traj: number of trajectories to draw
        lim: axis limit
    """
    methods = list(trajectories_dict.keys())
    n_methods = len(methods)

    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]

    for ax, name in zip(axes, methods):
        traj = trajectories_dict[name][:, :n_traj, :].cpu().numpy()  # (T, n_traj, 2)
        T_steps = traj.shape[0]

        # Color trajectories by time
        for i in range(min(n_traj, traj.shape[1])):
            points = traj[:, i, :]  # (T, 2)
            segments = np.stack([points[:-1], points[1:]], axis=1)
            colors = plt.cm.viridis(np.linspace(0, 1, T_steps - 1))

            lc = LineCollection(segments, colors=colors, linewidths=0.5, alpha=0.4)
            ax.add_collection(lc)

        # Mark start and end points
        ax.scatter(traj[0, :, 0], traj[0, :, 1], c='blue', s=1, alpha=0.3, zorder=5)
        ax.scatter(traj[-1, :, 0], traj[-1, :, 1], c='red', s=1, alpha=0.3, zorder=5)

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_title(name)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')

    fig.suptitle('Sampling Trajectories (blue=start, red=end)', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Figure 2: Density evolution at t = 0, 1/3, 2/3, 1
# ---------------------------------------------------------------------------

def plot_density_evolution(trajectories_dict, save_path, lim=3.5):
    """
    Plot sample density at t = 0, 1/3, 2/3, 1 for each method.

    Args:
        trajectories_dict: {method_name: (T, B, 2) tensor}
        save_path: output file path
    """
    time_fracs = [0.0, 1 / 3, 2 / 3, 1.0]
    methods = list(trajectories_dict.keys())

    fig, axes = plt.subplots(len(methods), 4, figsize=(16, 4 * len(methods)))
    if len(methods) == 1:
        axes = axes[np.newaxis, :]

    for row, name in enumerate(methods):
        traj = trajectories_dict[name].cpu().numpy()  # (T, B, 2)
        T_steps = traj.shape[0]

        for col, frac in enumerate(time_fracs):
            idx = int(frac * (T_steps - 1))
            points = traj[idx]  # (B, 2)

            ax = axes[row, col]
            ax.hist2d(points[:, 0], points[:, 1], bins=100,
                      range=[[-lim, lim], [-lim, lim]], cmap='inferno',
                      norm=mcolors.LogNorm())
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect('equal')
            ax.set_title(f't = {frac:.2f}')
            if col == 0:
                ax.set_ylabel(name)

    fig.suptitle('Density Evolution: $p_t$ at Different Time Steps', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Figure 4 (right): NFE vs sample quality
# ---------------------------------------------------------------------------

def compute_checkerboard_energy(samples, board_size=4, scale=4.0):
    """
    Simple quality metric: fraction of samples landing in correct (black) cells.
    Higher is better (max = 1.0 if all samples are in black cells).
    """
    cell_size = scale / board_size
    # Map to cell indices
    x_idx = ((samples[:, 0] + scale / 2) / cell_size).long().clamp(0, board_size - 1)
    y_idx = ((samples[:, 1] + scale / 2) / cell_size).long().clamp(0, board_size - 1)
    # Black cells: (i + j) % 2 == 0
    is_black = ((x_idx + y_idx) % 2 == 0).float()
    # Also check if in bounds
    in_bounds = ((samples[:, 0].abs() < scale / 2) & (samples[:, 1].abs() < scale / 2)).float()
    return (is_black * in_bounds).mean().item()


def plot_nfe_comparison(velocity_fns, save_path, n_samples=2048, device='cpu'):
    """
    Plot sample quality vs NFE for different solvers and methods.

    Args:
        velocity_fns: {method_name: callable(x, t) -> v}
        save_path: output file path
    """
    from samplers import euler_solve, midpoint_solve, rk4_solve

    nfe_values = [2, 4, 6, 8, 10, 15, 20, 30, 50, 100]
    solver_configs = {
        'Euler': (euler_solve, lambda n: n),
        'Midpoint': (midpoint_solve, lambda n: 2 * n),
        'RK4': (rk4_solve, lambda n: 4 * n),
    }

    methods = list(velocity_fns.keys())
    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 5))
    if len(methods) == 1:
        axes = [axes]

    for ax, method_name in zip(axes, methods):
        vel_fn = velocity_fns[method_name]

        for solver_name, (solver_fn, nfe_fn) in solver_configs.items():
            qualities = []
            actual_nfes = []

            for n_steps in nfe_values:
                x0 = torch.randn(n_samples, 2, device=device)
                samples = solver_fn(vel_fn, x0, n_steps=n_steps)
                q = compute_checkerboard_energy(samples)
                qualities.append(q)
                actual_nfes.append(nfe_fn(n_steps))

            ax.plot(actual_nfes, qualities, 'o-', label=solver_name, markersize=4)

        ax.set_xlabel('NFE (Number of Function Evaluations)')
        ax.set_ylabel('Quality (fraction in correct cells)')
        ax.set_title(method_name)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    fig.suptitle('Sample Quality vs. NFE', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Training loss curves
# ---------------------------------------------------------------------------

def plot_training_curves(loss_dict, save_path):
    """
    Plot training loss curves for all methods.

    Args:
        loss_dict: {method_name: list of (step, loss) tuples}
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, losses in loss_dict.items():
        steps, vals = zip(*losses)
        ax.plot(steps, vals, label=name, alpha=0.8)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Final samples scatter
# ---------------------------------------------------------------------------

def plot_samples(samples_dict, save_path, lim=3.5):
    """
    Scatter plot of final generated samples for each method.
    """
    methods = list(samples_dict.keys())
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 5))
    if len(methods) == 1:
        axes = [axes]

    for ax, name in zip(axes, methods):
        pts = samples_dict[name].cpu().numpy()
        ax.scatter(pts[:, 0], pts[:, 1], s=0.5, alpha=0.3, c='steelblue')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_title(f'{name}\n({len(pts)} samples)')
        ax.grid(True, alpha=0.2)

    fig.suptitle('Generated Samples (dopri5 solver)', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Vector field visualization
# ---------------------------------------------------------------------------

def plot_vector_fields(velocity_fns, save_path, lim=3.5, grid_res=25):
    """
    Quiver plot of the learned vector fields at t = 0, 1/3, 2/3.
    """
    time_points = [0.0, 1 / 3, 2 / 3]
    methods = list(velocity_fns.keys())

    fig, axes = plt.subplots(len(methods), len(time_points),
                             figsize=(5 * len(time_points), 5 * len(methods)))
    if len(methods) == 1:
        axes = axes[np.newaxis, :]

    xx = np.linspace(-lim, lim, grid_res)
    yy = np.linspace(-lim, lim, grid_res)
    X, Y = np.meshgrid(xx, yy)
    grid = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1),
                        dtype=torch.float32)

    for row, name in enumerate(methods):
        vel_fn = velocity_fns[name]
        for col, t_val in enumerate(time_points):
            t_tensor = torch.full((grid.shape[0], 1), t_val)
            with torch.no_grad():
                v = vel_fn(grid, t_tensor).numpy()

            ax = axes[row, col]
            magnitude = np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2).reshape(grid_res, grid_res)
            U = v[:, 0].reshape(grid_res, grid_res)
            V = v[:, 1].reshape(grid_res, grid_res)

            ax.quiver(X, Y, U, V, magnitude, cmap='coolwarm', alpha=0.8,
                      scale=None)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect('equal')
            ax.set_title(f't = {t_val:.2f}')
            if col == 0:
                ax.set_ylabel(name)

    fig.suptitle('Learned Vector Fields', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")
