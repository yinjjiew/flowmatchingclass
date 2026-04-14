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
# Figure 4 (left): Density snapshots with trajectory overlay
# ---------------------------------------------------------------------------
# Paper Figure 4 left shows a grid of [methods × time steps] where each cell
# is a 2D density heatmap of the *marginal* distribution p_t at that time,
# overlaid with a subset of ODE sampling trajectories.  This is the key figure
# for comparing how quickly each method "reveals" the checkerboard pattern:
#   - SM-Dif: noise dominates until the very end; pattern appears only near t≈1.
#   - FM-Dif: similar to SM-Dif but slightly earlier emergence.
#   - FM-OT:  checkerboard structure appears much earlier (by t≈0.5).
# The trajectory overlay shows the ODE paths particles take from noise to data.
# OT paths are visually straight, diffusion paths are curved.

def plot_figure4_left(trajectories_dict, save_path, n_traj=150, lim=3.5):
    """
    Reproduce Figure 4 (left) from the paper: density heatmaps with
    trajectory overlays at multiple time snapshots for all 3 methods.

    Layout: rows = methods, columns = time steps t ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0}

    Each cell shows:
      - Background: 2D histogram (density heatmap) of sample positions at time t,
        approximating the marginal distribution p_t. Plotted with hist2d using
        'inferno' colormap on a log scale so both dense and sparse regions are visible.
      - Foreground: a subset of ODE trajectories drawn as colored lines.
        Line color encodes time progression (viridis: purple=t=0 → yellow=t=1).
        A vertical dashed line marks the current time snapshot.

    What to look for:
      - How early the checkerboard pattern emerges (FM-OT >> FM-Dif ≈ SM-Dif).
      - Trajectory shape: FM-OT paths are nearly straight; diffusion paths curve.
      - FM-OT particles spread less and take more direct routes to their targets.

    Args:
        trajectories_dict: {method_name: (T, B, 2) tensor of ODE trajectories}
        save_path: output file path
        n_traj: number of trajectories to overlay (too many = visual clutter)
        lim: axis limit [-lim, lim]
    """
    time_fracs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    methods = list(trajectories_dict.keys())
    n_cols = len(time_fracs)
    n_rows = len(methods)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, name in enumerate(methods):
        traj = trajectories_dict[name].cpu().numpy()  # (T, B, 2)
        T_steps = traj.shape[0]
        n_draw = min(n_traj, traj.shape[1])

        for col, frac in enumerate(time_fracs):
            ax = axes[row, col]
            t_idx = int(frac * (T_steps - 1))
            points = traj[t_idx]  # (B, 2) — all samples at this time

            # --- density heatmap (background) ---
            ax.hist2d(points[:, 0], points[:, 1], bins=120,
                      range=[[-lim, lim], [-lim, lim]], cmap='inferno',
                      norm=mcolors.LogNorm(), zorder=0)

            # --- trajectory overlay (foreground) ---
            # Draw trajectories up to the current time step
            traj_seg = traj[:t_idx + 1, :n_draw, :]  # (t_idx+1, n_draw, 2)
            if traj_seg.shape[0] >= 2:
                for i in range(n_draw):
                    pts = traj_seg[:, i, :]  # (t_idx+1, 2)
                    segments = np.stack([pts[:-1], pts[1:]], axis=1)
                    n_seg = segments.shape[0]
                    colors = plt.cm.viridis(np.linspace(0, frac, n_seg))
                    lc = LineCollection(segments, colors=colors,
                                        linewidths=0.3, alpha=0.35, zorder=1)
                    ax.add_collection(lc)

            # --- current position markers ---
            ax.scatter(points[:n_draw, 0], points[:n_draw, 1],
                       c='white', s=0.3, alpha=0.4, zorder=2, edgecolors='none')

            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                ax.set_title(f'$t = {frac:.1f}$', fontsize=13)
            if col == 0:
                ax.set_ylabel(name, fontsize=13, fontweight='bold')

    fig.suptitle('Figure 4 (left): Density $p_t$ + Sampling Trajectories',
                 fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Figure 4 (right): Low-NFE sample quality (midpoint scheme)
# ---------------------------------------------------------------------------
# Paper Figure 4 right shows generated samples at NFE = 4, 8, 10, 20 using
# the midpoint ODE solver for FM-OT.  This demonstrates that OT paths produce
# recognizable samples even at very low NFE, while diffusion paths need many
# more steps.  We extend this to show all 3 methods side by side.

def plot_figure4_right(velocity_fns, save_path, n_samples=4096,
                       nfe_list=None, device='cpu'):
    """
    Reproduce Figure 4 (right): scatter plots of generated samples at different
    NFE budgets using the midpoint ODE solver.

    Layout: rows = methods, columns = NFE values.

    Each cell shows a 2D scatter plot of samples generated by solving the ODE
    with the midpoint scheme using a fixed number of steps.  The midpoint method
    uses 2 function evaluations per step, so NFE = 2 × n_steps.

    What to look for:
      - FM-OT produces recognizable checkerboard at NFE=8 (4 midpoint steps).
      - FM-Dif / SM-Dif need NFE≥20 for comparable quality.
      - At very low NFE, diffusion paths produce blurry/spread-out samples
        because the curved trajectories are poorly approximated by few steps.

    Args:
        velocity_fns: {method_name: callable(x, t) -> velocity}
        save_path: output file path
        n_samples: number of samples per cell
        nfe_list: list of NFE values (default: [4, 8, 10, 20] as in paper)
        device: torch device
    """
    from samplers import midpoint_solve

    if nfe_list is None:
        nfe_list = [4, 8, 10, 20]

    methods = list(velocity_fns.keys())
    n_rows = len(methods)
    n_cols = len(nfe_list)
    lim = 3.5

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Use same initial noise for fair comparison across methods
    x0_shared = torch.randn(n_samples, 2, device=device)

    for row, name in enumerate(methods):
        vel_fn = velocity_fns[name]
        for col, nfe in enumerate(nfe_list):
            n_steps = nfe // 2  # midpoint uses 2 evaluations per step
            n_steps = max(n_steps, 1)

            samples = midpoint_solve(vel_fn, x0_shared.clone(), n_steps=n_steps)
            pts = samples.cpu().numpy()

            ax = axes[row, col]
            ax.scatter(pts[:, 0], pts[:, 1], s=0.3, alpha=0.25, c='steelblue',
                       edgecolors='none')
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                ax.set_title(f'NFE={nfe}', fontsize=13)
            if col == 0:
                ax.set_ylabel(name, fontsize=13, fontweight='bold')

    fig.suptitle('Figure 4 (right): Low-NFE Samples (Midpoint Solver)',
                 fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Standalone trajectory plot (supplementary, not in paper Figure 4)
# ---------------------------------------------------------------------------
# This shows ONLY the ODE trajectories without density background.
# Useful for clearly seeing the path geometry: OT = straight, Diffusion = curved.

def plot_trajectories(trajectories_dict, save_path, n_traj=200, lim=3.5):
    """
    Plot sampling trajectories for each method side by side (trajectory-only view).

    Each panel shows ODE integration paths from noise (t=0, blue dots) to data
    (t=1, red dots).  Line color encodes time via the viridis colormap.

    What to look for:
      - FM-OT: trajectories are nearly straight lines (constant velocity OT map).
      - FM-Dif / SM-Dif: trajectories are curved; particles may "overshoot"
        and backtrack, especially visible in SM-Dif.
      - Endpoint red dots should form a checkerboard pattern if training succeeded.

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
# Density evolution with finer time resolution
# ---------------------------------------------------------------------------
# This is an extended version of the density snapshots: more time columns
# than Figure 4 left, giving a smooth "filmstrip" view of how p_t evolves.
# Helps diagnose whether the model is learning the correct marginal path.

def plot_density_evolution(trajectories_dict, save_path, lim=3.5,
                           time_fracs=None):
    """
    Plot p_t density heatmaps at many time snapshots for each method.

    Layout: rows = methods, columns = time steps.

    Each cell is a 2D histogram (density heatmap) of the sample positions at
    the given time t, approximating the marginal distribution p_t(x).  We use
    the 'inferno' colormap with log-scale normalization so that both the
    concentrated peaks (bright) and diffuse tails (dark) are visible.

    Default uses 8 time steps: t ∈ {0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0}
    to provide a finer-grained view than the paper's 4-column Figure 4.

    What to look for:
      - t=0: all methods start from the same isotropic Gaussian blob.
      - Early t (0.15–0.3): FM-OT already shows rectangular structure;
        diffusion methods remain blobby.
      - Mid t (0.45–0.6): FM-OT has clear checkerboard; diffusion still noisy.
      - Late t (0.75–1.0): all methods converge to checkerboard, but FM-OT
        has sharper cell boundaries.
      - The speed of pattern emergence directly reflects trajectory efficiency:
        straight OT paths "deliver" mass to the right place faster than curved
        diffusion paths that only denoise near the end.

    Args:
        trajectories_dict: {method_name: (T, B, 2) tensor}
        save_path: output file path
        lim: axis limit
        time_fracs: list of time fractions to show (default: 8 evenly spaced)
    """
    if time_fracs is None:
        time_fracs = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]

    methods = list(trajectories_dict.keys())
    n_cols = len(time_fracs)
    n_rows = len(methods)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.8 * n_cols, 2.8 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, name in enumerate(methods):
        traj = trajectories_dict[name].cpu().numpy()  # (T, B, 2)
        T_steps = traj.shape[0]

        for col, frac in enumerate(time_fracs):
            idx = int(frac * (T_steps - 1))
            points = traj[idx]  # (B, 2)

            ax = axes[row, col]
            ax.hist2d(points[:, 0], points[:, 1], bins=120,
                      range=[[-lim, lim], [-lim, lim]], cmap='inferno',
                      norm=mcolors.LogNorm())
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                ax.set_title(f'$t={frac:.2f}$', fontsize=10)
            if col == 0:
                ax.set_ylabel(name, fontsize=11, fontweight='bold')

    fig.suptitle('Density Evolution $p_t$ (8 time snapshots per method)',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# NFE vs quality curve (quantitative version of Figure 4 right)
# ---------------------------------------------------------------------------
# While plot_figure4_right shows sample scatter at fixed NFE values,
# this plot provides a *quantitative* view: sample quality as a function of
# NFE for three different ODE solvers (Euler, Midpoint, RK4).
#
# Quality metric: fraction of samples landing in correct checkerboard cells.
# This is a proxy for FID in our 2D setting — a perfect model would place
# 100% of samples in the black cells.

def compute_checkerboard_energy(samples, board_size=4, scale=4.0):
    """
    Compute sample quality as fraction of samples in correct checkerboard cells.

    We divide [-scale/2, scale/2]^2 into board_size × board_size cells.
    "Correct" cells are the black cells where (row + col) % 2 == 0.
    Returns a float in [0, 1]; higher is better.

    This serves as a 2D proxy for FID: it measures whether the model has
    learned the correct *support* of the distribution.
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

    Layout: one subplot per method.  Each subplot has 3 curves (Euler, Midpoint,
    RK4) showing quality (y-axis) vs NFE (x-axis).

    NFE = Number of Function Evaluations = number of times the neural network
    is called during sampling.  Different solvers use different numbers of
    evaluations per step:
      - Euler:    NFE = n_steps      (1 eval/step)
      - Midpoint: NFE = 2 × n_steps  (2 evals/step)
      - RK4:      NFE = 4 × n_steps  (4 evals/step)

    What to look for:
      - FM-OT reaches high quality at much lower NFE than diffusion methods.
      - Higher-order solvers (RK4) are more efficient per-NFE than Euler.
      - For FM-OT, even Euler at NFE=10 produces reasonable quality,
        while SM-Dif may need NFE=50+ for comparable results.
      - This demonstrates the practical advantage of straighter OT paths:
        they are easier to integrate numerically with fewer steps.

    Args:
        velocity_fns: {method_name: callable(x, t) -> v}
        save_path: output file path
        n_samples: samples per evaluation point
        device: torch device
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
# Not from a specific paper figure, but essential for diagnosing convergence.
# All three methods minimize different losses (CFM for FM-OT/FM-Dif, weighted
# SM for SM-Dif), so absolute loss values aren't directly comparable.
# What matters is convergence speed and stability.

def plot_training_curves(loss_dict, save_path):
    """
    Plot training loss curves for all methods on a single log-scale plot.

    What to look for:
      - FM-OT typically converges fastest and most smoothly because its
        regression target (x₁ - (1-σ_min)x₀) is independent of t.
      - FM-Dif and SM-Dif may show noisier curves because the target
        magnitude varies with t (large near t=0 where σ is small).
      - If SM-Dif loss is much higher than FM losses, that's expected —
        different loss scales due to the σ² weighting.

    Args:
        loss_dict: {method_name: list of (step, loss) tuples}
        save_path: output file path
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
# Final samples scatter (high-quality adaptive solver)
# ---------------------------------------------------------------------------
# Shows the best-quality samples each model can produce using the adaptive
# dopri5 solver (tolerance 1e-5).  This eliminates ODE solver error, so any
# imperfections are due to the model itself, not numerical integration.

def plot_samples(samples_dict, save_path, lim=3.5):
    """
    Scatter plot of final generated samples using the dopri5 adaptive solver.

    Each panel shows N samples from one method.  The dopri5 solver adaptively
    chooses step sizes to keep integration error below tolerance, so these
    represent the model's "true" output quality.

    What to look for:
      - All methods should produce a clear checkerboard pattern if trained
        sufficiently (≥15k steps).
      - FM-OT cells tend to have sharper boundaries and more uniform density.
      - SM-Dif may show slight bleeding between cells or density variation.
      - Points outside [-2, 2]² are model failures (should be rare).

    Args:
        samples_dict: {method_name: (N, 2) tensor}
        save_path: output file path
        lim: axis limit
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
# Vector field visualization (cf. Figure 2 / Figure 8 in paper)
# ---------------------------------------------------------------------------
# The paper's Figure 2 compares the *conditional* score/VF for a single x₁.
# Here we show the *learned marginal* vector field v_t(x; θ) on a grid,
# which is what the model actually predicts at test time.  The marginal VF
# is an average over all conditional VFs weighted by the data distribution.

def plot_vector_fields(velocity_fns, save_path, lim=3.5, grid_res=25):
    """
    Quiver plot of the learned vector field v_t(x; θ) at different times.

    Layout: rows = methods, columns = time steps (t=0.0, 0.33, 0.67).

    Each cell shows a 2D quiver (arrow) plot of the velocity field on a
    uniform grid.  Arrow direction = flow direction, arrow color = magnitude
    (blue=small, red=large via 'coolwarm' colormap).

    What to look for:
      - FM-OT at all times: arrows point roughly in the same direction at each
        location, confirming the constant-direction property of OT paths.
        The field should look like it's "sorting" points into checkerboard cells.
      - FM-Dif / SM-Dif at t=0: arrows are large and somewhat radial (the
        diffusion VF has strong initial dynamics).  At t=0.67: arrows become
        more structured as the field refines sample positions.
      - Magnitude contrast: FM-OT arrows have more uniform magnitude across
        time (constant speed), while diffusion methods show large magnitude
        variation (most work happens near t=1).

    Args:
        velocity_fns: {method_name: callable(x, t) -> v}
        save_path: output file path
        lim: axis limit
        grid_res: number of grid points per axis for quiver plot
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
