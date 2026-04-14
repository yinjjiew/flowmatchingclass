#!/usr/bin/env python3
"""
Flow Matching 2D Experiments — Full Reproduction
================================================

Reproduces Figure 4 (trajectories + NFE ablation), Figure 2 (density evolution),
and additional diagnostics from "Flow Matching for Generative Modeling"
(Lipman et al., 2022).

Usage:
    python main.py                        # Train all 3 methods + generate all figures
    python main.py --method fm_ot         # Train only FM-OT
    python main.py --epochs 30000         # More training steps
    python main.py --device cuda          # Use GPU
"""

import argparse
import os
import time
import torch
import numpy as np
from tqdm import tqdm

from model import VectorFieldMLP
from dataset import sample_checkerboard, sample_noise
from methods import FlowMatchingOT, FlowMatchingDiffusion, ScoreMatchingDiffusion
from samplers import euler_solve, midpoint_solve, rk4_solve, dopri5_solve
from visualize import (
    plot_trajectories, plot_density_evolution, plot_nfe_comparison,
    plot_training_curves, plot_samples, plot_vector_fields,
)


# ============================= Training =====================================

def train_method(method, model, args):
    """
    Train a single method.

    Args:
        method: one of FlowMatchingOT, FlowMatchingDiffusion, ScoreMatchingDiffusion
        model: VectorFieldMLP instance
        args: parsed command-line arguments

    Returns:
        loss_history: list of (step, loss_value) tuples
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=(0.9, 0.999), weight_decay=0.0)
    loss_history = []

    pbar = tqdm(range(1, args.epochs + 1), desc=f"Training {method.name}")
    for step in pbar:
        # Sample data and noise
        x1 = sample_checkerboard(args.batch_size).to(args.device)  # (B, 2)
        x0 = sample_noise(args.batch_size).to(args.device)          # (B, 2)
        t = torch.rand(args.batch_size, 1, device=args.device)      # (B, 1)

        # Compute loss
        loss = method.compute_loss(model, x1, x0, t)

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log
        if step % 100 == 0 or step == 1:
            loss_val = loss.item()
            loss_history.append((step, loss_val))
            pbar.set_postfix(loss=f"{loss_val:.4f}")

    return loss_history


# ========================== Sampling Helpers ================================

def make_velocity_fn(method, model, device):
    """Create a velocity function closure for ODE solvers."""
    model.eval()

    @torch.no_grad()
    def velocity_fn(x, t):
        return method.get_velocity(model, x.to(device), t.to(device)).cpu()

    return velocity_fn


def generate_trajectories(velocity_fn, n_samples=4096, n_steps=200, device='cpu'):
    """Generate samples with trajectory recording using midpoint solver."""
    x0 = sample_noise(n_samples).to(device)
    traj = midpoint_solve(velocity_fn, x0, n_steps=n_steps, return_trajectory=True)
    return traj


def generate_samples_dopri5(velocity_fn, n_samples=4096, device='cpu'):
    """Generate samples using adaptive dopri5 solver."""
    x0 = sample_noise(n_samples).to(device)
    samples, nfe = dopri5_solve(velocity_fn, x0, atol=1e-5, rtol=1e-5)
    return samples, nfe


# ================================ Main ======================================

def main():
    parser = argparse.ArgumentParser(description='Flow Matching 2D Experiments')
    parser.add_argument('--method', type=str, default='all',
                        choices=['all', 'fm_ot', 'fm_dif', 'sm_dif'],
                        help='Which method to train')
    parser.add_argument('--epochs', type=int, default=20000,
                        help='Number of training iterations')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='MLP hidden layer width')
    parser.add_argument('--n_layers', type=int, default=5,
                        help='Number of MLP hidden layers')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: cpu, cuda, or auto')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory for output figures')
    parser.add_argument('--n_traj_samples', type=int, default=4096,
                        help='Number of samples for trajectory visualization')
    parser.add_argument('--n_eval_samples', type=int, default=4096,
                        help='Number of samples for evaluation')
    args = parser.parse_args()

    # ---- Setup ----
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {args.device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Define methods ----
    method_registry = {
        'fm_ot':  FlowMatchingOT(sigma_min=1e-4),
        'fm_dif': FlowMatchingDiffusion(),
        'sm_dif': ScoreMatchingDiffusion(),
    }

    if args.method == 'all':
        methods_to_train = list(method_registry.keys())
    else:
        methods_to_train = [args.method]

    # ---- Train ----
    trained = {}  # {key: (method, model, loss_history)}
    all_loss_histories = {}

    for key in methods_to_train:
        method = method_registry[key]
        model = VectorFieldMLP(
            input_dim=2,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
        ).to(args.device)

        print(f"\n{'=' * 60}")
        print(f"Training: {method.name}")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")
        print(f"{'=' * 60}")

        t0 = time.time()
        loss_history = train_method(method, model, args)
        elapsed = time.time() - t0
        print(f"Training time: {elapsed:.1f}s")

        trained[key] = (method, model, loss_history)
        all_loss_histories[method.name] = loss_history

        # Save checkpoint
        ckpt_path = os.path.join(args.output_dir, f"checkpoint_{key}.pt")
        torch.save({
            'model_state': model.state_dict(),
            'method': key,
            'args': vars(args),
            'loss_history': loss_history,
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    # ---- Generate Figures ----
    print(f"\n{'=' * 60}")
    print("Generating figures...")
    print(f"{'=' * 60}")

    # Build velocity functions
    velocity_fns = {}
    for key, (method, model, _) in trained.items():
        velocity_fns[method.name] = make_velocity_fn(method, model, args.device)

    # 1. Training curves
    if len(all_loss_histories) > 0:
        print("\n[1/6] Training curves...")
        plot_training_curves(
            all_loss_histories,
            os.path.join(args.output_dir, 'training_curves.png'),
        )

    # 2. Trajectories (Figure 4 left)
    print("\n[2/6] Sampling trajectories (Figure 4 left)...")
    trajectories = {}
    for key, (method, model, _) in trained.items():
        vel_fn = velocity_fns[method.name]
        traj = generate_trajectories(vel_fn, n_samples=args.n_traj_samples,
                                     n_steps=200, device='cpu')
        trajectories[method.name] = traj
        print(f"  {method.name}: trajectory shape {traj.shape}")

    plot_trajectories(
        trajectories,
        os.path.join(args.output_dir, 'trajectories_comparison.png'),
        n_traj=300,
    )

    # 3. Density evolution (Figure 2)
    print("\n[3/6] Density evolution (Figure 2)...")
    plot_density_evolution(
        trajectories,
        os.path.join(args.output_dir, 'density_evolution.png'),
    )

    # 4. Final samples with dopri5
    print("\n[4/6] Final samples (dopri5)...")
    samples_dict = {}
    for key, (method, model, _) in trained.items():
        vel_fn = velocity_fns[method.name]
        samples, nfe = generate_samples_dopri5(vel_fn, n_samples=args.n_eval_samples,
                                               device='cpu')
        samples_dict[method.name] = samples
        print(f"  {method.name}: {nfe} NFE (dopri5, tol=1e-5)")

    plot_samples(
        samples_dict,
        os.path.join(args.output_dir, 'samples_dopri5.png'),
    )

    # 5. NFE comparison (Figure 4 right)
    print("\n[5/6] NFE comparison (Figure 4 right)...")
    plot_nfe_comparison(
        velocity_fns,
        os.path.join(args.output_dir, 'nfe_comparison.png'),
        n_samples=2048,
        device='cpu',
    )

    # 6. Vector field visualization
    print("\n[6/6] Vector field visualization...")
    plot_vector_fields(
        velocity_fns,
        os.path.join(args.output_dir, 'vector_fields.png'),
    )

    print(f"\n{'=' * 60}")
    print(f"All figures saved to: {os.path.abspath(args.output_dir)}/")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
