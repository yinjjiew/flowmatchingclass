"""
Checkerboard 2D dataset — the target distribution q(x1).

Generates samples from a 4x4 checkerboard pattern on [-2, 2]^2,
matching the setup in Figure 4 of the Flow Matching paper.
"""

import torch
import numpy as np


def sample_checkerboard(n: int, board_size: int = 4, scale: float = 4.0) -> torch.Tensor:
    """
    Sample n points from a 2D checkerboard distribution.

    The checkerboard has `board_size x board_size` cells on [-scale/2, scale/2]^2.
    Points are sampled uniformly from the "black" cells.

    Args:
        n: Number of samples.
        board_size: Number of cells per side (default 4).
        scale: Total width of the board (default 4.0, so range is [-2, 2]).

    Returns:
        Tensor of shape (n, 2).
    """
    cell_size = scale / board_size

    # Enumerate black cells: (i, j) where (i + j) % 2 == 0
    black_cells = []
    for i in range(board_size):
        for j in range(board_size):
            if (i + j) % 2 == 0:
                black_cells.append((i, j))

    # Pick random black cells for each sample
    cell_indices = np.random.randint(0, len(black_cells), size=n)

    # Sample uniformly within chosen cells
    samples = np.zeros((n, 2))
    for k in range(n):
        i, j = black_cells[cell_indices[k]]
        x = -scale / 2 + i * cell_size + np.random.uniform(0, cell_size)
        y = -scale / 2 + j * cell_size + np.random.uniform(0, cell_size)
        samples[k] = [x, y]

    return torch.tensor(samples, dtype=torch.float32)


def sample_noise(n: int, dim: int = 2) -> torch.Tensor:
    """Sample from p0 = N(0, I)."""
    return torch.randn(n, dim)
