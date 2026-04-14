"""
Neural network architecture for the vector field / score function.

Paper spec (Appendix E): MLP with 5 hidden layers of 512 neurons, SiLU activation.
Input: (x, t) ∈ R^3  |  Output: R^2
"""

import torch
import torch.nn as nn


class VectorFieldMLP(nn.Module):
    """
    Time-conditioned MLP that predicts a 2D vector field or score.

    Architecture:
        [x (2D) ; t (1D)] → Linear → SiLU → ... → Linear → output (2D)

    Args:
        input_dim: Spatial dimension (default 2).
        hidden_dim: Width of hidden layers (default 512).
        n_layers: Number of hidden layers (default 5).
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 512, n_layers: int = 5):
        super().__init__()

        layers = []
        # Input layer: dim + 1 (for time)
        layers.append(nn.Linear(input_dim + 1, hidden_dim))
        layers.append(nn.SiLU())

        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())

        # Output layer
        layers.append(nn.Linear(hidden_dim, input_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 2) spatial coordinates.
            t: (batch, 1) or (batch,) time values in [0, 1].

        Returns:
            (batch, 2) predicted vector field.
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (batch,) -> (batch, 1)
        xt = torch.cat([x, t], dim=-1)  # (batch, 3)
        return self.net(xt)
