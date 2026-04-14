"""
Three training methods from the Flow Matching paper.

Each method implements:
  - compute_loss(model, x1, x0, t) -> scalar loss
  - get_velocity(model, x, t)      -> vector field for ODE sampling

==========================================================================
Method 1: FM-OT   — Flow Matching with Optimal Transport path  (Eq. 20-23)
Method 2: FM-Dif  — Flow Matching with VP Diffusion path       (Eq. 18-19)
Method 3: SM-Dif  — Score Matching with VP Diffusion path      (Eq. 42-46)
==========================================================================
"""

import torch
import math


# ---------------------------------------------------------------------------
# VP Diffusion schedule helpers (shared by FM-Dif and SM-Dif)
# ---------------------------------------------------------------------------

BETA_MIN = 0.1
BETA_MAX = 20.0
EPS = 1e-5  # time clipping to avoid singularity at sigma=0


def _beta(s: torch.Tensor) -> torch.Tensor:
    """β(s) = β_min + s(β_max - β_min)"""
    return BETA_MIN + s * (BETA_MAX - BETA_MIN)


def _T(s: torch.Tensor) -> torch.Tensor:
    """T(s) = s·β_min + ½s²(β_max - β_min)"""
    return s * BETA_MIN + 0.5 * s ** 2 * (BETA_MAX - BETA_MIN)


def _alpha(s: torch.Tensor) -> torch.Tensor:
    """α_s = exp(-½ T(s))"""
    return torch.exp(-0.5 * _T(s))


def _vp_schedule(t: torch.Tensor):
    """
    Compute μ_t, σ_t and their time derivatives for the reversed VP path.

    Convention: t=0 is noise, t=1 is data. We define s = 1 - t.

    Returns: (mu_coeff, sigma, d_mu_coeff, d_sigma)
        where mu_t(x1) = mu_coeff * x1, sigma_t = sigma
    """
    s = 1.0 - t  # s goes from 1 (noise end) to 0 (data end)

    # Clamp s to avoid numerical issues at s=0
    s = s.clamp(min=EPS, max=1.0 - EPS)

    alpha_s = _alpha(s)
    sigma = torch.sqrt(1.0 - alpha_s ** 2)

    # Derivatives w.r.t. t (using ds/dt = -1):
    # dα_s/ds = -½ β(s) α_s
    # dα_s/dt = -dα_s/ds = ½ β(s) α_s
    beta_s = _beta(s)
    d_alpha_s_dt = 0.5 * beta_s * alpha_s

    # dσ/dt = d/dt sqrt(1 - α_s²) = -α_s · (dα_s/dt) / σ
    # Wait: σ² = 1 - α² → 2σ dσ/dt = -2α dα/dt → dσ/dt = -α dα/dt / σ
    # But dα/dt here is dα_s/dt = +0.5 β(s) α_s (since ds/dt = -1 flips sign)
    d_sigma_dt = -alpha_s * d_alpha_s_dt / sigma

    return alpha_s, sigma, d_alpha_s_dt, d_sigma_dt


# ===========================================================================
# Method 1: Flow Matching with Optimal Transport Path
# ===========================================================================

class FlowMatchingOT:
    """
    FM-OT: Conditional Flow Matching with OT displacement interpolant.

    Probability path (Eq. 20):
        μ_t(x₁) = t·x₁
        σ_t(x₁) = 1 - (1 - σ_min)t

    Conditional flow (Eq. 22):
        ψ_t(x₀) = [1 - (1-σ_min)t]·x₀ + t·x₁

    Training target (Eq. 23):
        x₁ - (1 - σ_min)·x₀   (independent of t!)

    The key property: OT paths produce STRAIGHT trajectories with
    CONSTANT velocity, making the regression target simpler.
    """

    def __init__(self, sigma_min: float = 1e-4):
        self.sigma_min = sigma_min
        self.name = "FM-OT"

    def compute_loss(self, model, x1, x0, t):
        """
        CFM loss (Eq. 23):
        L = E || v_θ(ψ_t(x₀), t) - (x₁ - (1-σ_min)x₀) ||²

        Args:
            model: VectorFieldMLP
            x1: (B, 2) data samples from checkerboard
            x0: (B, 2) noise samples from N(0, I)
            t:  (B, 1) uniform times in [0, 1]
        """
        # Conditional flow: ψ_t(x₀) = (1 - (1-σ_min)t)·x₀ + t·x₁
        psi_t = (1.0 - (1.0 - self.sigma_min) * t) * x0 + t * x1

        # Target: x₁ - (1-σ_min)·x₀  — note: INDEPENDENT of t
        target = x1 - (1.0 - self.sigma_min) * x0

        # Predict and compute MSE
        pred = model(psi_t, t)
        loss = (pred - target).pow(2).mean()
        return loss

    def get_velocity(self, model, x, t):
        """Return v_t(x; θ) directly — the model IS the vector field."""
        return model(x, t)


# ===========================================================================
# Method 2: Flow Matching with VP Diffusion Path
# ===========================================================================

class FlowMatchingDiffusion:
    """
    FM-Dif: Conditional Flow Matching with Variance Preserving diffusion path.

    Probability path (Eq. 18, reversed):
        μ_t(x₁) = α_{1-t} · x₁
        σ_t(x₁) = √(1 - α²_{1-t})

    Conditional flow (Eq. 11):
        ψ_t(x₀) = σ_t · x₀ + μ_t · x₁

    Training target (Eq. 14):
        d/dt ψ_t(x₀) = (dσ_t/dt)·x₀ + (dμ_t/dt)·x₁

    Conditional VF (Eq. 19, Theorem 3):
        u_t(x|x₁) = (σ'_t/σ_t)(x - μ_t·x₁) + μ'_t·x₁
    """

    def __init__(self):
        self.name = "FM-Dif"

    def compute_loss(self, model, x1, x0, t):
        """
        CFM loss (Eq. 14):
        L = E || v_θ(ψ_t(x₀), t) - d/dt ψ_t(x₀) ||²
        """
        alpha_s, sigma, d_alpha_dt, d_sigma_dt = _vp_schedule(t)

        # Conditional flow: ψ_t(x₀) = σ_t·x₀ + α_{1-t}·x₁
        psi_t = sigma * x0 + alpha_s * x1

        # Target: d/dt ψ_t(x₀) = (dσ/dt)·x₀ + (dα/dt)·x₁
        target = d_sigma_dt * x0 + d_alpha_dt * x1

        pred = model(psi_t, t)
        loss = (pred - target).pow(2).mean()
        return loss

    def get_velocity(self, model, x, t):
        """Return v_t(x; θ) directly — the model IS the vector field."""
        return model(x, t)


# ===========================================================================
# Method 3: Score Matching with VP Diffusion Path
# ===========================================================================

class ScoreMatchingDiffusion:
    """
    SM-Dif: Denoising Score Matching with VP diffusion path.

    Same probability path as FM-Dif.

    The network learns the SCORE s_t(x; θ) ≈ ∇_x log p_t(x|x₁).

    Training target (Eq. 43):
        ∇_x log p_t(x|x₁) = -(x - μ_t·x₁) / σ_t²

    Score Matching loss (Eq. 42, with λ(t) = σ_t²):
        L = E[ σ_t² · || s_θ(x, t) + (x - μ_t·x₁)/σ_t² ||² ]
          = E[ || s_θ(x, t)·σ_t + (x - μ_t·x₁)/σ_t ||² ]    (simplified)

    We use the equivalent denoising formulation:
        L = E[ || s_θ(ψ_t(x₀), t) + x₀/σ_t ||² ]

    Sampling VF (Eq. 46):
        v_t(x) = -T'(1-t)/2 · [s_t(x; θ) - x]
    """

    def __init__(self):
        self.name = "SM-Dif"

    def compute_loss(self, model, x1, x0, t):
        """
        Score Matching loss with λ(t) = σ_t² weighting.

        Using the identity: (ψ_t - μ_t·x₁)/σ_t² = x₀/σ_t

        L = E[ σ_t² · || s_θ(ψ_t, t) - (-x₀/σ_t) ||² ]
          = E[ || σ_t · s_θ(ψ_t, t) + x₀ ||² ]    (after simplification)

        This is equivalent to the standard denoising score matching loss.
        """
        alpha_s, sigma, _, _ = _vp_schedule(t)

        # Interpolated point
        psi_t = sigma * x0 + alpha_s * x1

        # Score target: ∇log p_t(x|x₁) = -(x - μ_t x₁)/σ_t² = -x₀/σ_t
        # With λ(t) = σ_t², the weighted loss becomes:
        # σ_t² || s_θ - (-x₀/σ_t) ||² = || σ_t s_θ + x₀ ||²
        pred_score = model(psi_t, t)
        loss = (sigma * pred_score + x0).pow(2).mean()
        return loss

    def get_velocity(self, model, x, t):
        """
        Convert learned score to vector field for ODE sampling (Eq. 46):
            v_t(x) = -T'(1-t)/2 · [s_t(x) - x]

        where T'(s) = β(s).
        """
        s = (1.0 - t).clamp(min=EPS, max=1.0 - EPS)
        beta_s = _beta(s)

        score = model(x, t)
        velocity = 0.5 * beta_s * (x + score)
        return velocity
