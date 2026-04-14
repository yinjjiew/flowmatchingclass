# Flow Matching 2D Experiments — Full Reproduction

Reproduction of the 2D checkerboard experiments from:
**"Flow Matching for Generative Modeling"** (Lipman et al., 2022) — [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)

This repo implements **three methods** on a 2D checkerboard distribution and compares their
trajectories, density evolution, and sampling efficiency (NFE ablation).

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt --break-system-packages

# Run all three experiments + generate all figures
python main.py

# Or run individual methods
python main.py --method fm_ot          # Flow Matching with OT path
python main.py --method fm_dif         # Flow Matching with VP Diffusion path
python main.py --method sm_dif         # Score Matching with VP Diffusion path

# Customize training
python main.py --epochs 20000 --lr 1e-3 --batch_size 4096 --hidden_dim 512 --n_layers 5
```

All figures are saved to `./outputs/`.

---

## Methods & Equations

All three methods share the same goal: learn a vector field $v_t(x;\theta)$ that transports a
simple prior $p_0 = \mathcal{N}(0, I)$ to the data distribution $q(x_1)$ (the checkerboard).
They differ in which **probability path** connects noise to data and which **training objective** is used.

### Common Setup

| Symbol | Meaning |
|--------|---------|
| $x_0 \sim p_0 = \mathcal{N}(0, I)$ | Noise sample |
| $x_1 \sim q(x_1)$ | Data sample (checkerboard) |
| $t \in [0, 1]$ | Time (0 = noise, 1 = data) |
| $v_t(x;\theta)$ | Learned vector field (neural net) |
| $\sigma_{\min} = 10^{-4}$ | Small std at $t=1$ |

### Neural Network Architecture (all methods)

- **MLP**: 5 hidden layers, 512 units each, SiLU activation
- **Input**: $(x, t) \in \mathbb{R}^3$ (2D point + scalar time)
- **Output**: $\mathbb{R}^2$ (predicted vector field or score)

---

## Method 1: Flow Matching with Optimal Transport Path (FM-OT)

> **Paper reference**: Section 4.1, Example II (Eq. 20–23), Figure 4

### Probability Path (Eq. 20)

The OT conditional probability path linearly interpolates mean and std:

$$\mu_t(x_1) = t \, x_1, \qquad \sigma_t(x_1) = 1 - (1 - \sigma_{\min}) t$$

So the conditional distribution is:

$$p_t(x | x_1) = \mathcal{N} \Big(x \;\Big|\; t \, x_1,\; \big[1 - (1-\sigma_{\min})t\big]^2 I\Big)$$

### Conditional Flow (Eq. 22)

The flow (affine map) that pushes $p_0$ to $p_t(\cdot|x_1)$:

$$\psi_t(x_0) = \big[1 - (1-\sigma_{\min})t\big] \, x_0 + t \, x_1$$

### Conditional Vector Field (Eq. 21)

Derived from Theorem 3 by plugging in the linear $\mu_t, \sigma_t$:

$$u_t(x | x_1) = \frac{x_1 - (1-\sigma_{\min})\,x}{1 - (1-\sigma_{\min})t}$$

### Training Target

For training, we only need the target at the interpolated point $\psi_t(x_0)$.
The CFM loss (Eq. 23) simplifies beautifully:

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, q(x_1), p(x_0)} \Big\| v_t \big(\psi_t(x_0);\theta\big) - \big(x_1 - (1-\sigma_{\min})\,x_0\big) \Big\|^2$$

**In code:**
```python
t = uniform(0, 1)
x0 = randn(...)                                          # noise
x1 = sample_checkerboard(...)                             # data
psi_t = (1 - (1 - sigma_min) * t) * x0 + t * x1          # interpolated point
target = x1 - (1 - sigma_min) * x0                        # constant direction
loss = ||v_theta(psi_t, t) - target||^2
```

**Key insight**: The regression target $x_1 - (1-\sigma_{\min})x_0$ is **independent of $t$** —
the OT vector field has constant direction in time (Figure 2 in the paper).

### Sampling

Solve the ODE forward from $x_0 \sim \mathcal{N}(0,I)$:

$$\frac{d}{dt} \phi_t(x) = v_t(\phi_t(x); \theta), \qquad \phi_0(x) = x_0$$

using any ODE solver (Euler, Midpoint, RK4, or adaptive dopri5).

---

## Method 2: Flow Matching with VP Diffusion Path (FM-Dif)

> **Paper reference**: Section 4.1, Example I (Eq. 18–19), Appendix E.1

### VP Diffusion Schedule

The Variance Preserving (VP) path is defined via the noise schedule:

$$\beta(s) = \beta_{\min} + s (\beta_{\max} - \beta_{\min}), \qquad \beta_{\min}=0.1,  \beta_{\max}=20$$

$$T(s) = \int_0^s \beta(r) dr = s \beta_{\min} + \tfrac{1}{2}s^2(\beta_{\max}-\beta_{\min})$$

$$\alpha_s = e^{-\frac{1}{2}T(s)}$$

### Probability Path (Eq. 18, reversed)

Using the paper's noise→data convention ($t=0$ noise, $t=1$ data):

$$\mu_t(x_1) = \alpha_{1-t}   x_1, \qquad \sigma_t(x_1) = \sqrt{1 - \alpha_{1-t}^2}$$

$$p_t(x|x_1) = \mathcal{N} \Big(x  \Big|  \alpha_{1-t}\,x_1,  (1-\alpha_{1-t}^2) I\Big)$$

At $t=0$: $\alpha_1 \approx 0$, so $\mu_0 \approx 0$, $\sigma_0 \approx 1$ (pure noise). 
At $t=1$: $\alpha_0 = 1$, so $\mu_1 = x_1$, $\sigma_1 = 0$ (pure data). 

### Conditional Flow (Eq. 11)

$$\psi_t(x_0) = \sigma_t(x_1)\,x_0 + \mu_t(x_1) = \sqrt{1-\alpha_{1-t}^2} x_0 + \alpha_{1-t} x_1$$

### Conditional Vector Field (Eq. 19, from Theorem 3)

$$u_t(x|x_1) = \frac{\sigma_t'(x_1)}{\sigma_t(x_1)} (x - \mu_t(x_1)) + \mu_t'(x_1)$$

Expanded using VP schedule (Eq. 19):

$$u_t(x|x_1) = -\frac{T'(1-t)}{2}\left[\frac{e^{-T(1-t)} x - e^{-\frac{1}{2}T(1-t)} x_1}{1 - e^{-T(1-t)}}\right]$$

where $T'(s) = \beta(s) = \beta_{\min} + s(\beta_{\max} - \beta_{\min})$.

### Training Loss (CFM, Eq. 14)

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, q(x_1), p(x_0)} \Big\| v_t \big(\psi_t(x_0);\theta\big) - \frac{d}{dt}\psi_t(x_0) \Big\|^2$$

**In code:**
```python
t = uniform(eps, 1 - eps)                                # avoid boundary singularity
s = 1 - t
T_s = s * beta_min + 0.5 * s**2 * (beta_max - beta_min)
alpha_s = exp(-0.5 * T_s)
sigma_t = sqrt(1 - alpha_s**2)
mu_t = alpha_s * x1

psi_t = sigma_t * x0 + mu_t                              # interpolated point

# Compute d/dt psi_t numerically or analytically
beta_s = beta_min + s * (beta_max - beta_min)
d_alpha_s = -0.5 * beta_s * alpha_s
# ds/dt = -1
d_mu_t = -d_alpha_s * x1                                 # = +0.5 * beta_s * alpha_s * x1
d_sigma_t = alpha_s * d_alpha_s / sigma_t                 # chain rule, then negate for ds/dt
d_sigma_t = -d_sigma_t
d_psi_t = d_sigma_t * x0 + d_mu_t                        # target vector field

loss = ||v_theta(psi_t, t) - d_psi_t||^2
```

### Sampling

Same ODE solver as FM-OT, but the learned vector field now approximates the diffusion-derived VF.

---

## Method 3: Score Matching with VP Diffusion Path (SM-Dif)

> **Paper reference**: Appendix E.1 (Eq. 42–46)

### Same VP Probability Path

Uses the identical VP path as FM-Dif above.

### Training: Learn the Score Function

Instead of learning a vector field directly, we learn the **score** $s_t(x;\theta) \approx \nabla_x \log p_t(x|x_1)$.

The conditional score of the Gaussian path is:

$$\nabla_x \log p_t(x|x_1) = -\frac{x - \mu_t(x_1)}{\sigma_t(x_1)^2}$$

### Score Matching Loss (Eq. 42–43)

$$\mathcal{L}_{\text{SM}}(\theta) = \mathbb{E}_{t, q(x_1), p_t(x|x_1)} \lambda(t) \Big\| s_t(x;\theta) - \nabla_x \log p_t(x|x_1) \Big\|^2$$

with weighting $\lambda(t) = \sigma_t^2$ (the standard SM weighting from Song & Ermon, 2019).

**In code:**
```python
t = uniform(eps, 1 - eps)
# ... compute mu_t, sigma_t same as FM-Dif ...
psi_t = sigma_t * x0 + mu_t
score_target = -(psi_t - mu_t) / sigma_t**2              # = -x0 / sigma_t
loss = sigma_t**2 * ||s_theta(psi_t, t) - score_target||^2
```

### Sampling: Convert Score to Vector Field (Eq. 46)

At inference, convert the learned score to a vector field for ODE sampling:

$$v_t(x) = -\frac{T'(1-t)}{2}\Big[s_t(x;\theta) - x\Big]$$

Then solve the same ODE as the other methods.

---

## Comparison Summary

| | FM-OT | FM-Dif | SM-Dif |
|---|---|---|---|
| **Probability path** | Linear interpolation (OT) | VP Diffusion | VP Diffusion |
| **What the network learns** | Vector field $v_t$ | Vector field $v_t$ | Score $s_t$ |
| **Training target** | $x_1 - (1-\sigma_{\min})x_0$ | $\frac{d}{dt}\psi_t(x_0)$ | $-x_0 / \sigma_t$ |
| **Target depends on $t$?** | **No** (constant direction) | Yes | Yes |
| **Trajectory shape** | Straight lines | Curved | Curved |
| **Expected NFE** | Lowest | Medium | Highest |

---

## Generated Figures — What Each Plot Shows

Running `python main.py` produces **8 figures** in `./outputs/`. Below is a detailed guide to each one: what it visualizes, how it's computed, and what to look for.

---

### 1. `figure4_left.png` — Density + Trajectory Overlay ⭐ (Paper Figure 4 left)

**Layout**: rows = methods (SM-Dif, FM-Dif, FM-OT), columns = time steps $t \in \{0, 0.2, 0.4, 0.6, 0.8, 1.0\}$.

**What it shows**: Each cell combines two layers:
- **Background (density heatmap)**: a 2D histogram (`hist2d`, 120×120 bins, `inferno` colormap, log-scale) of all sample positions at time $t$. This approximates the marginal distribution $p_t(x)$.
- **Foreground (trajectory lines)**: 150 randomly selected ODE trajectories drawn as colored line segments up to the current time. Color encodes time via viridis (purple=early → yellow=late). White dots mark current sample positions.

**How it's computed**: We solve the learned ODE $d\phi_t/dt = v_t(\phi_t; \theta)$ from $t=0$ to $t=1$ using the midpoint solver with 200 steps, recording all 201 intermediate states. Each time column slices into this trajectory tensor at the corresponding index.

**What to look for**:
- **FM-OT reveals the checkerboard pattern much earlier** (~$t=0.4$) than diffusion methods, which remain a Gaussian blob until $t \geq 0.8$. This is the paper's central visual claim.
- FM-OT trajectories are **straight lines**; diffusion trajectories are **curved** and may overshoot.
- The density at $t=0$ should be identical across all methods (same standard Gaussian prior).

---

### 2. `figure4_right.png` — Low-NFE Sample Quality ⭐ (Paper Figure 4 right)

**Layout**: rows = methods, columns = NFE values $\{4, 8, 10, 20\}$.

**What it shows**: Scatter plots of generated samples at a fixed computational budget (NFE = number of neural network forward passes), using the midpoint ODE solver.

**How it's computed**: Midpoint uses 2 function evaluations per step, so `n_steps = NFE // 2`. All methods start from the same shared noise $x_0 \sim \mathcal{N}(0, I)$ for fair comparison.

**What to look for**:
- **FM-OT at NFE=8** already shows clear checkerboard structure; diffusion methods are still blurry.
- At **NFE=20**, FM-OT is nearly perfect; diffusion methods are still rough.
- Straighter OT trajectories $\Rightarrow$ easier numerical integration $\Rightarrow$ fewer steps needed for good samples.

---

### 3. `density_evolution.png` — Extended Density Filmstrip (8 snapshots)

**Layout**: rows = methods, columns = 8 time steps $t \in \{0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0\}$.

**What it shows**: Pure density heatmaps (no trajectory overlay) at finer time resolution than Figure 4 left. Same `hist2d` + log-scale technique.

**How it's computed**: Same trajectory data, sliced at 8 evenly spaced time indices.

**What to look for**: A smoother "filmstrip" of how $p_t$ evolves. Helps pinpoint the exact time when structure first appears for each method. FM-OT's linear interpolation path means structure emerges roughly proportional to $t$.

---

### 4. `trajectories_only.png` — Trajectory Geometry (supplementary)

**Layout**: one panel per method, side by side.

**What it shows**: 300 ODE sampling paths from noise to data, **without** density background. Line color = time (viridis), blue dots = start ($t=0$), red dots = end ($t=1$).

**How it's computed**: Draws `LineCollection` segments for each trajectory, colored by time step.

**What to look for**: The clearest view of path geometry. FM-OT paths are nearly parallel straight lines; diffusion paths curve, spread, and may cross each other. Diffusion trajectories sometimes "overshoot" and backtrack (Figure 3 in the paper).

---

### 5. `nfe_comparison.png` — Quantitative NFE Tradeoff Curves

**Layout**: one subplot per method with 3 curves (Euler, Midpoint, RK4).

**What it shows**: Sample quality ($y$-axis: fraction of samples in correct checkerboard cells) vs. NFE ($x$-axis) for three ODE solver families.

**How it's computed**: For each `(method, solver, n_steps)` triple, generate 2048 samples and compute the quality metric. NFE = `n_steps × evals_per_step` (Euler: ×1, Midpoint: ×2, RK4: ×4).

**Quality metric**: We divide $[-2, 2]^2$ into a $4 \times 4$ grid and count what fraction of generated samples land in "black" cells (those where `(row + col) % 2 == 0`). A perfect model scores 1.0. This is a 2D proxy for FID.

**What to look for**: FM-OT dominates at every NFE budget. The gap is largest at low NFE ($\leq 20$). Higher-order solvers are more efficient per-NFE.

---

### 6. `training_curves.png` — Training Loss

**What it shows**: Training loss vs. step for all 3 methods on log scale.

**How it's computed**: Loss is recorded every 100 steps. Note: FM-OT/FM-Dif minimize CFM loss (Eq. 23/14); SM-Dif minimizes $\sigma_t^2$-weighted score matching loss (Eq. 42). Different loss scales are expected.

**What to look for**: FM-OT typically converges fastest and smoothest (its target $x_1 - (1-\sigma_{\min})x_0$ has constant magnitude across $t$). SM-Dif may be noisier or higher in absolute value.

---

### 7. `samples_dopri5.png` — Best-Quality Samples (adaptive solver)

**What it shows**: Scatter plots of samples using the dopri5 adaptive solver (`atol=rtol=1e-5`).

**How it's computed**: `scipy.integrate.solve_ivp` with RK45 method. The solver adaptively chooses step sizes to keep integration error below tolerance.

**What to look for**: These represent each model's **best achievable quality** (no solver error). Any imperfections (bleeding between cells, density non-uniformity, out-of-bounds points) are due to the model itself. Reports the average NFE used by the adaptive solver — FM-OT typically needs fewer.

---

### 8. `vector_fields.png` — Learned Vector Fields (cf. Paper Figure 2/8)

**Layout**: rows = methods, columns = time steps ($t = 0, 1/3, 2/3$).

**What it shows**: Quiver (arrow) plots of the learned velocity field $v_t(x; \theta)$ on a $25 \times 25$ grid. Arrow direction = flow direction; arrow color = magnitude (`coolwarm`: blue=small, red=large).

**How it's computed**: Evaluate the neural network on a uniform grid of query points at each time $t$.

**What to look for**:
- FM-OT arrows maintain **consistent direction** across all time columns (the constant-direction property from Eq. 21).
- Diffusion VFs show **dramatic direction changes** over time — the field does most of its "work" near $t=1$.
- Magnitude contrast: FM-OT arrows have more uniform magnitude (constant speed), while diffusion arrows vary wildly.

---

## Hyperparameters (from paper, Appendix E)

| Parameter | Value |
|-----------|-------|
| MLP layers | 5 |
| Hidden dim | 512 |
| Activation | SiLU |
| Optimizer | Adam ($\beta_1=0.9, \beta_2=0.999$) |
| Learning rate | $5 \times 10^{-4}$ |
| Batch size | 4096 |
| Training steps | 20,000 |
| $\sigma_{\min}$ | $10^{-4}$ |
| $\beta_{\min}$ (VP) | 0.1 |
| $\beta_{\max}$ (VP) | 20.0 |
| $\varepsilon$ (VP time clipping) | $10^{-5}$ |

---

## Project Structure

```
flow-matching-2d/
├── README.md              # This file
├── requirements.txt       # Dependencies
├── main.py                # Entry point — train + visualize
├── model.py               # MLP architecture
├── dataset.py             # Checkerboard dataset
├── methods.py             # FM-OT, FM-Dif, SM-Dif training logic
├── samplers.py            # ODE solvers (Euler, Midpoint, RK4, dopri5)
├── visualize.py           # All plotting functions
└── outputs/               # Generated figures (created at runtime)
    ├── figure4_left.png       # ⭐ Paper Fig 4 left: density + trajectory overlay
    ├── figure4_right.png      # ⭐ Paper Fig 4 right: low-NFE sample scatter
    ├── density_evolution.png  # Extended density filmstrip (8 time steps)
    ├── trajectories_only.png  # Trajectory-only view (no density background)
    ├── nfe_comparison.png     # Quantitative NFE vs quality curves
    ├── training_curves.png    # Training loss over iterations
    ├── samples_dopri5.png     # Best-quality samples (adaptive solver)
    ├── vector_fields.png      # Learned vector field quiver plots
    └── checkpoint_*.pt        # Model checkpoints
```

---

## References

- Lipman, Y., Chen, R.T.Q., Ben-Hamu, H., Nickel, M., & Le, M. (2022).
  *Flow Matching for Generative Modeling.* arXiv:2210.02747.
- Liu, X., Gong, C., & Liu, Q. (2022). *Rectified Flow.* arXiv:2209.03003.
- Song, Y., et al. (2020). *Score-Based Generative Modeling through SDEs.* arXiv:2011.13456.
