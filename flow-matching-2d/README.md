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

$$p_t(x | x_1) = \mathcal{N}\!\Big(x \;\Big|\; t \, x_1,\; \big[1 - (1-\sigma_{\min})t\big]^2 I\Big)$$

### Conditional Flow (Eq. 22)

The flow (affine map) that pushes $p_0$ to $p_t(\cdot|x_1)$:

$$\psi_t(x_0) = \big[1 - (1-\sigma_{\min})t\big] \, x_0 + t \, x_1$$

### Conditional Vector Field (Eq. 21)

Derived from Theorem 3 by plugging in the linear $\mu_t, \sigma_t$:

$$u_t(x | x_1) = \frac{x_1 - (1-\sigma_{\min})\,x}{1 - (1-\sigma_{\min})t}$$

### Training Target

For training, we only need the target at the interpolated point $\psi_t(x_0)$.
The CFM loss (Eq. 23) simplifies beautifully:

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, q(x_1), p(x_0)} \Big\| v_t\!\big(\psi_t(x_0);\theta\big) - \big(x_1 - (1-\sigma_{\min})\,x_0\big) \Big\|^2$$

**In code:**
```python
t = uniform(0, 1)
x0 = randn(...)                                          # noise
x1 = sample_checkerboard(...)                             # data
psi_t = (1 - (1 - sigma_min) * t) * x0 + t * x1          # interpolated point
target = x1 - (1 - sigma_min) * x0                        # constant direction!
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

$$\beta(s) = \beta_{\min} + s\,(\beta_{\max} - \beta_{\min}), \qquad \beta_{\min}=0.1,\; \beta_{\max}=20$$

$$T(s) = \int_0^s \beta(r)\,dr = s\,\beta_{\min} + \tfrac{1}{2}s^2(\beta_{\max}-\beta_{\min})$$

$$\alpha_s = e^{-\frac{1}{2}T(s)}$$

### Probability Path (Eq. 18, reversed)

Using the paper's noise→data convention ($t=0$ noise, $t=1$ data):

$$\mu_t(x_1) = \alpha_{1-t} \, x_1, \qquad \sigma_t(x_1) = \sqrt{1 - \alpha_{1-t}^2}$$

$$p_t(x|x_1) = \mathcal{N}\!\Big(x \;\Big|\; \alpha_{1-t}\,x_1,\; (1-\alpha_{1-t}^2)\,I\Big)$$

At $t=0$: $\alpha_1 \approx 0$, so $\mu_0 \approx 0$, $\sigma_0 \approx 1$ (pure noise). ✓
At $t=1$: $\alpha_0 = 1$, so $\mu_1 = x_1$, $\sigma_1 = 0$ (pure data). ✓

### Conditional Flow (Eq. 11)

$$\psi_t(x_0) = \sigma_t(x_1)\,x_0 + \mu_t(x_1) = \sqrt{1-\alpha_{1-t}^2}\;x_0 + \alpha_{1-t}\,x_1$$

### Conditional Vector Field (Eq. 19, from Theorem 3)

$$u_t(x|x_1) = \frac{\sigma_t'(x_1)}{\sigma_t(x_1)}\,(x - \mu_t(x_1)) + \mu_t'(x_1)$$

Expanded using VP schedule (Eq. 19):

$$u_t(x|x_1) = -\frac{T'(1-t)}{2}\left[\frac{e^{-T(1-t)}\,x - e^{-\frac{1}{2}T(1-t)}\,x_1}{1 - e^{-T(1-t)}}\right]$$

where $T'(s) = \beta(s) = \beta_{\min} + s(\beta_{\max} - \beta_{\min})$.

### Training Loss (CFM, Eq. 14)

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, q(x_1), p(x_0)} \Big\| v_t\!\big(\psi_t(x_0);\theta\big) - \frac{d}{dt}\psi_t(x_0) \Big\|^2$$

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

$$\mathcal{L}_{\text{SM}}(\theta) = \mathbb{E}_{t, q(x_1), p_t(x|x_1)} \;\lambda(t)\;\Big\| s_t(x;\theta) - \nabla_x \log p_t(x|x_1) \Big\|^2$$

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

## Expected Outputs

Running `python main.py` generates these figures in `./outputs/`:

| Figure | Description | Paper Reference |
|--------|-------------|-----------------|
| `trajectories_comparison.png` | Side-by-side sampling trajectories for all 3 methods | Figure 4 (left) |
| `density_evolution_*.png` | $p_t$ at $t=0, 1/3, 2/3, 1$ for each method | Figure 2 |
| `nfe_comparison.png` | Sample quality vs NFE using Euler/Midpoint/RK4 | Figure 4 (right) |
| `training_curves.png` | Training loss over iterations for all methods | — |
| `samples_*.png` | Final generated samples for each method | — |

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
```

---

## References

- Lipman, Y., Chen, R.T.Q., Ben-Hamu, H., Nickel, M., & Le, M. (2022).
  *Flow Matching for Generative Modeling.* arXiv:2210.02747.
- Liu, X., Gong, C., & Liu, Q. (2022). *Rectified Flow.* arXiv:2209.03003.
- Song, Y., et al. (2020). *Score-Based Generative Modeling through SDEs.* arXiv:2011.13456.
