"""Flow matching utilities for Flow-HWM.

Implements the flow matching framework from Section 2.3 of the paper:
- Path construction: X_t = t*X_1 + (1 - (1 - sigma_min)*t)*X_0
- Velocity computation: V_t = X_1 - (1 - sigma_min)*X_0
- Loss computation: MSE between predicted and target velocity

Flow matching (Lipman et al., 2023; Albergo et al., 2023) learns a velocity
field that transforms samples from a simple prior (Gaussian noise) into
samples from the target distribution. Unlike diffusion models that learn
a reversed stochastic process, flow matching directly learns a deterministic
velocity field via ODE integration.

Key equations from Section 2.3:
    X_t = t*X_1 + (1 - (1 - sigma_min)*t)*X_0  ... (1)
    V_t = dX_t/dt = X_1 - (1 - sigma_min)*X_0  ... (2)
    Loss = E[||u_theta(X_t, P, t) - V_t||^2]   ... (3)

where:
    X_0 ~ N(0, I) is Gaussian noise (no corruption or masking)
    X_1 is the target video latent
    t in [0, 1] is the timestep
    sigma_min is a small constant (0.001) ensuring non-zero support at t=1
    P = {v_p, a_p, a_f} is the conditioning context

Key differences from Masked-HWM:
    - No Copilot-4D-style corruption or random token replacement.
    - No masking or cosine schedule (continuous latents only).
    - No timestep-dependent noise schedule beyond the linear path.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional


def construct_flow_path(
    x0: Tensor,
    x1: Tensor,
    t: Tensor,
    sigma_min: float = 0.001,
) -> Tensor:
    """Construct sample X_t along the flow path.

    Implements Equation (1) from Section 2.3:
        X_t = t*X_1 + (1 - (1 - sigma_min)*t)*X_0

    At t=0: X_0 = noise
    At t=1: X_1 = target (with small noise sigma_min*X_0)

    Args:
        x0: Noise sample from N(0, I), shape (B, C, T, H, W) or (B, ...)
        x1: Target sample, same shape as x0
        t: Timesteps in [0, 1], shape (B,) or scalar
        sigma_min: Small constant for numerical stability (default: 0.001)

    Returns:
        X_t: Interpolated sample at time t, same shape as x0
    """
    # Ensure t has the right shape for broadcasting
    if t.dim() == 0:
        t = t.unsqueeze(0)

    # Reshape t for broadcasting to match x0/x1 dimensions
    # t: (B,) -> (B, 1, 1, 1, 1) for 5D tensors
    while t.dim() < x0.dim():
        t = t.unsqueeze(-1)

    # Compute X_t = t*X_1 + (1 - (1 - sigma_min)*t)*X_0
    # Equivalent to: X_t = t*X_1 + (1 - t + sigma_min*t)*X_0
    #              = t*X_1 + (1 - t*(1 - sigma_min))*X_0
    x_t = t * x1 + (1 - (1 - sigma_min) * t) * x0

    return x_t


def compute_target_velocity(
    x0: Tensor,
    x1: Tensor,
    sigma_min: float = 0.001,
) -> Tensor:
    """Compute target velocity V_t = dX_t/dt.

    Implements Equation (2) from Section 2.3:
        V_t = X_1 - (1 - sigma_min)*X_0

    Note: The velocity is constant along the path (does not depend on t).
    This is a property of the affine interpolation used.

    Args:
        x0: Noise sample from N(0, I)
        x1: Target sample
        sigma_min: Small constant (default: 0.001)

    Returns:
        V_t: Target velocity, same shape as x0
    """
    return x1 - (1 - sigma_min) * x0


def flow_matching_loss(
    predicted_velocity: Tensor,
    target_velocity: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """Compute flow matching loss (MSE).

    Implements Equation (3) from Section 2.3:
        Loss = E[||u_theta(X_t, P, t) - V_t||^2]

    Args:
        predicted_velocity: Model prediction, shape (B, C, T, H, W)
        target_velocity: Ground truth velocity, same shape
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Loss value (scalar if reduction='mean' or 'sum')
    """
    return F.mse_loss(predicted_velocity, target_velocity, reduction=reduction)


def sample_timesteps(
    batch_size: int,
    device: torch.device,
    min_t: float = 0.0,
    max_t: float = 1.0,
) -> Tensor:
    """Sample random timesteps for training.

    Samples t uniformly from [min_t, max_t].

    Args:
        batch_size: Number of timesteps to sample
        device: Device to create tensor on
        min_t: Minimum timestep (default: 0.0)
        max_t: Maximum timestep (default: 1.0)

    Returns:
        Timesteps tensor of shape (batch_size,)
    """
    return torch.rand(batch_size, device=device) * (max_t - min_t) + min_t


def sample_noise(
    shape: Tuple[int, ...],
    device: torch.device,
    std: float = 1.0,
) -> Tensor:
    """Sample Gaussian noise X_0 ~ N(0, std^2).

    IMPORTANT: The noise std should match the data distribution.
    For latents normalized to [-1, 1] with std ~0.5, use std=0.5.

    Args:
        shape: Shape of noise tensor
        device: Device to create tensor on
        std: Standard deviation of noise (default: 1.0, use 0.5 for normalized latents)

    Returns:
        Noise tensor of given shape with specified std
    """
    return torch.randn(shape, device=device) * std


class FlowMatchingTrainer:
    """Helper class for flow matching training step.

    Encapsulates the flow matching training logic:
    1. Sample timesteps t ~ U(0, 1)
    2. Sample noise x0 ~ N(0, noise_std^2)
    3. Construct X_t and target V_t
    4. Compute loss

    Args:
        sigma_min: Small constant for flow path
        noise_std: Std of noise distribution (should match data std, ~0.5 for normalized latents)
        cfg_drop_prob: Probability of dropping conditioning for CFG
    """

    def __init__(
        self,
        sigma_min: float = 0.001,
        noise_std: float = 0.5,
        cfg_drop_prob: float = 0.1,
    ):
        self.sigma_min = sigma_min
        self.noise_std = noise_std
        self.cfg_drop_prob = cfg_drop_prob

    def prepare_training_inputs(
        self,
        x1: Tensor,
        condition_tensors: Optional[list] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Prepare inputs for a training step.

        Args:
            x1: Target samples (B, C, T, H, W)
            condition_tensors: List of conditioning tensors to potentially drop

        Returns:
            Tuple of (x_t, target_velocity, t, drop_mask)
        """
        B = x1.shape[0]
        device = x1.device

        # Sample timesteps
        t = sample_timesteps(B, device)

        # Sample noise with std matching data distribution
        x0 = sample_noise(x1.shape, device, std=self.noise_std)

        # Construct X_t
        x_t = construct_flow_path(x0, x1, t, self.sigma_min)

        # Compute target velocity
        target_v = compute_target_velocity(x0, x1, self.sigma_min)

        # CFG: determine which samples should have conditioning dropped
        drop_mask = torch.rand(B, device=device) < self.cfg_drop_prob

        return x_t, target_v, t, drop_mask

    def compute_loss(
        self,
        predicted_velocity: Tensor,
        target_velocity: Tensor,
    ) -> Tensor:
        """Compute training loss.

        Args:
            predicted_velocity: Model output
            target_velocity: Target from compute_target_velocity

        Returns:
            Scalar loss value
        """
        return flow_matching_loss(predicted_velocity, target_velocity)


def euler_step(
    x: Tensor,
    velocity: Tensor,
    dt: float,
) -> Tensor:
    """Single Euler ODE integration step.

    Updates x according to: x_{t+dt} = x_t + dt * velocity

    Args:
        x: Current sample at time t
        velocity: Velocity field at current point
        dt: Time step size

    Returns:
        Updated sample at time t + dt
    """
    return x + dt * velocity


@torch.no_grad()
def euler_integrate(
    velocity_fn,
    x_init: Tensor,
    num_steps: int = 50,
    t_start: float = 0.0,
    t_end: float = 1.0,
    verbose: bool = False,
) -> Tensor:
    """Integrate ODE using Euler method.

    Solves dx/dt = velocity_fn(x, t) from t_start to t_end.

    Args:
        velocity_fn: Function that takes (x, t) and returns velocity
        x_init: Initial sample at t_start (typically Gaussian noise)
        num_steps: Number of integration steps
        t_start: Starting time (default: 0.0)
        t_end: Ending time (default: 1.0)
        verbose: Print progress

    Returns:
        Final sample at t_end
    """
    dt = (t_end - t_start) / num_steps
    x = x_init.clone()
    B = x.shape[0]
    device = x.device

    for step in range(num_steps):
        t = t_start + step * dt
        t_tensor = torch.full((B,), t, device=device)

        # Get velocity at current point
        velocity = velocity_fn(x, t_tensor)

        # Euler step
        x = euler_step(x, velocity, dt)

        if verbose and (step + 1) % 10 == 0:
            print(f"  Step {step + 1}/{num_steps}, t = {t + dt:.3f}")

    return x
