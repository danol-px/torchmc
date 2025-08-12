"""source: https://mc-stan.org/docs/2_21/reference-manual/hamiltonian-monte-carlo.html."""

from typing import Callable, Tuple

import torch


def vectorized_grad(position: torch.Tensor, log_prob_fn: Callable) -> torch.Tensor:
    """Compute gradient of negative log probability with respect to current parameters.

    Args:
        position: Parameter tensor (n_chains, n_dims)
        log_prob_fn: Function that computes log probability from position

    Returns:
        Gradient tensor (n_chains, n_dims)
    """
    position_grad = position.clone().requires_grad_()
    potential_energy = -log_prob_fn(position_grad)

    gradients = torch.autograd.grad(
        outputs=potential_energy,
        inputs=position_grad,
        grad_outputs=torch.ones_like(potential_energy),
    )[0]

    return gradients


def leapfrog(
    start_position: torch.Tensor,
    start_momentum: torch.Tensor,
    step_size: float,
    n_steps: int,
    log_prob_fn: Callable,
    inv_mass: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform leapfrog integration. Source: https://arxiv.org/pdf/1206.1901

    Args:
        start_position: Parameter tensor (n_chains, n_dims)
        start_momentum: Momentum tensor (n_chains, n_dims)
        step_size: Step size
        n_steps: Number of leapfrog steps
        log_prob_fn: Function that computes log probability
        inv_mass: Mass matrix for momentum sampling, can be diagonal (1d) or full (2d)

    Returns:
        Tuple of (q_new, p_new) after L steps of leapfrog integration
    """
    position = start_position.clone()
    momentum = start_momentum.clone()

    # Half-step update for momentum
    momentum -= 0.5 * step_size * vectorized_grad(position, log_prob_fn)

    # Alternate full position and momentum updates (updating position on half time steps for momentum)
    for i in range(n_steps):
        position += step_size * torch.einsum("bd, dd -> bd", momentum, inv_mass)

        if i < n_steps - 1:
            momentum -= step_size * vectorized_grad(position, log_prob_fn)

    # Final half-step and negation of momentum for symmetry
    momentum -= 0.5 * step_size * vectorized_grad(position, log_prob_fn)
    momentum = -momentum

    return position, momentum
