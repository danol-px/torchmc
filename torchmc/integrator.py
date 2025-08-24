"""source: https://mc-stan.org/docs/2_21/reference-manual/hamiltonian-monte-carlo.html."""

from typing import Callable, Tuple

import torch


def score(position: torch.Tensor, log_prob_fn: Callable) -> torch.Tensor:
    """Compute vectorized negative score.

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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform vectorized leapfrog integration. Source: https://arxiv.org/pdf/1206.1901

    Args:
        start_position: Parameter tensor (n_chains, n_dims)
        start_momentum: Momentum tensor (n_chains, n_dims)
        step_size: Step size
        n_steps: Number of leapfrog steps
        log_prob_fn: Function that computes log probability

    Returns:
        Tuple of (q_new, p_new) after L steps of leapfrog integration
    """
    position = start_position.clone()
    momentum = start_momentum.clone()

    momentum -= 0.5 * step_size * score(position, log_prob_fn)

    for i in range(n_steps):
        position += step_size * momentum

        if i < n_steps - 1:
            momentum -= step_size * score(position, log_prob_fn)

    momentum -= 0.5 * step_size * score(position, log_prob_fn)
    momentum = -momentum

    return position, momentum
