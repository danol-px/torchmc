"""HMC integrators, source: https://mc-stan.org/docs/2_21/reference-manual/hamiltonian-monte-carlo.html."""

from typing import Callable, Tuple

import torch
import random


class Integrator:
    """Integrator base class."""

    def vectorized_grad(
        self, position: torch.Tensor, log_prob_fn: Callable
    ) -> torch.Tensor:
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

    def sample_momentum(
        self, n_chains: int, n_dims: int, inv_mass: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Sample momentum.

        Args:
            n_chains: Number of chains
            n_dims: Dimensionality of the parameter space
            inv_mass: Mass matrix for momentum sampling
            device: Device to run computations on

        Returns:
            Sampled momentum tensor (n_chains, n_dims)
        """
        return torch.randn(n_chains, n_dims, device=device) @ torch.linalg.cholesky(
            inv_mass
        )

    def step(*args, **kwargs):
        """Abstract method for performing a step of the integrator."""
        raise NotImplementedError("Integrator step must be implemented in subclasses.")


class LeapfrogIntegrator(Integrator):
    """Leapfrog integrator for Hamiltonian dynamics."""

    def step(
        self,
        start_position: torch.Tensor,
        eps: float,
        L: int,
        log_prob_fn: Callable,
        inv_mass: torch.Tensor,
        eps_jitter: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform leapfrog integration.

        Source: https://arxiv.org/pdf/1206.1901

        Args:
            start_position: Parameter tensor (n_chains, n_dims)
            eps: Step size scalar
            L: Number of leapfrog steps
            log_prob_fn: Function that computes log probability
            mass_matrix: Mass matrix for momentum sampling, can be diagonal (1d) or full (2d)
            eps_jitter: [0, 1] float indicating proportion that may be added to or subtracted from the step size

        Returns:
            Tuple of (q_new, p_new) after L steps of leapfrog integration
        """
        position = start_position.clone()
        momentum = self.sample_momentum(*position.shape, inv_mass, position.device)

        if eps_jitter > 0:
            if eps_jitter > 1:
                raise ValueError("Step size jitter must be in the range [0, 1]")
            eps *= random.uniform(1 - eps_jitter, 1 + eps_jitter)

        # Half-step update for momentum
        momentum -= 0.5 * eps * self.vectorized_grad(position, log_prob_fn)

        # Alternate full position and momentum updates (updating position on half time steps for momentum)
        for i in range(L):
            position += eps * torch.einsum("bd, dd -> bd", momentum, inv_mass)

            if i < L - 1:
                momentum -= eps * self.vectorized_grad(position, log_prob_fn)

        # Final half-step and negation of momentum for symmetry
        momentum -= 0.5 * eps * self.vectorized_grad(position, log_prob_fn)
        momentum = -momentum

        return position, momentum
