"""HMC samplers."""

from typing import Any, Callable, Dict, Tuple

import torch

from integrators import Integrator, LeapfrogIntegrator
from tqdm import tqdm


class HMC:
    """Vectorized Hamiltonian Monte Carlo sampler built on pytorch."""

    def __init__(
        self,
        log_prob_fn: Callable[[torch.Tensor], torch.Tensor],
        integrator: Integrator = LeapfrogIntegrator(),
        step_size: float = 0.1,
        trajectory_length: int = 6,
    ):
        """Initialize.

        Args:
            log_prob_fn: Function that takes (n_chains, n_dims) and returns (n_chains,)
            integrator: Integrator instance (defaults to LeapfrogIntegrator)
            step_size: Initial step size (can be adapted later)
            trajectory_length: Total trajectory length in leapfrog steps
        """
        # TODO: what gets set here vs in sample()?
        self.log_prob_fn = log_prob_fn
        self.integrator = integrator
        self.step_size = step_size
        self.trajectory_length = trajectory_length
        self.diagnostics = {}

    def hamiltonian(
        self, theta: torch.Tensor, momentum: torch.Tensor, inv_mass: torch.Tensor
    ) -> torch.Tensor:
        """Compute Hamiltonian.

        Args:
            theta: Parameter tensor (n_chains, n_dims)
            momentum: Momentum tensor (n_chains, n_dims)

        Returns:
            Hamiltonian values (n_chains,)
        """
        kinetic_energy = 0.5 * torch.einsum("bd,dd,bd->b", momentum, inv_mass, momentum)
        potential_energy = -self.log_prob_fn(theta)
        return kinetic_energy + potential_energy

    def propose(
        self, theta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Generate proposal using Hamiltonian dynamics.

        Args:
            theta: Current parameter estimates (n_chains, n_dims)

        Returns:
            Tuple of (theta_proposal, accept_prob)
        """
        momentum = self.integrator.sample_momentum(
            n_chains=self.n_chains,
            n_dims=self.n_dims,
            inv_mass=self.inv_mass,
            device=self.device,
        )

        theta_proposal, momentum_proposal = self.integrator.step(
            theta_start=theta,
            eps=self.step_size,
            L=self.n_leapfrog,
            log_prob_fn=self.log_prob_fn,
            inv_mass=self.inv_mass,
            eps_jitter=self.eps_jitter,
        )

        H_current = self.hamiltonian(theta, momentum, self.inv_mass)
        H_proposal = self.hamiltonian(theta_proposal, momentum_proposal, self.inv_mass)
        accept_prob = torch.exp(torch.clamp(H_current - H_proposal, max=0.0))

        return theta_proposal, accept_prob

    def accept_reject(
        self,
        theta: torch.Tensor,
        theta_proposal: torch.Tensor,
        accept_prob: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform vectorized accept/reject step.

        Args:
            theta: Current positions (n_chains, n_dims)
            theta_proposal: Proposed positions (n_chains, n_dims)
            accept_prob: Acceptance probabilities (n_chains,)

        Returns:
            Tuple of (theta_new, accepted) where accepted is boolean mask
        """
        accepted = torch.rand(self.n_chains, device=self.device) < accept_prob
        theta_new = torch.where(accepted.unsqueeze(-1), theta_proposal, theta)

        return theta_new, accepted

    def sample(
        self,
        n_samples: int,
        theta_0: torch.Tensor,
        *,
        n_warmup: int = 0,
        eps_jitter: float = 0.5,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Run HMC sampling.

        Args:
            n_samples: Number of samples to generate
            theta_0: Initial positions (n_chains, n_dims)
            n_warmup: Number of warmup iterations, defaults to 0
            eps_jitter: Jitter for step size, defaults to 0.5

        Returns:
            Tuple of (samples, diagnostics)
            samples: (n_samples, n_chains, n_dims)
            diagnostics: Dictionary with sampling statistics
        """
        if (
            theta_0.ndim != 2
            or self.log_prob_fn(theta_0).ndim != 1
            or self.log_prob_fn(theta_0).shape[0] != theta_0.shape[0]
        ):
            raise ValueError("log_prob_fn must return a 1D tensor of shape (n_chains,)")

        self.n_chains, self.n_dims = theta_0.shape
        self.device = theta_0.device
        self.inv_mass = torch.eye(self.n_dims, device=self.device)
        self.eps_jitter = eps_jitter

        self.samples = torch.zeros(
            n_samples, self.n_chains, self.n_dims, device=self.device
        )
        self.accept_rates = torch.zeros(n_samples, device=self.device)
        self.n_accepted = 0
        self.n_proposed = 0

        theta = theta_0.clone()

        pbar = tqdm(range(n_warmup + n_samples), unit="step")
        for i in pbar:
            theta_proposal, accept_prob = self.propose(theta)
            theta, accepted = self.accept_reject(theta, theta_proposal, accept_prob)
            self.n_accepted += accepted.sum().item()
            self.n_proposed += self.n_chains

            if i < n_warmup:
                pbar.set_description(
                    f"Stage: Warmup | Accept rate: {(self.acceptance_fraction):.3f}"
                )

            elif i == n_warmup:
                self.n_accepted = 0
                self.n_proposed = 0

            else:
                pbar.set_description(
                    f"Stage: Sampling | Accept rate: {(self.acceptance_fraction):.3f}"
                )
                self.samples[i - n_warmup] = theta
                self.accept_rates[i - n_warmup] = accepted.float().mean()

        return self.samples

    @property
    def acceptance_fraction(self) -> float:
        """Compute acceptance fraction."""
        if self.n_proposed == 0:
            return 0.0
        return self.n_accepted / self.n_proposed

    @property
    def n_leapfrog(self) -> int:
        """Number of leapfrog steps."""
        return int(self.trajectory_length // self.step_size)
