"""HMC samplers."""

from typing import Callable, Tuple

import torch

from integrators import leapfrog
from adaption import StepSizeAdaptor, TrajectoryLengthAdaptor
from tqdm import tqdm


class HMC:
    """HMC with step size and trajectory length adaptation."""

    def __init__(
        self,
        log_prob_fn: Callable[[torch.Tensor], torch.Tensor],
        initial_state: torch.Tensor,
    ):
        """Initialize.

        Args:
            log_prob_fn: Function that takes (n_chains, n_dims) and returns (n_chains,)
            initial_state: Initial positions (n_chains, n_dims)
        """
        if (
            initial_state.ndim != 2
            or log_prob_fn(initial_state).ndim != 1
            or log_prob_fn(initial_state).shape[0] != initial_state.shape[0]
        ):
            raise ValueError("log_prob_fn must return a 1D tensor of shape (n_chains,)")

        self.log_prob_fn = log_prob_fn
        self.n_chains, self.n_dims = initial_state.shape
        self.device = initial_state.device
        self.inv_mass = torch.eye(self.n_dims, device=self.device)
        self.chain = [initial_state.clone()]
        self.warmup_mask = [False]
        self.mean_acceptance = 0.0

        self.set_initial_step_size(initial_state)
        self.step_size_adaptor = StepSizeAdaptor(self.step_size)

        self.trajectory_length = 6.14
        self.trajectory_length_adaptor = TrajectoryLengthAdaptor()

    def hamiltonian(
        self, position: torch.Tensor, momentum: torch.Tensor
    ) -> torch.Tensor:
        """Compute Hamiltonian.

        Args:
            position: Parameter tensor (n_chains, n_dims)
            momentum: Momentum tensor (n_chains, n_dims)

        Returns:
            Hamiltonian values (n_chains,)
        """
        ke = 0.5 * torch.einsum("bd,dd,bd->b", momentum, self.inv_mass, momentum)
        pe = -self.log_prob_fn(position)
        return ke + pe

    def propose(
        self, position: torch.Tensor, momentum: torch.Tensor, n_steps: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate proposal with acceptance probabilities.

        Args:
            position: Current parameter estimates (n_chains, n_dims)
            momentum: Current momentum (n_chains, n_dims)
            n_steps: Number of leapfrog steps to take

        Returns:
            Tuple of (position_proposal, acceptance_probabilities)
        """

        position_proposal, momentum_proposal = leapfrog(
            start_position=position,
            start_momentum=momentum,
            step_size=self.step_size,
            n_steps=n_steps,
            log_prob_fn=self.log_prob_fn,
            inv_mass=self.inv_mass,
        )

        H_current = self.hamiltonian(position, momentum)
        H_proposal = self.hamiltonian(position_proposal, momentum_proposal)
        delta_H = H_proposal - H_current
        delta_H = torch.where(torch.isnan(delta_H), float("inf"), delta_H)
        diverging = delta_H > 1000.0
        acceptance_probabilities = torch.exp(torch.clamp(-delta_H, max=0.0))

        return position_proposal, acceptance_probabilities, diverging

    def run(self, n_samples: int, n_warmup: int = 0):
        """Run HMC sampling.

        Args:
            n_samples: Number of samples to generate
            n_warmup: Number of warmup iterations, defaults to 0
        """
        pbar = tqdm(range(1, n_warmup + n_samples + 1), unit="step")
        for i in pbar:

            # Get jittered trajectory length for this iteration
            jittered_trajectory_length = self.trajectory_length_adaptor.get_jittered_trajectory_length()
            
            momentum = self.sample_momentum()
            position_proposal, acceptance_probabilities, diverging = self.propose(
                self.chain[-1], momentum, max(1, int(jittered_trajectory_length / self.step_size))
            )
            accepted_mask = (
                torch.rand(self.n_chains, device=self.device) < acceptance_probabilities
            )
            self.chain.append(
                torch.where(accepted_mask.unsqueeze(-1), position_proposal, self.chain[-1])
            )
            self.mean_acceptance += (acceptance_probabilities.mean().item() - self.mean_acceptance) / i

            if i < n_warmup:
                pbar.set_description(
                    f"Stage: Warmup | Accept rate: {(self.mean_acceptance):.3f} | Step size: {self.step_size:.3f} | TL: {self.trajectory_length:.3f}"
                )
                self.step_size = self.step_size_adaptor(i, acceptance_probabilities)
                # Update trajectory length using the adaptor (pass base length, not jittered)
                self.trajectory_length = self.trajectory_length_adaptor(
                    iteration=i,
                    proposal=position_proposal,
                    last_state=self.chain[-2],
                    momentum=momentum,
                    acceptance_probabilities=acceptance_probabilities,
                    trajectory_length=self.trajectory_length,
                    diverging=diverging,
                )
                self.warmup_mask.append(False)

            elif i == n_warmup:
                self.n_accepted = 0
                self.n_proposed = 0
                self.step_size = self.step_size_adaptor.final_step_size
                self.warmup_mask.append(False)

            else:
                pbar.set_description(
                    f"Stage: Sampling | Accept rate: {self.mean_acceptance:.3f} | Step size: {self.step_size:.3f} | TL: {self.trajectory_length:.3f}"
                )
                self.warmup_mask.append(True)

        return self.get_chain()

    def get_chain(self, thin: int = 1, flat: bool = False, include_warmup: bool = False) -> torch.Tensor:
        """Get the stored chain.

        Args:
            thin: Take only every `thin` steps from the chain.
            flat: If True, return a flattened version of the chain
            include_warmup: If True, include warmup samples in the chain

        Returns:
            Tensor of shape (n_samples, n_chains, n_dims)
        """
        chain = torch.stack(self.chain, dim=0)
        if not include_warmup:
            chain = chain[self.warmup_mask]
        chain = chain[::thin]
        if flat:
            chain = chain.view(-1, self.n_dims)
        return chain

    def sample_momentum(self) -> torch.Tensor:
        """Sample momentum."""
        return torch.randn(
            self.n_chains, self.n_dims, device=self.device
        ) @ torch.linalg.cholesky(self.inv_mass)

    def set_initial_step_size(self, position: torch.Tensor):
        """Source: https://arxiv.org/pdf/1111.4246 (Algorithm 4)."""

        self.step_size = 0.1
        momentum = self.sample_momentum()

        acceptance_probabilities = self.propose(position, momentum, 1)[1].mean().item()
        alpha = 2 * int(acceptance_probabilities > 0.5) - 1

        while acceptance_probabilities**alpha > 2**-alpha:
            self.step_size = 2**alpha * self.step_size
            acceptance_probabilities = self.propose(position, momentum, 1)[1].mean().item()
