"""HMC samplers."""

from typing import Callable, Tuple

import torch

from torchmc.integrator import leapfrog
from torchmc.adaption import ChEESAdaption
from tqdm.auto import trange


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
        if initial_state.ndim != 2:
            raise ValueError("Initial state must be shape (n_chains, n_dims)")
        if (_log_p := log_prob_fn(initial_state)).ndim != 1 or _log_p.shape[0] != initial_state.shape[0]:
            raise ValueError("log_prob_fn must return a 1D tensor of shape (n_chains,)")

        self.log_prob_fn = log_prob_fn
        self.n_chains, self.n_dims = initial_state.shape
        self.device = initial_state.device

        self.chain = [initial_state.clone()]
        self.n_warmup = None
        self.mean_acceptance = 0.0

        self.set_initial_step_size(initial_state)
        self.trajectory_length = self.step_size
        self.adaptor = ChEESAdaption(initial_step_size=self.step_size)

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
        return 0.5 * torch.sum(momentum**2, dim=1) - self.log_prob_fn(position)

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
        if self.n_warmup is not None and n_warmup != 0:
            raise ValueError("It is not valid to adapt state after sampling has started.")

        self.n_warmup = self.n_warmup or n_warmup

        pbar = trange(1, n_warmup + n_samples + 1, unit="step")
        for i in pbar:
            last_state = self.chain[-1]
            momentum = self.sample_momentum()
            jittered_trajectory_length = self.trajectory_length * self.adaptor.jitter_factor(i)
            n_steps = max(1, int(jittered_trajectory_length / self.step_size))

            position_proposal, acceptance_probs, diverging = self.propose(
                position=last_state, momentum=momentum, n_steps=n_steps
            )

            accepted = torch.rand(self.n_chains, device=self.device) < acceptance_probs
            new_state = torch.where(accepted.unsqueeze(-1), position_proposal, last_state)
            self.chain.append(new_state)
            self.mean_acceptance += (acceptance_probs.mean().item() - self.mean_acceptance) / i

            if i <= n_warmup:
                self.step_size, self.trajectory_length = self.adaptor(
                    iteration=i,
                    proposal=position_proposal,
                    last_state=last_state,
                    momentum=momentum,
                    acceptance_probabilities=acceptance_probs,
                    trajectory_length=self.trajectory_length,
                    diverging=diverging,
                )
            elif i == n_warmup + 1:
                self.step_size, self.trajectory_length = self.adaptor.finalize()

            stage = "Warmup" if i <= n_warmup else "Sampling"
            pbar.set_description(
                f"{stage} | Accept rate: {self.mean_acceptance:.3f} | "
                f"Step size: {self.step_size:.3e} | Trajectory length: {self.trajectory_length:.3e}"
            )

        return self.get_chain()

    def get_chain(self, thin: int = 1, flat: bool = False, include_warmup: bool = False) -> torch.Tensor:
        """Get the stored chain.

        Args:
            thin: Take only every `thin` steps from the chain.
            flat: If True, return a flattened version of the chain
            include_warmup: If True, include warmup samples in the chain

        Returns:
            Tensor of shape (n_chains, n_samples, n_dims)
        """
        chain = torch.stack(self.chain, dim=1)
        if not include_warmup:
            chain = chain[:, self.n_warmup + 1:, :]  # +1 for initial state
        chain = chain[:, ::thin, :]
        if flat:
            chain = chain.view(-1, self.n_dims)
        return chain

    def sample_momentum(self) -> torch.Tensor:
        """Sample momentum."""
        return torch.randn(self.n_chains, self.n_dims, device=self.device)

    def set_initial_step_size(self, position: torch.Tensor):
        """Source: https://arxiv.org/pdf/1111.4246 (Algorithm 4)."""

        self.step_size = 1
        momentum = self.sample_momentum()

        acceptance_probabilities = self.propose(position, momentum, 1)[1].mean().item()

        while acceptance_probabilities < 0.5:
            self.step_size *= 0.5
            acceptance_probabilities = self.propose(position, momentum, 1)[1].mean().item()