"""Adaption strategies for HMC sampling."""

import math
import torch
import numpy as np

# optimal tuning for HMC, see https://arxiv.org/abs/1001.4460
OPTIMAL_TARGET_ACCEPTANCE_RATE = 0.651


def harmonic_mean(a: torch.Tensor) -> float:
    """Compute the harmonic mean of a tensor."""
    return 1.0 / torch.mean(1.0 / a)


class TrajectoryLengthAdaptor:
    """Adapt trajectory length, source ChEES-HMC: http://proceedings.mlr.press/v130/hoffman21a/hoffman21a.pdf"""

    def __init__(self, jitter_amount: float = 1.0):
        self.log_trajectory_length_ma = 0
        self.eps_bar = 0
        self.jitter_amount = jitter_amount
        self.halton_index = 0
        self.decay_rate = 0.75

        # Adam (with no momentum)
        self.b2 = 0.95
        self.lr = 0.025
        self.v_t = 0
        self.t = 1

    def _halton_sequence(self, index: int, base: int = 2) -> float:
        """Generate Halton sequence for quasi-random jittering."""
        result = 0.0
        f = 1.0 / base
        i = index
        while i > 0:
            result += f * (i % base)
            i //= base
            f /= base
        return result

    def _get_jitter_factor(self) -> float:
        """Get jitter factor using Halton sequence."""
        jitter = self._halton_sequence(self.halton_index)
        self.halton_index += 1
        return jitter * self.jitter_amount + (1.0 - self.jitter_amount)

    def __call__(self, iteration, proposal, last_state, momentum, acceptance_probabilities, trajectory_length, diverging):
        """Update trajectory length adaptation.
        
        Args:
            iteration: Current iteration number
            proposal: Proposed positions (n_chains, n_dims)
            last_state: Previous positions (n_chains, n_dims) 
            momentum: Momentum vectors (n_chains, n_dims)
            acceptance_probabilities: Acceptance probabilities (n_chains,)
            trajectory_length: Trajectory length
            diverging: Boolean mask for divergent chains (n_chains,)
        """
        proposal_centered = proposal - proposal.mean(dim=0)
        previous_centered = last_state - last_state.mean(dim=0)

        trajectory_gradients = (
            self._get_jitter_factor() * trajectory_length *
            (torch.einsum("bd,bd->b", proposal_centered, proposal_centered) - 
             torch.einsum("bd,bd->b", previous_centered, previous_centered)) *
            torch.einsum("bd,bd->b", proposal_centered, momentum)
        )

        if diverging.any():
            non_divergent_mask = ~diverging
            trajectory_gradients = trajectory_gradients[non_divergent_mask]
            acceptance_probabilities = acceptance_probabilities[non_divergent_mask]
        
        # Avoid division by zero
        if len(acceptance_probabilities) == 0 or acceptance_probabilities.sum() == 0:
            return self.trajectory_length

        # Compute weighted average gradient
        trajectory_gradient = (acceptance_probabilities * trajectory_gradients).sum() / acceptance_probabilities.sum()        
        log_trajectory_length = math.log(trajectory_length)

        # Adam with no momentum (gradient ascent as maximizing ChEES criterion)
        v_t_plus_1 = self.b2 * self.v_t + (1 - self.b2) * trajectory_gradient**2
        v_hat = v_t_plus_1 / (1 - self.b2 ** self.t)
        new_log_trajectory_length = log_trajectory_length + (self.lr * trajectory_gradient) / (v_hat ** 0.5 + 1e-08)

        if not np.isfinite(new_log_trajectory_length):
            new_log_trajectory_length = log_trajectory_length
        else:
            self.v_t = v_t_plus_1
            self.t += 1

        weight = iteration ** (-self.decay_rate)
        self.log_trajectory_length_ma = (1.0 - weight) * self.log_trajectory_length_ma + weight * new_log_trajectory_length
        
        return self.trajectory_length
    
    @property
    def trajectory_length(self) -> float:
        """Get current base trajectory length estimate."""
        return math.exp(self.log_trajectory_length_ma)
    
    def get_jittered_trajectory_length(self, eps: float = 1e-6) -> float:
        """Get a jittered trajectory length for sampling."""
        base_length = max(self.trajectory_length, eps)
        jitter_factor = self._get_jitter_factor()
        return base_length * jitter_factor

class StepSizeAdaptor:
    """Adapt step size using dual averaging."""

    def __init__(
        self,
        initial_step_size: float,
        target_acceptance_rate: float = OPTIMAL_TARGET_ACCEPTANCE_RATE,
    ):
        self.target_acceptance_rate = target_acceptance_rate
        self.log_ss_ma = 0.0
        self.log_ss = initial_step_size
        self.h_t = 0.0
        self.mu = math.log(10 * initial_step_size)
        self.gamma = 0.05
        self.t_0 = 10
        self.decay_rate = 0.75

    def __call__(
        self,
        iteration: int,
        acceptance_probabilities: torch.Tensor,
    ):
        """Adapt step size. Algo 5 in https://arxiv.org/pdf/1111.4246.

        Args:
            iteration: Current iteration
            acceptance_rate: Current acceptance rate
        """
        _t = 1 / (iteration + self.t_0)
        self.h_t = (1 - _t) * self.h_t + _t * (
            self.target_acceptance_rate - harmonic_mean(acceptance_probabilities)
        )
        self.log_ss = self.mu - (iteration**0.5 / self.gamma) * self.h_t
        weight = iteration ** (-self.decay_rate)
        self.log_ss_ma = weight * self.log_ss + (1 - weight) * self.log_ss_ma

        return self.step_size

    @property
    def step_size(self) -> float:
        """Get current step size estimate."""
        return math.exp(self.log_ss)

    @property
    def final_step_size(self) -> float:
        """Get final step size estimate after adaptation."""
        return math.exp(self.log_ss_ma)
