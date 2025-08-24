"""
Implements step size and trajectory length adaptation as proposed in Hoffman et al (2021).
"""

import math
import torch
import numpy as np

OPTIMAL_TARGET_ACCEPTANCE_RATE = 0.651  # Source: https://arxiv.org/abs/1001.4460

def harmonic_mean(a: torch.Tensor) -> float:
    return 1.0 / torch.mean(1.0 / a)


def halton_sequence(index: int, base: int = 2) -> float:
    """Halton sequence for quasi-random jittering."""
    result = 0.0
    f = 1.0 / base
    i = index
    while i > 0:
        result += f * (i % base)
        i //= base
        f /= base
    return result


class Adam:
    """Adam with no momentum."""

    def __init__(self, learning_rate: float = 0.025, beta_2: float = 0.95, epsilon: float = 1e-08):
        """Hyperparameter values as per original paper."""
        self.learning_rate = learning_rate
        self.variance_estimate = 0.0
        self.beta_2 = beta_2
        self.iteration = 1
        self.epsilon = epsilon

    def __call__(self, gradient: float, current_value: float) -> float:
        new_variance_estimate = self.beta_2 * self.variance_estimate + (1 - self.beta_2) * gradient**2
        bias_corrected_variance = new_variance_estimate / (1 - self.beta_2 ** self.iteration)
        step_size = self.learning_rate / (bias_corrected_variance ** 0.5 + self.epsilon)
        # Gradient ascent for maximizing ChEES objective
        new_value = current_value + step_size * gradient

        # Only update the internal state if the new value is valid
        if np.isfinite(new_value):
            self.variance_estimate = new_variance_estimate
            self.iteration += 1
            return new_value
        else:
            return current_value


class DualAveraging:

    def __init__(self, initial_step_size: float, target_acceptance_rate: float):
        self.target_acceptance_rate = target_acceptance_rate
        self.dual_averaging_state = 0.0
        self.offset = math.log(10 * initial_step_size)
        self.adaptation_rate = 0.05
        self.initial_adaptation_delay = 10
        self.iteration = 1

    def __call__(self, acceptance_probabilities: torch.Tensor) -> float:
        weight = 1 / (self.iteration + self.initial_adaptation_delay)
        self.dual_averaging_state = (1 - weight) * self.dual_averaging_state + weight * (
            self.target_acceptance_rate - harmonic_mean(acceptance_probabilities)
        )
        new_value = self.offset - (self.iteration ** 0.5 / self.adaptation_rate) * self.dual_averaging_state
        self.iteration += 1
        return new_value


class ChEESAdaption:
    """Source: http://proceedings.mlr.press/v130/hoffman21a/hoffman21a.pdf"""

    def __init__(
        self,
        initial_step_size: float,
        target_acceptance_rate: float = OPTIMAL_TARGET_ACCEPTANCE_RATE,
    ):
        self.dual_averaging = DualAveraging(initial_step_size, target_acceptance_rate)
        self.adam = Adam()
        self.jitter_factor = halton_sequence
        self.weight = 0.9
        
        self.trajectory_length_ma = 0.0
        self.step_size_ma = 0.0

    def __call__(
        self,
        iteration: int,
        proposal: torch.Tensor,
        last_state: torch.Tensor,
        momentum: torch.Tensor,
        acceptance_probabilities: torch.Tensor,
        trajectory_length: float,
        diverging: torch.Tensor,
    ) -> float:
        new_log_step_size = self.dual_averaging(acceptance_probabilities)
        new_step_size = math.exp(new_log_step_size)
        self.step_size_ma = self.weight * self.step_size_ma + (1 - self.weight) * new_step_size

        if diverging.all():
            return new_step_size, trajectory_length

        proposal_centered = proposal - proposal.mean(dim=0)
        previous_centered = last_state - last_state.mean(dim=0)

        trajectory_gradients = (
            self.jitter_factor(iteration) * trajectory_length *
            (torch.einsum("bd,bd->b", proposal_centered, proposal_centered) - 
             torch.einsum("bd,bd->b", previous_centered, previous_centered)) *
            torch.einsum("bd,bd->b", proposal_centered, momentum)
        )

        if diverging.any():
            trajectory_gradients = trajectory_gradients[~diverging]
            acceptance_probabilities = acceptance_probabilities[~diverging]

        trajectory_gradient = (acceptance_probabilities * trajectory_gradients).sum() / acceptance_probabilities.sum()

        new_log_trajectory_length = self.adam(trajectory_gradient, math.log(trajectory_length))
        new_trajectory_length = math.exp(new_log_trajectory_length)        
        self.trajectory_length_ma = self.weight * self.trajectory_length_ma + (1 - self.weight) * new_trajectory_length

        return new_step_size, new_trajectory_length

    def finalize(self):
        """Set final values after warmup."""
        return self.step_size_ma, self.trajectory_length_ma