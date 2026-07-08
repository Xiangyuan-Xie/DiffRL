"""Interfaces for differentiable vectorized environments."""

from __future__ import annotations

from typing import Protocol

import torch


class DifferentiableVecEnv(Protocol):
    """Minimal environment contract required by SHAC.

    Rewards are maximized by the environment and converted to losses inside
    SHAC, preserving DiffRL's historical negative-reward loss convention.
    """

    num_envs: int
    num_obs: int
    num_actions: int
    episode_length: int

    def initialize_trajectory(self) -> torch.Tensor:
        """Reset or prepare rollout state while preserving action gradients."""

    def reset(self) -> torch.Tensor:
        """Reset the vectorized environment and return observations."""

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step all environments.

        Returns observations, rewards, done flags, and an info dictionary.
        ``info["obs_before_reset"]`` is optional; SHAC falls back to next
        observations when it is absent.
        """

    def clear_grad(self) -> None:
        """Clear simulator-side gradient buffers if the backend owns any."""

