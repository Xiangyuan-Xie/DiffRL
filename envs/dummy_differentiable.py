"""Small differentiable vectorized environment for SHAC smoke tests."""

from __future__ import annotations

import torch


class DummyDifferentiableVecEnv:
    """A tiny batched control problem with differentiable dynamics."""

    def __init__(
        self,
        num_envs: int = 16,
        num_obs: int = 3,
        num_actions: int = 2,
        episode_length: int = 32,
        device: str | torch.device = "cpu",
    ):
        self.num_envs = num_envs
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.episode_length = episode_length
        self.device = torch.device(device)
        self._action_map = torch.zeros(num_actions, num_obs, device=self.device)
        for i in range(num_actions):
            self._action_map[i, i % num_obs] = 1.0
        self.reset()

    def clear_grad(self) -> None:
        pass

    def reset(self) -> torch.Tensor:
        self._step = 0
        self.state = torch.zeros(self.num_envs, self.num_obs, device=self.device)
        return self.state

    def initialize_trajectory(self) -> torch.Tensor:
        return self.reset()

    def step(self, actions: torch.Tensor):
        self._step += 1
        projected_action = actions @ self._action_map
        next_state = 0.85 * self.state + 0.25 * projected_action
        reward = -(next_state.square().sum(dim=-1) + 0.01 * actions.square().sum(dim=-1))
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self._step >= self.episode_length:
            done[:] = True
        obs_before_reset = next_state
        self.state = torch.where(done[:, None], torch.zeros_like(next_state), next_state)
        return self.state, reward, done, {"obs_before_reset": obs_before_reset}

