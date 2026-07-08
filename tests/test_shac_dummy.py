from __future__ import annotations

from pathlib import Path

import pytest
import torch

from diffrl.algorithms.shac import SHAC
from diffrl.envs.dummy_differentiable import DummyDifferentiableVecEnv


def _cfg(logdir: Path) -> dict:
    return {
        "params": {
            "diff_env": {"name": "DummyDifferentiableVecEnv", "episode_length": 16},
            "general": {
                "seed": 0,
                "device": "cpu",
                "render": False,
                "train": True,
                "logdir": str(logdir),
            },
            "network": {
                "actor": "ActorStochasticMLP",
                "actor_mlp": {"units": [16], "activation": "elu"},
                "critic": "CriticMLP",
                "critic_mlp": {"units": [16], "activation": "elu"},
            },
            "config": {
                "name": "dummy_shac",
                "actor_learning_rate": 1e-3,
                "critic_learning_rate": 1e-3,
                "lr_schedule": "constant",
                "target_critic_alpha": 0.2,
                "obs_rms": False,
                "ret_rms": False,
                "critic_iterations": 2,
                "critic_method": "one-step",
                "num_batch": 2,
                "gamma": 0.95,
                "betas": [0.7, 0.95],
                "max_epochs": 2,
                "steps_num": 4,
                "grad_norm": 1.0,
                "truncate_grads": True,
                "num_actors": 8,
                "save_interval": 0,
                "player": {"deterministic": True, "games_num": 1, "num_actors": 1},
            },
        }
    }


def test_shac_import_and_dummy_differentiable_training(tmp_path):
    env = DummyDifferentiableVecEnv(num_envs=8, episode_length=16, device="cpu")
    alg = SHAC(_cfg(tmp_path), env=env)
    actor_loss = alg.compute_actor_loss()
    assert torch.isfinite(actor_loss)
    alg.actor_optimizer.zero_grad()
    actor_loss.backward()
    grad_norm = torch.linalg.vector_norm(
        torch.stack([p.grad.norm() for p in alg.actor.parameters() if p.grad is not None])
    )
    assert torch.isfinite(grad_norm)
    assert grad_norm > 0

    alg.train()
    checkpoint = tmp_path / "final_policy.pt"
    assert checkpoint.is_file()

    restored = SHAC(_cfg(tmp_path / "restored"), env=DummyDifferentiableVecEnv(num_envs=8, device="cpu"))
    restored.load(str(checkpoint))


def test_shac_save_interval_avoids_inf_reward_checkpoint_names(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["params"]["config"]["save_interval"] = 1
    cfg["params"]["config"]["max_epochs"] = 1
    env = DummyDifferentiableVecEnv(num_envs=8, episode_length=16, device="cpu")

    alg = SHAC(cfg, env=env)
    alg.train()

    checkpoint_names = {path.name for path in tmp_path.glob("*.pt")}
    assert "dummy_shacpolicy_iter1.pt" in checkpoint_names
    assert not any("inf" in name.lower() or "nan" in name.lower() for name in checkpoint_names)


def test_shac_logging_uses_finite_placeholders_when_no_episode_finished(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    cfg["params"]["config"]["max_epochs"] = 1
    cfg["params"]["config"]["steps_num"] = 1
    cfg["params"]["config"]["save_interval"] = 0
    env = DummyDifferentiableVecEnv(num_envs=8, episode_length=128, device="cpu")

    alg = SHAC(cfg, env=env)
    alg.train()

    output = capsys.readouterr().out.lower()
    iter_lines = [line for line in output.splitlines() if line.startswith("iter ")]
    assert iter_lines
    assert not any("inf" in line or "nan" in line for line in iter_lines)


def test_shac_can_disable_actor_loss_critic_bootstrap(tmp_path):
    class ZeroRewardActionStateEnv:
        num_envs = 4
        num_obs = 1
        num_actions = 1
        episode_length = 4
        device = torch.device("cpu")

        def __init__(self):
            self.reset()

        def clear_grad(self):
            pass

        def reset(self):
            self._step = 0
            self.state = torch.zeros(self.num_envs, self.num_obs)
            return self.state

        def initialize_trajectory(self):
            return self.reset()

        def step(self, actions):
            self._step += 1
            next_state = actions[:, :1]
            reward = actions[:, 0] * 0.0
            done = torch.zeros(self.num_envs, dtype=torch.bool)
            return next_state, reward, done, {"obs_before_reset": next_state}

    cfg = _cfg(tmp_path)
    cfg["params"]["diff_env"]["episode_length"] = 4
    cfg["params"]["network"]["actor"] = "ActorDeterministicMLP"
    cfg["params"]["network"]["actor_mlp"] = {"units": [4], "activation": "elu"}
    cfg["params"]["network"]["actor_output_bias_init"] = 0.25
    cfg["params"]["network"]["actor_output_weight_init_scale"] = 0.0
    cfg["params"]["config"]["num_actors"] = 4
    cfg["params"]["config"]["steps_num"] = 1
    cfg["params"]["config"]["actor_loss_critic_bootstrap"] = False

    alg = SHAC(cfg, env=ZeroRewardActionStateEnv())
    alg.target_critic = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        alg.target_critic.weight.fill_(1.0)

    loss = alg.compute_actor_loss()
    alg.actor_optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.linalg.vector_norm(
        torch.stack([p.grad.norm() for p in alg.actor.parameters() if p.grad is not None])
    )

    assert torch.isfinite(grad_norm)
    assert grad_norm == 0.0


def test_shac_accumulates_termination_reason_counts_from_extras(tmp_path):
    class TerminationReasonEnv:
        num_envs = 4
        num_obs = 1
        num_actions = 1
        episode_length = 2
        device = torch.device("cpu")

        def clear_grad(self):
            pass

        def reset(self):
            return torch.zeros(self.num_envs, self.num_obs)

        def initialize_trajectory(self):
            return self.reset()

        def step(self, actions):
            next_state = actions[:, :1]
            reward = actions[:, 0]
            done = torch.tensor([True, False, True, True])
            term_dones = torch.tensor(
                [
                    [False, True],
                    [False, False],
                    [True, False],
                    [False, False],
                ]
            )
            extras = {
                "obs_before_reset": next_state,
                "termination_terms": {"names": ["time_out", "crash"], "dones": term_dones},
            }
            return next_state, reward, done, extras

    cfg = _cfg(tmp_path)
    cfg["params"]["diff_env"]["episode_length"] = 2
    cfg["params"]["network"]["actor"] = "ActorDeterministicMLP"
    cfg["params"]["network"]["actor_mlp"] = {"units": [4], "activation": "elu"}
    cfg["params"]["network"]["actor_output_bias_init"] = 0.0
    cfg["params"]["network"]["actor_output_weight_init_scale"] = 0.0
    cfg["params"]["config"]["critic_iterations"] = 0
    cfg["params"]["config"]["num_actors"] = 4
    cfg["params"]["config"]["steps_num"] = 1
    cfg["params"]["config"]["save_interval"] = 0

    alg = SHAC(cfg, env=TerminationReasonEnv())

    loss = alg.compute_actor_loss()

    assert torch.isfinite(loss)
    assert alg.termination_reason_counts == {"crash": 1, "time_out": 1}
    assert alg.termination_done_count == 3
    assert alg.termination_unmatched_done_count == 1


def test_shac_accumulates_rollout_invalid_reason_counts_from_extras(tmp_path):
    class InvalidReasonEnv:
        num_envs = 4
        num_obs = 1
        num_actions = 1
        episode_length = 2
        device = torch.device("cpu")

        def clear_grad(self):
            pass

        def reset(self):
            return torch.zeros(self.num_envs, self.num_obs)

        def initialize_trajectory(self):
            return self.reset()

        def step(self, actions):
            next_state = actions[:, :1]
            reward = actions[:, 0]
            done = torch.tensor([True, False, True, False])
            extras = {
                "obs_before_reset": next_state,
                "rollout_invalid_envs": torch.tensor([True, False, True, False]),
                "rollout_new_invalid_envs": torch.tensor([False, False, True, False]),
                "rollout_invalid_sources": {
                    "done": torch.tensor([True, False, False, False]),
                    "state_token": torch.tensor([False, False, True, False]),
                },
            }
            return next_state, reward, done, extras

    cfg = _cfg(tmp_path)
    cfg["params"]["diff_env"]["episode_length"] = 2
    cfg["params"]["network"]["actor"] = "ActorDeterministicMLP"
    cfg["params"]["network"]["actor_mlp"] = {"units": [4], "activation": "elu"}
    cfg["params"]["network"]["actor_output_bias_init"] = 0.0
    cfg["params"]["network"]["actor_output_weight_init_scale"] = 0.0
    cfg["params"]["config"]["critic_iterations"] = 0
    cfg["params"]["config"]["num_actors"] = 4
    cfg["params"]["config"]["steps_num"] = 1
    cfg["params"]["config"]["save_interval"] = 0

    alg = SHAC(cfg, env=InvalidReasonEnv())

    loss = alg.compute_actor_loss()

    assert torch.isfinite(loss)
    assert alg.rollout_invalid_counts == {"invalid": 2, "new_invalid": 1, "done": 1, "state_token": 1}


def test_shac_records_actor_action_profile_after_tanh(tmp_path):
    class ActionProfileEnv:
        num_envs = 4
        num_obs = 1
        num_actions = 1
        episode_length = 2
        device = torch.device("cpu")

        def clear_grad(self):
            pass

        def reset(self):
            return torch.zeros(self.num_envs, self.num_obs)

        def initialize_trajectory(self):
            return self.reset()

        def step(self, actions):
            return actions[:, :1], actions[:, 0], torch.zeros(self.num_envs, dtype=torch.bool), {}

    cfg = _cfg(tmp_path)
    cfg["params"]["diff_env"]["episode_length"] = 2
    cfg["params"]["network"]["actor"] = "ActorDeterministicMLP"
    cfg["params"]["network"]["actor_mlp"] = {"units": [4], "activation": "elu"}
    cfg["params"]["network"]["actor_output_bias_init"] = 0.5
    cfg["params"]["network"]["actor_output_weight_init_scale"] = 0.0
    cfg["params"]["config"]["critic_iterations"] = 0
    cfg["params"]["config"]["num_actors"] = 4
    cfg["params"]["config"]["steps_num"] = 2
    cfg["params"]["config"]["save_interval"] = 0

    alg = SHAC(cfg, env=ActionProfileEnv())

    loss = alg.compute_actor_loss()

    expected_action = torch.tanh(torch.tensor(0.5)).item()
    assert torch.isfinite(loss)
    assert alg.actor_action_profile["count"] == 8
    assert alg.actor_action_profile["min"] == pytest.approx(expected_action)
    assert alg.actor_action_profile["max"] == pytest.approx(expected_action)
    assert alg.actor_action_profile["mean"] == pytest.approx(expected_action)


def test_shac_clips_large_finite_actor_gradients_instead_of_rejecting_them(tmp_path):
    class LargeFiniteGradientEnv:
        num_envs = 4
        num_obs = 1
        num_actions = 1
        episode_length = 1
        device = torch.device("cpu")

        def __init__(self):
            self.reset()

        def clear_grad(self):
            pass

        def reset(self):
            self._step = 0
            self.state = torch.zeros(self.num_envs, self.num_obs)
            return self.state

        def initialize_trajectory(self):
            return self.reset()

        def step(self, actions):
            self._step += 1
            next_state = actions[:, :1]
            reward = actions[:, 0] * 1.0e9
            done = torch.ones(self.num_envs, dtype=torch.bool)
            self.state = torch.zeros_like(next_state)
            return self.state, reward, done, {"obs_before_reset": next_state}

    cfg = _cfg(tmp_path)
    cfg["params"]["diff_env"]["episode_length"] = 1
    cfg["params"]["network"]["actor"] = "ActorDeterministicMLP"
    cfg["params"]["network"]["actor_mlp"] = {"units": [4], "activation": "elu"}
    cfg["params"]["network"]["actor_output_bias_init"] = 0.0
    cfg["params"]["network"]["actor_output_weight_init_scale"] = 0.0
    cfg["params"]["config"]["critic_iterations"] = 0
    cfg["params"]["config"]["max_epochs"] = 1
    cfg["params"]["config"]["num_actors"] = 4
    cfg["params"]["config"]["steps_num"] = 1
    cfg["params"]["config"]["save_interval"] = 0
    cfg["params"]["config"]["grad_norm"] = 1.0
    cfg["params"]["config"]["truncate_grads"] = True

    alg = SHAC(cfg, env=LargeFiniteGradientEnv())

    alg.train()

    assert torch.isfinite(alg.grad_norm_before_clip)
    assert alg.grad_norm_before_clip > 1.0e6
    assert torch.isfinite(alg.grad_norm_after_clip)
    assert alg.grad_norm_after_clip <= 1.0 + 1e-6


def test_shac_clips_extreme_finite_actor_gradients_before_norm_overflow(tmp_path):
    class ExtremeFiniteGradientEnv:
        num_envs = 4
        num_obs = 1
        num_actions = 1
        episode_length = 1
        device = torch.device("cpu")

        def clear_grad(self):
            pass

        def reset(self):
            return torch.zeros(self.num_envs, self.num_obs)

        def initialize_trajectory(self):
            return self.reset()

        def step(self, actions):
            next_state = actions[:, :1]
            reward = actions[:, 0] * 1.0e30
            done = torch.ones(self.num_envs, dtype=torch.bool)
            return next_state, reward, done, {"obs_before_reset": next_state}

    cfg = _cfg(tmp_path)
    cfg["params"]["diff_env"]["episode_length"] = 1
    cfg["params"]["network"]["actor"] = "ActorDeterministicMLP"
    cfg["params"]["network"]["actor_mlp"] = {"units": [4], "activation": "elu"}
    cfg["params"]["network"]["actor_output_bias_init"] = 0.0
    cfg["params"]["network"]["actor_output_weight_init_scale"] = 0.0
    cfg["params"]["config"]["critic_iterations"] = 0
    cfg["params"]["config"]["max_epochs"] = 1
    cfg["params"]["config"]["num_actors"] = 4
    cfg["params"]["config"]["steps_num"] = 1
    cfg["params"]["config"]["save_interval"] = 0
    cfg["params"]["config"]["grad_value_clip"] = 1.0e6

    alg = SHAC(cfg, env=ExtremeFiniteGradientEnv())

    alg.train()

    assert alg.actor_raw_grad_profile["nonfinite_count"] == 0
    assert alg.actor_raw_grad_profile["abs_max"] > cfg["params"]["config"]["grad_value_clip"]
    assert alg.actor_raw_grad_profile["norm"] > alg.grad_norm_before_clip
    assert torch.isfinite(alg.grad_norm_before_clip)
    assert torch.isfinite(alg.grad_norm_after_clip)


def test_shac_preprocesses_extreme_observations_without_cutting_gradients(tmp_path):
    env = DummyDifferentiableVecEnv(num_envs=4, episode_length=16, device="cpu")
    cfg = _cfg(tmp_path)
    cfg["params"]["config"]["obs_clip"] = 10.0
    alg = SHAC(cfg, env=env)
    obs = torch.tensor(
        [[float("nan"), float("inf")], [1.0e6, -1.0e6], [1.0, -2.0], [0.0, 3.0]],
        dtype=torch.float32,
        requires_grad=True,
    )

    processed = alg._preprocess_obs(obs)
    loss = processed[-1].sum()
    grad = torch.autograd.grad(loss, obs, allow_unused=True)[0]

    assert torch.isfinite(processed).all()
    assert processed.abs().max() <= 10.0
    assert torch.allclose(processed[0], torch.zeros_like(processed[0]))
    assert torch.allclose(grad[-1], torch.ones_like(grad[-1]))


def test_shac_sanitizes_nonfinite_actor_gradients_before_clipping(tmp_path):
    class NonFiniteGradientEnv:
        num_envs = 4
        num_obs = 1
        num_actions = 1
        episode_length = 1
        device = torch.device("cpu")

        def clear_grad(self):
            pass

        def reset(self):
            return torch.zeros(self.num_envs, self.num_obs)

        def initialize_trajectory(self):
            return self.reset()

        def step(self, actions):
            class _NaNBackward(torch.autograd.Function):
                @staticmethod
                def forward(ctx, value):
                    return value

                @staticmethod
                def backward(ctx, grad_output):
                    return torch.full_like(grad_output, float("nan"))

            reward = _NaNBackward.apply(actions[:, 0])
            done = torch.ones(self.num_envs, dtype=torch.bool)
            obs = torch.zeros(self.num_envs, self.num_obs)
            return obs, reward, done, {"obs_before_reset": obs}

    cfg = _cfg(tmp_path)
    cfg["params"]["diff_env"]["episode_length"] = 1
    cfg["params"]["network"]["actor"] = "ActorDeterministicMLP"
    cfg["params"]["network"]["actor_mlp"] = {"units": [4], "activation": "elu"}
    cfg["params"]["network"]["actor_output_bias_init"] = 0.0
    cfg["params"]["network"]["actor_output_weight_init_scale"] = 0.0
    cfg["params"]["config"]["critic_iterations"] = 0
    cfg["params"]["config"]["max_epochs"] = 1
    cfg["params"]["config"]["num_actors"] = 4
    cfg["params"]["config"]["steps_num"] = 1
    cfg["params"]["config"]["save_interval"] = 0

    alg = SHAC(cfg, env=NonFiniteGradientEnv())

    alg.train()

    assert alg.actor_raw_grad_profile["nonfinite_count"] > 0
    assert alg.actor_raw_grad_profile["finite_count"] == 0
    assert torch.isfinite(alg.grad_norm_before_clip)
    assert alg.grad_norm_before_clip == 0.0
