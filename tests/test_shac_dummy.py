from __future__ import annotations

from pathlib import Path

import pytest
import torch

from diffrl.algorithms.shac import SHAC
from diffrl.envs.dummy_differentiable import DummyDifferentiableVecEnv


class RecordingWriter:
    def __init__(self):
        self.scalars = []
        self.flushed = False
        self.closed = False

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def flush(self):
        self.flushed = True

    def close(self):
        self.closed = True


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


def test_shac_tensorboard_logging_matches_original_diffrl_tags_without_debug_tags(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["params"]["config"]["max_epochs"] = 1
    cfg["params"]["config"]["steps_num"] = 1
    cfg["params"]["config"]["save_interval"] = 0
    env = DummyDifferentiableVecEnv(num_envs=8, episode_length=1, device="cpu")
    alg = SHAC(cfg, env=env)
    writer = RecordingWriter()
    alg.writer = writer

    alg.train()

    tags = {tag for tag, _value, _step in writer.scalars}
    expected_original_tags = {
        "lr/iter",
        "actor_loss/step",
        "actor_loss/iter",
        "value_loss/step",
        "value_loss/iter",
        "policy_loss/step",
        "policy_loss/time",
        "policy_loss/iter",
        "rewards/step",
        "rewards/time",
        "rewards/iter",
        "policy_discounted_loss/step",
        "policy_discounted_loss/iter",
        "best_policy_loss/step",
        "best_policy_loss/iter",
        "episode_lengths/iter",
        "episode_lengths/step",
        "episode_lengths/time",
    }
    assert expected_original_tags <= tags
    assert not any(tag.startswith("actor_grad_norm/") for tag in tags)
    assert not any(tag.startswith("memory/") for tag in tags)


def test_shac_tensorboard_logs_episode_extras_like_rsl_rl(tmp_path):
    class EpisodeExtrasEnv(DummyDifferentiableVecEnv):
        def step(self, actions):
            obs, reward, done, extras = super().step(actions)
            extras = dict(extras)
            extras["log"] = {
                "Episode_Reward/pos": torch.tensor([1.0, 3.0]),
                "Episode_Reward/att": 2.0,
                "Episode_Termination/time_out": torch.tensor(1.0),
                "curriculum_level": torch.tensor([4.0]),
                "debug_text": "ignored",
            }
            return obs, reward, done, extras

    cfg = _cfg(tmp_path)
    cfg["params"]["config"]["max_epochs"] = 1
    cfg["params"]["config"]["steps_num"] = 1
    cfg["params"]["config"]["save_interval"] = 0
    env = EpisodeExtrasEnv(num_envs=8, episode_length=1, device="cpu")
    alg = SHAC(cfg, env=env)
    writer = RecordingWriter()
    alg.writer = writer

    alg.train()

    scalar_by_tag = {tag: (value, step) for tag, value, step in writer.scalars}
    assert "Episode_Reward/pos" in scalar_by_tag
    assert "Episode_Reward/att" in scalar_by_tag
    assert "Episode_Termination/time_out" in scalar_by_tag
    assert "Episode/curriculum_level" in scalar_by_tag
    assert "debug_text" not in scalar_by_tag
    assert torch.as_tensor(scalar_by_tag["Episode_Reward/pos"][0]).item() == pytest.approx(2.0)
    assert torch.as_tensor(scalar_by_tag["Episode_Reward/att"][0]).item() == pytest.approx(2.0)
    assert torch.as_tensor(scalar_by_tag["Episode_Termination/time_out"][0]).item() == pytest.approx(1.0)
    assert torch.as_tensor(scalar_by_tag["Episode/curriculum_level"][0]).item() == pytest.approx(4.0)
    assert scalar_by_tag["Episode_Reward/pos"][1] == alg.iter_count
    assert scalar_by_tag["Episode_Reward/att"][1] == alg.iter_count
    assert scalar_by_tag["Episode_Termination/time_out"][1] == alg.iter_count
    assert scalar_by_tag["Episode/curriculum_level"][1] == alg.iter_count


def test_shac_tensorboard_prefers_episode_extras_over_log_like_rsl_rl(tmp_path):
    class EpisodePreferredExtrasEnv(DummyDifferentiableVecEnv):
        def step(self, actions):
            obs, reward, done, extras = super().step(actions)
            extras = dict(extras)
            extras["episode"] = {"manager_metric": torch.tensor(5.0)}
            extras["log"] = {"manager_metric": torch.tensor(1.0)}
            return obs, reward, done, extras

    cfg = _cfg(tmp_path)
    cfg["params"]["config"]["max_epochs"] = 1
    cfg["params"]["config"]["steps_num"] = 1
    cfg["params"]["config"]["save_interval"] = 0
    env = EpisodePreferredExtrasEnv(num_envs=8, episode_length=1, device="cpu")
    alg = SHAC(cfg, env=env)
    writer = RecordingWriter()
    alg.writer = writer

    alg.train()

    scalar_by_tag = {tag: (value, step) for tag, value, step in writer.scalars}
    assert "Episode/manager_metric" in scalar_by_tag
    assert torch.as_tensor(scalar_by_tag["Episode/manager_metric"][0]).item() == pytest.approx(5.0)


def test_shac_tensorboard_ignores_stale_episode_log_when_no_env_done(tmp_path):
    class StaleResetLogEnv(DummyDifferentiableVecEnv):
        def step(self, actions):
            obs, reward, done, extras = super().step(actions)
            extras = dict(extras)
            done = torch.zeros_like(done, dtype=torch.bool)
            extras["log"] = {
                "Episode_Reward/pos": torch.tensor(0.0),
                "Episode_Termination/time_out": torch.tensor(0.0),
            }
            return obs, reward, done, extras

    cfg = _cfg(tmp_path)
    cfg["params"]["config"]["steps_num"] = 1
    env = StaleResetLogEnv(num_envs=8, episode_length=128, device="cpu")
    alg = SHAC(cfg, env=env)
    writer = RecordingWriter()
    alg.writer = writer

    loss = alg.compute_actor_loss()
    assert torch.isfinite(loss)
    alg._log_episode_extra_scalars()

    tags = {tag for tag, _value, _step in writer.scalars}
    assert "Episode_Reward/pos" not in tags
    assert "Episode_Termination/time_out" not in tags


def test_shac_target_critic_keeps_input_gradient_without_parameter_gradients(tmp_path):
    env = DummyDifferentiableVecEnv(num_envs=4, episode_length=16, device="cpu")
    alg = SHAC(_cfg(tmp_path), env=env)

    assert all(not parameter.requires_grad for parameter in alg.target_critic.parameters())

    obs = torch.randn(4, alg.num_obs, requires_grad=True)
    value = alg.target_critic(obs).sum()
    obs_grad = torch.autograd.grad(value, obs)[0]

    assert torch.isfinite(obs_grad).all()
    assert torch.linalg.vector_norm(obs_grad) > 0
    assert all(parameter.grad is None for parameter in alg.target_critic.parameters())


def test_shac_clears_rollout_graph_after_actor_update(tmp_path):
    class CleanupEnv(DummyDifferentiableVecEnv):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.cleanup_calls = 0

        def clear_rollout_graph_after_update(self):
            self.cleanup_calls += 1
            self.clear_grad()

    cfg = _cfg(tmp_path)
    cfg["params"]["config"]["max_epochs"] = 2
    env = CleanupEnv(num_envs=8, episode_length=16, device="cpu")

    alg = SHAC(cfg, env=env)
    alg.train()

    assert env.cleanup_calls == cfg["params"]["config"]["max_epochs"]


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


def test_shac_actor_loss_critic_bootstrap_warmup_defers_target_critic_gradients(tmp_path):
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
    cfg["params"]["config"]["actor_loss_critic_bootstrap"] = True
    cfg["params"]["config"]["actor_loss_critic_bootstrap_warmup_epochs"] = 2

    alg = SHAC(cfg, env=ZeroRewardActionStateEnv())
    alg.target_critic = torch.nn.Linear(1, 1, bias=False)
    alg.target_critic.requires_grad_(False)
    with torch.no_grad():
        alg.target_critic.weight.fill_(1.0)

    loss = alg.compute_actor_loss()
    alg.actor_optimizer.zero_grad()
    loss.backward()
    warmup_grad_norm = torch.linalg.vector_norm(
        torch.stack([p.grad.norm() for p in alg.actor.parameters() if p.grad is not None])
    )

    alg.iter_count = 2
    loss = alg.compute_actor_loss()
    alg.actor_optimizer.zero_grad()
    loss.backward()
    active_grad_norm = torch.linalg.vector_norm(
        torch.stack([p.grad.norm() for p in alg.actor.parameters() if p.grad is not None])
    )

    assert torch.isfinite(warmup_grad_norm)
    assert warmup_grad_norm == 0.0
    assert torch.isfinite(active_grad_norm)
    assert active_grad_norm > 0.0


def test_shac_contract_uses_tanh_actions_and_obs_before_reset_for_timeout_bootstrap(tmp_path):
    class TimeoutBootstrapEnv:
        num_envs = 2
        num_obs = 1
        num_actions = 1
        episode_length = 1
        device = torch.device("cpu")

        def __init__(self):
            self.received_actions = None

        def clear_grad(self):
            pass

        def reset(self):
            return torch.zeros(self.num_envs, self.num_obs)

        def initialize_trajectory(self):
            return self.reset()

        def step(self, actions):
            self.received_actions = actions
            next_state = actions[:, :1] + 1.0
            done = torch.tensor([True, False])
            reward = torch.zeros(self.num_envs)
            obs_before_reset = next_state + torch.tensor([[2.0], [9.0]])
            return next_state, reward, done, {"obs_before_reset": obs_before_reset}

    cfg = _cfg(tmp_path)
    cfg["params"]["diff_env"]["episode_length"] = 1
    cfg["params"]["network"]["actor"] = "ActorDeterministicMLP"
    cfg["params"]["network"]["actor_mlp"] = {"units": [4], "activation": "elu"}
    cfg["params"]["network"]["actor_output_bias_init"] = 0.4
    cfg["params"]["network"]["actor_output_weight_init_scale"] = 0.0
    cfg["params"]["config"]["num_actors"] = 2
    cfg["params"]["config"]["steps_num"] = 1

    env = TimeoutBootstrapEnv()
    alg = SHAC(cfg, env=env)
    alg.target_critic = torch.nn.Linear(1, 1, bias=False)
    alg.target_critic.requires_grad_(False)
    with torch.no_grad():
        alg.target_critic.weight.fill_(1.0)

    loss = alg.compute_actor_loss()

    actor_action = torch.tanh(torch.tensor(0.4))
    assert env.received_actions is not None
    assert torch.allclose(env.received_actions, torch.full((2, 1), actor_action))
    expected_timeout_value = actor_action + 3.0
    expected_non_done_value = actor_action + 1.0
    expected_loss = -alg.gamma * (expected_timeout_value + expected_non_done_value) / env.num_envs
    assert torch.allclose(loss, expected_loss)


def test_shac_contract_zeros_critic_bootstrap_for_early_termination(tmp_path):
    class EarlyTerminationEnv:
        num_envs = 2
        num_obs = 1
        num_actions = 1
        episode_length = 4
        device = torch.device("cpu")

        def clear_grad(self):
            pass

        def reset(self):
            return torch.zeros(self.num_envs, self.num_obs)

        def initialize_trajectory(self):
            return self.reset()

        def step(self, actions):
            next_state = actions[:, :1] + 1.0
            done = torch.tensor([True, False])
            reward = torch.zeros(self.num_envs)
            obs_before_reset = next_state + 100.0
            return next_state, reward, done, {"obs_before_reset": obs_before_reset}

    cfg = _cfg(tmp_path)
    cfg["params"]["diff_env"]["episode_length"] = 4
    cfg["params"]["network"]["actor"] = "ActorDeterministicMLP"
    cfg["params"]["network"]["actor_mlp"] = {"units": [4], "activation": "elu"}
    cfg["params"]["network"]["actor_output_bias_init"] = 0.4
    cfg["params"]["network"]["actor_output_weight_init_scale"] = 0.0
    cfg["params"]["config"]["num_actors"] = 2
    cfg["params"]["config"]["steps_num"] = 1

    alg = SHAC(cfg, env=EarlyTerminationEnv())
    alg.target_critic = torch.nn.Linear(1, 1, bias=False)
    alg.target_critic.requires_grad_(False)
    with torch.no_grad():
        alg.target_critic.weight.fill_(1.0)

    loss = alg.compute_actor_loss()

    actor_action = torch.tanh(torch.tensor(0.4))
    expected_non_done_value = actor_action + 1.0
    expected_loss = -alg.gamma * expected_non_done_value / 2
    assert torch.allclose(loss, expected_loss)


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


def test_shac_clips_finite_extreme_observations_without_cutting_gradients(tmp_path):
    env = DummyDifferentiableVecEnv(num_envs=4, episode_length=16, device="cpu")
    cfg = _cfg(tmp_path)
    cfg["params"]["config"]["obs_clip"] = 10.0
    alg = SHAC(cfg, env=env)
    obs = torch.tensor(
        [[1.0e6, -1.0e6], [20.0, -20.0], [1.0, -2.0], [0.0, 3.0]],
        dtype=torch.float32,
        requires_grad=True,
    )

    processed = alg._preprocess_obs(obs)
    loss = processed[-1].sum()
    grad = torch.autograd.grad(loss, obs, allow_unused=True)[0]

    assert torch.isfinite(processed).all()
    assert processed.abs().max() <= 10.0
    assert torch.allclose(grad[-1], torch.ones_like(grad[-1]))


def test_shac_rejects_nonfinite_observations_instead_of_sanitizing(tmp_path):
    env = DummyDifferentiableVecEnv(num_envs=4, episode_length=16, device="cpu")
    cfg = _cfg(tmp_path)
    cfg["params"]["config"]["obs_clip"] = 10.0
    alg = SHAC(cfg, env=env)
    obs = torch.tensor(
        [[float("nan"), float("inf")], [1.0e6, -1.0e6], [1.0, -2.0], [0.0, 3.0]],
        dtype=torch.float32,
        requires_grad=True,
    )

    with pytest.raises(RuntimeError, match="observation contains non-finite"):
        alg._preprocess_obs(obs)


def test_shac_rejects_nonfinite_actor_gradients_before_clipping(tmp_path):
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

    with pytest.raises(RuntimeError, match="non-finite gradients"):
        alg.train()

    assert alg.actor_raw_grad_profile["nonfinite_count"] > 0
    assert alg.actor_raw_grad_profile["finite_count"] == 0
    assert not (tmp_path / "final_policy.pt").exists()
