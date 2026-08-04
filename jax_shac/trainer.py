"""Differentiable short-horizon actor-critic trainer for ACELab MJX.

Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
Licensed under the NVIDIA Source Code License for DiffRL; see LICENSE.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax

from diffrl.jax_shac.losses import (
    actor_objective,
    critic_loss,
    polyak_update,
    td_lambda_returns,
)
from diffrl.jax_shac.models import actor_init, actor_step, critic_init, critic_sequence, critic_step


def _empty_running_statistics(feature_dim):
    return {
        "mean": jnp.zeros((feature_dim,)),
        "variance": jnp.ones((feature_dim,)),
        "count": jnp.array(0.0),
    }


def _update_running_statistics(statistics, observations, axis_name):
    """Merge a distributed observation batch into numerically stable running moments."""
    reduction_axes = tuple(range(observations.ndim - 1))
    local_count = jnp.array(observations.size // observations.shape[-1], dtype=observations.dtype)
    batch_count = jax.lax.psum(local_count, axis_name=axis_name)
    batch_sum = jax.lax.psum(jnp.sum(observations, axis=reduction_axes), axis_name=axis_name)
    batch_square_sum = jax.lax.psum(
        jnp.sum(jnp.square(observations), axis=reduction_axes),
        axis_name=axis_name,
    )
    batch_mean = batch_sum / batch_count
    batch_variance = jnp.maximum(batch_square_sum / batch_count - jnp.square(batch_mean), 0.0)

    previous_count = statistics["count"]
    total_count = previous_count + batch_count
    delta = batch_mean - statistics["mean"]
    mean = statistics["mean"] + delta * batch_count / total_count
    previous_m2 = statistics["variance"] * previous_count
    batch_m2 = batch_variance * batch_count
    correction = jnp.square(delta) * previous_count * batch_count / total_count
    variance = jnp.maximum((previous_m2 + batch_m2 + correction) / total_count, 0.0)
    return {"mean": mean, "variance": variance, "count": total_count}


def _normalize_critic_observations(statistics, observations):
    normalized = (observations - statistics["mean"]) * jax.lax.rsqrt(statistics["variance"] + 1.0e-4)
    return jnp.clip(normalized, -10.0, 10.0)


@dataclass(frozen=True)
class SHACConfig:
    num_envs: int
    observation_dim: int
    critic_observation_dim: int
    action_dim: int = 4
    hidden_dim: int = 32
    critic_hidden_dim: int = 32
    gamma: float = 0.99
    lambda_: float = 0.95
    learning_rate: float = 3.0e-4
    critic_learning_rate: float = 3.0e-4
    critic_iterations: int = 16
    critic_minibatches: int = 4
    target_alpha: float = 0.4
    horizon: int = 128
    gradient_horizon: int = 32


class SHACTrainer:
    """JIT-compatible SHAC updates with gradients through MJX dynamics."""

    def __init__(self, env, config: SHACConfig, seed=0, devices=None):
        self.env = env
        self.config = config
        if config.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {config.horizon}.")
        if config.gradient_horizon <= 0:
            raise ValueError(f"gradient_horizon must be positive, got {config.gradient_horizon}.")
        if config.horizon % config.gradient_horizon:
            raise ValueError(
                f"horizon={config.horizon} must be divisible by gradient_horizon={config.gradient_horizon}."
            )
        if config.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {config.hidden_dim}.")
        if config.critic_hidden_dim <= 0:
            raise ValueError(f"critic_hidden_dim must be positive, got {config.critic_hidden_dim}.")
        if not 0.0 <= config.target_alpha <= 1.0:
            raise ValueError(f"target_alpha must be in [0, 1], got {config.target_alpha}.")
        if config.critic_iterations <= 0:
            raise ValueError(f"critic_iterations must be positive, got {config.critic_iterations}.")
        if config.critic_minibatches <= 0:
            raise ValueError(f"critic_minibatches must be positive, got {config.critic_minibatches}.")
        self.devices = tuple(jax.local_devices() if devices is None else devices)
        if not self.devices:
            raise ValueError("SHACTrainer requires at least one local JAX device.")
        self.device_count = len(self.devices)
        if config.num_envs % self.device_count:
            raise ValueError(f"num_envs={config.num_envs} must be divisible by {self.device_count} local JAX devices.")
        self.envs_per_device = config.num_envs // self.device_count
        if self.envs_per_device % config.critic_minibatches:
            raise ValueError(
                f"Environments per device {self.envs_per_device} must be divisible by "
                f"critic_minibatches={config.critic_minibatches}; recurrent critic minibatches keep complete sequences."
            )
        key = jax.random.PRNGKey(seed)
        actor_key, critic_key, reset_key = jax.random.split(key, 3)
        actor_params = actor_init(actor_key, config.observation_dim, config.action_dim, config.hidden_dim)
        critic_params = critic_init(critic_key, config.critic_observation_dim, config.critic_hidden_dim)
        self.actor_optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adam(config.learning_rate, b1=0.7, b2=0.95)
        )
        self.critic_optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adam(config.critic_learning_rate, b1=0.7, b2=0.95)
        )

        def replicate(tree):
            return jax.tree.map(lambda value: jnp.broadcast_to(value, (self.device_count, *value.shape)), tree)

        self.actor_params = replicate(actor_params)
        self.critic_params = replicate(critic_params)
        self.target_critic_params = replicate(critic_params)
        self.actor_optimizer_state = replicate(self.actor_optimizer.init(actor_params))
        self.critic_optimizer_state = replicate(self.critic_optimizer.init(critic_params))
        reset_keys = jax.random.split(reset_key, config.num_envs).reshape(self.device_count, self.envs_per_device, 2)
        self.states = jax.pmap(jax.vmap(env.reset), devices=self.devices)(reset_keys)
        self.hidden = jnp.zeros((self.device_count, 1, self.envs_per_device, config.hidden_dim))
        self.critic_hidden = jnp.zeros(
            (self.device_count, 1, self.envs_per_device, config.critic_hidden_dim)
        )
        self.target_critic_hidden = jnp.zeros_like(self.critic_hidden)
        empty_statistics = replicate(_empty_running_statistics(config.critic_observation_dim))
        self.critic_normalizer_state = jax.pmap(
            lambda statistics, observations: _update_running_statistics(statistics, observations, "devices"),
            axis_name="devices",
            devices=self.devices,
        )(empty_statistics, self.states.obs["privileged_state"])
        self.key = jax.random.split(key, self.device_count)
        self.update_index = 0
        self._compiled_update = jax.pmap(
            lambda *args: self._update(*args, config.horizon),
            axis_name="devices",
            devices=self.devices,
        )

    @property
    def inference_actor_params(self):
        """Return one synchronized actor replica for evaluation and export."""
        return jax.tree.map(lambda value: value[0], self.actor_params)

    def _rollout(
        self,
        actor_params,
        target_critic_params,
        states,
        hidden,
        target_critic_hidden,
        critic_normalizer_state,
        horizon,
    ):
        def rollout_step(carry, _):
            states, hidden, target_critic_hidden = carry
            critic_values, next_target_critic_hidden = critic_step(
                target_critic_params,
                _normalize_critic_observations(
                    critic_normalizer_state,
                    states.obs["privileged_state"],
                ),
                target_critic_hidden,
            )
            actions, next_hidden, _ = actor_step(actor_params, states.obs["state"], hidden, deterministic=True)
            next_states = jax.vmap(self.env.step)(states, actions)
            dones = next_states.done
            next_hidden = next_hidden * (1.0 - dones)[None, :, None]
            next_target_critic_hidden = next_target_critic_hidden * (1.0 - dones)[None, :, None]
            transition = {
                "observations": states.obs["privileged_state"],
                "critic_values": critic_values,
                "rewards": next_states.reward,
                "dones": dones,
                **{f"episode_{name}": value for name, value in next_states.metrics.items()},
            }

            def reset_done_environments(states):
                split_keys = jax.vmap(jax.random.split)(states.info["rng"])
                reset_states = jax.vmap(self.env.reset)(split_keys[:, 0])
                continuing_info = dict(states.info)
                continuing_info["rng"] = split_keys[:, 1]
                continuing_states = states.replace(info=continuing_info)

                def select(reset_value, continuing_value):
                    mask = dones.reshape(dones.shape + (1,) * (reset_value.ndim - dones.ndim))
                    return jnp.where(mask, reset_value, continuing_value)

                return jax.tree.map(select, reset_states, continuing_states)

            rollout_states = jax.lax.cond(
                jnp.any(dones.astype(bool)),
                reset_done_environments,
                lambda states: states,
                next_states,
            )
            return (rollout_states, next_hidden, next_target_critic_hidden), transition

        return jax.lax.scan(
            rollout_step,
            (states, hidden, target_critic_hidden),
            None,
            length=horizon,
        )

    def _chunked_actor_rollout(
        self,
        actor_params,
        target_critic_params,
        states,
        hidden,
        target_critic_hidden,
        critic_normalizer_state,
        horizon,
    ):
        gradient_horizon = self.config.gradient_horizon
        chunk_count = horizon // gradient_horizon

        def rollout_chunk(carry, _):
            states, hidden, target_critic_hidden = jax.tree.map(jax.lax.stop_gradient, carry)
            (states, hidden, target_critic_hidden), rollout = self._rollout(
                actor_params,
                target_critic_params,
                states,
                hidden,
                target_critic_hidden,
                critic_normalizer_state,
                gradient_horizon,
            )
            bootstrap, _ = critic_step(
                target_critic_params,
                _normalize_critic_observations(
                    critic_normalizer_state,
                    states.obs["privileged_state"],
                ),
                target_critic_hidden,
            )
            loss = actor_objective(rollout["rewards"], rollout["dones"], bootstrap, self.config.gamma)
            return (states, hidden, target_critic_hidden), (rollout, bootstrap, loss)

        (states, hidden, target_critic_hidden), (chunked_rollout, chunk_bootstrap, chunk_loss) = jax.lax.scan(
            rollout_chunk,
            (states, hidden, target_critic_hidden),
            None,
            length=chunk_count,
        )
        rollout = jax.tree.map(
            lambda value: value.reshape((horizon, *value.shape[2:])),
            chunked_rollout,
        )
        return (
            states,
            hidden,
            target_critic_hidden,
        ), rollout, chunk_bootstrap[-1], jnp.mean(chunk_loss)

    def _update(
        self,
        actor_params,
        critic_params,
        target_critic_params,
        actor_optimizer_state,
        critic_optimizer_state,
        states,
        hidden,
        critic_hidden,
        target_critic_hidden,
        critic_normalizer_state,
        key,
        horizon,
    ):
        def actor_loss_fn(actor_params):
            (next_states, next_hidden, next_target_critic_hidden), rollout, bootstrap, loss = (
                self._chunked_actor_rollout(
                    actor_params,
                    target_critic_params,
                    states,
                    hidden,
                    target_critic_hidden,
                    critic_normalizer_state,
                    horizon,
                )
            )
            return loss, (
                next_states,
                next_hidden,
                next_target_critic_hidden,
                rollout,
                bootstrap,
            )

        (actor_loss_value, auxiliary), gradients = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_params)
        gradients = jax.lax.pmean(gradients, axis_name="devices")
        actor_gradient_norm = optax.tree.norm(gradients)
        local_actor_gradient_finite = (
            jnp.isfinite(actor_loss_value)
            & jnp.isfinite(actor_gradient_norm)
            & jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(gradients)]))
        )
        actor_gradient_finite = jax.lax.pmin(
            local_actor_gradient_finite.astype(jnp.int32), axis_name="devices"
        ).astype(bool)

        def apply_actor_update(_):
            actor_updates, next_optimizer_state = self.actor_optimizer.update(
                gradients,
                actor_optimizer_state,
                actor_params,
            )
            return optax.apply_updates(actor_params, actor_updates), next_optimizer_state

        actor_params, actor_optimizer_state = jax.lax.cond(
            actor_gradient_finite,
            apply_actor_update,
            lambda _: (actor_params, actor_optimizer_state),
            operand=None,
        )
        states, hidden, _, rollout, bootstrap = auxiliary
        values = rollout["critic_values"]
        targets = td_lambda_returns(
            rollout["rewards"],
            rollout["dones"],
            values,
            bootstrap,
            self.config.gamma,
            self.config.lambda_,
        )
        observations = rollout["observations"]
        normalized_observations = _normalize_critic_observations(critic_normalizer_state, observations)
        dones = rollout["dones"]
        environment_count = observations.shape[1]

        key, critic_key = jax.random.split(key)

        def critic_iteration(carry, iteration):
            critic_params, optimizer_state, _, all_gradients_finite, max_gradient_norm = carry
            permutation = jax.random.permutation(jax.random.fold_in(critic_key, iteration), environment_count)
            minibatches = permutation.reshape(self.config.critic_minibatches, -1)

            def critic_minibatch(carry, minibatch):
                critic_params, optimizer_state, _, all_gradients_finite, max_gradient_norm = carry

                def loss_fn(params):
                    predictions, _ = critic_sequence(
                        params,
                        normalized_observations[:, minibatch],
                        critic_hidden[:, minibatch],
                        dones[:, minibatch],
                    )
                    return critic_loss(predictions, targets[:, minibatch])

                loss, critic_gradients = jax.value_and_grad(loss_fn)(critic_params)
                critic_gradients = jax.lax.pmean(critic_gradients, axis_name="devices")
                gradient_norm = optax.tree.norm(critic_gradients)
                local_gradients_finite = (
                    jnp.isfinite(loss)
                    & jnp.isfinite(gradient_norm)
                    & jnp.all(
                        jnp.stack([jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(critic_gradients)])
                    )
                )
                gradients_finite = jax.lax.pmin(
                    local_gradients_finite.astype(jnp.int32), axis_name="devices"
                ).astype(bool)

                def apply_critic_update(_):
                    critic_updates, next_optimizer_state = self.critic_optimizer.update(
                        critic_gradients,
                        optimizer_state,
                        critic_params,
                    )
                    return optax.apply_updates(critic_params, critic_updates), next_optimizer_state

                critic_params, optimizer_state = jax.lax.cond(
                    gradients_finite,
                    apply_critic_update,
                    lambda _: (critic_params, optimizer_state),
                    operand=None,
                )
                all_gradients_finite &= gradients_finite
                max_gradient_norm = jnp.maximum(
                    max_gradient_norm,
                    jnp.where(gradients_finite, gradient_norm, jnp.inf),
                )
                return (
                    critic_params,
                    optimizer_state,
                    loss,
                    all_gradients_finite,
                    max_gradient_norm,
                ), None

            return (
                jax.lax.scan(
                    critic_minibatch,
                    (
                        critic_params,
                        optimizer_state,
                        jnp.array(0.0),
                        all_gradients_finite,
                        max_gradient_norm,
                    ),
                    minibatches,
                )[0],
                None,
            )

        (
            critic_params,
            critic_optimizer_state,
            critic_loss_value,
            critic_gradient_finite,
            critic_gradient_norm,
        ), _ = jax.lax.scan(
            critic_iteration,
            (
                critic_params,
                critic_optimizer_state,
                jnp.array(0.0),
                jnp.array(True),
                jnp.array(0.0),
            ),
            jnp.arange(self.config.critic_iterations),
        )
        target_critic_params = jax.lax.cond(
            critic_gradient_finite,
            lambda _: polyak_update(target_critic_params, critic_params, self.config.target_alpha),
            lambda _: target_critic_params,
            operand=None,
        )
        critic_normalizer_state = _update_running_statistics(
            critic_normalizer_state,
            observations,
            "devices",
        )
        normalized_observations = _normalize_critic_observations(critic_normalizer_state, observations)
        _, critic_hidden = critic_sequence(
            critic_params,
            normalized_observations,
            critic_hidden,
            dones,
        )
        _, target_critic_hidden = critic_sequence(
            target_critic_params,
            normalized_observations,
            target_critic_hidden,
            dones,
        )
        metrics = {
            "actor_loss": jax.lax.pmean(actor_loss_value, axis_name="devices"),
            "critic_loss": jax.lax.pmean(critic_loss_value, axis_name="devices"),
            "mean_reward": jax.lax.pmean(jnp.mean(rollout["rewards"]), axis_name="devices"),
            "mean_rollout_return": jax.lax.pmean(jnp.mean(jnp.sum(rollout["rewards"], axis=0)), axis_name="devices"),
            "horizon": jnp.array(horizon),
            "gradient_horizon": jnp.array(self.config.gradient_horizon),
            "actor_gradient_norm": actor_gradient_norm,
            "actor_gradient_finite": actor_gradient_finite.astype(jnp.float32),
            "critic_gradient_norm": critic_gradient_norm,
            "critic_gradient_finite": critic_gradient_finite.astype(jnp.float32),
            "critic_normalizer_count": critic_normalizer_state["count"],
            **{
                name: jax.lax.pmean(jnp.mean(value), axis_name="devices")
                for name, value in rollout.items()
                if name.startswith("episode_")
            },
        }
        return (
            actor_params,
            critic_params,
            target_critic_params,
            actor_optimizer_state,
            critic_optimizer_state,
            states,
            hidden,
            critic_hidden,
            target_critic_hidden,
            critic_normalizer_state,
            key,
            metrics,
        )

    def update(self):
        (
            self.actor_params,
            self.critic_params,
            self.target_critic_params,
            self.actor_optimizer_state,
            self.critic_optimizer_state,
            self.states,
            self.hidden,
            self.critic_hidden,
            self.target_critic_hidden,
            self.critic_normalizer_state,
            self.key,
            metrics,
        ) = self._compiled_update(
            self.actor_params,
            self.critic_params,
            self.target_critic_params,
            self.actor_optimizer_state,
            self.critic_optimizer_state,
            self.states,
            self.hidden,
            self.critic_hidden,
            self.target_critic_hidden,
            self.critic_normalizer_state,
            self.key,
        )
        metrics = jax.tree.map(lambda value: value[0], metrics)
        if not bool(metrics["actor_gradient_finite"]):
            raise RuntimeError(
                "SHAC actor gradients are non-finite; training stopped before applying a corrupted update."
            )
        if not bool(metrics["critic_gradient_finite"]):
            raise RuntimeError(
                "SHAC critic gradients are non-finite; training stopped before updating the target critic."
            )
        self.update_index += 1
        return metrics
