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
from diffrl.jax_shac.models import actor_init, actor_step, critic_apply, critic_init


@dataclass(frozen=True)
class SHACConfig:
    num_envs: int
    observation_dim: int
    critic_observation_dim: int
    action_dim: int = 4
    hidden_dim: int = 32
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
        self.devices = tuple(jax.local_devices() if devices is None else devices)
        if not self.devices:
            raise ValueError("SHACTrainer requires at least one local JAX device.")
        self.device_count = len(self.devices)
        if config.num_envs % self.device_count:
            raise ValueError(f"num_envs={config.num_envs} must be divisible by {self.device_count} local JAX devices.")
        self.envs_per_device = config.num_envs // self.device_count
        key = jax.random.PRNGKey(seed)
        actor_key, critic_key, reset_key = jax.random.split(key, 3)
        actor_params = actor_init(actor_key, config.observation_dim, config.action_dim, config.hidden_dim)
        critic_params = critic_init(critic_key, config.critic_observation_dim)
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

    def _rollout(self, actor_params, states, hidden, horizon):
        def rollout_step(carry, _):
            states, hidden = carry
            actions, next_hidden, _ = actor_step(actor_params, states.obs["state"], hidden, deterministic=True)
            next_states = jax.vmap(self.env.step)(states, actions)
            dones = next_states.done
            next_hidden = next_hidden * (1.0 - dones)[None, :, None]
            transition = {
                "observations": states.obs["privileged_state"],
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
            return (rollout_states, next_hidden), transition

        return jax.lax.scan(rollout_step, (states, hidden), None, length=horizon)

    def _chunked_actor_rollout(self, actor_params, target_critic_params, states, hidden, horizon):
        gradient_horizon = self.config.gradient_horizon
        chunk_count = horizon // gradient_horizon

        def rollout_chunk(carry, _):
            states, hidden = jax.tree.map(jax.lax.stop_gradient, carry)
            (states, hidden), rollout = self._rollout(
                actor_params,
                states,
                hidden,
                gradient_horizon,
            )
            bootstrap = critic_apply(target_critic_params, states.obs["privileged_state"])
            loss = actor_objective(rollout["rewards"], rollout["dones"], bootstrap, self.config.gamma)
            return (states, hidden), (rollout, bootstrap, loss)

        (states, hidden), (chunked_rollout, chunk_bootstrap, chunk_loss) = jax.lax.scan(
            rollout_chunk,
            (states, hidden),
            None,
            length=chunk_count,
        )
        rollout = jax.tree.map(
            lambda value: value.reshape((horizon, *value.shape[2:])),
            chunked_rollout,
        )
        return (states, hidden), rollout, chunk_bootstrap[-1], jnp.mean(chunk_loss)

    def _update(
        self,
        actor_params,
        critic_params,
        target_critic_params,
        actor_optimizer_state,
        critic_optimizer_state,
        states,
        hidden,
        key,
        horizon,
    ):
        def actor_loss_fn(actor_params):
            (next_states, next_hidden), rollout, bootstrap, loss = self._chunked_actor_rollout(
                actor_params,
                target_critic_params,
                states,
                hidden,
                horizon,
            )
            return loss, (next_states, next_hidden, rollout, bootstrap)

        (actor_loss_value, auxiliary), gradients = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_params)
        gradients = jax.lax.pmean(gradients, axis_name="devices")
        actor_gradient_norm = optax.tree.norm(gradients)
        actor_gradient_finite = jnp.isfinite(actor_gradient_norm) & jnp.all(
            jnp.stack([jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(gradients)])
        )

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
        states, hidden, rollout, bootstrap = auxiliary
        values = critic_apply(target_critic_params, rollout["observations"])
        targets = td_lambda_returns(
            rollout["rewards"],
            rollout["dones"],
            values,
            bootstrap,
            self.config.gamma,
            self.config.lambda_,
        )
        observations = rollout["observations"].reshape(-1, self.config.critic_observation_dim)
        targets = targets.reshape(-1)
        batch_size = observations.shape[0]
        if batch_size % self.config.critic_minibatches:
            raise ValueError(
                f"Rollout batch size {batch_size} must be divisible by "
                f"{self.config.critic_minibatches} critic minibatches."
            )

        key, critic_key = jax.random.split(key)

        def critic_iteration(carry, iteration):
            critic_params, optimizer_state, _ = carry
            permutation = jax.random.permutation(jax.random.fold_in(critic_key, iteration), batch_size)
            minibatches = permutation.reshape(self.config.critic_minibatches, -1)

            def critic_minibatch(carry, minibatch):
                critic_params, optimizer_state, _ = carry

                def loss_fn(params):
                    return critic_loss(critic_apply(params, observations[minibatch]), targets[minibatch])

                loss, critic_gradients = jax.value_and_grad(loss_fn)(critic_params)
                critic_gradients = jax.lax.pmean(critic_gradients, axis_name="devices")
                critic_updates, optimizer_state = self.critic_optimizer.update(
                    critic_gradients,
                    optimizer_state,
                    critic_params,
                )
                critic_params = optax.apply_updates(critic_params, critic_updates)
                return (critic_params, optimizer_state, loss), None

            return (
                jax.lax.scan(
                    critic_minibatch,
                    (critic_params, optimizer_state, jnp.array(0.0)),
                    minibatches,
                )[0],
                None,
            )

        (critic_params, critic_optimizer_state, critic_loss_value), _ = jax.lax.scan(
            critic_iteration,
            (critic_params, critic_optimizer_state, jnp.array(0.0)),
            jnp.arange(self.config.critic_iterations),
        )
        target_critic_params = polyak_update(target_critic_params, critic_params, self.config.target_alpha)
        metrics = {
            "actor_loss": jax.lax.pmean(actor_loss_value, axis_name="devices"),
            "critic_loss": jax.lax.pmean(critic_loss_value, axis_name="devices"),
            "mean_reward": jax.lax.pmean(jnp.mean(rollout["rewards"]), axis_name="devices"),
            "mean_rollout_return": jax.lax.pmean(jnp.mean(jnp.sum(rollout["rewards"], axis=0)), axis_name="devices"),
            "horizon": jnp.array(horizon),
            "gradient_horizon": jnp.array(self.config.gradient_horizon),
            "actor_gradient_norm": actor_gradient_norm,
            "actor_gradient_finite": actor_gradient_finite.astype(jnp.float32),
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
            self.key,
        )
        self.update_index += 1
        return jax.tree.map(lambda value: value[0], metrics)
