"""DiffRL-derived SHAC return and target-network updates in JAX.

Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
Licensed under the NVIDIA Source Code License for DiffRL; see LICENSE.md.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def td_lambda_returns(rewards, dones, values, bootstrap_value, gamma=0.99, lambda_=0.95):
    """Compute bootstrapped TD-lambda targets over a time-major rollout."""
    next_values = jnp.concatenate((values[1:], bootstrap_value[None, ...]), axis=0)

    def backward(carry, transition):
        reward, done, next_value = transition
        continuation = 1.0 - done
        target = reward + gamma * continuation * ((1.0 - lambda_) * next_value + lambda_ * carry)
        return target, target

    _, reversed_returns = jax.lax.scan(
        backward,
        bootstrap_value,
        (rewards[::-1], dones[::-1], next_values[::-1]),
    )
    return reversed_returns[::-1]


def actor_objective(rewards, dones, bootstrap_value, gamma=0.99):
    """Return the per-step negative truncated differentiable rollout objective."""
    discounts = jnp.cumprod(
        jnp.concatenate((jnp.ones_like(dones[:1]), gamma * (1.0 - dones[:-1])), axis=0),
        axis=0,
    )
    rollout_return = jnp.sum(discounts * rewards, axis=0)
    terminal_discount = discounts[-1] * gamma * (1.0 - dones[-1])
    return -jnp.mean(rollout_return + terminal_discount * bootstrap_value) / rewards.shape[0]


def critic_loss(predictions, targets):
    return jnp.mean((predictions - jax.lax.stop_gradient(targets)) ** 2)


def polyak_update(target_params, source_params, alpha=0.4):
    """Retain ``alpha`` of the target and blend in the source critic."""
    return jax.tree_util.tree_map(
        lambda target, source: alpha * target + (1.0 - alpha) * source, target_params, source_params
    )
