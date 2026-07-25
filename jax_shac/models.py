"""JAX recurrent actor and critic used by the DiffRL-derived SHAC trainer.

Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
Licensed under the NVIDIA Source Code License for DiffRL; see LICENSE.md.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _linear_init(key, input_dim, output_dim, scale=1.0):
    limit = scale * jnp.sqrt(6.0 / (input_dim + output_dim))
    return {
        "kernel": jax.random.uniform(key, (input_dim, output_dim), minval=-limit, maxval=limit),
        "bias": jnp.zeros((output_dim,)),
    }


def _linear(params, inputs):
    return inputs @ params["kernel"] + params["bias"]


def actor_init(key, observation_dim, action_dim=4, hidden_dim=32):
    """Initialize a Dense[64,32] + GRU32 Beta actor."""
    keys = jax.random.split(key, 7)
    return {
        "dense_0": _linear_init(keys[0], observation_dim, 64),
        "dense_1": _linear_init(keys[1], 64, 32),
        "gru_input": _linear_init(keys[2], 32, 3 * hidden_dim),
        "gru_hidden": _linear_init(keys[3], hidden_dim, 3 * hidden_dim),
        "alpha": _linear_init(keys[4], hidden_dim, action_dim, scale=0.01),
        "beta": _linear_init(keys[5], hidden_dim, action_dim, scale=0.01),
    }


def actor_step(params, observations, hidden_state, key=None, deterministic=True):
    """Evaluate one recurrent step and return Beta actions and next hidden state."""
    hidden = hidden_state[0] if hidden_state.ndim == 3 else hidden_state
    features = jax.nn.elu(_linear(params["dense_0"], observations))
    features = jax.nn.elu(_linear(params["dense_1"], features))
    input_gates = _linear(params["gru_input"], features)
    hidden_gates = _linear(params["gru_hidden"], hidden)
    input_z, input_r, input_n = jnp.split(input_gates, 3, axis=-1)
    hidden_z, hidden_r, hidden_n = jnp.split(hidden_gates, 3, axis=-1)
    update = jax.nn.sigmoid(input_z + hidden_z)
    reset = jax.nn.sigmoid(input_r + hidden_r)
    candidate = jnp.tanh(input_n + reset * hidden_n)
    next_hidden = update * hidden + (1.0 - update) * candidate
    alpha = jax.nn.softplus(_linear(params["alpha"], next_hidden)) + 1.0
    beta = jax.nn.softplus(_linear(params["beta"], next_hidden)) + 1.0
    if deterministic:
        actions = alpha / (alpha + beta)
    else:
        if key is None:
            raise ValueError("A PRNG key is required for stochastic actor inference.")
        actions = jax.random.beta(key, alpha, beta)
    return actions, next_hidden[None, ...], (alpha, beta)


def critic_init(key, observation_dim):
    keys = jax.random.split(key, 3)
    return {
        "dense_0": _linear_init(keys[0], observation_dim, 128),
        "dense_1": _linear_init(keys[1], 128, 128),
        "value": _linear_init(keys[2], 128, 1, scale=0.01),
    }


def critic_apply(params, observations):
    features = jax.nn.elu(_linear(params["dense_0"], observations))
    features = jax.nn.elu(_linear(params["dense_1"], features))
    return _linear(params["value"], features)[..., 0]
