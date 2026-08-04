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


def _gru_step(input_params, hidden_params, inputs, hidden):
    """Apply one GRU step without changing the public hidden-state layout."""
    input_gates = _linear(input_params, inputs)
    hidden_gates = _linear(hidden_params, hidden)
    input_z, input_r, input_n = jnp.split(input_gates, 3, axis=-1)
    hidden_z, hidden_r, hidden_n = jnp.split(hidden_gates, 3, axis=-1)
    update = jax.nn.sigmoid(input_z + hidden_z)
    reset = jax.nn.sigmoid(input_r + hidden_r)
    candidate = jnp.tanh(input_n + reset * hidden_n)
    return update * hidden + (1.0 - update) * candidate


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
    next_hidden = _gru_step(params["gru_input"], params["gru_hidden"], features, hidden)
    alpha = jax.nn.softplus(_linear(params["alpha"], next_hidden)) + 1.0
    beta = jax.nn.softplus(_linear(params["beta"], next_hidden)) + 1.0
    if deterministic:
        actions = alpha / (alpha + beta)
    else:
        if key is None:
            raise ValueError("A PRNG key is required for stochastic actor inference.")
        actions = jax.random.beta(key, alpha, beta)
    return actions, next_hidden[None, ...], (alpha, beta)


def critic_init(key, observation_dim, hidden_dim=32):
    """Initialize a Dense[128,128] + GRU critic."""
    keys = jax.random.split(key, 5)
    return {
        "dense_0": _linear_init(keys[0], observation_dim, 128),
        "dense_1": _linear_init(keys[1], 128, 128),
        "gru_input": _linear_init(keys[2], 128, 3 * hidden_dim),
        "gru_hidden": _linear_init(keys[3], hidden_dim, 3 * hidden_dim),
        "value": _linear_init(keys[4], hidden_dim, 1, scale=0.01),
    }


def critic_step(params, observations, hidden_state):
    """Evaluate one critic step and return its value and recurrent state."""
    hidden = hidden_state[0] if hidden_state.ndim == 3 else hidden_state
    features = jax.nn.elu(_linear(params["dense_0"], observations))
    features = jax.nn.elu(_linear(params["dense_1"], features))
    next_hidden = _gru_step(params["gru_input"], params["gru_hidden"], features, hidden)
    values = _linear(params["value"], next_hidden)[..., 0]
    return values, next_hidden[None, ...]


def critic_sequence(params, observations, hidden_state, dones):
    """Evaluate a time-major sequence and reset memory after terminal transitions."""

    def sequence_step(hidden, transition):
        observation, done = transition
        value, next_hidden = critic_step(params, observation, hidden)
        continuation = 1.0 - done.astype(observation.dtype)
        next_hidden = next_hidden * continuation[None, :, None]
        return next_hidden, value

    next_hidden, values = jax.lax.scan(sequence_step, hidden_state, (observations, dones))
    return values, next_hidden
