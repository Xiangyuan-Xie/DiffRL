import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from diffrl.jax_shac.models import critic_init, critic_sequence, critic_step
from diffrl.jax_shac.trainer import SHACConfig, SHACTrainer


def test_shac_config_uses_chunked_rollout_default():
    config = SHACConfig(num_envs=64, observation_dim=4, critic_observation_dim=6)

    assert config.horizon == 128
    assert config.gradient_horizon == 32


def test_recurrent_critic_preserves_history_and_resets_after_done():
    params = critic_init(jax.random.PRNGKey(3), observation_dim=6, hidden_dim=5)
    hidden = jnp.zeros((1, 1, 5))
    current = jnp.full((1, 6), 0.25)
    observations = jnp.stack((jnp.ones((1, 6)), current))

    uninterrupted_values, uninterrupted_hidden = critic_sequence(
        params,
        observations,
        hidden,
        jnp.zeros((2, 1)),
    )
    reset_values, reset_hidden = critic_sequence(
        params,
        observations,
        hidden,
        jnp.array([[1.0], [0.0]]),
    )
    zero_history_value, zero_history_hidden = critic_step(params, current, hidden)

    assert uninterrupted_values.shape == (2, 1)
    assert uninterrupted_hidden.shape == (1, 1, 5)
    assert not jnp.allclose(uninterrupted_values[1], zero_history_value)
    assert reset_values[1] == pytest.approx(zero_history_value)
    assert reset_hidden == pytest.approx(zero_history_hidden)


def test_update_rejects_nonfinite_actor_gradients_before_advancing():
    trainer = object.__new__(SHACTrainer)
    for name in (
        "actor_params",
        "critic_params",
        "target_critic_params",
        "actor_optimizer_state",
        "critic_optimizer_state",
        "states",
        "hidden",
        "critic_hidden",
        "target_critic_hidden",
        "critic_normalizer_state",
        "key",
    ):
        setattr(trainer, name, None)
    trainer.update_index = 7
    trainer._compiled_update = lambda *_: (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        {
            "actor_gradient_finite": jnp.array([0.0]),
            "critic_gradient_finite": jnp.array([1.0]),
        },
    )

    with pytest.raises(RuntimeError, match="actor gradients are non-finite"):
        trainer.update()

    assert trainer.update_index == 7


def test_update_rejects_nonfinite_critic_gradients_before_advancing():
    trainer = object.__new__(SHACTrainer)
    for name in (
        "actor_params",
        "critic_params",
        "target_critic_params",
        "actor_optimizer_state",
        "critic_optimizer_state",
        "states",
        "hidden",
        "critic_hidden",
        "target_critic_hidden",
        "critic_normalizer_state",
        "key",
    ):
        setattr(trainer, name, None)
    trainer.update_index = 9
    trainer._compiled_update = lambda *_: (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        {
            "actor_gradient_finite": jnp.array([1.0]),
            "critic_gradient_finite": jnp.array([0.0]),
        },
    )

    with pytest.raises(RuntimeError, match="critic gradients are non-finite"):
        trainer.update()

    assert trainer.update_index == 9
