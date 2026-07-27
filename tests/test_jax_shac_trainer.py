import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from diffrl.jax_shac.trainer import SHACTrainer


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
        {"actor_gradient_finite": jnp.array([0.0])},
    )

    with pytest.raises(RuntimeError, match="actor gradients are non-finite"):
        trainer.update()

    assert trainer.update_index == 7
