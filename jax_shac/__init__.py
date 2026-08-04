"""Non-commercial JAX SHAC implementation derived from NVIDIA DiffRL."""

from diffrl.jax_shac.losses import polyak_update, td_lambda_returns
from diffrl.jax_shac.models import actor_init, actor_step, critic_init, critic_sequence, critic_step

__all__ = [
    "actor_init",
    "actor_step",
    "critic_init",
    "critic_sequence",
    "critic_step",
    "polyak_update",
    "td_lambda_returns",
]
