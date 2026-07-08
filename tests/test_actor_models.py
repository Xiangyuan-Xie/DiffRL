from __future__ import annotations

import torch

from diffrl.models.actor import ActorDeterministicMLP


def test_deterministic_actor_can_initialize_output_bias():
    actor = ActorDeterministicMLP(
        obs_dim=3,
        action_dim=2,
        cfg_network={
            "actor_mlp": {"units": [4], "activation": "elu"},
            "actor_output_bias_init": [-0.25, 0.125],
        },
        device="cpu",
    )

    output = actor(torch.zeros(5, 3))

    assert torch.allclose(output, torch.tensor([[-0.25, 0.125]]).expand(5, -1))


def test_deterministic_actor_can_zero_initialize_output_weights_for_trim_policy():
    actor = ActorDeterministicMLP(
        obs_dim=3,
        action_dim=2,
        cfg_network={
            "actor_mlp": {"units": [4], "activation": "elu"},
            "actor_output_bias_init": [-0.25, 0.125],
            "actor_output_weight_init_scale": 0.0,
        },
        device="cpu",
    )

    output = actor(torch.randn(5, 3))

    assert torch.allclose(output, torch.tensor([[-0.25, 0.125]]).expand(5, -1), atol=1e-6)
