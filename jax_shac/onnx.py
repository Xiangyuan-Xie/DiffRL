"""Single-file ONNX export for recurrent JAX actors."""

from __future__ import annotations

from pathlib import Path

from diffrl.jax_shac.checkpoint import load_actor_checkpoint, resolve_actor_timing_metadata


def export_actor_onnx(checkpoint, output, *, control_dt_s=None):
    """Export a deterministic dynamic-batch actor without external tensor data."""
    import onnx
    from jax2onnx import to_onnx

    from diffrl.jax_shac.models import actor_step

    params, metadata = load_actor_checkpoint(checkpoint)
    metadata = resolve_actor_timing_metadata(metadata, control_dt_s)
    observation_dim = int(metadata["observation_dim"])
    if observation_dim not in (134, 140):
        raise ValueError(f"Expected an AM Pose actor with 134 or 140 observations, got {observation_dim}.")
    action_dim = int(metadata.get("action_dim", 4))
    if action_dim != 4:
        raise ValueError(f"Expected a four-action rotor actor, got {action_dim} actions.")
    hidden_dim = int(metadata.get("hidden_dim", 32))
    if hidden_dim != 32:
        raise ValueError(f"Expected a GRU hidden size of 32, got {hidden_dim}.")

    def deterministic_actor(observations, hidden_state):
        actions, next_hidden_state, _ = actor_step(params, observations, hidden_state, deterministic=True)
        return actions, next_hidden_state

    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    to_onnx(
        deterministic_actor,
        [("B", observation_dim), (1, "B", hidden_dim)],
        input_names=["observations", "hidden_state"],
        output_names=["actions", "next_hidden_state"],
        return_mode="file",
        output_path=str(output),
    )
    model = onnx.load(output, load_external_data=False)
    external_tensors = [
        tensor.name
        for tensor in model.graph.initializer
        if tensor.data_location == onnx.TensorProto.EXTERNAL or tensor.external_data
    ]
    if external_tensors:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"ONNX export produced external tensor data: {external_tensors}.")
    exported_metadata = (
        "task",
        "algorithm",
        "control_dt_s",
        "sim_dt_s",
        "policy_rate_hz",
        "control_dt_source",
        "gradient_mode",
        "gradient_model",
        "gradient_forward_model",
    )
    for key in exported_metadata:
        if key not in metadata:
            continue
        property_entry = model.metadata_props.add()
        property_entry.key = key
        property_entry.value = str(metadata[key])
    onnx.save_model(model, output, save_as_external_data=False)
    data_path = output.with_name(output.name + ".data")
    if data_path.exists():
        data_path.unlink()
        output.unlink(missing_ok=True)
        raise RuntimeError("ONNX export produced a forbidden external .data file.")
    return output
