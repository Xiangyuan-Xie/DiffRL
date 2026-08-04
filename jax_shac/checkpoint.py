"""Orbax checkpoint helpers for ACELab JAX policies."""

from __future__ import annotations

import json
import math
from pathlib import Path


def save_actor_checkpoint(path, actor_params, metadata):
    """Save actor parameters and their public contract."""
    import orbax.checkpoint as ocp

    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    ocp.PyTreeCheckpointer().save(path / "actor", actor_params, force=True)
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_actor_checkpoint(path):
    """Restore actor parameters and metadata."""
    import orbax.checkpoint as ocp

    path = Path(path).resolve()
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    actor_params = ocp.PyTreeCheckpointer().restore(path / "actor")
    return actor_params, metadata


def resolve_actor_timing_metadata(metadata, control_dt_s=None):
    """Validate checkpoint timing or add an explicit legacy override."""

    resolved = dict(metadata)
    saved_control_dt = resolved.get("control_dt_s")
    if control_dt_s is not None:
        control_dt_s = float(control_dt_s)
        if not math.isfinite(control_dt_s) or control_dt_s <= 0.0:
            raise ValueError(f"control_dt_s must be finite and positive, got {control_dt_s!r}.")
        if saved_control_dt is not None and not math.isclose(
            control_dt_s, float(saved_control_dt), rel_tol=1.0e-9, abs_tol=1.0e-12
        ):
            raise ValueError(
                f"control_dt_s={control_dt_s} conflicts with checkpoint control_dt_s={saved_control_dt}."
            )
        if saved_control_dt is None:
            resolved["control_dt_s"] = control_dt_s
            resolved["control_dt_source"] = "cli_override"
    if "control_dt_s" not in resolved:
        raise ValueError(
            "Checkpoint metadata has no control_dt_s; pass an explicit verified control_dt_s for this legacy model."
        )

    effective_control_dt = float(resolved["control_dt_s"])
    if not math.isfinite(effective_control_dt) or effective_control_dt <= 0.0:
        raise ValueError(f"Checkpoint control_dt_s must be finite and positive, got {effective_control_dt!r}.")
    expected_rate = 1.0 / effective_control_dt
    saved_rate = resolved.get("policy_rate_hz")
    if saved_rate is not None and not math.isclose(
        float(saved_rate), expected_rate, rel_tol=1.0e-9, abs_tol=1.0e-9
    ):
        raise ValueError(
            f"Checkpoint policy_rate_hz={saved_rate} conflicts with control_dt_s={effective_control_dt}."
        )
    resolved["policy_rate_hz"] = expected_rate
    return resolved
