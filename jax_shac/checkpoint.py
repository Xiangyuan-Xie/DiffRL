"""Orbax checkpoint helpers for ACELab JAX policies."""

from __future__ import annotations

import json
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
