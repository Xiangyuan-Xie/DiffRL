"""DiffRL package shim for ACELab integrations.

This repository keeps the original DiffRL source layout at the repository
root.  The package metadata maps the import package ``diffrl`` to this root so
new integrations can import ``diffrl.algorithms`` while legacy scripts can
continue to run from the checkout.
"""

__all__ = ["algorithms", "models", "utils"]

