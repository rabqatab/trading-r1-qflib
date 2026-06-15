"""Bootstrap: make the qf-lib-harness submodule's `alpha_lab` importable.

compare_lab lives in the project root repo; alpha_lab lives in the
qf-lib-harness submodule. We reuse alpha_lab.core's data loaders (read-only)
without vendoring or installing the harness as a package.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HARNESS = _REPO_ROOT / "qf-lib-harness"


def ensure_harness_on_path() -> Path:
    """Insert the submodule root on sys.path so `import alpha_lab` works.
    Returns the harness path. Idempotent."""
    if not _HARNESS.is_dir():
        raise RuntimeError(
            f"qf-lib-harness submodule not found at {_HARNESS}; "
            f"run: git submodule update --init"
        )
    p = str(_HARNESS)
    if p not in sys.path:
        sys.path.insert(0, p)
    return _HARNESS
