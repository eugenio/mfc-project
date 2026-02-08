"""STEP model loader for pump with parametric fallback.

Attempts to load vendor STEP file; falls back to parametric pump_head.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent.parent.parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import cadquery as cq

from cad.cad_config import StackCADConfig
from cad.components import pump_head


def load_pump_step(path: Path | str) -> cq.Workplane | None:
    """Try to import a STEP file. Returns None on failure."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return cq.importers.importStep(str(p))
    except Exception:
        return None


def build(
    config: StackCADConfig,
    step_file: Path | str | None = None,
) -> cq.Workplane:
    """Load STEP pump model if available, else parametric fallback."""
    if step_file is not None:
        result = load_pump_step(step_file)
        if result is not None:
            return result
    return pump_head.build(config)
