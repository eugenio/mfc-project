"""Shared fixtures for CAD tests."""

from __future__ import annotations

import os
import sys

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from cad.cad_config import (
    CathodeType,
    ElectrodeDimensions,
    FlowConfiguration,
    MembraneDimensions,
    SemiCellDimensions,
    StackCADConfig,
)


@pytest.fixture()
def default_config() -> StackCADConfig:
    """Default 10-cell stack configuration with standard dimensions."""
    return StackCADConfig()


@pytest.fixture()
def single_cell_config() -> StackCADConfig:
    """Single-cell stack for simpler geometry checks."""
    return StackCADConfig(num_cells=1)


@pytest.fixture()
def small_config() -> StackCADConfig:
    """Small 5 cm electrode stack matching the original simulator scale."""
    return StackCADConfig(
        num_cells=5,
        electrode=ElectrodeDimensions(side_length=0.05, thickness=0.005),
        semi_cell=SemiCellDimensions(
            inner_side=0.05,
            depth=0.011,
            wall_thickness=0.010,
        ),
        membrane=MembraneDimensions(
            thickness=1.78e-4,
            gasket_thickness=0.002,
            active_side=0.05,
        ),
    )
