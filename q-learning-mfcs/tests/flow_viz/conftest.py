"""Shared fixtures for flow visualization tests."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
from cad.cad_config import FlowConfiguration, StackCADConfig
from flow_viz.config import FlowVizConfig


@pytest.fixture
def default_config() -> StackCADConfig:
    """Return default 10-cell stack configuration."""
    return StackCADConfig()


@pytest.fixture
def single_cell_config() -> StackCADConfig:
    """Single-cell stack for simpler checks."""
    return StackCADConfig(num_cells=1)


@pytest.fixture
def parallel_config() -> StackCADConfig:
    """Parallel flow configuration."""
    return StackCADConfig(flow_config=FlowConfiguration.PARALLEL)


@pytest.fixture
def viz_config(tmp_path: Path) -> FlowVizConfig:
    """Visualization config with temp output directory."""
    return FlowVizConfig(output_dir=tmp_path / "flow_viz")
