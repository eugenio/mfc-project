#!/usr/bin/env python3
"""Test density property."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.electrode_config import ElectrodeGeometry, ElectrodeGeometrySpec


def test_density_in_init():
    """Test density parameter in __init__."""
    geometry = ElectrodeGeometrySpec(
        geometry_type=ElectrodeGeometry.RECTANGULAR_PLATE,
        length=0.05,
        width=0.05,
        thickness=0.005,
        density=2700
    )
    assert geometry.density == 2700