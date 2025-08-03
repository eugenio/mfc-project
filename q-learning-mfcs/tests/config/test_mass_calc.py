#!/usr/bin/env python3
"""Test mass calculation method."""

import pytest
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.electrode_config import ElectrodeGeometry, ElectrodeGeometrySpec


def test_calculate_mass_rectangular():
    """Test mass calculation for rectangular plate."""
    geometry = ElectrodeGeometrySpec(
        geometry_type=ElectrodeGeometry.RECTANGULAR_PLATE,
        length=0.05,  # 5 cm
        width=0.05,   # 5 cm
        thickness=0.005,  # 5 mm
        density=2500  # kg/m³ (graphite density)
    )
    
    expected_volume = 0.05 * 0.05 * 0.005  # m³
    expected_mass = expected_volume * 2500  # kg
    
    assert geometry.calculate_mass() == pytest.approx(expected_mass)