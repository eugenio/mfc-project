#!/usr/bin/env python3
"""Test mass calculation error handling."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.electrode_config import ElectrodeGeometry, ElectrodeGeometrySpec


def test_calculate_mass_no_density():
    """Test that mass calculation fails without density."""
    geometry = ElectrodeGeometrySpec(
        geometry_type=ElectrodeGeometry.RECTANGULAR_PLATE,
        length=0.05,
        width=0.05,
        thickness=0.005
        # No density specified
    )
    
    with pytest.raises(ValueError, match="Density not specified"):
        geometry.calculate_mass()