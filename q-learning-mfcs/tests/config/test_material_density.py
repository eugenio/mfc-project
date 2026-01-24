#!/usr/bin/env python3
"""Test material density in MaterialProperties."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.electrode_config import MaterialProperties


def test_material_properties_density():
    """Test that MaterialProperties can store density."""
    props = MaterialProperties(
        specific_conductance=25000,
        contact_resistance=0.1,
        surface_charge_density=-0.05,
        hydrophobicity_angle=75,
        surface_roughness=1.2,
        biofilm_adhesion_coefficient=1.0,
        attachment_energy=-12.5,
        density=2500  # kg/m³
    )
    
    assert props.density == 2500