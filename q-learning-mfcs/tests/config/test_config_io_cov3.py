"""Tests for config_io.py - coverage part 3.

Targets missing lines: cathode_config deserialization (203-220).
"""
import sys
import os
import json
from copy import deepcopy

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from config.config_io import (
    dataclass_to_dict,
    dict_to_dataclass,
)
from config.qlearning_config import QLearningConfig, DEFAULT_QLEARNING_CONFIG
from config.electrode_config import (
    ElectrodeConfiguration,
    ElectrodeGeometry,
    ElectrodeGeometrySpec,
    ElectrodeMaterial,
    MaterialProperties,
)


@pytest.mark.coverage_extra
class TestDictToDataclassCathodeConfig:
    """Cover cathode_config dict branch (lines 203-220)."""

    def test_cathode_config_deserialization_full(self):
        """Cover all sub-branches: material str, geometry dict, material_properties dict."""
        cfg = deepcopy(DEFAULT_QLEARNING_CONFIG)
        d = dataclass_to_dict(cfg)

        # Inject cathode_config as dict with all sub-fields as dicts/strings
        d["cathode_config"] = {
            "material": "graphite_plate",
            "geometry": {
                "geometry_type": "rectangular_plate",
                "length": 0.05,
                "width": 0.05,
                "thickness": 0.005,
            },
            "material_properties": {
                "specific_conductance": 25000,
                "contact_resistance": 0.1,
                "surface_charge_density": -0.05,
                "hydrophobicity_angle": 75,
                "surface_roughness": 1.5,
                "biofilm_adhesion_coefficient": 0.7,
                "attachment_energy": -15.0,
            },
        }

        result = dict_to_dataclass(d, QLearningConfig)
        assert isinstance(result, QLearningConfig)
        assert isinstance(result.cathode_config, ElectrodeConfiguration)
        assert result.cathode_config.material == ElectrodeMaterial.GRAPHITE_PLATE
        assert isinstance(result.cathode_config.geometry, ElectrodeGeometrySpec)
        assert result.cathode_config.geometry.geometry_type == ElectrodeGeometry.RECTANGULAR_PLATE
        assert isinstance(result.cathode_config.material_properties, MaterialProperties)
        assert result.cathode_config.material_properties.specific_conductance == 25000

    def test_cathode_config_material_already_enum(self):
        """Cover when material is already an ElectrodeMaterial."""
        cfg = deepcopy(DEFAULT_QLEARNING_CONFIG)
        d = dataclass_to_dict(cfg)

        d["cathode_config"] = {
            "material": ElectrodeMaterial.CARBON_FELT,
            "geometry": ElectrodeGeometrySpec(
                geometry_type=ElectrodeGeometry.RECTANGULAR_PLATE,
                length=0.03,
                width=0.03,
                thickness=0.01,
            ),
            "material_properties": MaterialProperties(
                specific_conductance=500,
                contact_resistance=0.8,
                surface_charge_density=-0.08,
                hydrophobicity_angle=85,
                surface_roughness=15.0,
                biofilm_adhesion_coefficient=2.5,
                attachment_energy=-18.0,
            ),
        }

        result = dict_to_dataclass(d, QLearningConfig)
        assert isinstance(result.cathode_config, ElectrodeConfiguration)

    def test_cathode_config_without_material_str(self):
        """Cover cathode_config without material being a string."""
        cfg = deepcopy(DEFAULT_QLEARNING_CONFIG)
        d = dataclass_to_dict(cfg)

        # No material key - should still create ElectrodeConfiguration
        d["cathode_config"] = {
            "material": ElectrodeMaterial.PLATINUM,
            "geometry": {
                "geometry_type": "cylindrical_rod",
                "diameter": 0.01,
                "length": 0.05,
            },
            "material_properties": {
                "specific_conductance": 9600000,
                "contact_resistance": 0.01,
                "surface_charge_density": 0.0,
                "hydrophobicity_angle": 40,
                "surface_roughness": 1.0,
                "biofilm_adhesion_coefficient": 0.2,
                "attachment_energy": -3.0,
            },
        }

        result = dict_to_dataclass(d, QLearningConfig)
        assert isinstance(result.cathode_config, ElectrodeConfiguration)

    def test_anode_config_deserialization(self):
        """Also cover anode_config dict branch for completeness."""
        cfg = deepcopy(DEFAULT_QLEARNING_CONFIG)
        d = dataclass_to_dict(cfg)

        d["anode_config"] = {
            "material": "carbon_cloth",
            "geometry": {
                "geometry_type": "rectangular_plate",
                "length": 0.04,
                "width": 0.04,
                "thickness": 0.003,
            },
            "material_properties": {
                "specific_conductance": 800,
                "contact_resistance": 0.6,
                "surface_charge_density": -0.06,
                "hydrophobicity_angle": 80,
                "surface_roughness": 8.0,
                "biofilm_adhesion_coefficient": 2.0,
                "attachment_energy": -15.5,
            },
        }

        result = dict_to_dataclass(d, QLearningConfig)
        assert isinstance(result, QLearningConfig)
        assert isinstance(result.anode_config, ElectrodeConfiguration)
        assert result.anode_config.material == ElectrodeMaterial.CARBON_CLOTH
