"""Tests for config/electrode_config.py - coverage target 98%+."""
import sys
import os
import math

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from config.electrode_config import (
    ElectrodeConfiguration,
    ElectrodeGeometry,
    ElectrodeGeometrySpec,
    ElectrodeMaterial,
    MaterialProperties,
    MATERIAL_PROPERTIES_DATABASE,
    create_electrode_config,
    DEFAULT_GRAPHITE_PLATE_CONFIG,
    DEFAULT_CARBON_FELT_CONFIG,
)


class TestElectrodeGeometrySpec:
    def test_rectangular_specific_area(self):
        g = ElectrodeGeometrySpec(
            geometry_type=ElectrodeGeometry.RECTANGULAR_PLATE,
            length=0.05, width=0.05, thickness=0.005,
        )
        area = g.calculate_specific_surface_area()
        assert area == pytest.approx(0.05 * 0.05)

    def test_cylindrical_rod_specific_area(self):
        g = ElectrodeGeometrySpec(
            geometry_type=ElectrodeGeometry.CYLINDRICAL_ROD,
            diameter=0.01, length=0.05,
        )
        area = g.calculate_specific_surface_area()
        assert area == pytest.approx(math.pi * (0.005) ** 2)

    def test_cylindrical_tube_specific_area(self):
        g = ElectrodeGeometrySpec(
            geometry_type=ElectrodeGeometry.CYLINDRICAL_TUBE,
            diameter=0.02, length=0.05, thickness=0.002,
        )
        area = g.calculate_specific_surface_area()
        assert area == pytest.approx(math.pi * (0.01) ** 2)

    def test_spherical_specific_area(self):
        g = ElectrodeGeometrySpec(
            geometry_type=ElectrodeGeometry.SPHERICAL,
            diameter=0.02,
        )
        area = g.calculate_specific_surface_area()
        assert area == pytest.approx(math.pi * (0.01) ** 2)

    def test_custom_specific_area(self):
        g = ElectrodeGeometrySpec(
            geometry_type=ElectrodeGeometry.CUSTOM,
            specific_surface_area=0.123,
        )
        assert g.calculate_specific_surface_area() == pytest.approx(0.123)

    def test_insufficient_dimensions(self):
        g = ElectrodeGeometrySpec(
            geometry_type=ElectrodeGeometry.RECTANGULAR_PLATE,
        )
        with pytest.raises(ValueError, match="Insufficient"):
            g.calculate_specific_surface_area()

    def test_rectangular_total_area(self):
        g = ElectrodeGeometrySpec(
            geometry_type=ElectrodeGeometry.RECTANGULAR_PLATE,
            length=0.05, width=0.05, thickness=0.005,
        )
        total = g.calculate_total_surface_area()
        expected = 2 * (0.05 * 0.05 + 0.05 * 0.005 + 0.05 * 0.005)
        assert total == pytest.approx(expected)

    def test_cylindrical_rod_total_area(self):
        g = ElectrodeGeometrySpec(
            geometry_type=ElectrodeGeometry.CYLINDRICAL_ROD,
            diameter=0.01, length=0.05,
        )
        total = g.calculate_total_surface_area()
        expected = math.pi * 0.01 * 0.05 + 2 * math.pi * (0.005) ** 2
        assert total == pytest.approx(expected)

    def test_cylindrical_tube_total_area(self):
        g = ElectrodeGeometrySpec(
            geometry_type=ElectrodeGeometry.CYLINDRICAL_TUBE,
            diameter=0.02, length=0.05, thickness=0.002,
        )
        total = g.calculate_total_surface_area()
        assert total > 0

    def test_spherical_total_area(self):
        g = ElectrodeGeometrySpec(
            geometry_type=ElectrodeGeometry.SPHERICAL,
            diameter=0.02,
        )
        total = g.calculate_total_surface_area()
        assert total == pytest.approx(4 * math.pi * (0.01) ** 2)

    def test_total_area_provided(self):
        g = ElectrodeGeometrySpec(
            geometry_type=ElectrodeGeometry.CUSTOM,
            total_surface_area=0.5,
        )
        assert g.calculate_total_surface_area() == pytest.approx(0.5)

    def test_total_area_insufficient(self):
        g = ElectrodeGeometrySpec(
            geometry_type=ElectrodeGeometry.RECTANGULAR_PLATE,
        )
        with pytest.raises(ValueError, match="Insufficient"):
            g.calculate_total_surface_area()

    def test_rectangular_volume(self):
        g = ElectrodeGeometrySpec(
            geometry_type=ElectrodeGeometry.RECTANGULAR_PLATE,
            length=0.05, width=0.05, thickness=0.005,
        )
        vol = g.calculate_volume()
        assert vol == pytest.approx(0.05 * 0.05 * 0.005)

    def test_cylindrical_rod_volume(self):
        g = ElectrodeGeometrySpec(
            geometry_type=ElectrodeGeometry.CYLINDRICAL_ROD,
            diameter=0.01, length=0.05,
        )
        vol = g.calculate_volume()
        assert vol == pytest.approx(math.pi * (0.005) ** 2 * 0.05)

    def test_cylindrical_tube_volume(self):
        g = ElectrodeGeometrySpec(
            geometry_type=ElectrodeGeometry.CYLINDRICAL_TUBE,
            diameter=0.02, length=0.05, thickness=0.002,
        )
        vol = g.calculate_volume()
        assert vol > 0

    def test_spherical_volume(self):
        g = ElectrodeGeometrySpec(
            geometry_type=ElectrodeGeometry.SPHERICAL,
            diameter=0.02,
        )
        vol = g.calculate_volume()
        assert vol == pytest.approx((4 / 3) * math.pi * (0.01) ** 3)

    def test_volume_insufficient(self):
        g = ElectrodeGeometrySpec(
            geometry_type=ElectrodeGeometry.RECTANGULAR_PLATE,
        )
        with pytest.raises(ValueError, match="Insufficient"):
            g.calculate_volume()

    def test_calculate_mass(self):
        g = ElectrodeGeometrySpec(
            geometry_type=ElectrodeGeometry.RECTANGULAR_PLATE,
            length=0.05, width=0.05, thickness=0.005,
            density=2000.0,
        )
        mass = g.calculate_mass()
        assert mass == pytest.approx(0.05 * 0.05 * 0.005 * 2000.0)

    def test_calculate_mass_no_density(self):
        g = ElectrodeGeometrySpec(
            geometry_type=ElectrodeGeometry.RECTANGULAR_PLATE,
            length=0.05, width=0.05, thickness=0.005,
        )
        with pytest.raises(ValueError, match="Density"):
            g.calculate_mass()


class TestElectrodeConfiguration:
    def _make_config(self, porous=False):
        props = MaterialProperties(
            specific_conductance=25000, contact_resistance=0.1,
            surface_charge_density=-0.05, hydrophobicity_angle=75,
            surface_roughness=1.2, biofilm_adhesion_coefficient=1.0,
            attachment_energy=-12.5,
            specific_surface_area=1500 if porous else None,
            porosity=0.95 if porous else None,
        )
        geom = ElectrodeGeometrySpec(
            geometry_type=ElectrodeGeometry.RECTANGULAR_PLATE,
            length=0.05, width=0.05, thickness=0.005,
        )
        return ElectrodeConfiguration(
            material=ElectrodeMaterial.GRAPHITE_PLATE,
            geometry=geom,
            material_properties=props,
        )

    def test_effective_area_non_porous(self):
        cfg = self._make_config(porous=False)
        area = cfg.calculate_effective_surface_area()
        assert area > 0

    def test_biofilm_capacity(self):
        cfg = self._make_config()
        cap = cfg.calculate_biofilm_capacity()
        assert cap > 0

    def test_charge_transfer_coefficient(self):
        cfg = self._make_config()
        coeff = cfg.calculate_charge_transfer_coefficient()
        assert 0 < coeff <= 1.0


class TestCreateElectrodeConfig:
    def test_graphite_plate(self):
        cfg = create_electrode_config(
            ElectrodeMaterial.GRAPHITE_PLATE,
            ElectrodeGeometry.RECTANGULAR_PLATE,
            {"length": 0.05, "width": 0.05, "thickness": 0.005},
        )
        assert cfg.material == ElectrodeMaterial.GRAPHITE_PLATE

    def test_custom_properties(self):
        custom = MaterialProperties(
            specific_conductance=100, contact_resistance=1.0,
            surface_charge_density=0.0, hydrophobicity_angle=60,
            surface_roughness=1.0, biofilm_adhesion_coefficient=0.5,
            attachment_energy=-5.0,
        )
        cfg = create_electrode_config(
            ElectrodeMaterial.CUSTOM,
            ElectrodeGeometry.RECTANGULAR_PLATE,
            {"length": 0.05, "width": 0.05, "thickness": 0.005},
            custom_properties=custom,
        )
        assert cfg.material_properties.specific_conductance == 100

    def test_unknown_material_no_props(self):
        with pytest.raises(ValueError, match="No predefined"):
            create_electrode_config(
                ElectrodeMaterial.GOLD,
                ElectrodeGeometry.RECTANGULAR_PLATE,
                {"length": 0.05, "width": 0.05, "thickness": 0.005},
            )


class TestDefaultConfigs:
    def test_graphite_plate_exists(self):
        assert DEFAULT_GRAPHITE_PLATE_CONFIG is not None

    def test_carbon_felt_exists(self):
        assert DEFAULT_CARBON_FELT_CONFIG is not None


class TestMaterialDatabase:
    def test_all_materials_in_db(self):
        expected = [
            ElectrodeMaterial.GRAPHITE_PLATE,
            ElectrodeMaterial.GRAPHITE_ROD,
            ElectrodeMaterial.CARBON_FELT,
            ElectrodeMaterial.CARBON_CLOTH,
            ElectrodeMaterial.CARBON_PAPER,
            ElectrodeMaterial.STAINLESS_STEEL,
            ElectrodeMaterial.PLATINUM,
        ]
        for mat in expected:
            assert mat in MATERIAL_PROPERTIES_DATABASE
