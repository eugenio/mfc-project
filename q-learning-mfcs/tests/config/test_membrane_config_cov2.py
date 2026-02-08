"""Coverage boost tests for membrane_config.py."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from config.membrane_config import (
    MEMBRANE_PROPERTIES_DATABASE,
    MembraneConfiguration,
    MembraneMaterial,
    MembraneProperties,
    create_membrane_config,
)


class TestMembraneConfiguration:
    def _make_props(self):
        return MembraneProperties(
            proton_conductivity=0.10,
            ion_exchange_capacity=0.9,
            permselectivity=0.95,
            thickness=175,
            water_uptake=38,
            density=1.98,
            area_resistance=1.5,
            oxygen_permeability=1.8e-12,
            substrate_permeability=5e-14,
        )

    def test_calculate_resistance(self):
        props = self._make_props()
        config = MembraneConfiguration(
            material=MembraneMaterial.NAFION_117,
            properties=props,
            area=0.001,
        )
        resistance = config.calculate_resistance()
        assert resistance > 0

    def test_calculate_proton_flux(self):
        props = self._make_props()
        config = MembraneConfiguration(
            material=MembraneMaterial.NAFION_117,
            properties=props,
            area=0.001,
        )
        flux = config.calculate_proton_flux(100.0)
        assert flux > 0

    def test_estimate_lifetime_factor(self):
        props = self._make_props()
        config = MembraneConfiguration(
            material=MembraneMaterial.NAFION_117,
            properties=props,
            area=0.001,
            operating_temperature=25.0,
            ph_anode=7.0,
            ph_cathode=7.0,
        )
        factor = config.estimate_lifetime_factor(100.0)
        assert 0 < factor <= 1.5

    def test_estimate_lifetime_factor_high_temp(self):
        props = self._make_props()
        config = MembraneConfiguration(
            material=MembraneMaterial.NAFION_117,
            properties=props,
            area=0.001,
            operating_temperature=50.0,
            ph_anode=6.0,
            ph_cathode=8.0,
        )
        factor = config.estimate_lifetime_factor(500.0)
        assert factor > 0

    def test_estimate_lifetime_ph_gradient(self):
        props = self._make_props()
        config = MembraneConfiguration(
            material=MembraneMaterial.NAFION_117,
            properties=props,
            area=0.001,
            ph_anode=5.0,
            ph_cathode=9.0,
        )
        factor = config.estimate_lifetime_factor(100.0)
        assert factor > 0


class TestCreateMembraneConfig:
    def test_create_nafion_117(self):
        config = create_membrane_config(MembraneMaterial.NAFION_117, 0.001)
        assert config.material == MembraneMaterial.NAFION_117

    def test_create_nafion_112(self):
        config = create_membrane_config(MembraneMaterial.NAFION_112, 0.002)
        assert config.material == MembraneMaterial.NAFION_112

    def test_create_ultrex(self):
        config = create_membrane_config(MembraneMaterial.ULTREX_CMI_7000, 0.003)
        assert config.material == MembraneMaterial.ULTREX_CMI_7000

    def test_create_fumasep(self):
        config = create_membrane_config(MembraneMaterial.FUMASEP_FKE, 0.001)
        assert config.material == MembraneMaterial.FUMASEP_FKE

    def test_create_j_cloth(self):
        config = create_membrane_config(MembraneMaterial.J_CLOTH, 0.005)
        assert config.material == MembraneMaterial.J_CLOTH

    def test_create_custom_with_properties(self):
        custom_props = MembraneProperties(
            proton_conductivity=0.05,
            ion_exchange_capacity=1.0,
            permselectivity=0.9,
            thickness=100,
            water_uptake=30,
            density=1.5,
            area_resistance=2.0,
            oxygen_permeability=1e-12,
            substrate_permeability=3e-14,
        )
        config = create_membrane_config(
            MembraneMaterial.CUSTOM, 0.001, custom_properties=custom_props
        )
        assert config.material == MembraneMaterial.CUSTOM

    def test_create_custom_without_properties_raises(self):
        with pytest.raises(ValueError):
            create_membrane_config(MembraneMaterial.CUSTOM, 0.001)


class TestMembranePropertiesDatabase:
    def test_database_has_entries(self):
        assert len(MEMBRANE_PROPERTIES_DATABASE) >= 5

    def test_nafion_117_properties(self):
        props = MEMBRANE_PROPERTIES_DATABASE[MembraneMaterial.NAFION_117]
        assert props.proton_conductivity == 0.10
        assert props.thickness == 175

    def test_all_properties_have_reference(self):
        for material, props in MEMBRANE_PROPERTIES_DATABASE.items():
            assert props.reference is not None
