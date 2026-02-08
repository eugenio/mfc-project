"""Tests for biofilm_kinetics/substrate_params.py - coverage target 98%+."""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from biofilm_kinetics.substrate_params import SubstrateParameters, SubstrateProperties


class TestSubstrateProperties:
    def test_acetate_properties(self):
        db = SubstrateParameters()
        props = db.get_substrate_properties("acetate")
        assert isinstance(props, SubstrateProperties)
        assert props.molecular_weight == pytest.approx(82.03)
        assert props.electrons_per_mole == 8
        assert props.ph_optimum == 7.0

    def test_lactate_properties(self):
        db = SubstrateParameters()
        props = db.get_substrate_properties("lactate")
        assert isinstance(props, SubstrateProperties)
        assert props.molecular_weight == pytest.approx(112.06)
        assert props.electrons_per_mole == 4
        assert props.ph_optimum == 7.2

    def test_invalid_substrate(self):
        db = SubstrateParameters()
        with pytest.raises(ValueError, match="not recognized"):
            db.get_substrate_properties("glucose")


class TestSubstrateParameters:
    def test_default_substrate(self):
        db = SubstrateParameters()
        assert db.get_default_substrate() == "lactate"

    def test_list_available(self):
        db = SubstrateParameters()
        avail = db.list_available_substrates()
        assert "acetate" in avail
        assert "lactate" in avail

    def test_nernst_potential_acetate(self):
        db = SubstrateParameters()
        E = db.calculate_nernst_potential("acetate", 0.01, ph=7.0)
        assert isinstance(E, float)

    def test_nernst_potential_lactate(self):
        db = SubstrateParameters()
        E = db.calculate_nernst_potential("lactate", 0.01, ph=7.0)
        assert isinstance(E, float)

    def test_nernst_potential_unknown_fallback(self):
        """Test fallback Q=1 branch for unknown substrate."""
        db = SubstrateParameters()
        # Add a temporary fake substrate to test fallback
        db._substrates["fake"] = SubstrateProperties(
            molecular_weight=100.0, density=1.0, diffusivity=1e-9,
            standard_potential=-0.2, gibbs_free_energy=-500.0,
            electrons_per_mole=2, base_consumption_rate=0.1,
            mass_transfer_coeff=1e-5, ph_optimum=7.0, ph_sensitivity=0.1,
        )
        E = db.calculate_nernst_potential("fake", 0.01, ph=7.0)
        assert isinstance(E, float)

    def test_nernst_potential_different_temp(self):
        db = SubstrateParameters()
        E1 = db.calculate_nernst_potential("acetate", 0.01, temperature=298.15)
        E2 = db.calculate_nernst_potential("acetate", 0.01, temperature=310.0)
        assert E1 != E2

    def test_theoretical_current_acetate(self):
        db = SubstrateParameters()
        I = db.calculate_theoretical_current("acetate", 0.5)
        assert I > 0

    def test_theoretical_current_lactate(self):
        db = SubstrateParameters()
        I = db.calculate_theoretical_current("lactate", 0.8)
        assert I > 0

    def test_ph_correction_at_optimum(self):
        db = SubstrateParameters()
        corrected = db.apply_ph_correction("acetate", 1.0, 7.0)
        assert corrected == pytest.approx(1.0, abs=0.01)

    def test_ph_correction_off_optimum(self):
        db = SubstrateParameters()
        corrected = db.apply_ph_correction("acetate", 1.0, 5.0)
        assert corrected < 1.0

    def test_ph_correction_lactate(self):
        db = SubstrateParameters()
        corrected = db.apply_ph_correction("lactate", 1.0, 7.2)
        assert corrected == pytest.approx(1.0, abs=0.01)

    def test_stoichiometric_acetate(self):
        db = SubstrateParameters()
        coeffs = db.get_stoichiometric_coefficients("acetate")
        assert coeffs["electrons"] == 8.0
        assert coeffs["substrate"] == -1.0
        assert coeffs["co2"] == 2.0

    def test_stoichiometric_lactate(self):
        db = SubstrateParameters()
        coeffs = db.get_stoichiometric_coefficients("lactate")
        assert coeffs["electrons"] == 4.0
        assert coeffs["pyruvate"] == 1.0

    def test_stoichiometric_unknown(self):
        db = SubstrateParameters()
        coeffs = db.get_stoichiometric_coefficients("unknown")
        assert coeffs == {}

    def test_mass_balance_acetate(self):
        db = SubstrateParameters()
        eq = db.get_mass_balance_equation("acetate")
        assert "CH_3COO" in eq

    def test_mass_balance_lactate(self):
        db = SubstrateParameters()
        eq = db.get_mass_balance_equation("lactate")
        assert "C_3H_5O_3" in eq

    def test_mass_balance_unknown(self):
        db = SubstrateParameters()
        eq = db.get_mass_balance_equation("unknown")
        assert "Unknown" in eq

    def test_nernst_potential_very_low_concentration(self):
        db = SubstrateParameters()
        E = db.calculate_nernst_potential("acetate", 1e-15, ph=7.0)
        assert isinstance(E, float)
