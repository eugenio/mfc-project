"""Tests for biofilm_kinetics/biofilm_model.py - coverage target 98%+."""
import sys
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from biofilm_kinetics.biofilm_model import BiofilmKineticsModel


class TestBiofilmKineticsModelInit:
    def test_default_init(self):
        m = BiofilmKineticsModel(use_gpu=False)
        assert m.species == "mixed"
        assert m.substrate == "lactate"
        assert m.temperature == 303.0
        assert m.ph == 7.0

    def test_geobacter_init(self):
        m = BiofilmKineticsModel(species="geobacter", use_gpu=False)
        assert m.species == "geobacter"

    def test_shewanella_init(self):
        m = BiofilmKineticsModel(species="shewanella", use_gpu=False)
        assert m.species == "shewanella"

    def test_acetate_substrate(self):
        m = BiofilmKineticsModel(substrate="acetate", use_gpu=False)
        assert m.substrate == "acetate"

    def test_custom_temperature(self):
        m = BiofilmKineticsModel(temperature=310.0, use_gpu=False)
        assert m.temperature == 310.0

    def test_custom_ph(self):
        m = BiofilmKineticsModel(ph=6.0, use_gpu=False)
        assert m.ph == 6.0


class TestResetState:
    def test_reset(self):
        m = BiofilmKineticsModel(use_gpu=False)
        m.biofilm_thickness = 50.0
        m.biomass_density = 10.0
        m.reset_state()
        assert m.biofilm_thickness == pytest.approx(0.1)
        assert m.biomass_density == pytest.approx(0.01)
        assert m.substrate_concentration == pytest.approx(10.0)
        assert m.time == 0.0


class TestNernstMonodGrowthRate:
    def test_positive_growth(self):
        m = BiofilmKineticsModel(use_gpu=False)
        rate = m.calculate_nernst_monod_growth_rate(10.0, -0.2)
        assert rate >= 0.0

    def test_zero_substrate(self):
        m = BiofilmKineticsModel(use_gpu=False)
        rate = m.calculate_nernst_monod_growth_rate(0.0, -0.2)
        assert rate == pytest.approx(0.0, abs=1e-10)

    def test_very_low_potential(self):
        m = BiofilmKineticsModel(use_gpu=False)
        rate = m.calculate_nernst_monod_growth_rate(10.0, -1.0)
        assert rate >= 0.0

    def test_equal_potentials(self):
        """When E_ka == E_an, division by zero protection."""
        m = BiofilmKineticsModel(use_gpu=False)
        m.E_ka = -0.5
        m.E_an = -0.5
        rate = m.calculate_nernst_monod_growth_rate(10.0, -0.2)
        assert rate >= 0.0


class TestStochasticAttachment:
    def test_basic_attachment(self):
        m = BiofilmKineticsModel(use_gpu=False)
        rate = m.calculate_stochastic_attachment(1e12, 1.0)
        assert rate > 0

    def test_high_coverage_reduces_attachment(self):
        m = BiofilmKineticsModel(use_gpu=False)
        m.biofilm_thickness = 0.1
        rate1 = m.calculate_stochastic_attachment(1e12, 1.0)
        m.biofilm_thickness = 90.0
        rate2 = m.calculate_stochastic_attachment(1e12, 1.0)
        assert rate2 < rate1


class TestBiofilmCurrentDensity:
    def test_basic_current(self):
        m = BiofilmKineticsModel(use_gpu=False)
        j = m.calculate_biofilm_current_density(10.0, 5.0)
        assert j > 0

    def test_zero_biomass(self):
        m = BiofilmKineticsModel(use_gpu=False)
        j = m.calculate_biofilm_current_density(10.0, 0.0)
        assert j == pytest.approx(0.0)

    def test_thick_biofilm_resistance(self):
        m = BiofilmKineticsModel(use_gpu=False)
        j_thin = m.calculate_biofilm_current_density(1.0, 10.0)
        j_thick = m.calculate_biofilm_current_density(100.0, 10.0)
        assert j_thick <= j_thin


class TestSubstrateConsumption:
    def test_positive_consumption(self):
        m = BiofilmKineticsModel(use_gpu=False)
        c = m.calculate_substrate_consumption(0.1, 5.0)
        assert c > 0

    def test_zero_growth(self):
        m = BiofilmKineticsModel(use_gpu=False)
        c = m.calculate_substrate_consumption(0.0, 5.0)
        assert c == pytest.approx(0.0)


class TestMixedCultureSynergy:
    def test_non_mixed_no_synergy(self):
        m = BiofilmKineticsModel(species="geobacter", use_gpu=False)
        total = m.calculate_mixed_culture_synergy(5.0, 3.0)
        assert total == pytest.approx(8.0)

    def test_mixed_synergy(self):
        m = BiofilmKineticsModel(species="mixed", use_gpu=False)
        total = m.calculate_mixed_culture_synergy(5.0, 3.0)
        assert total > 0


class TestStepBiofilmDynamics:
    def test_basic_step(self):
        m = BiofilmKineticsModel(use_gpu=False)
        result = m.step_biofilm_dynamics(0.1, -0.2)
        assert "time" in result
        assert "biofilm_thickness" in result
        assert "biomass_density" in result
        assert "current_density" in result
        assert result["time"] == pytest.approx(0.1)

    def test_step_with_supply(self):
        m = BiofilmKineticsModel(use_gpu=False)
        result = m.step_biofilm_dynamics(0.1, -0.2, substrate_supply=5.0)
        assert result["substrate_concentration"] > 0

    def test_multiple_steps(self):
        m = BiofilmKineticsModel(use_gpu=False)
        for _ in range(10):
            result = m.step_biofilm_dynamics(0.1, -0.2, substrate_supply=1.0)
        assert result["time"] == pytest.approx(1.0)


class TestGetModelParameters:
    def test_returns_dict(self):
        m = BiofilmKineticsModel(use_gpu=False)
        params = m.get_model_parameters()
        assert params["species"] == "mixed"
        assert params["substrate"] == "lactate"
        assert "kinetic_params" in params


class TestSetEnvironmentalConditions:
    def test_set_temperature(self):
        m = BiofilmKineticsModel(use_gpu=False)
        m.set_environmental_conditions(temperature=310.0)
        assert m.temperature == 310.0

    def test_set_ph(self):
        m = BiofilmKineticsModel(use_gpu=False)
        m.set_environmental_conditions(ph=6.5)
        assert m.ph == 6.5

    def test_set_both(self):
        m = BiofilmKineticsModel(use_gpu=False)
        m.set_environmental_conditions(temperature=310.0, ph=6.5)
        assert m.temperature == 310.0
        assert m.ph == 6.5


class TestTheoreticalMaxCurrent:
    def test_max_current(self):
        m = BiofilmKineticsModel(use_gpu=False)
        I_max = m.calculate_theoretical_maximum_current()
        assert I_max > 0


class TestMassBalanceEquations:
    def test_returns_dict(self):
        m = BiofilmKineticsModel(use_gpu=False)
        eq = m.get_mass_balance_equations()
        assert "substrate_equation" in eq
        assert "biomass_balance" in eq
        assert "nernst_monod" in eq
