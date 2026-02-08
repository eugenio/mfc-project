"""Coverage boost tests for biofilm_kinetics/species_params.py."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from biofilm_kinetics.species_params import KineticParameters, SpeciesParameters


@pytest.fixture
def sp():
    return SpeciesParameters()


class TestSpeciesParameters:
    def test_get_geobacter(self, sp):
        p = sp.get_parameters("geobacter")
        assert isinstance(p, KineticParameters)
        assert p.mu_max == 0.15
        assert p.K_s == 0.5
        assert p.j_max == pytest.approx(0.39)

    def test_get_shewanella(self, sp):
        p = sp.get_parameters("shewanella")
        assert p.mu_max == 0.12
        assert p.K_s == 1.0

    def test_get_mixed(self, sp):
        p = sp.get_parameters("mixed")
        assert p.mu_max == 0.14
        assert p.j_max == pytest.approx(0.54)

    def test_get_unknown(self, sp):
        with pytest.raises(ValueError, match="not recognized"):
            sp.get_parameters("unknown_species")

    def test_list_available(self, sp):
        species = sp.list_available_species()
        assert "geobacter" in species
        assert "shewanella" in species
        assert "mixed" in species
        assert len(species) == 3


class TestSynergyCoefficient:
    def test_geo_shew(self, sp):
        assert sp.get_synergy_coefficient("geobacter", "shewanella") == 1.38

    def test_shew_geo(self, sp):
        assert sp.get_synergy_coefficient("shewanella", "geobacter") == 1.38

    def test_same_species(self, sp):
        assert sp.get_synergy_coefficient("geobacter", "geobacter") == 1.0

    def test_unknown_pair(self, sp):
        assert sp.get_synergy_coefficient("geobacter", "mixed") == 1.0


class TestTemperatureCompensation:
    def test_ref_temperature(self, sp):
        p = sp.get_parameters("geobacter")
        comp = sp.apply_temperature_compensation(p, p.temp_ref)
        assert comp.mu_max == pytest.approx(p.mu_max, rel=1e-3)
        assert comp.K_s == p.K_s
        assert comp.Y_xs == p.Y_xs

    def test_higher_temperature(self, sp):
        p = sp.get_parameters("geobacter")
        comp = sp.apply_temperature_compensation(p, 310.0)
        assert comp.mu_max > p.mu_max
        assert comp.j_max > p.j_max
        assert comp.diffusion_coeff > p.diffusion_coeff

    def test_lower_temperature(self, sp):
        p = sp.get_parameters("geobacter")
        comp = sp.apply_temperature_compensation(p, 290.0)
        assert comp.mu_max < p.mu_max
        assert comp.j_max < p.j_max

    def test_unchanged_params(self, sp):
        p = sp.get_parameters("shewanella")
        comp = sp.apply_temperature_compensation(p, 310.0)
        assert comp.E_ka == p.E_ka
        assert comp.E_an == p.E_an
        assert comp.biofilm_thickness_max == p.biofilm_thickness_max
        assert comp.activation_energy == p.activation_energy

    def test_attachment_capped(self, sp):
        p = sp.get_parameters("geobacter")
        comp = sp.apply_temperature_compensation(p, 290.0)
        assert comp.attachment_prob <= p.attachment_prob


class TestPHCompensation:
    def test_ref_ph(self, sp):
        p = sp.get_parameters("geobacter")
        comp = sp.apply_ph_compensation(p, 7.0)
        assert comp.E_ka == p.E_ka
        assert comp.mu_max <= p.mu_max

    def test_optimal_ph(self, sp):
        p = sp.get_parameters("mixed")
        comp = sp.apply_ph_compensation(p, 7.1)
        assert comp.mu_max == pytest.approx(p.mu_max, rel=0.01)

    def test_acidic_ph(self, sp):
        p = sp.get_parameters("geobacter")
        comp = sp.apply_ph_compensation(p, 5.0)
        assert comp.mu_max < p.mu_max
        # nernst_factor=-0.059, pH<7 => positive shift => E_ka > p.E_ka
        assert comp.E_ka > p.E_ka

    def test_alkaline_ph(self, sp):
        p = sp.get_parameters("geobacter")
        comp = sp.apply_ph_compensation(p, 9.0)
        assert comp.mu_max < p.mu_max
        # nernst_factor=-0.059, pH>7 => negative shift => E_ka < p.E_ka
        assert comp.E_ka < p.E_ka

    def test_unchanged_params(self, sp):
        p = sp.get_parameters("geobacter")
        comp = sp.apply_ph_compensation(p, 6.0)
        assert comp.K_s == p.K_s
        assert comp.Y_xs == p.Y_xs
        assert comp.sigma_biofilm == p.sigma_biofilm
        assert comp.biofilm_thickness_max == p.biofilm_thickness_max


class TestKineticParameters:
    def test_dataclass(self):
        kp = KineticParameters(
            mu_max=0.1, K_s=0.5, Y_xs=0.08,
            j_max=0.3, sigma_biofilm=1e-4, E_ka=-0.3, E_an=-0.5,
            attachment_prob=0.7, biofilm_thickness_max=50.0,
            diffusion_coeff=1e-9, activation_energy=65.0, temp_ref=303.0,
        )
        assert kp.mu_max == 0.1
        assert kp.temp_ref == 303.0
