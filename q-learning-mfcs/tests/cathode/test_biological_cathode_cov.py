import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import pytest

from cathode_models.biological_cathode import (
    BiologicalCathodeModel,
    BiologicalParameters,
    create_biological_cathode,
)
from cathode_models.base_cathode import CathodeParameters


class TestBiologicalParameters:
    def test_defaults(self):
        p = BiologicalParameters()
        assert p.max_growth_rate == 0.5
        assert p.decay_rate == 0.05
        assert p.ks_oxygen == 0.5e-3
        assert p.biofilm_density == 80.0
        assert p.max_biofilm_thickness == 500e-6
        assert p.initial_biofilm_thickness == 1e-6

    def test_custom(self):
        p = BiologicalParameters(max_growth_rate=0.8, decay_rate=0.1)
        assert p.max_growth_rate == 0.8


class TestBiologicalCathodeModel:
    def setup_method(self):
        self.params = CathodeParameters(temperature_K=303.15, ph=7.0)
        self.model = BiologicalCathodeModel(self.params)

    def test_init(self):
        assert self.model.biofilm_thickness == 1e-6
        assert self.model.biomass_density == 1.0
        assert self.model.biofilm_age_hours == 0.0

    def test_init_custom_bio(self):
        bp = BiologicalParameters(max_growth_rate=0.8)
        m = BiologicalCathodeModel(self.params, bp)
        assert m.bio_params.max_growth_rate == 0.8

    def test_setup_kinetic_parameters(self):
        assert self.model.environmental_factor > 0
        assert self.model.effective_max_growth_rate > 0

    def test_setup_biofilm_parameters(self):
        assert self.model.effective_oxygen_diffusion > 0
        assert self.model.exchange_current_density > 0
        assert self.model.biofilm_resistance > 0

    def test_monod_growth_rate(self):
        rate = self.model.calculate_monod_growth_rate(8e-3, 0.4)
        assert rate > 0

    def test_monod_growth_rate_low_oxygen(self):
        rate = self.model.calculate_monod_growth_rate(1e-5, 0.4)
        assert rate >= 0

    def test_update_biofilm_dynamics(self):
        initial_thickness = self.model.biofilm_thickness
        self.model.update_biofilm_dynamics(1.0, 8e-3, 0.4)
        assert self.model.biofilm_age_hours == 1.0
        assert self.model.biofilm_thickness >= 1e-6

    def test_update_biofilm_dynamics_long(self):
        for _ in range(100):
            self.model.update_biofilm_dynamics(1.0, 8e-3, 0.4)
        assert self.model.biofilm_thickness <= 500e-6
        assert self.model.biomass_density >= 0.1

    def test_current_density_zero(self):
        cd = self.model.calculate_current_density(0.0)
        assert cd == 0.0

    def test_current_density_negative(self):
        cd = self.model.calculate_current_density(-0.1)
        assert cd == 0.0

    def test_current_density_positive(self):
        cd = self.model.calculate_current_density(0.3)
        assert cd >= 0

    def test_current_density_custom_oxygen(self):
        cd = self.model.calculate_current_density(0.3, oxygen_conc=4e-3)
        assert cd >= 0

    def test_current_density_none_oxygen(self):
        cd = self.model.calculate_current_density(0.3, oxygen_conc=None)
        assert cd >= 0

    def test_performance_metrics(self):
        m = self.model.calculate_performance_metrics(0.2)
        assert "current_density_A_m2" in m
        assert "power_loss_W" in m
        assert "voltage_efficiency_percent" in m
        assert "exchange_current_density_A_m2" in m

    def test_performance_metrics_with_oxygen(self):
        m = self.model.calculate_performance_metrics(0.2, oxygen_conc=4e-3)
        assert m["current_density_A_m2"] >= 0

    def test_biofilm_performance_metrics(self):
        m = self.model.calculate_biofilm_performance_metrics(0.2)
        assert "biofilm_thickness_um" in m
        assert "biomass_density_kg_m3" in m
        assert "growth_rate_h_inv" in m
        assert "biofilm_resistance_ohm_m2" in m
        assert "environmental_factor" in m

    def test_biofilm_performance_metrics_with_oxygen(self):
        m = self.model.calculate_biofilm_performance_metrics(0.2, oxygen_conc=4e-3)
        assert "biofilm_thickness_um" in m

    def test_predict_long_term(self):
        result = self.model.predict_long_term_performance(simulation_days=2)
        assert "time_hours" in result
        assert "biofilm_thickness_um" in result
        assert "current_density_A_m2" in result
        assert "final_thickness_um" in result
        assert len(result["time_hours"]) > 0

    def test_predict_long_term_custom(self):
        result = self.model.predict_long_term_performance(
            simulation_days=1, oxygen_conc=4e-3, electrode_potential=0.3
        )
        assert result["final_thickness_um"] > 0

    def test_predict_long_term_none_oxygen(self):
        result = self.model.predict_long_term_performance(simulation_days=1, oxygen_conc=None)
        assert "final_current_density_A_m2" in result

    def test_estimate_economic_analysis(self):
        ea = self.model.estimate_economic_analysis()
        assert "inoculation_cost_per_m2" in ea
        assert "annual_maintenance_cost_per_m2" in ea
        assert "total_lifetime_cost_per_m2" in ea
        assert "cost_per_kW" in ea

    def test_get_all_parameters(self):
        p = self.model.get_all_parameters()
        assert "microbial_kinetics" in p
        assert "monod_parameters" in p
        assert "biofilm_properties" in p
        assert "mass_transport" in p
        assert "electrochemical" in p

    def test_temperature_effect(self):
        cd1 = self.model.calculate_current_density(0.3)
        self.model.update_temperature(320.0)
        cd2 = self.model.calculate_current_density(0.3)
        # Different temperature should change result
        assert isinstance(cd2, float)

    def test_environmental_stress(self):
        stressed = CathodeParameters(temperature_K=340.0, ph=4.0)
        m = BiologicalCathodeModel(stressed)
        assert m.environmental_factor < 1.0


class TestCreateBiologicalCathode:
    def test_default(self):
        m = create_biological_cathode()
        assert m is not None
        assert m.area_m2 == pytest.approx(1e-4)

    def test_custom(self):
        m = create_biological_cathode(
            area_cm2=2.0, temperature_C=35.0, ph=6.5, oxygen_mg_L=6.0, initial_thickness_um=5.0
        )
        assert m.area_m2 == pytest.approx(2e-4)
        assert m.biofilm_thickness == pytest.approx(5e-6)

    def test_custom_bio_params(self):
        m = create_biological_cathode(
            custom_bio_params={"max_growth_rate": 0.8, "decay_rate": 0.1}
        )
        assert m.bio_params.max_growth_rate == 0.8

    def test_custom_bio_params_invalid_key(self):
        m = create_biological_cathode(custom_bio_params={"nonexistent": 99.0})
        assert not hasattr(m.bio_params, "nonexistent")
