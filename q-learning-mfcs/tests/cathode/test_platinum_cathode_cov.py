import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import pytest

from cathode_models.platinum_cathode import (
    PlatinumCathodeModel,
    PlatinumParameters,
    create_platinum_cathode,
)
from cathode_models.base_cathode import CathodeParameters


class TestPlatinumParameters:
    def test_defaults(self):
        p = PlatinumParameters()
        assert p.exchange_current_density_ref == 3.0e-5
        assert p.transfer_coefficient == 0.5
        assert p.tafel_slope_low == 0.060
        assert p.tafel_slope_high == 0.120
        assert p.overpotential_transition == 0.100
        assert p.platinum_loading == 0.5
        assert p.activation_energy == 20000.0
        assert p.porosity == 0.4
        assert p.tortuosity == 2.5

    def test_custom(self):
        p = PlatinumParameters(platinum_loading=1.0, transfer_coefficient=0.6)
        assert p.platinum_loading == 1.0
        assert p.transfer_coefficient == 0.6


class TestPlatinumCathodeModel:
    def setup_method(self):
        self.params = CathodeParameters()
        self.model = PlatinumCathodeModel(self.params)

    def test_init_default_pt_params(self):
        assert self.model.pt_params is not None
        assert self.model.exchange_current_density > 0

    def test_init_custom_pt_params(self):
        pt = PlatinumParameters(platinum_loading=1.0)
        m = PlatinumCathodeModel(self.params, pt)
        assert m.pt_params.platinum_loading == 1.0

    def test_setup_kinetic_parameters(self):
        assert self.model.exchange_current_density > 0
        assert self.model.oxygen_diffusion_coeff > 0
        assert self.model.transfer_coefficient == 0.5

    def test_setup_mass_transport(self):
        assert self.model.limiting_current_density > 0

    def test_current_density_zero_overpotential(self):
        cd = self.model.calculate_current_density(0.0)
        assert cd == 0.0

    def test_current_density_negative_overpotential(self):
        cd = self.model.calculate_current_density(-0.1)
        assert cd == 0.0

    def test_current_density_low_overpotential(self):
        cd = self.model.calculate_current_density(0.05)
        assert cd > 0

    def test_current_density_high_overpotential(self):
        cd = self.model.calculate_current_density(0.5)
        assert cd > 0

    def test_current_density_custom_oxygen(self):
        cd = self.model.calculate_current_density(0.3, oxygen_conc=4e-3)
        assert cd > 0

    def test_current_density_mass_transport_limited(self):
        cd = self.model.calculate_current_density(2.0)
        assert cd > 0

    def test_performance_metrics(self):
        m = self.model.calculate_performance_metrics(0.2)
        assert "current_density_A_m2" in m
        assert "current_A" in m
        assert "power_loss_W" in m
        assert "power_density_mW_m2" in m
        assert "voltage_efficiency_percent" in m
        assert "kinetic_limited" in m
        assert "mass_transport_utilization_percent" in m
        assert "platinum_utilization_percent" in m
        assert isinstance(m["kinetic_limited"], bool)

    def test_performance_metrics_with_oxygen(self):
        m = self.model.calculate_performance_metrics(0.2, oxygen_conc=4e-3)
        assert m["current_density_A_m2"] >= 0

    def test_estimate_cost_per_area(self):
        cost = self.model.estimate_cost_per_area()
        assert cost > 0

    def test_estimate_cost_analysis(self):
        ca = self.model.estimate_cost_analysis()
        assert "material_cost_per_m2" in ca
        assert "total_cost_per_m2" in ca
        assert "cost_per_kW" in ca
        assert ca["material_cost_per_m2"] > 0

    def test_compare_to_benchmark(self):
        b = self.model.compare_to_benchmark()
        assert "performance_vs_benchmark" in b
        assert "operating_conditions" in b
        assert "cost_analysis" in b
        assert b["performance_vs_benchmark"]["power_density_ratio"] > 0

    def test_get_all_parameters(self):
        p = self.model.get_all_parameters()
        assert "cathode_parameters" in p
        assert "platinum_kinetic_parameters" in p
        assert "catalyst_parameters" in p
        assert "mass_transport_parameters" in p
        assert "economic_parameters" in p
        assert "benchmark_parameters" in p

    def test_temperature_effect(self):
        cd1 = self.model.calculate_current_density(0.3)
        self.model.update_temperature(320.0)
        cd2 = self.model.calculate_current_density(0.3)
        assert cd1 != cd2


class TestCreatePlatinumCathode:
    def test_default(self):
        m = create_platinum_cathode()
        assert m is not None
        assert m.area_m2 == pytest.approx(1e-4)

    def test_custom_params(self):
        m = create_platinum_cathode(
            area_cm2=2.0, temperature_C=30.0, oxygen_mg_L=6.0, platinum_loading_mg_cm2=1.0
        )
        assert m.area_m2 == pytest.approx(2e-4)

    def test_custom_pt_params(self):
        m = create_platinum_cathode(
            custom_pt_params={"transfer_coefficient": 0.6, "porosity": 0.5}
        )
        assert m.pt_params.transfer_coefficient == 0.6
        assert m.pt_params.porosity == 0.5

    def test_custom_pt_params_invalid_key(self):
        m = create_platinum_cathode(custom_pt_params={"nonexistent_key": 99.0})
        assert not hasattr(m.pt_params, "nonexistent_key")
