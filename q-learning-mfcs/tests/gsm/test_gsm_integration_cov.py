"""Tests for gsm_integration module - targeting 98%+ coverage."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest
import numpy as np

from gsm.gsm_integration import (
    MetabolicReaction,
    Metabolite,
    GSMModelConfig,
    ShewanellaGSMModel,
    GSMPhysicsIntegrator,
)


class TestMetabolicReaction:
    def test_dataclass_defaults(self):
        rxn = MetabolicReaction(
            id="r1", name="Test", stoichiometry={"a": -1, "b": 1}
        )
        assert rxn.id == "r1"
        assert rxn.lower_bound == 0.0
        assert rxn.upper_bound == 1000.0
        assert rxn.objective_coefficient == 0.0
        assert rxn.subsystem == ""
        assert rxn.ec_number == ""
        assert rxn.gene_reaction_rule == ""

    def test_custom_bounds(self):
        rxn = MetabolicReaction(
            id="r2", name="T", stoichiometry={},
            lower_bound=-5.0, upper_bound=10.0,
            objective_coefficient=1.0, subsystem="ETC",
            ec_number="1.1.1.1", gene_reaction_rule="geneA and geneB",
        )
        assert rxn.lower_bound == -5.0
        assert rxn.upper_bound == 10.0
        assert rxn.objective_coefficient == 1.0
        assert rxn.subsystem == "ETC"


class TestMetabolite:
    def test_defaults(self):
        met = Metabolite(id="m1", name="Test met")
        assert met.formula == ""
        assert met.charge == 0
        assert met.compartment == "c"
        assert met.boundary is False

    def test_custom(self):
        met = Metabolite(
            id="m2", name="Ext", formula="CO2",
            charge=-1, compartment="e", boundary=True,
        )
        assert met.boundary is True
        assert met.compartment == "e"


class TestGSMModelConfig:
    def test_defaults(self):
        cfg = GSMModelConfig()
        assert cfg.organism == "Shewanella oneidensis MR-1"
        assert cfg.model_id == "iSO783"
        assert cfg.max_growth_rate == 0.085
        assert cfg.maintenance_atp == 1.03
        assert cfg.growth_atp == 220.22
        assert cfg.max_lactate_uptake == 4.11
        assert cfg.max_oxygen_uptake == 10.0
        assert cfg.max_riboflavin_export == 0.01
        assert cfg.flavin_transfer_efficiency == 0.7
        assert cfg.direct_transfer_efficiency == 0.3
        assert cfg.temperature == 30.0
        assert cfg.ph == 7.0
        assert cfg.electrode_potential == 0.2

    def test_custom_config(self):
        cfg = GSMModelConfig(max_growth_rate=0.1, temperature=37.0)
        assert cfg.max_growth_rate == 0.1
        assert cfg.temperature == 37.0


class TestShewanellaGSMModel:
    def test_init_default(self):
        model = ShewanellaGSMModel()
        assert model.config.organism == "Shewanella oneidensis MR-1"
        assert len(model.metabolites) > 0
        assert len(model.reactions) > 0
        assert model.current_growth_rate == 0.0
        assert model.current_electron_production == 0.0
        assert model.current_fluxes == {}

    def test_init_custom_config(self):
        cfg = GSMModelConfig(max_oxygen_uptake=0.0)
        model = ShewanellaGSMModel(cfg)
        assert model.config.max_oxygen_uptake == 0.0

    def test_build_metabolic_network_metabolites(self):
        model = ShewanellaGSMModel()
        expected_mets = [
            "lactate_ext", "oxygen_ext", "co2_ext", "acetate_ext",
            "riboflavin_ext", "lactate_c", "pyruvate_c", "acetyl_coa_c",
            "acetate_c", "co2_c", "atp_c", "adp_c", "nad_c", "nadh_c",
            "quinone_c", "quinol_c", "cytc_ox_p", "cytc_red_p",
            "riboflavin_c", "riboflavin_red_c", "biomass_c",
        ]
        for met_id in expected_mets:
            assert met_id in model.metabolites

    def test_build_metabolic_network_reactions(self):
        model = ShewanellaGSMModel()
        expected_rxns = [
            "lactate_uptake", "oxygen_uptake", "lactate_dehydrogenase",
            "pyruvate_dehydrogenase", "acetyl_coa_hydrolysis",
            "nadh_quinone_oxidoreductase", "bc1_complex", "cytochrome_oxidase",
            "riboflavin_synthesis", "riboflavin_reduction", "riboflavin_export",
            "electrode_electron_transfer", "co2_export", "acetate_export",
            "atp_maintenance", "biomass_reaction",
        ]
        for rxn_id in expected_rxns:
            assert rxn_id in model.reactions

    def test_solve_fba_aerobic(self):
        model = ShewanellaGSMModel()
        fluxes = model.solve_fba()
        assert "lactate_uptake" in fluxes
        assert fluxes["lactate_uptake"] > 0
        assert "electrode_electron_transfer" in fluxes
        assert fluxes["electrode_electron_transfer"] >= 0
        assert model.current_fluxes is fluxes
        assert model.current_growth_rate == fluxes.get("biomass_reaction", 0)
        assert model.current_electron_production == fluxes.get(
            "electrode_electron_transfer", 0
        )

    def test_solve_fba_anaerobic(self):
        cfg = GSMModelConfig(max_oxygen_uptake=0.0)
        model = ShewanellaGSMModel(cfg)
        fluxes = model.solve_fba()
        assert fluxes["cytochrome_oxidase"] == 0
        assert fluxes["oxygen_uptake"] == 0
        # No oxygen means no ATP from respiration, growth should be 0
        assert fluxes["biomass_reaction"] == 0

    def test_solve_fba_custom_objective(self):
        model = ShewanellaGSMModel()
        fluxes = model.solve_fba("biomass_reaction")
        assert len(fluxes) > 0

    def test_calculate_metabolic_objectives(self):
        model = ShewanellaGSMModel()
        objectives = model.calculate_metabolic_objectives()
        expected_keys = [
            "maximize_electron_production",
            "maximize_growth_rate",
            "maximize_substrate_utilization",
            "minimize_substrate_waste",
            "maximize_energy_efficiency",
            "maximize_flavin_utilization",
            "metabolic_burden",
            "electron_transfer_efficiency",
            "cofactor_balance",
        ]
        for key in expected_keys:
            assert key in objectives

    def test_update_environmental_conditions_basic(self):
        model = ShewanellaGSMModel()
        model.update_environmental_conditions(
            substrate_concentration=20.0,
            oxygen_availability=0.5,
            electrode_potential=0.3,
        )
        assert model.config.max_oxygen_uptake == 5.0
        assert model.config.electrode_potential == 0.3
        assert model.config.flavin_transfer_efficiency == min(0.9, 0.5 + 0.3)

    def test_update_environmental_conditions_low_potential(self):
        model = ShewanellaGSMModel()
        model.update_environmental_conditions(
            substrate_concentration=10.0,
            oxygen_availability=0.0,
            electrode_potential=-0.1,
        )
        assert model.config.flavin_transfer_efficiency == max(0.1, 0.5 + (-0.1))

    def test_update_environmental_conditions_with_temp_and_ph(self):
        model = ShewanellaGSMModel()
        model.update_environmental_conditions(
            substrate_concentration=25.0,
            oxygen_availability=1.0,
            electrode_potential=0.2,
            temperature=37.0,
            ph=6.5,
        )
        assert model.config.temperature == 37.0
        assert model.config.ph == 6.5

    def test_update_environmental_conditions_no_temp_ph(self):
        model = ShewanellaGSMModel()
        original_temp = model.config.temperature
        original_ph = model.config.ph
        model.update_environmental_conditions(
            substrate_concentration=25.0,
            oxygen_availability=1.0,
            electrode_potential=0.2,
        )
        assert model.config.temperature == original_temp
        assert model.config.ph == original_ph

    def test_get_metabolic_summary(self):
        model = ShewanellaGSMModel()
        summary = model.get_metabolic_summary()
        assert summary["organism"] == "Shewanella oneidensis MR-1"
        assert summary["model_id"] == "iSO783"
        assert "current_fluxes" in summary
        assert "objectives" in summary
        assert "environmental_conditions" in summary
        assert "performance_metrics" in summary
        assert "electron_production_rate" in summary["performance_metrics"]


class TestGSMPhysicsIntegrator:
    def test_init(self):
        model = ShewanellaGSMModel()
        integrator = GSMPhysicsIntegrator(model)
        assert integrator.gsm_model is model
        assert integrator.integration_history == []

    def test_integrate_with_electrode_model(self):
        model = ShewanellaGSMModel()
        integrator = GSMPhysicsIntegrator(model)

        electrode_results = {
            "performance_metrics": {
                "avg_substrate_mM": 20.0,
                "avg_current_density_A_m2": 0.5,
                "avg_biofilm_density_kg_m3": 15.0,
            }
        }

        objectives = integrator.integrate_with_electrode_model(electrode_results)
        assert "maximize_bioelectrochemical_performance" in objectives
        assert "maximize_integrated_substrate_efficiency" in objectives
        assert "maximize_biofilm_metabolic_activity" in objectives
        assert "maximize_integrated_energy_efficiency" in objectives
        assert "maximize_system_stability" in objectives
        assert "minimize_integrated_losses" in objectives
        assert len(integrator.integration_history) == 1

    def test_integrate_missing_performance_metrics(self):
        model = ShewanellaGSMModel()
        integrator = GSMPhysicsIntegrator(model)
        objectives = integrator.integrate_with_electrode_model({})
        assert isinstance(objectives, dict)
        assert len(objectives) > 0

    def test_get_optimization_targets_gsm_no_history(self):
        model = ShewanellaGSMModel()
        integrator = GSMPhysicsIntegrator(model)
        targets = integrator.get_optimization_targets_gsm()
        # Should run default integration
        assert len(integrator.integration_history) == 1
        assert isinstance(targets, dict)

    def test_get_optimization_targets_gsm_with_history(self):
        model = ShewanellaGSMModel()
        integrator = GSMPhysicsIntegrator(model)
        # First integration
        integrator.integrate_with_electrode_model({
            "performance_metrics": {
                "avg_substrate_mM": 10.0,
                "avg_current_density_A_m2": 1.0,
                "avg_biofilm_density_kg_m3": 20.0,
            }
        })
        targets = integrator.get_optimization_targets_gsm()
        assert len(integrator.integration_history) == 1
        assert isinstance(targets, dict)

    def test_multiple_integrations(self):
        model = ShewanellaGSMModel()
        integrator = GSMPhysicsIntegrator(model)
        for i in range(3):
            integrator.integrate_with_electrode_model({
                "performance_metrics": {
                    "avg_substrate_mM": 20.0 + i,
                    "avg_current_density_A_m2": 0.5,
                    "avg_biofilm_density_kg_m3": 10.0,
                }
            })
        assert len(integrator.integration_history) == 3
        # Check integration_history structure
        entry = integrator.integration_history[-1]
        assert "metabolic_objectives" in entry
        assert "physics_objectives" in entry
        assert "integrated_objectives" in entry
        assert "gsm_summary" in entry
