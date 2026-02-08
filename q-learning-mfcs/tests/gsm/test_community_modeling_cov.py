"""Tests for community_modeling module - targeting 98%+ coverage.

cobra is mocked since it is not installed.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

# Mock cobra before importing
mock_cobra = MagicMock()
sys.modules["cobra"] = mock_cobra
sys.modules["cobra.flux_analysis"] = MagicMock()
sys.modules["cobra.flux_analysis.parsimonious"] = MagicMock()
sys.modules["cobra.util"] = MagicMock()
sys.modules["cobra.util.solver"] = MagicMock()
sys.modules["mackinac"] = MagicMock()

from gsm.community_modeling import (
    OrganismAbundance,
    CommunityInteraction,
    CommunityState,
    MFCCommunityModel,
    CommunityElectrodeIntegration,
)


class TestOrganismAbundance:
    def test_update_abundance(self):
        ab = OrganismAbundance(
            organism_id="org1",
            initial_abundance=0.5,
            current_abundance=0.5,
            growth_rate=0.1,
        )
        ab.update_abundance(dt=1.0)
        expected = 0.5 * np.exp(0.1 * 1.0)
        assert abs(ab.current_abundance - expected) < 1e-6

    def test_update_abundance_zero_growth(self):
        ab = OrganismAbundance(
            organism_id="org1",
            initial_abundance=0.5,
            current_abundance=0.5,
            growth_rate=0.0,
        )
        ab.update_abundance(dt=1.0)
        assert abs(ab.current_abundance - 0.5) < 1e-6


class TestCommunityInteraction:
    def test_dataclass_fields(self):
        interaction = CommunityInteraction(
            producer_id="p1",
            consumer_id="c1",
            metabolite_id="lactate",
            interaction_type="cross-feeding",
            strength=0.8,
        )
        assert interaction.producer_id == "p1"
        assert interaction.consumer_id == "c1"
        assert interaction.strength == 0.8


class TestCommunityState:
    def test_to_dataframe(self):
        state = CommunityState(
            time=1.0,
            abundances={"org1": 0.5, "org2": 0.3},
            metabolite_concentrations={"lactate": 10.0, "acetate": 5.0, "co2": 3.0},
            community_growth_rate=0.2,
            electron_production_rate=0.1,
        )
        df = state.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "time" in df.columns
        assert "community_growth" in df.columns
        assert "org1_abundance" in df.columns
        assert "lactate_mM" in df.columns
        assert "acetate_mM" in df.columns
        # co2 should not be included (not in lactate/acetate/formate list)
        assert "co2_mM" not in df.columns

    def test_to_dataframe_formate(self):
        state = CommunityState(
            time=0.0,
            abundances={},
            metabolite_concentrations={"formate": 2.0},
            community_growth_rate=0.0,
            electron_production_rate=0.0,
        )
        df = state.to_dataframe()
        assert "formate_mM" in df.columns


class TestMFCCommunityModel:
    def test_init(self):
        model = MFCCommunityModel(electrode_area=0.02)
        assert model.electrode_area == 0.02
        assert model.organisms == {}
        assert model.abundances == {}
        assert model.interactions == []
        assert model.time == 0.0

    def test_add_organism(self):
        model = MFCCommunityModel()
        mock_wrapper = MagicMock()
        model.add_organism("shewanella", mock_wrapper, 0.3)
        assert "shewanella" in model.organisms
        assert model.abundances["shewanella"].initial_abundance == 0.3

    def test_add_interaction(self):
        model = MFCCommunityModel()
        model.add_interaction("p", "c", "lactate", "cross-feeding", 0.5)
        assert len(model.interactions) == 1
        assert model.interactions[0].producer_id == "p"

    def test_setup_mfc_community(self):
        model = MFCCommunityModel()
        model.setup_mfc_community()
        assert len(model.interactions) == 3
        assert "lactate" in model.shared_metabolites
        assert "riboflavin" in model.shared_metabolites

    def test_simulate_step_no_organisms(self):
        model = MFCCommunityModel()
        model.setup_mfc_community()
        state = model.simulate_step(dt=0.1)
        assert isinstance(state, CommunityState)
        assert state.time == pytest.approx(0.1)
        assert state.community_growth_rate == 0.0

    def test_simulate_step_with_organism_optimal(self):
        model = MFCCommunityModel()
        model.setup_mfc_community()

        mock_wrapper = MagicMock()
        fba_result = MagicMock()
        fba_result.status = "optimal"
        fba_result.objective_value = 0.5
        fba_result.fluxes = {"EX_lactate_e": -2.0, "cytochrome_c": 1.0}
        mock_wrapper.optimize.return_value = fba_result

        model.add_organism("shewanella_test", mock_wrapper, 0.4)
        state = model.simulate_step(dt=0.1)
        assert state.community_growth_rate > 0

    def test_simulate_step_organism_infeasible(self):
        model = MFCCommunityModel()
        model.setup_mfc_community()

        mock_wrapper = MagicMock()
        fba_result = MagicMock()
        fba_result.status = "infeasible"
        fba_result.objective_value = 0.0
        mock_wrapper.optimize.return_value = fba_result

        model.add_organism("org1", mock_wrapper, 0.3)
        state = model.simulate_step(dt=0.1)
        assert isinstance(state, CommunityState)

    def test_simulate_step_fba_exception(self):
        model = MFCCommunityModel()
        model.setup_mfc_community()

        mock_wrapper = MagicMock()
        mock_wrapper.optimize.side_effect = Exception("FBA error")

        model.add_organism("org1", mock_wrapper, 0.3)
        state = model.simulate_step(dt=0.1)
        assert isinstance(state, CommunityState)

    def test_simulate_step_organism_low_abundance(self):
        model = MFCCommunityModel()
        model.setup_mfc_community()

        mock_wrapper = MagicMock()
        model.add_organism("org1", mock_wrapper, 1e-8)
        state = model.simulate_step(dt=0.1)
        assert isinstance(state, CommunityState)

    def test_simulate_step_normalization(self):
        model = MFCCommunityModel()
        model.setup_mfc_community()

        # Add organisms with high initial abundances that will exceed 1.0
        for i in range(5):
            mock_wrapper = MagicMock()
            fba_result = MagicMock()
            fba_result.status = "optimal"
            fba_result.objective_value = 2.0
            fba_result.fluxes = {"r1": 1.0}
            mock_wrapper.optimize.return_value = fba_result
            model.add_organism(f"org{i}", mock_wrapper, 0.5)

        state = model.simulate_step(dt=0.1)
        total = sum(state.abundances.values())
        assert total <= 1.0 + 1e-6

    def test_prepare_media_for_organism_shewanella(self):
        model = MFCCommunityModel()
        model.setup_mfc_community()
        media = model._prepare_media_for_organism("shewanella_test")
        assert isinstance(media, dict)
        if "lactate" in media:
            # Shewanella has a 1.5x multiplier on lactate
            assert media["lactate"] < 0

    def test_prepare_media_for_organism_geobacter(self):
        model = MFCCommunityModel()
        model.setup_mfc_community()
        media = model._prepare_media_for_organism("geobacter_test")
        assert isinstance(media, dict)

    def test_prepare_media_for_organism_generic(self):
        model = MFCCommunityModel()
        model.setup_mfc_community()
        media = model._prepare_media_for_organism("generic_organism")
        assert isinstance(media, dict)

    def test_prepare_media_low_concentration(self):
        model = MFCCommunityModel()
        model.shared_metabolites = {"lactate": 0.05}  # Below 0.1 threshold
        media = model._prepare_media_for_organism("org1")
        assert "lactate" not in media

    def test_update_metabolite_pool(self):
        model = MFCCommunityModel()
        model.setup_mfc_community()
        fluxes = {
            "org1": {"EX_lactate_e": -5.0, "EX_acetate_e": 2.0, "r_internal": 1.0}
        }
        initial_lactate = model.shared_metabolites["lactate"]
        model._update_metabolite_pool(fluxes, dt=0.1)
        # Lactate should decrease (negative flux + diffusion)
        assert model.shared_metabolites["lactate"] != initial_lactate

    def test_update_metabolite_pool_negative_clamped(self):
        model = MFCCommunityModel()
        model.shared_metabolites = {"lactate": 0.001}
        fluxes = {"org1": {"EX_lactate_e": -100.0}}
        model._update_metabolite_pool(fluxes, dt=1.0)
        assert model.shared_metabolites["lactate"] >= 0.0

    def test_apply_interactions_cross_feeding(self):
        model = MFCCommunityModel()
        model.add_interaction("org1", "org2", "lactate", "cross-feeding", 1.0)
        growth_rates = {"org1": 0.5, "org2": 0.1}
        model._apply_interactions(growth_rates)
        assert growth_rates["org2"] > 0.1  # Should get bonus

    def test_apply_interactions_competition(self):
        model = MFCCommunityModel()
        model.add_interaction("org1", "org2", "lactate", "competition", 1.0)
        growth_rates = {"org1": 0.5, "org2": 0.3}
        model._apply_interactions(growth_rates)
        assert growth_rates["org2"] < 0.3  # Should get penalty

    def test_apply_interactions_competition_clamp(self):
        model = MFCCommunityModel()
        model.add_interaction("org1", "org2", "lactate", "competition", 100.0)
        growth_rates = {"org1": 0.5, "org2": 0.01}
        model._apply_interactions(growth_rates)
        assert growth_rates["org2"] >= 0

    def test_apply_interactions_producer_not_present(self):
        model = MFCCommunityModel()
        model.add_interaction("org_missing", "org2", "lactate", "cross-feeding")
        growth_rates = {"org2": 0.1}
        model._apply_interactions(growth_rates)
        assert growth_rates["org2"] == 0.1  # No change

    def test_apply_interactions_consumer_not_present(self):
        model = MFCCommunityModel()
        model.add_interaction("org1", "org_missing", "lactate", "cross-feeding")
        growth_rates = {"org1": 0.5}
        model._apply_interactions(growth_rates)
        # Should not raise

    def test_calculate_electron_production(self):
        model = MFCCommunityModel()
        model.setup_mfc_community()
        model.shared_metabolites["riboflavin"] = 0.1
        fluxes = {"org1": {"cytochrome_c": 1.0, "r1": 2.0}}
        rate = model._calculate_electron_production(fluxes)
        assert rate > 0

    def test_calculate_electron_production_no_matches(self):
        model = MFCCommunityModel()
        model.shared_metabolites = {}
        model.electrode_mediators = {}
        fluxes = {"org1": {"r1": 1.0}}
        rate = model._calculate_electron_production(fluxes)
        assert rate == 0.0

    def test_get_dominant_organism_empty(self):
        model = MFCCommunityModel()
        assert model.get_dominant_organism() == "None"

    def test_get_dominant_organism(self):
        model = MFCCommunityModel()
        model.add_organism("org1", MagicMock(), 0.3)
        model.add_organism("org2", MagicMock(), 0.7)
        assert model.get_dominant_organism() == "org2"

    def test_get_diversity_index_empty(self):
        model = MFCCommunityModel()
        assert model.get_diversity_index() == 0.0

    def test_get_diversity_index(self):
        model = MFCCommunityModel()
        model.add_organism("org1", MagicMock(), 0.5)
        model.add_organism("org2", MagicMock(), 0.5)
        diversity = model.get_diversity_index()
        assert diversity > 0

    def test_get_diversity_index_zero_total(self):
        model = MFCCommunityModel()
        mock_wrapper = MagicMock()
        model.add_organism("org1", mock_wrapper, 0.0)
        assert model.get_diversity_index() == 0.0

    def test_simulate_succession(self):
        model = MFCCommunityModel()
        model.setup_mfc_community()
        df = model.simulate_succession(duration=0.5, dt=0.1)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5  # 0.5/0.1 = 5 steps

    def test_simulate_succession_empty_history(self):
        model = MFCCommunityModel()
        model.setup_mfc_community()
        # Duration 0 means 0 steps
        df = model.simulate_succession(duration=0.0, dt=0.1)
        assert isinstance(df, pd.DataFrame)
        assert df.empty


class TestCommunityElectrodeIntegration:
    def test_init(self):
        community = MFCCommunityModel()
        integrator = CommunityElectrodeIntegration(community)
        assert integrator.biofilm_density == 50.0
        assert integrator.biofilm_thickness == 100e-6

    def test_calculate_current_density_no_history(self):
        community = MFCCommunityModel()
        integrator = CommunityElectrodeIntegration(community)
        current = integrator.calculate_current_density()
        assert current == 0.0

    def test_calculate_current_density_with_history(self):
        community = MFCCommunityModel()
        state = CommunityState(
            time=1.0,
            abundances={},
            metabolite_concentrations={},
            community_growth_rate=0.1,
            electron_production_rate=1.0,
        )
        community.history.append(state)
        integrator = CommunityElectrodeIntegration(community)
        current = integrator.calculate_current_density()
        assert current > 0

    def test_get_optimization_objectives_no_history(self):
        community = MFCCommunityModel()
        integrator = CommunityElectrodeIntegration(community)
        objectives = integrator.get_optimization_objectives()
        assert "maximize_current_density" in objectives
        assert objectives["maximize_electron_production"] == 0.0

    def test_get_optimization_objectives_with_history(self):
        community = MFCCommunityModel()
        state = CommunityState(
            time=1.0,
            abundances={},
            metabolite_concentrations={"lactate": 5.0},
            community_growth_rate=0.1,
            electron_production_rate=2.0,
        )
        community.history.append(state)
        integrator = CommunityElectrodeIntegration(community)
        objectives = integrator.get_optimization_objectives()
        assert objectives["maximize_electron_production"] == 2.0
        assert objectives["minimize_substrate_waste"] > 0

    def test_get_optimization_objectives_no_lactate(self):
        community = MFCCommunityModel()
        state = CommunityState(
            time=1.0,
            abundances={},
            metabolite_concentrations={"acetate": 5.0},
            community_growth_rate=0.1,
            electron_production_rate=2.0,
        )
        community.history.append(state)
        integrator = CommunityElectrodeIntegration(community)
        objectives = integrator.get_optimization_objectives()
        assert objectives["minimize_substrate_waste"] == 0.0
