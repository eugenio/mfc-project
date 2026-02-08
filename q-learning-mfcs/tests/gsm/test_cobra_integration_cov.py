"""Tests for cobra_integration module - targeting 98%+ coverage.

cobra and mackinac are mocked since they are not installed.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest
from unittest.mock import MagicMock, patch, mock_open
import pandas as pd


# ---- Mock cobra before importing the module ----
mock_cobra = MagicMock()
mock_cobra.io.load_json_model = MagicMock(return_value=MagicMock())
mock_cobra.io.read_sbml_model = MagicMock(return_value=MagicMock())
mock_cobra.manipulation.knock_out_model_genes = MagicMock()

mock_fva = MagicMock()
mock_pfba = MagicMock()
mock_lrc = MagicMock()

sys.modules["cobra"] = mock_cobra
sys.modules["cobra.flux_analysis"] = MagicMock(flux_variability_analysis=mock_fva)
sys.modules["cobra.flux_analysis.parsimonious"] = MagicMock(pfba=mock_pfba)
sys.modules["cobra.util"] = MagicMock()
sys.modules["cobra.util.solver"] = MagicMock(linear_reaction_coefficients=mock_lrc)

# Mock mackinac
mock_mackinac = MagicMock()
sys.modules["mackinac"] = mock_mackinac

# Now import
from gsm.cobra_integration import (
    FBAResult,
    FVAResult,
    COBRAModelWrapper,
    COBRA_AVAILABLE,
    MACKINAC_AVAILABLE,
)


class TestFBAResult:
    def test_get_active_reactions(self):
        result = FBAResult(
            objective_value=0.5,
            fluxes={"r1": 1.0, "r2": 0.0, "r3": -0.5, "r4": 1e-8},
            shadow_prices={},
            reduced_costs={},
            status="optimal",
        )
        active = result.get_active_reactions()
        assert "r1" in active
        assert "r3" in active
        assert "r2" not in active
        assert "r4" not in active

    def test_get_active_reactions_custom_threshold(self):
        result = FBAResult(
            objective_value=0.5,
            fluxes={"r1": 0.001},
            shadow_prices={},
            reduced_costs={},
            status="optimal",
        )
        assert result.get_active_reactions(threshold=0.01) == []
        assert result.get_active_reactions(threshold=0.0001) == ["r1"]

    def test_to_dataframe(self):
        result = FBAResult(
            objective_value=1.0,
            fluxes={"r1": 1.0, "r2": 2.0},
            shadow_prices={},
            reduced_costs={},
            status="optimal",
        )
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "reaction_id" in df.columns
        assert "flux" in df.columns
        assert len(df) == 2


class TestFVAResult:
    def test_get_variable_reactions(self):
        result = FVAResult(
            minimum_fluxes={"r1": 0.0, "r2": 1.0, "r3": 5.0},
            maximum_fluxes={"r1": 10.0, "r2": 1.0, "r3": 5.0000001},
        )
        variable = result.get_variable_reactions()
        assert "r1" in variable
        assert "r2" not in variable
        assert "r3" not in variable

    def test_to_dataframe(self):
        result = FVAResult(
            minimum_fluxes={"r1": 0.0, "r2": 1.0},
            maximum_fluxes={"r1": 10.0, "r2": 5.0},
        )
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "minimum" in df.columns
        assert "maximum" in df.columns
        assert "range" in df.columns


class TestCOBRAModelWrapper:
    def _make_mock_model(self):
        model = MagicMock()
        rxn1 = MagicMock()
        rxn1.id = "r1"
        rxn1.lower_bound = 0.0
        rxn1.upper_bound = 1000.0
        rxn1.name = "Reaction 1"
        rxn1.subsystem = "Central"

        rxn2 = MagicMock()
        rxn2.id = "EX_o2_e"
        rxn2.lower_bound = -10.0
        rxn2.upper_bound = 0.0
        rxn2.name = "O2 exchange"
        rxn2.subsystem = "Exchange"

        met1 = MagicMock()
        met1.id = "glc_e"

        rxn2.metabolites = [met1]
        model.reactions = [rxn1, rxn2]
        model.exchanges = [rxn2]
        model.metabolites = [met1]
        model.genes = [MagicMock()]
        model.compartments = {"c": "cytoplasm", "e": "extracellular"}
        model.objective = MagicMock()
        model.solver = MagicMock()
        model.solver.interface.__name__ = "glpk"
        model.copy.return_value = model
        return model

    def test_init_with_model(self):
        model = self._make_mock_model()
        wrapper = COBRAModelWrapper(model)
        assert wrapper.model is model
        assert len(wrapper.original_bounds) == 2

    def test_init_no_model(self):
        wrapper = COBRAModelWrapper(None)
        assert wrapper.model is None
        assert wrapper.original_bounds == {}

    def test_from_sbml(self):
        wrapper = COBRAModelWrapper.from_sbml("test.xml")
        mock_cobra.io.read_sbml_model.assert_called_with("test.xml")

    def test_from_modelseed(self):
        wrapper = COBRAModelWrapper.from_modelseed("model123")
        mock_mackinac.create_cobra_model_from_modelseed.assert_called_with("model123")

    def test_from_bigg_cached(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        model_file = cache_dir / "e_coli_core.json"
        model_file.write_text("{}")

        wrapper = COBRAModelWrapper.from_bigg("e_coli_core", str(cache_dir))
        mock_cobra.io.load_json_model.assert_called()

    def test_from_bigg_download(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_requests = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "{}"
        mock_resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_resp
        with patch.dict(sys.modules, {"requests": mock_requests}):
            wrapper = COBRAModelWrapper.from_bigg("test_model", str(cache_dir))

    def test_optimize_no_model(self):
        wrapper = COBRAModelWrapper(None)
        with pytest.raises(ValueError, match="No model loaded"):
            wrapper.optimize()

    def test_optimize_default_objective(self):
        model = self._make_mock_model()
        solution = MagicMock()
        solution.objective_value = 0.5
        solution.fluxes = MagicMock()
        solution.fluxes.to_dict.return_value = {"r1": 1.0}
        solution.shadow_prices = MagicMock()
        solution.shadow_prices.to_dict.return_value = {}
        solution.reduced_costs = MagicMock()
        solution.reduced_costs.to_dict.return_value = {}
        solution.status = "optimal"
        model.optimize.return_value = solution

        wrapper = COBRAModelWrapper(model)
        result = wrapper.optimize()
        assert result.objective_value == 0.5
        assert result.status == "optimal"

    def test_optimize_string_objective(self):
        model = self._make_mock_model()
        solution = MagicMock()
        solution.objective_value = 1.0
        solution.fluxes = {"r1": 0.5}
        solution.shadow_prices = {}
        solution.reduced_costs = {}
        solution.status = "optimal"
        model.optimize.return_value = solution

        wrapper = COBRAModelWrapper(model)
        result = wrapper.optimize("r1")
        assert result.objective_value == 1.0

    def test_optimize_dict_objective(self):
        model = self._make_mock_model()
        solution = MagicMock()
        solution.objective_value = 0.3
        solution.fluxes = {"r1": 0.3}
        solution.shadow_prices = {}
        solution.reduced_costs = {}
        solution.status = "optimal"
        model.optimize.return_value = solution

        wrapper = COBRAModelWrapper(model)
        result = wrapper.optimize({"r1": 1.0})
        assert result.objective_value == 0.3

    def test_fva_no_model(self):
        wrapper = COBRAModelWrapper(None)
        with pytest.raises(ValueError):
            wrapper.flux_variability_analysis()

    def test_fva(self):
        model = self._make_mock_model()
        fva_df = MagicMock()
        fva_df.__getitem__ = MagicMock(side_effect=[
            MagicMock(to_dict=MagicMock(return_value={"r1": 0.0})),
            MagicMock(to_dict=MagicMock(return_value={"r1": 10.0})),
        ])

        with patch("gsm.cobra_integration.flux_variability_analysis", return_value=fva_df):
            wrapper = COBRAModelWrapper(model)
            result = wrapper.flux_variability_analysis(["r1"], 0.9)
            assert isinstance(result, FVAResult)

    def test_parsimonious_fba_no_model(self):
        wrapper = COBRAModelWrapper(None)
        with pytest.raises(ValueError):
            wrapper.parsimonious_fba()

    def test_parsimonious_fba(self):
        model = self._make_mock_model()
        solution = MagicMock()
        solution.objective_value = 0.4
        solution.fluxes = MagicMock()
        solution.fluxes.to_dict.return_value = {"r1": 0.2}
        solution.status = "optimal"

        with patch("gsm.cobra_integration.pfba", return_value=solution):
            wrapper = COBRAModelWrapper(model)
            result = wrapper.parsimonious_fba()
            assert result.objective_value == 0.4

    def test_set_media_conditions_no_model(self):
        wrapper = COBRAModelWrapper(None)
        with pytest.raises(ValueError):
            wrapper.set_media_conditions({"glc": -10.0})

    def test_set_media_conditions(self):
        model = self._make_mock_model()
        wrapper = COBRAModelWrapper(model)
        wrapper.set_media_conditions({"glc": -10.0})

    def test_set_oxygen_no_model(self):
        wrapper = COBRAModelWrapper(None)
        with pytest.raises(ValueError):
            wrapper.set_oxygen_availability(-10.0)

    def test_set_oxygen_found(self):
        model = self._make_mock_model()
        o2_rxn = MagicMock()
        o2_rxn.id = "EX_o2_e"
        model.exchanges = [o2_rxn]
        wrapper = COBRAModelWrapper(model)
        wrapper.set_oxygen_availability(-5.0)
        assert o2_rxn.lower_bound == -5.0

    def test_set_oxygen_not_found(self):
        model = self._make_mock_model()
        rxn = MagicMock()
        rxn.id = "EX_glc_e"
        model.exchanges = [rxn]
        wrapper = COBRAModelWrapper(model)
        wrapper.set_oxygen_availability(-5.0)

    def test_knock_out_genes_no_model(self):
        wrapper = COBRAModelWrapper(None)
        with pytest.raises(ValueError):
            wrapper.knock_out_genes(["geneA"])

    def test_knock_out_genes(self):
        model = self._make_mock_model()
        gene = MagicMock()
        gene.id = "geneA"
        ko_model = MagicMock()
        ko_model.genes = {"geneA": gene}
        model.copy.return_value = ko_model

        solution = MagicMock()
        solution.objective_value = 0.1
        solution.fluxes = {"r1": 0.1}
        solution.status = "optimal"
        ko_model.optimize.return_value = solution

        wrapper = COBRAModelWrapper(model)
        result = wrapper.knock_out_genes(["geneA", "unknown_gene"])
        assert result.status == "optimal"

    def test_get_model_statistics_no_model(self):
        wrapper = COBRAModelWrapper(None)
        assert wrapper.get_model_statistics() == {}

    def test_get_model_statistics(self):
        model = self._make_mock_model()
        wrapper = COBRAModelWrapper(model)
        stats = wrapper.get_model_statistics()
        assert "num_reactions" in stats
        assert "num_metabolites" in stats
        assert "num_genes" in stats

    def test_find_electron_transfer_reactions_no_model(self):
        wrapper = COBRAModelWrapper(None)
        assert wrapper.find_electron_transfer_reactions() == []

    def test_find_electron_transfer_reactions(self):
        model = self._make_mock_model()
        rxn = MagicMock()
        rxn.id = "CYT_COMPLEX"
        rxn.name = "cytochrome oxidase"
        rxn.subsystem = "Electron Transport"
        model.reactions = [rxn]
        wrapper = COBRAModelWrapper(model)
        result = wrapper.find_electron_transfer_reactions()
        assert "CYT_COMPLEX" in result

    def test_get_exchange_reactions_no_model(self):
        wrapper = COBRAModelWrapper(None)
        assert wrapper.get_exchange_reactions() == {}

    def test_get_exchange_reactions(self):
        model = self._make_mock_model()
        wrapper = COBRAModelWrapper(model)
        exchanges = wrapper.get_exchange_reactions()
        assert isinstance(exchanges, dict)

    def test_init_cobra_not_available(self):
        with patch("gsm.cobra_integration.COBRA_AVAILABLE", False):
            with pytest.raises(ImportError, match="COBRApy is required"):
                COBRAModelWrapper(None)

    def test_from_modelseed_mackinac_not_available(self):
        with patch("gsm.cobra_integration.MACKINAC_AVAILABLE", False):
            with pytest.raises(ImportError, match="Mackinac is required"):
                COBRAModelWrapper.from_modelseed("model123")

    def test_from_bigg_download_failure(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_requests = MagicMock()
        mock_requests.get.side_effect = Exception("Network error")
        with patch.dict(sys.modules, {"requests": mock_requests}):
            with pytest.raises(Exception, match="Network error"):
                COBRAModelWrapper.from_bigg("bad_model", str(cache_dir))
