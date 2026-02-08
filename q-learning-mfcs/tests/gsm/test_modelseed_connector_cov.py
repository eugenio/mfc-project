"""Tests for modelseed_connector module - targeting 98%+ coverage."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest
from unittest.mock import MagicMock, patch
import json

# Mock cobra/mackinac before import
mock_cobra = MagicMock()
sys.modules["cobra"] = mock_cobra
sys.modules["cobra.flux_analysis"] = MagicMock()
sys.modules["cobra.flux_analysis.parsimonious"] = MagicMock()
sys.modules["cobra.util"] = MagicMock()
sys.modules["cobra.util.solver"] = MagicMock()
sys.modules["mackinac"] = MagicMock()

from gsm.modelseed_connector import (
    ModelSEEDModelInfo,
    ModelSEEDReaction,
    ModelSEEDConnector,
    get_mfc_relevant_models,
    setup_mfc_community_from_modelseed,
)


class TestModelSEEDModelInfo:
    def test_defaults(self):
        info = ModelSEEDModelInfo(
            model_id="m1", organism_name="Org", genome_id="g1"
        )
        assert info.model_type == "Single organism"
        assert info.reactions_count == 0


class TestModelSEEDReaction:
    def test_defaults(self):
        rxn = ModelSEEDReaction(
            reaction_id="r1", name="Test", equation="A->B",
            stoichiometry={"A": -1, "B": 1},
        )
        assert rxn.reversibility is True
        assert rxn.ec_numbers == []


class TestModelSEEDConnector:
    def test_init(self, tmp_path):
        connector = ModelSEEDConnector(cache_dir=str(tmp_path / "cache"))
        assert connector.model_cache == {}

    def test_init_mackinac_not_available(self, tmp_path):
        with patch("gsm.modelseed_connector.MACKINAC_AVAILABLE", False):
            connector = ModelSEEDConnector(cache_dir=str(tmp_path / "cache"))
            assert connector is not None

    def test_search_models_no_mackinac(self, tmp_path):
        with patch("gsm.modelseed_connector.MACKINAC_AVAILABLE", False):
            connector = ModelSEEDConnector(cache_dir=str(tmp_path / "cache"))
            result = connector.search_models()
            assert result == []

    def test_search_models_all(self, tmp_path):
        connector = ModelSEEDConnector(cache_dir=str(tmp_path / "cache"))
        result = connector.search_models()
        assert len(result) == 3

    def test_search_models_by_organism(self, tmp_path):
        connector = ModelSEEDConnector(cache_dir=str(tmp_path / "cache"))
        result = connector.search_models(organism_name="Shewanella")
        assert len(result) == 1
        assert "Shewanella" in result[0].organism_name

    def test_search_models_by_genome(self, tmp_path):
        connector = ModelSEEDConnector(cache_dir=str(tmp_path / "cache"))
        result = connector.search_models(genome_id="211586")
        assert len(result) == 1

    def test_search_models_limit(self, tmp_path):
        connector = ModelSEEDConnector(cache_dir=str(tmp_path / "cache"))
        result = connector.search_models(limit=1)
        assert len(result) == 1

    def test_load_model_no_mackinac(self, tmp_path):
        with patch("gsm.modelseed_connector.MACKINAC_AVAILABLE", False):
            connector = ModelSEEDConnector(cache_dir=str(tmp_path / "cache"))
            result = connector.load_model_from_modelseed("m1")
            assert result is None

    def test_load_model_no_cobra(self, tmp_path):
        with patch("gsm.modelseed_connector.COBRA_AVAILABLE", False):
            connector = ModelSEEDConnector(cache_dir=str(tmp_path / "cache"))
            result = connector.load_model_from_modelseed("m1")
            assert result is None

    def test_load_model_success(self, tmp_path):
        connector = ModelSEEDConnector(cache_dir=str(tmp_path / "cache"))
        result = connector.load_model_from_modelseed("test_model")
        assert result is not None
        assert "test_model" in connector.model_cache

    def test_load_model_cached(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "test_model_modelseed.pkl"
        cache_file.write_text("cached")
        connector = ModelSEEDConnector(cache_dir=str(cache_dir))
        result = connector.load_model_from_modelseed("test_model")
        assert result is not None

    def test_search_models_exception(self, tmp_path):
        connector = ModelSEEDConnector(cache_dir=str(tmp_path / "cache"))
        with patch("gsm.modelseed_connector.ModelSEEDModelInfo", side_effect=Exception("boom")):
            result = connector.search_models()
            assert result == []

    def test_load_model_exception(self, tmp_path):
        connector = ModelSEEDConnector(cache_dir=str(tmp_path / "cache"))
        with patch("gsm.modelseed_connector.COBRAModelWrapper", side_effect=Exception("fail")):
            result = connector.load_model_from_modelseed("m1")
            assert result is None

    def test_get_reaction_database_exception(self, tmp_path):
        connector = ModelSEEDConnector(cache_dir=str(tmp_path / "cache"))
        with patch("gsm.modelseed_connector.ModelSEEDReaction", side_effect=Exception("db error")):
            result = connector.get_reaction_database()
            assert result == []

    def test_create_community_exception(self, tmp_path):
        connector = ModelSEEDConnector(cache_dir=str(tmp_path / "cache"))
        with patch.object(connector, "load_model_from_modelseed", side_effect=Exception("fail")):
            result = connector.create_community_model(["m1"])
            assert result is None

    def test_validate_model_exception(self, tmp_path):
        connector = ModelSEEDConnector(cache_dir=str(tmp_path / "cache"))
        wrapper = MagicMock()
        wrapper.model_metadata = property(lambda self: (_ for _ in ()).throw(Exception("boom")))
        type(wrapper).model_metadata = property(lambda self: (_ for _ in ()).throw(Exception("boom")))
        result = connector.validate_model_quality(wrapper)
        assert result["status"] == "error"

    def test_export_model_summary_exception(self, tmp_path):
        connector = ModelSEEDConnector(cache_dir=str(tmp_path / "cache"))
        with patch("gsm.modelseed_connector.pd.Timestamp") as mock_ts:
            mock_ts.now.side_effect = Exception("timestamp error")
            wrapper = MagicMock()
            result = connector.export_model_summary(wrapper)
            assert "error" in result

    def test_get_reaction_database_no_mackinac(self, tmp_path):
        with patch("gsm.modelseed_connector.MACKINAC_AVAILABLE", False):
            connector = ModelSEEDConnector(cache_dir=str(tmp_path / "cache"))
            result = connector.get_reaction_database()
            assert result == []

    def test_get_reaction_database(self, tmp_path):
        connector = ModelSEEDConnector(cache_dir=str(tmp_path / "cache"))
        result = connector.get_reaction_database()
        assert len(result) == 3

    def test_get_reaction_database_filtered(self, tmp_path):
        connector = ModelSEEDConnector(cache_dir=str(tmp_path / "cache"))
        result = connector.get_reaction_database(pathway_filter="Electron")
        assert len(result) == 1

    def test_create_community_no_deps(self, tmp_path):
        with patch("gsm.modelseed_connector.MACKINAC_AVAILABLE", False):
            connector = ModelSEEDConnector(cache_dir=str(tmp_path / "cache"))
            result = connector.create_community_model(["m1"])
            assert result is None

    def test_create_community_success(self, tmp_path):
        connector = ModelSEEDConnector(cache_dir=str(tmp_path / "cache"))
        result = connector.create_community_model(["m1", "m2"], "comm1")
        assert result is not None

    def test_create_community_no_models_loaded(self, tmp_path):
        connector = ModelSEEDConnector(cache_dir=str(tmp_path / "cache"))
        with patch.object(connector, "load_model_from_modelseed", return_value=None):
            result = connector.create_community_model(["m1"])
            assert result is None

    def test_validate_model_no_mackinac(self, tmp_path):
        with patch("gsm.modelseed_connector.MACKINAC_AVAILABLE", False):
            connector = ModelSEEDConnector(cache_dir=str(tmp_path / "cache"))
            result = connector.validate_model_quality(MagicMock())
            assert result["status"] == "unavailable"

    def test_validate_model_with_model(self, tmp_path):
        connector = ModelSEEDConnector(cache_dir=str(tmp_path / "cache"))
        wrapper = MagicMock()
        wrapper.model_metadata = {"id": "test"}
        model = MagicMock()
        rxn = MagicMock()
        rxn.id = "biomass_rxn"
        model.reactions = [rxn]
        model.metabolites = [MagicMock()]
        model.genes = [MagicMock()]
        model.exchanges = [MagicMock()]
        wrapper.model = model
        result = connector.validate_model_quality(wrapper)
        assert result["status"] == "success"
        assert result["score"] > 0

    def test_validate_model_no_biomass(self, tmp_path):
        connector = ModelSEEDConnector(cache_dir=str(tmp_path / "cache"))
        wrapper = MagicMock()
        model = MagicMock()
        rxn = MagicMock()
        rxn.id = "some_reaction"
        model.reactions = [rxn]
        model.metabolites = [MagicMock()]
        model.genes = [MagicMock()]
        model.exchanges = [MagicMock()]
        wrapper.model = model
        result = connector.validate_model_quality(wrapper)
        assert "Add biomass reaction" in str(result["recommendations"])

    def test_validate_model_no_valid_model(self, tmp_path):
        connector = ModelSEEDConnector(cache_dir=str(tmp_path / "cache"))
        wrapper = MagicMock()
        wrapper.model = None
        result = connector.validate_model_quality(wrapper)
        assert result["status"] == "error"

    def test_validate_model_low_score(self, tmp_path):
        connector = ModelSEEDConnector(cache_dir=str(tmp_path / "cache"))
        wrapper = MagicMock()
        model = MagicMock()
        rxn = MagicMock()
        rxn.id = "r1"
        model.reactions = []
        model.metabolites = []
        model.genes = []
        model.exchanges = []
        wrapper.model = model
        result = connector.validate_model_quality(wrapper)
        assert result["score"] < 0.8

    def test_export_model_summary(self, tmp_path):
        connector = ModelSEEDConnector(cache_dir=str(tmp_path / "cache"))
        wrapper = MagicMock()
        wrapper.model_metadata = {"id": "test"}
        wrapper.get_model_statistics.return_value = {"num_reactions": 10}
        wrapper.find_electron_transfer_reactions.return_value = ["r1"]
        wrapper.model = None
        summary = connector.export_model_summary(wrapper)
        assert "model_info" in summary
        assert "statistics" in summary

    def test_export_model_summary_to_file(self, tmp_path):
        connector = ModelSEEDConnector(cache_dir=str(tmp_path / "cache"))
        wrapper = MagicMock()
        wrapper.model_metadata = {"id": "test"}
        wrapper.model = None
        out_file = str(tmp_path / "output" / "summary.json")
        summary = connector.export_model_summary(wrapper, output_file=out_file)
        assert os.path.exists(out_file)

    def test_export_model_summary_no_methods(self, tmp_path):
        connector = ModelSEEDConnector(cache_dir=str(tmp_path / "cache"))
        wrapper = MagicMock(spec=[])
        summary = connector.export_model_summary(wrapper)
        assert "model_info" in summary


class TestModuleFunctions:
    def test_get_mfc_relevant_models(self):
        models = get_mfc_relevant_models()
        assert len(models) == 5
        assert "Shewanella_oneidensis_MR1" in models

    def test_setup_mfc_community_default(self):
        result = setup_mfc_community_from_modelseed()
        assert result is not None

    def test_setup_mfc_community_custom(self):
        result = setup_mfc_community_from_modelseed(["model1"])
        assert result is not None

    def test_setup_mfc_community_failure(self):
        with patch("gsm.modelseed_connector.ModelSEEDConnector") as MockConnector:
            instance = MockConnector.return_value
            instance.create_community_model.return_value = None
            result = setup_mfc_community_from_modelseed(["model1"])
            assert result is None

    def test_setup_mfc_community_exception(self):
        with patch("gsm.modelseed_connector.ModelSEEDConnector") as MockConnector:
            instance = MockConnector.return_value
            instance.create_community_model.side_effect = Exception("error")
            result = setup_mfc_community_from_modelseed(["model1"])
            assert result is None
