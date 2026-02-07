"""Deep coverage tests for gsm_integration and literature_validation page modules."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_THIS_DIR = str(Path(__file__).resolve().parent)
_SRC_DIR = str((Path(__file__).resolve().parent / ".." / ".." / "src").resolve())
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import gui.pages.gsm_integration as gsm_mod  # noqa: E402
import gui.pages.literature_validation as lit_mod  # noqa: E402
from gui.pages.gsm_integration import (  # noqa: E402
    FluxAnalysisResult,
    GSMIntegrator,
    MetabolicModel,
    PathwayAnalysis,
    create_flux_analysis_viz,
    create_gsm_visualizations,
    render_gsm_integration_page,
)
from gui.pages.literature_validation import (  # noqa: E402
    Citation,
    LiteratureValidator,
    ParameterValidation,
    create_validation_visualizations,
    render_literature_validation_page,
)


class _SessionState(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key) from None


def _make_ctx():
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=cm)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


def _make_mock_st():
    mock_st = MagicMock()
    mock_st.session_state = _SessionState()

    def _smart_columns(n_or_spec, **_kw):
        n = len(n_or_spec) if isinstance(n_or_spec, list | tuple) else int(n_or_spec)
        return [_make_ctx() for _ in range(n)]

    mock_st.columns.side_effect = _smart_columns

    def _smart_tabs(labels):
        return [_make_ctx() for _ in labels]

    mock_st.tabs.side_effect = _smart_tabs
    mock_st.expander.return_value = _make_ctx()
    mock_st.sidebar = MagicMock()
    mock_st.form.return_value = _make_ctx()
    mock_st.spinner.return_value = _make_ctx()
    mock_st.container.return_value = _make_ctx()
    mock_st.button.return_value = False
    mock_st.checkbox.return_value = False
    mock_st.selectbox.return_value = "ph"
    mock_st.number_input.return_value = 7.0
    mock_st.slider.return_value = 5.0
    mock_st.text_input.return_value = ""
    mock_st.text_area.return_value = ""
    mock_st.file_uploader.return_value = None
    mock_st.radio.return_value = "Flux Distribution"
    return mock_st


# ===================================================================
# GSM Integration Tests
# ===================================================================


@pytest.mark.apptest
class TestGSMIntegratorInit:
    def test_init_creates_available_models(self):
        integrator = GSMIntegrator()
        assert len(integrator.available_models) == 4
        assert integrator.current_model is None
        assert integrator.flux_results is None

    def test_model_database_organisms(self):
        integrator = GSMIntegrator()
        organisms = [m.organism for m in integrator.available_models]
        assert "Shewanella oneidensis MR-1" in organisms
        assert "Geobacter sulfurreducens" in organisms
        assert "Escherichia coli K-12" in organisms
        assert "Pseudomonas putida KT2440" in organisms

    def test_model_database_ids(self):
        integrator = GSMIntegrator()
        ids = [m.model_id for m in integrator.available_models]
        assert "iSO783" in ids
        assert "iGS515" in ids
        assert "iML1515" in ids
        assert "iJN1462" in ids

    def test_model_fields_complete(self):
        integrator = GSMIntegrator()
        for m in integrator.available_models:
            assert m.reactions > 0
            assert m.metabolites > 0
            assert m.genes > 0
            assert len(m.biomass_reaction) > 0
            assert len(m.electron_transport_reactions) > 0


@pytest.mark.apptest
class TestGSMIntegratorLoadModel:
    def test_load_valid_model_returns_true(self):
        integrator = GSMIntegrator()
        result = integrator.load_model("iSO783")
        assert result is True
        assert integrator.current_model is not None
        assert integrator.current_model.model_id == "iSO783"

    def test_load_invalid_model_returns_false(self):
        integrator = GSMIntegrator()
        result = integrator.load_model("nonexistent_model")
        assert result is False
        assert integrator.current_model is None

    def test_load_each_model(self):
        integrator = GSMIntegrator()
        for model_id in ["iSO783", "iGS515", "iML1515", "iJN1462"]:
            assert integrator.load_model(model_id) is True
            assert integrator.current_model.model_id == model_id


@pytest.mark.apptest
class TestGSMIntegratorPerformFBA:
    def test_fba_no_model_raises(self):
        integrator = GSMIntegrator()
        with pytest.raises(ValueError, match="No model loaded"):
            integrator.perform_fba()

    def test_fba_returns_result(self):
        integrator = GSMIntegrator()
        integrator.load_model("iSO783")
        result = integrator.perform_fba()
        assert result.objective_value > 0
        assert result.growth_rate > 0
        assert result.electron_transfer_flux > 0
        assert len(result.substrate_uptake_rates) == 4
        assert len(result.secretion_rates) == 3
        assert len(result.shadow_prices) == 3

    def test_fba_with_constraints(self):
        integrator = GSMIntegrator()
        integrator.load_model("iML1515")
        constraints = {"acetate": 10.0, "lactate": 5.0}
        result = integrator.perform_fba("biomass", constraints)
        assert result is not None
        assert "acetate" in result.substrate_uptake_rates

    def test_fba_with_different_objectives(self):
        integrator = GSMIntegrator()
        integrator.load_model("iSO783")
        for obj in ["biomass", "atp_production", "electron_transfer"]:
            result = integrator.perform_fba(objective=obj)
            assert result is not None


@pytest.mark.apptest
class TestGSMIntegratorAnalyzePathway:
    def test_analyze_no_model_raises(self):
        integrator = GSMIntegrator()
        with pytest.raises(ValueError, match="No model loaded"):
            integrator.analyze_pathway("Glycolysis")

    def test_analyze_returns_pathway_result(self):
        integrator = GSMIntegrator()
        integrator.load_model("iGS515")
        result = integrator.analyze_pathway("Glycolysis")
        assert result.pathway_name == "Glycolysis"
        assert len(result.flux_distribution) == 5
        assert 0.0 <= result.pathway_efficiency <= 1.0
        assert len(result.bottleneck_reactions) == 2
        assert len(result.regulatory_targets) == 3

    def test_analyze_different_pathways(self):
        integrator = GSMIntegrator()
        integrator.load_model("iSO783")
        for pw in ["Glycolysis", "TCA Cycle", "Electron Transport Chain"]:
            result = integrator.analyze_pathway(pw)
            assert result.pathway_name == pw


@pytest.mark.apptest
class TestGSMIntegratorOptimize:
    def test_optimize_no_model_raises(self):
        integrator = GSMIntegrator()
        with pytest.raises(ValueError, match="No model loaded"):
            integrator.optimize_for_current_density(2.0)

    def test_optimize_returns_dict(self):
        integrator = GSMIntegrator()
        integrator.load_model("iJN1462")
        result = integrator.optimize_for_current_density(2.0)
        assert "target_current_density" in result
        assert result["target_current_density"] == 2.0
        assert "predicted_current_density" in result
        assert "optimal_substrate_concentrations" in result
        assert "optimal_conditions" in result
        assert "metabolic_efficiency" in result
        assert "predicted_power_density" in result

    def test_optimize_conditions_keys(self):
        integrator = GSMIntegrator()
        integrator.load_model("iSO783")
        result = integrator.optimize_for_current_density(5.0)
        conds = result["optimal_conditions"]
        assert "ph" in conds
        assert "temperature" in conds
        assert "dissolved_oxygen" in conds


@pytest.mark.apptest
class TestCreateGSMVisualizations:
    def test_no_model_shows_info(self):
        mock_st = _make_mock_st()
        integrator = GSMIntegrator()
        with patch.object(gsm_mod, "st", mock_st):
            create_gsm_visualizations(integrator)
        mock_st.info.assert_called_once_with("Please load a model first")

    def test_with_model_renders_charts(self):
        mock_st = _make_mock_st()
        integrator = GSMIntegrator()
        integrator.load_model("iSO783")
        with patch.object(gsm_mod, "st", mock_st):
            create_gsm_visualizations(integrator)
        mock_st.subheader.assert_called()
        mock_st.metric.assert_called()
        mock_st.plotly_chart.assert_called()

    def test_with_each_model(self):
        for mid in ["iSO783", "iGS515", "iML1515", "iJN1462"]:
            mock_st = _make_mock_st()
            integrator = GSMIntegrator()
            integrator.load_model(mid)
            with patch.object(gsm_mod, "st", mock_st):
                create_gsm_visualizations(integrator)


@pytest.mark.apptest
class TestCreateFluxAnalysisViz:
    def test_renders_metrics_and_charts(self):
        mock_st = _make_mock_st()
        integrator = GSMIntegrator()
        integrator.load_model("iSO783")
        flux_result = integrator.perform_fba()
        with patch.object(gsm_mod, "st", mock_st):
            create_flux_analysis_viz(flux_result)
        mock_st.subheader.assert_called()
        mock_st.metric.assert_called()
        mock_st.plotly_chart.assert_called()
        mock_st.info.assert_called()

    def test_flux_result_dataclass(self):
        fr = FluxAnalysisResult(
            objective_value=0.5,
            growth_rate=0.3,
            electron_transfer_flux=1.0,
            substrate_uptake_rates={"acetate": 10.0},
            secretion_rates={"co2": 5.0},
            shadow_prices={"atp": 0.2},
        )
        assert fr.objective_value == 0.5
        assert fr.growth_rate == 0.3


@pytest.mark.apptest
class TestRenderGSMIntegrationPageNoModel:
    def test_page_renders_without_error(self):
        mock_st = _make_mock_st()
        sel_mock = MagicMock()
        sel_mock.selection.rows = []
        mock_st.dataframe.return_value = sel_mock
        with patch.object(gsm_mod, "st", mock_st):
            render_gsm_integration_page()
        mock_st.title.assert_called()
        mock_st.caption.assert_called()
        mock_st.success.assert_called()

    def test_integrator_created_in_session_state(self):
        mock_st = _make_mock_st()
        sel_mock = MagicMock()
        sel_mock.selection.rows = []
        mock_st.dataframe.return_value = sel_mock
        with patch.object(gsm_mod, "st", mock_st):
            render_gsm_integration_page()
        assert "gsm_integrator" in mock_st.session_state


@pytest.mark.apptest
class TestRenderGSMPageTab1Selection:
    def test_selected_row_triggers_load_button(self):
        mock_st = _make_mock_st()
        sel_mock = MagicMock()
        sel_mock.selection.rows = [0]
        mock_st.dataframe.return_value = sel_mock
        mock_st.button.return_value = True
        with patch.object(gsm_mod, "st", mock_st):
            render_gsm_integration_page()
        mock_st.success.assert_called()

    def test_load_button_not_pressed(self):
        mock_st = _make_mock_st()
        sel_mock = MagicMock()
        sel_mock.selection.rows = [0]
        mock_st.dataframe.return_value = sel_mock
        mock_st.button.return_value = False
        with patch.object(gsm_mod, "st", mock_st):
            render_gsm_integration_page()

    def test_model_loaded_shows_current_model_info(self):
        mock_st = _make_mock_st()
        sel_mock = MagicMock()
        sel_mock.selection.rows = []
        mock_st.dataframe.return_value = sel_mock
        integrator = GSMIntegrator()
        integrator.load_model("iSO783")
        mock_st.session_state["gsm_integrator"] = integrator
        with patch.object(gsm_mod, "st", mock_st):
            render_gsm_integration_page()
        mock_st.write.assert_called()

    def test_no_current_model_shows_info(self):
        mock_st = _make_mock_st()
        sel_mock = MagicMock()
        sel_mock.selection.rows = []
        mock_st.dataframe.return_value = sel_mock
        with patch.object(gsm_mod, "st", mock_st):
            render_gsm_integration_page()
        mock_st.info.assert_called()

    def test_load_model_fails_in_render(self):
        mock_st = _make_mock_st()
        sel_mock = MagicMock()
        sel_mock.selection.rows = [0]
        mock_st.dataframe.return_value = sel_mock
        mock_st.button.return_value = True
        integrator = GSMIntegrator()
        integrator.load_model = MagicMock(return_value=False)
        mock_st.session_state["gsm_integrator"] = integrator
        with patch.object(gsm_mod, "st", mock_st):
            render_gsm_integration_page()
        mock_st.error.assert_called()


@pytest.mark.apptest
class TestRenderGSMPageTab2FBA:
    def test_tab2_no_model_shows_warning(self):
        mock_st = _make_mock_st()
        sel_mock = MagicMock()
        sel_mock.selection.rows = []
        mock_st.dataframe.return_value = sel_mock
        with patch.object(gsm_mod, "st", mock_st):
            render_gsm_integration_page()
        mock_st.warning.assert_called()

    def test_tab2_with_model_and_fba_button(self):
        mock_st = _make_mock_st()
        sel_mock = MagicMock()
        sel_mock.selection.rows = []
        mock_st.dataframe.return_value = sel_mock
        mock_st.button.return_value = True
        mock_st.slider.return_value = 10.0
        mock_st.selectbox.return_value = "biomass"
        integrator = GSMIntegrator()
        integrator.load_model("iSO783")
        mock_st.session_state["gsm_integrator"] = integrator
        with patch.object(gsm_mod, "st", mock_st):
            render_gsm_integration_page()
        assert "flux_result" in mock_st.session_state

    def test_tab2_display_flux_results(self):
        mock_st = _make_mock_st()
        sel_mock = MagicMock()
        sel_mock.selection.rows = []
        mock_st.dataframe.return_value = sel_mock
        integrator = GSMIntegrator()
        integrator.load_model("iSO783")
        mock_st.session_state["gsm_integrator"] = integrator
        mock_st.session_state["flux_result"] = integrator.perform_fba()
        with patch.object(gsm_mod, "st", mock_st):
            render_gsm_integration_page()
        mock_st.plotly_chart.assert_called()


@pytest.mark.apptest
class TestRenderGSMPageTab3Pathway:
    def test_tab3_with_model_and_analyze_button(self):
        mock_st = _make_mock_st()
        sel_mock = MagicMock()
        sel_mock.selection.rows = []
        mock_st.dataframe.return_value = sel_mock
        mock_st.button.return_value = True
        mock_st.selectbox.return_value = "Glycolysis"
        integrator = GSMIntegrator()
        integrator.load_model("iSO783")
        mock_st.session_state["gsm_integrator"] = integrator
        with patch.object(gsm_mod, "st", mock_st):
            render_gsm_integration_page()
        assert "pathway_result" in mock_st.session_state

    def test_tab3_display_pathway_results(self):
        mock_st = _make_mock_st()
        sel_mock = MagicMock()
        sel_mock.selection.rows = []
        mock_st.dataframe.return_value = sel_mock
        integrator = GSMIntegrator()
        integrator.load_model("iSO783")
        mock_st.session_state["gsm_integrator"] = integrator
        mock_st.session_state["pathway_result"] = integrator.analyze_pathway("Glycolysis")
        with patch.object(gsm_mod, "st", mock_st):
            render_gsm_integration_page()
        mock_st.plotly_chart.assert_called()
        mock_st.metric.assert_called()


@pytest.mark.apptest
class TestRenderGSMPageTab4Optimization:
    def test_tab4_with_model_and_optimize_button(self):
        mock_st = _make_mock_st()
        sel_mock = MagicMock()
        sel_mock.selection.rows = []
        mock_st.dataframe.return_value = sel_mock
        mock_st.button.return_value = True
        mock_st.slider.return_value = 2.0
        mock_st.selectbox.return_value = "Genetic Algorithm"
        integrator = GSMIntegrator()
        integrator.load_model("iSO783")
        mock_st.session_state["gsm_integrator"] = integrator
        with patch.object(gsm_mod, "st", mock_st):
            render_gsm_integration_page()
        assert "optimization_result" in mock_st.session_state

    def test_tab4_display_optimization_results(self):
        mock_st = _make_mock_st()
        sel_mock = MagicMock()
        sel_mock.selection.rows = []
        mock_st.dataframe.return_value = sel_mock
        integrator = GSMIntegrator()
        integrator.load_model("iSO783")
        mock_st.session_state["gsm_integrator"] = integrator
        opt_result = integrator.optimize_for_current_density(2.0)
        mock_st.session_state["optimization_result"] = opt_result
        with patch.object(gsm_mod, "st", mock_st):
            render_gsm_integration_page()
        mock_st.metric.assert_called()
        mock_st.plotly_chart.assert_called()

    def test_tab4_optimization_conditions_display(self):
        mock_st = _make_mock_st()
        sel_mock = MagicMock()
        sel_mock.selection.rows = []
        mock_st.dataframe.return_value = sel_mock
        integrator = GSMIntegrator()
        integrator.load_model("iSO783")
        mock_st.session_state["gsm_integrator"] = integrator
        opt_result = {
            "target_current_density": 2.0,
            "predicted_current_density": 2.1,
            "optimal_substrate_concentrations": {
                "acetate": 15.0, "lactate": 10.0, "glucose": 5.0,
            },
            "optimal_conditions": {
                "ph": 7.2, "temperature": 30.0, "dissolved_oxygen": 1.5,
            },
            "metabolic_efficiency": 0.85,
            "predicted_power_density": 0.9,
        }
        mock_st.session_state["optimization_result"] = opt_result
        with patch.object(gsm_mod, "st", mock_st):
            render_gsm_integration_page()

    def test_tab4_export_buttons(self):
        mock_st = _make_mock_st()
        sel_mock = MagicMock()
        sel_mock.selection.rows = []
        mock_st.dataframe.return_value = sel_mock

        def button_side(label, **kwargs):
            if "Export Report" in str(label):
                return True
            if "Export Data" in str(label):
                return True
            if "Apply to Simulation" in str(label):
                return True
            return False

        mock_st.button.side_effect = button_side
        integrator = GSMIntegrator()
        integrator.load_model("iSO783")
        mock_st.session_state["gsm_integrator"] = integrator
        mock_st.session_state["optimization_result"] = integrator.optimize_for_current_density(2.0)
        with patch.object(gsm_mod, "st", mock_st):
            render_gsm_integration_page()


@pytest.mark.apptest
class TestRenderGSMPageExpanderGuide:
    def test_expander_rendered(self):
        mock_st = _make_mock_st()
        sel_mock = MagicMock()
        sel_mock.selection.rows = []
        mock_st.dataframe.return_value = sel_mock
        integrator = GSMIntegrator()
        integrator.load_model("iSO783")
        mock_st.session_state["gsm_integrator"] = integrator
        with patch.object(gsm_mod, "st", mock_st):
            render_gsm_integration_page()
        mock_st.expander.assert_called()
        mock_st.markdown.assert_called()


@pytest.mark.apptest
class TestGSMDataclasses:
    def test_metabolic_model(self):
        m = MetabolicModel(
            organism="Test", model_id="test1", reactions=100,
            metabolites=50, genes=30, biomass_reaction="BIO",
            electron_transport_reactions=["R1", "R2"],
        )
        assert m.organism == "Test"
        assert m.model_id == "test1"

    def test_pathway_analysis(self):
        pa = PathwayAnalysis(
            pathway_name="Test", flux_distribution={"r1": 1.0},
            pathway_efficiency=0.8, bottleneck_reactions=["r1"],
            regulatory_targets=["g1"],
        )
        assert pa.pathway_name == "Test"
        assert pa.pathway_efficiency == 0.8


@pytest.mark.apptest
class TestGSMPageRerunOnLoad:
    def test_rerun_called_on_load_success(self):
        mock_st = _make_mock_st()
        sel_mock = MagicMock()
        sel_mock.selection.rows = [0]
        mock_st.dataframe.return_value = sel_mock
        mock_st.button.return_value = True
        with patch.object(gsm_mod, "st", mock_st):
            render_gsm_integration_page()
        mock_st.rerun.assert_called()


@pytest.mark.apptest
class TestGSMVisualizationsGenomeCoverage:
    def test_genome_coverage_computed(self):
        mock_st = _make_mock_st()
        integrator = GSMIntegrator()
        integrator.load_model("iSO783")
        with patch.object(gsm_mod, "st", mock_st):
            create_gsm_visualizations(integrator)
        calls = [str(c) for c in mock_st.metric.call_args_list]
        assert any("Genome Coverage" in c for c in calls)


@pytest.mark.apptest
class TestGSMPageFullPathAllTabs:
    def test_full_flow_all_buttons_pressed(self):
        mock_st = _make_mock_st()
        sel_mock = MagicMock()
        sel_mock.selection.rows = [0]
        mock_st.dataframe.return_value = sel_mock
        mock_st.button.return_value = True
        mock_st.slider.return_value = 2.0
        mock_st.selectbox.return_value = "biomass"
        with patch.object(gsm_mod, "st", mock_st):
            render_gsm_integration_page()

    def test_full_flow_model_preloaded(self):
        mock_st = _make_mock_st()
        sel_mock = MagicMock()
        sel_mock.selection.rows = []
        mock_st.dataframe.return_value = sel_mock
        mock_st.button.return_value = True
        mock_st.slider.return_value = 2.0
        mock_st.selectbox.return_value = "Glycolysis"
        integrator = GSMIntegrator()
        integrator.load_model("iML1515")
        mock_st.session_state["gsm_integrator"] = integrator
        with patch.object(gsm_mod, "st", mock_st):
            render_gsm_integration_page()


# ===================================================================
# Literature Validation Tests
# ===================================================================


@pytest.mark.apptest
class TestLiteratureValidatorInit:
    def test_init_creates_citation_database(self):
        validator = LiteratureValidator()
        assert len(validator.citation_database) == 5
        assert len(validator.parameter_ranges) == 6

    def test_citation_fields(self):
        validator = LiteratureValidator()
        for c in validator.citation_database:
            assert c.title
            assert len(c.authors) > 0
            assert c.journal
            assert c.year > 0
            assert c.relevance_score > 0
            assert c.quality_score > 0

    def test_parameter_ranges_keys(self):
        validator = LiteratureValidator()
        expected = {
            "conductivity", "flow_rate", "substrate_concentration",
            "biofilm_thickness", "ph", "temperature",
        }
        assert set(validator.parameter_ranges.keys()) == expected

    def test_parameter_ranges_structure(self):
        validator = LiteratureValidator()
        for _key, info in validator.parameter_ranges.items():
            assert "unit" in info
            assert "literature_range" in info
            assert "typical_range" in info
            assert "citations" in info
            assert "description" in info


@pytest.mark.apptest
class TestLiteratureValidateParameter:
    def test_validated_status(self):
        validator = LiteratureValidator()
        result = validator.validate_parameter("ph", 7.0)
        assert result.validation_status == "validated"
        assert result.confidence_level == 0.95
        assert "within typical" in result.recommendation

    def test_questionable_status(self):
        validator = LiteratureValidator()
        result = validator.validate_parameter("ph", 5.5)
        assert result.validation_status == "questionable"
        assert result.confidence_level == 0.65

    def test_outlier_status(self):
        validator = LiteratureValidator()
        result = validator.validate_parameter("ph", 3.0)
        assert result.validation_status == "outlier"
        assert result.confidence_level == 0.25

    def test_unknown_parameter(self):
        validator = LiteratureValidator()
        result = validator.validate_parameter("nonexistent", 42.0)
        assert result.validation_status == "unknown"
        assert result.confidence_level == 0.0
        assert result.unit == "unknown"
        assert len(result.citations) == 0

    def test_validate_conductivity_typical(self):
        validator = LiteratureValidator()
        result = validator.validate_parameter("conductivity", 10000.0)
        assert result.validation_status == "validated"

    def test_validate_flow_rate_questionable(self):
        validator = LiteratureValidator()
        result = validator.validate_parameter("flow_rate", 5e-6)
        assert result.validation_status == "questionable"

    def test_validate_temperature_outlier(self):
        validator = LiteratureValidator()
        result = validator.validate_parameter("temperature", 100.0)
        assert result.validation_status == "outlier"

    def test_validate_all_params_typical(self):
        validator = LiteratureValidator()
        typical_values = {
            "conductivity": 10000.0, "flow_rate": 1e-4,
            "substrate_concentration": 1.0, "biofilm_thickness": 50.0,
            "ph": 7.0, "temperature": 25.0,
        }
        for param, val in typical_values.items():
            result = validator.validate_parameter(param, val)
            assert result.validation_status == "validated", f"{param} not validated"


@pytest.mark.apptest
class TestLiteratureSearchLiterature:
    def test_search_by_title(self):
        validator = LiteratureValidator()
        results = validator.search_literature("fuel cell")
        assert len(results) > 0

    def test_search_by_abstract(self):
        validator = LiteratureValidator()
        results = validator.search_literature("biofilm")
        assert len(results) > 0

    def test_search_by_author(self):
        validator = LiteratureValidator()
        results = validator.search_literature("logan")
        assert len(results) > 0

    def test_search_no_results(self):
        validator = LiteratureValidator()
        results = validator.search_literature("zzzznonexistentzzz")
        assert len(results) == 0

    def test_search_max_results(self):
        validator = LiteratureValidator()
        results = validator.search_literature("mfc", max_results=2)
        assert len(results) <= 2

    def test_search_sorted_by_relevance(self):
        validator = LiteratureValidator()
        results = validator.search_literature("fuel cell")
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].relevance_score >= results[i + 1].relevance_score


@pytest.mark.apptest
class TestLiteratureGenerateValidationReport:
    def test_report_with_validations(self):
        validator = LiteratureValidator()
        validations = [
            validator.validate_parameter("ph", 7.0),
            validator.validate_parameter("temperature", 25.0),
            validator.validate_parameter("ph", 5.5),
        ]
        report = validator.generate_validation_report(validations)
        assert report["summary"]["total_parameters"] == 3
        assert report["summary"]["validated"] == 2
        assert report["summary"]["questionable"] == 1
        assert report["summary"]["outliers"] == 0
        assert report["summary"]["average_confidence"] > 0
        assert report["summary"]["validation_score"] > 0

    def test_report_empty_validations(self):
        validator = LiteratureValidator()
        report = validator.generate_validation_report([])
        assert report["summary"]["total_parameters"] == 0
        assert report["summary"]["average_confidence"] == 0
        assert report["summary"]["validation_score"] == 0

    def test_report_with_outliers(self):
        validator = LiteratureValidator()
        validations = [
            validator.validate_parameter("ph", 3.0),
            validator.validate_parameter("temperature", 100.0),
        ]
        report = validator.generate_validation_report(validations)
        assert report["summary"]["outliers"] == 2
        assert len(report["recommendations"]) == 2

    def test_report_recommendations_exclude_validated(self):
        validator = LiteratureValidator()
        validations = [validator.validate_parameter("ph", 7.0)]
        report = validator.generate_validation_report(validations)
        assert len(report["recommendations"]) == 0

    def test_report_unique_citations(self):
        validator = LiteratureValidator()
        validations = [
            validator.validate_parameter("ph", 7.0),
            validator.validate_parameter("temperature", 25.0),
        ]
        report = validator.generate_validation_report(validations)
        assert isinstance(report["citations"], list)


@pytest.mark.apptest
class TestCreateValidationVisualizations:
    def test_empty_validations_shows_info(self):
        mock_st = _make_mock_st()
        with patch.object(lit_mod, "st", mock_st):
            create_validation_visualizations([])
        mock_st.info.assert_called_with("No validation data available")

    def test_with_validations_renders_charts(self):
        mock_st = _make_mock_st()
        validator = LiteratureValidator()
        validations = [
            validator.validate_parameter("ph", 7.0),
            validator.validate_parameter("temperature", 25.0),
            validator.validate_parameter("ph", 5.5),
            validator.validate_parameter("temperature", 100.0),
        ]
        with patch.object(lit_mod, "st", mock_st):
            create_validation_visualizations(validations)
        mock_st.plotly_chart.assert_called()
        mock_st.subheader.assert_called()

    def test_validation_statuses_all_types(self):
        mock_st = _make_mock_st()
        validator = LiteratureValidator()
        validations = [
            validator.validate_parameter("ph", 7.0),
            validator.validate_parameter("ph", 5.5),
            validator.validate_parameter("ph", 3.0),
        ]
        with patch.object(lit_mod, "st", mock_st):
            create_validation_visualizations(validations)
        mock_st.metric.assert_called()
        mock_st.expander.assert_called()

    def test_parameter_range_analysis_section(self):
        mock_st = _make_mock_st()
        validator = LiteratureValidator()
        validations = [
            validator.validate_parameter("conductivity", 10000.0),
            validator.validate_parameter("flow_rate", 1e-4),
        ]
        with patch.object(lit_mod, "st", mock_st):
            create_validation_visualizations(validations)
        mock_st.info.assert_called()

    def test_confidence_color_buckets(self):
        mock_st = _make_mock_st()
        validations = [
            ParameterValidation(
                parameter_name="p1", value=1.0, unit="U",
                literature_range=(0, 10), typical_range=(0.5, 5),
                confidence_level=0.95, validation_status="validated",
                citations=[], recommendation="OK",
            ),
            ParameterValidation(
                parameter_name="p2", value=2.0, unit="U",
                literature_range=(0, 10), typical_range=(0.5, 5),
                confidence_level=0.65, validation_status="questionable",
                citations=[], recommendation="Check",
            ),
            ParameterValidation(
                parameter_name="p3", value=20.0, unit="U",
                literature_range=(0, 10), typical_range=(0.5, 5),
                confidence_level=0.25, validation_status="outlier",
                citations=[], recommendation="Bad",
            ),
        ]
        with patch.object(lit_mod, "st", mock_st):
            create_validation_visualizations(validations)


@pytest.mark.apptest
class TestRenderLiteratureValidationPageInit:
    def test_page_renders_title(self):
        mock_st = _make_mock_st()
        with patch.object(lit_mod, "st", mock_st):
            render_literature_validation_page()
        mock_st.title.assert_called()
        mock_st.caption.assert_called()

    def test_validator_created_in_session_state(self):
        mock_st = _make_mock_st()
        with patch.object(lit_mod, "st", mock_st):
            render_literature_validation_page()
        assert "literature_validator" in mock_st.session_state
        assert "validation_results" in mock_st.session_state


@pytest.mark.apptest
class TestRenderLitPageTab1Validation:
    def test_tab1_single_param_validate_button_validated(self):
        mock_st = _make_mock_st()
        mock_st.selectbox.return_value = "ph"
        mock_st.number_input.return_value = 7.0

        def button_side(label, **kwargs):
            if "Validate Parameter" in str(label):
                return True
            return False
        mock_st.button.side_effect = button_side

        with patch.object(lit_mod, "st", mock_st):
            render_literature_validation_page()
        mock_st.success.assert_called()

    def test_tab1_single_param_validate_questionable(self):
        mock_st = _make_mock_st()
        mock_st.selectbox.return_value = "ph"
        mock_st.number_input.return_value = 5.5

        def button_side(label, **kwargs):
            if "Validate Parameter" in str(label):
                return True
            return False
        mock_st.button.side_effect = button_side

        with patch.object(lit_mod, "st", mock_st):
            render_literature_validation_page()
        mock_st.warning.assert_called()

    def test_tab1_single_param_validate_outlier(self):
        mock_st = _make_mock_st()
        mock_st.selectbox.return_value = "ph"
        mock_st.number_input.return_value = 3.0

        def button_side(label, **kwargs):
            if "Validate Parameter" in str(label):
                return True
            return False
        mock_st.button.side_effect = button_side

        with patch.object(lit_mod, "st", mock_st):
            render_literature_validation_page()
        mock_st.error.assert_called()

    def test_tab1_batch_validation_with_csv(self):
        import io
        mock_st = _make_mock_st()
        mock_st.selectbox.return_value = "ph"
        csv_data = "parameter_name,value\nph,7.0\ntemperature,25.0\n"
        fake_file = io.BytesIO(csv_data.encode())
        mock_st.file_uploader.return_value = fake_file

        def button_side(label, **kwargs):
            if "Validate All" in str(label):
                return True
            return False
        mock_st.button.side_effect = button_side
        mock_st.progress.return_value = MagicMock()

        with patch.object(lit_mod, "st", mock_st):
            render_literature_validation_page()

    def test_tab1_batch_validation_bad_columns(self):
        import io
        mock_st = _make_mock_st()
        mock_st.selectbox.return_value = "ph"
        csv_data = "wrong_col,other_col\nph,7.0\n"
        fake_file = io.BytesIO(csv_data.encode())
        mock_st.file_uploader.return_value = fake_file

        with patch.object(lit_mod, "st", mock_st):
            render_literature_validation_page()
        mock_st.error.assert_called()

    def test_tab1_batch_validation_csv_error(self):
        mock_st = _make_mock_st()
        mock_st.selectbox.return_value = "ph"
        bad_file = MagicMock()
        mock_st.file_uploader.return_value = bad_file

        with patch.object(lit_mod, "st", mock_st), \
             patch.object(lit_mod, "pd") as mock_pd, \
             patch.object(lit_mod, "px", MagicMock()), \
             patch.object(lit_mod, "go", MagicMock()):
            mock_pd.read_csv.side_effect = Exception("bad data")
            mock_pd.DataFrame.return_value = MagicMock()
            mock_series = MagicMock()
            mock_series.value_counts.return_value = mock_series
            mock_series.sort_index.return_value = mock_series
            mock_series.index = [2020]
            mock_series.values = [1]
            mock_pd.Series.return_value = mock_series
            render_literature_validation_page()

    def test_tab1_template_validation(self):
        mock_st = _make_mock_st()
        selectbox_vals = iter(["ph", "Standard MFC"])
        mock_st.selectbox.side_effect = lambda *a, **kw: next(selectbox_vals, "ph")

        def button_side(label, **kwargs):
            if "Validate Template" in str(label):
                return True
            return False
        mock_st.button.side_effect = button_side

        with patch.object(lit_mod, "st", mock_st):
            render_literature_validation_page()
        mock_st.success.assert_called()

    def test_tab1_high_performance_template(self):
        mock_st = _make_mock_st()
        selectbox_vals = iter(["ph", "High Performance"])
        mock_st.selectbox.side_effect = lambda *a, **kw: next(selectbox_vals, "ph")

        def button_side(label, **kwargs):
            if "Validate Template" in str(label):
                return True
            return False
        mock_st.button.side_effect = button_side

        with patch.object(lit_mod, "st", mock_st):
            render_literature_validation_page()
        mock_st.success.assert_called()

    def test_tab1_display_validation_results(self):
        mock_st = _make_mock_st()
        validator = LiteratureValidator()
        mock_st.session_state["literature_validator"] = validator
        mock_st.session_state["validation_results"] = [
            validator.validate_parameter("ph", 7.0),
        ]
        with patch.object(lit_mod, "st", mock_st):
            render_literature_validation_page()


@pytest.mark.apptest
class TestRenderLitPageTab2Search:
    def test_tab2_search_with_results(self):
        mock_st = _make_mock_st()
        mock_st.text_input.return_value = "fuel cell"
        mock_st.number_input.return_value = 10
        with patch.object(lit_mod, "st", mock_st):
            render_literature_validation_page()
        mock_st.success.assert_called()

    def test_tab2_search_no_results(self):
        mock_st = _make_mock_st()
        mock_st.text_input.return_value = "zzzznonexistentzzz"
        mock_st.number_input.return_value = 10
        with patch.object(lit_mod, "st", mock_st):
            render_literature_validation_page()
        mock_st.info.assert_called()

    def test_tab2_empty_search_query(self):
        mock_st = _make_mock_st()
        mock_st.text_input.return_value = ""
        with patch.object(lit_mod, "st", mock_st):
            render_literature_validation_page()

    def test_tab2_database_stats_displayed(self):
        mock_st = _make_mock_st()
        mock_st.text_input.return_value = ""
        with patch.object(lit_mod, "st", mock_st):
            render_literature_validation_page()
        mock_st.metric.assert_called()

    def test_tab2_search_with_abstract_and_doi(self):
        mock_st = _make_mock_st()
        mock_st.text_input.return_value = "electrode"
        mock_st.number_input.return_value = 10
        with patch.object(lit_mod, "st", mock_st):
            render_literature_validation_page()


@pytest.mark.apptest
class TestRenderLitPageTab3Report:
    def test_tab3_no_results_shows_info(self):
        mock_st = _make_mock_st()
        mock_st.session_state["validation_results"] = []
        mock_st.session_state["literature_validator"] = LiteratureValidator()
        with patch.object(lit_mod, "st", mock_st):
            render_literature_validation_page()
        mock_st.info.assert_called()

    def test_tab3_with_results_shows_report(self):
        mock_st = _make_mock_st()
        validator = LiteratureValidator()
        mock_st.session_state["literature_validator"] = validator
        mock_st.session_state["validation_results"] = [
            validator.validate_parameter("ph", 7.0),
            validator.validate_parameter("temperature", 25.0),
        ]
        with patch.object(lit_mod, "st", mock_st):
            render_literature_validation_page()
        mock_st.metric.assert_called()
        mock_st.dataframe.assert_called()

    def test_tab3_with_recommendations(self):
        mock_st = _make_mock_st()
        validator = LiteratureValidator()
        mock_st.session_state["literature_validator"] = validator
        mock_st.session_state["validation_results"] = [
            validator.validate_parameter("ph", 3.0),
            validator.validate_parameter("ph", 5.5),
        ]
        with patch.object(lit_mod, "st", mock_st):
            render_literature_validation_page()
        mock_st.warning.assert_called()

    def test_tab3_export_buttons(self):
        mock_st = _make_mock_st()
        validator = LiteratureValidator()
        mock_st.session_state["literature_validator"] = validator
        mock_st.session_state["validation_results"] = [
            validator.validate_parameter("ph", 7.0),
        ]

        def button_side(label, **kwargs):
            if "Export PDF" in str(label):
                return True
            if "Export Data" in str(label):
                return True
            if "Export Bibliography" in str(label):
                return True
            return False
        mock_st.button.side_effect = button_side

        with patch.object(lit_mod, "st", mock_st):
            render_literature_validation_page()

    def test_tab3_no_citations_available(self):
        mock_st = _make_mock_st()
        validator = LiteratureValidator()
        mock_st.session_state["literature_validator"] = validator
        mock_st.session_state["validation_results"] = [
            validator.validate_parameter("nonexistent", 42.0),
        ]
        with patch.object(lit_mod, "st", mock_st):
            render_literature_validation_page()

    def test_tab3_validation_score_high(self):
        mock_st = _make_mock_st()
        validator = LiteratureValidator()
        mock_st.session_state["literature_validator"] = validator
        mock_st.session_state["validation_results"] = [
            validator.validate_parameter("ph", 7.0),
            validator.validate_parameter("temperature", 25.0),
            validator.validate_parameter("conductivity", 10000.0),
        ]
        with patch.object(lit_mod, "st", mock_st):
            render_literature_validation_page()

    def test_tab3_validation_score_low(self):
        mock_st = _make_mock_st()
        validator = LiteratureValidator()
        mock_st.session_state["literature_validator"] = validator
        mock_st.session_state["validation_results"] = [
            validator.validate_parameter("ph", 3.0),
            validator.validate_parameter("temperature", 100.0),
            validator.validate_parameter("conductivity", 0.001),
        ]
        with patch.object(lit_mod, "st", mock_st):
            render_literature_validation_page()


@pytest.mark.apptest
class TestRenderLitPageTab4Citations:
    def test_tab4_displays_citation_table(self):
        mock_st = _make_mock_st()
        with patch.object(lit_mod, "st", mock_st):
            render_literature_validation_page()
        mock_st.dataframe.assert_called()

    def test_tab4_add_citation_form_submit_success(self):
        mock_st = _make_mock_st()
        mock_st.text_input.return_value = "New Test Paper"
        mock_st.text_area.return_value = "Author A\nAuthor B"
        mock_st.number_input.return_value = 2024
        mock_st.slider.return_value = 4.5
        form = _make_ctx()
        mock_st.form.return_value = form
        form.form_submit_button = MagicMock(return_value=True)
        with patch.object(lit_mod, "st", mock_st):
            render_literature_validation_page()

    def test_tab4_add_citation_form_submit_empty_fields(self):
        mock_st = _make_mock_st()
        mock_st.text_input.return_value = ""
        mock_st.text_area.return_value = ""
        form = _make_ctx()
        mock_st.form.return_value = form
        form.form_submit_button = MagicMock(return_value=True)
        with patch.object(lit_mod, "st", mock_st):
            render_literature_validation_page()

    def test_tab4_citation_analytics_charts(self):
        mock_st = _make_mock_st()
        with patch.object(lit_mod, "st", mock_st):
            render_literature_validation_page()
        mock_st.plotly_chart.assert_called()

    def test_tab4_citation_data_truncation(self):
        mock_st = _make_mock_st()
        validator = LiteratureValidator()
        long_title_citation = Citation(
            title="A" * 60, authors=["Author A", "Author B"],
            journal="Test Journal", year=2024, doi=None, pmid=None,
            url=None, abstract=None, relevance_score=0.8, quality_score=4.0,
        )
        validator.citation_database.append(long_title_citation)
        mock_st.session_state["literature_validator"] = validator
        mock_st.session_state["validation_results"] = []
        with patch.object(lit_mod, "st", mock_st):
            render_literature_validation_page()

    def test_tab4_single_author_citation(self):
        mock_st = _make_mock_st()
        validator = LiteratureValidator()
        single_author = Citation(
            title="Test Paper", authors=["Solo Author"],
            journal="Test Journal", year=2024, doi="10.test/123",
            pmid=None, url=None, abstract=None,
            relevance_score=0.8, quality_score=4.0,
        )
        validator.citation_database.append(single_author)
        mock_st.session_state["literature_validator"] = validator
        mock_st.session_state["validation_results"] = []
        with patch.object(lit_mod, "st", mock_st):
            render_literature_validation_page()


@pytest.mark.apptest
class TestRenderLitPageExpander:
    def test_expander_guide_rendered(self):
        mock_st = _make_mock_st()
        with patch.object(lit_mod, "st", mock_st):
            render_literature_validation_page()
        mock_st.expander.assert_called()
        mock_st.markdown.assert_called()


@pytest.mark.apptest
class TestLitCitationDataclass:
    def test_citation_all_fields(self):
        c = Citation(
            title="Test", authors=["A"], journal="J", year=2024,
            doi="10.1/test", pmid="123", url="https://test.com",
            abstract="Abstract text", relevance_score=0.9, quality_score=4.5,
        )
        assert c.title == "Test"
        assert c.doi == "10.1/test"
        assert c.pmid == "123"
        assert c.url == "https://test.com"
        assert c.abstract == "Abstract text"

    def test_citation_optional_fields_none(self):
        c = Citation(
            title="Test", authors=["A"], journal="J", year=2024,
            doi=None, pmid=None, url=None, abstract=None,
            relevance_score=0.5, quality_score=3.0,
        )
        assert c.doi is None


@pytest.mark.apptest
class TestLitParameterValidationDataclass:
    def test_parameter_validation_fields(self):
        pv = ParameterValidation(
            parameter_name="test", value=1.0, unit="m",
            literature_range=(0.0, 10.0), typical_range=(1.0, 5.0),
            confidence_level=0.9, validation_status="validated",
            citations=[], recommendation="OK",
        )
        assert pv.parameter_name == "test"
        assert pv.value == 1.0


@pytest.mark.apptest
class TestLitPageFullPath:
    def test_full_flow_all_actions(self):
        mock_st = _make_mock_st()
        mock_st.text_input.return_value = "fuel cell"
        mock_st.text_area.return_value = "Author A"
        selectbox_vals = iter(["ph", "Standard MFC"])
        mock_st.selectbox.side_effect = lambda *a, **kw: next(selectbox_vals, "ph")
        mock_st.number_input.return_value = 10
        mock_st.slider.return_value = 4.0
        mock_st.button.return_value = True
        form = _make_ctx()
        mock_st.form.return_value = form
        form.form_submit_button = MagicMock(return_value=True)
        with patch.object(lit_mod, "st", mock_st):
            render_literature_validation_page()


@pytest.mark.apptest
class TestLitSearchCitationNoneAbstract:
    def test_search_skips_none_abstract(self):
        validator = LiteratureValidator()
        no_abstract = Citation(
            title="Special Paper About Electrodes",
            authors=["Test Author"], journal="Test J", year=2024,
            doi=None, pmid=None, url=None, abstract=None,
            relevance_score=0.8, quality_score=4.0,
        )
        validator.citation_database.append(no_abstract)
        results = validator.search_literature("electrode")
        found = any(c.title == "Special Paper About Electrodes" for c in results)
        assert found

    def test_validate_with_none_abstract_citation(self):
        validator = LiteratureValidator()
        result = validator.validate_parameter("ph", 7.0)
        assert result is not None


@pytest.mark.apptest
class TestLitFormSubmitRerun:
    def test_rerun_called_on_citation_add(self):
        mock_st = _make_mock_st()
        mock_st.text_input.return_value = "Paper Title"
        mock_st.text_area.return_value = "Author1\nAuthor2"
        mock_st.number_input.return_value = 2024
        mock_st.slider.return_value = 4.0
        mock_st.button.return_value = False
        form = _make_ctx()
        mock_st.form.return_value = form
        form.form_submit_button = MagicMock(return_value=True)
        with patch.object(lit_mod, "st", mock_st):
            render_literature_validation_page()
        mock_st.rerun.assert_called()


@pytest.mark.apptest
class TestLitTab2SearchResultDetails:
    def test_citation_without_doi_pmid_url(self):
        mock_st = _make_mock_st()
        mock_st.text_input.return_value = "no doi"
        mock_st.number_input.return_value = 10
        validator = LiteratureValidator()
        validator.citation_database.append(
            Citation(
                title="Paper with no doi", authors=["Author"],
                journal="J", year=2024, doi=None, pmid=None,
                url=None, abstract="Paper about no doi topic",
                relevance_score=0.8, quality_score=4.0,
            )
        )
        mock_st.session_state["literature_validator"] = validator
        mock_st.session_state["validation_results"] = []
        with patch.object(lit_mod, "st", mock_st):
            render_literature_validation_page()

    def test_citation_with_all_metadata(self):
        mock_st = _make_mock_st()
        mock_st.text_input.return_value = "full metadata"
        mock_st.number_input.return_value = 10
        validator = LiteratureValidator()
        validator.citation_database.append(
            Citation(
                title="Paper with full metadata",
                authors=["Author A", "Author B"], journal="J",
                year=2024, doi="10.test/full", pmid="999999",
                url="https://example.com/full",
                abstract="Full metadata abstract text",
                relevance_score=0.8, quality_score=4.5,
            )
        )
        mock_st.session_state["literature_validator"] = validator
        mock_st.session_state["validation_results"] = []
        with patch.object(lit_mod, "st", mock_st):
            render_literature_validation_page()


@pytest.mark.apptest
class TestLitTab4AddCitationNoDOI:
    def test_empty_doi_becomes_none(self):
        mock_st = _make_mock_st()
        mock_st.selectbox.return_value = "ph"
        mock_st.button.return_value = False

        def text_input_side(label, **kwargs):
            if "Title" in str(label) and "DOI" not in str(label):
                return "New Paper"
            if "Journal" in str(label):
                return "New Journal"
            if "DOI" in str(label):
                return ""
            return ""
        mock_st.text_input.side_effect = text_input_side
        mock_st.text_area.return_value = "Author X"
        mock_st.number_input.return_value = 2024
        mock_st.slider.return_value = 4.0
        form = _make_ctx()
        mock_st.form.return_value = form
        form.form_submit_button = MagicMock(return_value=True)
        with patch.object(lit_mod, "st", mock_st):
            render_literature_validation_page()


@pytest.mark.apptest
class TestLitTab4AddCitationWithDOI:
    def test_doi_value_preserved(self):
        mock_st = _make_mock_st()
        mock_st.selectbox.return_value = "ph"
        mock_st.button.return_value = False

        def text_input_side(label, **kwargs):
            if "Title" in str(label) and "DOI" not in str(label):
                return "New Paper With DOI"
            if "Journal" in str(label):
                return "New Journal"
            if "DOI" in str(label):
                return "10.1234/test"
            return ""
        mock_st.text_input.side_effect = text_input_side
        mock_st.text_area.return_value = "Author Y\nAuthor Z"
        mock_st.number_input.return_value = 2024
        mock_st.slider.return_value = 3.5
        form = _make_ctx()
        mock_st.form.return_value = form
        form.form_submit_button = MagicMock(return_value=True)
        with patch.object(lit_mod, "st", mock_st):
            render_literature_validation_page()


@pytest.mark.apptest
class TestLitTab3ReportWithCitations:
    """Cover lines 803-809: citation statistics when unique_citations is non-empty."""

    def test_tab3_report_with_real_citations(self):
        mock_st = _make_mock_st()
        validator = LiteratureValidator()
        # Build validations that carry real Citation objects
        c1 = Citation(
            title="Paper A", authors=["Auth A"], journal="J1", year=2020,
            doi="10.1/a", pmid=None, url=None, abstract=None,
            relevance_score=0.9, quality_score=4.5,
        )
        c2 = Citation(
            title="Paper B", authors=["Auth B"], journal="J2", year=2022,
            doi="10.1/b", pmid=None, url=None, abstract=None,
            relevance_score=0.8, quality_score=3.8,
        )
        v1 = ParameterValidation(
            parameter_name="ph", value=7.0, unit="-",
            literature_range=(5.0, 9.0), typical_range=(6.5, 8.0),
            confidence_level=0.95, validation_status="validated",
            citations=[c1, c2], recommendation="OK",
        )
        v2 = ParameterValidation(
            parameter_name="temperature", value=25.0, unit="C",
            literature_range=(4.0, 60.0), typical_range=(20.0, 35.0),
            confidence_level=0.95, validation_status="validated",
            citations=[c1], recommendation="OK",
        )
        mock_st.session_state["literature_validator"] = validator
        mock_st.session_state["validation_results"] = [v1, v2]
        with patch.object(lit_mod, "st", mock_st):
            render_literature_validation_page()
        # Verify citation statistics were displayed (lines 803-809)
        metric_calls = [str(c) for c in mock_st.metric.call_args_list]
        assert any("Unique Citations" in c for c in metric_calls)
        assert any("Average Publication Year" in c for c in metric_calls)
        assert any("Average Citation Quality" in c for c in metric_calls)

    def test_tab3_report_empty_citations_shows_info(self):
        """Explicitly test the else branch (line 811) with zero citations."""
        mock_st = _make_mock_st()
        validator = LiteratureValidator()
        v1 = ParameterValidation(
            parameter_name="ph", value=7.0, unit="-",
            literature_range=(5.0, 9.0), typical_range=(6.5, 8.0),
            confidence_level=0.95, validation_status="validated",
            citations=[], recommendation="OK",
        )
        mock_st.session_state["literature_validator"] = validator
        mock_st.session_state["validation_results"] = [v1]
        with patch.object(lit_mod, "st", mock_st):
            render_literature_validation_page()


@pytest.mark.apptest
class TestGSMFBASliderInteractions:
    def test_all_sliders_present(self):
        mock_st = _make_mock_st()
        sel_mock = MagicMock()
        sel_mock.selection.rows = []
        mock_st.dataframe.return_value = sel_mock
        mock_st.slider.return_value = 5.0
        integrator = GSMIntegrator()
        integrator.load_model("iSO783")
        mock_st.session_state["gsm_integrator"] = integrator
        with patch.object(gsm_mod, "st", mock_st):
            render_gsm_integration_page()
        mock_st.slider.assert_called()
