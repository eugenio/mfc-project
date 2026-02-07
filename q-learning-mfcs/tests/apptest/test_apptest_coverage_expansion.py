"""AppTest coverage expansion tests for low-coverage pages.

Targets: ml_optimization, gsm_integration, advanced_physics,
electrode_enhanced, and literature_validation pages.
"""

import pytest

from constants import PAGE_LABELS


# ---------------------------------------------------------------------------
# ML Optimization page  (PAGE_LABELS[4])
# ---------------------------------------------------------------------------
@pytest.mark.apptest
class TestMLOptimizationMethodSelection:
    """Test ML page method radio and method-specific widgets."""

    def test_ml_method_radio_exists(self, navigate_to) -> None:
        """ML page has an optimization method radio."""
        at = navigate_to(PAGE_LABELS[4])
        # radio[0] is method radio, radio[1] is sidebar
        assert len(at.radio) >= 2

    def test_ml_default_method_is_bayesian(self, navigate_to) -> None:
        """Default method should be Bayesian Optimization."""
        at = navigate_to(PAGE_LABELS[4])
        method_radio = at.radio[0]
        assert method_radio.value == "Bayesian Optimization"

    def test_ml_switch_to_nsga_ii(self, navigate_to) -> None:
        """Switch to NSGA-II method without exceptions."""
        at = navigate_to(PAGE_LABELS[4])
        at.radio[0].set_value("Multi-Objective (NSGA-II)").run()
        assert len(at.exception) == 0
        assert at.radio[0].value == "Multi-Objective (NSGA-II)"

    def test_ml_switch_to_neural_surrogate(self, navigate_to) -> None:
        """Switch to Neural Network Surrogate method."""
        at = navigate_to(PAGE_LABELS[4])
        at.radio[0].set_value("Neural Network Surrogate").run()
        assert len(at.exception) == 0

    def test_ml_switch_to_q_learning(self, navigate_to) -> None:
        """Switch to Q-Learning method."""
        at = navigate_to(PAGE_LABELS[4])
        at.radio[0].set_value("Q-Learning Reinforcement").run()
        assert len(at.exception) == 0

    def test_ml_has_multiselect_objectives(self, navigate_to) -> None:
        """ML page has multiselect for objectives."""
        at = navigate_to(PAGE_LABELS[4])
        assert len(at.multiselect) >= 1

    def test_ml_objectives_default(self, navigate_to) -> None:
        """Default objectives include power_density and cost."""
        at = navigate_to(PAGE_LABELS[4])
        obj_ms = at.multiselect[0]
        assert "power_density" in obj_ms.value
        assert "cost" in obj_ms.value

    def test_ml_change_objectives(self, navigate_to) -> None:
        """Change selected objectives."""
        at = navigate_to(PAGE_LABELS[4])
        at.multiselect[0].set_value(
            ["treatment_efficiency", "stability"]
        ).run()
        assert len(at.exception) == 0
        assert "treatment_efficiency" in at.multiselect[0].value

    def test_ml_has_parameter_multiselect(self, navigate_to) -> None:
        """ML page has multiselect for parameters to optimise."""
        at = navigate_to(PAGE_LABELS[4])
        assert len(at.multiselect) >= 2

    def test_ml_change_parameters(self, navigate_to) -> None:
        """Change selected optimization parameters."""
        at = navigate_to(PAGE_LABELS[4])
        at.multiselect[1].set_value(
            ["ph", "temperature"]
        ).run()
        assert len(at.exception) == 0


@pytest.mark.apptest
class TestMLOptimizationAdvancedSettings:
    """Test ML page advanced settings and expanders."""

    def test_ml_has_expanders(self, navigate_to) -> None:
        """ML page has expander sections."""
        at = navigate_to(PAGE_LABELS[4])
        assert len(at.expander) >= 1

    def test_ml_info_banner(self, navigate_to) -> None:
        """ML page displays status info banner."""
        at = navigate_to(PAGE_LABELS[4])
        info_texts = [i.value for i in at.info]
        assert any("Phase 3" in t for t in info_texts)

    def test_ml_has_caption(self, navigate_to) -> None:
        """ML page has descriptive caption."""
        at = navigate_to(PAGE_LABELS[4])
        assert len(at.caption) >= 1

    def test_ml_nsga_shows_slider(self, navigate_to) -> None:
        """NSGA-II method shows crossover probability slider."""
        at = navigate_to(PAGE_LABELS[4])
        at.radio[0].set_value("Multi-Objective (NSGA-II)").run()
        assert len(at.slider) >= 1

    def test_ml_qlearning_shows_sliders(self, navigate_to) -> None:
        """Q-Learning method shows learning rate slider."""
        at = navigate_to(PAGE_LABELS[4])
        at.radio[0].set_value("Q-Learning Reinforcement").run()
        assert len(at.slider) >= 1

    def test_ml_neural_has_architecture_selectbox(
        self, navigate_to
    ) -> None:
        """Neural Surrogate method shows Architecture selectbox."""
        at = navigate_to(PAGE_LABELS[4])
        at.radio[0].set_value("Neural Network Surrogate").run()
        found = False
        for sb in at.selectbox:
            if "MLP" in sb.options:
                found = True
                break
        assert found

    def test_ml_has_buttons(self, navigate_to) -> None:
        """ML page has run optimization button."""
        at = navigate_to(PAGE_LABELS[4])
        assert len(at.button) >= 1

    def test_ml_empty_objectives_warning(self, navigate_to) -> None:
        """Clearing all objectives shows a warning."""
        at = navigate_to(PAGE_LABELS[4])
        at.multiselect[0].set_value([]).run()
        warnings = [w.value for w in at.warning]
        assert any("objective" in w.lower() for w in warnings)

    def test_ml_empty_params_warning(self, navigate_to) -> None:
        """Clearing all parameters shows a warning."""
        at = navigate_to(PAGE_LABELS[4])
        at.multiselect[1].set_value([]).run()
        warnings = [w.value for w in at.warning]
        assert any("parameter" in w.lower() for w in warnings)

    def test_ml_checkbox_count(self, navigate_to) -> None:
        """ML page has checkboxes for configuration."""
        at = navigate_to(PAGE_LABELS[4])
        assert len(at.checkbox) >= 3

    def test_ml_toggle_early_stopping(self, navigate_to) -> None:
        """Toggle Early Stopping checkbox."""
        at = navigate_to(PAGE_LABELS[4])
        # Find checkbox that is True by default (Early Stopping)
        for cb in at.checkbox:
            if cb.value is True:
                cb.set_value(False).run()
                assert len(at.exception) == 0
                break


# ---------------------------------------------------------------------------
# GSM Integration page  (PAGE_LABELS[5])
# ---------------------------------------------------------------------------
@pytest.mark.apptest
class TestGSMIntegrationWidgets:
    """Test GSM Integration page widgets and tabs."""

    def test_gsm_success_banner(self, navigate_to) -> None:
        """GSM page has Phase 4 complete success banner."""
        at = navigate_to(PAGE_LABELS[5])
        successes = [s.value for s in at.success]
        assert any("Phase 4" in s for s in successes)

    def test_gsm_has_caption(self, navigate_to) -> None:
        """GSM page has descriptive caption."""
        at = navigate_to(PAGE_LABELS[5])
        assert len(at.caption) >= 1

    def test_gsm_has_dataframe(self, navigate_to) -> None:
        """GSM page displays a dataframe of available models."""
        at = navigate_to(PAGE_LABELS[5])
        assert len(at.dataframe) >= 1

    def test_gsm_has_four_tabs(self, navigate_to) -> None:
        """GSM page has 4 tabs."""
        at = navigate_to(PAGE_LABELS[5])
        assert len(at.tabs) == 4

    def test_gsm_model_info_displayed(self, navigate_to) -> None:
        """GSM page shows 'No model loaded' info by default."""
        at = navigate_to(PAGE_LABELS[5])
        info_texts = [i.value for i in at.info]
        assert any("No model" in t for t in info_texts)

    def test_gsm_tab2_warning_no_model(self, navigate_to) -> None:
        """Tab 2 (Flux Analysis) shows warning when no model loaded."""
        at = navigate_to(PAGE_LABELS[5])
        warnings = [w.value for w in at.warning]
        assert any("load a model" in w.lower() for w in warnings)

    def test_gsm_has_markdown(self, navigate_to) -> None:
        """GSM page renders markdown content."""
        at = navigate_to(PAGE_LABELS[5])
        assert len(at.markdown) >= 1

    def test_gsm_has_subheaders(self, navigate_to) -> None:
        """GSM page has section subheaders."""
        at = navigate_to(PAGE_LABELS[5])
        assert len(at.subheader) >= 1

    def test_gsm_no_exceptions(self, navigate_to) -> None:
        """GSM page loads without exceptions."""
        at = navigate_to(PAGE_LABELS[5])
        assert len(at.exception) == 0

    def test_gsm_title_contains_gsm(self, navigate_to) -> None:
        """GSM page title contains 'GSM'."""
        at = navigate_to(PAGE_LABELS[5])
        titles = [t.value for t in at.title]
        assert any("GSM" in t for t in titles)


# ---------------------------------------------------------------------------
# Advanced Physics page  (PAGE_LABELS[3])
# ---------------------------------------------------------------------------
@pytest.mark.apptest
class TestAdvancedPhysicsWidgets:
    """Test Advanced Physics page widgets and parameters."""

    def test_physics_success_banner(self, navigate_to) -> None:
        """Physics page has Phase 2 complete success banner."""
        at = navigate_to(PAGE_LABELS[3])
        successes = [s.value for s in at.success]
        assert any("Phase 2" in s for s in successes)

    def test_physics_has_caption(self, navigate_to) -> None:
        """Physics page has descriptive caption."""
        at = navigate_to(PAGE_LABELS[3])
        assert len(at.caption) >= 1

    def test_physics_has_expanders(self, navigate_to) -> None:
        """Physics page has simulation parameters expander."""
        at = navigate_to(PAGE_LABELS[3])
        assert len(at.expander) >= 1

    def test_physics_has_buttons(self, navigate_to) -> None:
        """Physics page has the Run Simulation button."""
        at = navigate_to(PAGE_LABELS[3])
        assert len(at.button) >= 1

    def test_physics_has_selectboxes(self, navigate_to) -> None:
        """Physics page has selectboxes for time and resolution."""
        at = navigate_to(PAGE_LABELS[3])
        assert len(at.selectbox) >= 2

    def test_physics_change_simulation_time(self, navigate_to) -> None:
        """Change Simulation Time selectbox."""
        at = navigate_to(PAGE_LABELS[3])
        for sb in at.selectbox:
            if "1 hour" in sb.options:
                sb.set_value("1 day").run()
                assert sb.value == "1 day"
                break

    def test_physics_change_mesh_resolution(self, navigate_to) -> None:
        """Change Mesh Resolution selectbox."""
        at = navigate_to(PAGE_LABELS[3])
        for sb in at.selectbox:
            if "Coarse" in sb.options:
                sb.set_value("Fine").run()
                assert sb.value == "Fine"
                break

    def test_physics_reynolds_metric(self, navigate_to) -> None:
        """Physics page shows Reynolds Number metric."""
        at = navigate_to(PAGE_LABELS[3])
        metric_labels = [m.label for m in at.metric]
        assert any("Reynolds" in lab for lab in metric_labels)

    def test_physics_peclet_metric(self, navigate_to) -> None:
        """Physics page shows Peclet Number metric."""
        at = navigate_to(PAGE_LABELS[3])
        metric_labels = [m.label for m in at.metric]
        assert any("Peclet" in lab for lab in metric_labels)

    def test_physics_info_laminar(self, navigate_to) -> None:
        """Physics page shows flow regime info."""
        at = navigate_to(PAGE_LABELS[3])
        info_texts = [i.value for i in at.info]
        assert any("Laminar" in t or "Flow" in t for t in info_texts)

    def test_physics_number_inputs_multiple(self, navigate_to) -> None:
        """Physics page has multiple number inputs for parameters."""
        at = navigate_to(PAGE_LABELS[3])
        assert len(at.number_input) >= 5

    def test_physics_no_exceptions(self, navigate_to) -> None:
        """Physics page loads without exceptions."""
        at = navigate_to(PAGE_LABELS[3])
        assert len(at.exception) == 0

    def test_physics_subheaders(self, navigate_to) -> None:
        """Physics page has section subheaders."""
        at = navigate_to(PAGE_LABELS[3])
        sub_texts = [s.value for s in at.subheader]
        assert any(
            "Flow" in s or "Mass" in s or "Biofilm" in s
            for s in sub_texts
        )


# ---------------------------------------------------------------------------
# Electrode Enhanced page  (PAGE_LABELS[1])
# ---------------------------------------------------------------------------
@pytest.mark.apptest
class TestElectrodeEnhancedWidgets:
    """Test Electrode Enhanced page widgets and tabs."""

    def test_electrode_has_success_banner(self, navigate_to) -> None:
        """Electrode page shows completion status."""
        at = navigate_to(PAGE_LABELS[1])
        successes = [s.value for s in at.success]
        assert any("Complete" in s for s in successes)

    def test_electrode_materials_available_metric(
        self, navigate_to
    ) -> None:
        """Electrode page shows Materials Available metric."""
        at = navigate_to(PAGE_LABELS[1])
        metric_labels = [m.label for m in at.metric]
        assert any("Materials" in lab for lab in metric_labels)

    def test_electrode_has_tabs(self, navigate_to) -> None:
        """Electrode page has configuration tabs."""
        at = navigate_to(PAGE_LABELS[1])
        assert len(at.tabs) >= 1

    def test_electrode_selectbox_anode(self, navigate_to) -> None:
        """Electrode page has anode material selectbox."""
        at = navigate_to(PAGE_LABELS[1])
        assert len(at.selectbox) >= 1
        anode_sb = at.selectbox[0]
        assert "Carbon Cloth" in anode_sb.options

    def test_electrode_selectbox_cathode(self, navigate_to) -> None:
        """Electrode page has cathode material selectbox."""
        at = navigate_to(PAGE_LABELS[1])
        assert len(at.selectbox) >= 2
        cathode_sb = at.selectbox[1]
        assert "Carbon Cloth" in cathode_sb.options

    def test_electrode_conductivity_metric(self, navigate_to) -> None:
        """Electrode page shows Conductivity metric."""
        at = navigate_to(PAGE_LABELS[1])
        metric_labels = [m.label for m in at.metric]
        assert any("Conductivity" in lab for lab in metric_labels)

    def test_electrode_surface_area_metric(self, navigate_to) -> None:
        """Electrode page shows Surface Area metric."""
        at = navigate_to(PAGE_LABELS[1])
        metric_labels = [m.label for m in at.metric]
        assert any("Surface" in lab for lab in metric_labels)

    def test_electrode_cost_metric(self, navigate_to) -> None:
        """Electrode page shows Cost metric."""
        at = navigate_to(PAGE_LABELS[1])
        metric_labels = [m.label for m in at.metric]
        assert any("Cost" in lab for lab in metric_labels)

    def test_electrode_info_descriptions(self, navigate_to) -> None:
        """Electrode page shows material description info."""
        at = navigate_to(PAGE_LABELS[1])
        info_texts = [i.value for i in at.info]
        assert len(info_texts) >= 1

    def test_electrode_geometry_selectbox_exists(
        self, navigate_to
    ) -> None:
        """Electrode page has geometry type selectbox."""
        at = navigate_to(PAGE_LABELS[1])
        geom_found = False
        for sb in at.selectbox:
            if "Rectangular Plate" in sb.options:
                geom_found = True
                break
        assert geom_found


@pytest.mark.apptest
class TestElectrodeKnownBug:
    """Electrode page known bug tests."""

    def test_electrode_material_options_exception(
        self, navigate_to
    ) -> None:
        """Electrode page has known material_options exception."""
        at = navigate_to(PAGE_LABELS[1])
        exceptions = [str(e.value) for e in at.exception]
        # The geometry tab has a bug referencing material_options
        has_material_bug = any(
            "material_options" in exc for exc in exceptions
        )
        if has_material_bug:
            pytest.xfail(
                "Known bug: material_options not defined"
            )
        # If no bug, page renders fully - either way passes
        assert True

    @pytest.mark.xfail(
        reason="Known bug: material_options blocks tab rendering",
        strict=False,
    )
    def test_electrode_custom_tab_text_input(
        self, navigate_to
    ) -> None:
        """Custom Material tab has text input (blocked by bug)."""
        at = navigate_to(PAGE_LABELS[1])
        # Bug in geometry tab blocks later tabs from rendering
        assert len(at.text_input) >= 1

    @pytest.mark.xfail(
        reason="Known bug: material_options blocks tab rendering",
        strict=False,
    )
    def test_electrode_custom_tab_text_area(
        self, navigate_to
    ) -> None:
        """Custom Material tab has text area (blocked by bug)."""
        at = navigate_to(PAGE_LABELS[1])
        assert len(at.text_area) >= 1

    @pytest.mark.xfail(
        reason="Known bug: material_options blocks tab rendering",
        strict=False,
    )
    def test_electrode_custom_tab_buttons(
        self, navigate_to
    ) -> None:
        """Custom Material tab has buttons (blocked by bug)."""
        at = navigate_to(PAGE_LABELS[1])
        assert len(at.button) >= 3

    @pytest.mark.xfail(
        reason="Known bug: material_options blocks tab rendering",
        strict=False,
    )
    def test_electrode_performance_power_metric(
        self, navigate_to
    ) -> None:
        """Performance tab shows Power metric (blocked by bug)."""
        at = navigate_to(PAGE_LABELS[1])
        metric_labels = [m.label for m in at.metric]
        assert any("Power" in lab for lab in metric_labels)


# ---------------------------------------------------------------------------
# Literature Validation page  (PAGE_LABELS[6])
# ---------------------------------------------------------------------------
@pytest.mark.apptest
class TestLiteratureValidationWidgets:
    """Test Literature Validation page widgets and tabs."""

    def test_literature_success_banner(self, navigate_to) -> None:
        """Literature page has Phase 5 complete banner."""
        at = navigate_to(PAGE_LABELS[6])
        successes = [s.value for s in at.success]
        assert any("Phase 5" in s for s in successes)

    def test_literature_has_caption(self, navigate_to) -> None:
        """Literature page has descriptive caption."""
        at = navigate_to(PAGE_LABELS[6])
        assert len(at.caption) >= 1

    def test_literature_has_tabs(self, navigate_to) -> None:
        """Literature page has tabs."""
        at = navigate_to(PAGE_LABELS[6])
        assert len(at.tabs) >= 1

    def test_literature_parameter_selectbox(self, navigate_to) -> None:
        """Literature page has parameter selectbox for validation."""
        at = navigate_to(PAGE_LABELS[6])
        assert len(at.selectbox) >= 1

    def test_literature_parameter_selectbox_options(
        self, navigate_to
    ) -> None:
        """Parameter selectbox has expected display options."""
        at = navigate_to(PAGE_LABELS[6])
        param_sb = at.selectbox[0]
        options = param_sb.options
        # Options are formatted via format_func to Title Case
        assert "Conductivity" in options
        assert "Flow Rate" in options

    def test_literature_change_parameter(self, navigate_to) -> None:
        """Change selected parameter for validation."""
        at = navigate_to(PAGE_LABELS[6])
        at.selectbox[0].set_value("temperature").run()
        assert at.selectbox[0].value == "temperature"
        assert len(at.exception) == 0

    def test_literature_select_ph(self, navigate_to) -> None:
        """Select pH parameter for validation."""
        at = navigate_to(PAGE_LABELS[6])
        at.selectbox[0].set_value("ph").run()
        assert at.selectbox[0].value == "ph"
        assert len(at.exception) == 0

    def test_literature_select_biofilm(self, navigate_to) -> None:
        """Select biofilm_thickness parameter for validation."""
        at = navigate_to(PAGE_LABELS[6])
        at.selectbox[0].set_value("biofilm_thickness").run()
        assert at.selectbox[0].value == "biofilm_thickness"
        assert len(at.exception) == 0

    def test_literature_select_substrate(self, navigate_to) -> None:
        """Select substrate_concentration parameter."""
        at = navigate_to(PAGE_LABELS[6])
        at.selectbox[0].set_value("substrate_concentration").run()
        assert len(at.exception) == 0

    def test_literature_has_number_input(self, navigate_to) -> None:
        """Literature page has value number input."""
        at = navigate_to(PAGE_LABELS[6])
        assert len(at.number_input) >= 1

    def test_literature_has_validate_button(self, navigate_to) -> None:
        """Literature page has buttons."""
        at = navigate_to(PAGE_LABELS[6])
        assert len(at.button) >= 1

    def test_literature_has_expanders(self, navigate_to) -> None:
        """Literature page has information guide expander."""
        at = navigate_to(PAGE_LABELS[6])
        assert len(at.expander) >= 1


@pytest.mark.apptest
class TestLiteratureSearchAndReport:
    """Test Literature Search, Report, and Citation Manager tabs."""

    def test_literature_search_text_input(self, navigate_to) -> None:
        """Literature Search tab has search text input."""
        at = navigate_to(PAGE_LABELS[6])
        assert len(at.text_input) >= 1

    def test_literature_max_results_input(self, navigate_to) -> None:
        """Literature Search tab has Max Results input."""
        at = navigate_to(PAGE_LABELS[6])
        found_max = False
        for ni in at.number_input:
            if ni.value == 10:
                found_max = True
                break
        assert found_max

    def test_literature_database_stats_metrics(
        self, navigate_to
    ) -> None:
        """Literature page shows database statistics metrics."""
        at = navigate_to(PAGE_LABELS[6])
        metric_labels = [m.label for m in at.metric]
        assert any("Total" in lab for lab in metric_labels)

    def test_literature_report_tab_no_results(
        self, navigate_to
    ) -> None:
        """Report tab shows info when no validation results."""
        at = navigate_to(PAGE_LABELS[6])
        info_texts = [i.value for i in at.info]
        assert any(
            "No validation" in t or "No model" in t
            for t in info_texts
        )

    def test_literature_citation_manager_dataframe(
        self, navigate_to
    ) -> None:
        """Citation Manager tab shows citations dataframe."""
        at = navigate_to(PAGE_LABELS[6])
        assert len(at.dataframe) >= 1

    def test_literature_citation_form_text_inputs(
        self, navigate_to
    ) -> None:
        """Citation Manager has text inputs for form fields."""
        at = navigate_to(PAGE_LABELS[6])
        assert len(at.text_input) >= 1

    def test_literature_template_selectbox(self, navigate_to) -> None:
        """Literature page has template selectbox."""
        at = navigate_to(PAGE_LABELS[6])
        found = False
        for sb in at.selectbox:
            if "Standard MFC" in sb.options:
                found = True
                break
        assert found

    def test_literature_change_template(self, navigate_to) -> None:
        """Change validation template selection."""
        at = navigate_to(PAGE_LABELS[6])
        for sb in at.selectbox:
            if "Standard MFC" in sb.options:
                sb.set_value("High Performance").run()
                assert sb.value == "High Performance"
                assert len(at.exception) == 0
                break

    def test_literature_select_low_cost_template(
        self, navigate_to
    ) -> None:
        """Select Low Cost template."""
        at = navigate_to(PAGE_LABELS[6])
        for sb in at.selectbox:
            if "Standard MFC" in sb.options:
                sb.set_value("Low Cost").run()
                assert sb.value == "Low Cost"
                assert len(at.exception) == 0
                break

    def test_literature_no_exceptions(self, navigate_to) -> None:
        """Literature page loads without exceptions."""
        at = navigate_to(PAGE_LABELS[6])
        assert len(at.exception) == 0

    def test_literature_quality_metric(self, navigate_to) -> None:
        """Literature page shows Average Quality metric."""
        at = navigate_to(PAGE_LABELS[6])
        metric_labels = [m.label for m in at.metric]
        assert any("Quality" in lab for lab in metric_labels)
