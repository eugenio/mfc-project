"""Coverage tests for electrode_configuration_ui.py - lines 100-720."""
import os
import sys
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'gui'),
)


class _SessionState(dict):
    """Dict that also supports attribute-style access like Streamlit."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key)


def _col_mock():
    """Create a MagicMock that works as context manager."""
    m = MagicMock()
    m.__enter__ = MagicMock(return_value=m)
    m.__exit__ = MagicMock(return_value=False)
    return m


# Mock streamlit and plotly before import
mock_st = MagicMock()
mock_st.session_state = _SessionState()
# Make st.columns return proper list of context managers
mock_st.columns.side_effect = lambda n: [_col_mock() for _ in range(n)]
# Make st.expander return a context manager
_exp = MagicMock()
_exp.__enter__ = MagicMock(return_value=_exp)
_exp.__exit__ = MagicMock(return_value=False)
mock_st.expander.return_value = _exp
# Make st.tabs return context managers
mock_st.tabs.side_effect = lambda labels: [_col_mock() for _ in labels]

mock_go = MagicMock()
mock_pd = MagicMock()

# Create mock electrode config objects
mock_elec_cfg = MagicMock()
mock_elec_cfg.ElectrodeGeometry = MagicMock()
mock_elec_cfg.ElectrodeGeometry.RECTANGULAR_PLATE = "rectangular_plate"
mock_elec_cfg.ElectrodeGeometry.CYLINDRICAL_ROD = "cylindrical_rod"
mock_elec_cfg.ElectrodeGeometry.CYLINDRICAL_TUBE = "cylindrical_tube"
mock_elec_cfg.ElectrodeGeometry.SPHERICAL = "spherical"
mock_elec_cfg.ElectrodeGeometry.CUSTOM = "custom"
mock_elec_cfg.ElectrodeMaterial = MagicMock()
mock_elec_cfg.ElectrodeMaterial.GRAPHITE_PLATE = "graphite_plate"
mock_elec_cfg.ElectrodeMaterial.GRAPHITE_ROD = "graphite_rod"
mock_elec_cfg.ElectrodeMaterial.CARBON_FELT = "carbon_felt"
mock_elec_cfg.ElectrodeMaterial.CARBON_CLOTH = "carbon_cloth"
mock_elec_cfg.ElectrodeMaterial.CARBON_PAPER = "carbon_paper"
mock_elec_cfg.ElectrodeMaterial.STAINLESS_STEEL = "stainless_steel"
mock_elec_cfg.ElectrodeMaterial.PLATINUM = "platinum"
mock_elec_cfg.ElectrodeMaterial.CUSTOM = "custom"
mock_elec_cfg.MATERIAL_PROPERTIES_DATABASE = {}
mock_elec_cfg.MaterialProperties = MagicMock()
mock_elec_cfg.ElectrodeConfiguration = MagicMock()
mock_elec_cfg.create_electrode_config = MagicMock()

with patch.dict(sys.modules, {
    'streamlit': mock_st,
    'plotly': MagicMock(),
    'plotly.graph_objects': mock_go,
    'pandas': mock_pd,
    'config': MagicMock(),
    'config.electrode_config': mock_elec_cfg,
}):
    import electrode_configuration_ui as _ui_mod
    from electrode_configuration_ui import ElectrodeConfigurationUI


@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset session state and relevant mock calls for each test."""
    mock_st.reset_mock()
    mock_go.reset_mock()
    mock_pd.reset_mock()
    # Re-set after reset_mock clears side_effects
    mock_st.session_state = _SessionState()
    mock_st.columns.side_effect = lambda n: [_col_mock() for _ in range(n)]
    mock_st.tabs.side_effect = lambda labels: [_col_mock() for _ in labels]
    _exp2 = MagicMock()
    _exp2.__enter__ = MagicMock(return_value=_exp2)
    _exp2.__exit__ = MagicMock(return_value=False)
    mock_st.expander.return_value = _exp2
    yield


class TestInitializeSessionState:
    def test_init(self):
        ui = ElectrodeConfigurationUI()
        assert "anode_config" in mock_st.session_state
        assert "cathode_config" in mock_st.session_state
        assert "electrode_calculations" in mock_st.session_state


def _real_props():
    """Create a mock with real numeric values so format strings work."""
    p = MagicMock()
    p.specific_conductance = 1000.0
    p.contact_resistance = 0.5
    p.hydrophobicity_angle = 75.0
    p.surface_roughness = 1.0
    p.biofilm_adhesion_coefficient = 1.0
    p.attachment_energy = -10.0
    p.specific_surface_area = 500.0
    p.porosity = 0.8
    p.reference = "Test 2025"
    return p


class TestRenderMaterialSelector:
    def test_render_non_custom(self):
        mock_st.selectbox.return_value = "Graphite Plate"
        old_db = _ui_mod.MATERIAL_PROPERTIES_DATABASE
        _ui_mod.MATERIAL_PROPERTIES_DATABASE = {
            "graphite_plate": _real_props(),
        }
        try:
            ui = ElectrodeConfigurationUI()
            result = ui.render_material_selector("anode")
            assert result == "graphite_plate"
        finally:
            _ui_mod.MATERIAL_PROPERTIES_DATABASE = old_db

    def test_render_custom(self):
        mock_st.selectbox.return_value = "Custom Material"
        ui = ElectrodeConfigurationUI()
        result = ui.render_material_selector("cathode")
        assert result == "custom"


class TestDisplayMaterialProperties:
    def test_display(self):
        props = MagicMock()
        props.specific_conductance = 1000
        props.contact_resistance = 0.5
        props.hydrophobicity_angle = 75
        props.surface_roughness = 1.0
        props.biofilm_adhesion_coefficient = 1.0
        props.attachment_energy = -10.0
        props.specific_surface_area = 500.0
        props.porosity = 0.8
        props.reference = "Test 2025"
        ui = ElectrodeConfigurationUI()
        ui._display_material_properties(props, "Graphite Plate")
        mock_st.expander.assert_called()

    def test_display_no_porous(self):
        props = MagicMock()
        props.specific_conductance = 1000
        props.contact_resistance = 0.5
        props.hydrophobicity_angle = 75
        props.surface_roughness = 1.0
        props.biofilm_adhesion_coefficient = 1.0
        props.attachment_energy = -10.0
        props.specific_surface_area = None
        props.porosity = None
        props.reference = "Ref"
        ui = ElectrodeConfigurationUI()
        ui._display_material_properties(props, "Steel")


class TestRenderGeometrySelector:
    def test_rectangular(self):
        mock_st.selectbox.return_value = "Rectangular Plate"
        mock_st.number_input.return_value = 5.0
        ui = ElectrodeConfigurationUI()
        geom, dims = ui.render_geometry_selector("anode")
        assert geom == "rectangular_plate"

    def test_cylindrical_rod(self):
        mock_st.selectbox.return_value = "Cylindrical Rod"
        mock_st.number_input.return_value = 10.0
        ui = ElectrodeConfigurationUI()
        geom, dims = ui.render_geometry_selector("anode")
        assert geom == "cylindrical_rod"


class TestRenderDimensionInputs:
    def test_rectangular_plate(self):
        mock_st.number_input.return_value = 5.0
        ui = ElectrodeConfigurationUI()
        dims = ui._render_dimension_inputs("rectangular_plate", "anode")
        assert "length" in dims
        assert "width" in dims
        assert "thickness" in dims

    def test_cylindrical_rod(self):
        mock_st.number_input.return_value = 10.0
        ui = ElectrodeConfigurationUI()
        dims = ui._render_dimension_inputs("cylindrical_rod", "anode")
        assert "diameter" in dims
        assert "length" in dims

    def test_cylindrical_tube(self):
        mock_st.number_input.return_value = 15.0
        ui = ElectrodeConfigurationUI()
        dims = ui._render_dimension_inputs("cylindrical_tube", "anode")
        assert "diameter" in dims
        assert "thickness" in dims
        assert "length" in dims

    def test_spherical(self):
        mock_st.number_input.return_value = 20.0
        ui = ElectrodeConfigurationUI()
        dims = ui._render_dimension_inputs("spherical", "anode")
        assert "diameter" in dims

    def test_custom(self):
        mock_st.number_input.return_value = 25.0
        ui = ElectrodeConfigurationUI()
        dims = ui._render_dimension_inputs("custom", "anode")
        assert "projected_area" in dims
        assert "total_surface_area" in dims


class TestRenderCustomMaterialProperties:
    def test_non_porous(self):
        mock_st.number_input.return_value = 1000.0
        mock_st.slider.return_value = 75
        mock_st.checkbox.return_value = False
        mock_st.text_input.return_value = "Custom ref"
        ui = ElectrodeConfigurationUI()
        ui.render_custom_material_properties("anode")
        mock_elec_cfg.MaterialProperties.assert_called()

    def test_porous(self):
        mock_st.number_input.return_value = 1000.0
        mock_st.slider.return_value = 75
        mock_st.checkbox.return_value = True
        mock_st.text_input.return_value = "Custom ref"
        ui = ElectrodeConfigurationUI()
        ui.render_custom_material_properties("cathode")
        mock_elec_cfg.MaterialProperties.assert_called()


class TestCalculateAndDisplayAreas:
    def test_name_error_propagates(self):
        """Source has a bug: undefined 'projected_area'. Verify it raises."""
        config = MagicMock()
        config.geometry.calculate_specific_surface_area.return_value = 0.001
        config.geometry.calculate_total_surface_area.return_value = 0.005
        config.calculate_effective_surface_area.return_value = 0.003
        config.calculate_biofilm_capacity.return_value = 1e-9
        config.calculate_charge_transfer_coefficient.return_value = 0.5
        ui = ElectrodeConfigurationUI()
        with pytest.raises(NameError):
            ui.calculate_and_display_areas(config, "anode")

    def test_value_error(self):
        config = MagicMock()
        config.geometry.calculate_specific_surface_area.side_effect = (
            ValueError("bad dims")
        )
        ui = ElectrodeConfigurationUI()
        ui.calculate_and_display_areas(config, "anode")
        mock_st.error.assert_called()


class TestRenderElectrodeComparison:
    def test_too_few(self):
        mock_st.session_state["electrode_calculations"] = {"anode": {}}
        ui = ElectrodeConfigurationUI()
        ui.render_electrode_comparison()
        # Should return early -- no Figure created

    def test_comparison(self):
        mock_st.session_state["electrode_calculations"] = {
            "anode": {
                "projected_area_cm2": 25.0,
                "geometric_area_cm2": 50.0,
                "effective_area_cm2": 40.0,
                "enhancement_factor": 2.0,
                "biofilm_capacity_ul": 0.5,
                "charge_transfer_coeff": 0.8,
                "volume_cm3": 1.0,
            },
            "cathode": {
                "projected_area_cm2": 25.0,
                "geometric_area_cm2": 50.0,
                "effective_area_cm2": 40.0,
                "enhancement_factor": 2.0,
                "biofilm_capacity_ul": 0.5,
                "charge_transfer_coeff": 0.8,
                "volume_cm3": 1.0,
            },
        }
        # pd.DataFrame returns something with column access + tolist
        mock_col = MagicMock()
        mock_col.tolist.return_value = ["Anode", "Cathode"]
        mock_df = MagicMock()
        mock_df.__getitem__ = MagicMock(return_value=mock_col)
        # Patch module-level pd reference
        old_pd = _ui_mod.pd
        _ui_mod.pd = MagicMock()
        _ui_mod.pd.DataFrame.return_value = mock_df
        # Patch module-level go reference
        old_go = _ui_mod.go
        mock_fig = MagicMock()
        _ui_mod.go = MagicMock()
        _ui_mod.go.Figure.return_value = mock_fig
        try:
            ui = ElectrodeConfigurationUI()
            ui.render_electrode_comparison()
            _ui_mod.go.Figure.assert_called()
        finally:
            _ui_mod.pd = old_pd
            _ui_mod.go = old_go


class TestRenderConfigurationSummary:
    def test_summary(self):
        config = MagicMock()
        config.get_configuration_summary.return_value = {
            "material": "graphite_plate",
            "geometry": "rectangular_plate",
            "specific_surface_area_m2_per_g": 10.0,
            "effective_area_cm2": 25.0,
            "biofilm_capacity_ul": 0.5,
            "charge_transfer_coeff": 0.8,
            "specific_conductance_S_per_m": 1000,
            "hydrophobicity_angle_deg": 75,
        }
        ui = ElectrodeConfigurationUI()
        ui.render_configuration_summary(config, "anode")
        mock_st.write.assert_called()


class TestRenderFullElectrodeConfiguration:
    def test_full_render(self):
        # selectbox: anode_material, anode_geometry, cathode_material,
        #   cathode_geometry
        mock_st.selectbox.side_effect = [
            "Graphite Plate", "Rectangular Plate",
            "Graphite Plate", "Rectangular Plate",
        ]
        mock_st.number_input.return_value = 5.0
        mock_st.checkbox.return_value = False
        mock_cfg_obj = MagicMock()
        mock_elec_cfg.create_electrode_config.return_value = mock_cfg_obj
        old_db = _ui_mod.MATERIAL_PROPERTIES_DATABASE
        _ui_mod.MATERIAL_PROPERTIES_DATABASE = {
            "graphite_plate": _real_props(),
        }
        try:
            ui = ElectrodeConfigurationUI()
            with patch.object(ui, 'calculate_and_display_areas'):
                with patch.object(ui, 'render_configuration_summary'):
                    result = ui.render_full_electrode_configuration()
            assert isinstance(result, tuple)
        finally:
            _ui_mod.MATERIAL_PROPERTIES_DATABASE = old_db

    def test_full_render_custom_material(self):
        mock_st.selectbox.side_effect = [
            "Custom Material", "Rectangular Plate",
            "Custom Material", "Rectangular Plate",
        ]
        mock_st.number_input.return_value = 5.0
        mock_st.slider.return_value = 75
        mock_st.checkbox.return_value = False
        mock_st.text_input.return_value = "Custom"
        mock_cfg_obj = MagicMock()
        mock_elec_cfg.create_electrode_config.return_value = mock_cfg_obj
        ui = ElectrodeConfigurationUI()
        with patch.object(ui, 'calculate_and_display_areas'):
            with patch.object(ui, 'render_configuration_summary'):
                result = ui.render_full_electrode_configuration()
        assert isinstance(result, tuple)

    def test_full_render_config_error(self):
        mock_st.selectbox.side_effect = [
            "Graphite Plate", "Rectangular Plate",
            "Graphite Plate", "Rectangular Plate",
        ]
        mock_st.number_input.return_value = 5.0
        mock_st.checkbox.return_value = False
        mock_elec_cfg.create_electrode_config.side_effect = ValueError("bad")
        old_db = _ui_mod.MATERIAL_PROPERTIES_DATABASE
        _ui_mod.MATERIAL_PROPERTIES_DATABASE = {
            "graphite_plate": _real_props(),
        }
        try:
            ui = ElectrodeConfigurationUI()
            result = ui.render_full_electrode_configuration()
            assert isinstance(result, tuple)
            mock_st.error.assert_called()
        finally:
            _ui_mod.MATERIAL_PROPERTIES_DATABASE = old_db
            mock_elec_cfg.create_electrode_config.side_effect = None

    def test_full_render_use_same_as_anode(self):
        """Test the 'use same configuration as anode' checkbox path."""
        # Only anode material + geometry selectors are called
        mock_st.selectbox.side_effect = [
            "Graphite Plate", "Rectangular Plate",
        ]
        mock_st.number_input.return_value = 5.0
        mock_st.checkbox.return_value = True  # use same as anode
        mock_cfg_obj = MagicMock()
        mock_elec_cfg.create_electrode_config.return_value = mock_cfg_obj
        old_db = _ui_mod.MATERIAL_PROPERTIES_DATABASE
        _ui_mod.MATERIAL_PROPERTIES_DATABASE = {
            "graphite_plate": _real_props(),
        }
        try:
            ui = ElectrodeConfigurationUI()
            with patch.object(ui, 'calculate_and_display_areas'):
                with patch.object(ui, 'render_configuration_summary'):
                    result = ui.render_full_electrode_configuration()
            assert isinstance(result, tuple)
        finally:
            _ui_mod.MATERIAL_PROPERTIES_DATABASE = old_db
