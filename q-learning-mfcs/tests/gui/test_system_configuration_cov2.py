"""Coverage tests for gui/pages/system_configuration.py -- lines 302-303.

The uncovered lines are inside the export button handler where
export_options is truthy: st.success and st.info calls.
"""
import importlib.util
import os
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

_SYS_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "src", "gui", "pages", "system_configuration.py",
)


def _make_col():
    col = MagicMock()
    col.__enter__ = MagicMock(return_value=col)
    col.__exit__ = MagicMock(return_value=False)
    return col


def _make_st():
    st = MagicMock()
    st.columns.side_effect = lambda *a, **kw: [
        _make_col()
        for _ in range(a[0] if isinstance(a[0], int) else len(a[0]))
    ]
    st.tabs.side_effect = lambda labels: [_make_col() for _ in labels]
    st.selectbox.return_value = "scientific"
    st.number_input.return_value = 3
    st.checkbox.return_value = True
    st.slider.return_value = 4
    st.text_input.return_value = "./exports"
    st.multiselect.return_value = ["Simulation Results"]
    st.session_state = MagicMock()
    st.color_picker.return_value = "#1f77b4"
    return st


@contextmanager
def _sys_config_env(st_mock):
    """Load system_configuration.py in an isolated module environment.

    Yields the loaded module.
    """
    mod_name = "_sys_config_isolated"
    saved = {}
    keys_to_mock = {
        "streamlit": st_mock,
        "numpy": MagicMock(),
        "pandas": MagicMock(),
    }
    for k, v in keys_to_mock.items():
        saved[k] = sys.modules.get(k)
        sys.modules[k] = v

    try:
        spec = importlib.util.spec_from_file_location(
            mod_name,
            _SYS_CONFIG_PATH,
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop(mod_name, None)
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


@pytest.mark.coverage_extra
class TestExportWithOptions:
    """Cover lines 302-303: export button with export_options truthy."""

    def test_export_selected_data_with_options(self):
        st = _make_st()

        # Use label-based button behavior: only "Export Selected Data" returns True
        def button_side_effect(*args, **kwargs):
            label = args[0] if args else kwargs.get("label", "")
            return "Export Selected Data" in label

        st.button.side_effect = button_side_effect

        st.multiselect.return_value = [
            "Simulation Results",
            "Parameter Configurations",
        ]

        with _sys_config_env(st) as mod:
            mod.render_export_management()

        success_calls = [str(c) for c in st.success.call_args_list]
        info_calls = [str(c) for c in st.info.call_args_list]
        assert any("Exported" in c for c in success_calls)
        assert any("Files saved" in c or "exports" in c for c in info_calls)

    def test_export_selected_data_empty_options(self):
        """Cover else branch: export_options is empty."""
        st = _make_st()

        def button_side_effect(*args, **kwargs):
            label = args[0] if args else kwargs.get("label", "")
            return "Export Selected Data" in label

        st.button.side_effect = button_side_effect

        st.multiselect.return_value = []

        with _sys_config_env(st) as mod:
            mod.render_export_management()

        warning_calls = [str(c) for c in st.warning.call_args_list]
        assert any("select data" in c.lower() for c in warning_calls)
