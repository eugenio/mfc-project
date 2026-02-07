"""AppTest tests for enhanced components.

Tests individual GUI components in isolation.
"""

import os
import tempfile
from typing import Generator

import pytest
from streamlit.testing.v1 import AppTest

# Persistent temp directory for test scripts
_TEMP_DIR = tempfile.mkdtemp(prefix="apptest_")
_COUNTER = 0


def _app_from_code(code: str) -> AppTest:
    """Create AppTest from inline code string.

    File persists for the process lifetime so
    AppTest can re-read it on subsequent .run() calls.
    """
    global _COUNTER
    _COUNTER += 1
    path = os.path.join(_TEMP_DIR, f"app_{_COUNTER}.py")
    with open(path, "w") as f:
        f.write(code)
    at = AppTest.from_file(path, default_timeout=10)
    at.run()
    return at


@pytest.mark.apptest
class TestScientificParameterWidgets:
    """Test scientific parameter input widgets."""

    def test_number_input_renders(self) -> None:
        """Number input widget renders correctly."""
        at = _app_from_code(
            'import streamlit as st\n'
            'st.number_input("Flow Rate", value=0.001)\n'
        )
        assert len(at.number_input) == 1
        assert at.number_input[0].value == 0.001

    def test_number_input_change(self) -> None:
        """Number input value can be changed."""
        at = _app_from_code(
            'import streamlit as st\n'
            'val = st.number_input("Temp", value=25.0)\n'
            'st.write(f"Value: {val}")\n'
        )
        at.number_input[0].set_value(37.0).run()
        assert at.number_input[0].value == 37.0

    def test_selectbox_renders(self) -> None:
        """Selectbox widget renders correctly."""
        at = _app_from_code(
            'import streamlit as st\n'
            'st.selectbox("Material",'
            ' ["Carbon", "Graphite", "Steel"])\n'
        )
        assert len(at.selectbox) == 1
        assert at.selectbox[0].value == "Carbon"

    def test_selectbox_change(self) -> None:
        """Selectbox value can be changed."""
        at = _app_from_code(
            'import streamlit as st\n'
            'st.selectbox("Material",'
            ' ["Carbon", "Graphite", "Steel"])\n'
        )
        at.selectbox[0].set_value("Graphite").run()
        assert at.selectbox[0].value == "Graphite"

    def test_checkbox_toggle(self) -> None:
        """Checkbox can be toggled."""
        at = _app_from_code(
            'import streamlit as st\n'
            'on = st.checkbox("GPU Acceleration")\n'
            'if on:\n'
            '    st.success("GPU enabled")\n'
        )
        assert at.checkbox[0].value is False
        at.checkbox[0].set_value(True).run()
        assert at.checkbox[0].value is True
        assert len(at.success) == 1  # type: ignore[unreachable]


@pytest.mark.apptest
class TestLayoutComponents:
    """Test layout components rendering."""

    def test_columns_layout(self) -> None:
        """Columns layout renders correctly."""
        at = _app_from_code(
            'import streamlit as st\n'
            'c1, c2 = st.columns(2)\n'
            'with c1:\n'
            '    st.metric("CPU", "45%")\n'
            'with c2:\n'
            '    st.metric("Memory", "60%")\n'
        )
        assert len(at.metric) == 2

    def test_tabs_layout(self) -> None:
        """Tab layout renders correctly."""
        at = _app_from_code(
            'import streamlit as st\n'
            't1, t2 = st.tabs(["Config", "Results"])\n'
            'with t1:\n'
            '    st.write("Configuration")\n'
            'with t2:\n'
            '    st.write("Results here")\n'
        )
        assert len(at.tabs) >= 1

    def test_expander_content(self) -> None:
        """Expander renders with content."""
        at = _app_from_code(
            'import streamlit as st\n'
            'with st.expander("Details"):\n'
            '    st.write("Hidden content")\n'
        )
        assert len(at.expander) == 1
