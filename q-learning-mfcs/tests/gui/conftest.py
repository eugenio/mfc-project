"""Pytest configuration for GUI tests.

This module provides fixtures that properly manage streamlit and related
module mocking for GUI tests without polluting the global module cache.

The key issue being solved: Many GUI test files mock streamlit, numpy, pandas,
plotly, etc. at module level by directly modifying sys.modules. This pollutes
the module cache and causes subsequent tests (that need real imports) to fail
with errors like "AttributeError: __version__" when matplotlib checks numpy.

Solution: Store and restore original modules around the test session.
"""

import sys
from unittest.mock import MagicMock

import pytest

# Store original modules that may be mocked by GUI tests
_original_modules: dict[str, object] = {}
_modules_to_track = [
    "streamlit",
    "numpy",
    "pandas",
    "plotly",
    "plotly.graph_objects",
    "plotly.express",
    "plotly.subplots",
    "altair",
]


def _save_original_modules() -> None:
    """Save references to original modules before they're mocked."""
    for mod_name in _modules_to_track:
        if mod_name in sys.modules:
            _original_modules[mod_name] = sys.modules[mod_name]


def _restore_original_modules() -> None:
    """Restore original modules after GUI tests complete."""
    for mod_name in _modules_to_track:
        if mod_name in _original_modules:
            sys.modules[mod_name] = _original_modules[mod_name]
        elif mod_name in sys.modules and isinstance(sys.modules[mod_name], MagicMock):
            # If a mock was inserted and there was no original, remove it
            del sys.modules[mod_name]


@pytest.fixture(scope="session", autouse=True)
def manage_gui_module_mocks() -> None:
    """Session-scoped fixture to manage module mocking for GUI tests.

    This fixture runs before any GUI tests and restores original modules
    after all GUI tests complete.
    """
    _save_original_modules()
    yield
    _restore_original_modules()


@pytest.fixture
def mock_streamlit() -> MagicMock:
    """Provide a properly configured streamlit mock for individual tests.

    Usage:
        def test_something(mock_streamlit):
            mock_streamlit.title.return_value = None
            # ... test code that imports gui modules
    """
    mock_st = MagicMock()
    mock_st.columns = MagicMock(return_value=[MagicMock() for _ in range(3)])
    mock_st.title = MagicMock()
    mock_st.caption = MagicMock()
    mock_st.success = MagicMock()
    mock_st.info = MagicMock()
    mock_st.warning = MagicMock()
    mock_st.error = MagicMock()
    mock_st.metric = MagicMock()
    mock_st.markdown = MagicMock()
    mock_st.write = MagicMock()
    mock_st.container = MagicMock()
    mock_st.empty = MagicMock()
    mock_st.form = MagicMock()
    mock_st.form_submit_button = MagicMock(return_value=False)
    mock_st.selectbox = MagicMock(return_value="Option1")
    mock_st.number_input = MagicMock(return_value=0.5)
    mock_st.checkbox = MagicMock(return_value=True)
    mock_st.button = MagicMock(return_value=False)
    mock_st.dataframe = MagicMock()
    mock_st.plotly_chart = MagicMock()
    mock_st.tabs = MagicMock(return_value=[MagicMock() for _ in range(5)])
    mock_st.header = MagicMock()
    mock_st.subheader = MagicMock()
    mock_st.text = MagicMock()
    mock_st.sidebar = MagicMock()
    mock_st.sidebar.title = MagicMock()
    mock_st.sidebar.radio = MagicMock(return_value="Dashboard")
    mock_st.sidebar.subheader = MagicMock()
    mock_st.sidebar.text = MagicMock()
    mock_st.session_state = {}
    return mock_st
