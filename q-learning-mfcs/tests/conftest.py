"""Root test conftest — protects real modules from GUI test mock pollution.

Several files in tests/gui/ replace plotly, streamlit, and psutil in
sys.modules with MagicMock objects at *module level* (collection time).
This pollutes the module cache for the entire test session.

This conftest captures originals at load time (before gui test files are
collected) and restores them before every non-gui test runs.
"""

import contextlib
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

# Eagerly import modules that AppTest and other tests need so they are
# captured in _original_modules before gui test files can pollute them.
for _eager in ("streamlit", "plotly", "psutil"):
    with contextlib.suppress(ImportError):
        __import__(_eager)

# Explicit modules that gui tests replace with MagicMock.
_EXPLICIT_MODULES = [
    "streamlit",
    "plotly",
    "plotly.graph_objects",
    "plotly.express",
    "plotly.subplots",
    "matplotlib.pyplot",
    "psutil",
]

# Prefixes whose sub-modules are bulk-saved (gui tests may import
# streamlit or plotly sub-modules that cache mock references).
_PREFIX_PATTERNS = ["streamlit.", "plotly."]

# Save originals NOW — conftest at tests/ root loads before any
# subdirectory conftest or test file, so modules are still real.
_original_modules: dict[str, object] = {}
for _mod in _EXPLICIT_MODULES:
    if _mod in sys.modules:
        _original_modules[_mod] = sys.modules[_mod]
for _key in list(sys.modules):
    if any(_key.startswith(p) for p in _PREFIX_PATTERNS):
        _original_modules[_key] = sys.modules[_key]

_GUI_TEST_DIR = str(Path(__file__).resolve().parent / "gui")

# Snapshot of ALL modules as they exist after collection (gui tests have
# already injected their mocks).  Taken once before the first test runs.
_post_collection_snapshot: dict[str, object] = {}
_snapshot_taken = False


def _restore_gui_state() -> None:
    """Restore post-collection mocked state for gui tests."""
    for mod_name, mod_obj in _post_collection_snapshot.items():
        sys.modules[mod_name] = mod_obj  # type: ignore[assignment]


def _restore_real_state() -> None:
    """Restore real library modules and clear gui caches for non-gui tests."""
    for mod_name, mod_obj in _original_modules.items():
        sys.modules[mod_name] = mod_obj  # type: ignore[assignment]

    for key in list(sys.modules):
        if key.startswith("gui.") or key == "gui":
            del sys.modules[key]
        elif any(key.startswith(p) for p in _PREFIX_PATTERNS):
            if key not in _original_modules and isinstance(
                sys.modules[key],
                MagicMock,
            ):
                del sys.modules[key]
        elif (
            key in _EXPLICIT_MODULES
            and key not in _original_modules
            and isinstance(sys.modules[key], MagicMock)
        ):
            del sys.modules[key]


def pytest_runtest_setup(item: Any) -> None:  # noqa: ANN401
    """Restore real or mocked modules depending on test location."""
    global _snapshot_taken  # noqa: PLW0603
    item_path = str(Path(item.fspath).resolve())

    # Take a one-time snapshot right before the first test executes.
    # At this point collection is complete and gui module-level mocks are
    # active.  We save everything so gui tests can be restored later.
    if not _snapshot_taken:
        for key in list(sys.modules):
            if (
                key.startswith("gui.") or key == "gui"
                or key in _EXPLICIT_MODULES
            ):
                _post_collection_snapshot[key] = sys.modules[key]
        _snapshot_taken = True

    if item_path.startswith(_GUI_TEST_DIR):
        _restore_gui_state()
    else:
        _restore_real_state()
