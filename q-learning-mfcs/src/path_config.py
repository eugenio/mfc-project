"""Path configuration for MFC Q-Learning project outputs.

This module provides standardized paths for all output types.
Supports debug mode for temporary file output.
"""

import os
from pathlib import Path

# Get the project root directory (q-learning-mfcs)
PROJECT_ROOT = Path(__file__).parent.parent

# Debug mode configuration
DEBUG_BASE_PATH = Path("/tmp/mfc_debug_simulation")  # noqa: S108
_debug_state = {"enabled": False}

# Subdirectories for output
_OUTPUT_SUBDIRS = [
    "data/figures",
    "data/simulation_data",
    "data/logs",
    "q_learning_models",
    "reports",
]


def _check_env_debug_mode() -> bool:
    """Check if debug mode is enabled via environment variable."""
    env_value = os.environ.get("MFC_DEBUG_MODE", "").lower()
    return env_value in ("true", "1", "yes")


def is_debug_mode() -> bool:
    """Check if debug mode is currently enabled.

    Debug mode can be enabled via:
    - Environment variable MFC_DEBUG_MODE=true/1/yes
    - Programmatically via enable_debug_mode()
    """
    return _debug_state["enabled"] or _check_env_debug_mode()


def enable_debug_mode() -> None:
    """Enable debug mode programmatically.

    When enabled, all output paths will be redirected to a temporary directory.
    """
    _debug_state["enabled"] = True
    _ensure_debug_directories()


def disable_debug_mode() -> None:
    """Disable debug mode programmatically."""
    _debug_state["enabled"] = False


def get_current_base_path() -> Path:
    """Get the current base path for outputs.

    Returns the debug directory if debug mode is enabled,
    otherwise returns the project root.
    """
    if is_debug_mode():
        return DEBUG_BASE_PATH
    return PROJECT_ROOT


def _get_output_dir(subpath: str) -> Path:
    """Get an output directory, respecting debug mode."""
    return get_current_base_path() / subpath


def _ensure_debug_directories() -> None:
    """Ensure debug output directories exist."""
    if is_debug_mode():
        base = DEBUG_BASE_PATH
        for subpath in _OUTPUT_SUBDIRS:
            (base / subpath).mkdir(parents=True, exist_ok=True)


# Define standard output directories (for normal mode initialization)
FIGURES_DIR = PROJECT_ROOT / "data" / "figures"
SIMULATION_DATA_DIR = PROJECT_ROOT / "data" / "simulation_data"
LOGS_DIR = PROJECT_ROOT / "data" / "logs"
MODELS_DIR = PROJECT_ROOT / "q_learning_models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Create directories if they don't exist
for directory in [FIGURES_DIR, SIMULATION_DATA_DIR, LOGS_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def get_figure_path(filename: str) -> str:
    """Get the full path for a figure file."""
    output_dir = _get_output_dir("data/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir / filename)


def get_simulation_data_path(filename: str) -> str:
    """Get the full path for a simulation data file."""
    output_dir = _get_output_dir("data/simulation_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir / filename)


def get_log_path(filename: str) -> str:
    """Get the full path for a log file."""
    output_dir = _get_output_dir("data/logs")
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir / filename)


def get_model_path(filename: str) -> str:
    """Get the full path for a model file."""
    output_dir = _get_output_dir("q_learning_models")
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir / filename)


def get_report_path(filename: str) -> str:
    """Get the full path for a report file."""
    output_dir = _get_output_dir("reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir / filename)
