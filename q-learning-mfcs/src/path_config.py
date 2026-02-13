"""
Path configuration for MFC Q-Learning project outputs.
This module provides standardized paths for all output types.
"""
from pathlib import Path
import tempfile
import os

# Get the project root directory (q-learning-mfcs)
PROJECT_ROOT = Path(__file__).parent.parent

# Base directory for compatibility with tests
BASE_DIR = PROJECT_ROOT

# Debug mode flag - when True, outputs go to temporary directory
DEBUG_MODE = os.environ.get('MFC_DEBUG_MODE', '').lower() in ('true', '1', 'yes')

def _get_base_dir():
    """Get the current base directory, allowing for dynamic changes."""
    if DEBUG_MODE:
        # Create a temporary directory for debug mode
        temp_dir = Path(tempfile.gettempdir()) / "mfc_debug_simulation"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir
    return Path(BASE_DIR)

# Define standard output directories (as functions for dynamic base)
def _get_figures_dir():
    return _get_base_dir() / "data" / "figures"

def _get_simulation_data_dir():
    return _get_base_dir() / "data" / "simulation_data"

def _get_logs_dir():
    return _get_base_dir() / "data" / "logs"

def _get_models_dir():
    return _get_base_dir() / "q_learning_models"

def _get_reports_dir():
    return _get_base_dir() / "reports"

# Static directories for backward compatibility
FIGURES_DIR = _get_figures_dir()
SIMULATION_DATA_DIR = _get_simulation_data_dir()
LOGS_DIR = _get_logs_dir()
MODELS_DIR = _get_models_dir()
REPORTS_DIR = _get_reports_dir()

# Create directories if they don't exist
for directory in [FIGURES_DIR, SIMULATION_DATA_DIR, LOGS_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

def get_figure_path(filename):
    """Get the full path for a figure file."""
    path = _get_figures_dir() / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)

def get_simulation_data_path(filename):
    """Get the full path for a simulation data file."""
    path = _get_simulation_data_dir() / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)

def get_log_path(filename):
    """Get the full path for a log file."""
    path = _get_logs_dir() / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)

def get_model_path(filename):
    """Get the full path for a model file."""
    path = _get_models_dir() / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)

def get_report_path(filename):
    """Get the full path for a report file."""
    path = _get_reports_dir() / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)

def is_debug_mode():
    """Check if debug mode is enabled."""
    return DEBUG_MODE

def get_current_base_path():
    """Get the current base path being used for outputs."""
    return str(_get_base_dir())

def enable_debug_mode():
    """Enable debug mode programmatically."""
    global DEBUG_MODE
    DEBUG_MODE = True
    # Refresh static directories
    global FIGURES_DIR, SIMULATION_DATA_DIR, LOGS_DIR, MODELS_DIR, REPORTS_DIR
    FIGURES_DIR = _get_figures_dir()
    SIMULATION_DATA_DIR = _get_simulation_data_dir()
    LOGS_DIR = _get_logs_dir()
    MODELS_DIR = _get_models_dir()
    REPORTS_DIR = _get_reports_dir()

def disable_debug_mode():
    """Disable debug mode programmatically."""
    global DEBUG_MODE
    DEBUG_MODE = False
    # Refresh static directories
    global FIGURES_DIR, SIMULATION_DATA_DIR, LOGS_DIR, MODELS_DIR, REPORTS_DIR
    FIGURES_DIR = _get_figures_dir()
    SIMULATION_DATA_DIR = _get_simulation_data_dir()
    LOGS_DIR = _get_logs_dir()
    MODELS_DIR = _get_models_dir()
    REPORTS_DIR = _get_reports_dir()