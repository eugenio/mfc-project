"""Path configuration for MFC Q-Learning project outputs.
This module provides standardized paths for all output types.
"""

from pathlib import Path

# Get the project root directory (q-learning-mfcs)
PROJECT_ROOT = Path(__file__).parent.parent

# Define standard output directories
FIGURES_DIR = PROJECT_ROOT / "data" / "figures"
SIMULATION_DATA_DIR = PROJECT_ROOT / "data" / "simulation_data"
LOGS_DIR = PROJECT_ROOT / "data" / "logs"
MODELS_DIR = PROJECT_ROOT / "q_learning_models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Create directories if they don't exist
for directory in [FIGURES_DIR, SIMULATION_DATA_DIR, LOGS_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def get_figure_path(filename):
    """Get the full path for a figure file."""
    return str(FIGURES_DIR / filename)


def get_simulation_data_path(filename):
    """Get the full path for a simulation data file."""
    return str(SIMULATION_DATA_DIR / filename)


def get_log_path(filename):
    """Get the full path for a log file."""
    return str(LOGS_DIR / filename)


def get_model_path(filename):
    """Get the full path for a model file."""
    return str(MODELS_DIR / filename)


def get_report_path(filename):
    """Get the full path for a report file."""
    return str(REPORTS_DIR / filename)
