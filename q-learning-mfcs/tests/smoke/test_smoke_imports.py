"""Smoke tests: verify core modules import without errors.

Each test should complete in < 1 second. Failures here indicate missing
dependencies, syntax errors, or broken top-level code.
"""

import importlib

import pytest

# Core simulation modules (no torch dependency)
CORE_MODULES = [
    "biofilm_kinetics",
    "biofilm_kinetics.biofilm_model",
    "biofilm_kinetics.species_params",
    "biofilm_kinetics.substrate_params",
    "metabolic_model",
    "metabolic_model.metabolic_core",
    "metabolic_model.electron_shuttles",
    "metabolic_model.membrane_transport",
    "metabolic_model.pathway_database",
    "gpu_acceleration",
    "mfc_recirculation_control",
    "path_config",
    "simulation_helpers",
    "plotting_system",
    "visualization_analysis",
    "energy_sustainability_analysis",
    "flow_rate_optimization",
    "mfc_dynamic_substrate_control",
]

CONFIG_MODULES = [
    "config",
    "config.electrode_config",
    "config.membrane_config",
    "config.sensor_config",
    "config.qlearning_config",
    "config.substrate_config",
    "config.biological_config",
    "config.control_config",
    "config.visualization_config",
    "config.config_manager",
    "config.config_utils",
    "config.unit_converter",
    "config.parameter_validation",
]

SIMULATION_MODULES = [
    "run_simulation",
    "integrated_mfc_model",
    "mfc_qlearning_optimization",
    "sensor_integrated_mfc_model",
]

# Modules requiring odes (Mojo compiled module) — skip if unavailable
MOJO_MODULES = [
    "mfc_stack_simulation",
    "mfc_100h_simulation",
    "mfc_stack_demo",
]

VISUALIZATION_MODULES = [
    "create_summary_plots",
    "generate_all_figures",
    "generate_performance_graphs",
]

# Modules requiring seaborn — skip if unavailable
SEABORN_MODULES = [
    "three_model_comparison_plots",
]

# Modules requiring torch — skip if torch not available
TORCH_MODULES = [
    "deep_rl_controller",
    "federated_learning_controller",
    "transfer_learning_controller",
    "transformer_controller",
    "base_controller",
    "ml_optimization",
]


@pytest.mark.parametrize("module_name", CORE_MODULES)
def test_core_module_imports(module_name):
    """Core module can be imported without errors."""
    mod = importlib.import_module(module_name)
    assert mod is not None


@pytest.mark.parametrize("module_name", CONFIG_MODULES)
def test_config_module_imports(module_name):
    """Config module can be imported without errors."""
    mod = importlib.import_module(module_name)
    assert mod is not None


@pytest.mark.parametrize("module_name", SIMULATION_MODULES)
def test_simulation_module_imports(module_name):
    """Simulation module can be imported without errors."""
    mod = importlib.import_module(module_name)
    assert mod is not None


@pytest.mark.parametrize("module_name", VISUALIZATION_MODULES)
def test_visualization_module_imports(module_name):
    """Visualization module can be imported without errors."""
    mod = importlib.import_module(module_name)
    assert mod is not None


@pytest.mark.parametrize("module_name", MOJO_MODULES)
def test_mojo_module_imports(module_name):
    """Module requiring odes (Mojo) can be imported (skipped if unavailable)."""
    pytest.importorskip("odes")
    mod = importlib.import_module(module_name)
    assert mod is not None


@pytest.mark.parametrize("module_name", SEABORN_MODULES)
def test_seaborn_module_imports(module_name):
    """Module requiring seaborn can be imported (skipped if unavailable)."""
    pytest.importorskip("seaborn")
    mod = importlib.import_module(module_name)
    assert mod is not None


@pytest.mark.parametrize("module_name", TORCH_MODULES)
def test_torch_module_imports(module_name):
    """Torch-dependent module can be imported (skipped if torch unavailable)."""
    pytest.importorskip("torch")
    mod = importlib.import_module(module_name)
    assert mod is not None
