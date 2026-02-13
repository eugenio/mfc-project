# Test Coverage Expansion TODO

**Date**: 2026-02-08
**Current overall coverage**: 70.84% (45,563 stmts, 13,286 missing)
**Target**: 99%+

## Summary

117 `_cov.py` test files + 27 apptest files already created. Need to cover remaining
~13,286 missing lines across ~60 modules to reach 99%+.

## Known Issues

- `test_apptest_electrode_sim_98.py` crashes with segfault (greenlet/numpy extension conflict)
- Running full apptest suite in one pytest invocation causes segfault; must run files individually
- `tests/config/` and `tests/membrane_models/` directories shadow `src/` counterparts in combined runs
- `test_flow_rate_optimization_cov.py`, `test_parameter_input.py`, `test_real_time_validator.py` fail collection in combined runs
- Some original tests hang indefinitely; need timeout or isolation
- Cross-test `sys.modules` pollution causes failures in combined runs

## Coverage Measurement Command

Must run in batches with `coverage run --append`:
```bash
# Clean
rm -f .coverage

# Batch 1: clean directories
pixi run -e default python -m coverage run --source=q-learning-mfcs/src -m pytest \
  q-learning-mfcs/tests/stability/ q-learning-mfcs/tests/gsm/ q-learning-mfcs/tests/qlearning/ \
  q-learning-mfcs/tests/visualization/ q-learning-mfcs/tests/metabolic_model/ q-learning-mfcs/tests/utils/ \
  q-learning-mfcs/tests/analysis/ q-learning-mfcs/tests/substrate/ q-learning-mfcs/tests/physics/ \
  q-learning-mfcs/tests/energy/ q-learning-mfcs/tests/deployment/ q-learning-mfcs/tests/quantum/ \
  q-learning-mfcs/tests/notification_system/ -q --tb=no

# Batch 2: compliance, gpu, monitoring, performance
pixi run -e default python -m coverage run --append --source=q-learning-mfcs/src -m pytest \
  q-learning-mfcs/tests/compliance/ q-learning-mfcs/tests/gpu/ q-learning-mfcs/tests/monitoring/ \
  q-learning-mfcs/tests/performance/ q-learning-mfcs/tests/gui/test_web_download_server_cov.py \
  q-learning-mfcs/tests/gui/test_mfc_streamlit_gui_cov.py q-learning-mfcs/tests/test_torch_compat_cov.py \
  --ignore=q-learning-mfcs/tests/monitoring/test_observability_manager.py -q --tb=no

# Batch 3: config + core (namespace conflict dirs, explicit file list)
pixi run -e default python -m coverage run --append --source=q-learning-mfcs/src -m pytest \
  q-learning-mfcs/tests/config/test_*_cov.py q-learning-mfcs/tests/core/test_*_cov.py \
  --ignore=q-learning-mfcs/tests/core/test_flow_rate_optimization_cov.py -q --tb=no

# Batch 4: membrane_models + sensing_models + integrated
pixi run -e default python -m coverage run --append --source=q-learning-mfcs/src -m pytest \
  q-learning-mfcs/tests/membrane_models/test_*_cov.py q-learning-mfcs/tests/sensing_models/test_*_cov.py \
  q-learning-mfcs/tests/integrated/test_*_cov.py -q --tb=no

# Batch 5: apptest files (one by one to avoid segfault)
for f in q-learning-mfcs/tests/apptest/test_apptest_*.py; do
  pixi run -e default python -m coverage run --append --source=q-learning-mfcs/src \
    -m pytest "$f" -q --no-header --tb=no 2>&1 | tail -1
done

# Report
pixi run -e default python -m coverage report --include="q-learning-mfcs/src/*" --sort=-Miss
```

---

## Priority 1: 0% Coverage Modules (sorted by stmts desc)

These modules have zero test coverage and need new test files.

### Config Modules (3,267 stmts total)

| Module | Stmts | Test File Location |
|--------|-------|--------------------|
| `config/advanced_visualization.py` | 543 | `tests/config/test_advanced_visualization_cov.py` |
| `config/statistical_analysis.py` | 494 | `tests/config/test_statistical_analysis_cov.py` |
| `config/model_validation.py` | 461 | `tests/config/test_model_validation_cov.py` |
| `config/experimental_data_integration.py` | 418 | `tests/config/test_experimental_data_integration_cov.py` |
| `config/parameter_optimization.py` | 393 | `tests/config/test_parameter_optimization_cov.py` |
| `config/uncertainty_quantification.py` | 388 | `tests/config/test_uncertainty_quantification_cov.py` |
| `config/real_time_validator.py` | 191 | `tests/config/test_real_time_validator_cov.py` |
| `config/unit_converter.py` | 81 | `tests/config/test_unit_converter_cov.py` |

### Controller Modules (torch-dependent, 2,511 stmts total)

| Module | Stmts | Test File Location |
|--------|-------|--------------------|
| `transfer_learning_controller.py` | 593 | `tests/controllers/test_transfer_learning_controller_cov.py` |
| `deep_rl_controller.py` | 491 | `tests/controllers/test_deep_rl_controller_cov.py` |
| `federated_learning_controller.py` | 488 | `tests/controllers/test_federated_learning_controller_cov.py` |
| `transformer_controller.py` | 323 | `tests/controllers/test_transformer_controller_cov.py` |
| `base_controller.py` | 116 | `tests/controllers/test_base_controller_cov.py` |

### MLOps Modules (1,095 stmts total)

| Module | Stmts | Test File Location |
|--------|-------|--------------------|
| `mlops/deployment_manager.py` | 596 | `tests/mlops/test_deployment_manager_cov.py` |
| `mlops/experiment_tracker.py` | 287 | `tests/mlops/test_experiment_tracker_cov.py` |
| `mlops/model_registry.py` | 212 | `tests/mlops/test_model_registry_cov.py` |

### Simulation Modules (1,188 stmts total)

| Module | Stmts | Test File Location |
|--------|-------|--------------------|
| `run_simulation.py` | 359 | `tests/simulation/test_run_simulation_cov.py` |
| `integrated_model.py` | 315 | `tests/simulation/test_integrated_model_cov.py` |
| `flow_rate_optimization.py` | 304 | `tests/simulation/test_flow_rate_optimization_cov.py` |
| `mfc_stack_simulation.py` | 302 | `tests/simulation/test_mfc_stack_simulation_cov.py` |
| `mfc_100h_simulation.py` | 227 | `tests/simulation/test_mfc_100h_simulation_cov.py` |

### Stability + Analysis (1,086 stmts total)

| Module | Stmts | Test File Location |
|--------|-------|--------------------|
| `stability/stability_visualizer.py` | 294 | `tests/stability/test_stability_visualizer_cov.py` |
| `stability/stability_framework.py` | 289 | `tests/stability/test_stability_framework_cov.py` |
| `analysis/policy_evolution_tracker.py` | 280 | `tests/analysis/test_policy_evolution_tracker_cov.py` |
| `analysis/qtable_analyzer.py` | 223 | `tests/analysis/test_qtable_analyzer_cov.py` |

### Monitoring + Misc (1,145 stmts total)

| Module | Stmts | Test File Location |
|--------|-------|--------------------|
| `monitoring/start_monitoring.py` | 291 | `tests/monitoring/test_start_monitoring_deep_cov.py` |
| `testing/coverage_utilities.py` | 249 | `tests/testing/test_coverage_utilities_cov.py` |
| `cathode_models/biological_cathode.py` | 167 | `tests/cathode/test_biological_cathode_cov.py` |
| `cathode_models/platinum_cathode.py` | 123 | `tests/cathode/test_platinum_cathode_cov.py` |
| `cathode_models/base_cathode.py` | 73 | `tests/cathode/test_base_cathode_cov.py` |
| `build_qlearning.py` | 31 | `tests/qlearning/test_build_qlearning_cov.py` |
| `test.py` | 20 | (skip - test utility) |
| `cathode_models/__init__.py` | 5 | (include in cathode tests) |
| `matplotlib_config.py` | 2 | (include in any viz test) |

---

## Priority 2: Low Coverage Modules (<60%)

These already have partial tests but need significant expansion.

| Module | Stmts | Miss | Cover | Gap |
|--------|-------|------|-------|-----|
| `mfc_dynamic_substrate_control.py` | 543 | 318 | 41.4% | 318 |
| `sensor_integrated_mfc_model.py` | 396 | 292 | 26.3% | 292 |
| `adaptive_mfc_controller.py` | 371 | 286 | 22.9% | 286 |
| `sensing_enhanced_q_controller.py` | 269 | 229 | 14.9% | 229 |
| `mfc_streamlit_gui.py` | 208 | 199 | 4.3% | 199 |
| `integrated_mfc_model.py` | 245 | 195 | 20.4% | 195 |
| `monitoring/dashboard_api.py` | 287 | 167 | 41.8% | 167 |
| `biofilm_kinetics/enhanced_biofilm_model.py` | 213 | 151 | 29.1% | 151 |
| `run_gpu_simulation.py` | 321 | 146 | 54.5% | 146 |
| `gui/electrode_configuration_ui.py` | 202 | 136 | 32.7% | 136 |
| `gui/simulation_runner.py` | 266 | 118 | 55.6% | 118 |
| `web_download_server.py` | 169 | 110 | 34.9% | 110 |
| `controller_models/model_inference.py` | 310 | 101 | 67.4% | 101 |
| `biofilm_kinetics/biofilm_model.py` | 167 | 99 | 40.7% | 99 |
| `config/electrode_config.py` | 147 | 63 | 57.1% | 63 |
| `config/config_io.py` | 152 | 60 | 60.5% | 60 |
| `phase2_demonstration.py` | 96 | 44 | 54.2% | 44 |
| `inspect_live_biofilm.py` | 46 | 41 | 10.9% | 41 |
| `biofilm_kinetics/substrate_params.py` | 66 | 34 | 48.5% | 34 |
| `path_config.py` | 54 | 29 | 46.3% | 29 |
| `mfc_model.py` | 21 | 17 | 19.1% | 17 |
| `biofilm_kinetics/species_params.py` | 47 | 16 | 66.0% | 16 |

**Subtotal Priority 2 gap**: ~3,482 stmts

---

## Priority 3: Medium Coverage Modules (60-95%)

These need smaller targeted improvements.

| Module | Stmts | Miss | Cover |
|--------|-------|------|-------|
| `config/real_time_processing.py` | 483 | 59 | 87.8% |
| `config/sensitivity_analysis.py` | 391 | 48 | 87.7% |
| `deployment/service_orchestrator.py` | 541 | 69 | 87.3% |
| `deployment/process_manager.py` | 396 | 64 | 83.8% |
| `deployment/log_management.py` | 324 | 62 | 80.9% |
| `controller_models/real_time_controller.py` | 347 | 37 | 89.3% |
| `config/config_manager.py` | 275 | 27 | 90.2% |
| `config/membrane_config.py` | 61 | 18 | 70.5% |
| `config/literature_database.py` | 113 | 17 | 85.0% |
| `mfc_unified_qlearning_control.py` | 578 | 39 | 93.3% |
| `mfc_qlearning_optimization_parallel.py` | 490 | 25 | 94.9% |
| `gui/pages/electrode_enhanced.py` | 184 | 12 | 93.5% |
| `gui/pages/cell_config.py` | 158 | 8 | 94.9% |
| `monitoring/realtime_streamer.py` | 254 | 14 | 94.5% |
| `performance/gpu_memory_manager.py` | 190 | 13 | 93.2% |
| `biofilm_health_monitor.py` | 343 | 17 | 95.0% |
| `config/qlearning_config.py` | 192 | 8 | 95.8% |
| `notifications/__init__.py` | 42 | 7 | 83.3% |
| `email_notification.py` | 69 | 6 | 91.3% |
| `config/config_utils.py` | 254 | 18 | 92.9% |
| `config/simulation_chronology.py` | 157 | 7 | 95.5% |

**Subtotal Priority 3 gap**: ~576 stmts

---

## Test File Patterns

Every `_cov.py` test file must follow this template:
```python
"""Tests for <module_name> module."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Mock external deps BEFORE importing target module
from unittest.mock import MagicMock, patch
# ... mock torch, optuna, etc. as needed ...

import pytest
from <package>.<module> import TargetClass, target_function

class TestTargetClass:
    def test_something(self):
        ...
```

- Max 450 lines per file (hook blocks larger)
- Use `pixi run -e default python -m pytest <file> -q` to verify
- Use `git add -f` to stage (gitignore has `*test*.py` pattern)
- Run `pixi run -e default ruff check <file>` before committing

---

## Estimated Effort

| Priority | Missing Stmts | Est. Tests | Est. Files |
|----------|--------------|------------|------------|
| P1 (0% modules) | ~10,567 | ~800 | ~30 |
| P2 (low coverage) | ~3,482 | ~250 | ~20 |
| P3 (medium coverage) | ~576 | ~50 | ~15 |
| **Total** | **~13,286** | **~1,100** | **~65** |

Reaching 99%+ requires covering ~13,000 of the 13,286 missing lines.
