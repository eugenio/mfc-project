# PRD: MFC Test Suite Analysis & Verification

## Overview

This PRD defines a systematic approach to analyze, categorize, and fix all failing tests in the MFC (Microbial Fuel Cell) project repository. The goal is to achieve a fully passing test suite with clear understanding of test categories and failure root causes.

## Problem Statement

The MFC project test suite has significant issues:

- **580 tests collected** with **4 collection errors** (tests that fail to import)
- **~145 failing tests** across multiple categories
- **~89 skipped tests** (legitimate skips for missing dependencies)
- **~332 passing tests** (57% pass rate)

Test failures fall into distinct categories requiring different remediation approaches:

1. **Import/Module Errors** - Tests that can't even be collected
2. **SciPy/NumPy Compatibility** - Environment-level issues
3. **Missing Functions/APIs** - Code changes broke test assumptions
4. **Streamlit/GUI Mocking** - Tests require Streamlit context
5. **Hook Module Dependencies** - Tests for Claude hooks need special setup
6. **Integration Tests** - Require full system setup
7. **File I/O Tests** - Tests expecting specific file outputs

## Target Users

| User Type | Benefit |
|-----------|---------|
| **Developers** | Confidence in code changes through passing tests |
| **CI/CD** | Automated quality gates that actually work |
| **Contributors** | Clear test expectations for new features |

## Goals & Success Metrics

| Goal | Metric | Target |
|------|--------|--------|
| Fix collection errors | Tests that import successfully | 100% |
| Fix failing tests | Test pass rate | >95% |
| Reduce flaky tests | Consistent results across runs | <2% flaky |
| Document skip reasons | Skipped tests with clear reasons | 100% |

---

## User Stories

### Epic 1: Collection Error Fixes (Blocking)

#### US-001: Fix SciPy/NumPy Compatibility Error

**As a** developer running tests
**I want to** tests to import without numpy.dtype errors
**So that** I can run the full test suite

**Acceptance Criteria:**
- [ ] `test_observability_manager.py` imports successfully
- [ ] `test_sensing_models.py` imports successfully
- [ ] Root cause identified (scipy.signal import issue)
- [ ] Fix applied (lazy import or dependency update)
- [ ] `pixi run pytest --collect-only` shows no collection errors

**Priority:** Critical
**Estimate:** S

**Analysis:** The error `TypeError: numpy.dtype is not a type object` occurs when importing `scipy.signal`. This is typically a NumPy/SciPy version mismatch.

---

#### US-002: Fix Missing Import `is_debug_mode`

**As a** developer running tests
**I want to** `test_debug_mode.py` to import correctly
**So that** debug mode tests can run

**Acceptance Criteria:**
- [ ] `is_debug_mode` function added to `path_config.py` OR
- [ ] Test updated to use correct function name
- [ ] Test passes or is properly skipped
- [ ] `pixi run ruff check` passes

**Priority:** High
**Estimate:** S

**Analysis:** `test_debug_mode.py:13` tries to import `is_debug_mode` from `path_config` but it doesn't exist.

---

#### US-003: Fix Missing Module `optuna`

**As a** developer running tests
**I want to** `test_optimized_config.py` to handle missing optuna gracefully
**So that** tests work without optional dependencies

**Acceptance Criteria:**
- [ ] Test uses `pytest.importorskip("optuna")` OR
- [ ] Optuna added to test dependencies OR
- [ ] Test skipped with clear reason
- [ ] No import errors during collection

**Priority:** High
**Estimate:** S

**Analysis:** `hyperparameter_optimization.py` imports optuna at module level, causing collection failure.

---

### Epic 2: Config Module Test Fixes

#### US-004: Fix config_io Save/Load Tests

**As a** developer working on configuration
**I want to** config I/O tests to pass
**So that** configuration serialization is verified

**Acceptance Criteria:**
- [ ] `test_save_config_yaml` passes
- [ ] `test_save_config_json` passes
- [ ] `test_load_config_yaml` passes
- [ ] `test_merge_configs_*` tests pass (5 tests)
- [ ] Root cause identified and documented

**Priority:** High
**Estimate:** M

**Files:** `q-learning-mfcs/tests/config/test_config_io.py`

---

#### US-005: Fix Sensitivity Analysis Tests

**As a** developer working on parameter analysis
**I want to** sensitivity analysis tests to pass
**So that** parameter sensitivity features are verified

**Acceptance Criteria:**
- [ ] `test_sobol_sampling` passes
- [ ] `test_sobol_analysis` passes
- [ ] `test_plot_sensitivity_indices` passes
- [ ] `test_plot_morris_results` passes
- [ ] `test_plot_parameter_ranking` passes
- [ ] SALib dependency verified/added if needed

**Priority:** Medium
**Estimate:** M

**Files:** `q-learning-mfcs/tests/config/test_sensitivity_analysis.py`

---

#### US-006: Fix Mass Calculation Test

**As a** developer working on electrode config
**I want to** mass calculation test to pass
**So that** electrode mass calculations are verified

**Acceptance Criteria:**
- [ ] `test_calculate_mass_rectangular` passes
- [ ] Density parameter correctly used in calculation
- [ ] Test assertions match implementation

**Priority:** Medium
**Estimate:** S

**Files:** `q-learning-mfcs/tests/config/test_mass_calc.py`

---

### Epic 3: GUI/Streamlit Test Fixes

#### US-007: Fix Core Layout Tests

**As a** developer working on GUI
**I want to** core layout tests to pass
**So that** GUI layout functions are verified

**Acceptance Criteria:**
- [ ] All `test_core_layout.py` tests pass (10 tests)
- [ ] Streamlit mocking properly configured
- [ ] Tests use `unittest.mock` or `streamlit.testing`
- [ ] No Streamlit context errors

**Priority:** Medium
**Estimate:** M

**Files:** `q-learning-mfcs/tests/gui/test_core_layout.py`

---

#### US-008: Fix Alert Configuration Tests

**As a** developer working on alerts
**I want to** alert tests to pass
**So that** alert system is verified

**Acceptance Criteria:**
- [ ] `test_alert_functions` passes
- [ ] `test_module_import` passes
- [ ] `test_render_alert_configuration` passes
- [ ] Mock Streamlit context properly

**Priority:** Medium
**Estimate:** S

**Files:** `q-learning-mfcs/tests/gui/test_alerts.py`

---

#### US-009: Fix Page Module Tests

**As a** developer working on GUI pages
**I want to** page module tests to pass
**So that** individual page functionality is verified

**Acceptance Criteria:**
- [ ] `test_dashboard_page` passes
- [ ] `test_cell_config_page` passes
- [ ] `test_electrode_page` passes
- [ ] `test_advanced_physics_page` passes
- [ ] `test_ml_optimization_page` passes
- [ ] `test_performance_monitor_page` passes

**Priority:** Medium
**Estimate:** M

**Files:** `q-learning-mfcs/tests/gui/test_page_modules.py`

---

#### US-010: Fix Remaining GUI Tests

**As a** developer working on GUI
**I want to** all remaining GUI tests to pass
**So that** GUI functionality is fully verified

**Acceptance Criteria:**
- [ ] `test_live_monitoring.py` passes
- [ ] `test_policy_evolution.py` passes
- [ ] `test_qtable.py` passes
- [ ] `test_parameter_input.py` passes
- [ ] `test_browser_download.py` passes
- [ ] `test_advanced_visualizations.py` passes

**Priority:** Medium
**Estimate:** L

---

### Epic 4: Hook Module Test Fixes

#### US-011: Fix Security Guardian Tests

**As a** developer working on hooks
**I want to** security guardian tests to pass
**So that** security features are verified

**Acceptance Criteria:**
- [ ] All `test_enhanced_security_guardian.py` tests pass (15 tests)
- [ ] Hook module imports work from test location
- [ ] Mocking properly configured for hook functions

**Priority:** Medium
**Estimate:** M

**Files:** `q-learning-mfcs/tests/hooks/test_enhanced_security_guardian.py`

---

#### US-012: Fix Git Guardian Integration Tests

**As a** developer working on hooks
**I want to** git guardian tests to pass
**So that** git integration is verified

**Acceptance Criteria:**
- [ ] All `test_git_guardian_integration.py` tests pass (10 tests)
- [ ] Guardian functions properly mocked
- [ ] No subprocess side effects

**Priority:** Medium
**Estimate:** M

**Files:** `q-learning-mfcs/tests/hooks/test_git_guardian_integration.py`

---

#### US-013: Fix Hook Guardian Behavior Tests

**As a** developer working on hooks
**I want to** hook behavior tests to pass
**So that** hook execution is verified

**Acceptance Criteria:**
- [ ] All `test_hook_guardian_behavior.py` tests pass (8 tests)
- [ ] Code analysis functions work
- [ ] Security blocks properly detected

**Priority:** Medium
**Estimate:** M

**Files:** `q-learning-mfcs/tests/hooks/test_hook_guardian_behavior.py`

---

### Epic 5: Model/Simulation Test Fixes

#### US-014: Fix GPU Acceleration Tests

**As a** developer working on GPU support
**I want to** GPU acceleration tests to pass
**So that** GPU fallback behavior is verified

**Acceptance Criteria:**
- [ ] All `test_gpu_acceleration.py` tests pass (10 tests)
- [ ] CPU fallback mode works correctly
- [ ] Array operations work without GPU

**Priority:** High
**Estimate:** M

**Files:** `q-learning-mfcs/tests/test_gpu_acceleration.py`

---

#### US-015: Fix Integrated Model Tests

**As a** developer working on simulation
**I want to** integrated model tests to pass
**So that** simulation accuracy is verified

**Acceptance Criteria:**
- [ ] All `test_integrated_model.py` tests pass (10 tests)
- [ ] Biofilm-metabolic coupling works
- [ ] Multi-step simulation stable

**Priority:** High
**Estimate:** M

**Files:** `q-learning-mfcs/tests/test_integrated_model.py`

---

#### US-016: Fix Metabolic Model Tests

**As a** developer working on metabolic model
**I want to** metabolic model tests to pass
**So that** metabolic calculations are verified

**Acceptance Criteria:**
- [ ] All `test_metabolic_model.py` tests pass (9 tests)
- [ ] Electron shuttle calculations correct
- [ ] Oxygen crossover effects work

**Priority:** High
**Estimate:** M

**Files:** `q-learning-mfcs/tests/metabolic_model/test_metabolic_model.py`

---

### Epic 6: Analysis/Output Test Fixes

#### US-017: Fix Q-Table Analysis Tests

**As a** developer working on Q-learning
**I want to** Q-table analysis tests to pass
**So that** Q-table operations are verified

**Acceptance Criteria:**
- [ ] All `test_qtable_analysis.py` tests pass (8 tests)
- [ ] Pickle loading works
- [ ] Analysis export works

**Priority:** Medium
**Estimate:** M

**Files:** `q-learning-mfcs/tests/test_qtable_analysis.py`

---

#### US-018: Fix Policy Evolution Tests

**As a** developer working on Q-learning
**I want to** policy evolution tests to pass
**So that** policy tracking is verified

**Acceptance Criteria:**
- [ ] All `test_policy_evolution_tracker.py` tests pass (10 tests)
- [ ] Policy snapshots work
- [ ] Convergence detection works

**Priority:** Medium
**Estimate:** M

**Files:** `q-learning-mfcs/tests/test_policy_evolution_tracker.py`

---

#### US-019: Fix File Output Tests

**As a** developer working on data output
**I want to** file output tests to pass
**So that** data persistence is verified

**Acceptance Criteria:**
- [ ] `test_file_outputs.py` tests pass (3 tests)
- [ ] `test_path_outputs.py` tests pass (3 tests)
- [ ] `test_parquet_io.py` tests pass (3 tests)
- [ ] Output directories properly created

**Priority:** Medium
**Estimate:** M

---

### Epic 7: Integration & Performance Test Fixes

#### US-020: Fix Security Integration Tests

**As a** developer working on security
**I want to** security integration tests to pass
**So that** security features are verified

**Acceptance Criteria:**
- [ ] `test_authentication_flow` passes
- [ ] `test_session_security_features` passes
- [ ] `test_rate_limiting` passes
- [ ] `test_csrf_protection` passes

**Priority:** Medium
**Estimate:** M

**Files:** `q-learning-mfcs/tests/integration/test_security_integration.py`

---

#### US-021: Fix System Integration Tests

**As a** developer working on system
**I want to** system integration tests to pass or be properly skipped
**So that** system-level features are verified

**Acceptance Criteria:**
- [ ] All 11 `test_system_integration.py` tests pass or have clear skip reasons
- [ ] Performance tests have realistic expectations
- [ ] Concurrent operation tests stable

**Priority:** Low
**Estimate:** L

**Files:** `q-learning-mfcs/tests/integration/test_system_integration.py`

---

#### US-022: Fix Performance Stress Tests

**As a** developer working on performance
**I want to** performance tests to pass
**So that** performance benchmarks are verified

**Acceptance Criteria:**
- [ ] All `test_performance_stress.py` tests pass (8 tests)
- [ ] Memory efficiency tests work
- [ ] CPU utilization tests stable

**Priority:** Low
**Estimate:** M

**Files:** `q-learning-mfcs/tests/test_performance_stress.py`

---

### Epic 8: Miscellaneous Test Fixes

#### US-023: Fix Edge Case Tests

**As a** developer
**I want to** edge case tests to pass
**So that** boundary conditions are verified

**Acceptance Criteria:**
- [ ] All `test_comprehensive_edge_cases.py` tests pass (6 tests)
- [ ] Zero value handling correct
- [ ] Numerical precision verified

**Priority:** Medium
**Estimate:** M

**Files:** `q-learning-mfcs/tests/test_comprehensive_edge_cases.py`

---

#### US-024: Fix Epsilon and Biological Tests

**As a** developer
**I want to** remaining miscellaneous tests to pass
**So that** all features are verified

**Acceptance Criteria:**
- [ ] `test_epsilon_fix.py` passes
- [ ] `test_biological_constraints.py::test_energy_conservation` passes
- [ ] `test_actual_executions.py` passes or skips cleanly

**Priority:** Low
**Estimate:** S

---

## Technical Considerations

### Test Categories by Root Cause

1. **Environment Issues** (US-001)
   - SciPy/NumPy version mismatch
   - Fix: Update pixi.toml dependencies

2. **Missing Code** (US-002, US-003)
   - Functions removed or renamed during refactoring
   - Fix: Add functions or update tests

3. **Streamlit Context** (US-007-010)
   - Tests call Streamlit functions outside app context
   - Fix: Use `unittest.mock.patch` for Streamlit functions

4. **Hook Module Paths** (US-011-013)
   - Relative imports don't work from test directory
   - Fix: Use absolute imports or pytest fixtures

5. **API Changes** (US-004-006, US-014-019)
   - Code was refactored, tests not updated
   - Fix: Update test expectations

6. **Missing Test Data** (US-017-019)
   - Tests expect files that don't exist
   - Fix: Create fixtures or mock file operations

### Testing Strategy

Run tests in groups to isolate failures:
```bash
# Group 1: Config tests
pixi run pytest q-learning-mfcs/tests/config -v

# Group 2: GUI tests
pixi run pytest q-learning-mfcs/tests/gui -v

# Group 3: Hook tests
pixi run pytest q-learning-mfcs/tests/hooks -v

# Group 4: Model tests
pixi run pytest q-learning-mfcs/tests/biofilm_kinetics q-learning-mfcs/tests/metabolic_model -v

# Group 5: Integration tests
pixi run pytest q-learning-mfcs/tests/integration -v
```

---

## Implementation Order

**Phase 1: Unblock Collection (Critical)**
- US-001, US-002, US-003

**Phase 2: Core Config Tests**
- US-004, US-005, US-006

**Phase 3: GPU/Model Tests**
- US-014, US-015, US-016

**Phase 4: GUI Tests**
- US-007, US-008, US-009, US-010

**Phase 5: Hook Tests**
- US-011, US-012, US-013

**Phase 6: Analysis Tests**
- US-017, US-018, US-019

**Phase 7: Integration Tests**
- US-020, US-021, US-022

**Phase 8: Cleanup**
- US-023, US-024

---

## Appendix: Test Failure Summary

### Collection Errors (4)
```
test_observability_manager.py - scipy.signal numpy.dtype error
test_sensing_models.py - scipy.signal numpy.dtype error
test_debug_mode.py - ImportError: is_debug_mode
test_optimized_config.py - ModuleNotFoundError: optuna
```

### Failed Tests by Category

**Config (16 failures)**
- config_io: 10 tests
- sensitivity_analysis: 5 tests
- mass_calc: 1 test

**GUI (24 failures)**
- core_layout: 8 tests
- page_modules: 6 tests
- alerts: 3 tests
- others: 7 tests

**Hooks (27 failures)**
- enhanced_security_guardian: 11 tests
- git_guardian_integration: 10 tests
- hook_guardian_behavior: 6 tests

**Models (29 failures)**
- gpu_acceleration: 10 tests
- integrated_model: 10 tests
- metabolic_model: 9 tests

**Analysis (21 failures)**
- qtable_analysis: 8 tests
- policy_evolution_tracker: 10 tests
- file outputs: 3 tests

**Integration (22 failures)**
- security_integration: 5 tests
- system_integration: 11 errors
- performance_stress: 6 tests

**Other (6 failures)**
- edge_cases: 4 tests
- epsilon_fix: 1 test
- biological_constraints: 1 test

---

Generated with [Claude Code](https://claude.com/claude-code)
