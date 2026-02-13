# PRD: MFC Repository Refactoring & Debloating

## Overview

This PRD defines a systematic approach to refactor and debloat the MFC (Microbial Fuel Cell) simulation project repository. The goal is to reduce code duplication, eliminate dead code, improve organization, and enhance maintainability while preserving the existing 5-phase enhancement architecture and GPU-accelerated performance.

## Problem Statement

The MFC project has grown organically through 5 enhancement phases, resulting in:

- **Code duplication**: 15+ pairs of variant/duplicate files (e.g., `config_io.py` + `config_io_fix.py` + `config_io_fixed.py`)
- **Dead code**: Stub files, experimental scripts, and test files mixed with source code
- **Poor organization**: Nested duplicate directories, scattered scripts, 200+ modified files
- **Large monolithic files**: 8+ files exceeding 1,400 lines (largest: 2,348 lines)
- **Configuration sprawl**: 18.5K lines across config modules with overlapping responsibilities
- **Documentation bloat**: 25 markdown files (~330KB) with duplicate/superseded content
- **Test fragmentation**: 72 test files scattered across multiple locations

This technical debt increases maintenance burden, slows onboarding, and makes the codebase harder to navigate.

## Target Users

| User Type | Benefit |
|-----------|---------|
| **Internal developers** | Faster navigation, clearer code ownership, reduced merge conflicts |
| **External contributors** | Easier onboarding, clearer contribution guidelines |
| **Research scientists** | Cleaner API surface, better documentation |
| **CI/CD systems** | Faster builds, reduced test flakiness |

## Goals & Success Metrics

| Goal | Metric | Target |
|------|--------|--------|
| Reduce code duplication | Lines of duplicate code | -50% |
| Eliminate dead code | Unused files removed | 30+ files |
| Improve organization | Files in wrong locations | 0 |
| Reduce file count | Total Python files | -15% |
| Maintain test coverage | Coverage percentage | No regression |
| Preserve performance | GPU speedup | 8400x maintained |
| Improve maintainability | Avg file size (lines) | <500 for new files |

---

## User Stories

### Epic 1: Configuration Consolidation

#### US-001: Consolidate Config I/O Variants

**As a** developer maintaining the configuration system
**I want to** have a single config I/O implementation
**So that** I don't have to maintain three variant files with unclear purposes

**Acceptance Criteria:**
- [ ] `config_io.py`, `config_io_fix.py`, and `config_io_fixed.py` merged into single `config_io.py`
- [ ] Enum serialization fix from variant files integrated
- [ ] All existing tests pass
- [ ] No imports break across codebase
- [ ] `pixi run ruff check` passes
- [ ] `pixi run mypy` passes (or existing errors unchanged)

**Priority:** High
**Estimate:** S

---

#### US-002: Merge SSL Configuration Variants

**As a** developer working on the monitoring system
**I want to** have environment-based SSL configuration instead of separate files
**So that** I can switch between dev and production without duplicate code

**Acceptance Criteria:**
- [ ] `ssl_config.py` and `ssl_dev_config.py` merged into single file
- [ ] Environment variable or config flag controls dev vs production mode
- [ ] All SSL-related tests pass
- [ ] Monitoring dashboard still functions correctly
- [ ] `pixi run ruff check` passes

**Priority:** High
**Estimate:** S

---

#### US-003: Consolidate Validation Modules

**As a** developer working on parameter validation
**I want to** have clear separation between config data structures and validation logic
**So that** I don't have overlapping modules doing similar things

**Acceptance Criteria:**
- [ ] `model_validation.py` and `biological_validation.py` consolidated where appropriate
- [ ] Clear interface between validation logic and config data
- [ ] No functionality lost
- [ ] Tests updated and passing
- [ ] `pixi run ruff check` passes

**Priority:** Medium
**Estimate:** M

---

### Epic 2: Duplicate Code Elimination

#### US-004: Consolidate PDF Report Generators

**As a** developer generating reports
**I want to** have a single PDF report generation module
**So that** I don't maintain two implementations with unclear differences

**Acceptance Criteria:**
- [ ] `generate_pdf_report.py` and `generate_enhanced_pdf_report.py` consolidated
- [ ] "Enhanced" features available via parameter/flag
- [ ] Backward compatibility maintained for existing scripts
- [ ] Sample report generation verified
- [ ] `pixi run ruff check` passes

**Priority:** High
**Estimate:** S

---

#### US-005: Consolidate Sensor Fusion Models

**As a** developer working on sensing models
**I want to** have a single sensor fusion implementation
**So that** the advanced features supersede the basic version cleanly

**Acceptance Criteria:**
- [ ] `sensor_fusion.py` retired in favor of `advanced_sensor_fusion.py`
- [ ] Any unique functionality from basic version preserved
- [ ] All imports updated across codebase
- [ ] Tests pass
- [ ] `pixi run ruff check` passes

**Priority:** High
**Estimate:** S

---

#### US-006: Consolidate Flow Rate Optimization

**As a** developer working on optimization
**I want to** have a single flow rate optimization module
**So that** "realistic" mode is a configuration option, not a separate file

**Acceptance Criteria:**
- [ ] `flow_rate_optimization.py` and `flow_rate_optimization_realistic.py` merged
- [ ] Realistic mode available via parameter
- [ ] Documentation updated
- [ ] Tests pass
- [ ] `pixi run ruff check` passes

**Priority:** Medium
**Estimate:** S

---

#### US-007: Consolidate Q-Learning Control Variants

**As a** developer working on the Q-learning system
**I want to** have clear distinction between control variants
**So that** I know which implementation is production vs experimental

**Acceptance Criteria:**
- [ ] `mfc_unified_qlearning_control.py` and `mfc_unified_qlearning_optimized.py` consolidated or clearly documented
- [ ] Production implementation clearly marked
- [ ] GPU speedup preserved (8400x)
- [ ] Q-table compatibility maintained
- [ ] Tests pass

**Priority:** High
**Estimate:** M

---

#### US-008: Consolidate Dashboard Components

**As a** developer working on the GUI
**I want to** have a unified dashboard module structure
**So that** I don't maintain 5 overlapping dashboard implementations

**Acceptance Criteria:**
- [ ] `live_monitoring_dashboard.py`, `dashboard.py`, `dashboard_api.py`, `dashboard_frontend.py`, `simple_dashboard_api.py` consolidated
- [ ] Clear separation: GUI components vs API vs backend
- [ ] Simple vs advanced modes available via configuration
- [ ] All dashboard functionality preserved
- [ ] Tests pass
- [ ] `pixi run ruff check` passes

**Priority:** Medium
**Estimate:** L

---

### Epic 3: Dead Code Removal

#### US-009: Remove Stub Files

**As a** developer navigating the codebase
**I want to** not encounter stub/placeholder files
**So that** I don't waste time investigating empty implementations

**Acceptance Criteria:**
- [ ] `create_https_issue.py` (1-line stub) removed
- [ ] `create_https_enhancement_issue.py` (1-line stub) removed
- [ ] `check_issues.py` (obsolete) archived or removed
- [ ] Any other stub files identified and removed
- [ ] No broken imports

**Priority:** High
**Estimate:** S

---

#### US-010: Move Test Files from Source Directory

**As a** developer following project conventions
**I want to** test files only in test directories
**So that** source directories contain only production code

**Acceptance Criteria:**
- [ ] `test_debug_mode.py` moved from `src/` to `tests/`
- [ ] `test_epsilon_fix.py` moved from `src/` to `tests/`
- [ ] `test_gitlab_cli.py` moved from `src/` to `tests/`
- [ ] `test_optimized_config.py` moved from `src/` to `tests/`
- [ ] `test_path_outputs.py` moved from `src/` to `tests/`
- [ ] `test_qtable_loading.py` moved from `src/` to `tests/`
- [ ] All imports updated
- [ ] CI configuration updated if needed
- [ ] Tests still pass from new locations

**Priority:** High
**Estimate:** S

---

#### US-011: Consolidate Issue Management Scripts

**As a** developer managing GitLab issues
**I want to** have a single issue management module
**So that** I don't maintain scattered scripts at multiple levels

**Acceptance Criteria:**
- [ ] Root-level issue scripts (`create_mfc_stack_issue.py`, `create_test_failure_issues.py`) consolidated
- [ ] Q-learning level issue scripts consolidated with root
- [ ] Single `utils/gitlab_issue_manager.py` or similar
- [ ] All functionality preserved
- [ ] `pixi run ruff check` passes

**Priority:** Medium
**Estimate:** M

---

#### US-012: Remove Nested Duplicate Directory

**As a** developer navigating the codebase
**I want to** not encounter nested duplicate directory structures
**So that** the repository structure is clear and not confusing

**Acceptance Criteria:**
- [ ] `q-learning-mfcs/q-learning-mfcs/` directory analyzed
- [ ] Unique content (deployment, performance, testing subdirs) moved to appropriate locations
- [ ] Duplicate directory removed
- [ ] No broken imports or references

**Priority:** High
**Estimate:** S

---

### Epic 4: Large File Decomposition

#### US-013: Decompose Streamlit GUI Module

**As a** developer working on the GUI
**I want to** have the 2,348-line `mfc_streamlit_gui.py` split into page modules
**So that** I can work on individual pages without navigating a massive file

**Acceptance Criteria:**
- [ ] `mfc_streamlit_gui.py` split into logical page modules
- [ ] Main entry point imports and orchestrates pages
- [ ] Each page module <500 lines
- [ ] GUI functionality unchanged
- [ ] Verify in browser that all pages work
- [ ] `pixi run ruff check` passes

**Priority:** Medium
**Estimate:** L

---

#### US-014: Extract Base Controller Class

**As a** developer working on ML controllers
**I want to** have a base controller class for common functionality
**So that** the 5 controller implementations (6.2K+ lines total) share common code

**Acceptance Criteria:**
- [ ] `BaseController` class created with common methods
- [ ] `transfer_learning_controller.py` (1,409 lines) refactored
- [ ] `deep_rl_controller.py` (1,368 lines) refactored
- [ ] `federated_learning_controller.py` (1,272 lines) refactored
- [ ] `adaptive_mfc_controller.py` (1,149 lines) refactored
- [ ] `transformer_controller.py` (1,093 lines) refactored
- [ ] Each controller reduced by 20-30%
- [ ] All controller tests pass
- [ ] GPU acceleration preserved

**Priority:** Medium
**Estimate:** L

---

### Epic 5: Simulation/Demo Consolidation

#### US-015: Create Unified Simulation CLI

**As a** user running simulations
**I want to** have a single simulation entry point with options
**So that** I don't have to choose between 8+ demo/simulation scripts

**Acceptance Criteria:**
- [ ] Single `run_simulation.py` CLI created
- [ ] Supports modes: demo, 100h, 1year, gpu, stack, comprehensive
- [ ] Preserves all existing simulation capabilities
- [ ] Old scripts deprecated with redirect messages (or removed)
- [ ] Documentation updated
- [ ] `pixi run ruff check` passes

**Priority:** Medium
**Estimate:** M

---

### Epic 6: Documentation Cleanup

#### US-016: Consolidate Architecture Documentation

**As a** developer understanding the system
**I want to** have a single authoritative architecture document
**So that** I don't read conflicting or duplicate information

**Acceptance Criteria:**
- [ ] `PROJECT_ARCHITECTURE.md` and `SYSTEM_ARCHITECTURE.md` merged
- [ ] Single source of truth for architecture
- [ ] Outdated information removed
- [ ] Cross-references updated

**Priority:** Low
**Estimate:** S

---

#### US-017: Archive Phase Reports

**As a** developer reviewing project history
**I want to** phase completion reports archived to `docs/history/`
**So that** root directory is cleaner and history is preserved

**Acceptance Criteria:**
- [ ] `PHASES_1_4_COMPLETION_REPORT.md` moved to `docs/history/`
- [ ] `PHASE_5_LITERATURE_VALIDATION_REPORT.md` moved to `docs/history/`
- [ ] `TDD_Agent_54_Final_Report.md` moved to `docs/history/`
- [ ] `TDD_DEPLOYMENT_FINAL_REPORT.md` moved to `docs/history/`
- [ ] `DATA_PERSISTENCE_TDD_FINAL_REPORT.md` moved to `docs/history/`
- [ ] Demo markdown files moved to `docs/examples/`
- [ ] References updated

**Priority:** Low
**Estimate:** S

---

### Epic 7: Test Organization

#### US-018: Reorganize Test Directory Structure

**As a** developer running tests
**I want to** test files organized to mirror source structure
**So that** I can easily find tests for specific modules

**Acceptance Criteria:**
- [ ] Tests in `.claude/hooks/tests/` moved to main `tests/`
- [ ] Test directory structure mirrors `src/` structure
- [ ] No duplicate test files
- [ ] All tests discoverable by pytest
- [ ] CI configuration updated
- [ ] All tests pass

**Priority:** Medium
**Estimate:** M

---

### Epic 8: Scripts Organization

#### US-019: Consolidate Scripts Directory

**As a** developer using utility scripts
**I want to** scripts organized by purpose
**So that** I can find and use them easily

**Acceptance Criteria:**
- [ ] `scripts/` directory organized into subdirectories (e.g., `scripts/gpu/`, `scripts/ci/`, `scripts/utils/`)
- [ ] Large script `tdd_worktree_manager.py` (36KB) evaluated for decomposition
- [ ] `scripts/ralph/` purpose documented or removed if unused
- [ ] All scripts still functional
- [ ] `pixi run ruff check` passes

**Priority:** Low
**Estimate:** S

---

#### US-020: Clarify Ralph Directory

**As a** developer exploring the codebase
**I want to** understand what `scripts/ralph/` is
**So that** I know if it should be kept, removed, or documented

**Acceptance Criteria:**
- [ ] `scripts/ralph/` purpose determined
- [ ] If external dependency: documented in README
- [ ] If unused: removed
- [ ] If active: properly integrated with documentation

**Priority:** Low
**Estimate:** S

---

## Technical Considerations

### Architecture Constraints
- Must preserve the 5-phase enhancement architecture (see `architecture/brownfield-mfc-enhancement-architecture.md`)
- Must maintain GPU acceleration (8400x speedup)
- Must preserve backward compatibility with existing Q-tables and configurations
- Must maintain queue-based streaming architecture (Phase 1 data optimization)

### Database/Data Changes
- No schema changes required
- Q-table format must remain compatible
- Literature database unaffected

### API Endpoints
- No external API changes
- Internal module interfaces may change (document breaking changes)

### Security Considerations
- No security implications
- Rate limiting and input validation preserved

### Testing Strategy
- Run full test suite after each user story
- Verify GPU acceleration benchmarks
- Test GUI in browser for UI-related changes
- Maintain or improve test coverage

---

## Out of Scope

- **Rewriting core algorithms** - Only refactoring, not reimplementation
- **Changing the GUI framework** - Streamlit remains the GUI framework
- **Migrating to different ML frameworks** - PyTorch/JAX usage unchanged
- **Breaking API compatibility** - External interfaces preserved
- **Performance optimization** - Focus is organization, not speed improvements
- **Adding new features** - This is cleanup only
- **Dependency upgrades** - Unless directly related to removing bloat

---

## Dependencies

### External Systems
- GitLab CI/CD (for test validation)
- pixi environment (for running tests)

### Internal Dependencies
- Architecture documents must be consulted before major changes
- Q-learning configuration system is central to many modules

### Third-party Services
- None affected

---

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Breaking GPU acceleration | High | Low | Benchmark before/after each controller change |
| Breaking Q-table compatibility | High | Low | Test loading existing Q-tables after changes |
| Introducing import errors | Medium | Medium | Run full test suite after each consolidation |
| Losing functionality in merges | Medium | Medium | Careful diff review; comprehensive testing |
| Breaking CI pipeline | Medium | Low | Test CI locally before pushing |
| Merge conflicts with active development | Medium | Medium | Coordinate with team; work on feature branches |

---

## Implementation Order

**Phase 1: Quick Wins (Low Risk)** - Stories: US-009, US-010, US-012
- Remove stubs, move misplaced tests, remove nested duplicate

**Phase 2: Configuration Cleanup** - Stories: US-001, US-002
- Consolidate config I/O and SSL variants

**Phase 3: Duplicate Elimination** - Stories: US-004, US-005, US-006, US-007
- Merge PDF generators, sensor fusion, flow rate, Q-learning variants

**Phase 4: Dashboard Consolidation** - Story: US-008
- Unify dashboard components

**Phase 5: Issue Scripts Cleanup** - Story: US-011
- Consolidate GitLab issue management

**Phase 6: Large File Refactoring** - Stories: US-013, US-014
- Split GUI, extract base controller

**Phase 7: Simulation Consolidation** - Story: US-015
- Create unified simulation CLI

**Phase 8: Documentation & Tests** - Stories: US-016, US-017, US-018
- Archive docs, reorganize tests

**Phase 9: Scripts Cleanup** - Stories: US-019, US-020
- Organize scripts directory

---

## Appendix: Files to Remove/Archive

### Immediate Removal (Stubs/Dead Code)
```
/home/uge/mfc-project/create_https_issue.py
/home/uge/mfc-project/create_https_enhancement_issue.py
/home/uge/mfc-project/check_issues.py
```

### Move to Tests Directory
```
/home/uge/mfc-project/q-learning-mfcs/src/test_debug_mode.py
/home/uge/mfc-project/q-learning-mfcs/src/test_epsilon_fix.py
/home/uge/mfc-project/q-learning-mfcs/src/test_gitlab_cli.py
/home/uge/mfc-project/q-learning-mfcs/src/test_optimized_config.py
/home/uge/mfc-project/q-learning-mfcs/src/test_path_outputs.py
/home/uge/mfc-project/q-learning-mfcs/src/test_qtable_loading.py
```

### Archive to docs/history/
```
PHASES_1_4_COMPLETION_REPORT.md
PHASE_5_LITERATURE_VALIDATION_REPORT.md
TDD_Agent_54_Final_Report.md
TDD_DEPLOYMENT_FINAL_REPORT.md
DATA_PERSISTENCE_TDD_FINAL_REPORT.md
```

### Consolidation Targets
```
config_io.py + config_io_fix.py + config_io_fixed.py -> config_io.py
ssl_config.py + ssl_dev_config.py -> ssl_config.py
generate_pdf_report.py + generate_enhanced_pdf_report.py -> generate_pdf_report.py
sensor_fusion.py + advanced_sensor_fusion.py -> sensor_fusion.py
flow_rate_optimization.py + flow_rate_optimization_realistic.py -> flow_rate_optimization.py
```

---

Generated with [Claude Code](https://claude.com/claude-code)
