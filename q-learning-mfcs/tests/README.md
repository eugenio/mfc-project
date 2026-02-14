# MFC Q-Learning Project - Test Suite

Comprehensive test suite for the MFC Q-learning project with ~492 test files across 44 subdirectories.

## Quick Start

```bash
# Run all tests with coverage (batched to avoid OOM)
pixi run test-coverage

# Run fast subset (skip coverage-extra tests)
pixi run -e default python -m pytest -m "not coverage_extra" -q

# Run a single subdirectory
pixi run -e default python -m pytest q-learning-mfcs/tests/config/ -v

# Run with old single-process mode (may OOM on large suites)
pixi run test-coverage-single
```

## Directory Structure

```
tests/
├── conftest.py                    # Root conftest — sys.modules leak guard
├── run_tests.py                   # Legacy unittest runner
├── adaptive/                      # Adaptive controller tests
├── analysis/                      # Analysis module tests
├── apptest/                       # Streamlit AppTest tests
├── biofilm_kinetics/              # Biofilm kinetics tests
├── cad/                           # CAD export/viewer tests
├── cathode/                       # Biological cathode tests
├── compliance/                    # Security & compliance tests
├── config/                        # Configuration module tests
├── controllers/                   # Controller tests
├── core/                          # Core model tests
├── deep_rl/                       # Deep RL controller tests
├── deployment/                    # Deployment & service tests
├── e2e/                           # End-to-end workflow tests
├── federated/                     # Federated learning tests
├── gpu/                           # GPU acceleration tests
├── gui/                           # GUI/Streamlit tests
├── integrated/                    # Integration tests
├── mlops/                         # MLOps pipeline tests
├── monitoring/                    # Monitoring & dashboard tests
├── notification_system/           # Notification tests
├── performance/                   # Performance & benchmarking tests
├── playwright/                    # Playwright E2E browser tests
├── qlearning/                     # Q-learning optimization tests
├── simulation/                    # Simulation runner tests
├── smoke/                         # Fast health-check tests
├── visualization/                 # Visualization tests
└── ...                            # + 15 more subdirectories
```

## Pytest Markers

| Marker | Description | Example |
|--------|-------------|---------|
| `unit` | Unit tests | `pytest -m unit` |
| `integration` | Integration tests | `pytest -m integration` |
| `coverage_extra` | Supplemental coverage tests (cov2/cov3/cov_extra) | `pytest -m "not coverage_extra"` |
| `gpu` | GPU-dependent tests | `pytest -m gpu` |
| `slow` | Long-running tests | `pytest -m "not slow"` |
| `smoke` | Fast health checks (<5s each) | `pytest -m smoke` |
| `e2e` | End-to-end workflow tests | `pytest -m e2e` |
| `apptest` | Streamlit AppTest tests | `pytest -m apptest` |
| `playwright` | Playwright browser tests | `pytest -m playwright` |
| `selenium` | Selenium browser tests | `pytest -m selenium` |
| `gui` | GUI tests | `pytest -m gui` |
| `performance` | Performance benchmarks | `pytest -m performance` |
| `mlops` | MLOps integration tests | `pytest -m mlops` |

## Pixi Tasks

| Task | Description |
|------|-------------|
| `test` | Run legacy unittest runner |
| `test-coverage` | Batched coverage run (per-subdirectory, avoids OOM) |
| `test-coverage-single` | Single-process coverage (may OOM) |
| `test-fast` | Unit tests only, skip slow/integration/browser |
| `test-smoke` | Fast health checks |
| `test-e2e` | End-to-end tests |
| `test-gui` | GUI tests |
| `test-config` | Config module tests |
| `test-cad` | CAD tests |
| `test-ci` | CI-friendly run with XML + coverage output |

## Coverage File Naming Convention

| Suffix | Meaning |
|--------|---------|
| `_cov.py` | Primary coverage tests |
| `_cov2.py` | Additional coverage (round 2) |
| `_cov3.py` | Additional coverage (round 3) |
| `_cov4.py` | Additional coverage (round 4) |
| `_cov_extra.py` | Supplemental edge-case coverage |

All `_cov2`, `_cov3`, `_cov4`, and `_cov_extra` files are marked with `@pytest.mark.coverage_extra` and can be excluded for faster test runs.

## Memory Management

The test suite uses module-level `sys.modules` mocking to avoid importing heavy dependencies (torch, numpy, pandas, etc.) in coverage tests. To prevent memory leaks:

1. **Root `conftest.py`** provides two autouse fixtures:
   - `_guard_sys_modules` (session-scoped): snapshots and restores `sys.modules`
   - `_per_test_mock_cleanup` (function-scoped): removes mocks added during each test

2. **Batched coverage runner** (`scripts/run_coverage_batched.py`): runs pytest per-subdirectory with `--cov-append` to keep each process under ~1 GB memory.

3. **Module-level cleanup**: files with `sys.modules` assignments include `_original_modules` snapshot/restore blocks.

## Writing New Tests

- Place tests in the appropriate subdirectory
- Use `@pytest.mark.coverage_extra` for supplemental coverage tests
- If mocking `sys.modules` at module level, add the snapshot/restore pattern:

```python
from unittest.mock import MagicMock

_original_modules = dict(sys.modules)

sys.modules["torch"] = MagicMock()
# ... imports ...

# Restore after imports
for _k in list(sys.modules):
    if _k not in _original_modules and isinstance(sys.modules[_k], MagicMock):
        del sys.modules[_k]
```
