# Test Suite Documentation

## Overview

The MFC Q-learning project includes a comprehensive test suite with ~492 test files across 44 subdirectories, validating functionality across configuration, controllers, simulation, GPU acceleration, monitoring, deployment, and more. The suite uses **pytest** as its primary test framework with batched coverage collection to manage memory.

## Test Suite Architecture

The test suite is located in `q-learning-mfcs/tests/`:

```text
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

## Running Tests

### Recommended Commands (Pixi)

```bash
# Run all tests with coverage (batched to avoid OOM)
pixi run test-coverage

# Run fast subset (skip coverage-extra tests)
pixi run -e default python -m pytest -m "not coverage_extra" -q

# Run a single subdirectory
pixi run -e default python -m pytest q-learning-mfcs/tests/config/ -v

# Run with old single-process mode (may OOM on large suites)
pixi run test-coverage-single

# Run smoke tests only
pixi run test-smoke

# Run CI-friendly with XML + coverage output
pixi run test-ci
```

### Pixi Tasks

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

The test suite uses module-level `sys.modules` mocking to avoid importing heavy dependencies (torch, numpy, pandas, etc.) in coverage tests. Without safeguards, this can cause 3-4+ GB memory usage and OOM. Three mechanisms prevent this:

### 1. Root `conftest.py` — Module Leak Guard

Located at `q-learning-mfcs/tests/conftest.py`, provides two autouse fixtures:

- **`_guard_sys_modules`** (session-scoped): Snapshots `sys.modules` at session start and restores it at session end, removing any MagicMock entries that were added during the session.
- **`_per_test_mock_cleanup`** (function-scoped): After each test, removes any MagicMock entries added to `sys.modules` during that test, preventing mock leakage between tests.

### 2. Batched Coverage Runner

Located at `scripts/run_coverage_batched.py`, invoked via `pixi run test-coverage`:

1. Clears any existing `.coverage` file
2. Discovers all subdirectories in `q-learning-mfcs/tests/`
3. Runs `pytest <subdir> --cov=../src --cov-append -q -p no:playwright` for each
4. Also runs pytest on any top-level `test_*.py` files
5. Generates `--cov-report=html --cov-report=term` at the end

This keeps each pytest process to <50 test files and ~900 MB memory.

### 3. Module-Level Cleanup Pattern

Files with `sys.modules` assignments include `_original_modules` snapshot/restore blocks:

```python
import sys
from unittest.mock import MagicMock

_original_modules = dict(sys.modules)

sys.modules["torch"] = MagicMock()
# ... imports ...

# Restore after imports
for _k in list(sys.modules):
    if _k not in _original_modules and isinstance(sys.modules[_k], MagicMock):
        del sys.modules[_k]
```

## Writing New Tests

- Place tests in the appropriate subdirectory
- Use `@pytest.mark.coverage_extra` for supplemental coverage tests (cov2/cov3/cov_extra variants)
- If mocking `sys.modules` at module level, use the snapshot/restore pattern shown above
- Use **pytest** (preferred) or **unittest**, never standalone scripts
- Structure tests with **Arrange-Act-Assert**
- Use descriptive names: `test_<behavior>_when_<condition>`
- Mock at boundaries (DB, HTTP, filesystem), never mock the unit under test

## Continuous Integration

```yaml
# Example CI workflow
test:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - name: Install pixi
      uses: prefix-dev/setup-pixi@v0.8.0
    - name: Install dependencies
      run: pixi install
    - name: Run tests with coverage
      run: pixi run test-ci
```

## Troubleshooting

### Common Issues

1. **OOM during coverage run**: Use `pixi run test-coverage` (batched) instead of `test-coverage-single`
2. **Playwright import errors**: The batched runner disables playwright with `-p no:playwright`. If running manually, add this flag.
3. **Import errors**: Ensure pixi environment is activated or use `pixi run`
4. **GPU tests failing**: May indicate missing GPU drivers or libraries; these tests are skipped gracefully
5. **Mock leakage warnings**: The root conftest guards against this, but check for missing `_original_modules` cleanup in new test files
