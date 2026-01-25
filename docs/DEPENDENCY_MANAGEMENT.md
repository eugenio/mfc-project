# Dependency Management Guide

This document describes the pixi dependency structure for the mfc-project, including environment configurations, package sources, and best practices for adding new dependencies.

## Overview

The project uses [pixi](https://pixi.sh/) for dependency management, which provides:

- Reproducible environments via lock files
- Environment isolation via `PYTHONNOUSERSITE=1`
- Support for both conda-forge and PyPI packages
- Feature-based modular dependency organization

## Environment Configurations

### Available Environments

| Environment | Features Included | Use Case |
|-------------|-------------------|----------|
| `runtime` | (base only) | Minimal production runtime |
| `test-runtime` | dev | Minimal testing without hooks |
| `dev` | dev, tools, project | Standard development & testing |
| `full-dev` | dev, tools, project, docs-build | Full development with docs |
| `gui-dev` | dev, gui, tools | GUI/Streamlit development |
| `ml-research` | dev, ml, advanced-ml, gui | ML experimentation |
| `gsm-research` | dev, gsm, ml, gui | Genome-scale metabolic modeling |
| `api-server` | api, project | REST API deployment |
| `mojo-dev` | dev, modular, tools | Mojo/Modular development |
| `amd-gpu` | amd-gpu, ml, api, gui, dev | AMD GPU (ROCm) acceleration |
| `nvidia-gpu` | nvidia-gpu, ml | NVIDIA GPU (CUDA) acceleration |
| `complete` | all features | Full environment (avoid in production) |

### Recommended Environment for Tests

**Use the `dev` environment for running tests:**

```bash
# Run tests in isolated environment
PYTHONNOUSERSITE=1 pixi run -e dev pytest q-learning-mfcs/tests -v

# Run specific test categories
pixi run -e dev pytest q-learning-mfcs/tests/config -v
pixi run -e dev pytest q-learning-mfcs/tests/gui -v
pixi run -e dev pytest q-learning-mfcs/tests/integration -v
```

The `test-runtime` environment is a minimal subset for fast CI execution.

## Feature to Dependency Mapping

### Core Dependencies (`[dependencies]`)

Base packages available in all environments:

| Package | Version | Purpose |
|---------|---------|---------|
| python | >=3.12.11,\<3.13 | Python runtime |
| numpy | >=2.3.1,\<3 | Numerical computing |
| scipy | >=1.16.0,\<2 | Scientific computing |
| pandas | >=2.3.1,\<3 | Data analysis |
| matplotlib | >=3.10.3,\<4 | Plotting |
| pyyaml | >=6.0.2,\<7 | YAML parsing |
| streamlit | >=1.47.1,\<2 | Web UI framework |
| plotly | >=6.2.0,\<7 | Interactive plots |
| altair | >=5.5.0,\<6 | Declarative visualization |
| pydantic | >=2.10.4 | Data validation |
| fastapi | >=0.115.14,\<0.117 | Web framework |
| uvicorn | >=0.35.0,\<0.36 | ASGI server |
| pytest | >=8.4.1,\<9 | Testing framework |
| pytest-cov | >=6.2.1,\<7 | Coverage reporting |
| ruff | >=0.12.7,\<0.13 | Linting |
| mypy | >=1.17.1,\<2 | Type checking |
| selenium | >=4.34.2,\<5 | Browser automation |
| webdriver-manager | >=4.0.2,\<5 | WebDriver management |
| glab | >=1.62.0,\<2 | GitLab CLI |

### Feature: dev (Development Tools)

**Conda packages** (`[feature.dev.dependencies]`):

- pytest, pytest-cov - Testing
- ruff, mypy, black - Linting & formatting
- pre-commit - Git hooks
- bandit, safety - Security scanning
- types-\* - Type stubs (pyyaml, psutil, requests, jsonschema, seaborn)
- yamllint - YAML validation

**PyPI packages** (`[feature.dev.pypi-dependencies]`):

- build - Package building
- pip-audit - Dependency auditing
- pytest-benchmark - Performance testing
- detect-secrets - Secret detection
- mdformat - Markdown formatting
- pymupdf - PDF processing
- streamlit-autorefresh - Auto-refresh for Streamlit
- selenium, webdriver-manager - Browser testing

### Feature: docs-build (Documentation)

**Conda packages** (`[feature.docs-build.dependencies]`):

- tectonic - LaTeX processing
- pandoc - Document conversion
- mdformat - Markdown formatting

### Feature: tools (Project Tools)

**Conda packages** (`[feature.tools.dependencies]`):

- git-lfs - Large file storage
- gh - GitHub CLI
- nox - Task automation

### Feature: gui (GUI Development)

**Conda packages** (`[feature.gui.dependencies]`):

- streamlit - Web UI framework
- plotly - Interactive plotting
- seaborn - Statistical visualization

### Feature: ml (Machine Learning)

**Conda packages** (`[feature.ml.dependencies]`):

- optuna - Hyperparameter optimization

### Feature: gsm (Genome-Scale Modeling)

**Conda packages** (`[feature.gsm.dependencies]`):

- python >=3.12.11,\<3.13
- glpk - Linear programming solver

**PyPI packages** (`[feature.gsm.pypi-dependencies]`):

- cobra - COBRApy metabolic modeling
- mackinac - ModelSEED bridge
- escher - Pathway visualization
- memote - Model testing
- swiglpk - GLPK Python bindings

### Feature: modular (Mojo Runtime)

**Conda packages** (`[feature.modular.dependencies]`):

- modular - Modular runtime
- magic - Modular tooling

### Feature: api (API Server)

**PyPI packages** (`[feature.api.pypi-dependencies]`):

- fastapi - Web framework
- uvicorn[standard] - ASGI server
- websockets - WebSocket support
- cryptography - TLS/encryption
- pyjwt - JWT authentication
- python-multipart - Form handling
- starlette - ASGI toolkit

### Feature: project (Project Management)

**PyPI packages** (`[feature.project.pypi-dependencies]`):

- python-gitlab - GitLab API client

### Feature: advanced-ml (Advanced ML)

**PyPI packages** (`[feature.advanced-ml.pypi-dependencies]`):

- ray[tune] - Distributed ML
- scikit-optimize - Bayesian optimization

### Feature: amd-gpu (AMD GPU Acceleration)

**PyPI packages** (`[feature.amd-gpu.pypi-dependencies]`):

- jax==0.5.0 - JAX (frozen version)
- jaxlib==0.5.0 - JAXlib (frozen version)
- jax-rocm60-plugin==0.5.0 - ROCm plugin
- jax-rocm60-pjrt==0.5.0 - ROCm PJRT

### Feature: nvidia-gpu (NVIDIA GPU Acceleration)

**Conda packages** (`[feature.nvidia-gpu.dependencies]`):

- cuda-toolkit >=12.0
- cudnn >=8.0

**PyPI packages** (`[feature.nvidia-gpu.pypi-dependencies]`):

- jax[cuda12] - JAX with CUDA
- jaxlib[cuda12] - JAXlib with CUDA
- cupy-cuda12x - CuPy GPU arrays
- torch - PyTorch
- nvidia-ml-py3 - NVIDIA ML bindings
- streamlit-autorefresh - Auto-refresh

## Core PyPI Dependencies (`[pypi-dependencies]`)

Available in all environments:

- requests - HTTP library
- python-crontab - Cron job management
- types-python-crontab - Type stubs
- pubmedclient - PubMed API client
- cchooks - Claude Code hooks

## Adding New Dependencies

### Decision Tree

1. **Is the package on conda-forge?**

   - Yes: Add to `[feature.<name>.dependencies]` (preferred)
   - No: Add to `[feature.<name>.pypi-dependencies]`

1. **Which feature should it go in?**

   - Testing only: `feature.dev`
   - Documentation: `feature.docs-build`
   - GUI/visualization: `feature.gui`
   - Machine learning: `feature.ml` or `feature.advanced-ml`
   - API/web server: `feature.api`
   - All environments: Base `[dependencies]` or `[pypi-dependencies]`

### Adding a Conda Package

Edit `pixi.toml`:

```toml
[feature.dev.dependencies]
new-package = ">=1.0.0,<2"
```

### Adding a PyPI Package

Edit `pixi.toml`:

```toml
[feature.dev.pypi-dependencies]
new-package = ">=1.0.0, <2"
```

### Verification After Adding

```bash
# Verify pixi.toml is valid
pixi install --dry-run

# Install the environment
pixi install -e dev

# Run tests to verify
PYTHONNOUSERSITE=1 pixi run -e dev pytest q-learning-mfcs/tests -v
```

## Version Pinning Strategy

### Recommended Practices

1. **Compatible release specifiers**: `>=X.Y.Z,<X+1` (allows patch updates)
1. **Exact versions**: Only for known incompatibilities (e.g., JAX/ROCm)
1. **Wildcards**: `*` only for stable packages with infrequent breaking changes

### Examples

```toml
# Good: Allows patch updates within major version
numpy = ">=2.3.1,<3"

# Good: Exact version for compatibility-sensitive packages
jax = "==0.5.0"

# Avoid: Too permissive
some-package = "*"
```

## Environment Isolation

### Running Tests in Isolation

Always use `PYTHONNOUSERSITE=1` to prevent user site-packages from being used:

```bash
# Correct: Isolated test run
PYTHONNOUSERSITE=1 pixi run -e dev pytest q-learning-mfcs/tests -v

# Incorrect: May use packages from ~/.local
pixi run -e dev pytest q-learning-mfcs/tests -v
```

### CI/CD Best Practices

1. Always set `PYTHONNOUSERSITE=1` in CI
1. Use `pixi run -e dev pytest` (not system pytest)
1. Verify no imports from user site-packages

### Checking Package Source

```bash
# Check where a package is installed
pixi run -e dev python -c "import numpy; print(numpy.__file__)"

# List all installed packages
pixi run -e dev pip list

# Check for user site-packages (should be empty with PYTHONNOUSERSITE=1)
pixi run -e dev python -c "import site; print(site.getusersitepackages())"
```

## Troubleshooting

### Common Issues

1. **Import error in isolated mode but works normally**

   - Package is in user site-packages but not in pixi.toml
   - Solution: Add the missing package to appropriate feature section

1. **Version conflict during install**

   - Check if multiple features pin incompatible versions
   - Use `pixi install -e <env> --verbose` for details

1. **Package not found on conda-forge**

   - Use PyPI fallback: `[feature.<name>.pypi-dependencies]`

### Debugging Commands

```bash
# Check pixi.toml validity
pixi install --dry-run

# Verbose install for debugging
pixi install -e dev --verbose

# List environment packages
pixi run -e dev pip list

# Check import paths
pixi run -e dev python -c "import sys; print('\n'.join(sys.path))"
```

## Related Documentation

- [Test Suite Documentation](TEST_SUITE_DOCUMENTATION.md)
- [GPU Acceleration Guide](GPU_ACCELERATION_GUIDE.md)

______________________________________________________________________

Generated with [Claude Code](https://claude.ai/code)
