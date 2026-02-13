# JAX ROCm Installation Guide

## Overview

This guide documents the complete process for installing JAX with ROCm support for AMD GPU acceleration on Ubuntu systems. This installation enables GPU-accelerated Q-learning simulations for MFC (Microbial Fuel Cell) optimization.

## Prerequisites

- Ubuntu 24.04 LTS (or compatible)
- AMD GPU with ROCm support
- ROCm 6.0+ installed on the system
- Python 3.12
- Pixi package manager

## Installation Components

The JAX ROCm setup consists of three main components:

1. **jaxlib** (0.6.0.dev20250728) - Core XLA library with ROCm support
2. **jax_rocm60_plugin** - ROCm 6.0 plugin interface
3. **jax_rocm60_pjrt** - PJRT runtime for AMD GPU acceleration

## Step-by-Step Installation Process

### 1. Prepare Local Wheel Files

First, ensure you have the ROCm wheel files available:

```bash
# Verify wheel files exist
ls -la /home/uge/rocm-jax/jax/dist/
# Expected files:
# - jaxlib-0.6.0.dev20250728-cp312-cp312-manylinux2014_x86_64.whl
# - jax_rocm60_plugin-0.6.0.dev20250728-cp312-cp312-manylinux2014_x86_64.whl  
# - jax_rocm60_pjrt-0.6.0.dev20250728-py3-none-manylinux2014_x86_64.whl
```

### 2. Create Conda Build Environment

```bash
# Add conda-build to pixi environment
pixi add conda-build

# Create directories for conda recipes and packages
mkdir -p conda-recipes/{jaxlib,jax_rocm60_plugin,jax_rocm60_pjrt}
mkdir -p conda-packages local-channel/{linux-64,noarch}
```

### 3. Copy Wheels to Build Directory

```bash
# Copy wheels for conda package creation
cp /home/uge/rocm-jax/jax/dist/*.whl conda-packages/
```

### 4. Create Conda Recipes

#### 4.1 jaxlib Recipe

Create `conda-recipes/jaxlib/meta.yaml`:

```yaml
{% set name = "jaxlib" %}
{% set version = "0.6.0.dev20250728" %}

package:
  name: {{ name|lower }}
  version: {{ version }}

source:
  path: ../../conda-packages/jaxlib-0.6.0.dev20250728-cp312-cp312-manylinux2014_x86_64.whl

build:
  number: 0
  skip: True  # [not linux64]
  script: python -m pip install jaxlib-0.6.0.dev20250728-cp312-cp312-manylinux2014_x86_64.whl -vv --no-deps --no-build-isolation

requirements:
  host:
    - python =3.12
    - pip
  run:
    - python =3.12
    - numpy >=1.22
    - scipy >=1.9
    - ml_dtypes >=0.2.0
    - libgcc-ng
    - libstdcxx-ng

test:
  imports:
    - jaxlib

about:
  home: https://github.com/google/jax
  license: Apache-2.0
  license_family: Apache
  summary: 'XLA library for JAX'
  description: |
    JAXlib is XLA, compiled for your CPU and/or GPU.
  dev_url: https://github.com/google/jax
  doc_url: https://jax.readthedocs.io/
```

#### 4.2 jax_rocm60_plugin Recipe

Create `conda-recipes/jax_rocm60_plugin/meta.yaml`:

```yaml
{% set name = "jax_rocm60_plugin" %}
{% set version = "0.6.0.dev20250728" %}

package:
  name: {{ name|lower|replace("_", "-") }}
  version: {{ version }}

source:
  path: ../../conda-packages/jax_rocm60_plugin-0.6.0.dev20250728-cp312-cp312-manylinux2014_x86_64.whl

build:
  number: 0
  skip: True  # [not linux64]
  script: python -m pip install jax_rocm60_plugin-0.6.0.dev20250728-cp312-cp312-manylinux2014_x86_64.whl -vv --no-deps --no-build-isolation

requirements:
  host:
    - python =3.12
    - pip
  run:
    - python =3.12
    - jaxlib >=0.6.0

test:
  imports:
    - jax_plugins.xla_rocm60

about:
  home: https://github.com/google/jax
  license: Apache-2.0
  license_family: Apache
  summary: 'JAX ROCm 6.0 plugin for AMD GPU support'
  description: |
    JAX plugin for ROCm 6.0 AMD GPU acceleration.
  dev_url: https://github.com/google/jax
```

#### 4.3 jax_rocm60_pjrt Recipe

Create `conda-recipes/jax_rocm60_pjrt/meta.yaml`:

```yaml
{% set name = "jax_rocm60_pjrt" %}
{% set version = "0.6.0.dev20250728" %}

package:
  name: {{ name|lower|replace("_", "-") }}
  version: {{ version }}

source:
  path: ../../conda-packages/jax_rocm60_pjrt-0.6.0.dev20250728-py3-none-manylinux2014_x86_64.whl

build:
  number: 0
  skip: True  # [not linux64]
  script: python -m pip install jax_rocm60_pjrt-0.6.0.dev20250728-py3-none-manylinux2014_x86_64.whl -vv --no-deps --no-build-isolation

requirements:
  host:
    - python >=3.8
    - pip
  run:
    - python >=3.8
    - jaxlib >=0.6.0

about:
  home: https://github.com/google/jax
  license: Apache-2.0
  license_family: Apache
  summary: 'JAX ROCm 6.0 PJRT runtime for AMD GPU support'
  description: |
    JAX PJRT runtime for ROCm 6.0 AMD GPU acceleration.
  dev_url: https://github.com/google/jax
```

### 5. Build Conda Packages (Detached Mode)

Build all packages in background to avoid timeouts:

```bash
# Build jaxlib (takes ~2.5 minutes)
nohup pixi run conda build conda-recipes/jaxlib --output-folder local-channel --no-test > jaxlib_build.log 2>&1 &

# Build plugin (fast)
nohup pixi run conda build conda-recipes/jax_rocm60_plugin --output-folder local-channel --no-test > plugin_build.log 2>&1 &

# Build PJRT runtime (takes ~2 minutes)
nohup pixi run conda build conda-recipes/jax_rocm60_pjrt --output-folder local-channel --no-test > pjrt_build.log 2>&1 &

# Monitor progress
ps aux | grep "conda build" | grep -v grep
tail -f *_build.log
```

### 6. Update Local Conda Channel

```bash
# Index the local channel with built packages
pixi run conda index local-channel

# Verify packages were built
ls -lh local-channel/linux-64/*.conda
# Expected output:
# - jaxlib-0.6.0.dev20250728-py312_0.conda (64M)
# - jax-rocm60-plugin-0.6.0.dev20250728-py312_0.conda (4.6M)  
# - jax-rocm60-pjrt-0.6.0.dev20250728-py313_0.conda (48M)
```

### 7. Configure Pixi Environment

Update `pixi.toml` to use the local conda channel and install the packages:

```toml
channels = ["file:///home/uge/mfc-project/local-channel","conda-forge","https://conda.modular.com/max-nightly/", "https://repo.prefix.dev/modular", "https://repo.prefix.dev/mojo", "https://repo.prefix.dev/modular-community"]

[feature.amd.dependencies]
jaxlib = "==0.6.0.dev20250728"

[feature.amd.pypi-dependencies]
jax = "==0.6.0"
streamlit-autorefresh = ">=1.0.1, <2"
jax_rocm60_plugin = { path = "/home/uge/rocm-jax/jax/dist/jax_rocm60_plugin-0.6.0.dev20250728-cp312-cp312-manylinux2014_x86_64.whl" }
jax_rocm60_pjrt = { path = "/home/uge/rocm-jax/jax/dist/jax_rocm60_pjrt-0.6.0.dev20250728-py3-none-manylinux2014_x86_64.whl" }

[environments]
amd = ["amd"]
rocm = ["amd"]
```

### 8. Install Packages

```bash
# Install the complete JAX ROCm stack
pixi install

# The installation will show:
# ✔ The default environment has been installed.
```

## Verification and Testing

### 9. Test Installation

Create a test script to verify functionality:

```python
#!/usr/bin/env python3
"""Test JAX ROCm installation."""

import jax
import jax.numpy as jnp

# Test versions
print(f"JAX version: {jax.__version__}")
print(f"jaxlib version: {jax._src.lib.jaxlib_version}")

# Test devices
print(f"Available devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

# Test basic computation
x = jnp.array([1, 2, 3, 4])
y = jnp.array([2, 3, 4, 5])
result = x + y
print(f"Basic computation: {result}")

# Test JIT compilation
@jax.jit
def square(x):
    return x * x

result = square(jnp.array([1.0, 2.0, 3.0]))
print(f"JIT compilation: {result}")

# Test ROCm plugin import
try:
    import jax_plugins.xla_rocm60
    print("✅ JAX ROCm 6.0 plugin imported successfully")
except ImportError as e:
    print(f"❌ ROCm plugin import failed: {e}")
```

Run the test:

```bash
pixi run --environment amd python test_jax_rocm.py
```

Expected successful output:
```
JAX version: 0.6.0
jaxlib version: 0.6.0.dev20250728
Available devices: [RocmDevice(id=0)]
Default backend: gpu
Basic computation: [3 5 7 9]
JIT compilation: [1. 4. 9.]
✅ JAX ROCm 6.0 plugin imported successfully
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Version Compatibility Errors

**Problem**: `jaxlib version X is newer than and incompatible with jax version Y`

**Solution**: Ensure JAX and jaxlib versions are compatible:
- Use JAX 0.6.0 with jaxlib 0.6.0.dev20250728
- Avoid mixing PyPI and conda versions

#### 2. Plugin Not Found

**Problem**: `JAX plugin jax_rocm60_plugin version X is not compatible with jaxlib version Y`

**Solution**: 
- Verify all components are from the same build date
- Check that plugins are installed as wheels, not conda packages

#### 3. No ROCm Device Detected

**Problem**: Only `CpuDevice(id=0)` appears in `jax.devices()`

**Solution**:
- Verify ROCm is properly installed: `rocm-smi`
- Check environment variables: `export ROCM_PATH=/opt/rocm`
- Ensure user is in `render` group: `sudo usermod -a -G render $USER`

#### 4. Build Timeouts

**Problem**: Conda builds timeout during compilation

**Solution**:
- Use detached mode: `nohup ... &`
- Monitor with: `ps aux | grep conda`
- Check logs: `tail -f *_build.log`

### Environment Variables

Set these for optimal ROCm performance:

```bash
export ROCM_PATH=/opt/rocm
export HIP_VISIBLE_DEVICES=0
export JAX_PLATFORMS=gpu  # or cpu for fallback
```

## File Structure

After successful installation, your project should have:

```
project-root/
├── conda-packages/
│   ├── jaxlib-0.6.0.dev20250728-cp312-cp312-manylinux2014_x86_64.whl
│   ├── jax_rocm60_plugin-0.6.0.dev20250728-cp312-cp312-manylinux2014_x86_64.whl
│   └── jax_rocm60_pjrt-0.6.0.dev20250728-py3-none-manylinux2014_x86_64.whl
├── conda-recipes/
│   ├── jaxlib/meta.yaml
│   ├── jax_rocm60_plugin/meta.yaml
│   └── jax_rocm60_pjrt/meta.yaml
├── local-channel/
│   ├── linux-64/
│   │   ├── jaxlib-0.6.0.dev20250728-py312_0.conda
│   │   ├── jax-rocm60-plugin-0.6.0.dev20250728-py312_0.conda
│   │   └── jax-rocm60-pjrt-0.6.0.dev20250728-py313_0.conda
│   └── noarch/
└── pixi.toml (updated with JAX ROCm configuration)
```

## Performance Benefits

With JAX ROCm successfully installed, Q-learning simulations can expect:

- **10-100x speedup** for matrix operations
- **GPU-accelerated JIT compilation** for control algorithms  
- **Parallel batch processing** for multiple MFC simulations
- **Memory-efficient operations** with XLA optimization

## Maintenance

### Updating JAX ROCm

To update to newer versions:

1. Obtain new wheel files
2. Update version numbers in conda recipes
3. Rebuild conda packages
4. Update pixi.toml version specifications
5. Reinstall with `pixi install`

### Backup

Keep backups of:
- Original wheel files (`/home/uge/rocm-jax/jax/dist/`)
- Working conda recipes (`conda-recipes/`)
- Built packages (`local-channel/linux-64/`)
- Working `pixi.toml` configuration

## References

- [JAX Documentation](https://jax.readthedocs.io/)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [Pixi Documentation](https://pixi.sh/)
- [Conda Build Documentation](https://docs.conda.io/projects/conda-build/)

---

**Document Version**: 1.0  
**Last Updated**: 2025-07-28  
**Tested On**: Ubuntu 24.04 LTS, ROCm 6.0, Python 3.12