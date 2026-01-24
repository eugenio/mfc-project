#!/usr/bin/env python3
"""GPU-specific Dependency Installation Script.

Automatically installs the correct ML framework dependencies based on detected GPU hardware.
Works with pixi to manage environment-specific packages.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def load_gpu_report():
    """Load the GPU detection report."""
    report_file = Path(__file__).parent / "gpu_detection_report.json"
    if not report_file.exists():
        subprocess.run(
            [sys.executable, str(Path(__file__).parent / "detect_gpu.py")],
            check=False,
        )

    try:
        with open(report_file) as f:
            return json.load(f)
    except Exception:
        return None


def install_dependencies(environment) -> bool | None:
    """Install dependencies for the specified environment.

    Args:
        environment: 'default', 'nvidia', or 'amd'

    """
    try:
        # Use pixi to install environment-specific dependencies
        cmd = ["pixi", "install", "--environment", environment]
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)

        return result.returncode == 0

    except Exception:
        return False


def verify_installation(gpu_type) -> bool | None:
    """Verify that the correct ML frameworks are installed and working.

    Args:
        gpu_type: 'nvidia', 'amd', 'intel', or 'cpu'

    """
    # Test basic imports
    verification_code = """
import sys
try:
    import jax
    import jax.numpy as jnp
    print(f"✅ JAX {jax.__version__} imported successfully")

    # Test device detection
    devices = jax.devices()
    print(f"🎮 JAX devices: {[str(d) for d in devices]}")

    # Test basic computation
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.sum(x)
    print(f"🧮 JAX computation test: sum([1,2,3]) = {y}")

except ImportError as e:
    print(f"❌ JAX import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"⚠️ JAX test failed: {e}")
    sys.exit(1)

print("✅ All verifications passed!")
"""

    try:
        result = subprocess.run(
            [sys.executable, "-c", verification_code],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def setup_environment_variables(gpu_type) -> None:
    """Set up environment variables for the detected GPU type.

    Args:
        gpu_type: GPU type from detection

    """
    env_vars = {}

    if gpu_type == "amd":
        env_vars.update(
            {
                "LLVM_PATH": "/opt/rocm/llvm",
                "ROC_ENABLE_PRE_VEGA": "1",
                "HIP_VISIBLE_DEVICES": "0",
            },
        )
    elif gpu_type == "nvidia":
        env_vars.update(
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",  # Prevent memory hogging
            },
        )

    if env_vars:
        for key, value in env_vars.items():
            os.environ[key] = value


def main() -> bool:
    """Main installation workflow."""
    # Load GPU detection report
    report = load_gpu_report()
    if not report:
        return False

    gpu_type = report["gpu_info"]["type"]
    recommended_env = report["recommended_environment"]

    # Set up environment variables
    setup_environment_variables(gpu_type)

    # Install dependencies
    success = install_dependencies(recommended_env)
    if not success:
        return False

    # Verify installation
    verification_success = verify_installation(gpu_type)

    if verification_success:
        if gpu_type in {"amd", "nvidia"}:
            pass

        return True
    return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception:
        sys.exit(1)
