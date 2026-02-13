#!/usr/bin/env python3
"""
GPU-specific Dependency Installation Script

Automatically installs the correct ML framework dependencies based on detected GPU hardware.
Works with pixi to manage environment-specific packages.
"""

import subprocess
import sys
import json
import os
from pathlib import Path


def load_gpu_report():
    """Load the GPU detection report."""
    report_file = Path(__file__).parent / "gpu_detection_report.json"
    if not report_file.exists():
        print("⚠️ GPU detection report not found. Running detection first...")
        subprocess.run([sys.executable, str(Path(__file__).parent / "detect_gpu.py")])
    
    try:
        with open(report_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to load GPU report: {e}")
        return None


def install_dependencies(environment):
    """
    Install dependencies for the specified environment.
    
    Args:
        environment: 'default', 'nvidia', or 'amd'
    """
    print(f"📦 Installing dependencies for {environment} environment...")
    
    try:
        # Use pixi to install environment-specific dependencies
        cmd = ["pixi", "install", "--environment", environment]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Successfully installed {environment} dependencies")
            return True
        else:
            print(f"❌ Failed to install {environment} dependencies:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False


def verify_installation(gpu_type):
    """
    Verify that the correct ML frameworks are installed and working.
    
    Args:
        gpu_type: 'nvidia', 'amd', 'intel', or 'cpu'
    """
    print(f"🔍 Verifying installation for {gpu_type} setup...")
    
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
        result = subprocess.run([sys.executable, "-c", verification_code], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Installation verification successful!")
            print(result.stdout)
            return True
        else:
            print("❌ Installation verification failed:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Verification timed out - this may indicate GPU driver issues")
        return False
    except Exception as e:
        print(f"❌ Verification error: {e}")
        return False


def setup_environment_variables(gpu_type):
    """
    Set up environment variables for the detected GPU type.
    
    Args:
        gpu_type: GPU type from detection
    """
    env_vars = {}
    
    if gpu_type == 'amd':
        env_vars.update({
            'LLVM_PATH': '/opt/rocm/llvm',
            'ROC_ENABLE_PRE_VEGA': '1',
            'HIP_VISIBLE_DEVICES': '0'
        })
    elif gpu_type == 'nvidia':
        env_vars.update({
            'CUDA_VISIBLE_DEVICES': '0',
            'XLA_PYTHON_CLIENT_PREALLOCATE': 'false'  # Prevent memory hogging
        })
    
    if env_vars:
        print(f"🔧 Setting up environment variables for {gpu_type}:")
        for key, value in env_vars.items():
            os.environ[key] = value
            print(f"   {key}={value}")


def main():
    """Main installation workflow."""
    print("🚀 Starting GPU-specific dependency installation...")
    
    # Load GPU detection report
    report = load_gpu_report()
    if not report:
        print("❌ Cannot proceed without GPU detection report")
        return False
    
    gpu_type = report['gpu_info']['type']
    recommended_env = report['recommended_environment']
    
    print("\n📋 Installation Plan:")
    print(f"   GPU Type: {gpu_type}")
    print(f"   Environment: {recommended_env}")
    print(f"   Devices: {len(report['gpu_info']['devices'])}")
    
    # Set up environment variables
    setup_environment_variables(gpu_type)
    
    # Install dependencies
    success = install_dependencies(recommended_env)
    if not success:
        print("❌ Dependency installation failed")
        return False
    
    # Verify installation
    verification_success = verify_installation(gpu_type)
    
    if verification_success:
        print("\n🎉 GPU-specific dependencies successfully installed and verified!")
        print("\n🚀 To use the optimized environment, run:")
        print(f"   pixi shell {recommended_env}")
        
        if gpu_type == 'amd':
            print("\n💡 For ROCm acceleration, ensure ROCm drivers are installed:")
            print("   See: https://rocm.docs.amd.com/projects/install-on-linux/")
        elif gpu_type == 'nvidia':
            print("\n💡 For CUDA acceleration, ensure NVIDIA drivers are installed:")
            print("   Check with: nvidia-smi")
            
        return True
    else:
        print("\n⚠️ Dependencies installed but verification failed")
        print("💡 You may still be able to use CPU-only mode")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)