#!/usr/bin/env python3
"""
GPU Detection Script for Pixi Environment Configuration

Detects the GPU hardware present in the system and determines which
ML framework dependencies should be installed (ROCm for AMD, CUDA for NVIDIA, CPU-only for Intel/others).
"""

import subprocess
import sys
import json
from pathlib import Path


def detect_gpu_type():
    """
    Detect the primary GPU type in the system.
    
    Returns:
        str: 'amd', 'nvidia', 'intel', or 'cpu' based on detected hardware
    """
    gpu_info = {
        'type': 'cpu',  # Default fallback
        'devices': [],
        'details': {}
    }
    
    # Check for NVIDIA GPUs first
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            devices = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    name, memory = line.split(',', 1)
                    devices.append({
                        'name': name.strip(),
                        'memory_mb': int(memory.strip())
                    })
            if devices:
                gpu_info.update({
                    'type': 'nvidia',
                    'devices': devices,
                    'details': {'driver_available': True}
                })
                print(f"🎮 Detected NVIDIA GPU(s): {[d['name'] for d in devices]}")
                return gpu_info
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, ValueError):
        pass
    
    # Check for AMD ROCm support
    try:
        result = subprocess.run(['rocm-smi', '--showproductname'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and 'GPU' in result.stdout:
            # Parse ROCm output
            devices = []
            for line in result.stdout.split('\n'):
                if 'GPU' in line and ':' in line:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        devices.append({
                            'name': parts[1].strip(),
                            'rocm_available': True
                        })
            if devices:
                gpu_info.update({
                    'type': 'amd',
                    'devices': devices,
                    'details': {'rocm_available': True}
                })
                print(f"🎮 Detected AMD GPU(s) with ROCm: {[d['name'] for d in devices]}")
                return gpu_info
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Check for AMD GPUs using lspci (fallback)
    try:
        result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            amd_lines = [line for line in result.stdout.split('\n') 
                        if 'VGA' in line and ('AMD' in line or 'ATI' in line or 'Radeon' in line)]
            if amd_lines:
                devices = []
                for line in amd_lines:
                    devices.append({
                        'name': line.split(':')[-1].strip() if ':' in line else line.strip(),
                        'rocm_available': False  # Unknown without rocm-smi
                    })
                gpu_info.update({
                    'type': 'amd',
                    'devices': devices,
                    'details': {'rocm_available': False}
                })
                print(f"🎮 Detected AMD GPU(s): {[d['name'] for d in devices]}")
                return gpu_info
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Check for Intel GPUs
    try:
        result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            intel_lines = [line for line in result.stdout.split('\n') 
                          if 'VGA' in line and ('Intel' in line)]
            if intel_lines:
                devices = []
                for line in intel_lines:
                    devices.append({
                        'name': line.split(':')[-1].strip() if ':' in line else line.strip(),
                        'integrated': True
                    })
                gpu_info.update({
                    'type': 'intel',
                    'devices': devices,
                    'details': {'integrated': True}
                })
                print(f"🎮 Detected Intel GPU(s): {[d['name'] for d in devices]}")
                return gpu_info
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Fallback to CPU-only
    print("🖥️ No discrete GPU detected, using CPU-only configuration")
    return gpu_info


def get_recommended_environment():
    """
    Get the recommended pixi environment based on detected GPU.
    
    Returns:
        str: Environment name ('amd', 'nvidia', 'default')
    """
    gpu_info = detect_gpu_type()
    gpu_type = gpu_info['type']
    
    if gpu_type == 'nvidia':
        return 'nvidia'
    elif gpu_type == 'amd':
        return 'amd'
    else:
        return 'default'


def check_gpu_drivers():
    """
    Check if appropriate GPU drivers are installed.
    
    Returns:
        dict: Driver status information
    """
    driver_status = {
        'nvidia': False,
        'amd_rocm': False,
        'amd_opencl': False
    }
    
    # Check NVIDIA drivers
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
        driver_status['nvidia'] = result.returncode == 0
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Check ROCm
    try:
        result = subprocess.run(['rocm-smi'], capture_output=True, timeout=5)
        driver_status['amd_rocm'] = result.returncode == 0
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Check OpenCL
    try:
        result = subprocess.run(['clinfo'], capture_output=True, timeout=5)
        driver_status['amd_opencl'] = result.returncode == 0 and b'AMD' in result.stdout
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    return driver_status


def main():
    """Main function to detect GPU and recommend configuration."""
    print("🔍 Detecting GPU hardware for optimal ML framework configuration...")
    
    # Detect GPU
    gpu_info = detect_gpu_type()
    recommended_env = get_recommended_environment()
    
    # Check drivers
    driver_status = check_gpu_drivers()
    
    # Create comprehensive report
    report = {
        'gpu_info': gpu_info,
        'recommended_environment': recommended_env,
        'driver_status': driver_status,
        'recommendations': []
    }
    
    # Generate recommendations
    if gpu_info['type'] == 'nvidia':
        if driver_status['nvidia']:
            report['recommendations'].append("✅ Use 'pixi shell nvidia' for CUDA acceleration")
            report['recommendations'].append("✅ NVIDIA drivers detected and working")
        else:
            report['recommendations'].append("⚠️ Install NVIDIA drivers for GPU acceleration")
            report['recommendations'].append("💡 Fallback to 'pixi shell default' for CPU-only")
    
    elif gpu_info['type'] == 'amd':
        if driver_status['amd_rocm']:
            report['recommendations'].append("✅ Use 'pixi shell amd' for ROCm acceleration")
            report['recommendations'].append("✅ ROCm drivers detected and working")
        else:
            report['recommendations'].append("⚠️ Install ROCm drivers for GPU acceleration")
            report['recommendations'].append("💡 Instructions: https://rocm.docs.amd.com/projects/install-on-linux/")
            report['recommendations'].append("💡 Fallback to 'pixi shell default' for CPU-only")
    
    else:
        report['recommendations'].append("💻 Use 'pixi shell default' for CPU-only ML workloads")
        if gpu_info['type'] == 'intel':
            report['recommendations'].append("💡 Consider Intel Extension for PyTorch for integrated GPU acceleration")
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 GPU DETECTION SUMMARY")
    print("="*60)
    print(f"GPU Type: {gpu_info['type'].upper()}")
    print(f"Recommended Environment: {recommended_env}")
    print(f"Number of devices: {len(gpu_info['devices'])}")
    
    if gpu_info['devices']:
        print("\nDetected Devices:")
        for i, device in enumerate(gpu_info['devices']):
            print(f"  {i+1}. {device['name']}")
    
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  {rec}")
    
    # Save report to file
    report_file = Path(__file__).parent / "gpu_detection_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    print("\n🚀 To activate the recommended environment, run:")
    print(f"   pixi shell {recommended_env}")
    
    return report


if __name__ == "__main__":
    try:
        report = main()
        # Return environment name as exit code mapping for shell scripts
        env_codes = {'default': 0, 'nvidia': 1, 'amd': 2}
        sys.exit(env_codes.get(report['recommended_environment'], 0))
    except Exception as e:
        print(f"❌ Error during GPU detection: {e}")
        print("💻 Falling back to default CPU environment")
        sys.exit(0)