#!/usr/bin/env python3
"""GPU Detection Script for Pixi Environment Configuration.

Detects the GPU hardware present in the system and determines which
ML framework dependencies should be installed (ROCm for AMD, CUDA for NVIDIA, CPU-only for Intel/others).
"""

import json
import subprocess
import sys
from pathlib import Path


def detect_gpu_type():
    """Detect the primary GPU type in the system.

    Returns:
        str: 'amd', 'nvidia', 'intel', or 'cpu' based on detected hardware

    """
    gpu_info = {"type": "cpu", "devices": [], "details": {}}  # Default fallback

    # Check for NVIDIA GPUs first
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            devices = []
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    name, memory = line.split(",", 1)
                    devices.append(
                        {"name": name.strip(), "memory_mb": int(memory.strip())},
                    )
            if devices:
                gpu_info.update(
                    {
                        "type": "nvidia",
                        "devices": devices,
                        "details": {"driver_available": True},
                    },
                )
                return gpu_info
    except (
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        FileNotFoundError,
        ValueError,
    ):
        pass

    # Check for AMD ROCm support
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and "GPU" in result.stdout:
            # Parse ROCm output
            devices = []
            for line in result.stdout.split("\n"):
                if "GPU" in line and ":" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        devices.append(
                            {"name": parts[1].strip(), "rocm_available": True},
                        )
            if devices:
                gpu_info.update(
                    {
                        "type": "amd",
                        "devices": devices,
                        "details": {"rocm_available": True},
                    },
                )
                return gpu_info
    except (
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        pass

    # Check for AMD GPUs using lspci (fallback)
    try:
        result = subprocess.run(
            ["lspci"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            amd_lines = [
                line
                for line in result.stdout.split("\n")
                if "VGA" in line
                and ("AMD" in line or "ATI" in line or "Radeon" in line)
            ]
            if amd_lines:
                devices = []
                for line in amd_lines:
                    devices.append(
                        {
                            "name": (
                                line.split(":")[-1].strip()
                                if ":" in line
                                else line.strip()
                            ),
                            "rocm_available": False,  # Unknown without rocm-smi
                        },
                    )
                gpu_info.update(
                    {
                        "type": "amd",
                        "devices": devices,
                        "details": {"rocm_available": False},
                    },
                )
                return gpu_info
    except (
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        pass

    # Check for Intel GPUs
    try:
        result = subprocess.run(
            ["lspci"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            intel_lines = [
                line
                for line in result.stdout.split("\n")
                if "VGA" in line and ("Intel" in line)
            ]
            if intel_lines:
                devices = []
                for line in intel_lines:
                    devices.append(
                        {
                            "name": (
                                line.split(":")[-1].strip()
                                if ":" in line
                                else line.strip()
                            ),
                            "integrated": True,
                        },
                    )
                gpu_info.update(
                    {
                        "type": "intel",
                        "devices": devices,
                        "details": {"integrated": True},
                    },
                )
                return gpu_info
    except (
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        pass

    # Fallback to CPU-only
    return gpu_info


def get_recommended_environment() -> str:
    """Get the recommended pixi environment based on detected GPU.

    Returns:
        str: Environment name ('amd', 'nvidia', 'default')

    """
    gpu_info = detect_gpu_type()
    gpu_type = gpu_info["type"]

    if gpu_type == "nvidia":
        return "nvidia"
    if gpu_type == "amd":
        return "amd"
    return "default"


def check_gpu_drivers():
    """Check if appropriate GPU drivers are installed.

    Returns:
        dict: Driver status information

    """
    driver_status = {"nvidia": False, "amd_rocm": False, "amd_opencl": False}

    # Check NVIDIA drivers
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            check=False,
            capture_output=True,
            timeout=5,
        )
        driver_status["nvidia"] = result.returncode == 0
    except (
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        pass

    # Check ROCm
    try:
        result = subprocess.run(
            ["rocm-smi"],
            check=False,
            capture_output=True,
            timeout=5,
        )
        driver_status["amd_rocm"] = result.returncode == 0
    except (
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        pass

    # Check OpenCL
    try:
        result = subprocess.run(["clinfo"], check=False, capture_output=True, timeout=5)
        driver_status["amd_opencl"] = result.returncode == 0 and b"AMD" in result.stdout
    except (
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        pass

    return driver_status


def main():
    """Main function to detect GPU and recommend configuration."""
    # Detect GPU
    gpu_info = detect_gpu_type()
    recommended_env = get_recommended_environment()

    # Check drivers
    driver_status = check_gpu_drivers()

    # Create comprehensive report
    report = {
        "gpu_info": gpu_info,
        "recommended_environment": recommended_env,
        "driver_status": driver_status,
        "recommendations": [],
    }

    # Generate recommendations
    if gpu_info["type"] == "nvidia":
        if driver_status["nvidia"]:
            report["recommendations"].append(
                "✅ Use 'pixi shell nvidia' for CUDA acceleration",
            )
            report["recommendations"].append("✅ NVIDIA drivers detected and working")
        else:
            report["recommendations"].append(
                "⚠️ Install NVIDIA drivers for GPU acceleration",
            )
            report["recommendations"].append(
                "💡 Fallback to 'pixi shell default' for CPU-only",
            )

    elif gpu_info["type"] == "amd":
        if driver_status["amd_rocm"]:
            report["recommendations"].append(
                "✅ Use 'pixi shell amd' for ROCm acceleration",
            )
            report["recommendations"].append("✅ ROCm drivers detected and working")
        else:
            report["recommendations"].append(
                "⚠️ Install ROCm drivers for GPU acceleration",
            )
            report["recommendations"].append(
                "💡 Instructions: https://rocm.docs.amd.com/projects/install-on-linux/",
            )
            report["recommendations"].append(
                "💡 Fallback to 'pixi shell default' for CPU-only",
            )

    else:
        report["recommendations"].append(
            "💻 Use 'pixi shell default' for CPU-only ML workloads",
        )
        if gpu_info["type"] == "intel":
            report["recommendations"].append(
                "💡 Consider Intel Extension for PyTorch for integrated GPU acceleration",
            )

    # Print summary

    if gpu_info["devices"]:
        for _i, _device in enumerate(gpu_info["devices"]):
            pass

    for _rec in report["recommendations"]:
        pass

    # Save report to file
    report_file = Path(__file__).parent / "gpu_detection_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    try:
        report = main()
        # Return environment name as exit code mapping for shell scripts
        env_codes = {"default": 0, "nvidia": 1, "amd": 2}
        sys.exit(env_codes.get(report["recommended_environment"], 0))
    except Exception:
        sys.exit(0)
