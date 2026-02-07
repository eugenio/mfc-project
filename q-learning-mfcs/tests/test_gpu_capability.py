#!/usr/bin/env python3
"""GPU Capability Assessment Tests for MFC Q-Learning Project.
Tests system GPU capabilities and available acceleration libraries.
"""

import os
import subprocess
import sys
import unittest

import pytest

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# Helper functions to detect GPU availability at module level
def _check_nvidia_available() -> bool:
    """Check if NVIDIA GPU and drivers are available."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False, capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _check_cuda_runtime_available() -> bool:
    """Check if CUDA runtime is properly available (not just installed)."""
    try:
        import torch  # type: ignore[import-not-found]  # noqa: PLC0415
        return torch.cuda.is_available()  # type: ignore[no-any-return]  # noqa: TRY300
    except ImportError:
        # Try with cupy
        try:
            import cupy as cp  # type: ignore[import-not-found]  # noqa: PLC0415
            # Try to create a simple array to verify CUDA runtime works
            _ = cp.zeros(1)
            return True  # noqa: TRY300
        except (ImportError, Exception):  # noqa: BLE001
            return False


def _check_rocm_available() -> bool:
    """Check if ROCm is available."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--version"],
            check=False, capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0  # noqa: TRY300
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _check_pytorch_available() -> bool:
    """Check if PyTorch is installed."""
    try:
        import torch  # noqa: F401, PLC0415
        return True  # noqa: TRY300
    except ImportError:
        return False


def _check_pytorch_cuda_available() -> bool:
    """Check if PyTorch CUDA is available."""
    try:
        import torch  # noqa: PLC0415
        return bool(torch.cuda.is_available())  # noqa: TRY300
    except ImportError:
        return False


def _check_pytorch_rocm_available() -> bool:
    """Check if PyTorch ROCm is available."""
    try:
        import torch  # noqa: PLC0415
        return hasattr(torch.version, "hip") and torch.version.hip is not None  # noqa: TRY300
    except ImportError:
        return False


def _check_cupy_available() -> bool:
    """Check if CuPy is installed and CUDA runtime works."""
    try:
        import cupy as cp  # noqa: PLC0415
        # Try to create a simple array to verify CUDA runtime works
        _ = cp.zeros(1)
        return True  # noqa: TRY300
    except ImportError:
        return False
    except Exception:  # noqa: BLE001
        # CUDA runtime error or driver issue
        return False


# Module-level flags for skipif decorators
NVIDIA_AVAILABLE = _check_nvidia_available()
CUDA_RUNTIME_AVAILABLE = _check_cuda_runtime_available()
ROCM_AVAILABLE = _check_rocm_available()
PYTORCH_AVAILABLE = _check_pytorch_available()
PYTORCH_CUDA_AVAILABLE = _check_pytorch_cuda_available()
PYTORCH_ROCM_AVAILABLE = _check_pytorch_rocm_available()
CUPY_AVAILABLE = _check_cupy_available()


def _is_cuda_runtime_error(exc: Exception) -> bool:
    """Check if exception is a CUDA runtime/driver error."""
    error_indicators = [
        "cudaErrorInsufficientDriver",
        "CUDA driver version is insufficient",
        "CUDA runtime version",
        "no CUDA-capable device",
        "CUDA error",
        "cudaErrorNoDevice",
        "cuda initialization",
    ]
    error_msg = str(exc).lower()
    return any(indicator.lower() in error_msg for indicator in error_indicators)


def _is_rocm_runtime_error(exc: Exception) -> bool:
    """Check if exception is a ROCm runtime/driver error."""
    error_indicators = [
        "hip error",
        "hipErrorNoBinaryForGpu",
        "hip runtime",
        "rocm",
        "no device",
    ]
    error_msg = str(exc).lower()
    return any(indicator.lower() in error_msg for indicator in error_indicators)


class TestGPUCapability(unittest.TestCase):
    """Test GPU hardware and software capabilities for both NVIDIA and AMD."""

    def setUp(self):
        """Set up test environment."""
        self.gpu_info = {}

    def test_nvidia_gpu_hardware(self):
        """Test for NVIDIA GPU hardware presence."""
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                                   "--format=csv,noheader,nounits"],
                                  check=False, capture_output=True, text=True, timeout=10)

            if result.returncode == 0 and result.stdout.strip():
                gpu_lines = result.stdout.strip().split("\n")
                self.gpu_info["nvidia_gpus"] = []

                for line in gpu_lines:
                    parts = line.split(", ")
                    if len(parts) >= 3:
                        gpu_data = {
                            "name": parts[0].strip(),
                            "memory_mb": int(parts[1].strip()),
                            "compute_capability": parts[2].strip(),
                        }
                        self.gpu_info["nvidia_gpus"].append(gpu_data)

                print(f"Found {len(self.gpu_info['nvidia_gpus'])} NVIDIA GPU(s):")
                for i, gpu in enumerate(self.gpu_info["nvidia_gpus"]):
                    print(f"  GPU {i}: {gpu['name']} ({gpu['memory_mb']} MB, CC {gpu['compute_capability']})")

                self.assertGreater(len(self.gpu_info["nvidia_gpus"]), 0)
            else:
                self.skipTest("No NVIDIA GPUs detected or nvidia-smi not available")

        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.skipTest("nvidia-smi not available or timeout")

    def test_cuda_availability(self):
        """Test CUDA availability and version."""
        try:
            result = subprocess.run(["nvcc", "--version"],
                                  check=False, capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                cuda_info = result.stdout
                # Extract CUDA version
                for line in cuda_info.split("\n"):
                    if "release" in line.lower():
                        self.gpu_info["cuda_version"] = line.strip()
                        print(f"CUDA available: {line.strip()}")
                        break

                self.assertIsNotNone(self.gpu_info.get("cuda_version"))
            else:
                self.skipTest("CUDA not available")

        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.skipTest("nvcc not available")

    def test_amd_gpu_hardware(self):
        """Test for AMD GPU hardware presence."""
        try:
            # Try rocm-smi first
            result = subprocess.run(["rocm-smi", "--showproductname", "--showmeminfo", "vram"],
                                  check=False, capture_output=True, text=True, timeout=10)

            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split("\n")
                self.gpu_info["amd_gpus"] = []

                current_gpu = {}
                for line in lines:
                    line = line.strip()
                    if "GPU[" in line and "Card series:" in line:
                        if current_gpu:
                            self.gpu_info["amd_gpus"].append(current_gpu)
                        current_gpu = {"name": line.split("Card series:")[-1].strip()}
                    elif "GPU[" in line and "VRAM Total Memory (B):" in line:
                        memory_bytes = int(line.split(":")[-1].strip())
                        current_gpu["memory_mb"] = memory_bytes // (1024 * 1024)

                if current_gpu:
                    self.gpu_info["amd_gpus"].append(current_gpu)

                if self.gpu_info["amd_gpus"]:
                    print(f"Found {len(self.gpu_info['amd_gpus'])} AMD GPU(s):")
                    for i, gpu in enumerate(self.gpu_info["amd_gpus"]):
                        print(f"  GPU {i}: {gpu.get('name', 'Unknown')} ({gpu.get('memory_mb', 'Unknown')} MB)")
                    self.assertGreater(len(self.gpu_info["amd_gpus"]), 0)
                else:
                    # Fallback: try lspci to detect AMD GPUs
                    lspci_result = subprocess.run(["lspci", "-d", "1002:", "-v"],
                                                check=False, capture_output=True, text=True, timeout=5)
                    if lspci_result.returncode == 0 and "VGA" in lspci_result.stdout:
                        amd_gpus = []
                        for line in lspci_result.stdout.split("\n"):
                            if "VGA" in line and ("AMD" in line or "ATI" in line):
                                gpu_name = line.split("VGA compatible controller:")[-1].strip()
                                amd_gpus.append({"name": gpu_name, "detection_method": "lspci"})

                        if amd_gpus:
                            self.gpu_info["amd_gpus"] = amd_gpus
                            print(f"Found {len(amd_gpus)} AMD GPU(s) via lspci:")
                            for i, gpu in enumerate(amd_gpus):
                                print(f"  GPU {i}: {gpu['name']}")
                            self.assertGreater(len(amd_gpus), 0)
                        else:
                            self.skipTest("No AMD GPUs detected")
                    else:
                        self.skipTest("No AMD GPUs detected")
            else:
                self.skipTest("rocm-smi not available or no AMD GPUs")

        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Fallback: check for AMD GPUs using lspci
            try:
                result = subprocess.run(["lspci", "-d", "1002:"],
                                      check=False, capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and result.stdout.strip():
                    amd_devices = []
                    for line in result.stdout.split("\n"):
                        if line.strip() and ("VGA" in line or "Display" in line):
                            device_info = line.split(":", 2)[-1].strip()
                            amd_devices.append({"name": device_info, "detection_method": "lspci"})

                    if amd_devices:
                        self.gpu_info["amd_gpus"] = amd_devices
                        print(f"Found {len(amd_devices)} AMD GPU device(s) via lspci:")
                        for i, device in enumerate(amd_devices):
                            print(f"  Device {i}: {device['name']}")
                        self.assertGreater(len(amd_devices), 0)
                    else:
                        self.skipTest("No AMD GPU devices detected")
                else:
                    self.skipTest("lspci not available or no AMD devices")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self.skipTest("AMD GPU detection tools not available")

    def test_rocm_availability(self):
        """Test ROCm availability and version."""
        try:
            # Check ROCm version
            result = subprocess.run(["rocm-smi", "--version"],
                                  check=False, capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                rocm_info = result.stdout
                self.gpu_info["rocm_version"] = rocm_info.strip()
                print(f"ROCm available: {rocm_info.strip()}")
                self.assertIsNotNone(self.gpu_info.get("rocm_version"))
            else:
                # Try alternative method
                try:
                    with open("/opt/rocm/.info/version") as f:
                        version = f.read().strip()
                        self.gpu_info["rocm_version"] = f"ROCm {version}"
                        print(f"ROCm version found: {version}")
                        self.assertIsNotNone(version)
                except FileNotFoundError:
                    self.skipTest("ROCm not available")

        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.skipTest("ROCm tools not available")

    def test_hip_availability(self):
        """Test HIP (Heterogeneous-Compute Interface for Portability) availability."""
        try:
            result = subprocess.run(["hipcc", "--version"],
                                  check=False, capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                hip_info = result.stdout
                # Extract HIP version
                for line in hip_info.split("\n"):
                    if "HIP version" in line or "clang version" in line:
                        self.gpu_info["hip_version"] = line.strip()
                        print(f"HIP available: {line.strip()}")
                        break

                self.assertIsNotNone(self.gpu_info.get("hip_version"))
            else:
                self.skipTest("HIP not available")

        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.skipTest("hipcc not available")

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available or CUDA runtime error")
    def test_cupy_availability(self):
        """Test CuPy library availability and functionality."""
        try:
            import cupy as cp

            # Test basic CuPy functionality
            a = cp.array([1, 2, 3, 4, 5])
            b = cp.array([2, 3, 4, 5, 6])
            c = a + b
            result = cp.asnumpy(c)

            expected = [3, 5, 7, 9, 11]
            self.assertEqual(result.tolist(), expected)

            # Get device info
            device = cp.cuda.Device()
            self.gpu_info["cupy_device"] = {
                "id": device.id,
                "name": device.name.decode() if hasattr(device.name, "decode") else str(device.name),
                "compute_capability": device.compute_capability,
                "memory_info": device.mem_info,
            }

            print(f"CuPy available - Device: {self.gpu_info['cupy_device']['name']}")
            print(f"  Compute Capability: {self.gpu_info['cupy_device']['compute_capability']}")
            print(f"  Memory: {self.gpu_info['cupy_device']['memory_info'][1] // 1024**2} MB total")

        except ImportError:
            self.skipTest("CuPy not available")
        except Exception as e:
            if _is_cuda_runtime_error(e):
                self.skipTest(f"CUDA runtime not available: {e}")
            self.fail(f"CuPy available but failed basic test: {e}")

    @pytest.mark.skipif(not CUDA_RUNTIME_AVAILABLE, reason="CUDA runtime not available")
    def test_numba_cuda_availability(self):
        """Test Numba CUDA JIT compilation availability."""
        try:
            from numba import cuda

            # Test if CUDA is available
            if not cuda.is_available():
                self.skipTest("Numba CUDA not available")

            # Test basic CUDA kernel
            @cuda.jit
            def add_kernel(a, b, c):
                idx = cuda.grid(1)
                if idx < len(c):
                    c[idx] = a[idx] + b[idx]

            import numpy as np

            # Test data
            n = 1000
            a = np.ones(n, dtype=np.float32)
            b = np.ones(n, dtype=np.float32) * 2
            c = np.zeros(n, dtype=np.float32)

            # Copy to device
            d_a = cuda.to_device(a)
            d_b = cuda.to_device(b)
            d_c = cuda.to_device(c)

            # Configure and launch kernel
            threads_per_block = 256
            blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
            add_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_c)

            # Copy result back
            result = d_c.copy_to_host()

            # Verify result
            expected = np.ones(n, dtype=np.float32) * 3
            np.testing.assert_array_almost_equal(result, expected)

            # Get device info
            device = cuda.get_current_device()
            self.gpu_info["numba_cuda"] = {
                "name": device.name.decode() if hasattr(device.name, "decode") else str(device.name),
                "compute_capability": device.compute_capability,
                "multiprocessors": device.MULTIPROCESSOR_COUNT,
                "max_threads_per_block": device.MAX_THREADS_PER_BLOCK,
            }

            print(f"Numba CUDA available - Device: {self.gpu_info['numba_cuda']['name']}")
            print(f"  Compute Capability: {self.gpu_info['numba_cuda']['compute_capability']}")
            print(f"  Multiprocessors: {self.gpu_info['numba_cuda']['multiprocessors']}")

        except ImportError:
            self.skipTest("Numba CUDA not available")
        except Exception as e:
            if _is_cuda_runtime_error(e):
                self.skipTest(f"CUDA runtime not available: {e}")
            self.fail(f"Numba CUDA available but failed test: {e}")

    @pytest.mark.skipif(not PYTORCH_CUDA_AVAILABLE, reason="PyTorch CUDA not available")
    def test_pytorch_cuda_availability(self):
        """Test PyTorch CUDA availability."""
        try:
            import torch

            if not torch.cuda.is_available():
                self.skipTest("PyTorch CUDA not available")

            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)

            # Test basic PyTorch CUDA operation
            a = torch.tensor([1.0, 2.0, 3.0]).cuda()
            b = torch.tensor([2.0, 3.0, 4.0]).cuda()
            c = a + b
            result = c.cpu().numpy()

            expected = [3.0, 5.0, 7.0]
            self.assertEqual(result.tolist(), expected)

            self.gpu_info["pytorch_cuda"] = {
                "device_count": device_count,
                "current_device": current_device,
                "device_name": device_name,
                "cuda_version": torch.version.cuda,
            }

            print(f"PyTorch CUDA available - {device_count} device(s)")
            print(f"  Current device: {device_name}")
            print(f"  CUDA version: {torch.version.cuda}")

        except ImportError:
            self.skipTest("PyTorch not available")
        except unittest.SkipTest:
            raise  # Re-raise skip test exceptions
        except Exception as e:
            if _is_cuda_runtime_error(e):
                self.skipTest(f"CUDA runtime not available: {e}")
            self.fail(f"PyTorch CUDA available but failed test: {e}")

    @pytest.mark.skipif(not PYTORCH_ROCM_AVAILABLE, reason="PyTorch ROCm not available")
    def test_pytorch_rocm_availability(self):
        """Test PyTorch ROCm availability."""
        try:
            import torch

            # Check if ROCm support is available
            if not hasattr(torch.version, "hip") or torch.version.hip is None:
                self.skipTest("PyTorch not compiled with ROCm support")

            if not torch.cuda.is_available():  # In ROCm, torch.cuda functions work for AMD GPUs
                self.skipTest("PyTorch ROCm not available")

            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)

            # Test basic PyTorch ROCm operation
            a = torch.tensor([1.0, 2.0, 3.0]).cuda()
            b = torch.tensor([2.0, 3.0, 4.0]).cuda()
            c = a + b
            result = c.cpu().numpy()

            expected = [3.0, 5.0, 7.0]
            self.assertEqual(result.tolist(), expected)

            self.gpu_info["pytorch_rocm"] = {
                "device_count": device_count,
                "current_device": current_device,
                "device_name": device_name,
                "hip_version": torch.version.hip,
                "rocm_version": getattr(torch.version, "rocm", "unknown"),
            }

            print(f"PyTorch ROCm available - {device_count} device(s)")
            print(f"  Current device: {device_name}")
            print(f"  HIP version: {torch.version.hip}")
            print(f"  ROCm version: {getattr(torch.version, 'rocm', 'unknown')}")

        except ImportError:
            self.skipTest("PyTorch not available")
        except unittest.SkipTest:
            raise  # Re-raise skip test exceptions
        except Exception as e:
            if _is_rocm_runtime_error(e):
                self.skipTest(f"ROCm runtime not available: {e}")
            self.fail(f"PyTorch ROCm available but failed test: {e}")

    @pytest.mark.skipif(not ROCM_AVAILABLE, reason="ROCm not available")
    def test_tensorflow_rocm_availability(self):
        """Test TensorFlow ROCm availability."""
        try:
            import tensorflow as tf

            # Suppress TensorFlow warnings
            tf.get_logger().setLevel("ERROR")

            gpu_devices = tf.config.list_physical_devices("GPU")

            if not gpu_devices:
                self.skipTest("TensorFlow ROCm not available")

            # Check if it's ROCm build
            build_info = tf.sysconfig.get_build_info()
            is_rocm_build = any("rocm" in str(v).lower() or "hip" in str(v).lower()
                              for v in build_info.values())

            # Test basic TensorFlow ROCm operation
            with tf.device("/GPU:0"):
                a = tf.constant([1.0, 2.0, 3.0])
                b = tf.constant([2.0, 3.0, 4.0])
                c = tf.add(a, b)
                result = c.numpy()

            expected = [3.0, 5.0, 7.0]
            self.assertEqual(result.tolist(), expected)

            self.gpu_info["tensorflow_rocm"] = {
                "gpu_count": len(gpu_devices),
                "devices": [device.name for device in gpu_devices],
                "version": tf.__version__,
                "is_rocm_build": is_rocm_build,
                "build_info": build_info,
            }

            print(f"TensorFlow {'ROCm' if is_rocm_build else 'GPU'} available - {len(gpu_devices)} device(s)")
            print(f"  Version: {tf.__version__}")
            print(f"  ROCm build: {is_rocm_build}")
            for device in gpu_devices:
                print(f"  Device: {device.name}")

        except ImportError:
            self.skipTest("TensorFlow not available")
        except Exception as e:
            if _is_rocm_runtime_error(e):
                self.skipTest(f"ROCm runtime not available: {e}")
            self.fail(f"TensorFlow ROCm available but failed test: {e}")

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available or GPU runtime error")
    def test_cupy_rocm_availability(self):
        """Test CuPy with ROCm backend availability."""
        try:
            # CuPy with ROCm support (if available)
            import cupy as cp

            # Check if this is ROCm build
            try:
                runtime_info = cp.cuda.runtime.getVersion()
                if runtime_info == 0:  # ROCm typically returns 0
                    backend = "ROCm"
                else:
                    backend = "CUDA"
            except Exception:
                backend = "Unknown"

            # Test basic CuPy functionality
            a = cp.array([1, 2, 3, 4, 5])
            b = cp.array([2, 3, 4, 5, 6])
            c = a + b
            result = cp.asnumpy(c)

            expected = [3, 5, 7, 9, 11]
            self.assertEqual(result.tolist(), expected)

            # Get device info
            device = cp.cuda.Device()
            self.gpu_info["cupy_rocm"] = {
                "id": device.id,
                "backend": backend,
                "compute_capability": getattr(device, "compute_capability", "N/A"),
                "memory_info": device.mem_info,
            }

            print(f"CuPy with {backend} backend available")
            print(f"  Device ID: {device.id}")
            print(f"  Memory: {device.mem_info[1] // 1024**2} MB total")

        except ImportError:
            self.skipTest("CuPy not available")
        except Exception as e:
            if _is_cuda_runtime_error(e) or _is_rocm_runtime_error(e):
                self.skipTest(f"GPU runtime not available: {e}")
            self.fail(f"CuPy available but failed basic test: {e}")

    def test_jax_gpu_availability(self):
        """Test JAX GPU availability (supports both CUDA and ROCm)."""
        try:
            import jax
            import jax.numpy as jnp

            # Check available devices
            devices = jax.devices()
            # Check for GPU devices by platform (works for both CUDA and ROCm)
            gpu_devices = [d for d in devices if d.platform == "gpu"]

            if not gpu_devices:
                self.skipTest("JAX GPU not available")

            # Test basic JAX GPU operation
            a = jnp.array([1.0, 2.0, 3.0])
            b = jnp.array([2.0, 3.0, 4.0])
            c = a + b
            result = c.tolist()

            expected = [3.0, 5.0, 7.0]
            self.assertEqual(result, expected)

            self.gpu_info["jax_gpu"] = {
                "total_devices": len(devices),
                "gpu_devices": len(gpu_devices),
                "device_info": [{"platform": d.platform, "id": d.id} for d in gpu_devices],
                "version": jax.__version__,
            }

            print(f"JAX GPU available - {len(gpu_devices)} GPU device(s)")
            print(f"  Version: {jax.__version__}")
            for device in gpu_devices:
                print(f"  Device: {device.platform} {device.id}")

        except ImportError:
            self.skipTest("JAX not available")
        except unittest.SkipTest:
            raise  # Re-raise skip test exceptions
        except Exception as e:
            if _is_cuda_runtime_error(e) or _is_rocm_runtime_error(e):
                self.skipTest(f"GPU runtime not available: {e}")
            self.fail(f"JAX GPU available but failed test: {e}")

    def test_tensorflow_gpu_availability(self):
        """Test TensorFlow GPU availability."""
        try:
            import tensorflow as tf

            # Suppress TensorFlow warnings
            tf.get_logger().setLevel("ERROR")

            gpu_devices = tf.config.list_physical_devices("GPU")

            if not gpu_devices:
                self.skipTest("TensorFlow GPU not available")

            # Test basic TensorFlow GPU operation
            with tf.device("/GPU:0"):
                a = tf.constant([1.0, 2.0, 3.0])
                b = tf.constant([2.0, 3.0, 4.0])
                c = tf.add(a, b)
                result = c.numpy()

            expected = [3.0, 5.0, 7.0]
            self.assertEqual(result.tolist(), expected)

            self.gpu_info["tensorflow_gpu"] = {
                "gpu_count": len(gpu_devices),
                "devices": [device.name for device in gpu_devices],
                "version": tf.__version__,
            }

            print(f"TensorFlow GPU available - {len(gpu_devices)} device(s)")
            print(f"  Version: {tf.__version__}")
            for device in gpu_devices:
                print(f"  Device: {device.name}")

        except ImportError:
            self.skipTest("TensorFlow not available")
        except Exception as e:
            if _is_cuda_runtime_error(e) or _is_rocm_runtime_error(e):
                self.skipTest(f"GPU runtime not available: {e}")
            self.fail(f"TensorFlow GPU available but failed test: {e}")


class TestGPUPerformance(unittest.TestCase):
    """Test GPU performance benchmarks."""

    def setUp(self):
        """Set up performance test environment."""
        self.performance_results = {}

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available or CUDA runtime error")
    def test_cupy_vs_numpy_performance(self):
        """Compare CuPy vs NumPy performance for matrix operations."""
        try:
            import time

            import cupy as cp
            import numpy as np

            # Test parameters
            size = 2000
            iterations = 5

            # NumPy test
            np_times = []
            for _ in range(iterations):
                a_np = np.random.rand(size, size).astype(np.float32)
                b_np = np.random.rand(size, size).astype(np.float32)

                start_time = time.time()
                _ = np.dot(a_np, b_np)  # Result not used, just timing
                np_time = time.time() - start_time
                np_times.append(np_time)

            avg_np_time = sum(np_times) / len(np_times)

            # CuPy test
            cp_times = []
            for _ in range(iterations):
                a_cp = cp.random.rand(size, size, dtype=cp.float32)
                b_cp = cp.random.rand(size, size, dtype=cp.float32)

                start_time = time.time()
                _ = cp.dot(a_cp, b_cp)  # Result not used, just timing
                cp.cuda.Stream.null.synchronize()  # Ensure GPU computation is complete
                cp_time = time.time() - start_time
                cp_times.append(cp_time)

            avg_cp_time = sum(cp_times) / len(cp_times)
            speedup = avg_np_time / avg_cp_time

            self.performance_results["matrix_multiplication"] = {
                "numpy_time": avg_np_time,
                "cupy_time": avg_cp_time,
                "speedup": speedup,
                "matrix_size": size,
            }

            print(f"Matrix Multiplication Performance ({size}x{size}):")
            print(f"  NumPy average time: {avg_np_time:.4f}s")
            print(f"  CuPy average time: {avg_cp_time:.4f}s")
            print(f"  Speedup: {speedup:.2f}x")

            # Verify correctness
            a_test = np.random.rand(100, 100).astype(np.float32)
            b_test = np.random.rand(100, 100).astype(np.float32)
            c_np_test = np.dot(a_test, b_test)
            c_cp_test = cp.asnumpy(cp.dot(cp.asarray(a_test), cp.asarray(b_test)))

            np.testing.assert_allclose(c_np_test, c_cp_test, rtol=1e-5)

        except ImportError:
            self.skipTest("CuPy not available for performance test")
        except Exception as e:
            if _is_cuda_runtime_error(e):
                self.skipTest(f"CUDA runtime not available: {e}")
            self.fail(f"Performance test failed: {e}")

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available or CUDA runtime error")
    def test_gpu_memory_bandwidth(self):
        """Test GPU memory bandwidth."""
        try:
            import time

            import cupy as cp

            # Test parameters
            size = 100_000_000  # 100M elements
            iterations = 10

            # Memory bandwidth test
            times = []
            for _ in range(iterations):
                # Create large arrays
                a = cp.random.rand(size, dtype=cp.float32)
                b = cp.random.rand(size, dtype=cp.float32)

                start_time = time.time()
                _ = a + b  # Simple element-wise operation, result not used
                cp.cuda.Stream.null.synchronize()
                elapsed_time = time.time() - start_time
                times.append(elapsed_time)

            avg_time = sum(times) / len(times)
            # Calculate bandwidth (reading 2 arrays + writing 1 array)
            bytes_transferred = size * 4 * 3  # 3 arrays * 4 bytes per float32
            bandwidth_gb_s = (bytes_transferred / avg_time) / (1024**3)

            self.performance_results["memory_bandwidth"] = {
                "bandwidth_gb_s": bandwidth_gb_s,
                "avg_time": avg_time,
                "array_size": size,
            }

            print("GPU Memory Bandwidth Test:")
            print(f"  Array size: {size:,} elements")
            print(f"  Average time: {avg_time:.4f}s")
            print(f"  Bandwidth: {bandwidth_gb_s:.2f} GB/s")

        except ImportError:
            self.skipTest("CuPy not available for bandwidth test")
        except Exception as e:
            if _is_cuda_runtime_error(e):
                self.skipTest(f"CUDA runtime not available: {e}")
            self.fail(f"Memory bandwidth test failed: {e}")

    @pytest.mark.skipif(
        not (PYTORCH_CUDA_AVAILABLE or PYTORCH_ROCM_AVAILABLE),
        reason="PyTorch GPU not available (no CUDA or ROCm)",
    )
    def test_pytorch_gpu_vs_cpu_performance(self):
        """Compare PyTorch GPU vs CPU performance (works for both CUDA and ROCm)."""
        try:
            import time

            import torch

            if not torch.cuda.is_available():
                self.skipTest("PyTorch GPU not available for performance test")

            # Test parameters
            size = 2000
            iterations = 5

            # CPU test
            cpu_times = []
            for _ in range(iterations):
                a_cpu = torch.randn(size, size, dtype=torch.float32)
                b_cpu = torch.randn(size, size, dtype=torch.float32)

                start_time = time.time()
                _ = torch.mm(a_cpu, b_cpu)  # Result not used, just timing
                cpu_time = time.time() - start_time
                cpu_times.append(cpu_time)

            avg_cpu_time = sum(cpu_times) / len(cpu_times)

            # GPU test
            gpu_times = []
            device = torch.cuda.current_device()

            for _ in range(iterations):
                a_gpu = torch.randn(size, size, dtype=torch.float32, device=device)
                b_gpu = torch.randn(size, size, dtype=torch.float32, device=device)

                start_time = time.time()
                _ = torch.mm(a_gpu, b_gpu)  # Result not used, just timing
                torch.cuda.synchronize()  # Ensure GPU computation is complete
                gpu_time = time.time() - start_time
                gpu_times.append(gpu_time)

            avg_gpu_time = sum(gpu_times) / len(gpu_times)
            speedup = avg_cpu_time / avg_gpu_time

            # Determine backend type
            backend_type = "CUDA"
            if hasattr(torch.version, "hip") and torch.version.hip is not None:
                backend_type = "ROCm"

            self.performance_results["pytorch_matrix_multiplication"] = {
                "cpu_time": avg_cpu_time,
                "gpu_time": avg_gpu_time,
                "speedup": speedup,
                "matrix_size": size,
                "backend": backend_type,
                "device_name": torch.cuda.get_device_name(device),
            }

            print(f"PyTorch {backend_type} Matrix Multiplication Performance ({size}x{size}):")
            print(f"  CPU average time: {avg_cpu_time:.4f}s")
            print(f"  GPU average time: {avg_gpu_time:.4f}s")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Device: {torch.cuda.get_device_name(device)}")

            # Verify correctness
            a_test = torch.randn(100, 100, dtype=torch.float32)
            b_test = torch.randn(100, 100, dtype=torch.float32)
            c_cpu_test = torch.mm(a_test, b_test)
            c_gpu_test = torch.mm(a_test.cuda(), b_test.cuda()).cpu()

            torch.testing.assert_close(c_cpu_test, c_gpu_test, rtol=1e-5, atol=1e-5)

        except ImportError:
            self.skipTest("PyTorch not available for performance test")
        except unittest.SkipTest:
            raise  # Re-raise skip test exceptions
        except Exception as e:
            if _is_cuda_runtime_error(e) or _is_rocm_runtime_error(e):
                self.skipTest(f"GPU runtime not available: {e}")
            self.fail(f"PyTorch performance test failed: {e}")


class TestSystemRequirements(unittest.TestCase):
    """Test system requirements for GPU-accelerated simulations."""

    def test_python_version(self):
        """Test Python version compatibility."""
        version = sys.version_info
        self.assertGreaterEqual(version.major, 3)
        self.assertGreaterEqual(version.minor, 8)
        print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    def test_required_packages(self):
        """Test availability of required packages."""
        required_packages = {
            "numpy": "numpy",
            "pandas": "pandas",
            "matplotlib": "matplotlib",
            "scipy": "scipy",
        }

        missing_packages = []
        available_packages = {}

        for name, import_name in required_packages.items():
            try:
                module = __import__(import_name)
                version = getattr(module, "__version__", "unknown")
                available_packages[name] = version
                print(f"{name}: {version}")
            except ImportError:
                missing_packages.append(name)

        if missing_packages:
            self.fail(f"Missing required packages: {', '.join(missing_packages)}")

    def test_optional_gpu_packages(self):
        """Test availability of optional GPU packages (CUDA and ROCm support)."""
        optional_packages = {
            "cupy": "cupy",
            "numba": "numba",
            "torch": "torch",
            "tensorflow": "tensorflow",
            "jax": "jax",
        }

        available_gpu_packages = {}

        for name, import_name in optional_packages.items():
            try:
                module = __import__(import_name)
                version = getattr(module, "__version__", "unknown")
                available_gpu_packages[name] = version
                print(f"GPU package {name}: {version}")
            except ImportError:
                print(f"GPU package {name}: not available")

        # At least one GPU package should be available for acceleration
        if not available_gpu_packages:
            self.skipTest("No GPU acceleration packages available")


class TestEnvironmentAssessment(unittest.TestCase):
    """Test environment and provide GPU-specific package recommendations."""

    def setUp(self):
        """Set up environment assessment."""
        self.gpu_detected = {"nvidia": False, "amd": False}
        self.available_packages = {}
        self.recommendations = {"nvidia": [], "amd": [], "general": []}

    def test_detect_gpu_vendors(self):
        """Detect which GPU vendors are present in the system."""
        # Check for NVIDIA GPUs
        try:
            result = subprocess.run(["nvidia-smi", "--list-gpus"],
                                  check=False, capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                self.gpu_detected["nvidia"] = True
                print("✅ NVIDIA GPU(s) detected")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Check for AMD GPUs
        try:
            result = subprocess.run(["rocm-smi", "--showproductname"],
                                  check=False, capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                self.gpu_detected["amd"] = True
                print("✅ AMD GPU(s) detected")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Fallback: check lspci for AMD devices
            try:
                result = subprocess.run(["lspci", "-d", "1002:"],
                                      check=False, capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and ("VGA" in result.stdout or "Display" in result.stdout):
                    self.gpu_detected["amd"] = True
                    print("✅ AMD GPU(s) detected (via lspci)")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

        if not self.gpu_detected["nvidia"] and not self.gpu_detected["amd"]:
            print("❌ No discrete GPUs detected")
            self.skipTest("No GPU hardware detected")

    def test_assess_cuda_packages(self):
        """Assess CUDA-related packages and their installation status."""
        cuda_packages = {
            "cupy": {
                "import_name": "cupy",
                "install_cmd": "pip install cupy-cuda12x",  # Most common CUDA version
                "description": "NumPy-compatible library for GPU arrays",
            },
            "numba": {
                "import_name": "numba",
                "install_cmd": "pip install numba",
                "description": "JIT compiler with CUDA support",
            },
            "pytorch_cuda": {
                "import_name": "torch",
                "install_cmd": "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
                "description": "PyTorch with CUDA support",
                "test_func": lambda: __import__("torch").cuda.is_available(),
            },
            "tensorflow_gpu": {
                "import_name": "tensorflow",
                "install_cmd": "pip install tensorflow[and-cuda]",
                "description": "TensorFlow with GPU support",
                "test_func": lambda: len(__import__("tensorflow").config.list_physical_devices("GPU")) > 0,
            },
            "jax_cuda": {
                "import_name": "jax",
                "install_cmd": 'pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html',
                "description": "JAX with CUDA support",
                "test_func": lambda: len([d for d in __import__("jax").devices() if d.device_kind == "gpu"]) > 0,
            },
        }

        if not self.gpu_detected["nvidia"]:
            self.skipTest("No NVIDIA GPU detected - skipping CUDA package assessment")

        print("\n🔍 Assessing CUDA packages...")
        for name, info in cuda_packages.items():
            try:
                module = __import__(info["import_name"])

                # Test GPU functionality if test function provided
                gpu_available = True
                if "test_func" in info:
                    try:
                        gpu_available = info["test_func"]()
                    except Exception:
                        gpu_available = False

                if gpu_available:
                    version = getattr(module, "__version__", "unknown")
                    self.available_packages[name] = version
                    print(f"  ✅ {name}: {version} (GPU support confirmed)")
                else:
                    print(f"  ⚠️  {name}: installed but no GPU support detected")
                    self.recommendations["nvidia"].append(f"Reinstall {name} with GPU support: {info['install_cmd']}")

            except ImportError:
                print(f"  ❌ {name}: not installed")
                self.recommendations["nvidia"].append(f"Install {info['description']}: {info['install_cmd']}")

    def test_assess_rocm_packages(self):
        """Assess ROCm-related packages and their installation status."""
        rocm_packages = {
            "pytorch_rocm": {
                "import_name": "torch",
                "install_cmd": "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6",
                "description": "PyTorch with ROCm support",
                "test_func": lambda: hasattr(__import__("torch").version, "hip") and __import__("torch").version.hip is not None,
            },
            "tensorflow_rocm": {
                "import_name": "tensorflow",
                "install_cmd": "pip install tensorflow-rocm",
                "description": "TensorFlow with ROCm support",
                "test_func": lambda: any("rocm" in str(v).lower() for v in __import__("tensorflow").sysconfig.get_build_info().values()),
            },
            "jax_rocm": {
                "import_name": "jax",
                "install_cmd": 'pip install -U "jax[rocm]" -f https://storage.googleapis.com/jax-releases/jax_rocm_releases.html',
                "description": "JAX with ROCm support",
                "test_func": lambda: len([d for d in __import__("jax").devices() if d.device_kind == "gpu"]) > 0,
            },
        }

        if not self.gpu_detected["amd"]:
            self.skipTest("No AMD GPU detected - skipping ROCm package assessment")

        print("\n🔍 Assessing ROCm packages...")
        for name, info in rocm_packages.items():
            try:
                module = __import__(info["import_name"])

                # Test ROCm functionality if test function provided
                rocm_available = True
                if "test_func" in info:
                    try:
                        rocm_available = info["test_func"]()
                    except Exception:
                        rocm_available = False

                if rocm_available:
                    version = getattr(module, "__version__", "unknown")
                    self.available_packages[name] = version
                    print(f"  ✅ {name}: {version} (ROCm support confirmed)")
                else:
                    print(f"  ⚠️  {name}: installed but no ROCm support detected")
                    self.recommendations["amd"].append(f"Install {info['description']}: {info['install_cmd']}")

            except ImportError:
                print(f"  ❌ {name}: not installed")
                self.recommendations["amd"].append(f"Install {info['description']}: {info['install_cmd']}")

    def test_assess_general_packages(self):
        """Assess general acceleration packages."""
        general_packages = {
            "numpy": {
                "import_name": "numpy",
                "install_cmd": "pip install numpy",
                "description": "Fundamental package for scientific computing",
            },
            "scipy": {
                "import_name": "scipy",
                "install_cmd": "pip install scipy",
                "description": "Scientific computing library",
            },
            "pandas": {
                "import_name": "pandas",
                "install_cmd": "pip install pandas",
                "description": "Data manipulation and analysis",
            },
            "matplotlib": {
                "import_name": "matplotlib",
                "install_cmd": "pip install matplotlib",
                "description": "Plotting library",
            },
            "scikit_learn": {
                "import_name": "sklearn",
                "install_cmd": "pip install scikit-learn",
                "description": "Machine learning library",
            },
        }

        print("\n🔍 Assessing general scientific packages...")
        for name, info in general_packages.items():
            try:
                module = __import__(info["import_name"])
                version = getattr(module, "__version__", "unknown")
                self.available_packages[name] = version
                print(f"  ✅ {name}: {version}")
            except ImportError:
                print(f"  ❌ {name}: not installed")
                self.recommendations["general"].append(f"Install {info['description']}: {info['install_cmd']}")

    def test_print_recommendations(self):
        """Print installation recommendations based on detected hardware."""
        print("\n" + "="*60)
        print("📋 INSTALLATION RECOMMENDATIONS")
        print("="*60)

        if self.gpu_detected["nvidia"] and self.recommendations["nvidia"]:
            print("\n🟢 For NVIDIA GPU acceleration:")
            print("  Detected NVIDIA GPU(s) - install CUDA packages for optimal performance")
            for rec in self.recommendations["nvidia"]:
                print(f"  • {rec}")

            print("\n  💡 Additional NVIDIA recommendations:")
            print("  • Ensure CUDA Toolkit is installed: https://developer.nvidia.com/cuda-downloads")
            print("  • Verify cuDNN installation for deep learning: https://developer.nvidia.com/cudnn")
            print("  • Check NVIDIA driver version: nvidia-smi")

        if self.gpu_detected["amd"] and self.recommendations["amd"]:
            print("\n🔴 For AMD GPU acceleration:")
            print("  Detected AMD GPU(s) - install ROCm packages for optimal performance")
            for rec in self.recommendations["amd"]:
                print(f"  • {rec}")

            print("\n  💡 Additional AMD recommendations:")
            print("  • Install ROCm platform: https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html")
            print("  • Verify ROCm installation: rocm-smi")
            print("  • Check supported GPU list: https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html#supported-gpus")

        if self.recommendations["general"]:
            print("\n🔧 General scientific computing packages:")
            for rec in self.recommendations["general"]:
                print(f"  • {rec}")

        if not self.gpu_detected["nvidia"] and not self.gpu_detected["amd"]:
            print("\n💻 CPU-only recommendations:")
            print("  No discrete GPU detected - using CPU-optimized packages")
            print("  • pip install numpy scipy pandas matplotlib scikit-learn")
            print("  • pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
            print("  • pip install tensorflow")
            print("  • Consider Intel optimization: pip install intel-extension-for-pytorch")

        print("\n🚀 Performance optimization tips:")
        print("  • Use conda/mamba for better dependency resolution")
        print("  • Consider containerized environments (Docker/Singularity)")
        print("  • Monitor GPU utilization during training/inference")
        print("  • Use mixed precision training when available")


def create_gpu_report():
    """Create a comprehensive GPU capability report with recommendations."""
    print("\n" + "="*60)
    print("GPU CAPABILITY ASSESSMENT REPORT")
    print("="*60)

    # Run tests and collect results
    suite = unittest.TestSuite()
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestGPUCapability))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestGPUPerformance))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestSystemRequirements))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestEnvironmentAssessment))

    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)

    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.failures:
        print("\n❌ Failures:")
        for test, error in result.failures:
            print(f"  {test}: {error.split('AssertionError:')[-1].strip()}")

    if result.errors:
        print("\n⚠️  Errors:")
        for test, error in result.errors:
            print(f"  {test}: {error.split('Exception:')[-1].strip()}")

    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    success = create_gpu_report()
    sys.exit(0 if success else 1)
