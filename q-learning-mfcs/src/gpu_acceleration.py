#!/usr/bin/env python3
"""
Universal GPU Acceleration Module for MFC Simulations.
Provides abstraction layer for both NVIDIA CUDA and AMD ROCm support.
"""

import sys
from typing import Any, Dict, Optional

import numpy as np


class GPUAccelerator:
    """
    Universal GPU acceleration interface supporting both NVIDIA CUDA and AMD ROCm.
    Automatically detects available GPU backends and provides unified interface.
    """

    def __init__(self, prefer_backend: str = 'auto'):
        """
        Initialize GPU accelerator with backend preference.
        
        Args:
            prefer_backend: 'auto', 'cuda', 'rocm', or 'cpu'
        """
        self.backend = None
        self.device_info = {}
        self.available_backends = []
        self._detect_backends()
        self._initialize_backend(prefer_backend)

    def _detect_backends(self):
        """Detect available GPU backends."""
        # Test for NVIDIA CUDA support
        try:
            import cupy as cp
            # Test basic operation
            test_array = cp.array([1, 2, 3])
            cp.asnumpy(test_array)  # Test GPU->CPU transfer
            self.available_backends.append('cuda')
            print("✅ NVIDIA CUDA backend detected (CuPy)")
        except ImportError:
            print("❌ CuPy not available - NVIDIA CUDA backend unavailable")
        except Exception as e:
            print(f"⚠️  CuPy available but CUDA not functional: {e}")

        # Test for PyTorch (supports both CUDA and ROCm)
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0)

                # Check if it's ROCm or CUDA build
                is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None

                if is_rocm:
                    self.available_backends.append('rocm')
                    print(f"✅ AMD ROCm backend detected (PyTorch) - {device_count} device(s)")
                    print(f"   Device: {device_name}")
                else:
                    if 'cuda' not in self.available_backends:
                        self.available_backends.append('cuda')
                    print(f"✅ NVIDIA CUDA backend detected (PyTorch) - {device_count} device(s)")
                    print(f"   Device: {device_name}")
        except ImportError:
            print("❌ PyTorch not available")
        except Exception as e:
            print(f"⚠️  PyTorch available but GPU not functional: {e}")

        # Test for JAX (supports both CUDA and ROCm)
        try:
            import jax
            devices = jax.devices()
            gpu_devices = [d for d in devices if d.device_kind == 'gpu']

            if gpu_devices:
                platform = gpu_devices[0].platform
                if platform == 'rocm':
                    if 'rocm' not in self.available_backends:
                        self.available_backends.append('rocm')
                    print(f"✅ AMD ROCm backend detected (JAX) - {len(gpu_devices)} device(s)")
                elif platform == 'gpu':  # JAX CUDA
                    if 'cuda' not in self.available_backends:
                        self.available_backends.append('cuda')
                    print(f"✅ NVIDIA CUDA backend detected (JAX) - {len(gpu_devices)} device(s)")
        except ImportError:
            print("❌ JAX not available")
        except Exception as e:
            print(f"⚠️  JAX available but GPU not functional: {e}")

        # Always add CPU as fallback option
        if 'cpu' not in self.available_backends:
            self.available_backends.append('cpu')

        if len(self.available_backends) == 1 and self.available_backends[0] == 'cpu':
            print("💻 No GPU backends detected - will use CPU-only mode")

    def _initialize_backend(self, prefer_backend: str):
        """Initialize the preferred backend."""
        if prefer_backend == 'auto':
            # Priority: CUDA > ROCm > CPU
            if 'cuda' in self.available_backends:
                self.backend = 'cuda'
            elif 'rocm' in self.available_backends:
                self.backend = 'rocm'
            else:
                self.backend = 'cpu'
        elif prefer_backend in self.available_backends:
            self.backend = prefer_backend
        else:
            print(f"⚠️  Preferred backend '{prefer_backend}' not available")
            print(f"   Available backends: {self.available_backends}")
            self.backend = self.available_backends[0] if self.available_backends else 'cpu'

        print(f"🚀 Using backend: {self.backend.upper()}")
        self._setup_backend()

    def _setup_backend(self):
        """Setup the selected backend."""
        if self.backend == 'cuda':
            self._setup_cuda()
        elif self.backend == 'rocm':
            self._setup_rocm()
        else:
            self._setup_cpu()

    def _setup_cuda(self):
        """Setup NVIDIA CUDA backend."""
        try:
            import cupy as cp
            self.cp = cp
            self.torch = None

            # Get device info
            device = cp.cuda.Device()
            self.device_info = {
                'backend': 'cuda',
                'device_count': 1,
                'device_name': device.name.decode() if hasattr(device.name, 'decode') else str(device.name),
                'memory_total': device.mem_info[1],
                'memory_free': device.mem_info[0],
                'compute_capability': device.compute_capability
            }

            print(f"   Device: {self.device_info['device_name']}")
            print(f"   Memory: {self.device_info['memory_total'] // 1024**2} MB")
            print(f"   Compute Capability: {self.device_info['compute_capability']}")

        except Exception as e:
            print(f"❌ Failed to setup CUDA backend: {e}")
            self._setup_cpu()

    def _setup_rocm(self):
        """Setup AMD ROCm backend."""
        try:
            import torch
            self.torch = torch
            self.cp = None

            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0)

                self.device_info = {
                    'backend': 'rocm',
                    'device_count': device_count,
                    'device_name': device_name,
                    'hip_version': getattr(torch.version, 'hip', 'unknown'),
                    'rocm_version': getattr(torch.version, 'rocm', 'unknown')
                }

                print(f"   Device: {device_name}")
                print(f"   HIP Version: {self.device_info['hip_version']}")
                print(f"   ROCm Version: {self.device_info['rocm_version']}")

        except Exception as e:
            print(f"❌ Failed to setup ROCm backend: {e}")
            self._setup_cpu()

    def _setup_cpu(self):
        """Setup CPU-only backend."""
        self.cp = None
        self.torch = None
        self.backend = 'cpu'
        self.device_info = {
            'backend': 'cpu',
            'device_count': 0,
            'device_name': 'CPU'
        }
        print("   Using CPU-only computation")

    def array(self, data, dtype=np.float32):
        """Create array on appropriate device."""
        if self.backend == 'cuda' and self.cp:
            return self.cp.asarray(data, dtype=dtype)
        elif self.backend == 'rocm' and self.torch:
            if isinstance(data, np.ndarray):
                tensor = self.torch.from_numpy(data.copy().astype(dtype))
            else:
                tensor = self.torch.tensor(data, dtype=self._np_to_torch_dtype(dtype))
            return tensor.cuda()
        else:
            return np.asarray(data, dtype=dtype)

    def zeros(self, shape, dtype=np.float32):
        """Create zeros array on appropriate device."""
        if self.backend == 'cuda' and self.cp:
            return self.cp.zeros(shape, dtype=dtype)
        elif self.backend == 'rocm' and self.torch:
            return self.torch.zeros(shape, dtype=self._np_to_torch_dtype(dtype)).cuda()
        else:
            return np.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=np.float32):
        """Create ones array on appropriate device."""
        if self.backend == 'cuda' and self.cp:
            return self.cp.ones(shape, dtype=dtype)
        elif self.backend == 'rocm' and self.torch:
            return self.torch.ones(shape, dtype=self._np_to_torch_dtype(dtype)).cuda()
        else:
            return np.ones(shape, dtype=dtype)

    def random_uniform(self, shape, low=0.0, high=1.0, dtype=np.float32):
        """Create random uniform array on appropriate device."""
        if self.backend == 'cuda' and self.cp:
            return self.cp.random.uniform(low, high, shape).astype(dtype)
        elif self.backend == 'rocm' and self.torch:
            return self.torch.rand(shape, dtype=self._np_to_torch_dtype(dtype)).cuda() * (high - low) + low
        else:
            return np.random.uniform(low, high, shape).astype(dtype)

    def to_cpu(self, array):
        """Transfer array to CPU."""
        if self.backend == 'cuda' and self.cp:
            return self.cp.asnumpy(array)
        elif self.backend == 'rocm' and self.torch:
            return array.detach().cpu().numpy()
        else:
            return np.asarray(array)

    def synchronize(self):
        """Synchronize GPU operations."""
        if self.backend == 'cuda' and self.cp:
            self.cp.cuda.Stream.null.synchronize()
        elif self.backend == 'rocm' and self.torch:
            self.torch.cuda.synchronize()

    def _np_to_torch_dtype(self, np_dtype):
        """Convert NumPy dtype to PyTorch dtype."""
        if not self.torch:
            return None

        dtype_map = {
            np.float32: self.torch.float32,
            np.float64: self.torch.float64,
            np.int32: self.torch.int32,
            np.int64: self.torch.int64,
            np.bool_: self.torch.bool
        }
        return dtype_map.get(np_dtype, self.torch.float32)

    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the current backend."""
        return {
            'backend': self.backend,
            'available_backends': self.available_backends,
            'device_info': self.device_info
        }

    def is_gpu_available(self) -> bool:
        """Check if GPU acceleration is available."""
        return self.backend in ['cuda', 'rocm']

    def get_memory_info(self) -> Optional[Dict[str, int]]:
        """Get GPU memory information."""
        if self.backend == 'cuda' and self.cp:
            device = self.cp.cuda.Device()
            mem_info = device.mem_info
            return {
                'free': mem_info[0],
                'total': mem_info[1],
                'used': mem_info[1] - mem_info[0]
            }
        elif self.backend == 'rocm' and self.torch:
            if self.torch.cuda.is_available():
                return {
                    'free': self.torch.cuda.memory_reserved() - self.torch.cuda.memory_allocated(),
                    'total': self.torch.cuda.memory_reserved(),
                    'used': self.torch.cuda.memory_allocated()
                }
        return None

    # Mathematical operations with automatic CPU fallback
    def abs(self, array):
        """Absolute value with CPU fallback."""
        if self.backend == 'cuda' and self.cp:
            return self.cp.abs(array)
        elif self.backend == 'rocm' and self.torch:
            return self.torch.abs(array)
        else:
            return np.abs(array)

    def where(self, condition, x, y):
        """Conditional selection with CPU fallback."""
        if self.backend == 'cuda' and self.cp:
            return self.cp.where(condition, x, y)
        elif self.backend == 'rocm' and self.torch:
            return self.torch.where(condition, x, y)
        else:
            return np.where(condition, x, y)

    def maximum(self, x, y):
        """Element-wise maximum with CPU fallback."""
        if self.backend == 'cuda' and self.cp:
            return self.cp.maximum(x, y)
        elif self.backend == 'rocm' and self.torch:
            return self.torch.maximum(x, y)
        else:
            return np.maximum(x, y)

    def minimum(self, x, y):
        """Element-wise minimum with CPU fallback."""
        if self.backend == 'cuda' and self.cp:
            return self.cp.minimum(x, y)
        elif self.backend == 'rocm' and self.torch:
            return self.torch.minimum(x, y)
        else:
            return np.minimum(x, y)

    def log(self, array):
        """Natural logarithm with CPU fallback."""
        if self.backend == 'cuda' and self.cp:
            return self.cp.log(array)
        elif self.backend == 'rocm' and self.torch:
            return self.torch.log(array)
        else:
            return np.log(array)

    def exp(self, array):
        """Exponential function with CPU fallback."""
        if self.backend == 'cuda' and self.cp:
            return self.cp.exp(array)
        elif self.backend == 'rocm' and self.torch:
            return self.torch.exp(array)
        else:
            return np.exp(array)

    def clip(self, array, min_val, max_val):
        """Clip values with CPU fallback."""
        if self.backend == 'cuda' and self.cp:
            return self.cp.clip(array, min_val, max_val)
        elif self.backend == 'rocm' and self.torch:
            return self.torch.clamp(array, min_val, max_val)
        else:
            return np.clip(array, min_val, max_val)

    def mean(self, array, axis=None):
        """Mean with CPU fallback."""
        if self.backend == 'cuda' and self.cp:
            return self.cp.mean(array, axis=axis)
        elif self.backend == 'rocm' and self.torch:
            if axis is None:
                return self.torch.mean(array)
            else:
                return self.torch.mean(array, dim=axis)
        else:
            return np.mean(array, axis=axis)

    def sum(self, array, axis=None):
        """Sum with CPU fallback."""
        if self.backend == 'cuda' and self.cp:
            return self.cp.sum(array, axis=axis)
        elif self.backend == 'rocm' and self.torch:
            if axis is None:
                return self.torch.sum(array)
            else:
                return self.torch.sum(array, dim=axis)
        else:
            return np.sum(array, axis=axis)

    def sqrt(self, array):
        """Square root with CPU fallback."""
        if self.backend == 'cuda' and self.cp:
            return self.cp.sqrt(array)
        elif self.backend == 'rocm' and self.torch:
            return self.torch.sqrt(array)
        else:
            return np.sqrt(array)

    def power(self, array, exponent):
        """Power function with CPU fallback."""
        if self.backend == 'cuda' and self.cp:
            return self.cp.power(array, exponent)
        elif self.backend == 'rocm' and self.torch:
            return self.torch.pow(array, exponent)
        else:
            return np.power(array, exponent)

    def random_normal(self, shape, mean=0.0, std=1.0, dtype=np.float32):
        """Generate random normal distribution with CPU fallback."""
        if self.backend == 'cuda' and self.cp:
            return self.cp.random.normal(mean, std, shape).astype(dtype)
        elif self.backend == 'rocm' and self.torch:
            return self.torch.normal(mean, std, shape, dtype=self._np_to_torch_dtype(dtype)).cuda()
        else:
            return np.random.normal(mean, std, shape).astype(dtype)

    def force_cpu_fallback(self):
        """Force CPU fallback mode for testing or when GPU fails."""
        print("🔄 Forcing CPU fallback mode")
        self.backend = 'cpu'
        self.cp = None
        self.torch = None
        self.device_info = {
            'backend': 'cpu',
            'device_count': 0,
            'device_name': 'CPU (Fallback Mode)'
        }
        print("💻 Now using CPU-only computation")

    def test_gpu_functionality(self) -> bool:
        """Test GPU functionality and fallback to CPU if needed."""
        if self.backend == 'cpu':
            return True

        try:
            print("🧪 Testing GPU functionality...")

            # Create test arrays
            a = self.array([1.0, 2.0, 3.0])
            b = self.array([4.0, 5.0, 6.0])

            # Test basic operations
            c = a + b
            self.abs(a - b)
            e = self.maximum(a, b)
            self.mean(c)

            # Test conversion back to CPU
            result = self.to_cpu(c)
            expected = np.array([5.0, 7.0, 9.0])

            if not np.allclose(result, expected, rtol=1e-5):
                raise ValueError("GPU computation results don't match expected values")

            print("✅ GPU functionality test passed")
            return True

        except Exception as e:
            print(f"❌ GPU functionality test failed: {e}")
            print("🔄 Falling back to CPU mode...")
            self.force_cpu_fallback()
            return False


# Global GPU accelerator instance
_gpu_accelerator = None

def get_gpu_accelerator(prefer_backend: str = 'auto', force_reinit: bool = False, test_functionality: bool = True) -> GPUAccelerator:
    """
    Get global GPU accelerator instance with automatic CPU fallback.
    
    Args:
        prefer_backend: Preferred backend ('auto', 'cuda', 'rocm', 'cpu')
        force_reinit: Force reinitialization even if already created
        test_functionality: Test GPU functionality and fallback to CPU if needed
    
    Returns:
        GPUAccelerator instance
    """
    global _gpu_accelerator

    if _gpu_accelerator is None or force_reinit:
        _gpu_accelerator = GPUAccelerator(prefer_backend)

        # Test GPU functionality and fallback to CPU if needed
        if test_functionality and _gpu_accelerator.is_gpu_available():
            _gpu_accelerator.test_gpu_functionality()

    return _gpu_accelerator


def benchmark_backends():
    """Benchmark available GPU backends for matrix operations."""
    print("\n🏁 GPU Backend Benchmark")
    print("=" * 50)

    # Test parameters
    size = 1000
    iterations = 3

    results = {}

    # Test CPU
    print("\n💻 Testing CPU (NumPy)...")
    import time
    cpu_times = []

    for _ in range(iterations):
        a = np.random.rand(size, size).astype(np.float32)
        b = np.random.rand(size, size).astype(np.float32)

        start_time = time.time()
        np.dot(a, b)
        cpu_time = time.time() - start_time
        cpu_times.append(cpu_time)

    results['cpu'] = sum(cpu_times) / len(cpu_times)
    print(f"   Average time: {results['cpu']:.4f}s")

    # Test available GPU backends
    for backend in ['cuda', 'rocm']:
        try:
            gpu_acc = GPUAccelerator(prefer_backend=backend)
            if gpu_acc.backend == backend:
                print(f"\n🚀 Testing {backend.upper()}...")

                gpu_times = []
                for _ in range(iterations):
                    a = gpu_acc.random_uniform((size, size))
                    b = gpu_acc.random_uniform((size, size))

                    start_time = time.time()
                    if backend == 'cuda':
                        gpu_acc.cp.dot(a, b)
                    else:  # rocm
                        gpu_acc.torch.mm(a, b)
                    gpu_acc.synchronize()
                    gpu_time = time.time() - start_time
                    gpu_times.append(gpu_time)

                results[backend] = sum(gpu_times) / len(gpu_times)
                speedup = results['cpu'] / results[backend]
                print(f"   Average time: {results[backend]:.4f}s")
                print(f"   Speedup vs CPU: {speedup:.2f}x")

        except Exception as e:
            print(f"❌ {backend.upper()} benchmark failed: {e}")

    print(f"\n📊 Benchmark Summary (Matrix size: {size}x{size})")
    print("-" * 30)
    for backend, time_taken in results.items():
        speedup = results['cpu'] / time_taken if backend != 'cpu' else 1.0
        print(f"{backend.upper():>8}: {time_taken:.4f}s ({speedup:.2f}x)")


if __name__ == '__main__':
    # Demo the GPU accelerator
    print("🔍 GPU Accelerator Demo")
    print("=" * 40)

    # Test auto-detection
    gpu_acc = get_gpu_accelerator()
    print(f"\nBackend info: {gpu_acc.get_backend_info()}")

    # Test basic operations
    print(f"\n🧪 Testing basic operations with {gpu_acc.backend.upper()} backend...")

    # Create test arrays
    a = gpu_acc.array([1, 2, 3, 4, 5])
    b = gpu_acc.array([2, 3, 4, 5, 6])

    print(f"Array a: {gpu_acc.to_cpu(a)}")
    print(f"Array b: {gpu_acc.to_cpu(b)}")

    # Test arithmetic operations
    if gpu_acc.backend == 'cuda':
        c = a + b
    elif gpu_acc.backend == 'rocm':
        c = a + b
    else:
        c = a + b

    print(f"Sum a+b: {gpu_acc.to_cpu(c)}")

    # Memory info
    mem_info = gpu_acc.get_memory_info()
    if mem_info:
        print("\n💾 Memory info:")
        print(f"   Total: {mem_info['total'] // 1024**2} MB")
        print(f"   Used:  {mem_info['used'] // 1024**2} MB")
        print(f"   Free:  {mem_info['free'] // 1024**2} MB")

    # Run benchmark if requested
    if len(sys.argv) > 1 and sys.argv[1] == '--benchmark':
        benchmark_backends()
