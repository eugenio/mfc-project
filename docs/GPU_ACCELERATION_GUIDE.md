# GPU Acceleration Guide

## Overview

The MFC Q-learning project now features universal GPU acceleration that supports multiple hardware vendors and provides automatic CPU fallback. This guide covers the GPU acceleration implementation, supported backends, and usage instructions.

## Supported Backends

### NVIDIA CUDA
- **Library**: CuPy
- **Requirements**: CUDA Toolkit, NVIDIA GPU
- **Installation**: `pip install cupy-cuda12x` (adjust version for your CUDA)
- **Detection**: Automatic via `nvidia-smi`

### AMD ROCm
- **Library**: PyTorch with ROCm support
- **Requirements**: ROCm drivers, AMD GPU (e.g., Radeon RX 7900 XTX)
- **Installation**: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0`
- **Detection**: Automatic via `rocm-smi` or PyTorch

### CPU Fallback
- **Library**: NumPy
- **Requirements**: None (always available)
- **Performance**: Baseline performance without acceleration

## Architecture

### GPU Acceleration Module (`gpu_acceleration.py`)

The module provides a unified interface for GPU operations:

```python
from gpu_acceleration import get_gpu_accelerator

# Initialize GPU accelerator (auto-detects backend)
gpu_acc = get_gpu_accelerator()

# Create arrays on GPU
a = gpu_acc.array([1.0, 2.0, 3.0])
b = gpu_acc.array([4.0, 5.0, 6.0])

# Perform operations (automatically uses GPU if available)
c = a + b
result = gpu_acc.abs(a - b)
maximum = gpu_acc.maximum(a, b)

# Convert back to CPU
cpu_result = gpu_acc.to_cpu(result)
```

### Supported Operations

#### Array Operations
- `array()`: Create arrays on appropriate device
- `zeros()`: Create zero arrays
- `to_cpu()`: Transfer arrays to CPU
- `is_gpu_available()`: Check GPU availability

#### Mathematical Operations
- `abs()`: Absolute value
- `log()`: Natural logarithm
- `exp()`: Exponential
- `sqrt()`: Square root
- `power()`: Power function

#### Conditional Operations
- `where()`: Conditional selection
- `maximum()`: Element-wise maximum
- `minimum()`: Element-wise minimum
- `clip()`: Clip values to range

#### Aggregation Operations
- `mean()`: Mean calculation
- `sum()`: Sum calculation

#### Random Generation
- `random_normal()`: Generate normal distribution

## Implementation in Simulation Files

All major simulation files have been converted to use universal GPU acceleration:

### 1. `mfc_unified_qlearning_control.py`
```python
# Import universal GPU acceleration
from gpu_acceleration import get_gpu_accelerator

# Initialize GPU accelerator
gpu_accelerator = get_gpu_accelerator()
GPU_AVAILABLE = gpu_accelerator.is_gpu_available()

# Use in computations
if self.use_gpu:
    delta_opt = gpu_accelerator.abs(thickness - self.optimal_biofilm_thickness)
    outlet_conc = gpu_accelerator.maximum(0.001, inlet_conc - consumed)
```

### 2. `mfc_qlearning_optimization.py`
- Q-learning flow controller with GPU-accelerated computations
- Automatic array transfers between CPU and GPU

### 3. `mfc_dynamic_substrate_control.py`
- Dynamic substrate control with GPU acceleration
- PID controller remains on CPU, simulations on GPU

### 4. `mfc_optimization_gpu.py`
- Multi-objective optimization leveraging GPU
- Removed Numba dependency in favor of universal interface

## Performance Considerations

### GPU Selection Priority
1. NVIDIA CUDA (if available) - Highest performance with CuPy
2. AMD ROCm (if available) - Good performance with PyTorch
3. CPU fallback - Baseline NumPy performance

### Memory Management
- Arrays are created directly on GPU to minimize transfers
- Use `to_cpu()` only when necessary (e.g., saving results)
- GPU memory is automatically managed by backend libraries

### Best Practices
1. **Batch Operations**: Group operations to minimize kernel launches
2. **Minimize Transfers**: Keep data on GPU throughout computation
3. **Check Availability**: Always verify GPU availability before assuming acceleration
4. **Error Handling**: The module automatically falls back to CPU on GPU errors

## Testing

### GPU Capability Tests
Run hardware detection tests:
```bash
python q-learning-mfcs/tests/run_tests.py -c gpu_capability
```

Tests include:
- NVIDIA GPU hardware detection
- AMD GPU hardware detection
- CUDA/ROCm software stack verification
- Library compatibility checks
- Performance benchmarking

### GPU Acceleration Tests
Run functionality tests:
```bash
python q-learning-mfcs/tests/run_tests.py -c gpu_acceleration
```

Tests cover:
- Array creation and conversion
- All mathematical operations
- CPU fallback functionality
- Memory management
- Backend-specific features

## Troubleshooting

### NVIDIA CUDA Issues
1. **CuPy not found**: Install with `pip install cupy-cuda12x`
2. **CUDA version mismatch**: Match CuPy version to CUDA toolkit
3. **Out of memory**: Reduce batch sizes or use CPU fallback

### AMD ROCm Issues
1. **PyTorch not detecting GPU**: Install ROCm-compatible PyTorch
2. **HIP errors**: Update ROCm drivers
3. **Performance issues**: Ensure GPU is in compute mode

### CPU Fallback
- Automatically activated when GPU unavailable
- No configuration needed
- Performance will be lower than GPU

## Environment Setup

### For NVIDIA GPUs
```bash
# Install CUDA Toolkit (if not present)
# Install CuPy
pip install cupy-cuda12x  # Adjust version
```

### For AMD GPUs
```bash
# Install ROCm drivers
# Install PyTorch with ROCm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

### Testing Installation
```python
from gpu_acceleration import get_gpu_accelerator

gpu_acc = get_gpu_accelerator()
print(f"Backend: {gpu_acc.backend}")
print(f"Device: {gpu_acc.device_info}")
```

## Future Enhancements

1. **JAX Backend**: Support for Google's JAX library
2. **Multi-GPU Support**: Distributed computing across multiple GPUs
3. **Mixed Precision**: FP16/BF16 support for newer GPUs
4. **Custom Kernels**: Optimized kernels for MFC-specific operations
5. **Profiling Tools**: Built-in performance profiling

## Conclusion

The universal GPU acceleration module provides seamless hardware acceleration across different vendors while maintaining code portability and reliability through automatic CPU fallback. This ensures optimal performance regardless of the available hardware.