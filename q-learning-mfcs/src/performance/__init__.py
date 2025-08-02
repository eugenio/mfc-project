"""
Performance Optimization and Benchmarking Module

Provides GPU memory management, performance benchmarking,
and system optimization for MFC simulations.
"""

from .benchmark_suite import BenchmarkResult, PerformanceBenchmark, run_benchmark_demo
from .gpu_memory_manager import (
    GPUMemoryManager,
    ManagedGPUContext,
    MemoryStats,
    PerformanceProfile,
)

__all__ = [
    'GPUMemoryManager',
    'ManagedGPUContext',
    'MemoryStats',
    'PerformanceProfile',
    'PerformanceBenchmark',
    'BenchmarkResult',
    'run_benchmark_demo'
]
