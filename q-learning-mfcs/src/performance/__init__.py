"""Performance Optimization and Benchmarking Module.

from .gpu_memory_manager import (
    GPUMemoryManager,
    ManagedGPUContext,
    MemoryStats,
    PerformanceProfile,
)
from .performance_metrics import (
    MetricsCollector,
    MetricSummary,
    MetricType,
    MetricValue,
    MFCMetricsCollector,
    get_default_collector,
    record_metric,
    start_metrics_collection,
    stop_metrics_collection,
)

__all__ = [
    "BenchmarkResult",
    "GPUMemoryManager",
    "ManagedGPUContext",
    "MemoryStats",
    "PerformanceBenchmark",
    "PerformanceProfile",
    "run_benchmark_demo",
]
