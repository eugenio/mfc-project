"""
Benchmarking Tools for MFC Systems
==================================

Provides comprehensive benchmarking utilities for performance testing
of MFC simulations, Q-learning algorithms, and system components.

Created: 2025-08-05
Author: TDD Agent 10
"""
import contextlib
import functools
import statistics
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import psutil
import torch

from .performance_metrics import MetricType, MFCMetricsCollector


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    duration_seconds: float
    memory_peak_mb: float
    memory_delta_mb: float
    cpu_percent_avg: float
    iterations: int
    success: bool
    error_message: str | None = None
    custom_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    name: str
    results: list[BenchmarkResult] = field(default_factory=list)
    total_duration: float = 0.0
    start_time: float | None = None
    end_time: float | None = None

    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result to the suite."""
        self.results.append(result)

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics for the benchmark suite."""
        if not self.results:
            return {"total_tests": 0, "success_rate": 0.0}

        successful_results = [r for r in self.results if r.success]
        durations = [r.duration_seconds for r in successful_results]
        memory_peaks = [r.memory_peak_mb for r in successful_results]

        return {
            "total_tests": len(self.results),
            "successful_tests": len(successful_results),
            "success_rate": len(successful_results) / len(self.results),
            "total_duration": self.total_duration,
            "avg_duration": statistics.mean(durations) if durations else 0.0,
            "median_duration": statistics.median(durations) if durations else 0.0,
            "max_duration": max(durations) if durations else 0.0,
            "avg_memory_peak": statistics.mean(memory_peaks) if memory_peaks else 0.0,
            "max_memory_peak": max(memory_peaks) if memory_peaks else 0.0,
        }


class PerformanceBenchmark:
    """Main benchmarking utility for MFC systems."""

    def __init__(self, metrics_collector: MFCMetricsCollector | None = None):
        self.metrics_collector = metrics_collector or MFCMetricsCollector()
        self.active_benchmarks: dict[str, dict[str, Any]] = {}
        self.benchmark_history: list[BenchmarkSuite] = []

    @contextlib.contextmanager
    def benchmark_context(self, name: str, iterations: int = 1) -> Iterator[dict[str, Any]]:
        """Context manager for benchmarking code blocks."""
        benchmark_data = {
            "name": name,
            "iterations": iterations,
            "start_time": time.time(),
            "start_memory": self._get_memory_usage(),
            "start_cpu_percent": psutil.cpu_percent(),
            "peak_memory": 0.0,
            "cpu_samples": [],
        }

        self.active_benchmarks[name] = benchmark_data

        try:
            yield benchmark_data
            benchmark_data["success"] = True
            benchmark_data["error_message"] = None
        except Exception as e:
            benchmark_data["success"] = False
            benchmark_data["error_message"] = str(e)
            raise
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()

            result = BenchmarkResult(
                name=name,
                duration_seconds=end_time - benchmark_data["start_time"],
                memory_peak_mb=max(benchmark_data["peak_memory"], end_memory),
                memory_delta_mb=end_memory - benchmark_data["start_memory"],
                cpu_percent_avg=statistics.mean(benchmark_data["cpu_samples"])
                    if benchmark_data["cpu_samples"] else 0.0,
                iterations=iterations,
                success=benchmark_data.get("success", False),
                error_message=benchmark_data.get("error_message"),
                custom_metrics=benchmark_data.get("custom_metrics", {}),
            )

            # Record metrics
            self._record_benchmark_metrics(result)

            # Clean up
            del self.active_benchmarks[name]

    def benchmark_function(self, iterations: int = 1, name: str | None = None):
        """Decorator for benchmarking functions."""
        def decorator(func: Callable) -> Callable:
            benchmark_name = name or f"{func.__module__}.{func.__name__}"

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.benchmark_context(benchmark_name, iterations) as ctx:
                    # Run function multiple times if specified
                    results = []
                    for _i in range(iterations):
                        # Update memory peak during execution
                        current_memory = self._get_memory_usage()
                        ctx["peak_memory"] = max(ctx["peak_memory"], current_memory)

                        # Sample CPU usage
                        ctx["cpu_samples"].append(psutil.cpu_percent())

                        result = func(*args, **kwargs)
                        results.append(result)

                    return results if iterations > 1 else results[0]

            return wrapper
        return decorator

    def run_benchmark_suite(self, suite_name: str,
                          benchmark_functions: list[Callable]) -> BenchmarkSuite:
        """Run a suite of benchmark functions."""
        suite = BenchmarkSuite(name=suite_name)
        suite.start_time = time.time()

        for func in benchmark_functions:
            try:
                if hasattr(func, '__wrapped__'):  # Decorated function
                    func()  # Benchmarking handled by decorator
                else:
                    # Wrap undecorated function
                    with self.benchmark_context(func.__name__):
                        func()
            except Exception as e:
                # Add failed result
                suite.add_result(BenchmarkResult(
                    name=func.__name__,
                    duration_seconds=0.0,
                    memory_peak_mb=0.0,
                    memory_delta_mb=0.0,
                    cpu_percent_avg=0.0,
                    iterations=1,
                    success=False,
                    error_message=str(e)
                ))

        suite.end_time = time.time()
        suite.total_duration = suite.end_time - suite.start_time
        self.benchmark_history.append(suite)

        return suite

    def profile_mfc_simulation(self, simulation_func: Callable,
                             test_cases: list[dict[str, Any]]) -> BenchmarkSuite:
        """Profile MFC simulation performance across different scenarios."""
        suite = BenchmarkSuite(name="MFC_Simulation_Profile")
        suite.start_time = time.time()

        for i, test_case in enumerate(test_cases):
            case_name = test_case.get("name", f"test_case_{i}")

            with self.benchmark_context(case_name) as ctx:
                # Add custom metrics for MFC-specific parameters
                ctx["custom_metrics"] = {
                    "substrate_concentration": test_case.get("substrate_conc", 0.0),
                    "electrode_area": test_case.get("electrode_area", 0.0),
                    "temperature": test_case.get("temperature", 298.15),
                }

                # Run simulation
                result = simulation_func(**test_case)

                # Extract MFC-specific metrics from result
                if isinstance(result, dict):
                    ctx["custom_metrics"].update({
                        "power_output": result.get("power_output", 0.0),
                        "efficiency": result.get("efficiency", 0.0),
                        "convergence_time": result.get("convergence_time", 0.0),
                    })

        suite.end_time = time.time()
        suite.total_duration = suite.end_time - suite.start_time
        return suite

    def profile_qlearning_performance(self, agent_func: Callable,
                                    episodes: int = 100) -> BenchmarkResult:
        """Profile Q-learning agent performance."""
        with self.benchmark_context("Q-Learning_Training", episodes) as ctx:
            # Track Q-learning specific metrics
            ctx["custom_metrics"] = {
                "episodes": episodes,
                "initial_epsilon": 1.0,
            }

            # Run Q-learning training
            training_results = agent_func(episodes=episodes)

            # Extract Q-learning metrics
            if isinstance(training_results, dict):
                ctx["custom_metrics"].update({
                    "final_epsilon": training_results.get("final_epsilon", 0.0),
                    "avg_reward": training_results.get("avg_reward", 0.0),
                    "convergence_episode": training_results.get("convergence_episode", episodes),
                    "qtable_size": training_results.get("qtable_size", 0),
                })

        return ctx

    def profile_gpu_performance(self, gpu_func: Callable,
                              tensor_sizes: list[tuple[int, ...]]) -> BenchmarkSuite:
        """Profile GPU computation performance."""
        suite = BenchmarkSuite(name="GPU_Performance_Profile")
        suite.start_time = time.time()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for size in tensor_sizes:
            size_name = f"tensor_{'x'.join(map(str, size))}"

            with self.benchmark_context(size_name) as ctx:
                # GPU-specific metrics
                if device.type == "cuda":
                    initial_gpu_memory = torch.cuda.memory_allocated()
                    ctx["custom_metrics"] = {
                        "tensor_size": np.prod(size),
                        "initial_gpu_memory_mb": initial_gpu_memory / (1024**2),
                    }

                # Create test tensor
                test_tensor = torch.randn(size, device=device)

                # Run GPU function
                result = gpu_func(test_tensor)

                # Update GPU metrics
                if device.type == "cuda":
                    final_gpu_memory = torch.cuda.memory_allocated()
                    ctx["custom_metrics"].update({
                        "final_gpu_memory_mb": final_gpu_memory / (1024**2),
                        "gpu_memory_delta_mb": (final_gpu_memory - initial_gpu_memory) / (1024**2),
                    })

                # Clean up
                del test_tensor, result
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        suite.end_time = time.time()
        suite.total_duration = suite.end_time - suite.start_time
        return suite

    def generate_performance_report(self, output_path: str | None = None) -> dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            "timestamp": time.time(),
            "system_info": self._get_system_info(),
            "benchmark_history": [
                {
                    "name": suite.name,
                    "summary": suite.get_summary(),
                    "results": [
                        {
                            "name": result.name,
                            "duration": result.duration_seconds,
                            "memory_peak": result.memory_peak_mb,
                            "success": result.success,
                            "custom_metrics": result.custom_metrics,
                        }
                        for result in suite.results
                    ]
                }
                for suite in self.benchmark_history
            ],
            "recommendations": self._generate_performance_recommendations(),
        }

        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)

        return report

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024**2)

    def _record_benchmark_metrics(self, result: BenchmarkResult) -> None:
        """Record benchmark results as metrics."""
        timestamp = None  # Use current time

        self.metrics_collector.record_metric(
            f"benchmark.{result.name}.duration",
            result.duration_seconds,
            timestamp,
            metric_type=MetricType.TIMER
        )

        self.metrics_collector.record_metric(
            f"benchmark.{result.name}.memory_peak",
            result.memory_peak_mb,
            timestamp
        )

        self.metrics_collector.record_metric(
            f"benchmark.{result.name}.success",
            1.0 if result.success else 0.0,
            timestamp,
            metric_type=MetricType.COUNTER
        )

        # Record custom metrics
        for metric_name, value in result.custom_metrics.items():
            self.metrics_collector.record_metric(
                f"benchmark.{result.name}.{metric_name}",
                value,
                timestamp
            )

    def _get_system_info(self) -> dict[str, Any]:
        """Get system information for the report."""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
            "torch_version": torch.__version__,
        }

    def _generate_performance_recommendations(self) -> list[str]:
        """Generate performance optimization recommendations."""
        recommendations = []

        if not self.benchmark_history:
            return ["Run benchmarks to get performance recommendations"]

        # Analyze recent benchmark results
        recent_results = []
        for suite in self.benchmark_history[-5:]:  # Last 5 suites
            recent_results.extend([r for r in suite.results if r.success])

        if not recent_results:
            return ["No successful benchmark results to analyze"]

        # Memory usage analysis
        avg_memory = statistics.mean(r.memory_peak_mb for r in recent_results)
        if avg_memory > 1000:  # > 1GB
            recommendations.append("Consider optimizing memory usage - average peak exceeds 1GB")

        # Duration analysis
        durations = [r.duration_seconds for r in recent_results]
        if len(durations) > 1:
            duration_variance = statistics.variance(durations)
            if duration_variance > statistics.mean(durations):
                recommendations.append("Performance variance is high - consider profiling for bottlenecks")

        # GPU utilization analysis
        gpu_results = [r for r in recent_results if "gpu" in r.name.lower()]
        if gpu_results and not torch.cuda.is_available():
            recommendations.append("GPU benchmarks detected but CUDA not available - consider GPU setup")

        return recommendations


# Utility functions for common benchmarking scenarios
def quick_performance_test(func: Callable, iterations: int = 10) -> BenchmarkResult:
    """Quick performance test for a single function."""
    benchmark = PerformanceBenchmark()

    @benchmark.benchmark_function(iterations=iterations)
    def test_func():
        return func()

    test_func()
    return benchmark.benchmark_history[-1].results[0] if benchmark.benchmark_history else None


def compare_implementations(implementations: dict[str, Callable],
                          iterations: int = 10) -> dict[str, BenchmarkResult]:
    """Compare performance of different implementations."""
    benchmark = PerformanceBenchmark()
    results = {}

    for name, func in implementations.items():
        with benchmark.benchmark_context(name, iterations):
            for _ in range(iterations):
                func()

    # Extract results
    if benchmark.benchmark_history:
        for result in benchmark.benchmark_history[-1].results:
            results[result.name] = result

    return results
