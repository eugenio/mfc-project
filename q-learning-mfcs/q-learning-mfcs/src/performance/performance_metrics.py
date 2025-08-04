"""
Performance metrics collection and analysis for MFC simulations.

This module provides comprehensive performance monitoring capabilities including:
- System resource usage tracking (CPU, memory, GPU)
- Simulation performance metrics (throughput, latency, efficiency)
- Hardware acceleration benchmarking
- Performance trend analysis and optimization suggestions
"""

import statistics
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import psutil


@dataclass
class CPUMetrics:
    """CPU performance metrics."""
    usage_percent: float
    usage_per_core: list[float]
    frequency_current: float
    frequency_max: float
    load_average: list[float]
    context_switches: int
    interrupts: int
    timestamp: float


@dataclass
class MemoryMetrics:
    """Memory usage metrics."""
    total_gb: float
    used_gb: float
    available_gb: float
    usage_percent: float
    swap_total_gb: float
    swap_used_gb: float
    swap_percent: float
    timestamp: float


@dataclass
class GPUMetrics:
    """GPU performance metrics."""
    gpu_count: int
    usage_percent: list[float]
    memory_used_gb: list[float]
    memory_total_gb: list[float]
    memory_percent: list[float]
    temperature_c: list[float]
    power_watts: list[float]
    backend_type: str  # 'cuda', 'rocm', 'mps', or 'cpu'
    timestamp: float


@dataclass
class SimulationMetrics:
    """Simulation-specific performance metrics."""
    name: str
    start_time: float
    end_time: float | None = None
    duration_sec: float | None = None
    iterations_completed: int = 0
    iterations_per_second: float = 0.0
    convergence_rate: float = 0.0
    memory_peak_gb: float = 0.0
    cpu_time_sec: float = 0.0
    gpu_time_sec: float = 0.0
    efficiency_ratio: float = 1.0  # GPU time / CPU time
    status: str = "running"  # running, completed, failed, paused


@dataclass
class BenchmarkResult:
    """Benchmark test results."""
    test_name: str
    backend: str
    operation_type: str
    data_size_mb: float
    execution_time_ms: float
    throughput_mbps: float
    operations_per_second: float
    memory_bandwidth_gbps: float
    timestamp: float


class PerformanceCollector:
    """Collects and aggregates system performance metrics."""

    def __init__(self, collection_interval: float = 1.0):
        self.collection_interval = collection_interval
        self.collecting = False
        self.collection_thread: threading.Thread | None = None

        # Historical data storage (circular buffers)
        self.max_history = 1000
        self.cpu_history: deque = deque(maxlen=self.max_history)
        self.memory_history: deque = deque(maxlen=self.max_history)
        self.gpu_history: deque = deque(maxlen=self.max_history)

        # Callbacks for real-time monitoring
        self.metric_callbacks: list[Callable] = []

    def add_callback(self, callback: Callable[[dict[str, Any]], None]):
        """Add callback for real-time metric updates."""
        self.metric_callbacks.append(callback)

    def start_collection(self):
        """Start background metric collection."""
        if self.collecting:
            return

        self.collecting = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()

    def stop_collection(self):
        """Stop background metric collection."""
        self.collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=2.0)

    def _collection_loop(self):
        """Main collection loop running in background thread."""
        while self.collecting:
            try:
                metrics = self.collect_current_metrics()

                # Store in history
                self.cpu_history.append(metrics['cpu'])
                self.memory_history.append(metrics['memory'])
                if metrics['gpu']:
                    self.gpu_history.append(metrics['gpu'])

                # Notify callbacks
                for callback in self.metric_callbacks:
                    callback(metrics)

            except Exception as e:
                print(f"Error in metric collection: {e}")

            time.sleep(self.collection_interval)

    def collect_current_metrics(self) -> dict[str, Any]:
        """Collect current system metrics."""
        timestamp = time.time()

        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
        cpu_freq = psutil.cpu_freq()
        load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0.0, 0.0, 0.0]
        psutil.boot_time()
        cpu_stats = psutil.cpu_stats()

        cpu_metrics = CPUMetrics(
            usage_percent=sum(cpu_percent) / len(cpu_percent),
            usage_per_core=cpu_percent,
            frequency_current=cpu_freq.current if cpu_freq else 0.0,
            frequency_max=cpu_freq.max if cpu_freq else 0.0,
            load_average=list(load_avg),
            context_switches=cpu_stats.ctx_switches,
            interrupts=cpu_stats.interrupts,
            timestamp=timestamp
        )

        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        memory_metrics = MemoryMetrics(
            total_gb=memory.total / (1024**3),
            used_gb=memory.used / (1024**3),
            available_gb=memory.available / (1024**3),
            usage_percent=memory.percent,
            swap_total_gb=swap.total / (1024**3),
            swap_used_gb=swap.used / (1024**3),
            swap_percent=swap.percent,
            timestamp=timestamp
        )

        # GPU metrics (attempt to collect from available backends)
        gpu_metrics = self._collect_gpu_metrics(timestamp)

        return {
            'cpu': cpu_metrics,
            'memory': memory_metrics,
            'gpu': gpu_metrics,
            'timestamp': timestamp
        }

    def _collect_gpu_metrics(self, timestamp: float) -> GPUMetrics | None:
        """Collect GPU metrics from available backends."""
        try:
            # Try NVIDIA first
            try:
                import pynvml
                pynvml.nvmlInit()
                gpu_count = pynvml.nvmlDeviceGetCount()

                usage_percent = []
                memory_used_gb = []
                memory_total_gb = []
                memory_percent = []
                temperature_c = []
                power_watts = []

                for i in range(gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                    # Utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    usage_percent.append(util.gpu)

                    # Memory
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    memory_used_gb.append(mem_info.used / (1024**3))
                    memory_total_gb.append(mem_info.total / (1024**3))
                    memory_percent.append((mem_info.used / mem_info.total) * 100)

                    # Temperature
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        temperature_c.append(temp)
                    except Exception:
                        temperature_c.append(0.0)

                    # Power
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                        power_watts.append(power)
                    except Exception:
                        power_watts.append(0.0)

                return GPUMetrics(
                    gpu_count=gpu_count,
                    usage_percent=usage_percent,
                    memory_used_gb=memory_used_gb,
                    memory_total_gb=memory_total_gb,
                    memory_percent=memory_percent,
                    temperature_c=temperature_c,
                    power_watts=power_watts,
                    backend_type='cuda',
                    timestamp=timestamp
                )

            except ImportError:
                pass

            # Try AMD ROCm
            try:
                import subprocess
                result = subprocess.run(['rocm-smi', '--showuse', '--showmemuse', '--showtemp'],
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    # Parse rocm-smi output
                    result.stdout.split('\n')
                    # This is a simplified parser - would need more robust parsing
                    return GPUMetrics(
                        gpu_count=1,  # Simplified
                        usage_percent=[0.0],
                        memory_used_gb=[0.0],
                        memory_total_gb=[8.0],  # Estimate
                        memory_percent=[0.0],
                        temperature_c=[0.0],
                        power_watts=[0.0],
                        backend_type='rocm',
                        timestamp=timestamp
                    )
            except Exception:
                pass

            # Try PyTorch for basic GPU info
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    memory_used = [torch.cuda.memory_allocated(i) / (1024**3) for i in range(gpu_count)]
                    memory_total = [torch.cuda.get_device_properties(i).total_memory / (1024**3) for i in range(gpu_count)]
                    memory_percent = [(used/total)*100 for used, total in zip(memory_used, memory_total, strict=False)]

                    backend_type = 'rocm' if hasattr(torch.version, 'hip') and torch.version.hip else 'cuda'

                    return GPUMetrics(
                        gpu_count=gpu_count,
                        usage_percent=[0.0] * gpu_count,  # Can't get utilization from PyTorch alone
                        memory_used_gb=memory_used,
                        memory_total_gb=memory_total,
                        memory_percent=memory_percent,
                        temperature_c=[0.0] * gpu_count,
                        power_watts=[0.0] * gpu_count,
                        backend_type=backend_type,
                        timestamp=timestamp
                    )
            except Exception:
                pass

        except Exception as e:
            print(f"Error collecting GPU metrics: {e}")

        return None

    def get_statistics(self, metric_type: str, duration_minutes: int = 10) -> dict[str, float]:
        """Get statistical summary of metrics over specified duration."""
        if metric_type == 'cpu' and self.cpu_history:
            recent_metrics = [m for m in self.cpu_history
                            if time.time() - m.timestamp <= duration_minutes * 60]
            if recent_metrics:
                usage_values = [m.usage_percent for m in recent_metrics]
                return {
                    'mean': statistics.mean(usage_values),
                    'median': statistics.median(usage_values),
                    'std_dev': statistics.stdev(usage_values) if len(usage_values) > 1 else 0.0,
                    'min': min(usage_values),
                    'max': max(usage_values),
                    'samples': len(usage_values)
                }

        elif metric_type == 'memory' and self.memory_history:
            recent_metrics = [m for m in self.memory_history
                            if time.time() - m.timestamp <= duration_minutes * 60]
            if recent_metrics:
                usage_values = [m.usage_percent for m in recent_metrics]
                return {
                    'mean': statistics.mean(usage_values),
                    'median': statistics.median(usage_values),
                    'std_dev': statistics.stdev(usage_values) if len(usage_values) > 1 else 0.0,
                    'min': min(usage_values),
                    'max': max(usage_values),
                    'samples': len(usage_values)
                }

        elif metric_type == 'gpu' and self.gpu_history:
            recent_metrics = [m for m in self.gpu_history
                            if time.time() - m.timestamp <= duration_minutes * 60]
            if recent_metrics:
                # Average across all GPUs
                usage_values = [sum(m.usage_percent)/len(m.usage_percent) for m in recent_metrics if m.usage_percent]
                if usage_values:
                    return {
                        'mean': statistics.mean(usage_values),
                        'median': statistics.median(usage_values),
                        'std_dev': statistics.stdev(usage_values) if len(usage_values) > 1 else 0.0,
                        'min': min(usage_values),
                        'max': max(usage_values),
                        'samples': len(usage_values)
                    }

        return {}


class SimulationTracker:
    """Tracks performance metrics for running simulations."""

    def __init__(self):
        self.active_simulations: dict[str, SimulationMetrics] = {}
        self.completed_simulations: list[SimulationMetrics] = []
        self.performance_collector = PerformanceCollector()

    def start_simulation(self, name: str) -> str:
        """Start tracking a new simulation."""
        sim_id = f"{name}_{int(time.time())}"
        self.active_simulations[sim_id] = SimulationMetrics(
            name=name,
            start_time=time.time()
        )
        return sim_id

    def update_simulation(self, sim_id: str, **kwargs):
        """Update simulation metrics."""
        if sim_id in self.active_simulations:
            sim = self.active_simulations[sim_id]
            for key, value in kwargs.items():
                if hasattr(sim, key):
                    setattr(sim, key, value)

    def complete_simulation(self, sim_id: str):
        """Mark simulation as completed and calculate final metrics."""
        if sim_id in self.active_simulations:
            sim = self.active_simulations[sim_id]
            sim.end_time = time.time()
            sim.duration_sec = sim.end_time - sim.start_time
            sim.status = "completed"

            # Calculate iterations per second
            if sim.duration_sec > 0:
                sim.iterations_per_second = sim.iterations_completed / sim.duration_sec

            self.completed_simulations.append(sim)
            del self.active_simulations[sim_id]

    def get_simulation_summary(self, sim_id: str) -> dict[str, Any]:
        """Get comprehensive summary of simulation performance."""
        sim = self.active_simulations.get(sim_id) or next(
            (s for s in self.completed_simulations if f"{s.name}_{int(s.start_time)}" == sim_id),
            None
        )

        if not sim:
            return {}

        return {
            'name': sim.name,
            'status': sim.status,
            'duration_sec': sim.duration_sec or (time.time() - sim.start_time),
            'iterations_completed': sim.iterations_completed,
            'iterations_per_second': sim.iterations_per_second,
            'memory_peak_gb': sim.memory_peak_gb,
            'cpu_time_sec': sim.cpu_time_sec,
            'gpu_time_sec': sim.gpu_time_sec,
            'efficiency_ratio': sim.efficiency_ratio,
            'convergence_rate': sim.convergence_rate
        }


class PerformanceBenchmark:
    """Benchmarking suite for different hardware backends."""

    def __init__(self):
        self.benchmark_results: list[BenchmarkResult] = []

    def benchmark_array_operations(self, backend: str, data_sizes: list[int]) -> list[BenchmarkResult]:
        """Benchmark array operations across different data sizes."""
        results = []

        for size in data_sizes:
            # Create test data
            data_size_mb = size * 4 / (1024 * 1024)  # float32 = 4 bytes

            try:
                if backend == 'numpy':
                    results.extend(self._benchmark_numpy(size, data_size_mb))
                elif backend == 'cuda':
                    results.extend(self._benchmark_cuda(size, data_size_mb))
                elif backend == 'rocm':
                    results.extend(self._benchmark_rocm(size, data_size_mb))

            except Exception as e:
                print(f"Benchmark failed for {backend} with size {size}: {e}")

        self.benchmark_results.extend(results)
        return results

    def _benchmark_numpy(self, size: int, data_size_mb: float) -> list[BenchmarkResult]:
        """Benchmark NumPy operations."""
        results = []

        # Create test arrays
        a = np.random.rand(size).astype(np.float32)
        b = np.random.rand(size).astype(np.float32)

        # Addition benchmark
        start_time = time.perf_counter()
        a + b
        end_time = time.perf_counter()

        execution_time_ms = (end_time - start_time) * 1000
        throughput_mbps = (data_size_mb * 2) / (execution_time_ms / 1000)  # 2 arrays
        ops_per_sec = size / (execution_time_ms / 1000)

        results.append(BenchmarkResult(
            test_name="array_addition",
            backend="numpy",
            operation_type="elementwise",
            data_size_mb=data_size_mb,
            execution_time_ms=execution_time_ms,
            throughput_mbps=throughput_mbps,
            operations_per_second=ops_per_sec,
            memory_bandwidth_gbps=throughput_mbps / 1000,
            timestamp=time.time()
        ))

        # Matrix multiplication benchmark (if size allows)
        if size <= 2048:  # Avoid excessive memory usage
            matrix_size = int(np.sqrt(size))
            if matrix_size * matrix_size == size:
                a_mat = a.reshape(matrix_size, matrix_size)
                b_mat = b.reshape(matrix_size, matrix_size)

                start_time = time.perf_counter()
                np.dot(a_mat, b_mat)
                end_time = time.perf_counter()

                execution_time_ms = (end_time - start_time) * 1000
                # FLOPS = 2 * N^3 for matrix multiplication
                flops = 2 * matrix_size**3
                flops_per_sec = flops / (execution_time_ms / 1000)

                results.append(BenchmarkResult(
                    test_name="matrix_multiplication",
                    backend="numpy",
                    operation_type="linear_algebra",
                    data_size_mb=data_size_mb,
                    execution_time_ms=execution_time_ms,
                    throughput_mbps=0.0,  # Not applicable for matrix ops
                    operations_per_second=flops_per_sec,
                    memory_bandwidth_gbps=0.0,  # Would need more detailed calculation
                    timestamp=time.time()
                ))

        return results

    def _benchmark_cuda(self, size: int, data_size_mb: float) -> list[BenchmarkResult]:
        """Benchmark CUDA operations."""
        try:
            import cupy as cp

            results = []

            # Create test arrays on GPU
            a = cp.random.rand(size, dtype=cp.float32)
            b = cp.random.rand(size, dtype=cp.float32)

            # Warm up
            a + b
            cp.cuda.Stream.null.synchronize()

            # Addition benchmark
            start_time = time.perf_counter()
            a + b
            cp.cuda.Stream.null.synchronize()
            end_time = time.perf_counter()

            execution_time_ms = (end_time - start_time) * 1000
            throughput_mbps = (data_size_mb * 2) / (execution_time_ms / 1000)
            ops_per_sec = size / (execution_time_ms / 1000)

            results.append(BenchmarkResult(
                test_name="array_addition",
                backend="cuda",
                operation_type="elementwise",
                data_size_mb=data_size_mb,
                execution_time_ms=execution_time_ms,
                throughput_mbps=throughput_mbps,
                operations_per_second=ops_per_sec,
                memory_bandwidth_gbps=throughput_mbps / 1000,
                timestamp=time.time()
            ))

            return results

        except ImportError:
            return []

    def _benchmark_rocm(self, size: int, data_size_mb: float) -> list[BenchmarkResult]:
        """Benchmark ROCm operations."""
        try:
            import torch

            if not torch.cuda.is_available():
                return []

            results = []

            # Create test tensors on GPU
            a = torch.rand(size, dtype=torch.float32, device='cuda')
            b = torch.rand(size, dtype=torch.float32, device='cuda')

            # Warm up
            a + b
            torch.cuda.synchronize()

            # Addition benchmark
            start_time = time.perf_counter()
            a + b
            torch.cuda.synchronize()
            end_time = time.perf_counter()

            execution_time_ms = (end_time - start_time) * 1000
            throughput_mbps = (data_size_mb * 2) / (execution_time_ms / 1000)
            ops_per_sec = size / (execution_time_ms / 1000)

            backend_name = 'rocm' if hasattr(torch.version, 'hip') and torch.version.hip else 'cuda'

            results.append(BenchmarkResult(
                test_name="array_addition",
                backend=backend_name,
                operation_type="elementwise",
                data_size_mb=data_size_mb,
                execution_time_ms=execution_time_ms,
                throughput_mbps=throughput_mbps,
                operations_per_second=ops_per_sec,
                memory_bandwidth_gbps=throughput_mbps / 1000,
                timestamp=time.time()
            ))

            return results

        except ImportError:
            return []

    def get_benchmark_summary(self) -> dict[str, Any]:
        """Get summary of all benchmark results."""
        if not self.benchmark_results:
            return {}

        # Group by backend
        backend_results = {}
        for result in self.benchmark_results:
            if result.backend not in backend_results:
                backend_results[result.backend] = []
            backend_results[result.backend].append(result)

        summary = {}
        for backend, results in backend_results.items():
            if results:
                throughputs = [r.throughput_mbps for r in results if r.throughput_mbps > 0]
                execution_times = [r.execution_time_ms for r in results]

                summary[backend] = {
                    'test_count': len(results),
                    'avg_throughput_mbps': statistics.mean(throughputs) if throughputs else 0.0,
                    'avg_execution_time_ms': statistics.mean(execution_times),
                    'best_throughput_mbps': max(throughputs) if throughputs else 0.0,
                    'fastest_execution_ms': min(execution_times)
                }

        return summary


class PerformanceAnalyzer:
    """Analyzes performance data and provides optimization recommendations."""

    def __init__(self):
        self.analysis_history: list[dict[str, Any]] = []

    def analyze_system_performance(self, collector: PerformanceCollector) -> dict[str, Any]:
        """Analyze current system performance and provide recommendations."""
        if not any([collector.cpu_history, collector.memory_history, collector.gpu_history]):
            return {'error': 'No performance data available'}

        analysis = {
            'timestamp': time.time(),
            'system_health': 'good',
            'bottlenecks': [],
            'recommendations': [],
            'resource_utilization': {},
            'performance_score': 100.0
        }

        # Analyze CPU performance
        if collector.cpu_history:
            recent_cpu = list(collector.cpu_history)[-10:]  # Last 10 measurements
            avg_cpu_usage = sum(m.usage_percent for m in recent_cpu) / len(recent_cpu)
            max_cpu_usage = max(m.usage_percent for m in recent_cpu)

            analysis['resource_utilization']['cpu'] = {
                'average_usage': avg_cpu_usage,
                'peak_usage': max_cpu_usage,
                'status': 'optimal' if avg_cpu_usage < 70 else 'high' if avg_cpu_usage < 90 else 'critical'
            }

            if avg_cpu_usage > 85:
                analysis['bottlenecks'].append('High CPU utilization')
                analysis['recommendations'].append('Consider reducing simulation complexity or using GPU acceleration')
                analysis['performance_score'] -= 20
            elif avg_cpu_usage > 70:
                analysis['system_health'] = 'warning'
                analysis['recommendations'].append('Monitor CPU usage and consider optimization')
                analysis['performance_score'] -= 10

        # Analyze memory performance
        if collector.memory_history:
            recent_memory = list(collector.memory_history)[-10:]
            avg_memory_usage = sum(m.usage_percent for m in recent_memory) / len(recent_memory)
            max_memory_usage = max(m.usage_percent for m in recent_memory)

            analysis['resource_utilization']['memory'] = {
                'average_usage': avg_memory_usage,
                'peak_usage': max_memory_usage,
                'status': 'optimal' if avg_memory_usage < 80 else 'high' if avg_memory_usage < 95 else 'critical'
            }

            if avg_memory_usage > 90:
                analysis['bottlenecks'].append('High memory utilization')
                analysis['recommendations'].append('Reduce batch size or enable memory optimization')
                analysis['performance_score'] -= 25
                analysis['system_health'] = 'critical'
            elif avg_memory_usage > 80:
                analysis['system_health'] = 'warning' if analysis['system_health'] == 'good' else analysis['system_health']
                analysis['recommendations'].append('Monitor memory usage closely')
                analysis['performance_score'] -= 15

        # Analyze GPU performance
        if collector.gpu_history:
            recent_gpu = list(collector.gpu_history)[-10:]
            if recent_gpu and recent_gpu[0].usage_percent:
                # Average across GPUs and time
                avg_gpu_usage = sum(
                    sum(m.usage_percent) / len(m.usage_percent)
                    for m in recent_gpu if m.usage_percent
                ) / len(recent_gpu)

                avg_gpu_memory = sum(
                    sum(m.memory_percent) / len(m.memory_percent)
                    for m in recent_gpu if m.memory_percent
                ) / len(recent_gpu)

                analysis['resource_utilization']['gpu'] = {
                    'average_usage': avg_gpu_usage,
                    'average_memory': avg_gpu_memory,
                    'backend': recent_gpu[0].backend_type,
                    'status': 'optimal' if avg_gpu_usage > 20 else 'underutilized'
                }

                if avg_gpu_usage < 10:
                    analysis['recommendations'].append('GPU is underutilized - consider CPU-only mode for small workloads')
                elif avg_gpu_usage > 95:
                    analysis['recommendations'].append('GPU is at maximum capacity - excellent utilization')
                    analysis['performance_score'] += 10

        self.analysis_history.append(analysis)
        return analysis

    def get_optimization_suggestions(self, simulation_history: list[SimulationMetrics]) -> list[str]:
        """Generate optimization suggestions based on simulation history."""
        if not simulation_history:
            return ["No simulation history available for analysis"]

        suggestions = []

        # Analyze completion times
        completion_times = [s.duration_sec for s in simulation_history if s.duration_sec]
        if completion_times:
            avg_time = statistics.mean(completion_times)
            if avg_time > 3600:  # More than 1 hour
                suggestions.append("Simulations are taking long - consider distributed computing or GPU acceleration")
            elif avg_time < 60:  # Less than 1 minute
                suggestions.append("Fast simulations - current setup is well optimized")

        # Analyze iteration rates
        iteration_rates = [s.iterations_per_second for s in simulation_history if s.iterations_per_second > 0]
        if iteration_rates:
            if statistics.mean(iteration_rates) < 10:
                suggestions.append("Low iteration rate - check for computational bottlenecks")

        # Analyze efficiency ratios
        efficiency_ratios = [s.efficiency_ratio for s in simulation_history if s.efficiency_ratio > 0]
        if efficiency_ratios:
            avg_efficiency = statistics.mean(efficiency_ratios)
            if avg_efficiency > 10:  # GPU much faster than CPU
                suggestions.append("Excellent GPU acceleration - consider increasing model complexity")
            elif avg_efficiency < 2:  # GPU not much faster
                suggestions.append("GPU acceleration may not be beneficial for current workload")

        return suggestions or ["System performance appears optimal"]
