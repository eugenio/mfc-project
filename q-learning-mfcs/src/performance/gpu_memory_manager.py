"""GPU Memory Management and Optimization.

Monitors and optimizes GPU memory usage for MFC simulations.
Provides dynamic memory allocation, garbage collection, and performance monitoring.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil
import torch


@dataclass
class MemoryStats:
    """GPU and system memory statistics."""

    gpu_memory_total: float  # GB
    gpu_memory_used: float  # GB
    gpu_memory_free: float  # GB
    gpu_utilization: float  # %
    system_memory_total: float  # GB
    system_memory_used: float  # GB
    system_memory_free: float  # GB
    timestamp: float


@dataclass
class PerformanceProfile:
    """Performance profile for different model sizes."""

    name: str
    max_gpu_memory: float  # GB
    max_system_memory: float  # GB
    batch_size: int
    model_complexity: str  # 'low', 'medium', 'high'
    expected_duration: float  # seconds


class GPUMemoryManager:
    """Manages GPU memory allocation and optimization."""

    def __init__(self, config_path: str | None = None) -> None:
        self.device = self._detect_device()
        self.performance_profiles = self._load_performance_profiles(config_path)
        self.memory_history: list[MemoryStats] = []
        self.optimization_enabled = True

        # Memory thresholds
        self.gpu_memory_warning_threshold = 0.8  # 80%
        self.gpu_memory_critical_threshold = 0.9  # 90%
        self.system_memory_warning_threshold = 0.85  # 85%

        # Performance tracking
        self.allocation_events: list[dict[str, Any]] = []

    def _detect_device(self) -> torch.device:
        """Detect and configure the best available device."""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        return device

    def _load_performance_profiles(
        self,
        config_path: str | None,
    ) -> dict[str, PerformanceProfile]:
        """Load performance profiles for different simulation types."""
        default_profiles = {
            "electrode_optimization": PerformanceProfile(
                name="electrode_optimization",
                max_gpu_memory=2.0,  # GB
                max_system_memory=4.0,  # GB
                batch_size=32,
                model_complexity="medium",
                expected_duration=300,  # 5 minutes
            ),
            "physics_simulation": PerformanceProfile(
                name="physics_simulation",
                max_gpu_memory=4.0,  # GB
                max_system_memory=8.0,  # GB
                batch_size=16,
                model_complexity="high",
                expected_duration=3600,  # 1 hour
            ),
            "ml_training": PerformanceProfile(
                name="ml_training",
                max_gpu_memory=6.0,  # GB
                max_system_memory=12.0,  # GB
                batch_size=64,
                model_complexity="high",
                expected_duration=1800,  # 30 minutes
            ),
            "gsm_analysis": PerformanceProfile(
                name="gsm_analysis",
                max_gpu_memory=1.0,  # GB
                max_system_memory=2.0,  # GB
                batch_size=8,
                model_complexity="low",
                expected_duration=60,  # 1 minute
            ),
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path) as f:
                    custom_config = json.load(f)
                    # Update default profiles with custom settings
                    for name, profile_data in custom_config.items():
                        default_profiles[name] = PerformanceProfile(**profile_data)
            except Exception:
                pass

        return default_profiles

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        # GPU memory
        gpu_memory_total = 0.0
        gpu_memory_used = 0.0
        gpu_utilization = 0.0

        if self.device.type == "cuda":
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (
                1024**3
            )
            gpu_memory_used = torch.cuda.memory_allocated(0) / (1024**3)
            gpu_utilization = self._get_gpu_utilization()
        elif self.device.type == "mps":
            # MPS doesn't provide detailed memory info, estimate
            gpu_memory_total = 8.0  # Estimate for M1/M2 Macs
            gpu_memory_used = (
                torch.mps.current_allocated_memory() / (1024**3)
                if hasattr(torch.mps, "current_allocated_memory")
                else 0.0
            )

        gpu_memory_free = gpu_memory_total - gpu_memory_used

        # System memory
        system_memory = psutil.virtual_memory()
        system_memory_total = system_memory.total / (1024**3)
        system_memory_used = system_memory.used / (1024**3)
        system_memory_free = system_memory.available / (1024**3)

        stats = MemoryStats(
            gpu_memory_total=gpu_memory_total,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_free=gpu_memory_free,
            gpu_utilization=gpu_utilization,
            system_memory_total=system_memory_total,
            system_memory_used=system_memory_used,
            system_memory_free=system_memory_free,
            timestamp=time.time(),
        )

        self.memory_history.append(stats)
        return stats

    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        try:
            if self.device.type == "cuda":
                # Try nvidia-smi
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    return float(result.stdout.strip())
            else:
                # Try rocm-smi for AMD GPUs
                result = subprocess.run(
                    ["rocm-smi", "--showuse"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    # Parse rocm-smi output for GPU utilization
                    lines = result.stdout.split("\n")
                    for line in lines:
                        if "GPU%" in line and "%" in line:
                            parts = line.split()
                            for part in parts:
                                if part.endswith("%") and part[:-1].isdigit():
                                    return float(part[:-1])
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError):
            pass

        return 0.0

    def check_memory_health(self) -> dict[str, Any]:
        """Check memory health and return status."""
        stats = self.get_memory_stats()

        health_status = {
            "status": "healthy",
            "warnings": [],
            "critical_issues": [],
            "recommendations": [],
        }

        # Check GPU memory
        if stats.gpu_memory_total > 0:
            gpu_usage_ratio = stats.gpu_memory_used / stats.gpu_memory_total

            if gpu_usage_ratio > self.gpu_memory_critical_threshold:
                health_status["status"] = "critical"
                health_status["critical_issues"].append(
                    f"GPU memory critically high: {gpu_usage_ratio:.1%}",
                )
                health_status["recommendations"].append(
                    "Reduce batch size or model complexity",
                )
            elif gpu_usage_ratio > self.gpu_memory_warning_threshold:
                health_status["status"] = "warning"
                health_status["warnings"].append(
                    f"GPU memory high: {gpu_usage_ratio:.1%}",
                )
                health_status["recommendations"].append("Consider reducing batch size")

        # Check system memory
        system_usage_ratio = stats.system_memory_used / stats.system_memory_total
        if system_usage_ratio > self.system_memory_warning_threshold:
            health_status["status"] = (
                "warning"
                if health_status["status"] == "healthy"
                else health_status["status"]
            )
            health_status["warnings"].append(
                f"System memory high: {system_usage_ratio:.1%}",
            )
            health_status["recommendations"].append("Close unnecessary applications")

        return health_status

    def optimize_for_profile(self, profile_name: str) -> dict[str, Any]:
        """Optimize memory settings for a specific performance profile."""
        if profile_name not in self.performance_profiles:
            msg = f"Unknown profile: {profile_name}"
            raise ValueError(msg)

        profile = self.performance_profiles[profile_name]
        stats = self.get_memory_stats()

        optimization_result = {
            "profile": profile_name,
            "original_settings": {},
            "optimized_settings": {},
            "memory_freed": 0.0,
            "recommendations": [],
        }

        # Clear GPU cache if needed
        if self.device.type in ["cuda", "mps"]:
            if stats.gpu_memory_used > profile.max_gpu_memory * 0.8:
                self.clear_gpu_cache()
                new_stats = self.get_memory_stats()
                optimization_result["memory_freed"] = (
                    stats.gpu_memory_used - new_stats.gpu_memory_used
                )

        # Suggest optimal batch size
        available_gpu_memory = stats.gpu_memory_free
        if available_gpu_memory < profile.max_gpu_memory:
            suggested_batch_size = max(
                1,
                int(
                    profile.batch_size
                    * (available_gpu_memory / profile.max_gpu_memory),
                ),
            )
            optimization_result["optimized_settings"]["batch_size"] = (
                suggested_batch_size
            )
            optimization_result["recommendations"].append(
                f"Reduce batch size to {suggested_batch_size}",
            )

        return optimization_result

    def clear_gpu_cache(self) -> None:
        """Clear GPU memory cache."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.empty_cache() if hasattr(torch.mps, "empty_cache") else None

    def monitor_allocation(self, name: str, size_mb: float) -> None:
        """Record memory allocation event."""
        event = {
            "name": name,
            "size_mb": size_mb,
            "timestamp": time.time(),
            "memory_stats": self.get_memory_stats(),
        }
        self.allocation_events.append(event)

    def get_performance_report(self) -> dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.memory_history:
            self.get_memory_stats()

        recent_stats = self.memory_history[-10:]  # Last 10 measurements

        return {
            "current_status": self.check_memory_health(),
            "device_info": {
                "type": str(self.device),
                "available": self.device.type != "cpu",
            },
            "memory_trends": {
                "gpu_memory_avg": sum(s.gpu_memory_used for s in recent_stats)
                / len(recent_stats),
                "gpu_memory_peak": max(s.gpu_memory_used for s in recent_stats),
                "system_memory_avg": sum(s.system_memory_used for s in recent_stats)
                / len(recent_stats),
                "gpu_utilization_avg": sum(s.gpu_utilization for s in recent_stats)
                / len(recent_stats),
            },
            "allocation_events": len(self.allocation_events),
            "optimization_suggestions": self._generate_optimization_suggestions(),
        }

    def _generate_optimization_suggestions(self) -> list[str]:
        """Generate optimization suggestions based on usage patterns."""
        suggestions = []

        if len(self.memory_history) < 5:
            return ["Collect more performance data for better suggestions"]

        recent_stats = self.memory_history[-10:]
        avg_gpu_usage = sum(s.gpu_memory_used for s in recent_stats) / len(recent_stats)
        peak_gpu_usage = max(s.gpu_memory_used for s in recent_stats)

        if peak_gpu_usage > avg_gpu_usage * 1.5:
            suggestions.append("Consider batch processing to reduce memory spikes")

        if avg_gpu_usage < 1.0:  # Less than 1GB average
            suggestions.append("GPU memory usage is efficient")
        elif avg_gpu_usage > 4.0:  # More than 4GB average
            suggestions.append(
                "Consider model size optimization or distributed computing",
            )

        avg_utilization = sum(s.gpu_utilization for s in recent_stats) / len(
            recent_stats,
        )
        if avg_utilization < 20:
            suggestions.append(
                "GPU utilization is low - consider CPU-only mode for small models",
            )
        elif avg_utilization > 90:
            suggestions.append("GPU is highly utilized - excellent performance")

        return suggestions

    def save_performance_log(self, filepath: str) -> None:
        """Save performance data to file."""
        log_data = {
            "device": str(self.device),
            "memory_history": [
                {
                    "gpu_memory_used": s.gpu_memory_used,
                    "gpu_utilization": s.gpu_utilization,
                    "system_memory_used": s.system_memory_used,
                    "timestamp": s.timestamp,
                }
                for s in self.memory_history
            ],
            "allocation_events": self.allocation_events,
            "performance_report": self.get_performance_report(),
        }

        with open(filepath, "w") as f:
            json.dump(log_data, f, indent=2)


# Context manager for automatic memory management
class ManagedGPUContext:
    """Context manager for automatic GPU memory management."""

    def __init__(self, manager: GPUMemoryManager, profile_name: str) -> None:
        self.manager = manager
        self.profile_name = profile_name
        self.initial_stats = None

    def __enter__(self):
        self.initial_stats = self.manager.get_memory_stats()
        self.manager.optimize_for_profile(self.profile_name)
        return self.manager

    def __exit__(self, exc_type, exc_val, exc_tb):
        final_stats = self.manager.get_memory_stats()
        memory_delta = final_stats.gpu_memory_used - self.initial_stats.gpu_memory_used

        if memory_delta > 0.1:  # Significant memory increase
            self.manager.clear_gpu_cache()


# Example usage functions
def demonstrate_memory_management():
    """Demonstrate GPU memory management capabilities."""
    manager = GPUMemoryManager()

    # Check initial status
    manager.get_memory_stats()

    # Health check
    health = manager.check_memory_health()
    if health["warnings"]:
        pass
    if health["recommendations"]:
        pass

    # Performance report
    manager.get_performance_report()

    return manager


if __name__ == "__main__":
    # Run demonstration
    manager = demonstrate_memory_management()

    # Example of using managed context
    with ManagedGPUContext(manager, "electrode_optimization") as gpu_manager:
        # Simulate some GPU work
        if gpu_manager.device.type != "cpu":
            dummy_tensor = torch.randn(1000, 1000, device=gpu_manager.device)
            gpu_manager.monitor_allocation(
                "dummy_tensor",
                dummy_tensor.nelement() * 4 / 1024**2,
            )
            time.sleep(1)  # Simulate computation
