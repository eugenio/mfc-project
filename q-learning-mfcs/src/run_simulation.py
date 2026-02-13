#!/usr/bin/env python3
"""Unified MFC Simulation CLI.

This CLI consolidates multiple simulation scripts into a single entry point with
different modes for various simulation scenarios.

Available Modes:
    demo        - Quick 1-hour demonstration (fast, for testing)
    100h        - 100-hour simulation with Q-learning control
    1year       - Extended 1000-hour (approx 42 days) simulation
    gpu         - GPU-accelerated simulation (requires CUDA)
    stack       - 5-cell MFC stack simulation with Q-learning
    comprehensive - Full sensor-integrated simulation with EIS/QCM

Examples:
    python run_simulation.py demo
    python run_simulation.py 100h --cells 5
    python run_simulation.py gpu --duration 1000
    python run_simulation.py comprehensive --output ./results

"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

# Add source paths for imports
sys.path.insert(0, str(Path(__file__).parent))

if TYPE_CHECKING:
    from collections.abc import Callable

# Import path_config for output directories
try:
    from path_config import get_figure_path, get_simulation_data_path
except ImportError:
    # Fallback if path_config not available

    def get_simulation_data_path(filename: str) -> str:
        """Fallback: return path in current directory."""
        return str(Path("simulation_data") / filename)

    def get_figure_path(filename: str) -> str:
        """Fallback: return path in current directory."""
        return str(Path("figures") / filename)


class SimulationConfig:
    """Configuration for unified simulation."""

    def __init__(
        self,
        mode: str = "demo",
        n_cells: int = 5,
        duration_hours: float = 1.0,
        time_step: float = 1.0,
        use_gpu: bool = False,
        enable_sensors: bool = False,
        output_dir: str | None = None,
        verbose: bool = True,
    ) -> None:
        """Initialize simulation configuration.

        Args:
            mode: Simulation mode (demo, 100h, 1year, gpu, stack, comprehensive)
            n_cells: Number of MFC cells in stack
            duration_hours: Simulation duration in hours
            time_step: Time step in seconds
            use_gpu: Enable GPU acceleration
            enable_sensors: Enable sensor integration (EIS/QCM)
            output_dir: Output directory for results
            verbose: Enable verbose output

        """
        self.mode = mode
        self.n_cells = n_cells
        self.duration_hours = duration_hours
        self.time_step = time_step
        self.use_gpu = use_gpu
        self.enable_sensors = enable_sensors
        self.output_dir = output_dir or self._default_output_dir()
        self.verbose = verbose

    def _default_output_dir(self) -> str:
        """Get default output directory based on mode."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return get_simulation_data_path(f"{self.mode}_simulation_{timestamp}")

    @classmethod
    def from_mode(cls, mode: str, **kwargs) -> SimulationConfig:
        """Create configuration from mode name with sensible defaults.

        Args:
            mode: Simulation mode name
            **kwargs: Override parameters

        Returns:
            SimulationConfig instance

        """
        mode_defaults = {
            "demo": {
                "duration_hours": 1.0,
                "time_step": 10.0,  # Faster for demo
                "n_cells": 3,
                "use_gpu": False,
                "enable_sensors": False,
            },
            "100h": {
                "duration_hours": 100.0,
                "time_step": 1.0,
                "n_cells": 5,
                "use_gpu": False,
                "enable_sensors": False,
            },
            "1year": {
                "duration_hours": 1000.0,  # ~42 days for practical purposes
                "time_step": 60.0,  # 1 minute steps for speed
                "n_cells": 5,
                "use_gpu": False,
                "enable_sensors": False,
            },
            "gpu": {
                "duration_hours": 1000.0,
                "time_step": 60.0,
                "n_cells": 5,
                "use_gpu": True,
                "enable_sensors": False,
            },
            "stack": {
                "duration_hours": 1000.0,
                "time_step": 1.0,
                "n_cells": 5,
                "use_gpu": False,
                "enable_sensors": False,
            },
            "comprehensive": {
                "duration_hours": 100.0,
                "time_step": 1.0,
                "n_cells": 5,
                "use_gpu": True,
                "enable_sensors": True,
            },
        }

        defaults = mode_defaults.get(mode, mode_defaults["demo"])
        defaults.update(kwargs)
        return cls(mode=mode, **defaults)


class UnifiedSimulationRunner:
    """Unified runner for all simulation modes."""

    def __init__(self, config: SimulationConfig) -> None:
        """Initialize the simulation runner.

        Args:
            config: Simulation configuration

        """
        self.config = config
        self.start_time: float | None = None
        self.results: dict[str, Any] = {}
        self._interrupted = False

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle interruption signals gracefully."""
        print("\nInterrupted! Saving partial results...")
        self._interrupted = True

    def log(self, message: str) -> None:
        """Log message if verbose mode is enabled."""
        if self.config.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")

    def run(self) -> dict[str, Any]:
        """Run the simulation based on configured mode.

        Returns:
            Dictionary containing simulation results

        """
        self.log(f"Starting {self.config.mode} simulation")
        self.log(f"Configuration: {self.config.n_cells} cells, {self.config.duration_hours}h")

        os.makedirs(self.config.output_dir, exist_ok=True)
        self.start_time = time.time()

        # Dispatch to appropriate simulation method
        mode_handlers: dict[str, Callable[[], dict[str, Any]]] = {
            "demo": self._run_demo,
            "100h": self._run_100h,
            "1year": self._run_extended,
            "gpu": self._run_gpu,
            "stack": self._run_stack,
            "comprehensive": self._run_comprehensive,
        }

        handler = mode_handlers.get(self.config.mode, self._run_demo)

        try:
            self.results = handler()
            self.results["success"] = True
        except Exception as e:
            self.log(f"Simulation failed: {e}")
            self.results["success"] = False
            self.results["error"] = str(e)
            import traceback

            self.results["traceback"] = traceback.format_exc()

        # Add metadata
        self.results["metadata"] = {
            "mode": self.config.mode,
            "n_cells": self.config.n_cells,
            "duration_hours": self.config.duration_hours,
            "execution_time": time.time() - self.start_time,
            "timestamp": datetime.now().isoformat(),
            "interrupted": self._interrupted,
        }

        # Save results
        self._save_results()

        return self.results

    def _run_demo(self) -> dict[str, Any]:
        """Run quick demonstration simulation."""
        self.log("Running demo simulation (quick test mode)")

        # Simple simulation without heavy dependencies
        n_steps = int(self.config.duration_hours * 3600 / self.config.time_step)
        n_steps = min(n_steps, 360)  # Cap at 1 hour with 10s steps

        # Initialize simple state
        power_history = []
        voltage_history = []
        time_history = []

        voltage = 0.5  # Starting voltage
        current = 0.1

        for step in range(n_steps):
            if self._interrupted:
                break

            # Simple dynamics
            voltage += np.random.normal(0.001, 0.01)
            voltage = np.clip(voltage, 0.1, 1.0)
            current += np.random.normal(0, 0.01)
            current = np.clip(current, 0.01, 0.5)
            power = voltage * current * self.config.n_cells

            time_hours = step * self.config.time_step / 3600
            time_history.append(time_hours)
            voltage_history.append(voltage)
            power_history.append(power)

            if step % (n_steps // 10 + 1) == 0:
                self.log(f"Progress: {step / n_steps * 100:.1f}%")

        return {
            "time_series": {
                "time": time_history,
                "voltage": voltage_history,
                "power": power_history,
            },
            "final_power": power_history[-1] if power_history else 0,
            "total_energy": sum(power_history) * self.config.time_step / 3600,
            "average_power": np.mean(power_history) if power_history else 0,
        }

    def _run_100h(self) -> dict[str, Any]:
        """Run 100-hour simulation with Q-learning control."""
        self.log("Running 100-hour Q-learning simulation")

        try:
            from mfc_stack_simulation import MFCStack, MFCStackQLearningController
        except ImportError:
            self.log("MFC stack module not available, using simplified simulation")
            return self._run_simplified_qlearning()

        try:
            from mfc_100h_simulation import LongTermController, LongTermMFCStack
        except ImportError:
            # Fall back to basic stack simulation
            stack = MFCStack()
            controller = MFCStackQLearningController(stack)

            simulation_time = self.config.duration_hours * 3600
            dt = self.config.time_step
            steps = int(simulation_time / dt)

            self.log(f"Running {steps} steps...")

            for step in range(steps):
                if self._interrupted:
                    break

                controller.train_step()

                if step % (steps // 10 + 1) == 0:
                    self.log(f"Progress: {step / steps * 100:.1f}%")

            return {
                "time_series": stack.data_log,
                "final_power": stack.stack_power,
                "total_energy": sum(stack.data_log.get("stack_power", [0]))
                * dt
                / 3600,
                "q_table_size": len(controller.q_table),
            }

        # Use extended long-term simulation
        stack = LongTermMFCStack()
        controller = LongTermController(stack)

        simulation_time = self.config.duration_hours * 3600
        dt = self.config.time_step
        steps = int(simulation_time / dt)

        for step in range(steps):
            if self._interrupted:
                break

            controller.train_step()

            if step % (steps // 10 + 1) == 0:
                self.log(f"Progress: {step / steps * 100:.1f}%")

        return {
            "hourly_data": stack.hourly_data,
            "final_power": stack.stack_power,
            "total_energy": stack.total_energy_produced,
            "maintenance_cycles": stack.maintenance_cycles,
            "q_table_size": len(controller.q_table),
        }

    def _run_extended(self) -> dict[str, Any]:
        """Run extended (1000+ hour) simulation."""
        self.log("Running extended simulation (may take significant time)")
        return self._run_simplified_qlearning()

    def _run_gpu(self) -> dict[str, Any]:
        """Run GPU-accelerated simulation."""
        self.log("Running GPU-accelerated simulation")

        try:
            from mfc_gpu_accelerated import GPUAcceleratedMFC
            from config.qlearning_config import DEFAULT_QLEARNING_CONFIG
        except ImportError:
            self.log("GPU module not available, falling back to CPU simulation")
            return self._run_simplified_qlearning()

        config = DEFAULT_QLEARNING_CONFIG
        config.n_cells = self.config.n_cells

        try:
            mfc_sim = GPUAcceleratedMFC(config)
            self.log(f"GPU initialized: {mfc_sim.gpu_available if hasattr(mfc_sim, 'gpu_available') else 'Unknown'}")
        except Exception as e:
            self.log(f"GPU initialization failed: {e}, falling back to CPU")
            return self._run_simplified_qlearning()

        dt_hours = 0.1
        n_steps = int(self.config.duration_hours / dt_hours)

        results_log = {
            "time_hours": [],
            "power": [],
            "reservoir_concentration": [],
        }

        for step in range(n_steps):
            if self._interrupted:
                break

            result = mfc_sim.simulate_timestep(dt_hours)

            if step % max(1, n_steps // 100) == 0:
                results_log["time_hours"].append(step * dt_hours)
                results_log["power"].append(result.get("total_power", 0))
                results_log["reservoir_concentration"].append(
                    mfc_sim.reservoir_concentration
                )

            if step % (n_steps // 10 + 1) == 0:
                self.log(f"Progress: {step / n_steps * 100:.1f}%")

        mfc_sim.cleanup_gpu_resources()

        return {
            "time_series": results_log,
            "final_power": results_log["power"][-1] if results_log["power"] else 0,
            "total_energy": sum(results_log["power"]) * dt_hours,
            "gpu_accelerated": True,
        }

    def _run_stack(self) -> dict[str, Any]:
        """Run 5-cell MFC stack simulation."""
        self.log("Running MFC stack simulation")

        try:
            from mfc_stack_simulation import MFCStack, MFCStackQLearningController
        except ImportError:
            self.log("MFC stack module not available, using simplified simulation")
            return self._run_simplified_qlearning()

        stack = MFCStack()
        controller = MFCStackQLearningController(stack)

        simulation_time = self.config.duration_hours * 3600
        dt = self.config.time_step
        steps = int(simulation_time / dt)

        for step in range(steps):
            if self._interrupted:
                break

            controller.train_step()

            if step % (steps // 10 + 1) == 0:
                health = stack.check_system_health()
                self.log(
                    f"Progress: {step / steps * 100:.1f}% - "
                    f"Power: {stack.stack_power:.3f}W - "
                    f"Reversed: {health['reversed_cells']}"
                )

        return {
            "time_series": stack.data_log,
            "final_power": stack.stack_power,
            "final_voltage": stack.stack_voltage,
            "total_energy": sum(stack.data_log.get("stack_power", [0])) * dt / 3600,
            "q_table_size": len(controller.q_table),
            "final_cell_states": [
                {
                    "cell_id": i,
                    "power": cell.get_power(),
                    "reversed": cell.is_reversed,
                }
                for i, cell in enumerate(stack.cells)
            ],
        }

    def _run_comprehensive(self) -> dict[str, Any]:
        """Run comprehensive sensor-integrated simulation."""
        self.log("Running comprehensive simulation with sensor integration")

        try:
            from sensing_models.sensor_fusion import FusionMethod
            from sensor_integrated_mfc_model import SensorIntegratedMFCModel
        except ImportError:
            self.log("Sensor modules not available, falling back to basic simulation")
            return self._run_100h()

        try:
            model = SensorIntegratedMFCModel(
                n_cells=self.config.n_cells,
                species="mixed",
                substrate="lactate",
                membrane_type="Nafion-117",
                use_gpu=self.config.use_gpu,
                simulation_hours=self.config.duration_hours,
                enable_eis=True,
                enable_qcm=True,
                sensor_fusion_method=FusionMethod.KALMAN_FILTER,
                recirculation_mode=True,
            )
        except Exception as e:
            self.log(f"Sensor-integrated model failed: {e}, falling back")
            return self._run_100h()

        self.log(f"GPU: {'Enabled' if model.gpu_available else 'Disabled'}")

        dt = self.config.time_step  # hours
        for hour in range(int(self.config.duration_hours / dt)):
            if self._interrupted:
                break

            state = model.step_integrated_dynamics(dt)

            if hour % max(1, int(self.config.duration_hours / 10)) == 0:
                self.log(
                    f"Progress: {hour / self.config.duration_hours * 100:.1f}% - "
                    f"Power: {state.average_power:.3f}W - "
                    f"CE: {state.coulombic_efficiency:.2%}"
                )

        results = model._compile_results()
        results["sensor_integration"] = {
            "eis_enabled": True,
            "qcm_enabled": True,
            "fusion_method": "kalman_filter",
        }

        return results

    def _run_simplified_qlearning(self) -> dict[str, Any]:
        """Run simplified Q-learning simulation (fallback mode)."""
        self.log("Running simplified Q-learning simulation")

        n_cells = self.config.n_cells
        simulation_hours = self.config.duration_hours
        time_step = max(self.config.time_step, 60)  # At least 1 minute
        steps_per_hour = int(3600 / time_step)
        total_steps = int(simulation_hours * steps_per_hour)

        # Initialize state
        cell_states = np.array(
            [[1.0, 0.05, 1e-4, 0.1, 0.25, 1e-7, 0.05, 0.01, -0.01, 1.0, 1.0]
             for _ in range(n_cells)]
        )
        cell_states += np.random.normal(0, 0.05, cell_states.shape)
        cell_states = np.clip(cell_states, 0.001, 10.0)

        # Q-learning
        epsilon = 0.3
        epsilon_decay = 0.9995
        epsilon_min = 0.01
        q_states: dict[tuple, np.ndarray] = {}

        # Logging
        log_interval = max(1, steps_per_hour)
        performance_log = {
            "time_hours": [],
            "stack_power": [],
            "stack_voltage": [],
            "total_energy": [],
        }

        total_energy = 0.0

        for step in range(total_steps):
            if self._interrupted:
                break

            # Q-learning action
            state_key = tuple(int(v * 5) % 5 for v in cell_states[0, :5])
            if np.random.random() < epsilon or state_key not in q_states:
                actions = np.random.uniform([0.1, 0.0, 0.0], [0.9, 1.0, 1.0], (n_cells, 3))
            else:
                actions = np.tile(q_states[state_key], (n_cells, 1))

            # Simple dynamics
            aging = cell_states[:, 9]
            voltages = cell_states[:, 7] - cell_states[:, 8]
            currents = actions[:, 0] * aging
            powers = voltages * currents

            stack_voltage = np.sum(voltages)
            stack_current = np.min(currents)
            stack_power = stack_voltage * stack_current

            # Update states (simplified)
            cell_states[:, 0] += np.random.normal(0, 0.01, n_cells)
            cell_states[:, 7] += np.random.normal(0.001, 0.005, n_cells)
            cell_states[:, 8] += np.random.normal(-0.001, 0.005, n_cells)
            cell_states = np.clip(cell_states, 0.001, 2.0)

            if step % steps_per_hour == 0:
                total_energy += stack_power * 1.0
                epsilon = max(epsilon_min, epsilon * epsilon_decay)

            # Log
            if step % log_interval == 0:
                performance_log["time_hours"].append(step * time_step / 3600)
                performance_log["stack_power"].append(float(stack_power))
                performance_log["stack_voltage"].append(float(stack_voltage))
                performance_log["total_energy"].append(float(total_energy))

            if step % max(1, total_steps // 10) == 0:
                self.log(f"Progress: {step / total_steps * 100:.1f}%")

        return {
            "time_series": performance_log,
            "final_power": performance_log["stack_power"][-1] if performance_log["stack_power"] else 0,
            "total_energy": total_energy,
            "average_power": np.mean(performance_log["stack_power"]) if performance_log["stack_power"] else 0,
            "q_table_size": len(q_states),
        }

    def _save_results(self) -> None:
        """Save simulation results to files."""
        # JSON results
        json_file = os.path.join(self.config.output_dir, "results.json")
        try:
            # Convert numpy arrays for JSON serialization
            json_results = self._prepare_for_json(self.results)
            with open(json_file, "w") as f:
                json.dump(json_results, f, indent=2)
            self.log(f"Results saved to: {json_file}")
        except Exception as e:
            self.log(f"Failed to save JSON results: {e}")

        # Pickle for full data preservation
        pkl_file = os.path.join(self.config.output_dir, "results.pkl")
        try:
            with open(pkl_file, "wb") as f:
                pickle.dump(self.results, f)
        except Exception as e:
            self.log(f"Failed to save pickle results: {e}")

        # Generate plots if matplotlib available
        try:
            self._generate_plots()
        except Exception as e:
            self.log(f"Failed to generate plots: {e}")

    def _prepare_for_json(self, obj: Any) -> Any:
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    def _generate_plots(self) -> None:
        """Generate visualization plots."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        time_series = self.results.get("time_series", {})
        if not time_series:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Power over time
        if "power" in time_series or "stack_power" in time_series:
            ax = axes[0, 0]
            power = time_series.get("power", time_series.get("stack_power", []))
            time_h = time_series.get("time", time_series.get("time_hours", range(len(power))))
            ax.plot(time_h, power, "b-", linewidth=1)
            ax.set_xlabel("Time (hours)")
            ax.set_ylabel("Power (W)")
            ax.set_title("(a) Power Output")
            ax.grid(True)

        # Energy accumulation
        if "total_energy" in time_series:
            ax = axes[0, 1]
            ax.plot(time_series.get("time_hours", []), time_series["total_energy"], "g-", linewidth=2)
            ax.set_xlabel("Time (hours)")
            ax.set_ylabel("Cumulative Energy (Wh)")
            ax.set_title("(b) Energy Production")
            ax.grid(True)

        # Voltage
        if "voltage" in time_series or "stack_voltage" in time_series:
            ax = axes[1, 0]
            voltage = time_series.get("voltage", time_series.get("stack_voltage", []))
            time_h = time_series.get("time", time_series.get("time_hours", range(len(voltage))))
            ax.plot(time_h, voltage, "r-", linewidth=1)
            ax.set_xlabel("Time (hours)")
            ax.set_ylabel("Voltage (V)")
            ax.set_title("(c) Stack Voltage")
            ax.grid(True)

        # Summary text
        ax = axes[1, 1]
        ax.axis("off")
        summary_text = f"""Simulation Summary
------------------------
Mode: {self.config.mode}
Duration: {self.config.duration_hours} hours
Cells: {self.config.n_cells}

Final Power: {self.results.get('final_power', 'N/A'):.4f} W
Total Energy: {self.results.get('total_energy', 'N/A'):.2f} Wh
Average Power: {self.results.get('average_power', 'N/A'):.4f} W
"""
        ax.text(0.1, 0.5, summary_text, fontsize=12, family="monospace",
                verticalalignment="center", transform=ax.transAxes)
        ax.set_title("(d) Summary")

        plt.tight_layout()
        plot_file = os.path.join(self.config.output_dir, f"simulation_{self.config.mode}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        self.log(f"Plot saved to: {plot_file}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Unified MFC Simulation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s demo                           Quick 1-hour demo
  %(prog)s 100h --cells 5                 100-hour simulation with 5 cells
  %(prog)s gpu --duration 500             GPU-accelerated 500-hour simulation
  %(prog)s comprehensive --output ./out   Full sensor simulation

Available modes:
  demo          Quick demonstration (1 hour, fast)
  100h          Standard 100-hour simulation
  1year         Extended 1000-hour simulation
  gpu           GPU-accelerated simulation
  stack         5-cell stack with Q-learning
  comprehensive Full sensor-integrated simulation
""",
    )

    parser.add_argument(
        "mode",
        nargs="?",
        choices=["demo", "100h", "1year", "gpu", "stack", "comprehensive"],
        default=None,
        help="Simulation mode to run",
    )

    parser.add_argument(
        "--cells", "-c",
        type=int,
        default=None,
        help="Number of MFC cells (default: mode-specific)",
    )

    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=None,
        help="Simulation duration in hours (default: mode-specific)",
    )

    parser.add_argument(
        "--timestep", "-t",
        type=float,
        default=None,
        help="Time step in seconds (default: mode-specific)",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for results",
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Force GPU acceleration (if available)",
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )

    parser.add_argument(
        "--list-modes",
        action="store_true",
        help="List available modes with descriptions and exit",
    )

    return parser


def list_modes() -> None:
    """Print available modes with descriptions."""
    modes = {
        "demo": (
            "Quick 1-hour demonstration",
            "Fast mode for testing. Uses simplified dynamics with 10-second steps.",
        ),
        "100h": (
            "100-hour Q-learning simulation",
            "Standard simulation with Q-learning control and cell aging effects.",
        ),
        "1year": (
            "Extended 1000-hour simulation",
            "Long-term simulation (~42 days) with maintenance cycles.",
        ),
        "gpu": (
            "GPU-accelerated simulation",
            "Uses CUDA for parallel computation. Requires compatible GPU.",
        ),
        "stack": (
            "5-cell MFC stack simulation",
            "Full stack model with sensors, actuators, and Q-learning control.",
        ),
        "comprehensive": (
            "Full sensor-integrated simulation",
            "Includes EIS/QCM sensors, Kalman filter fusion, mixed species.",
        ),
    }

    print("\nAvailable Simulation Modes:")
    print("=" * 60)
    for mode, (short_desc, long_desc) in modes.items():
        print(f"\n  {mode:15s} - {short_desc}")
        print(f"                    {long_desc}")
    print()


def main() -> int:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    if args.list_modes:
        list_modes()
        return 0

    if args.mode is None:
        parser.print_help()
        return 1

    # Build configuration
    kwargs = {}
    if args.cells is not None:
        kwargs["n_cells"] = args.cells
    if args.duration is not None:
        kwargs["duration_hours"] = args.duration
    if args.timestep is not None:
        kwargs["time_step"] = args.timestep
    if args.output is not None:
        kwargs["output_dir"] = args.output
    if args.gpu:
        kwargs["use_gpu"] = True
    if args.quiet:
        kwargs["verbose"] = False

    config = SimulationConfig.from_mode(args.mode, **kwargs)

    # Run simulation
    runner = UnifiedSimulationRunner(config)
    results = runner.run()

    # Print summary
    if config.verbose:
        print("\n" + "=" * 60)
        print("SIMULATION COMPLETE")
        print("=" * 60)
        print(f"Mode:           {config.mode}")
        print(f"Duration:       {config.duration_hours} hours")
        print(f"Cells:          {config.n_cells}")
        print(f"Success:        {results.get('success', False)}")
        print(f"Execution time: {results['metadata']['execution_time']:.1f} seconds")
        print(f"Output dir:     {config.output_dir}")

        if "total_energy" in results:
            print(f"Total energy:   {results['total_energy']:.2f} Wh")
        if "final_power" in results:
            print(f"Final power:    {results['final_power']:.4f} W")
        print("=" * 60)

    return 0 if results.get("success", False) else 1


if __name__ == "__main__":
    sys.exit(main())
