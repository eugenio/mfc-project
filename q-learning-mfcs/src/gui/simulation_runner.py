"""Thread-safe simulation runner for Streamlit GUI.

CRITICAL ARCHITECTURE COMPONENT:
=================================
This module implements the core data optimization architecture with:
- Phase 1: Memory queue streaming (data_queue, live_data_buffer)
- Phase 2: Incremental updates (change detection, smart refresh flags)
- Phase 3: Parquet Migration (columnar storage optimization)

MODIFICATION WARNING:
Any changes to this module must preserve:
1. Queue-based real-time data streaming patterns
2. Non-blocking data operations (get_nowait, put_nowait)
3. Incremental update mechanisms for GUI performance
4. Thread-safe operation between simulation and GUI threads
5. Change detection flags and caching mechanisms
"""

from __future__ import annotations

import os
import queue
import sys
import threading
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class SimulationRunner:
    """Thread-safe simulation runner for Streamlit.

    Critical methods that control data flow:
    - get_live_data(): Non-blocking queue data retrieval
    - get_buffered_data(): DataFrame generation from buffer
    - has_data_changed(): Change detection for performance
    - should_update_plots/metrics(): Smart refresh decisions
    """

    def __init__(self) -> None:
        self.simulation = None
        self.is_running = False
        self.should_stop = False
        self.results_queue = queue.Queue()  # For status messages
        self.data_queue = queue.Queue(maxsize=100)  # For real-time data streaming
        self.thread = None
        self.current_output_dir = None
        self.live_data_buffer = []  # In-memory buffer for GUI
        self.gui_refresh_interval = 5.0

        # Phase 2: Incremental Updates - Change Detection
        self.last_data_count = 0
        self.last_plot_hash = None
        self.last_metrics_hash = None
        self.plot_dirty_flag = True
        self.metrics_dirty_flag = True

        # Phase 3: Parquet Migration - Columnar Storage
        self.parquet_buffer = []
        self.parquet_batch_size = 100  # Write every 100 data points
        self.parquet_writer = None
        self.parquet_schema = None
        self.enable_parquet = True  # Feature flag for Parquet storage

    def start_simulation(
        self,
        config,
        duration_hours,
        n_cells=None,
        electrode_area_m2=None,
        target_conc=None,
        gui_refresh_interval=5.0,
    ) -> bool:
        """Start simulation in background thread.

        Args:
            config: Q-learning configuration
            duration_hours: Simulation duration in hours
            n_cells: Number of MFC cells
            electrode_area_m2: Electrode area per cell in m² (NOT cm²)
            target_conc: Target substrate concentration in mM
            gui_refresh_interval: GUI refresh interval in seconds (for data sync)

        """
        if self.is_running:
            return False

        self.is_running = True
        self.should_stop = False
        self.gui_refresh_interval = gui_refresh_interval
        self.thread = threading.Thread(
            target=self._run_simulation,
            args=(
                config,
                duration_hours,
                n_cells,
                electrode_area_m2,
                target_conc,
                gui_refresh_interval,
            ),
        )
        self.thread.start()
        return True

    def stop_simulation(self) -> bool:
        """Stop the running simulation."""
        if self.is_running:
            self.should_stop = True

            # Wait for thread to finish (with longer timeout for GPU cleanup)
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=10.0)  # Wait up to 10 seconds for GPU cleanup

                # If thread is still alive after timeout, force cleanup
                if self.thread.is_alive():
                    self._force_cleanup()
                    return False

            # Ensure stopped message is sent after thread completes
            self.results_queue.put(
                ("stopped", "Simulation stopped by user", self.current_output_dir),
            )

            # Final state cleanup
            self.is_running = False
            self.should_stop = False
            self.thread = None  # Clear thread reference

            # Clear data queue and buffer
            while not self.data_queue.empty():
                try:
                    self.data_queue.get_nowait()
                except queue.Empty:
                    break
            self.live_data_buffer.clear()

            return True
        return False

    def _force_cleanup(self) -> None:
        """Force cleanup when thread doesn't respond to stop signal."""
        self._cleanup_resources()
        self.is_running = False
        self.should_stop = False

    def _cleanup_resources(self) -> None:
        """Clean up GPU/CPU resources after simulation stops."""
        try:
            # Clear JAX GPU memory if available
            try:
                import jax

                if hasattr(jax, "clear_backends"):
                    jax.clear_backends()
                if hasattr(jax, "clear_caches"):
                    jax.clear_caches()
            except ImportError:
                pass

            # Clear CUDA cache if using NVIDIA
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except ImportError:
                pass

            # For ROCm/HIP, try to reset GPU state (without requiring sudo)
            try:
                os.environ.pop("HIP_VISIBLE_DEVICES", None)
                os.environ.pop("ROCR_VISIBLE_DEVICES", None)
            except Exception:
                pass

            # Force multiple garbage collections
            import gc

            for _ in range(3):
                gc.collect()

        except Exception:
            pass

    def _run_simulation(
        self,
        config,
        duration_hours,
        n_cells=None,
        electrode_area_m2=None,
        target_conc=None,
        gui_refresh_interval=5.0,
    ) -> None:
        """Run simulation in background."""
        try:
            # Import here to avoid circular imports
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            import json
            from datetime import datetime
            from pathlib import Path

            import pandas as pd
            from mfc_gpu_accelerated import GPUAcceleratedMFC

            # Create output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"../data/simulation_data/gui_simulation_{timestamp}")
            output_dir.mkdir(parents=True, exist_ok=True)
            self.current_output_dir = output_dir

            # Update config with GUI values if provided
            if n_cells is not None:
                config.n_cells = n_cells
            if electrode_area_m2 is not None:
                config.electrode_area_per_cell = electrode_area_m2
            if target_conc is not None:
                config.substrate_target_concentration = target_conc

            # Initialize simulation
            mfc_sim = GPUAcceleratedMFC(config)

            # Simulation parameters
            dt_hours = 0.1
            n_steps = int(duration_hours / dt_hours)

            # Calculate save interval based on GUI refresh rate
            min_save_steps = 30  # Minimum 3 hours of simulation time between saves
            max_save_steps = 100  # Maximum 10 hours of simulation time between saves

            gui_refresh_hours = gui_refresh_interval / 3600.0
            calculated_steps = max(1, int(gui_refresh_hours / dt_hours))
            save_interval_steps = max(
                min_save_steps,
                min(calculated_steps, max_save_steps),
            )

            # Progress tracking
            results = {
                "time_hours": [],
                "reservoir_concentration": [],
                "outlet_concentration": [],
                "total_power": [],
                "biofilm_thicknesses": [],
                "substrate_addition_rate": [],
                "q_action": [],
                "epsilon": [],
                "reward": [],
            }

            # Run simulation with stop check
            for step in range(n_steps):
                if self.should_stop:
                    break

                current_time = step * dt_hours

                # Simulate timestep
                step_results = mfc_sim.simulate_timestep(dt_hours)

                # Check for stop signal after GPU computation
                if self.should_stop:
                    break

                # Store results at GUI-synchronized intervals
                if step % save_interval_steps == 0:
                    results["time_hours"].append(current_time)
                    results["reservoir_concentration"].append(
                        float(mfc_sim.reservoir_concentration),
                    )
                    results["outlet_concentration"].append(
                        float(mfc_sim.outlet_concentration),
                    )
                    results["total_power"].append(step_results["total_power"])
                    results["biofilm_thicknesses"].append(
                        [float(x) for x in mfc_sim.biofilm_thicknesses],
                    )
                    results["substrate_addition_rate"].append(
                        step_results["substrate_addition"],
                    )
                    results["q_action"].append(step_results["action"])
                    results["epsilon"].append(step_results["epsilon"])
                    results["reward"].append(step_results["reward"])

                    # Send latest data point to GUI queue (Phase 1: Shared Memory)
                    latest_data_point = {
                        "time_hours": current_time,
                        "reservoir_concentration": float(
                            mfc_sim.reservoir_concentration,
                        ),
                        "outlet_concentration": float(mfc_sim.outlet_concentration),
                        "total_power": step_results["total_power"],
                        "biofilm_thicknesses": [
                            float(x) for x in mfc_sim.biofilm_thicknesses
                        ],
                        "substrate_addition_rate": step_results["substrate_addition"],
                        "q_action": step_results["action"],
                        "epsilon": step_results["epsilon"],
                        "reward": step_results["reward"],
                    }

                    try:
                        self.data_queue.put_nowait(latest_data_point)
                    except queue.Full:
                        pass  # Skip if queue full

                    # Phase 3: Parquet batch processing
                    if self.enable_parquet:
                        if self.parquet_writer is None and self.parquet_schema is None:
                            self.create_parquet_schema(latest_data_point)
                            self.init_parquet_writer(output_dir)

                        if self.parquet_writer is not None:
                            self.parquet_buffer.append(latest_data_point.copy())
                            self.write_parquet_batch()

                    # Async CSV.gz backup every 100 steps
                    if step % 100 == 0:
                        df = pd.DataFrame(results)
                        data_file = (
                            output_dir / f"gui_simulation_data_{timestamp}.csv.gz"
                        )
                        df.to_csv(data_file, compression="gzip", index=False)

            # Phase 3: Finalize Parquet storage
            if self.enable_parquet:
                self.close_parquet_writer()

            # Save final results (CSV.gz for backward compatibility)
            df = pd.DataFrame(results)
            data_file = output_dir / f"gui_simulation_data_{timestamp}.csv.gz"
            df.to_csv(data_file, compression="gzip", index=False)

            # Calculate metrics
            final_metrics = mfc_sim.calculate_final_metrics(results)

            # Save summary
            results_summary = {
                "simulation_info": {
                    "duration_hours": len(results["time_hours"]) * 0.1,
                    "timestamp": timestamp,
                    "stopped_early": self.should_stop,
                },
                "performance_metrics": final_metrics,
            }

            results_file = output_dir / f"gui_simulation_results_{timestamp}.json"
            with open(results_file, "w") as f:
                json.dump(results_summary, f, indent=2)

            if not self.should_stop:
                self.results_queue.put(("completed", results_summary, output_dir))

        except Exception as e:
            self.results_queue.put(("error", str(e), None))
        finally:
            # Clean up GPU resources from simulation
            try:
                if "mfc_sim" in locals():
                    mfc_sim.cleanup_gpu_resources()
            except Exception:
                pass

            # Clean up general resources
            self._cleanup_resources()
            self.is_running = False

    def get_status(self):
        """Get current simulation status."""
        try:
            while not self.results_queue.empty():
                return self.results_queue.get_nowait()
        except queue.Empty:
            pass
        return None

    def get_live_data(self):
        """Get live simulation data from memory queue (non-blocking)."""
        new_data = []
        try:
            while not self.data_queue.empty():
                data_point = self.data_queue.get_nowait()
                new_data.append(data_point)
                self.live_data_buffer.append(data_point)

                # Keep buffer size manageable (last 1000 points)
                if len(self.live_data_buffer) > 1000:
                    self.live_data_buffer = self.live_data_buffer[-1000:]

        except queue.Empty:
            pass
        return new_data

    def get_buffered_data(self):
        """Get current data buffer as DataFrame."""
        if not self.live_data_buffer:
            return None
        try:
            return pd.DataFrame(self.live_data_buffer)
        except Exception:
            return None

    def has_data_changed(self) -> bool:
        """Phase 2: Check if new data points are available."""
        current_count = len(self.live_data_buffer)
        if current_count != self.last_data_count:
            self.last_data_count = current_count
            self.plot_dirty_flag = True
            self.metrics_dirty_flag = True
            return True
        return False

    def should_update_plots(self, force=False) -> bool:
        """Phase 2: Check if plots need updating."""
        if force or self.plot_dirty_flag:
            self.plot_dirty_flag = False
            return True
        return False

    def should_update_metrics(self, force=False) -> bool:
        """Phase 2: Check if metrics need updating."""
        if force or self.metrics_dirty_flag:
            self.metrics_dirty_flag = False
            return True
        return False

    def get_incremental_update_info(self):
        """Phase 2: Get information about what needs updating."""
        return {
            "has_new_data": self.has_data_changed(),
            "data_count": len(self.live_data_buffer),
            "needs_plot_update": self.plot_dirty_flag,
            "needs_metrics_update": self.metrics_dirty_flag,
        }

    def create_parquet_schema(self, sample_data):
        """Phase 3: Create Parquet schema from sample data."""
        if not self.enable_parquet or not sample_data:
            return None
        try:
            df_sample = pd.DataFrame([sample_data])
            schema_fields = []
            for col, dtype in df_sample.dtypes.items():
                if dtype in ["float64", "float32"]:
                    schema_fields.append(pa.field(col, pa.float32()))
                elif dtype in ["int64", "int32"]:
                    schema_fields.append(pa.field(col, pa.int32()))
                else:
                    schema_fields.append(pa.field(col, pa.string()))
            self.parquet_schema = pa.schema(schema_fields)
            return self.parquet_schema
        except Exception:
            self.enable_parquet = False
            return None

    def init_parquet_writer(self, output_dir) -> bool | None:
        """Phase 3: Initialize Parquet writer."""
        if not self.enable_parquet or not self.parquet_schema:
            return False
        try:
            parquet_path = Path(output_dir) / "simulation_data.parquet"
            self.parquet_writer = pq.ParquetWriter(
                parquet_path,
                schema=self.parquet_schema,
                compression="snappy",
            )
            return True
        except Exception:
            self.enable_parquet = False
            return False

    def write_parquet_batch(self) -> bool | None:
        """Phase 3: Write Parquet batch when buffer reaches threshold."""
        if (
            not self.enable_parquet
            or not self.parquet_writer
            or len(self.parquet_buffer) < self.parquet_batch_size
        ):
            return None

        try:
            df_batch = pd.DataFrame(self.parquet_buffer)
            table = pa.Table.from_pandas(df_batch, schema=self.parquet_schema)
            self.parquet_writer.write_table(table)
            self.parquet_buffer.clear()
            return True
        except Exception:
            self.enable_parquet = False
            return False

    def close_parquet_writer(self) -> bool:
        """Phase 3: Close Parquet writer."""
        if self.parquet_writer:
            try:
                if self.parquet_buffer:
                    df_batch = pd.DataFrame(self.parquet_buffer)
                    table = pa.Table.from_pandas(df_batch, schema=self.parquet_schema)
                    self.parquet_writer.write_table(table)
                    self.parquet_buffer.clear()
                self.parquet_writer.close()
                self.parquet_writer = None
                return True
            except Exception:
                return False
        return True
