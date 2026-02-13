#!/usr/bin/env python3
"""
Streamlit GUI for MFC Simulation Control and Monitoring

CRITICAL ARCHITECTURE WARNING:
=================================
This file implements a 3-phase data optimization architecture:
- Phase 1: Shared Memory Queue (queue.Queue for real-time streaming)
- Phase 2: Incremental Updates (change detection, smart GUI updates)  
- Phase 3: Parquet Migration (planned - columnar storage optimization)

ANY MODIFICATIONS MUST:
1. Preserve the existing queue-based streaming architecture
2. Maintain incremental update mechanisms for performance
3. Respect the non-blocking data flow patterns
4. Follow existing method signatures and behavior
5. Test thoroughly as this controls real-time MFC simulation data

INTEGRATION REQUIREMENTS:
- New methods must integrate with SimulationRunner class structure
- GUI updates must use cached data and incremental refresh patterns
- Data flow: simulation -> queue -> incremental updates -> GUI rendering
- Performance-critical: changes here affect live simulation responsiveness
"""

import gzip
import json
import os
import queue
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Phase 3: Parquet Migration
import pyarrow as pa
import pyarrow.parquet as pq
import streamlit as st
from plotly.subplots import make_subplots

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.qlearning_config import DEFAULT_QLEARNING_CONFIG

# Set page config
st.set_page_config(
    page_title="MFC Simulation Control Panel",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 10px;
    margin: 5px 0;
}
.status-running { color: #00ff00; }
.status-stopped { color: #ff0000; }
.status-completed { color: #0066cc; }
</style>
""", unsafe_allow_html=True)

class SimulationRunner:
    """Thread-safe simulation runner for Streamlit
    
    CRITICAL ARCHITECTURE COMPONENT:
    =================================
    This class implements the core data optimization architecture with:
    - Phase 1: Memory queue streaming (data_queue, live_data_buffer)
    - Phase 2: Incremental updates (change detection, smart refresh flags)
    
    MODIFICATION WARNING:
    Any changes to this class must preserve:
    1. Queue-based real-time data streaming patterns
    2. Non-blocking data operations (get_nowait, put_nowait)
    3. Incremental update mechanisms for GUI performance
    4. Thread-safe operation between simulation and GUI threads
    5. Change detection flags and caching mechanisms
    
    Critical methods that control data flow:
    - get_live_data(): Non-blocking queue data retrieval
    - get_buffered_data(): DataFrame generation from buffer
    - has_data_changed(): Change detection for performance
    - should_update_plots/metrics(): Smart refresh decisions
    """

    def __init__(self):
        self.simulation = None
        self.is_running = False
        self.should_stop = False
        self.results_queue = queue.Queue()  # For status messages
        self.data_queue = queue.Queue(maxsize=100)  # For real-time data streaming
        self.thread = None
        self.current_output_dir = None
        self.live_data_buffer = []  # In-memory buffer for GUI

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

    def start_simulation(self, config, duration_hours, n_cells=None, electrode_area_m2=None, target_conc=None, gui_refresh_interval=5.0):
        """Start simulation in background thread
        
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
            args=(config, duration_hours, n_cells, electrode_area_m2, target_conc, gui_refresh_interval)
        )
        self.thread.start()
        return True

    def stop_simulation(self):
        """Stop the running simulation"""
        if self.is_running:
            self.should_stop = True

            # Wait for thread to finish (with longer timeout for GPU cleanup)
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=10.0)  # Wait up to 10 seconds for GPU cleanup

                # If thread is still alive after timeout, force cleanup
                if self.thread.is_alive():
                    print("Warning: Simulation thread did not stop gracefully")
                    self._force_cleanup()
                    return False

            # Ensure stopped message is sent after thread completes
            self.results_queue.put(('stopped', 'Simulation stopped by user', self.current_output_dir))

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

            print("✅ Simulation stopped successfully")
            return True
        return False

    def _cleanup_resources(self):
        """Clean up GPU/CPU resources after simulation stops"""
        try:
            # Clear JAX GPU memory if available
            try:
                import jax
                if hasattr(jax, 'clear_backends'):
                    jax.clear_backends()
                if hasattr(jax, 'clear_caches'):
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
                # Try to access ROCm environment variables instead of system commands
                import os
                os.environ.pop('HIP_VISIBLE_DEVICES', None)
                os.environ.pop('ROCR_VISIBLE_DEVICES', None)
            except Exception:
                pass

            # Force multiple garbage collections
            import gc
            for _ in range(3):
                gc.collect()

            print("🧹 Resources cleaned up")

        except Exception as e:
            print(f"Warning: Failed to clean up resources: {e}")

    def _run_simulation(self, config, duration_hours, n_cells=None, electrode_area_m2=None, target_conc=None, gui_refresh_interval=5.0):
        """Run simulation in background"""
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
            # For performance, save data at reasonable intervals regardless of GUI refresh
            # Minimum save interval: every 30 simulation steps (3 sim hours)
            # Maximum save interval: every 100 simulation steps (10 sim hours)
            min_save_steps = 30  # Minimum 3 hours of simulation time between saves
            max_save_steps = 100  # Maximum 10 hours of simulation time between saves

            # Calculate based on GUI refresh, but constrain to reasonable bounds
            gui_refresh_hours = gui_refresh_interval / 3600.0  # Convert seconds to hours
            calculated_steps = max(1, int(gui_refresh_hours / dt_hours))
            save_interval_steps = max(min_save_steps, min(calculated_steps, max_save_steps))

            print(f"GUI sync: Saving simulation data every {save_interval_steps} steps ({save_interval_steps * dt_hours:.2f} sim hours) for {gui_refresh_interval}s GUI refresh")

            # Progress tracking
            results = {
                'time_hours': [],
                'reservoir_concentration': [],
                'outlet_concentration': [],
                'total_power': [],
                'biofilm_thicknesses': [],
                'substrate_addition_rate': [],
                'q_action': [],
                'epsilon': [],
                'reward': []
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
                    results['time_hours'].append(current_time)
                    results['reservoir_concentration'].append(float(mfc_sim.reservoir_concentration))
                    results['outlet_concentration'].append(float(mfc_sim.outlet_concentration))
                    results['total_power'].append(step_results['total_power'])
                    results['biofilm_thicknesses'].append([float(x) for x in mfc_sim.biofilm_thicknesses])
                    results['substrate_addition_rate'].append(step_results['substrate_addition'])
                    results['q_action'].append(step_results['action'])
                    results['epsilon'].append(step_results['epsilon'])
                    results['reward'].append(step_results['reward'])

                    # Send latest data point to GUI queue (Phase 1: Shared Memory)
                    latest_data_point = {
                        'time_hours': current_time,
                        'reservoir_concentration': float(mfc_sim.reservoir_concentration),
                        'outlet_concentration': float(mfc_sim.outlet_concentration),
                        'total_power': step_results['total_power'],
                        'biofilm_thicknesses': [float(x) for x in mfc_sim.biofilm_thicknesses],
                        'substrate_addition_rate': step_results['substrate_addition'],
                        'q_action': step_results['action'],
                        'epsilon': step_results['epsilon'],
                        'reward': step_results['reward']
                    }

                    try:
                        self.data_queue.put_nowait(latest_data_point)
                    except queue.Full:
                        pass  # Skip if queue full

                    # Phase 3: Parquet batch processing
                    if self.enable_parquet:
                        # Initialize Parquet writer on first data point
                        if self.parquet_writer is None and self.parquet_schema is None:
                            self.create_parquet_schema(latest_data_point)
                            self.init_parquet_writer(output_dir)

                        # Add to Parquet buffer
                        if self.parquet_writer is not None:
                            self.parquet_buffer.append(latest_data_point.copy())

                            # Write batch when buffer is full
                            self.write_parquet_batch()

                    # Async CSV.gz backup every 100 steps (reduced frequency for legacy support)
                    if step % 100 == 0:
                        df = pd.DataFrame(results)
                        data_file = output_dir / f"gui_simulation_data_{timestamp}.csv.gz"
                        df.to_csv(data_file, compression='gzip', index=False)

            # Phase 3: Finalize Parquet storage
            if self.enable_parquet:
                self.close_parquet_writer()

            # Save final results (CSV.gz for backward compatibility)
            df = pd.DataFrame(results)
            data_file = output_dir / f"gui_simulation_data_{timestamp}.csv.gz"
            df.to_csv(data_file, compression='gzip', index=False)

            # Calculate metrics
            final_metrics = mfc_sim.calculate_final_metrics(results)

            # Save summary
            results_summary = {
                'simulation_info': {
                    'duration_hours': len(results['time_hours']) * 0.1,
                    'timestamp': timestamp,
                    'stopped_early': self.should_stop
                },
                'performance_metrics': final_metrics
            }

            results_file = output_dir / f"gui_simulation_results_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(results_summary, f, indent=2)

            if not self.should_stop:
                self.results_queue.put(('completed', results_summary, output_dir))

        except Exception as e:
            self.results_queue.put(('error', str(e), None))
        finally:
            # Clean up GPU resources from simulation
            try:
                if 'mfc_sim' in locals():
                    mfc_sim.cleanup_gpu_resources()
            except Exception:
                pass

            # Clean up general resources
            self._cleanup_resources()
            self.is_running = False

    def get_status(self):
        """Get current simulation status"""
        try:
            while not self.results_queue.empty():
                return self.results_queue.get_nowait()
        except queue.Empty:
            pass
        return None

    def get_live_data(self):
        """Get live simulation data from memory queue (non-blocking)"""
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
        """Get current data buffer as DataFrame"""
        if not self.live_data_buffer:
            return None
        try:
            return pd.DataFrame(self.live_data_buffer)
        except Exception as e:
            print(f"Error creating DataFrame from buffer: {e}")
            return None

    def has_data_changed(self):
        """Phase 2: Check if new data points are available"""
        current_count = len(self.live_data_buffer)
        if current_count != self.last_data_count:
            self.last_data_count = current_count
            self.plot_dirty_flag = True
            self.metrics_dirty_flag = True
            return True
        return False

    def should_update_plots(self, force=False):
        """Phase 2: Check if plots need updating"""
        if force or self.plot_dirty_flag:
            self.plot_dirty_flag = False
            return True
        return False

    def should_update_metrics(self, force=False):
        """Phase 2: Check if metrics need updating"""
        if force or self.metrics_dirty_flag:
            self.metrics_dirty_flag = False
            return True
        return False

    def get_incremental_update_info(self):
        """Phase 2: Get information about what needs updating"""
        return {
            'has_new_data': self.has_data_changed(),
            'data_count': len(self.live_data_buffer),
            'needs_plot_update': self.plot_dirty_flag,
            'needs_metrics_update': self.metrics_dirty_flag
        }

    def create_parquet_schema(self, sample_data):
        """Phase 3: Create Parquet schema from sample data"""
        if not self.enable_parquet or not sample_data:
            return None
        try:
            df_sample = pd.DataFrame([sample_data])
            schema_fields = []
            for col, dtype in df_sample.dtypes.items():
                if dtype in ['float64', 'float32']:
                    schema_fields.append(pa.field(col, pa.float32()))
                elif dtype in ['int64', 'int32']:
                    schema_fields.append(pa.field(col, pa.int32()))
                else:
                    schema_fields.append(pa.field(col, pa.string()))
            self.parquet_schema = pa.schema(schema_fields)
            return self.parquet_schema
        except Exception as e:
            print(f"Warning: Could not create Parquet schema: {e}")
            self.enable_parquet = False
            return None

    def init_parquet_writer(self, output_dir):
        """Phase 3: Initialize Parquet writer"""
        if not self.enable_parquet or not self.parquet_schema:
            return False
        try:
            parquet_path = Path(output_dir) / "simulation_data.parquet"
            self.parquet_writer = pq.ParquetWriter(
                parquet_path, schema=self.parquet_schema, compression='snappy'
            )
            return True
        except Exception as e:
            print(f"Warning: Could not initialize Parquet writer: {e}")
            self.enable_parquet = False
            return False

    def write_parquet_batch(self):
        """Phase 3: Write Parquet batch when buffer reaches threshold"""
        if not self.enable_parquet or not self.parquet_writer or len(self.parquet_buffer) < self.parquet_batch_size:
            return

        try:
            df_batch = pd.DataFrame(self.parquet_buffer)
            table = pa.Table.from_pandas(df_batch, schema=self.parquet_schema)
            self.parquet_writer.write_table(table)
            self.parquet_buffer.clear()
            return True
        except Exception as e:
            print(f"Warning: Could not write Parquet batch: {e}")
            self.enable_parquet = False
            return False

    def close_parquet_writer(self):
        """Phase 3: Close Parquet writer"""
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
            except Exception as e:
                print(f"Warning: Could not close Parquet writer: {e}")
                return False
        return True

# Initialize session state
if 'sim_runner' not in st.session_state:
    st.session_state.sim_runner = SimulationRunner()
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'last_output_dir' not in st.session_state:
    st.session_state.last_output_dir = None

def load_simulation_data(data_dir):
    """Load simulation data from directory"""
    data_dir = Path(data_dir)

    # Find compressed CSV file
    csv_files = list(data_dir.glob("*_data_*.csv.gz"))
    if not csv_files:
        return None

    csv_file = csv_files[0]

    try:
        with gzip.open(csv_file, 'rt') as f:
            df = pd.read_csv(f)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def load_recent_simulations():
    """Load list of recent simulation directories"""
    data_dir = Path("../data/simulation_data")

    if not data_dir.exists():
        return []

    sim_dirs = []
    for subdir in data_dir.iterdir():
        if subdir.is_dir() and subdir.name.startswith(('gpu_', 'lactate_')):
            # Check if it has results
            json_files = list(subdir.glob("*results*.json"))
            csv_files = list(subdir.glob("*data*.csv.gz"))

            if json_files and csv_files:
                try:
                    with open(json_files[0]) as f:
                        results = json.load(f)

                    sim_dirs.append({
                        'name': subdir.name,
                        'path': str(subdir),
                        'timestamp': results.get('simulation_info', {}).get('timestamp', ''),
                        'duration': results.get('simulation_info', {}).get('duration_hours', 0),
                        'final_conc': results.get('performance_metrics', {}).get('final_reservoir_concentration', 0),
                        'control_effectiveness': results.get('performance_metrics', {}).get('control_effectiveness_2mM', 0)
                    })
                except Exception:
                    continue

    return sorted(sim_dirs, key=lambda x: x['timestamp'], reverse=True)

def create_real_time_plots(df):
    """Create real-time monitoring plots"""

    # Create subplots - 4x3 grid to accommodate all plots
    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=('Substrate Concentration', 'Power Output', 'System Voltage',
                       'Q-Learning Actions', 'Biofilm Growth', 'Flow Control',
                       'Individual Cell Powers', 'Mixing & Control', 'Q-Values & Rewards',
                       'Cumulative Energy', '', ''),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": True}],
               [{"secondary_y": False}, {"secondary_y": True}, {"secondary_y": True}],
               [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )

    # Substrate concentration plot
    fig.add_trace(
        go.Scatter(x=df['time_hours'], y=df['reservoir_concentration'],
                  name='Reservoir', line=dict(color='blue', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['time_hours'], y=df['outlet_concentration'],
                  name='Outlet', line=dict(color='red', width=2)),
        row=1, col=1
    )
    # Target line
    fig.add_hline(y=25.0, line_dash="dash", line_color="green",
                  annotation_text="Target (25 mM)", row=1, col=1)

    # Power output
    fig.add_trace(
        go.Scatter(x=df['time_hours'], y=df['total_power'],
                  name='Power', line=dict(color='orange', width=2)),
        row=1, col=2
    )

    # Q-learning actions
    fig.add_trace(
        go.Scatter(x=df['time_hours'], y=df['q_action'],
                  mode='markers', name='Actions', marker=dict(color='purple', size=4)),
        row=2, col=1
    )

    # Biofilm thickness (average)
    if 'biofilm_thicknesses' in df.columns:
        # Calculate average biofilm thickness safely
        def parse_biofilm_data(x):
            """Safely parse biofilm thickness data from various formats"""
            try:
                if isinstance(x, (list, tuple)):
                    # Already a list/tuple
                    return sum(x) / len(x) if len(x) > 0 else 1.0
                elif isinstance(x, str) and x.strip():
                    # String representation - parse safely
                    x_clean = x.strip('[]() ').replace(' ', '')
                    if ',' in x_clean:
                        values = [float(val.strip()) for val in x_clean.split(',') if val.strip()]
                        return sum(values) / len(values) if len(values) > 0 else 1.0
                    else:
                        # Single value
                        return float(x_clean) if x_clean else 1.0
                elif isinstance(x, (int, float)):
                    # Single numeric value
                    return float(x)
                else:
                    return 1.0  # Default fallback
            except (ValueError, TypeError, ZeroDivisionError):
                return 1.0  # Default fallback

        biofilm_avg = df['biofilm_thicknesses'].apply(parse_biofilm_data)

        fig.add_trace(
            go.Scatter(x=df['time_hours'], y=biofilm_avg,
                      name='Avg Thickness', line=dict(color='brown', width=2)),
            row=2, col=2
        )

    # Plot 5: Flow Rate & Efficiency (dual y-axis)
    if 'flow_rate_ml_h' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['time_hours'], y=df['flow_rate_ml_h'],
                      name='Flow Rate (mL/h)', line=dict(color='cyan', width=2)),
            row=2, col=2
        )
    if 'substrate_efficiency' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['time_hours'], y=df['substrate_efficiency'],
                      name='Efficiency', line=dict(color='green', width=2)),
            row=2, col=2, secondary_y=True
        )

    # Plot 6: System Performance (voltage only, since conc error moved to plot 1)
    if 'system_voltage' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['time_hours'], y=df['system_voltage'],
                      name='Voltage (V)', line=dict(color='blue', width=2)),
            row=2, col=3
        )

    # Plot 7: Individual Cell Powers
    if 'individual_cell_powers' in df.columns:
        # Plot first few cells to avoid clutter
        for i in range(min(3, len(df['individual_cell_powers'].iloc[0]) if len(df) > 0 else 0)):
            cell_powers = [powers[i] if len(powers) > i else 0 for powers in df['individual_cell_powers']]
            fig.add_trace(
                go.Scatter(x=df['time_hours'], y=cell_powers,
                          name=f'Cell {i+1}', line=dict(width=1.5)),
                row=3, col=1
            )

    # Plot 8: Mixing & Control (dual y-axis)
    if 'mixing_efficiency' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['time_hours'], y=df['mixing_efficiency'],
                      name='Mixing Eff', line=dict(color='purple', width=2)),
            row=3, col=2
        )
    if 'biofilm_activity_factor' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['time_hours'], y=df['biofilm_activity_factor'],
                      name='Biofilm Activity', line=dict(color='orange', width=2)),
            row=3, col=2, secondary_y=True
        )

    # Plot 9: Q-Values & Rewards (dual y-axis)
    if 'q_value' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['time_hours'], y=df['q_value'],
                      name='Q-Value', line=dict(color='blue', width=2)),
            row=3, col=3
        )
    if 'reward' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['time_hours'], y=df['reward'],
                      name='Reward', line=dict(color='green', width=2)),
            row=3, col=3, secondary_y=True
        )

    # Plot 10: Cumulative Energy
    if 'total_power' in df.columns and 'time_hours' in df.columns:
        # Calculate cumulative energy from power data
        time_hours = df['time_hours'].values
        power_watts = df['total_power'].values

        # Calculate time differences in hours for integration
        dt_hours = np.diff(time_hours, prepend=0)  # Time step for each point

        # Energy = Power × Time (Wh = W × h)
        energy_increments = power_watts * dt_hours  # Wh per timestep
        cumulative_energy_wh = np.cumsum(energy_increments)  # Cumulative energy in Wh

        # Convert to more appropriate units
        if cumulative_energy_wh[-1] > 1000:
            # Use kWh for large values
            cumulative_energy_display = cumulative_energy_wh / 1000
            energy_unit = 'kWh'
        else:
            # Use Wh for smaller values
            cumulative_energy_display = cumulative_energy_wh
            energy_unit = 'Wh'

        fig.add_trace(
            go.Scatter(x=time_hours, y=cumulative_energy_display,
                      name=f'Cumulative Energy ({energy_unit})',
                      line=dict(color='darkgreen', width=3),
                      fill='tonexty' if len(fig.data) == 0 else 'tozeroy',
                      fillcolor='rgba(0,128,0,0.1)'),
            row=4, col=1
        )

        # Add energy efficiency indicator (energy per unit time)
        if len(time_hours) > 1:
            total_time = time_hours[-1] - time_hours[0]
            if total_time > 0:
                avg_power = np.mean(power_watts)
                energy_efficiency = cumulative_energy_display[-1] / total_time if total_time > 0 else 0

                # Add annotation for total energy and average power
                fig.add_annotation(
                    x=time_hours[-1] * 0.7, y=cumulative_energy_display[-1] * 0.8,
                    text=f"Total: {cumulative_energy_display[-1]:.2f} {energy_unit}<br>"
                         f"Avg Power: {avg_power:.3f} W<br>"
                         f"Rate: {energy_efficiency:.3f} {energy_unit}/h",
                    showarrow=True, arrowhead=2, arrowcolor='darkgreen',
                    bgcolor='rgba(255,255,255,0.8)', bordercolor='darkgreen',
                    row=4, col=1
                )

    # Update layout for expanded 4x3 grid
    fig.update_layout(
        height=1200,  # Increased height for 4 rows
        showlegend=True,
        title_text="Comprehensive MFC Simulation Monitoring Dashboard"
    )

    # Update axes labels for all plots
    # Row 1
    fig.update_yaxes(title_text="Concentration (mM)", row=1, col=1)
    fig.update_yaxes(title_text="Error (mM)", secondary_y=True, row=1, col=1)
    fig.update_yaxes(title_text="Power (W)", row=1, col=2)
    fig.update_yaxes(title_text="Current (A)", secondary_y=True, row=1, col=2)
    fig.update_yaxes(title_text="Action ID", row=1, col=3)

    # Row 2
    fig.update_yaxes(title_text="Thickness (μm)", row=2, col=1)
    fig.update_yaxes(title_text="Flow Rate (mL/h)", row=2, col=2)
    fig.update_yaxes(title_text="Efficiency", secondary_y=True, row=2, col=2)
    fig.update_yaxes(title_text="Voltage (V)", row=2, col=3)

    # Row 3
    fig.update_yaxes(title_text="Power (W)", row=3, col=1)
    fig.update_yaxes(title_text="Mixing Efficiency", row=3, col=2)
    fig.update_yaxes(title_text="Activity Factor", secondary_y=True, row=3, col=2)
    fig.update_yaxes(title_text="Q-Value", row=3, col=3)
    fig.update_yaxes(title_text="Reward", secondary_y=True, row=3, col=3)

    # Row 4 (Cumulative Energy)
    if 'total_power' in df.columns:
        energy_unit = 'kWh' if df['total_power'].sum() * len(df) / 1000 > 1 else 'Wh'
        fig.update_yaxes(title_text=f"Energy ({energy_unit})", row=4, col=1)

    # Add time labels to bottom row
    fig.update_xaxes(title_text="Time (hours)", row=4, col=1)
    fig.update_xaxes(title_text="Time (hours)", row=4, col=2)
    fig.update_xaxes(title_text="Time (hours)", row=4, col=3)

    return fig

def create_biofilm_analysis_plots(df):
    """Create comprehensive biofilm parameter visualization"""

    # Handle both dict and DataFrame inputs
    if isinstance(df, dict):
        columns = df.keys()
        data_dict = df
    else:
        columns = df.columns
        data_dict = df.to_dict('list') if hasattr(df, 'to_dict') else df

    # Check if we have biofilm data
    biofilm_cols = [col for col in columns if 'biofilm' in col.lower() or 'biomass' in col.lower() or 'attachment' in col.lower()]
    if not biofilm_cols:
        return None

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Biofilm Thickness per Cell', 'Biomass Density Distribution', 'Attachment Fraction',
            'Growth vs Detachment Rates', 'Biofilm Conductivity', 'Species Composition'
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": True}, {"secondary_y": False}, {"secondary_y": False}]
        ]
    )

    # Plot 1: Per-cell biofilm thickness
    if 'biofilm_thicknesses' in data_dict or 'biofilm_thickness_per_cell' in data_dict:
        time_data = data_dict.get('time_hours', data_dict.get('time', list(range(len(data_dict.get('biofilm_thicknesses', []))))))

        # Handle different data formats
        if 'biofilm_thicknesses' in data_dict:
            biofilm_data = data_dict['biofilm_thicknesses']
            if biofilm_data and isinstance(biofilm_data[0], list):
                # Data is list of lists (per timepoint, per cell)
                n_cells = len(biofilm_data[0])
                for i in range(min(n_cells, 5)):  # Max 5 cells for readability
                    cell_values = [timepoint[i] if i < len(timepoint) else 0 for timepoint in biofilm_data]
                    fig.add_trace(
                        go.Scatter(x=time_data, y=cell_values,
                                  name=f'Cell {i+1}', line=dict(width=2)),
                        row=1, col=1
                    )
        elif 'biofilm_thickness_per_cell' in data_dict:
            # Handle individual cell columns
            for i in range(5):
                col_name = f'biofilm_thickness_cell_{i}'
                if col_name in data_dict:
                    fig.add_trace(
                        go.Scatter(x=time_data, y=data_dict[col_name],
                                  name=f'Cell {i+1}', line=dict(width=2)),
                        row=1, col=1
                    )

    # Plot 2: Biomass density heatmap
    if 'biomass_density' in data_dict or 'biomass_density_per_cell' in data_dict:
        time_data = data_dict.get('time_hours', data_dict.get('time', list(range(len(data_dict.get('biomass_density', []))))))

        try:
            biomass_data = data_dict.get('biomass_density', data_dict.get('biomass_density_per_cell', []))
            if biomass_data and isinstance(biomass_data[0], list):
                # Transpose for heatmap: rows = cells, cols = time
                transposed_data = list(map(list, zip(*biomass_data)))

                fig.add_trace(
                    go.Heatmap(
                        z=transposed_data,
                        x=time_data,
                        y=[f'Cell {i+1}' for i in range(len(transposed_data))],
                        colorscale='Viridis',
                        name='Biomass Density'
                    ),
                    row=1, col=2
                )
        except Exception:
            # Add placeholder if data processing fails
            fig.add_trace(
                go.Scatter(x=[0], y=[0], mode='markers',
                          name='No biomass data', marker=dict(size=0)),
                row=1, col=2
            )

    # Plot 3: Attachment fraction (or use average biofilm thickness if attachment not available)
    if 'attachment_fraction' in data_dict:
        time_data = data_dict.get('time_hours', data_dict.get('time', list(range(len(data_dict['attachment_fraction'])))))
        avg_attachment = data_dict['attachment_fraction']
    elif 'biofilm_thicknesses' in data_dict:
        time_data = data_dict.get('time_hours', data_dict.get('time', list(range(len(data_dict['biofilm_thicknesses'])))))
        # Calculate average biofilm thickness as proxy for attachment
        biofilm_data = data_dict['biofilm_thicknesses']
        if biofilm_data and isinstance(biofilm_data[0], list):
            avg_attachment = [sum(timepoint)/len(timepoint) if timepoint else 0 for timepoint in biofilm_data]
        else:
            avg_attachment = biofilm_data if isinstance(biofilm_data, list) else [0]
    else:
        time_data = [0]
        avg_attachment = [0]

    # Add the attachment trace
    fig.add_trace(
        go.Scatter(x=time_data, y=avg_attachment,
                  name='Avg Attachment/Thickness',
                  line=dict(color='green', width=2)),
        row=1, col=3
    )

    # Add remaining plots with available data
    fig.update_layout(
        title="Biofilm Analysis - Per Cell Parameters",
        showlegend=True,
        height=600
    )
    fig.update_xaxes(title_text="Time (hours)")
    fig.update_yaxes(title_text="Thickness (μm)", row=1, col=1)
    fig.update_yaxes(title_text="Density (g/L)", row=1, col=2)
    fig.update_yaxes(title_text="Attachment", row=1, col=3)

    return fig

def create_metabolic_analysis_plots(df):
    """Create metabolic pathway visualization"""

    # Handle both dict and DataFrame inputs
    if isinstance(df, dict):
        columns = df.keys()
        data_dict = df
    else:
        columns = df.columns
        data_dict = df.to_dict('list') if hasattr(df, 'to_dict') else df

    metabolic_cols = [col for col in columns if any(term in col.lower() for term in ['nadh', 'atp', 'electron', 'metabolic'])]
    if not metabolic_cols:
        return None

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'NADH/NAD+ Ratios', 'ATP Levels', 'Electron Flux',
            'Substrate Uptake Rates', 'Metabolic Activity', 'Oxygen Crossover'
        ),
        specs=[[{"secondary_y": False} for _ in range(3)] for _ in range(2)]
    )

    time_data = data_dict.get('time_hours', data_dict.get('time', [0]))

    # Plot NADH ratios if available
    if 'nadh_ratios' in data_dict or 'nadh_ratio' in data_dict:
        nadh_data = data_dict.get('nadh_ratios', data_dict.get('nadh_ratio', []))
        if nadh_data and isinstance(nadh_data[0], list):
            for i in range(min(len(nadh_data[0]), 5)):
                cell_values = [timepoint[i] if i < len(timepoint) else 0.3 for timepoint in nadh_data]
                fig.add_trace(
                    go.Scatter(x=time_data, y=cell_values,
                              name=f'Cell {i+1} NADH', line=dict(width=2)),
                    row=1, col=1
                )
        else:
            # Single value or average
            fig.add_trace(
                go.Scatter(x=time_data, y=nadh_data if isinstance(nadh_data, list) else [0.3] * len(time_data),
                          name='Avg NADH Ratio', line=dict(width=2)),
                row=1, col=1
            )

    # Plot ATP levels
    if 'atp_levels' in data_dict or 'atp_level' in data_dict:
        atp_data = data_dict.get('atp_levels', data_dict.get('atp_level', []))
        if atp_data and isinstance(atp_data[0], list):
            # Average across cells
            avg_atp = [sum(timepoint)/len(timepoint) if timepoint else 2.0 for timepoint in atp_data]
        else:
            avg_atp = atp_data if isinstance(atp_data, list) else [2.0] * len(time_data)

        fig.add_trace(
            go.Scatter(x=time_data, y=avg_atp,
                      name='Avg ATP', line=dict(color='red', width=2)),
            row=1, col=2
        )

    # Plot electron flux
    if 'electron_flux' in data_dict:
        electron_data = data_dict['electron_flux']
        if electron_data and isinstance(electron_data[0], list):
            # Average across cells
            avg_flux = [sum(timepoint)/len(timepoint) if timepoint else 0.1 for timepoint in electron_data]
        else:
            avg_flux = electron_data if isinstance(electron_data, list) else [0.1] * len(time_data)

        fig.add_trace(
            go.Scatter(x=time_data, y=avg_flux,
                      name='Avg e- Flux', line=dict(color='blue', width=2)),
            row=1, col=3
        )
    elif 'total_current' in data_dict:
        fig.add_trace(
            go.Scatter(x=time_data, y=data_dict['total_current'],
                      name='Total Current (A)', line=dict(color='blue', width=2)),
            row=1, col=3
        )

    fig.update_layout(
        title="Metabolic Analysis - Per Cell Parameters",
        showlegend=True,
        height=600
    )
    fig.update_xaxes(title_text="Time (hours)")
    fig.update_yaxes(title_text="NADH Ratio", row=1, col=1)
    fig.update_yaxes(title_text="ATP (mM)", row=1, col=2)
    fig.update_yaxes(title_text="Electron Flux", row=1, col=3)

    return fig

def create_sensing_analysis_plots(df):
    """Create EIS and QCM sensing visualization"""

    # Handle both dict and DataFrame inputs
    if isinstance(df, dict):
        columns = df.keys()
        data_dict = df
    else:
        columns = df.columns
        data_dict = df.to_dict('list') if hasattr(df, 'to_dict') else df

    sensing_cols = [col for col in columns if any(term in col.lower() for term in ['eis', 'qcm', 'impedance', 'frequency'])]
    if not sensing_cols:
        return None

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'EIS Impedance Magnitude', 'EIS Phase Response', 'QCM Frequency Shift',
            'Charge Transfer Resistance', 'QCM Mass Loading', 'Sensor Calibration'
        ),
        specs=[[{"secondary_y": False} for _ in range(3)] for _ in range(2)]
    )

    # Get time data
    time_data = data_dict.get('time_hours', data_dict.get('time', [0]))
    if not time_data:
        time_data = [0]

    # EIS Impedance magnitude
    if 'eis_impedance_magnitude' in data_dict:
        impedance_data = data_dict.get('eis_impedance_magnitude', [1000] * len(time_data))
        fig.add_trace(
            go.Scatter(x=time_data, y=impedance_data if impedance_data else [1000] * len(time_data),
                      name='|Z| @ 1kHz', line=dict(color='purple', width=2)),
            row=1, col=1
        )

    # EIS Phase
    if 'eis_impedance_phase' in data_dict:
        phase_data = data_dict.get('eis_impedance_phase', [-45] * len(time_data))
        fig.add_trace(
            go.Scatter(x=time_data, y=phase_data if phase_data else [-45] * len(time_data),
                      name='Phase @ 1kHz', line=dict(color='orange', width=2)),
            row=1, col=2
        )

    # QCM frequency shift
    if 'qcm_frequency_shift' in data_dict:
        freq_data = data_dict.get('qcm_frequency_shift', [-500] * len(time_data))
        fig.add_trace(
            go.Scatter(x=time_data, y=freq_data if freq_data else [-500] * len(time_data),
                      name='Δf (Hz)', line=dict(color='green', width=2)),
            row=1, col=3
        )

    # Charge transfer resistance (additional subplot)
    if 'charge_transfer_resistance' in data_dict:
        rct_data = data_dict.get('charge_transfer_resistance', [100] * len(time_data))
        fig.add_trace(
            go.Scatter(x=time_data, y=rct_data if rct_data else [100] * len(time_data),
                      name='Rct (Ω)', line=dict(color='red', width=2)),
            row=2, col=1
        )

    # QCM mass loading
    if 'qcm_mass_loading' in data_dict:
        mass_data = data_dict.get('qcm_mass_loading', [0.1] * len(time_data))
        fig.add_trace(
            go.Scatter(x=time_data, y=mass_data if mass_data else [0.1] * len(time_data),
                      name='Mass Loading (μg/cm²)', line=dict(color='brown', width=2)),
            row=2, col=2
        )

    fig.update_layout(
        title="Sensing Analysis",
        height=600,
        showlegend=True
    )

    return fig

def create_spatial_distribution_plots(df, n_cells=5):
    """Create spatial distribution visualization for per-cell parameters"""

    # Handle both dict and DataFrame inputs
    if isinstance(df, dict):
        columns = df.keys()
        data_dict = df
    else:
        columns = df.columns
        data_dict = df.to_dict('list') if hasattr(df, 'to_dict') else df

    # Check for per-cell data
    cell_data_cols = [col for col in columns if 'per_cell' in col or 'cell_' in col or 'voltages' in col or 'densities' in col]
    if not cell_data_cols:
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Cell Voltages Distribution', 'Current Density Distribution',
            'Temperature Distribution', 'Biofilm Thickness Distribution'
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )

    # Get latest data point for spatial visualization
    def get_latest_value(data_list):
        if isinstance(data_list, list) and data_list:
            if isinstance(data_list[-1], list):
                return data_list[-1]  # Per-cell data at latest timepoint
            else:
                return data_list[-1]  # Single value at latest timepoint
        return None

    # Cell voltages
    if 'cell_voltages' in data_dict:
        cell_voltages_data = get_latest_value(data_dict['cell_voltages'])
        if cell_voltages_data:
            if isinstance(cell_voltages_data, list):
                cell_voltages = cell_voltages_data[:n_cells] if len(cell_voltages_data) >= n_cells else cell_voltages_data + [0.7] * (n_cells - len(cell_voltages_data))
            else:
                cell_voltages = [cell_voltages_data] * n_cells
        else:
            cell_voltages = [0.7] * n_cells

        fig.add_trace(
            go.Bar(x=[f'Cell {i+1}' for i in range(n_cells)], y=cell_voltages,
                   name='Cell Voltage (V)', marker_color='blue'),
            row=1, col=1
        )

    # Current density
    if 'current_densities' in data_dict or 'current_density_per_cell' in data_dict:
        current_key = 'current_densities' if 'current_densities' in data_dict else 'current_density_per_cell'
        current_densities_data = get_latest_value(data_dict[current_key])
        if current_densities_data:
            if isinstance(current_densities_data, list):
                current_densities = current_densities_data[:n_cells] if len(current_densities_data) >= n_cells else current_densities_data + [1.0] * (n_cells - len(current_densities_data))
            else:
                current_densities = [current_densities_data] * n_cells
        else:
            current_densities = [1.0] * n_cells

        fig.add_trace(
            go.Bar(x=[f'Cell {i+1}' for i in range(n_cells)], y=current_densities,
                   name='Current Density (A/m²)', marker_color='red'),
            row=1, col=2
        )

    # Temperature distribution
    if 'temperature_per_cell' in data_dict:
        temp_data = get_latest_value(data_dict['temperature_per_cell'])
        if temp_data:
            if isinstance(temp_data, list):
                temperatures = temp_data[:n_cells] if len(temp_data) >= n_cells else temp_data + [25.0] * (n_cells - len(temp_data))
            else:
                temperatures = [temp_data] * n_cells
        else:
            temperatures = [25.0] * n_cells

        fig.add_trace(
            go.Scatter(x=[f'Cell {i+1}' for i in range(n_cells)], y=temperatures,
                      mode='markers+lines', name='Temperature (°C)',
                      line=dict(color='orange', width=2), marker=dict(size=8)),
            row=2, col=1
        )

    # Biofilm thickness distribution
    if 'biofilm_thicknesses' in data_dict or 'biofilm_thickness_per_cell' in data_dict:
        biofilm_key = 'biofilm_thicknesses' if 'biofilm_thicknesses' in data_dict else 'biofilm_thickness_per_cell'
        biofilm_data = get_latest_value(data_dict[biofilm_key])
        if biofilm_data:
            if isinstance(biofilm_data, list):
                biofilm_thickness = biofilm_data[:n_cells] if len(biofilm_data) >= n_cells else biofilm_data + [10.0] * (n_cells - len(biofilm_data))
            else:
                biofilm_thickness = [biofilm_data] * n_cells
        else:
            biofilm_thickness = [10.0] * n_cells

        fig.add_trace(
            go.Scatter(x=[f'Cell {i+1}' for i in range(n_cells)], y=biofilm_thickness,
                      mode='markers+lines', name='Biofilm Thickness (μm)',
                      line=dict(color='green', width=2), marker=dict(size=8)),
            row=2, col=2
        )

    fig.update_layout(
        title="Spatial Distribution Analysis",
        height=600,
        showlegend=True
    )

    return fig

def create_performance_analysis_plots(df):
    """Create comprehensive performance analysis visualization"""

    # Handle both dict and DataFrame inputs
    if isinstance(df, dict):
        data_dict = df
    else:
        data_dict = df.to_dict('list') if hasattr(df, 'to_dict') else df

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Energy Efficiency Over Time', 'Coulombic Efficiency', 'Power Density',
            'Cumulative Energy Production', 'Control Performance', 'Economic Metrics'
        ),
        specs=[[{"secondary_y": True} for _ in range(3)] for _ in range(2)]
    )

    # Get time data
    time_data = data_dict.get('time_hours', data_dict.get('time', [0]))
    if not time_data:
        time_data = [0]

    # Energy efficiency
    if 'energy_efficiency' in data_dict:
        energy_eff_data = data_dict.get('energy_efficiency', [75] * len(time_data))
        fig.add_trace(
            go.Scatter(x=time_data, y=energy_eff_data if energy_eff_data else [75] * len(time_data),
                      name='Energy Efficiency (%)', line=dict(color='green', width=2)),
            row=1, col=1
        )

    # Coulombic efficiency
    if 'coulombic_efficiency' in data_dict or 'coulombic_efficiency_per_cell' in data_dict:
        ce_key = 'coulombic_efficiency' if 'coulombic_efficiency' in data_dict else 'coulombic_efficiency_per_cell'
        ce_data = data_dict.get(ce_key, [85] * len(time_data))
        # If per-cell data, take average
        if isinstance(ce_data, list) and ce_data and isinstance(ce_data[0], list):
            ce_data = [sum(timepoint)/len(timepoint) if timepoint else 85 for timepoint in ce_data]
        fig.add_trace(
            go.Scatter(x=time_data, y=ce_data if ce_data else [85] * len(time_data),
                      name='Coulombic Efficiency (%)', line=dict(color='blue', width=2)),
            row=1, col=2
        )

    # Power density
    if 'total_power' in data_dict or 'power_density_per_cell' in data_dict:
        power_key = 'total_power' if 'total_power' in data_dict else 'power_density_per_cell'
        power_data = data_dict.get(power_key, [2.0] * len(time_data))
        # If per-cell data, sum or average as appropriate
        if isinstance(power_data, list) and power_data and isinstance(power_data[0], list):
            if power_key == 'total_power':
                power_data = [sum(timepoint) if timepoint else 2.0 for timepoint in power_data]
            else:
                power_data = [sum(timepoint)/len(timepoint) if timepoint else 2.0 for timepoint in power_data]
        fig.add_trace(
            go.Scatter(x=time_data, y=power_data if power_data else [2.0] * len(time_data),
                      name='Power Density (W/m²)', line=dict(color='red', width=2)),
            row=1, col=3
        )

    # Cumulative energy
    if 'total_energy_produced' in data_dict or 'energy_produced' in data_dict:
        energy_key = 'total_energy_produced' if 'total_energy_produced' in data_dict else 'energy_produced'
        energy_data = data_dict.get(energy_key, list(range(len(time_data))))
        if energy_data:
            # Calculate cumulative sum
            cumulative_energy = []
            cumsum = 0
            for val in energy_data:
                if isinstance(val, (int, float)):
                    cumsum += val
                elif isinstance(val, list) and val:
                    cumsum += sum(val)
                cumulative_energy.append(cumsum)
        else:
            cumulative_energy = list(range(len(time_data)))

        fig.add_trace(
            go.Scatter(x=time_data, y=cumulative_energy,
                      name='Cumulative Energy (Wh)', line=dict(color='purple', width=2)),
            row=2, col=1
        )

    # Control performance (error from setpoint)
    if 'control_error' in data_dict:
        control_error = data_dict.get('control_error', [0] * len(time_data))
    elif 'outlet_concentration' in data_dict:
        outlet_conc = data_dict.get('outlet_concentration', [25] * len(time_data))
        control_error = [abs(25 - x) if isinstance(x, (int, float)) else 0 for x in outlet_conc]
    else:
        control_error = [0] * len(time_data)

    fig.add_trace(
        go.Scatter(x=time_data, y=control_error,
                  name='Control Error (mM)', line=dict(color='orange', width=2)),
        row=2, col=2
    )

    # Economic metrics (placeholder)
    if 'operating_cost' in data_dict or 'revenue' in data_dict:
        cost_data = data_dict.get('operating_cost', [1.0] * len(time_data))
        revenue_data = data_dict.get('revenue', [2.0] * len(time_data))
        profit = [(r - c) if isinstance(r, (int, float)) and isinstance(c, (int, float)) else 1.0
                 for r, c in zip(revenue_data, cost_data)]
        fig.add_trace(
            go.Scatter(x=time_data, y=profit,
                      name='Profit ($/h)', line=dict(color='gold', width=2)),
            row=2, col=3
        )

    fig.update_layout(
        title="Performance Analysis",
        height=600,
        showlegend=True
    )

    return fig

def create_parameter_correlation_matrix(df):
    """Create correlation matrix for key parameters"""

    # Handle both dict and DataFrame inputs
    if isinstance(df, dict):
        columns = df.keys()
        data_dict = df
    else:
        columns = df.columns
        data_dict = df.to_dict('list') if hasattr(df, 'to_dict') else df

    # Select numeric parameters for correlation analysis
    key_params = []
    numeric_data = {}

    for col in columns:
        if col.startswith('time') or col.startswith('step'):
            continue

        data = data_dict.get(col, [])
        if not data:
            continue

        # Handle different data types
        if isinstance(data, list):
            # For per-cell data, take mean at each timepoint
            if data and isinstance(data[0], list):
                try:
                    numeric_values = [sum(timepoint)/len(timepoint) if timepoint else 0 for timepoint in data]
                except (TypeError, ZeroDivisionError):
                    continue
            else:
                # Check if it's numeric data
                try:
                    numeric_values = [float(x) if isinstance(x, (int, float)) else 0 for x in data]
                except (ValueError, TypeError):
                    continue
        else:
            continue

        # Only include if we have valid numeric data
        if numeric_values and len(numeric_values) > 1:
            key_params.append(col)
            numeric_data[col] = numeric_values

    if len(key_params) < 2:
        return None

    # Ensure all arrays have the same length
    min_length = min(len(numeric_data[param]) for param in key_params)
    for param in key_params:
        numeric_data[param] = numeric_data[param][:min_length]

    # Calculate correlation matrix
    import pandas as pd
    temp_df = pd.DataFrame(numeric_data)

    try:
        corr_matrix = temp_df.corr()
    except Exception:
        # Fallback: return None if correlation calculation fails
        return None

    # Handle case where correlation matrix might have NaN values
    corr_matrix = corr_matrix.fillna(0)

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
    ))

    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="Real-Time MFC Simulation Monitoring"
    )

    # Update axes labels
    fig.update_xaxes(title_text="Time (hours)", row=2, col=1)
    fig.update_xaxes(title_text="Time (hours)", row=2, col=2)
    fig.update_yaxes(title_text="Concentration (mM)", row=1, col=1)
    fig.update_yaxes(title_text="Power (W)", row=1, col=2)
    fig.update_yaxes(title_text="Action ID", row=2, col=1)
    fig.update_yaxes(title_text="Thickness (μm)", row=2, col=2)

    return fig

def create_performance_dashboard(results):
    """Create performance metrics dashboard"""

    metrics = results.get('performance_metrics', {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Final Concentration",
            f"{metrics.get('final_reservoir_concentration', 0):.2f} mM",
            delta=f"{metrics.get('final_reservoir_concentration', 0) - 25:.2f}"
        )

    with col2:
        st.metric(
            "Control Effectiveness (±2mM)",
            f"{metrics.get('control_effectiveness_2mM', 0):.1f}%"
        )

    with col3:
        st.metric(
            "Mean Power",
            f"{metrics.get('mean_power', 0):.3f} W"
        )

    with col4:
        st.metric(
            "Substrate Consumed",
            f"{metrics.get('total_substrate_added', 0):.1f} mmol"
        )

def main():
    """Main Streamlit application"""

    # Title and header
    st.title("🔋 MFC Simulation Control Panel")
    st.markdown("Real-time monitoring and control for Microbial Fuel Cell simulations")

    # Sidebar controls
    st.sidebar.header("🔧 Simulation Parameters")

    # Simulation duration
    duration_options = {
        "1 Hour (Quick Test)": 1,
        "24 Hours (Daily)": 24,
        "1 Week": 168,
        "1 Month": 720,
        "1 Year": 8784
    }

    selected_duration = st.sidebar.selectbox(
        "Simulation Duration",
        options=list(duration_options.keys()),
        index=1
    )
    duration_hours = duration_options[selected_duration]

    # Q-learning parameters
    st.sidebar.subheader("Q-Learning Parameters")

    use_pretrained = st.sidebar.checkbox("Use Pre-trained Q-table", value=True)

    if not use_pretrained:
        st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1)
        st.sidebar.slider("Initial Epsilon", 0.1, 1.0, 0.4)
        st.sidebar.slider("Discount Factor", 0.8, 0.99, 0.95)

    # Target concentrations
    st.sidebar.subheader("Target Concentrations")
    target_conc = st.sidebar.number_input(
        "Target Substrate (mM)",
        min_value=10.0, max_value=40.0, value=25.0, step=0.1
    )

    # MFC cell configuration
    st.sidebar.subheader("MFC Configuration")
    n_cells = st.sidebar.number_input(
        "Number of Cells",
        min_value=1, max_value=10, value=5, step=1
    )

    # Separate anode and cathode electrode areas
    st.sidebar.markdown("**🔋 Working Electrodes**")

    anode_area_cm2 = st.sidebar.number_input(
        "Anode Area (cm²/cell)",
        min_value=0.1, value=10.0, step=0.1,
        help="Current-collecting anode area per cell - arbitrary size"
    )
    cathode_area_cm2 = st.sidebar.number_input(
        "Cathode Area (cm²/cell)",
        min_value=0.1, value=10.0, step=0.1,
        help="Cathode area per cell (can differ from anode) - arbitrary size"
    )

    # Convert to m² for internal use (all simulations use m²)
    anode_area_m2 = anode_area_cm2 * 1e-4
    # cathode_area_m2 = cathode_area_cm2 * 1e-4  # For future cathode-specific calculations

    # Show sensor areas (fixed for optimal sensing)
    st.sidebar.markdown("**📊 Sensor Electrodes (Fixed)**")
    st.sidebar.text("EIS sensor: 1.0 cm² (impedance sensing)")
    st.sidebar.text("QCM sensor: 0.196 cm² (mass sensing)")

    # Legacy compatibility
    electrode_area_m2 = anode_area_m2  # For backward compatibility with simulation code

    # Advanced settings
    with st.sidebar.expander("Advanced Settings"):
        gpu_backend = st.selectbox("GPU Backend", ["Auto-detect", "CUDA", "ROCm", "CPU"])
        st.slider("Save Interval (steps)", 1, 100, 10, help="Data saving is now synchronized with GUI refresh rate")
        st.checkbox("Email Notifications", value=False, help="Feature not yet implemented")

    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Run Simulation", "📊 Monitor", "📈 Results", "📁 History"])

    with tab1:
        st.header("Simulation Control")

        # Status display
        status = st.session_state.sim_runner.get_status()
        if status:
            if status[0] == 'completed':
                st.success("✅ Simulation completed successfully!")
                st.session_state.simulation_results = status[1]
                st.session_state.last_output_dir = status[2]
            elif status[0] == 'stopped':
                st.warning(f"⏹️ {status[1]}")
                st.session_state.last_output_dir = status[2]
            elif status[0] == 'error':
                st.error(f"❌ Simulation failed: {status[1]}")

        # Control buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("▶️ Start Simulation", disabled=st.session_state.sim_runner.is_running):
                # Get current refresh interval from sidebar
                current_refresh_interval = st.session_state.get('current_refresh_interval', 5.0)

                if st.session_state.sim_runner.start_simulation(
                    DEFAULT_QLEARNING_CONFIG,
                    duration_hours,
                    n_cells=n_cells,
                    electrode_area_m2=electrode_area_m2,
                    target_conc=target_conc,
                    gui_refresh_interval=current_refresh_interval
                ):
                    st.success(f"Started {selected_duration} simulation!")
                    st.info(f"📊 Data saving synchronized with {current_refresh_interval}s refresh rate")
                    st.rerun()
                else:
                    st.error("Simulation already running!")

        with col2:
            if st.button("⏹️ Stop Simulation", disabled=not st.session_state.sim_runner.is_running):
                if st.session_state.sim_runner.stop_simulation():
                    st.success("Stopping simulation...")
                    st.rerun()
                else:
                    st.error("No simulation is running")

        with col3:
            if st.button("🔄 Refresh Status"):
                st.rerun()

        # Simulation status
        if st.session_state.sim_runner.is_running:
            st.markdown('<p class="status-running">🟢 Simulation Running...</p>', unsafe_allow_html=True)
            st.info("💡 Switch to the Monitor tab to see real-time updates")
        else:
            st.markdown('<p class="status-stopped">🔴 Simulation Stopped</p>', unsafe_allow_html=True)

        # Configuration preview
        st.subheader("Current Configuration")
        current_refresh = st.session_state.get('current_refresh_interval', 5.0)
        # Calculate data save frequency (matching simulation logic)
        gui_refresh_hours = current_refresh / 3600.0
        min_save_steps = 30  # Minimum 3 hours of simulation time between saves
        max_save_steps = 100  # Maximum 10 hours of simulation time between saves
        calculated_steps = max(1, int(gui_refresh_hours / 0.1))  # 0.1h timestep
        actual_save_steps = max(min_save_steps, min(calculated_steps, max_save_steps))
        save_frequency_hours = actual_save_steps * 0.1

        config_data = {
            "Duration": f"{duration_hours:,} hours ({duration_hours/24:.1f} days)",
            "Target Concentration": f"{target_conc} mM",
            "Number of Cells": n_cells,
            "Anode Area": f"{anode_area_cm2:.1f} cm²/cell ({anode_area_cm2 * n_cells:.1f} cm² total)",
            "Cathode Area": f"{cathode_area_cm2:.1f} cm²/cell ({cathode_area_cm2 * n_cells:.1f} cm² total)",
            "Sensor Areas": "EIS: 1.0 cm², QCM: 0.196 cm² (fixed)",
            "Pre-trained Q-table": "✅ Enabled" if use_pretrained else "❌ Disabled",
            "GPU Backend": gpu_backend,
            "Data Save Sync": f"Every {save_frequency_hours:.2f} sim hours (GUI: {current_refresh}s)"
        }

        for key, value in config_data.items():
            st.text(f"{key}: {value}")

    with tab2:
        st.header("Real-Time Monitoring")

        # Auto-refresh controls
        col1, col2, col3 = st.columns([2, 1, 3])

        with col1:
            auto_refresh = st.checkbox("🔄 Enable Auto-refresh", value=True)

        with col2:
            refresh_interval = st.number_input(
                "Interval (s)",
                min_value=1,
                max_value=60,
                value=5,
                step=1,
                disabled=not auto_refresh,
                key="refresh_interval_input"
            )
            # Store refresh interval in session state for simulation sync
            st.session_state.current_refresh_interval = refresh_interval

        with col3:
            if auto_refresh:
                st.info(f"🔄 Auto-refreshing every {refresh_interval} seconds")
                if st.session_state.sim_runner.is_running:
                    st.success("📊 Data sync enabled with simulation")

        # Implement auto-refresh (non-blocking)
        if auto_refresh:
            # Use a more efficient refresh method
            import time
            current_time = time.time()
            if 'last_refresh_time' not in st.session_state:
                st.session_state.last_refresh_time = current_time

            if current_time - st.session_state.last_refresh_time >= refresh_interval:
                st.session_state.last_refresh_time = current_time
                st.rerun()

        # Check if simulation is running and show live data
        if st.session_state.sim_runner.is_running and st.session_state.sim_runner.current_output_dir:
            st.subheader("🟢 Live Simulation Data")

            # Load current simulation data from memory queue (Phase 1: Shared Memory)
            new_data = st.session_state.sim_runner.get_live_data()
            df = st.session_state.sim_runner.get_buffered_data()

            if df is not None and len(df) > 0:
                # Get actual elapsed time from simulation data
                actual_hours = df['time_hours'].iloc[-1] if 'time_hours' in df.columns else 0
                st.info(f"📊 Simulation running: {actual_hours:.1f} hours elapsed, {len(df)} data points")
                st.success(f"⚡ Memory streaming: {len(new_data)} new points this refresh")
            else:
                st.info("Waiting for simulation data...")
                df = None
        else:
            # Load most recent simulation data
            recent_sims = load_recent_simulations()
            if recent_sims:
                latest_sim = recent_sims[0]

                st.subheader(f"Latest Simulation: {latest_sim['name']}")

                # Load and display data
                df = load_simulation_data(latest_sim['path'])
            else:
                df = None

        if df is not None:
            # Phase 2: Smart plot updates - only recreate when needed
            if st.session_state.sim_runner.should_update_plots():
                fig = create_real_time_plots(df)
                st.session_state['cached_plot'] = fig
                st.plotly_chart(fig, use_container_width=True, key="monitor_plots")
            else:
                # Use cached plot if no new data
                cached_plot = st.session_state.get('cached_plot')
                if cached_plot is not None:
                    st.plotly_chart(cached_plot, use_container_width=True, key="monitor_plots")
                else:
                    # Fallback: create new plot
                    fig = create_real_time_plots(df)
                    st.session_state['cached_plot'] = fig
                    st.plotly_chart(fig, use_container_width=True, key="monitor_plots")

            # Current status metrics
            if len(df) > 0:
                latest = df.iloc[-1]

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Time", f"{latest['time_hours']:.1f} h")
                with col2:
                    st.metric("Reservoir Conc", f"{latest['reservoir_concentration']:.2f} mM")
                with col3:
                    st.metric("Power Output", f"{latest['total_power']:.3f} W")
                with col4:
                    st.metric("Current Action", int(latest['q_action']))
        else:
            if not st.session_state.sim_runner.is_running:
                st.info("No recent simulations found. Start a simulation to see real-time monitoring.")
            else:
                st.info("Waiting for simulation to generate data...")

    with tab3:
        st.header("Simulation Results")

        if st.session_state.simulation_results:
            results = st.session_state.simulation_results

            # Performance dashboard
            st.subheader("Performance Metrics")
            create_performance_dashboard(results)

            # Detailed results
            st.subheader("Detailed Results")

            col1, col2 = st.columns(2)

            with col1:
                st.json(results.get('performance_metrics', {}))

            with col2:
                st.json(results.get('simulation_info', {}))

        else:
            st.info("No simulation results available. Run a simulation first.")

    with tab4:
        st.header("Simulation History")

        recent_sims = load_recent_simulations()

        if recent_sims:
            # Create summary table
            df_history = pd.DataFrame(recent_sims)

            # Display table with metrics
            st.dataframe(
                df_history[['name', 'duration', 'final_conc', 'control_effectiveness']].rename(columns={
                    'name': 'Simulation',
                    'duration': 'Duration (h)',
                    'final_conc': 'Final Conc (mM)',
                    'control_effectiveness': 'Control Eff (%)'
                }),
                use_container_width=True
            )

            # Selection for detailed view
            selected_sim = st.selectbox(
                "Select simulation for detailed view:",
                options=recent_sims,
                format_func=lambda x: f"{x['name']} - {x['duration']}h"
            )

            if selected_sim:
                # Load and display selected simulation
                df = load_simulation_data(selected_sim['path'])
                if df is not None:
                    st.subheader(f"Detailed View: {selected_sim['name']}")
                    fig = create_real_time_plots(df)
                    st.plotly_chart(fig, use_container_width=True, key=f"history_plot_{selected_sim['name']}")

                    # Download option
                    st.download_button(
                        label="📥 Download CSV Data",
                        data=df.to_csv(index=False),
                        file_name=f"{selected_sim['name']}_data.csv",
                        mime="text/csv"
                    )
        else:
            st.info("No simulation history found.")

    # Footer
    st.markdown("---")
    st.markdown("🔬 MFC Simulation Control Panel | Built with Streamlit")

    # Cleanup on app close
    if st.session_state.sim_runner.is_running:
        st.sidebar.warning("⚠️ Simulation running - will cleanup on stop")

def cleanup_on_exit():
    """Cleanup function to be called when app exits"""
    try:
        if 'sim_runner' in st.session_state:
            if st.session_state.sim_runner.is_running:
                st.session_state.sim_runner.stop_simulation()
    except Exception:
        pass

if __name__ == "__main__":
    import atexit
    atexit.register(cleanup_on_exit)
    main()
