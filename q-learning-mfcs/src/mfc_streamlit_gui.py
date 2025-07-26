#!/usr/bin/env python3
"""
Streamlit GUI for MFC Simulation Control and Monitoring
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
import gzip
from pathlib import Path
import sys
import os
import threading
import queue

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
    """Thread-safe simulation runner for Streamlit"""
    
    def __init__(self):
        self.simulation = None
        self.is_running = False
        self.should_stop = False
        self.results_queue = queue.Queue()
        self.thread = None
        self.current_output_dir = None
        
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
            self.results_queue.put(('stopped', 'Simulation stopped by user', self.current_output_dir))
            
            # Wait for thread to finish (with timeout)
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=5.0)  # Wait up to 5 seconds
                
            # Clean up resources
            self._cleanup_resources()
            return True
        return False
    
    def _cleanup_resources(self):
        """Clean up GPU/CPU resources after simulation stops"""
        try:
            print("🧹 Starting aggressive GPU memory cleanup...")
            
            # Force delete any simulation objects first
            if hasattr(self, 'simulation'):
                del self.simulation
                
            # Clear JAX GPU memory aggressively
            try:
                import jax
                import jax.numpy as jnp
                
                # Clear all JAX compilation cache
                if hasattr(jax, 'clear_caches'):
                    jax.clear_caches()
                    print("   ✅ JAX caches cleared")
                    
                # Clear all backends
                if hasattr(jax, 'clear_backends'):
                    jax.clear_backends()
                    print("   ✅ JAX backends cleared")
                
                # For ROCm specifically, clear device arrays
                try:
                    # Get all live arrays and delete them
                    import gc
                    for obj in gc.get_objects():
                        if hasattr(obj, '__class__') and 'DeviceArray' in str(type(obj)):
                            del obj
                    print("   ✅ JAX device arrays cleared")
                except Exception:
                    pass
                    
            except ImportError:
                pass
            
            # Clear PyTorch CUDA cache if using NVIDIA
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    print("   ✅ CUDA cache cleared")
            except ImportError:
                pass
            
            # Force multiple garbage collections
            import gc
            for i in range(5):  # More aggressive cleanup
                collected = gc.collect()
                print(f"   🗑️ GC cycle {i+1}: {collected} objects collected")
                
            print("🧹 Aggressive GPU cleanup completed")
                
        except Exception as e:
            print(f"⚠️ Warning: Failed to clean up resources: {e}")
        
    def _run_simulation(self, config, duration_hours, n_cells=None, electrode_area_m2=None, target_conc=None, gui_refresh_interval=5.0):
        """Run simulation in background"""
        try:
            # Import here to avoid circular imports
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from mfc_gpu_accelerated import GPUAcceleratedMFC
            from datetime import datetime
            from pathlib import Path
            import pandas as pd
            import json
            
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
            # Save data every GUI refresh interval (in simulation time)
            # Convert GUI refresh seconds to simulation hours, then to steps
            gui_refresh_hours = gui_refresh_interval / 3600.0  # Convert seconds to hours
            save_interval_steps = max(1, int(gui_refresh_hours / dt_hours))
            
            # Also maintain a minimum save frequency for very slow refresh rates
            min_save_steps = 10  # Save at least every 1 hour of simulation time
            save_interval_steps = min(save_interval_steps, min_save_steps)
            
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
            
            # Run simulation with stop check and progress reporting
            for step in range(n_steps):
                if self.should_stop:
                    print(f"Simulation stopped by user at step {step}/{n_steps}")
                    break
                    
                current_time = step * dt_hours
                
                # Simulate timestep
                step_results = mfc_sim.simulate_timestep(dt_hours)
                
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
                    
                    # Save data file immediately for real-time monitoring
                    df = pd.DataFrame(results)
                    data_file = output_dir / f"gui_simulation_data_{timestamp}.csv.gz"
                    df.to_csv(data_file, compression='gzip', index=False)
                    
                    # Send progress update to GUI
                    progress_pct = (step / n_steps) * 100
                    self.results_queue.put(('progress', f"Step {step}/{n_steps} ({progress_pct:.1f}%)", current_time))
                
                # Progress logging for longer simulations
                if step > 0 and step % max(1, n_steps // 20) == 0:  # Log every 5% progress
                    progress_pct = (step / n_steps) * 100
                    print(f"GUI simulation progress: {progress_pct:.1f}% ({step}/{n_steps} steps, {current_time:.1f}h)")
            
            # Save final results
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
        # Get the most recent status from the queue
        latest_status = None
        try:
            while not self.results_queue.empty():
                latest_status = self.results_queue.get_nowait()
        except queue.Empty:
            pass
        return latest_status

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
    """Load simulation data from directory with real-time updates"""
    data_dir = Path(data_dir)
    
    # Find compressed CSV file
    csv_files = list(data_dir.glob("*_data_*.csv.gz"))
    if not csv_files:
        return None
        
    csv_file = csv_files[0]
    
    try:
        # Check if file was recently modified (for real-time detection)
        file_mtime = csv_file.stat().st_mtime
        current_time = time.time()
        is_recent = (current_time - file_mtime) < 60  # Modified within last minute
        
        with gzip.open(csv_file, 'rt') as f:
            df = pd.read_csv(f)
            
        if is_recent:
            # Add metadata about freshness for GUI display
            df.attrs['is_live_data'] = True
            df.attrs['last_modified'] = file_mtime
        else:
            df.attrs['is_live_data'] = False
            
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
                    with open(json_files[0], 'r') as f:
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
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Substrate Concentration', 'Power Output', 'Q-Learning Actions', 'Biofilm Growth'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
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
        try:
            biofilm_avg = df['biofilm_thicknesses'].apply(lambda x: 
                sum(eval(x)) / len(eval(x)) if isinstance(x, str) and x.strip() else 0)
            fig.add_trace(
                go.Scatter(x=df['time_hours'], y=biofilm_avg,
                          name='Avg Thickness', line=dict(color='brown', width=2)),
                row=2, col=2
            )
        except Exception:
            pass  # Skip if biofilm data is malformed
    
    # Update layout
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
        
        # Status display with enhanced debugging
        status = st.session_state.sim_runner.get_status()
        
        # Always show current simulation state
        if st.session_state.sim_runner.is_running:
            st.info("🔄 Simulation is running...")
            if st.session_state.sim_runner.current_output_dir:
                st.text(f"Output directory: {st.session_state.sim_runner.current_output_dir}")
        else:
            st.text("🔴 No simulation currently running")
        
        # Handle status messages
        if status:
            if status[0] == 'completed':
                st.success("✅ Simulation completed successfully!")
                st.session_state.simulation_results = status[1]
                st.session_state.last_output_dir = status[2]
                # Clear running state
                st.session_state.sim_runner.is_running = False
            elif status[0] == 'stopped':
                st.warning(f"⏹️ {status[1]}")
                st.session_state.last_output_dir = status[2]
                # Clear running state
                st.session_state.sim_runner.is_running = False
            elif status[0] == 'error':
                st.error(f"❌ Simulation failed: {status[1]}")
                # Clear running state
                st.session_state.sim_runner.is_running = False
            elif status[0] == 'progress':
                # Show real-time progress
                progress_text = status[1]
                sim_time = status[2]
                
                # Extract progress percentage from text
                import re
                progress_match = re.search(r'\((\d+\.?\d*)%\)', progress_text)
                if progress_match:
                    progress_pct = float(progress_match.group(1))
                    st.progress(progress_pct / 100.0, text=f"🔄 {progress_text} - Simulation time: {sim_time:.1f}h")
                else:
                    st.info(f"🔄 {progress_text} - Simulation time: {sim_time:.1f}h")
                
                # Store progress for monitoring tab
                st.session_state.last_progress = status
        
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
        
        # GPU cleanup button (separate row)
        col1, col2, col3 = st.columns(3)
        with col2:
            if st.button("🧹 Force GPU Cleanup", key="gpu_cleanup_btn"):
                st.info("🧹 Performing manual GPU cleanup...")
                st.session_state.sim_runner._cleanup_resources()
                st.success("✅ GPU cleanup completed!")
        
        # Simulation status
        if st.session_state.sim_runner.is_running:
            st.markdown('<p class="status-running">🟢 Simulation Running...</p>', unsafe_allow_html=True)
            st.info("💡 Switch to the Monitor tab to see real-time updates")
        else:
            st.markdown('<p class="status-stopped">🔴 Simulation Stopped</p>', unsafe_allow_html=True)
        
        # Configuration preview
        st.subheader("Current Configuration")
        current_refresh = st.session_state.get('current_refresh_interval', 5.0)
        # Calculate data save frequency
        gui_refresh_hours = current_refresh / 3600.0
        save_interval_steps = max(1, int(gui_refresh_hours / 0.1))  # 0.1h timestep
        min_save_steps = 10
        actual_save_steps = min(save_interval_steps, min_save_steps)
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
            auto_refresh = st.checkbox("Enable Auto-refresh", value=False)
        
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
                
        # Manual refresh button for immediate updates
        if st.button("🔄 Manual Refresh", key="manual_refresh_monitor"):
            st.rerun()
        
        # Implement auto-refresh using session state and JavaScript
        if auto_refresh:
            current_time = time.time()
            last_refresh = st.session_state.get('last_auto_refresh', 0)
            
            if current_time - last_refresh >= refresh_interval:
                st.session_state.last_auto_refresh = current_time
                st.rerun()
            else:
                time_until_refresh = refresh_interval - (current_time - last_refresh)
                st.write(f"⏱️ Next refresh in {time_until_refresh:.1f}s")
                
                # Force page refresh using meta tag
                st.markdown(f"""
                <meta http-equiv="refresh" content="{int(time_until_refresh) + 1}">
                """, unsafe_allow_html=True)
        
        # Check if simulation is running and show live data
        if st.session_state.sim_runner.is_running and st.session_state.sim_runner.current_output_dir:
            st.subheader("🟢 Live Simulation Data")
            
            # Load current simulation data
            df = load_simulation_data(st.session_state.sim_runner.current_output_dir)
            if df is not None and len(df) > 0:
                # Get actual elapsed time from simulation data
                actual_hours = df['time_hours'].iloc[-1] if 'time_hours' in df.columns else 0
                refresh_rate = st.session_state.get('current_refresh_interval', 5.0)
                
                # Show data freshness info
                is_live = getattr(df, 'attrs', {}).get('is_live_data', False)
                last_modified = getattr(df, 'attrs', {}).get('last_modified', 0)
                
                if is_live:
                    time_since_update = time.time() - last_modified
                    st.success(f"🟢 LIVE DATA: {actual_hours:.1f}h elapsed, {len(df)} points (updated {time_since_update:.0f}s ago)")
                else:
                    st.info(f"📊 Simulation data: {actual_hours:.1f} hours elapsed, {len(df)} data points")
                    
                st.info(f"⚡ Real-time sync: Data saved every {refresh_rate}s")
            else:
                st.warning("⏳ Waiting for simulation data...")
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
            # Real-time plots
            fig = create_real_time_plots(df)
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