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
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import psutil
from datetime import datetime

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
.debug-console {
    background-color: #1e1e1e;
    color: #ffffff;
    font-family: 'Courier New', monospace;
    font-size: 12px;
    padding: 10px;
    border-radius: 5px;
    max-height: 400px;
    overflow-y: auto;
    white-space: pre-wrap;
}
</style>
""", unsafe_allow_html=True)

class DebugLogger:
    """Thread-safe debug logger that captures console output and provides GUI display"""
    
    def __init__(self):
        self.messages = deque(maxlen=1000)  # Keep last 1000 messages
        self.lock = threading.Lock()
        self.enabled = False
        
    def enable_debug(self):
        """Enable debug logging"""
        self.enabled = True
        
    def disable_debug(self):
        """Disable debug logging"""
        self.enabled = False
        
    def log(self, message: str, level: str = "INFO"):
        """Add a debug message with timestamp"""
        if self.enabled:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            formatted_message = f"[{timestamp}] {level}: {message}"
            
            with self.lock:
                self.messages.append(formatted_message)
                
            # Also print to console if debug is enabled
            print(formatted_message)
        else:
            # Silent mode - only print to console without timestamp
            print(message)
    
    def get_messages(self) -> List[str]:
        """Get all debug messages (thread-safe)"""
        with self.lock:
            return list(self.messages)
    
    def clear_messages(self):
        """Clear all debug messages"""
        with self.lock:
            self.messages.clear()

# Global debug logger instance
debug_logger = DebugLogger()

class DebugPrintCapture:
    """Context manager to capture print statements and route them through debug logger"""
    
    def __init__(self, logger: DebugLogger, level: str = "INFO"):
        self.logger = logger
        self.level = level
        self.old_stdout: Optional[Any] = None
        
    def __enter__(self):
        self.old_stdout = sys.stdout
        sys.stdout = self
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_stdout
        
    def write(self, text: str):
        if text.strip():  # Only log non-empty messages
            self.logger.log(text.strip(), self.level)
        # Also write to original stdout
        if self.old_stdout is not None:
            self.old_stdout.write(text)
        
    def flush(self):
        if self.old_stdout is not None:
            self.old_stdout.flush()

@dataclass
class SimulationSnapshot:
    """Lightweight simulation data snapshot for GUI updates"""
    current_time: float
    reservoir_concentration: float
    outlet_concentration: float
    total_power: float
    total_current: float
    system_voltage: float
    flow_rate_ml_h: float
    substrate_efficiency: float
    biofilm_thickness_avg: float
    q_action: int
    reward: float
    epsilon: float
    step_number: int
    progress_percent: float

class OptimizedDataBuffer:
    """Memory-efficient data buffer for GUI updates"""
    
    def __init__(self, max_gui_points: int = 1000):
        self.max_gui_points = max_gui_points
        self.snapshots: deque[SimulationSnapshot] = deque(maxlen=max_gui_points)
        self.full_data_file = None
        self.last_save_time = 0
        self.save_interval = 30.0  # Save full data every 30 seconds
        
    def add_snapshot(self, snapshot: SimulationSnapshot):
        """Add a new data snapshot"""
        self.snapshots.append(snapshot)
        
    def get_gui_data(self) -> Dict[str, List]:
        """Get data optimized for GUI display (lightweight)"""
        if not self.snapshots:
            return {}
            
        # Convert recent snapshots to dict format for plotting
        data = {}
        fields = ['current_time', 'reservoir_concentration', 'outlet_concentration', 
                 'total_power', 'total_current', 'system_voltage', 'flow_rate_ml_h',
                 'substrate_efficiency', 'biofilm_thickness_avg', 'q_action', 'reward', 
                 'epsilon', 'progress_percent']
        
        for field in fields:
            data[field] = [getattr(snapshot, field) for snapshot in self.snapshots]
            
        return data
        
    def get_latest_metrics(self) -> Dict[str, float]:
        """Get latest key metrics for dashboard"""
        if not self.snapshots:
            return {}
            
        latest = self.snapshots[-1]
        return {
            'current_time': latest.current_time,
            'total_power': latest.total_power,
            'efficiency': latest.substrate_efficiency,
            'concentration_error': abs(latest.reservoir_concentration - latest.outlet_concentration),
            'progress': latest.progress_percent
        }

class SimulationRunner:
    """Thread-safe simulation runner for Streamlit with performance optimizations"""
    
    def __init__(self):
        self.simulation = None
        self.is_running = False
        self.should_stop = False
        self.results_queue = queue.Queue()
        self.thread = None
        self.current_output_dir = None
        
        # Optimized data management
        self.data_buffer = OptimizedDataBuffer()
        self.live_data_lock = threading.RLock()  # Allow multiple readers
        self.adaptive_refresh_interval = 5.0
        self.simulation_speed_avg = deque(maxlen=10)  # Track simulation speed
        
        # Performance monitoring
        self.memory_monitor = threading.Event()
        self.start_memory_mb = 0
    
    def is_actually_running(self):
        """Check if simulation is actually running by checking both flag and thread state"""
        return self.is_running and self.thread and self.thread.is_alive()
    
    def get_live_data(self):
        """Get current simulation data from memory (thread-safe, optimized)"""
        with self.live_data_lock:
            return self.data_buffer.get_gui_data()
    
    def get_latest_metrics(self):
        """Get latest simulation metrics (lightweight)"""
        with self.live_data_lock:
            return self.data_buffer.get_latest_metrics()
    
    def _calculate_adaptive_refresh(self, simulation_speed: float):
        """Calculate adaptive refresh interval based on simulation performance"""
        self.simulation_speed_avg.append(simulation_speed)
        
        if len(self.simulation_speed_avg) >= 3:
            avg_speed = sum(self.simulation_speed_avg) / len(self.simulation_speed_avg)
            
            # Adaptive refresh: faster simulations get slower GUI updates
            if avg_speed > 100:  # Very fast simulation
                self.adaptive_refresh_interval = min(10.0, self.adaptive_refresh_interval * 1.2)
            elif avg_speed < 10:  # Slow simulation  
                self.adaptive_refresh_interval = max(2.0, self.adaptive_refresh_interval * 0.8)
                
    def _monitor_memory_usage(self):
        """Monitor memory usage during simulation"""
        try:
            process = psutil.Process()
            current_memory_mb = process.memory_info().rss / 1024 / 1024
            memory_increase = current_memory_mb - self.start_memory_mb
            
            if memory_increase > 500:  # Alert if memory increased by >500MB
                print(f"⚠️ Memory usage increased by {memory_increase:.1f}MB")
                
        except Exception:
            pass  # Don't fail simulation for memory monitoring issues
        
    def start_simulation(self, config, duration_hours, n_cells=None, electrode_area_m2=None, target_conc=None, gui_refresh_interval=5.0, debug_mode=False):
        """Start simulation in background thread with performance optimizations
        
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
            
        # Initialize performance monitoring
        try:
            process = psutil.Process()
            self.start_memory_mb = process.memory_info().rss / 1024 / 1024
        except Exception:
            self.start_memory_mb = 0
            
        self.is_running = True
        self.should_stop = False
        self.adaptive_refresh_interval = gui_refresh_interval
        self.data_buffer = OptimizedDataBuffer()  # Reset buffer
        
        # Log simulation start
        if debug_logger.enabled:
            debug_logger.log(f"🚀 Starting simulation: {duration_hours}h, {n_cells} cells, debug={debug_mode}", "SIM")
        
        self.thread = threading.Thread(
            target=self._run_simulation_optimized,
            args=(config, duration_hours, n_cells, electrode_area_m2, target_conc, gui_refresh_interval, debug_mode)
        )
        self.thread.start()
        return True
        
    def stop_simulation(self):
        """Stop the running simulation"""
        if self.is_running:
            if debug_logger.enabled:
                debug_logger.log("⏹️ Stopping simulation by user request", "SIM")
                
            self.should_stop = True
            self.results_queue.put(('stopped', 'Simulation stopped by user', self.current_output_dir))
            
            # Wait for thread to finish (with timeout)
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=5.0)  # Wait up to 5 seconds
                
            # Clean up resources
            self._cleanup_resources(verbose=False)
            
            if debug_logger.enabled:
                debug_logger.log("✅ Simulation stopped and cleaned up", "SIM")
            return True
        return False
    
    def _cleanup_resources(self, verbose: bool = True):
        """Clean up GPU/CPU resources after simulation stops"""
        try:
            # Use debug logger if enabled
            if debug_logger.enabled:
                debug_logger.log("🧹 Starting GPU memory cleanup...", "CLEANUP")
                verbose = False  # Use debug logger instead of print
            elif verbose:
                print("🧹 Starting GPU memory cleanup...")
            
            # Force delete any simulation objects first
            if hasattr(self, 'simulation'):
                del self.simulation
                if debug_logger.enabled:
                    debug_logger.log("✅ Simulation object deleted", "CLEANUP")
                elif verbose:
                    print("   ✅ Simulation object deleted")
                
            # Clear JAX GPU memory using JAX 0.6.0 compatible methods
            try:
                import jax
                
                # 1. Move device arrays to host memory
                try:
                    import gc
                    device_arrays_found = 0
                    for obj in gc.get_objects():
                        if hasattr(obj, '__class__') and hasattr(obj, 'device'):
                            try:
                                # Move device arrays to host memory
                                jax.device_get(obj)
                                device_arrays_found += 1
                            except Exception:
                                pass
                    
                    if debug_logger.enabled:
                        if device_arrays_found > 0:
                            debug_logger.log(f"✅ Moved {device_arrays_found} device arrays to host", "CLEANUP")
                        else:
                            debug_logger.log("✅ No device arrays found to clean", "CLEANUP")
                    elif verbose:
                        if device_arrays_found > 0:
                            print(f"   ✅ Moved {device_arrays_found} device arrays to host")
                        else:
                            print("   ✅ No device arrays found to clean")
                        
                except Exception as e:
                    if debug_logger.enabled:
                        debug_logger.log(f"⚠️ Device array cleanup warning: {e}", "CLEANUP")
                    elif verbose:
                        print(f"   ⚠️ Device array cleanup warning: {e}")
                
                # 2. Clear JAX compilation cache (available in JAX 0.6.0)
                try:
                    if hasattr(jax, 'clear_caches'):
                        jax.clear_caches()
                        if debug_logger.enabled:
                            debug_logger.log("✅ JAX compilation cache cleared", "CLEANUP")
                        elif verbose:
                            print("   ✅ JAX compilation cache cleared")
                except Exception as e:
                    if debug_logger.enabled:
                        debug_logger.log(f"⚠️ Cache clear warning: {e}", "CLEANUP")
                    elif verbose:
                        print(f"   ⚠️ Cache clear warning: {e}")
                    
            except ImportError:
                if debug_logger.enabled:
                    debug_logger.log("⚠️ JAX not available for cleanup", "CLEANUP")
                elif verbose:
                    print("   ⚠️ JAX not available for cleanup")
            
            # Clear PyTorch CUDA cache if using NVIDIA (fallback)
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    if debug_logger.enabled:
                        debug_logger.log("✅ CUDA cache cleared", "CLEANUP")
                    elif verbose:
                        print("   ✅ CUDA cache cleared")
            except ImportError:
                pass
            
            # Force aggressive garbage collection
            import gc
            total_collected = 0
            for i in range(3):  # Multiple GC cycles
                collected = gc.collect()
                total_collected += collected
                if debug_logger.enabled and collected > 0:
                    debug_logger.log(f"🗑️ GC cycle {i+1}: {collected} objects collected", "CLEANUP")
                elif verbose and collected > 0:
                    print(f"   🗑️ GC cycle {i+1}: {collected} objects collected")
            
            if debug_logger.enabled:
                if total_collected == 0:
                    debug_logger.log("✅ No objects to garbage collect", "CLEANUP")
                else:
                    debug_logger.log(f"✅ Total objects collected: {total_collected}", "CLEANUP")
            elif verbose:
                if total_collected == 0:
                    print("   ✅ No objects to garbage collect")
                else:
                    print(f"   ✅ Total objects collected: {total_collected}")
                
            # ROCm memory monitoring
            try:
                import subprocess
                result = subprocess.run(['rocm-smi', '--showmeminfo', 'vram'], 
                                      capture_output=True, text=True, timeout=3)
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'VRAM Total Used Memory' in line:
                            used_bytes = int(line.split(':')[-1].strip())
                            used_mb = used_bytes / (1024 * 1024)
                            if debug_logger.enabled:
                                debug_logger.log(f"📊 Current VRAM usage: {used_mb:.1f} MB", "CLEANUP")
                            elif verbose:
                                print(f"   📊 Current VRAM usage: {used_mb:.1f} MB")
                            break
            except Exception:
                if debug_logger.enabled:
                    debug_logger.log("⚠️ ROCm memory stats unavailable", "CLEANUP")
                elif verbose:
                    print("   ⚠️ ROCm memory stats unavailable")
                
            if debug_logger.enabled:
                debug_logger.log("🧹 GPU cleanup completed", "CLEANUP")
            elif verbose:
                print("🧹 GPU cleanup completed")
                
        except Exception as e:
            print(f"⚠️ Warning: Failed to clean up resources: {e}")
        
    def _run_simulation_optimized(self, config, duration_hours, n_cells=None, electrode_area_m2=None, target_conc=None, gui_refresh_interval=5.0, debug_mode=False):
        """Optimized simulation runner with reduced overhead"""
        try:
            # Import here to avoid circular imports
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from mfc_gpu_accelerated import GPUAcceleratedMFC
            from datetime import datetime
            from pathlib import Path
            
            # Setup output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if debug_mode:
                from path_config import enable_debug_mode, get_simulation_data_path
                enable_debug_mode()
                output_dir = Path(get_simulation_data_path(f"gui_simulation_{timestamp}"))
            else:
                output_dir = Path(f"../data/simulation_data/gui_simulation_{timestamp}")
            
            output_dir.mkdir(parents=True, exist_ok=True)
            self.current_output_dir = output_dir
            
            # Update config
            if n_cells is not None:
                config.n_cells = n_cells
            if electrode_area_m2 is not None:
                config.electrode_area_per_cell = electrode_area_m2
            if target_conc is not None:
                config.substrate_target_concentration = target_conc
            
            # Initialize simulation
            mfc_sim = GPUAcceleratedMFC(config)
            
            # Optimized timing parameters
            gui_refresh_hours = self.adaptive_refresh_interval / 3600.0
            dt_hours = min(gui_refresh_hours / 3, 6.0 / 60.0)  # Max 6 minutes
            dt_hours = max(dt_hours, 30.0 / 3600.0)  # Min 30 seconds
            
            n_steps = int(duration_hours / dt_hours)
            save_interval_steps = max(1, int(gui_refresh_hours / dt_hours))
            
            if debug_logger.enabled:
                debug_logger.log(f"🚀 Optimized GUI simulation: {n_steps} steps, save every {save_interval_steps} steps", "SIM")
            else:
                print(f"🚀 Optimized GUI simulation: {n_steps} steps, save every {save_interval_steps} steps")
            
            # Lightweight results tracking (only for full data file)
            full_results = {
                'time_hours': [],
                'reservoir_concentration': [],
                'outlet_concentration': [],
                'total_power': [],
                'total_current': [],
                'system_voltage': [],
                'flow_rate_ml_h': [],
                'substrate_efficiency': [],
                'biofilm_thicknesses': [],
                'q_action': [],
                'reward': [],
                'epsilon': []
            }
            
            # Create initial snapshot
            initial_snapshot = SimulationSnapshot(
                current_time=0.0,
                reservoir_concentration=float(mfc_sim.reservoir_concentration),
                outlet_concentration=float(mfc_sim.outlet_concentration),
                total_power=0.0,
                total_current=0.0,
                system_voltage=0.7,
                flow_rate_ml_h=10.0,
                substrate_efficiency=1.0,
                biofilm_thickness_avg=float(np.mean(mfc_sim.biofilm_thicknesses)),
                q_action=0,
                reward=0.0,
                epsilon=getattr(mfc_sim, 'epsilon', 0.1),
                step_number=0,
                progress_percent=0.0
            )
            
            # Add initial snapshot
            with self.live_data_lock:
                self.data_buffer.add_snapshot(initial_snapshot)
            
            # Performance tracking
            last_time = time.time()
            steps_since_last_check = 0
            
            # Main simulation loop
            for step in range(n_steps):
                if self.should_stop:
                    print(f"Simulation stopped by user at step {step}/{n_steps}")
                    break
                    
                current_time = step * dt_hours
                
                try:
                    # Simulate timestep
                    step_results = mfc_sim.simulate_timestep(dt_hours)
                    
                    # Performance monitoring
                    steps_since_last_check += 1
                    
                    # Update adaptive refresh (every 100 steps)
                    if steps_since_last_check >= 100:
                        total_time = time.time() - last_time
                        simulation_speed = steps_since_last_check / total_time if total_time > 0 else 1.0
                        self._calculate_adaptive_refresh(simulation_speed)
                        
                        # Update save interval based on new refresh rate
                        gui_refresh_hours = self.adaptive_refresh_interval / 3600.0
                        save_interval_steps = max(1, int(gui_refresh_hours / dt_hours))
                        
                        last_time = time.time()
                        steps_since_last_check = 0
                        
                        # Memory monitoring
                        self._monitor_memory_usage()
                    
                    # Save data at intervals (reduced frequency)
                    if step % save_interval_steps == 0:
                        # Create lightweight snapshot for GUI
                        snapshot = SimulationSnapshot(
                            current_time=current_time,
                            reservoir_concentration=float(mfc_sim.reservoir_concentration),
                            outlet_concentration=float(mfc_sim.outlet_concentration),
                            total_power=step_results['total_power'],
                            total_current=step_results.get('total_current', 0.0),
                            system_voltage=step_results.get('system_voltage', 0.7),
                            flow_rate_ml_h=step_results.get('flow_rate_ml_h', 10.0),
                            substrate_efficiency=step_results.get('substrate_efficiency', 1.0),
                            biofilm_thickness_avg=float(np.mean(mfc_sim.biofilm_thicknesses)),
                            q_action=step_results['action'],
                            reward=step_results['reward'],
                            epsilon=step_results['epsilon'],
                            step_number=step,
                            progress_percent=(step / n_steps) * 100
                        )
                        
                        # Add to optimized buffer (thread-safe)
                        with self.live_data_lock:
                            self.data_buffer.add_snapshot(snapshot)
                        
                        # Add to full results (for final file save)
                        full_results['time_hours'].append(current_time)
                        full_results['reservoir_concentration'].append(float(mfc_sim.reservoir_concentration))
                        full_results['outlet_concentration'].append(float(mfc_sim.outlet_concentration))
                        full_results['total_power'].append(step_results['total_power'])
                        full_results['total_current'].append(step_results.get('total_current', 0.0))
                        full_results['system_voltage'].append(step_results.get('system_voltage', 0.7))
                        full_results['flow_rate_ml_h'].append(step_results.get('flow_rate_ml_h', 10.0))
                        full_results['substrate_efficiency'].append(step_results.get('substrate_efficiency', 1.0))
                        full_results['biofilm_thicknesses'].append([float(x) for x in mfc_sim.biofilm_thicknesses])
                        full_results['q_action'].append(step_results['action'])
                        full_results['reward'].append(step_results['reward'])
                        full_results['epsilon'].append(step_results['epsilon'])
                        
                        # Send progress update
                        progress_pct = (step / n_steps) * 100
                        self.results_queue.put(('progress', f"Step {step}/{n_steps} ({progress_pct:.1f}%)", current_time))
                    
                    # Progress logging (reduced frequency)
                    if step > 0 and step % max(1, n_steps // 10) == 0:  # Log every 10%
                        progress_pct = (step / n_steps) * 100
                        print(f"Optimized GUI simulation: {progress_pct:.1f}% ({step}/{n_steps} steps, {current_time:.1f}h)")
                
                except Exception as sim_error:
                    error_msg = f"Simulation error at step {step}: {sim_error}"
                    print(f"❌ {error_msg}")
                    self.results_queue.put(('error', error_msg, output_dir))
                    break
            
            # Save final results using efficient format (Parquet instead of CSV.gz)
            try:
                df = pd.DataFrame(full_results)
                parquet_file = output_dir / f"gui_simulation_data_{timestamp}.parquet"
                df.to_parquet(parquet_file, compression='snappy', index=False)
                print(f"💾 Final results saved to {parquet_file} (Parquet format)")
                
                # Also save as CSV.gz for backwards compatibility (but don't wait for it)
                threading.Thread(target=lambda: df.to_csv(
                    output_dir / f"gui_simulation_data_{timestamp}.csv.gz", 
                    compression='gzip', index=False
                )).start()
                
            except Exception as save_error:
                print(f"⚠️ Warning: Could not save results: {save_error}")
            
            # Calculate final metrics
            final_metrics = mfc_sim.calculate_final_metrics(full_results)
            
            # Success
            self.results_queue.put(('completed', 'Simulation completed successfully', output_dir, final_metrics))
            print(f"✅ Optimized GUI simulation completed: {duration_hours}h simulated")
            
        except Exception as e:
            error_msg = f"Simulation failed: {e}"
            print(f"❌ {error_msg}")
            self.results_queue.put(('error', error_msg, self.current_output_dir))
        finally:
            self.is_running = False
            self._cleanup_resources(verbose=False)
    
    def _run_simulation(self, config, duration_hours, n_cells=None, electrode_area_m2=None, target_conc=None, gui_refresh_interval=5.0, debug_mode=False):
        """Run simulation in background"""
        try:
            # Import here to avoid circular imports
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from mfc_gpu_accelerated import GPUAcceleratedMFC
            from datetime import datetime
            from pathlib import Path
            import pandas as pd
            import json
            
            # Handle debug mode
            if debug_mode:
                from path_config import enable_debug_mode, get_simulation_data_path
                enable_debug_mode()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = Path(get_simulation_data_path(f"gui_simulation_{timestamp}"))
            else:
                # Create output directory in normal location
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
            
            # Simulation parameters - use smaller timesteps for smoother visualization
            # Synchronize timestep with GUI refresh for optimal updates
            gui_refresh_hours = gui_refresh_interval / 3600.0  # Convert seconds to hours
            
            # Use timestep that's a fraction of GUI refresh interval for smooth updates
            # Aim for 2-5 simulation steps per GUI refresh for good temporal resolution
            target_steps_per_refresh = 3
            dt_hours = gui_refresh_hours / target_steps_per_refresh
            
            # Ensure timestep is reasonable (between 30 seconds and 6 minutes)
            min_dt_hours = 30.0 / 3600.0   # 30 seconds minimum
            max_dt_hours = 6.0 / 60.0      # 6 minutes maximum
            dt_hours = max(min_dt_hours, min(dt_hours, max_dt_hours))
            
            n_steps = int(duration_hours / dt_hours)
            
            # Calculate save interval - save every GUI refresh interval
            save_interval_steps = max(1, int(gui_refresh_hours / dt_hours))
            
            # Ensure we don't save too frequently (minimum 1 minute intervals)
            min_save_hours = 1.0 / 60.0  # 1 minute
            min_save_steps = max(1, int(min_save_hours / dt_hours))
            save_interval_steps = max(save_interval_steps, min_save_steps)
            
            print(f"GUI sync: Saving simulation data every {save_interval_steps} steps ({save_interval_steps * dt_hours:.2f} sim hours) for {gui_refresh_interval}s GUI refresh")
            
            # Progress tracking - comprehensive data collection
            results = {
                'time_hours': [],
                'reservoir_concentration': [],
                'outlet_concentration': [],
                'total_power': [],
                'biofilm_thicknesses': [],
                'substrate_addition_rate': [],
                'q_action': [],
                'epsilon': [],
                'reward': [],
                # Additional comprehensive monitoring parameters
                'individual_cell_powers': [],
                'total_current': [],
                'system_voltage': [],
                'flow_rate_ml_h': [],
                'substrate_efficiency': [],
                'concentration_error': [],
                'mixing_efficiency': [],
                'individual_biofilm_thicknesses': [],
                'individual_cell_concentrations': [],
                'biofilm_activity_factor': [],
                'q_value': []
            }
            
            # Add initial state as first data point
            initial_state = {
                'time_hours': 0.0,
                'reservoir_concentration': float(mfc_sim.reservoir_concentration),
                'outlet_concentration': float(mfc_sim.outlet_concentration),
                'total_power': 0.0,
                'biofilm_thicknesses': [float(x) for x in mfc_sim.biofilm_thicknesses],
                'substrate_addition_rate': 0.0,
                'q_action': 0,
                'epsilon': mfc_sim.epsilon if hasattr(mfc_sim, 'epsilon') else 0.1,
                'reward': 0.0,
                # Additional initial state parameters
                'individual_cell_powers': [0.0] * mfc_sim.n_cells,
                'total_current': 0.0,
                'system_voltage': 0.7,
                'flow_rate_ml_h': 10.0,
                'substrate_efficiency': 1.0,
                'concentration_error': 0.0,
                'mixing_efficiency': 1.0,
                'individual_biofilm_thicknesses': [float(x) for x in mfc_sim.biofilm_thicknesses],
                'individual_cell_concentrations': [float(x) for x in mfc_sim.cell_concentrations],
                'biofilm_activity_factor': 0.005,  # 1.0 μm / 200.0 μm
                'q_value': 0.0
            }
            
            for key, value in initial_state.items():
                results[key].append(value)
            
            # Save initial data point both to file and memory
            df = pd.DataFrame(results)
            data_file = output_dir / f"gui_simulation_data_{timestamp}.csv.gz"
            df.to_csv(data_file, compression='gzip', index=False)
            
            # Store in memory for GUI access (thread-safe)
            with self.live_data_lock:
                self.live_data = df.copy()
            
            print(f"💾 Initial state saved to {data_file} and memory")
            
            # Run simulation with stop check and progress reporting
            for step in range(n_steps):
                if self.should_stop:
                    print(f"Simulation stopped by user at step {step}/{n_steps}")
                    break
                    
                current_time = step * dt_hours
                
                try:
                    # Simulate timestep with error handling
                    step_results = mfc_sim.simulate_timestep(dt_hours)
                except Exception as sim_error:
                    error_msg = f"Simulation error at step {step}: {sim_error}"
                    print(f"❌ {error_msg}")
                    self.results_queue.put(('error', error_msg, output_dir))
                    # Even if simulation fails, we have initial data to show
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
                    # Collect additional comprehensive monitoring data
                    results['individual_cell_powers'].append(step_results.get('individual_cell_powers', [0.0] * mfc_sim.n_cells))
                    results['total_current'].append(step_results.get('total_current', 0.0))
                    results['system_voltage'].append(step_results.get('system_voltage', 0.7))
                    results['flow_rate_ml_h'].append(step_results.get('flow_rate_ml_h', 10.0))
                    results['substrate_efficiency'].append(step_results.get('substrate_efficiency', 1.0))
                    results['concentration_error'].append(step_results.get('concentration_error', 0.0))
                    results['mixing_efficiency'].append(step_results.get('mixing_efficiency', 1.0))
                    results['individual_biofilm_thicknesses'].append(step_results.get('individual_biofilm_thicknesses', [float(x) for x in mfc_sim.biofilm_thicknesses]))
                    results['individual_cell_concentrations'].append(step_results.get('individual_cell_concentrations', [float(x) for x in mfc_sim.cell_concentrations]))
                    results['biofilm_activity_factor'].append(step_results.get('biofilm_activity_factor', 0.005))
                    results['q_value'].append(step_results.get('q_value', 0.0))
                    
                    # Save data file and update memory for real-time monitoring
                    df = pd.DataFrame(results)
                    data_file = output_dir / f"gui_simulation_data_{timestamp}.csv.gz"
                    df.to_csv(data_file, compression='gzip', index=False)
                    
                    # Update live data in memory (thread-safe)
                    with self.live_data_lock:
                        self.live_data = df.copy()
                    
                    # Send progress update to GUI
                    progress_pct = (step / n_steps) * 100
                    self.results_queue.put(('progress', f"Step {step}/{n_steps} ({progress_pct:.1f}%)", current_time))
                
                # Progress logging for longer simulations
                if step > 0 and step % max(1, n_steps // 20) == 0:  # Log every 5% progress
                    progress_pct = (step / n_steps) * 100
                    print(f"GUI simulation progress: {progress_pct:.1f}% ({step}/{n_steps} steps, {current_time:.1f}h)")
            
            # Save final results both to file and memory
            df = pd.DataFrame(results)
            data_file = output_dir / f"gui_simulation_data_{timestamp}.csv.gz"
            df.to_csv(data_file, compression='gzip', index=False)
            
            # Update final data in memory (thread-safe)
            with self.live_data_lock:
                self.live_data = df.copy()
            
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
            self._cleanup_resources(verbose=False)
            self.is_running = False
            # Clear live data when simulation stops
            with self.live_data_lock:
                self.live_data = None
            
    def get_status(self):
        """Get current simulation status"""
        # Get the most recent status from the queue
        latest_status = None
        try:
            while not self.results_queue.empty():
                latest_status = self.results_queue.get_nowait()
        except queue.Empty:
            pass
        
        # Return a default status if no status is available
        if latest_status is None:
            if self.is_running:
                return ('running', 'Simulation in progress', self.current_output_dir)
            else:
                return ('idle', 'No simulation running', None)
        
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

def load_simulation_data_from_memory(sim_runner):
    """Load simulation data directly from memory (no file I/O)"""
    if not sim_runner or not sim_runner.is_actually_running():
        return None
    
    try:
        # Get live data from memory (thread-safe)
        data = sim_runner.get_live_data()
        
        # Handle both dict and DataFrame returns
        if data is None:
            st.info("📊 Waiting for simulation to generate data...")
            return None
        
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            if not data:  # Empty dict
                st.info("📊 Waiting for simulation to generate data...")
                return None
            df = pd.DataFrame(data)
        else:
            df = data
        
        # Check if DataFrame is empty
        if df.empty:
            st.info("📊 Waiting for simulation to generate data...")
            return None
        
        # Add metadata to indicate this is live data
        df.attrs['is_live_data'] = True
        df.attrs['last_modified'] = time.time()
        
        return df
    except Exception as e:
        st.error(f"Error loading data from memory: {e}")
        return None

def load_simulation_data(data_dir):
    """Load simulation data from directory with optimized format support"""
    data_dir = Path(data_dir)
    
    # Try to load Parquet file first (faster)
    parquet_files = list(data_dir.glob("*_data_*.parquet"))
    if parquet_files:
        parquet_file = parquet_files[0]
        try:
            df = pd.read_parquet(parquet_file)
            file_mtime = parquet_file.stat().st_mtime
            current_time = time.time()
            is_recent = (current_time - file_mtime) < 60
            
            if is_recent:
                df.attrs['is_live_data'] = True
                df.attrs['last_modified'] = file_mtime
            else:
                df.attrs['is_live_data'] = False
                
            df.attrs['format'] = 'parquet'
            return df
        except Exception as e:
            print(f"⚠️ Failed to load Parquet file: {e}")
    
    # Fallback to CSV.gz file
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
        
        # Check if dataframe is empty (only headers, no data rows)
        if df.empty:
            st.warning("📊 Simulation file found but contains no data yet - waiting for simulation to generate data...")
            return None
            
        if is_recent:
            # Add metadata about freshness for GUI display
            df.attrs['is_live_data'] = True
            df.attrs['last_modified'] = file_mtime
        else:
            df.attrs['is_live_data'] = False
            
        df.attrs['format'] = 'csv.gz'
        return df
    except pd.errors.EmptyDataError:
        st.warning("📊 Simulation data file is empty - waiting for simulation to generate data...")
        return None
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

def get_action_name(action_id):
    """Convert action ID to descriptive name"""
    # Substrate actions from mfc_gpu_accelerated.py
    substrate_actions = [-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    
    if isinstance(action_id, (int, float)) and 0 <= int(action_id) < len(substrate_actions):
        rate = substrate_actions[int(action_id)]
        if rate == 0.0:
            return f"#{int(action_id)}: Hold (0.0 mM/h)"
        elif rate > 0:
            return f"#{int(action_id)}: Add +{rate:.2f} mM/h"
        else:
            return f"#{int(action_id)}: Reduce {rate:.2f} mM/h"
    else:
        return f"#{int(action_id)}: Unknown"

def create_real_time_plots(df, target_conc=25.0):
    """Create real-time monitoring plots with comprehensive parameters"""
    
    # Create expanded subplots for comprehensive monitoring
    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=(
            'Substrate Concentration', 'Power & Current', 'Q-Learning Actions',
            'Biofilm Growth', 'Flow Rate & Efficiency', 'System Voltage',
            'Individual Cell Powers', 'Mixing & Control', 'Q-Values & Rewards',
            'Cumulative Energy', '', ''
        ),
        specs=[
            [{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": True}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": True}, {"secondary_y": True}],
            [{"secondary_y": False}, {}, {}]
        ]
    )
    
    # Plot 1: Substrate concentration 
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
    # Target line (dynamic based on interface setting)
    fig.add_hline(y=target_conc, line_dash="dash", line_color="green",
                  annotation_text=f"Target ({target_conc:.1f} mM)", row=1, col=1)
    
    # Add concentration error on secondary y-axis
    if 'concentration_error' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['time_hours'], y=df['concentration_error'],
                      name='Conc Error (mM)', line=dict(color='yellow', width=2, dash='dot')),
            row=1, col=1, secondary_y=True
        )
    
    # Plot 2: Power & Current (dual y-axis)
    fig.add_trace(
        go.Scatter(x=df['time_hours'], y=df['total_power'],
                  name='Power (W)', line=dict(color='orange', width=2)),
        row=1, col=2
    )
    if 'total_current' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['time_hours'], y=df['total_current'],
                      name='Current (A)', line=dict(color='red', width=2)),
            row=1, col=2, secondary_y=True
        )
    
    # Plot 3: Q-learning actions with descriptive hover text
    action_names = [get_action_name(action) for action in df['q_action']]
    fig.add_trace(
        go.Scatter(x=df['time_hours'], y=df['q_action'],
                  mode='markers', name='Actions', 
                  marker=dict(color='purple', size=6),
                  hovertemplate='<b>Time:</b> %{x:.1f}h<br>' +
                               '<b>Action:</b> %{text}<br>' +
                               '<extra></extra>',
                  text=action_names),
        row=1, col=3
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
        
        # Plot 4: Biofilm Growth
        fig.add_trace(
            go.Scatter(x=df['time_hours'], y=biofilm_avg,
                      name='Avg Thickness', line=dict(color='brown', width=2)),
            row=2, col=1
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

def create_performance_dashboard(results, target_conc=25.0):
    """Create performance metrics dashboard"""
    
    try:
        # Handle case where results might be None or not a dictionary
        if results is None or not isinstance(results, dict):
            st.warning("⚠️ No performance data available")
            return
        
        metrics = results.get('performance_metrics', {})
        
        # Ensure target_conc is a valid number
        if not isinstance(target_conc, (int, float)) or target_conc <= 0:
            target_conc = 25.0
            
    except Exception as e:
        st.error(f"❌ Error initializing performance dashboard: {str(e)}")
        st.info("Using default values...")
        metrics = {}
        target_conc = 25.0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Final Concentration",
            f"{metrics.get('final_reservoir_concentration', 0):.2f} mM",
            delta=f"{metrics.get('final_reservoir_concentration', 0) - target_conc:.2f}"
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
    
    with col5:
        # Calculate total energy from metrics
        total_energy_wh = metrics.get('total_energy_wh', 0)
        if total_energy_wh == 0:
            # Calculate from mean power and duration if not available
            mean_power = metrics.get('mean_power', 0)
            duration_hours = results.get('simulation_info', {}).get('duration_hours', 0)
            total_energy_wh = mean_power * duration_hours
        
        if total_energy_wh > 1:
            energy_display = f"{total_energy_wh:.2f} Wh"
        else:
            energy_display = f"{total_energy_wh*1000:.1f} mWh"
        
        st.metric(
            "Total Energy",
            energy_display
        )

def get_settings_file_path():
    """Get path to GUI settings file"""
    try:
        from path_config import get_simulation_data_path
        settings_dir = Path(get_simulation_data_path(""))
    except (ImportError, AttributeError, OSError):
        # Fallback to local directory
        settings_dir = Path.cwd() / "data" / "simulation_data"
    
    settings_dir.mkdir(parents=True, exist_ok=True)
    return settings_dir / "gui_settings.json"

def save_gui_settings(settings_dict):
    """Save GUI settings to file"""
    try:
        settings_file = get_settings_file_path()
        with open(settings_file, 'w') as f:
            json.dump(settings_dict, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Failed to save settings: {e}")
        return False

def load_gui_settings():
    """Load GUI settings from file"""
    try:
        settings_file = get_settings_file_path()
        if settings_file.exists():
            with open(settings_file, 'r') as f:
                settings = json.load(f)
            return settings
        return None
    except Exception as e:
        st.warning(f"Failed to load settings: {e}")
        return None

def get_current_gui_settings(duration_hours, target_conc, n_cells, anode_area_cm2, cathode_area_cm2, 
                           gpu_backend, debug_mode, use_pretrained, auto_refresh, refresh_interval):
    """Collect current GUI settings into a dictionary"""
    return {
        "duration_hours": duration_hours,
        "target_conc": target_conc,
        "n_cells": n_cells,
        "anode_area_cm2": anode_area_cm2,
        "cathode_area_cm2": cathode_area_cm2,
        "gpu_backend": gpu_backend,
        "debug_mode": debug_mode,
        "use_pretrained": use_pretrained,
        "auto_refresh": auto_refresh,
        "refresh_interval": refresh_interval,
        "timestamp": time.time()
    }

def main():
    """Main Streamlit application"""
    
    # Initialize debug mode from session state or query params
    if 'debug_mode_enabled' not in st.session_state:
        # Check for debug parameter in URL
        query_params = st.query_params
        debug_from_url = query_params.get('debug', ['false'])[0].lower() == 'true'
        st.session_state.debug_mode_enabled = debug_from_url
        
        if debug_from_url:
            debug_logger.enable_debug()
            debug_logger.log("Debug mode enabled from URL parameter", "DEBUG")
    
    # Initialize GPU acceleration once at startup
    if 'gpu_initialized' not in st.session_state:
        from gpu_acceleration import get_gpu_accelerator
        
        # Use debug mode to determine verbosity
        verbose_init = st.session_state.get('debug_mode_enabled', False)
        
        if verbose_init:
            debug_logger.log("🔍 Detecting available GPU backends...", "GPU")
        else:
            print("🔍 Detecting available GPU backends...")
            
        gpu_acc = get_gpu_accelerator(verbose=verbose_init)
        
        # Store GPU info in session state to avoid re-initialization
        st.session_state.gpu_initialized = True
        st.session_state.gpu_backend = gpu_acc.backend
        st.session_state.gpu_device_name = gpu_acc.device_info.get('device_name', 'Unknown')
        
        if verbose_init:
            debug_logger.log(f"GPU initialized: {gpu_acc.backend} - {gpu_acc.device_info.get('device_name', 'Unknown')}", "GPU")
    
    # Title and header
    st.title("🔋 MFC Simulation Control Panel")
    st.markdown("Real-time monitoring and control for Microbial Fuel Cell simulations")
    
    # Load saved settings
    saved_settings = load_gui_settings()
    use_saved_settings = False
    
    # Settings save/restore checkbox
    st.sidebar.subheader("⚙️ Settings Management")
    if saved_settings:
        saved_time = saved_settings.get('timestamp', 0)
        import datetime
        saved_date = datetime.datetime.fromtimestamp(saved_time).strftime("%Y-%m-%d %H:%M")
        use_saved_settings = st.sidebar.checkbox(
            f"📋 Use saved settings from {saved_date}", 
            value=False,
            help="Restore all parameter values from the last saved configuration"
        )
    
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
    
    # Determine default duration index based on saved settings
    default_duration_index = 1  # Default to "24 Hours (Daily)"
    if use_saved_settings and 'duration_hours' in saved_settings:
        saved_duration = saved_settings['duration_hours']
        # Find matching duration option
        for i, (name, hours) in enumerate(duration_options.items()):
            if hours == saved_duration:
                default_duration_index = i
                break
    
    selected_duration = st.sidebar.selectbox(
        "Simulation Duration",
        options=list(duration_options.keys()),
        index=default_duration_index
    )
    duration_hours = duration_options[selected_duration]
    
    # Q-learning parameters
    st.sidebar.subheader("Q-Learning Parameters")
    
    # Get default value from saved settings or use default
    default_use_pretrained = saved_settings.get('use_pretrained', True) if use_saved_settings else True
    use_pretrained = st.sidebar.checkbox("Use Pre-trained Q-table", value=default_use_pretrained)
    
    if not use_pretrained:
        st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1)
        st.sidebar.slider("Initial Epsilon", 0.1, 1.0, 0.4)
        st.sidebar.slider("Discount Factor", 0.8, 0.99, 0.95)
    
    # Target concentrations
    st.sidebar.subheader("Target Concentrations")
    # Get default value from saved settings
    default_target_conc = saved_settings.get('target_conc', 25.0) if use_saved_settings else 25.0
    target_conc = st.sidebar.number_input(
        "Target Substrate (mM)", 
        min_value=10.0, max_value=40.0, value=default_target_conc, step=0.1
    )
    
    # MFC cell configuration
    st.sidebar.subheader("MFC Configuration")
    # Get default value from saved settings
    default_n_cells = saved_settings.get('n_cells', 5) if use_saved_settings else 5
    n_cells = st.sidebar.number_input(
        "Number of Cells", 
        min_value=1, max_value=10, value=default_n_cells, step=1
    )
    
    # Separate anode and cathode electrode areas
    st.sidebar.markdown("**🔋 Working Electrodes**")
    
    # Get default values from saved settings
    default_anode_area = saved_settings.get('anode_area_cm2', 10.0) if use_saved_settings else 10.0
    default_cathode_area = saved_settings.get('cathode_area_cm2', 10.0) if use_saved_settings else 10.0
    
    anode_area_cm2 = st.sidebar.number_input(
        "Anode Area (cm²/cell)", 
        min_value=0.1, value=default_anode_area, step=0.1,
        help="Current-collecting anode area per cell - arbitrary size"
    )
    cathode_area_cm2 = st.sidebar.number_input(
        "Cathode Area (cm²/cell)", 
        min_value=0.1, value=default_cathode_area, step=0.1,
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
        # Get default values from saved settings
        gpu_backend_options = ["Auto-detect", "CUDA", "ROCm", "CPU"]
        default_gpu_backend_index = 0  # Default to "Auto-detect"
        if use_saved_settings and 'gpu_backend' in saved_settings:
            saved_backend = saved_settings['gpu_backend']
            try:
                default_gpu_backend_index = gpu_backend_options.index(saved_backend)
            except ValueError:
                pass  # Use default if saved value not found
        
        gpu_backend = st.selectbox("GPU Backend", gpu_backend_options, index=default_gpu_backend_index)
        st.slider("Save Interval (steps)", 1, 100, 10, help="Data saving is now synchronized with GUI refresh rate")
        
        # Get default debug mode from saved settings or session state
        default_debug_mode = saved_settings.get('debug_mode', False) if use_saved_settings else st.session_state.get('debug_mode_enabled', False)
        debug_mode = st.checkbox("🐛 Debug Mode", value=default_debug_mode, 
                                help="Enable verbose console output and show debug monitor. Also outputs files to temporary directory for testing")
        
        # Update debug logger when checkbox changes
        if debug_mode != st.session_state.get('debug_mode_enabled', False):
            st.session_state.debug_mode_enabled = debug_mode
            if debug_mode:
                debug_logger.enable_debug()
                debug_logger.log("Debug mode enabled from GUI", "DEBUG")
            else:
                debug_logger.log("Debug mode disabled from GUI", "DEBUG")
                debug_logger.disable_debug()
        
        st.checkbox("📧 Email Notifications", value=False, help="Feature not yet implemented")
    
    # Main content area - add debug tab if debug mode is enabled
    if debug_mode:
        tab1, tab2, tab3, tab4, debug_tab = st.tabs(["🚀 Run Simulation", "📊 Monitor", "📈 Results", "📁 History", "🐛 Debug Console"])
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["🚀 Run Simulation", "📊 Monitor", "📈 Results", "📁 History"])
    
    with tab1:
        st.header("Simulation Control")
        
        # Status display with enhanced debugging
        status = st.session_state.sim_runner.get_status()
        
        # Always show current simulation state - check both flag and thread
        if st.session_state.sim_runner.is_actually_running():
            st.info("🔄 Simulation is running...")
            if st.session_state.sim_runner.current_output_dir:
                st.text(f"Output directory: {st.session_state.sim_runner.current_output_dir}")
        else:
            # Sync the is_running flag with actual thread state
            if st.session_state.sim_runner.is_running and (not st.session_state.sim_runner.thread or not st.session_state.sim_runner.thread.is_alive()):
                st.session_state.sim_runner.is_running = False
            st.text("🔴 No simulation currently running")
        
        # Handle status messages
        if status:
            if status[0] == 'completed':
                st.success("✅ Simulation completed successfully!")
                st.session_state.simulation_results = status[1]
                st.session_state.last_output_dir = status[2]
                # Only clear running state if thread is actually finished
                if not st.session_state.sim_runner.thread or not st.session_state.sim_runner.thread.is_alive():
                    st.session_state.sim_runner.is_running = False
            elif status[0] == 'stopped':
                st.warning(f"⏹️ {status[1]}")
                st.session_state.last_output_dir = status[2]
                # Only clear running state if thread is actually finished
                if not st.session_state.sim_runner.thread or not st.session_state.sim_runner.thread.is_alive():
                    st.session_state.sim_runner.is_running = False
            elif status[0] == 'error':
                st.error(f"❌ Simulation error: {status[1]}")
                # Don't immediately stop on errors - let the thread finish cleanup
                # Only show the error but keep monitoring until thread actually stops
                if not st.session_state.sim_runner.thread or not st.session_state.sim_runner.thread.is_alive():
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
            if st.button("▶️ Start Simulation", disabled=st.session_state.sim_runner.is_actually_running()):
                # Get current refresh interval from sidebar
                current_refresh_interval = st.session_state.get('current_refresh_interval', 5.0)
                
                if st.session_state.sim_runner.start_simulation(
                    DEFAULT_QLEARNING_CONFIG, 
                    duration_hours,
                    n_cells=n_cells,
                    electrode_area_m2=electrode_area_m2,
                    target_conc=target_conc,
                    gui_refresh_interval=current_refresh_interval,
                    debug_mode=debug_mode
                ):
                    st.success(f"Started {selected_duration} simulation!")
                    st.info(f"📊 Data saving synchronized with {current_refresh_interval}s refresh rate")
                    if debug_mode:
                        st.warning("🐛 DEBUG MODE: Files will be saved to temporary directory")
                    st.rerun()
                else:
                    st.error("Simulation already running!")
        
        with col2:
            if st.button("⏹️ Stop Simulation", disabled=not st.session_state.sim_runner.is_actually_running()):
                if st.session_state.sim_runner.stop_simulation():
                    st.success("Stopping simulation...")
                    st.rerun()
                else:
                    st.error("No simulation is running")
        
        with col3:
            if st.button("🔄 Refresh Status"):
                st.rerun()
        
        # Simulation status
        if st.session_state.sim_runner.is_actually_running():
            st.markdown('<p class="status-running">🟢 Simulation Running...</p>', unsafe_allow_html=True)
            st.info("💡 Switch to the Monitor tab to see real-time updates")
        else:
            st.markdown('<p class="status-stopped">🔴 Simulation Stopped</p>', unsafe_allow_html=True)
        
        # Configuration preview
        st.subheader("Current Configuration")
        current_refresh = st.session_state.get('current_refresh_interval', 5.0)
        # Calculate timestep and save frequency using the same logic as simulation
        gui_refresh_hours = current_refresh / 3600.0
        target_steps_per_refresh = 3
        dt_hours = gui_refresh_hours / target_steps_per_refresh
        
        # Apply same constraints as in simulation
        min_dt_hours = 30.0 / 3600.0   # 30 seconds minimum
        max_dt_hours = 6.0 / 60.0      # 6 minutes maximum
        dt_hours = max(min_dt_hours, min(dt_hours, max_dt_hours))
        
        save_interval_steps = max(1, int(gui_refresh_hours / dt_hours))
        min_save_hours = 1.0 / 60.0  # 1 minute
        min_save_steps = max(1, int(min_save_hours / dt_hours))
        actual_save_steps = max(save_interval_steps, min_save_steps)
        save_frequency_hours = actual_save_steps * dt_hours
        
        config_data = {
            "Duration": f"{duration_hours:,} hours ({duration_hours/24:.1f} days)",
            "Target Concentration": f"{target_conc} mM",
            "Number of Cells": n_cells,
            "Anode Area": f"{anode_area_cm2:.1f} cm²/cell ({anode_area_cm2 * n_cells:.1f} cm² total)",
            "Cathode Area": f"{cathode_area_cm2:.1f} cm²/cell ({cathode_area_cm2 * n_cells:.1f} cm² total)",
            "Sensor Areas": "EIS: 1.0 cm², QCM: 0.196 cm² (fixed)",
            "Pre-trained Q-table": "✅ Enabled" if use_pretrained else "❌ Disabled",
            "GPU Backend": gpu_backend,
            "Simulation Timestep": f"{dt_hours*60:.1f} minutes ({dt_hours*3600:.0f} seconds)",
            "Steps per GUI Refresh": f"{target_steps_per_refresh} simulation steps", 
            "Data Save Sync": f"Every {save_frequency_hours:.2f} sim hours (GUI: {current_refresh}s)"
        }
        
        for key, value in config_data.items():
            st.text(f"{key}: {value}")
        
        # Secondary control buttons (separate row)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Use default values for auto_refresh if not defined yet
            default_auto_refresh_for_save = saved_settings.get('auto_refresh', False) if saved_settings else False
            default_refresh_interval_for_save = saved_settings.get('refresh_interval', 5) if saved_settings else 5
            
            if st.button("💾 Save Current Settings"):
                # Get current refresh interval for saving
                current_refresh_interval = st.session_state.get('current_refresh_interval', default_refresh_interval_for_save)
                # Collect current settings
                current_settings = get_current_gui_settings(
                    duration_hours, target_conc, n_cells, anode_area_cm2, cathode_area_cm2,
                    gpu_backend, debug_mode, use_pretrained, default_auto_refresh_for_save, current_refresh_interval
                )
                if save_gui_settings(current_settings):
                    st.success("✅ Settings saved! They will be available for next session.")
                    st.rerun()  # Refresh to show the saved settings checkbox

        with col2:
            if st.button("🧹 Force GPU Cleanup", key="gpu_cleanup_btn"):
                st.info("🧹 Performing manual GPU cleanup...")
                st.session_state.sim_runner._cleanup_resources()
                st.success("✅ GPU cleanup completed!")
    
    with tab2:
        st.header("Real-Time Monitoring")
        
        # Performance monitoring section  
        if st.session_state.sim_runner.is_actually_running():
            try:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                cpu_percent = process.cpu_percent()
                
                perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
                with perf_col1:
                    st.metric("Memory Usage", f"{memory_mb:.1f} MB", 
                             delta=f"+{memory_mb - st.session_state.sim_runner.start_memory_mb:.1f} MB" if hasattr(st.session_state.sim_runner, 'start_memory_mb') else None)
                with perf_col2:
                    st.metric("CPU Usage", f"{cpu_percent:.1f}%")
                with perf_col3:
                    refresh_interval = getattr(st.session_state.sim_runner, 'adaptive_refresh_interval', 5.0)
                    st.metric("Adaptive Refresh", f"{refresh_interval:.1f}s")
                with perf_col4:
                    buffer_size = len(st.session_state.sim_runner.data_buffer.snapshots) if hasattr(st.session_state.sim_runner, 'data_buffer') else 0
                    st.metric("Buffer Size", f"{buffer_size} points")
                    
            except Exception:
                pass  # Don't break UI for performance monitoring issues
        
        # GPU Status Section
        st.subheader("🖥️ GPU Status")
        
        # Create GPU status monitoring
        def get_gpu_status():
            """Get current GPU status and information."""
            try:
                # Import GPU acceleration module
                from gpu_acceleration import get_gpu_accelerator
                
                # Initialize GPU accelerator (silent to avoid console spam)
                gpu_acc = get_gpu_accelerator(verbose=False)
                
                gpu_info = {
                    'backend': gpu_acc.backend or 'cpu',
                    'available': gpu_acc.is_gpu_available(),
                    'device_count': len(gpu_acc.device_info.get('devices', [])),
                    'memory_info': gpu_acc.get_memory_info(),
                    'device_name': gpu_acc.device_info.get('name', 'Unknown'),
                    'available_backends': gpu_acc.available_backends
                }
                
                return gpu_info
                
            except Exception as e:
                return {
                    'backend': 'error',
                    'available': False,
                    'error': str(e),
                    'device_count': 0,
                    'memory_info': None,
                    'device_name': 'Error detecting GPU',
                    'available_backends': []
                }
        
        # Get and display GPU status
        gpu_status = get_gpu_status()
        
        if gpu_status['backend'] == 'error':
            st.error(f"❌ GPU Detection Error: {gpu_status['error']}")
        else:
            # GPU Status Metrics
            gpu_col1, gpu_col2, gpu_col3, gpu_col4 = st.columns(4)
            
            with gpu_col1:
                backend_display = gpu_status['backend'].upper() if gpu_status['backend'] != 'cpu' else 'CPU Only'
                if gpu_status['available']:
                    st.metric("GPU Backend", backend_display, delta="🟢 Active")
                else:
                    st.metric("GPU Backend", backend_display, delta="⚪ CPU Mode")
            
            with gpu_col2:
                device_name = gpu_status['device_name']
                if len(device_name) > 15:
                    device_name = device_name[:12] + "..."
                st.metric("Device", device_name)
            
            with gpu_col3:
                if gpu_status['available'] and gpu_status['device_count'] > 0:
                    st.metric("GPU Count", f"{gpu_status['device_count']}")
                else:
                    st.metric("GPU Count", "0", delta="No GPU")
            
            with gpu_col4:
                available_backends = ', '.join(gpu_status['available_backends']) if gpu_status['available_backends'] else 'None'
                if len(available_backends) > 12:
                    available_backends = available_backends[:9] + "..."
                st.metric("Available", available_backends)
            
            # GPU Memory Information (if available)
            if gpu_status['memory_info']:
                st.subheader("📊 GPU Memory Usage")
                mem_info = gpu_status['memory_info']
                
                total_mb = mem_info['total'] // (1024**2)
                used_mb = mem_info['used'] // (1024**2)
                free_mb = mem_info['free'] // (1024**2)
                usage_percent = (used_mb / total_mb * 100) if total_mb > 0 else 0
                
                mem_col1, mem_col2, mem_col3, mem_col4 = st.columns(4)
                
                with mem_col1:
                    st.metric("Total Memory", f"{total_mb:,} MB")
                with mem_col2:
                    st.metric("Used Memory", f"{used_mb:,} MB", 
                             delta=f"{usage_percent:.1f}% utilized")
                with mem_col3:
                    st.metric("Free Memory", f"{free_mb:,} MB")
                with mem_col4:
                    # Memory status indicator
                    if usage_percent < 50:
                        status = "🟢 Good"
                    elif usage_percent < 80:
                        status = "🟡 Moderate"
                    else:
                        status = "🔴 High"
                    st.metric("Memory Status", status)
                
                # Memory usage bar
                st.progress(usage_percent / 100, text=f"GPU Memory: {usage_percent:.1f}% ({used_mb:,}/{total_mb:,} MB)")
                
            else:
                if gpu_status['available']:
                    st.info("ℹ️ GPU memory information not available for this backend")
                else:
                    st.info("ℹ️ No GPU detected - running in CPU mode")
            
            # GPU Backend Details
            with st.expander("🔧 GPU Backend Details"):
                st.write(f"**Active Backend:** {gpu_status['backend']}")
                st.write(f"**GPU Available:** {'Yes' if gpu_status['available'] else 'No'}")
                st.write(f"**Available Backends:** {', '.join(gpu_status['available_backends']) if gpu_status['available_backends'] else 'None detected'}")
                st.write(f"**Device Name:** {gpu_status['device_name']}")
                st.write(f"**Device Count:** {gpu_status['device_count']}")
                
                if gpu_status['memory_info']:
                    st.write("**Memory Details:**")
                    st.json(gpu_status['memory_info'])
        
        st.divider()
        
        # Auto-refresh controls
        col1, col2, col3 = st.columns([2, 1, 3])
        
        with col1:
            # Get default auto_refresh from saved settings
            default_auto_refresh = saved_settings.get('auto_refresh', False) if use_saved_settings else False
            auto_refresh = st.checkbox("Enable Auto-refresh", value=default_auto_refresh)
        
        with col2:
            # Get default refresh_interval from saved settings
            default_refresh_interval = saved_settings.get('refresh_interval', 5) if use_saved_settings else 5
            refresh_interval = st.number_input(
                "Interval (s)", 
                min_value=1, 
                max_value=60, 
                value=default_refresh_interval, 
                step=1,
                disabled=not auto_refresh,
                key="refresh_interval_input"
            )
            # Store refresh interval in session state for simulation sync
            st.session_state.current_refresh_interval = refresh_interval
        
        with col3:
            if auto_refresh:
                st.info(f"🔄 Auto-refreshing every {refresh_interval} seconds")
                if st.session_state.sim_runner.is_actually_running():
                    st.success("📊 Data sync enabled with simulation")
        
        # Add save settings button in Monitor tab where auto_refresh is defined
        if st.button("💾 Save Current Settings (including refresh settings)", key="save_settings_monitor"):
            # Collect current settings with actual auto_refresh and refresh_interval values
            current_settings = get_current_gui_settings(
                duration_hours, target_conc, n_cells, anode_area_cm2, cathode_area_cm2,
                gpu_backend, debug_mode, use_pretrained, auto_refresh, refresh_interval
            )
            if save_gui_settings(current_settings):
                st.success("✅ Settings saved! They will be available for next session.")
                st.rerun()  # Refresh to show the saved settings checkbox
                
        # Manual refresh button for immediate updates
        if st.button("🔄 Manual Refresh", key="manual_refresh_monitor"):
            st.rerun()
        
        # Show auto-refresh status and implement using streamlit-autorefresh if available
        if auto_refresh:
            try:
                from streamlit_autorefresh import st_autorefresh
                # Use streamlit-autorefresh for seamless updates without tab switching
                count = st_autorefresh(interval=refresh_interval * 1000, key="data_refresh")
                st.success(f"🔄 Auto-refresh enabled (#{count}) - data updates every {refresh_interval}s")
            except ImportError:
                # Fallback to manual refresh instructions
                st.success("🔄 Auto-refresh enabled - click 'Manual Refresh' to see latest data")
                st.info("💡 Install streamlit-autorefresh for automatic updates: pip install streamlit-autorefresh")
        else:
            st.info("🔄 Auto-refresh disabled - click 'Manual Refresh' to update data")
        
        # Check if simulation is running and show live data
        if st.session_state.sim_runner.is_actually_running():
            st.subheader("🟢 Live Simulation Data")
            
            # Load current simulation data from memory (no file I/O race conditions)
            df = load_simulation_data_from_memory(st.session_state.sim_runner)
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
            fig = create_real_time_plots(df, target_conc)
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
                    action_id = int(latest['q_action'])
                    action_description = get_action_name(action_id)
                    st.metric("Current Action", action_description)
        else:
            if not st.session_state.sim_runner.is_actually_running():
                st.info("No recent simulations found. Start a simulation to see real-time monitoring.")
            else:
                st.info("Waiting for simulation to generate data...")
    
    with tab3:
        st.header("Simulation Results")
        
        if st.session_state.simulation_results:
            results = st.session_state.simulation_results
            
            # Performance dashboard
            st.subheader("Performance Metrics")
            try:
                # Ensure target_conc is available, use default if not
                dashboard_target_conc = target_conc if 'target_conc' in locals() else 25.0
                create_performance_dashboard(results, dashboard_target_conc)
            except Exception as e:
                st.error(f"❌ Error creating performance dashboard: {str(e)}")
                st.info("This might be due to incomplete simulation data. Try running a new simulation.")
            
            # Detailed results
            st.subheader("Detailed Results")
            
            col1, col2 = st.columns(2)
            
            try:
                with col1:
                    st.text("Performance Metrics")
                    if isinstance(results, dict) and 'performance_metrics' in results:
                        st.json(results.get('performance_metrics', {}))
                    else:
                        st.info("No performance metrics available")
                
                with col2:
                    st.text("Simulation Info")
                    if isinstance(results, dict) and 'simulation_info' in results:
                        st.json(results.get('simulation_info', {}))
                    else:
                        st.info("No simulation info available")
            except Exception as e:
                st.error(f"❌ Error displaying detailed results: {str(e)}")
                st.info("Raw results data:")
                try:
                    st.write(results)
                except Exception:
                    st.write(f"Results type: {type(results)}")
                    st.write(f"Results content: {str(results)[:200]}...")
                
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
                    # Try to get target concentration from simulation data or use current interface setting
                    historical_target = target_conc  # Use current interface setting as best estimate
                    fig = create_real_time_plots(df, historical_target)
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
    
    # Debug Console Tab (only shown when debug mode is enabled)
    if debug_mode and 'debug_tab' in locals():
        with debug_tab:
            st.header("🐛 Debug Console")
            st.markdown("Real-time monitoring of system events, GPU operations, and simulation diagnostics")
            
            # Debug control buttons
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("🔄 Refresh", help="Refresh debug messages"):
                    st.rerun()
                    
            with col2:
                if st.button("🗑️ Clear Log", help="Clear all debug messages"):
                    debug_logger.clear_messages()
                    st.rerun()
                    
            with col3:
                auto_refresh_debug = st.checkbox("🔄 Auto-refresh", value=False, 
                                               help="Automatically refresh debug console every 2 seconds")
                
            with col4:
                show_level_filter = st.selectbox("📊 Filter Level", 
                                               ["ALL", "DEBUG", "INFO", "GPU", "ERROR"], 
                                               help="Filter messages by level")
            
            # Get debug messages
            messages = debug_logger.get_messages()
            
            # Filter messages if needed
            if show_level_filter != "ALL":
                filtered_messages = []
                for msg in messages:
                    if f"] {show_level_filter}:" in msg:
                        filtered_messages.append(msg)
                messages = filtered_messages
            
            # Debug console display
            if messages:
                # Create scrollable debug console
                console_text = "\n".join(messages[-100:])  # Show last 100 messages
                
                # Create a text area with console styling
                st.text_area(
                    label="Debug Messages",
                    value=console_text,
                    height=400,
                    help=f"Showing last {len(messages)} debug messages. Auto-scroll to bottom.",
                    key="debug_console",
                    label_visibility="collapsed"
                )
                
                # System information panel
                st.subheader("📊 System Information")
                
                sys_col1, sys_col2, sys_col3 = st.columns(3)
                
                with sys_col1:
                    st.metric("GPU Backend", st.session_state.get('gpu_backend', 'Unknown'))
                    st.metric("GPU Device", st.session_state.get('gpu_device_name', 'Unknown'))
                    
                with sys_col2:
                    # Memory usage
                    memory_usage = psutil.virtual_memory()
                    st.metric("RAM Usage", f"{memory_usage.percent:.1f}%")
                    st.metric("Available RAM", f"{memory_usage.available / (1024**3):.1f} GB")
                    
                with sys_col3:
                    # Simulation status
                    sim_status = "Running" if st.session_state.sim_runner.is_actually_running() else "Stopped"
                    st.metric("Simulation Status", sim_status)
                    st.metric("Debug Messages", len(debug_logger.messages))
                
                # GPU Memory monitoring (if available)
                if st.session_state.get('gpu_backend') == 'rocm':
                    st.subheader("🎮 GPU Memory Status")
                    try:
                        from gpu_acceleration import get_gpu_accelerator
                        gpu_acc = get_gpu_accelerator(verbose=False)
                        mem_info = gpu_acc.get_memory_info()
                        
                        if mem_info:
                            gpu_col1, gpu_col2, gpu_col3 = st.columns(3)
                            with gpu_col1:
                                st.metric("GPU Memory Used", f"{mem_info['used'] / (1024**2):.0f} MB")
                            with gpu_col2:
                                st.metric("GPU Memory Free", f"{mem_info['free'] / (1024**2):.0f} MB")
                            with gpu_col3:
                                usage_percent = (mem_info['used'] / mem_info['total']) * 100 if mem_info['total'] > 0 else 0
                                st.metric("GPU Memory Usage", f"{usage_percent:.1f}%")
                    except Exception as e:
                        st.warning(f"Could not retrieve GPU memory info: {e}")
            else:
                st.info("🔇 No debug messages yet. Enable debug mode and perform some operations to see logs here.")
                
                # Show sample debug message
                if st.button("🧪 Test Debug Message"):
                    debug_logger.log("This is a test debug message from the GUI", "DEBUG")
                    st.rerun()
            
            # Auto-refresh functionality
            if auto_refresh_debug and messages:
                time.sleep(2)
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("🔬 MFC Simulation Control Panel | Built with Streamlit")
    
    # Cleanup on app close
    if st.session_state.sim_runner.is_actually_running():
        st.sidebar.warning("⚠️ Simulation running - will cleanup on stop")

def cleanup_on_exit():
    """Cleanup function to be called when app exits"""
    try:
        if 'sim_runner' in st.session_state:
            if st.session_state.sim_runner.is_actually_running():
                st.session_state.sim_runner.stop_simulation()
    except Exception:
        pass

if __name__ == "__main__":
    import atexit
    atexit.register(cleanup_on_exit)
    main()