#!/usr/bin/env python3
"""Streamlit GUI for MFC Simulation Control and Monitoring.

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

MODULE DECOMPOSITION:
======================
This main entry point orchestrates page modules located in:
- gui/simulation_runner.py: Thread-safe simulation runner class
- gui/plots/realtime_plots.py: Real-time monitoring visualizations
- gui/plots/analysis_plots.py: Advanced analysis visualizations
- gui/data_loaders.py: Data loading utilities
"""

from __future__ import annotations

import os
import sys
import time

import pandas as pd
import streamlit as st

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.qlearning_config import DEFAULT_QLEARNING_CONFIG
from gui.data_loaders import load_recent_simulations, load_simulation_data
from gui.plots import create_performance_dashboard, create_real_time_plots
from gui.simulation_runner import SimulationRunner

# Set page config
st.set_page_config(
    page_title="MFC Simulation Control Panel",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)


# Initialize session state
if "sim_runner" not in st.session_state:
    st.session_state.sim_runner = SimulationRunner()
if "simulation_results" not in st.session_state:
    st.session_state.simulation_results = None
if "last_update" not in st.session_state:
    st.session_state.last_update = None
if "last_output_dir" not in st.session_state:
    st.session_state.last_output_dir = None


def render_sidebar():
    """Render the sidebar with simulation parameters."""
    st.sidebar.header("Simulation Parameters")

    # Simulation duration
    duration_options = {
        "1 Hour (Quick Test)": 1,
        "24 Hours (Daily)": 24,
        "1 Week": 168,
        "1 Month": 720,
        "1 Year": 8784,
    }

    selected_duration = st.sidebar.selectbox(
        "Simulation Duration",
        options=list(duration_options.keys()),
        index=1,
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
        min_value=10.0,
        max_value=40.0,
        value=25.0,
        step=0.1,
    )

    # MFC cell configuration
    st.sidebar.subheader("MFC Configuration")
    n_cells = st.sidebar.number_input(
        "Number of Cells",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
    )

    # Separate anode and cathode electrode areas
    st.sidebar.markdown("**Working Electrodes**")

    anode_area_cm2 = st.sidebar.number_input(
        "Anode Area (cm2/cell)",
        min_value=0.1,
        value=10.0,
        step=0.1,
        help="Current-collecting anode area per cell - arbitrary size",
    )
    cathode_area_cm2 = st.sidebar.number_input(
        "Cathode Area (cm2/cell)",
        min_value=0.1,
        value=10.0,
        step=0.1,
        help="Cathode area per cell (can differ from anode) - arbitrary size",
    )

    # Convert to m2 for internal use
    anode_area_m2 = anode_area_cm2 * 1e-4

    # Show sensor areas (fixed for optimal sensing)
    st.sidebar.markdown("**Sensor Electrodes (Fixed)**")
    st.sidebar.text("EIS sensor: 1.0 cm2 (impedance sensing)")
    st.sidebar.text("QCM sensor: 0.196 cm2 (mass sensing)")

    # Legacy compatibility
    electrode_area_m2 = anode_area_m2

    # Advanced settings
    with st.sidebar.expander("Advanced Settings"):
        gpu_backend = st.selectbox(
            "GPU Backend",
            ["Auto-detect", "CUDA", "ROCm", "CPU"],
        )
        st.slider(
            "Save Interval (steps)",
            1,
            100,
            10,
            help="Data saving is now synchronized with GUI refresh rate",
        )
        st.checkbox(
            "Email Notifications",
            value=False,
            help="Feature not yet implemented",
        )

    return {
        "duration_hours": duration_hours,
        "selected_duration": selected_duration,
        "use_pretrained": use_pretrained,
        "target_conc": target_conc,
        "n_cells": n_cells,
        "anode_area_cm2": anode_area_cm2,
        "cathode_area_cm2": cathode_area_cm2,
        "electrode_area_m2": electrode_area_m2,
        "gpu_backend": gpu_backend,
    }


def render_simulation_tab(params):
    """Render the simulation control tab."""
    st.header("Simulation Control")

    # Status display
    status = st.session_state.sim_runner.get_status()
    if status:
        if status[0] == "completed":
            st.success("Simulation completed successfully!")
            st.session_state.simulation_results = status[1]
            st.session_state.last_output_dir = status[2]
        elif status[0] == "stopped":
            st.warning(f"{status[1]}")
            st.session_state.last_output_dir = status[2]
        elif status[0] == "error":
            st.error(f"Simulation failed: {status[1]}")

    # Control buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(
            "Start Simulation",
            disabled=st.session_state.sim_runner.is_running,
        ):
            current_refresh_interval = st.session_state.get(
                "current_refresh_interval",
                5.0,
            )

            if st.session_state.sim_runner.start_simulation(
                DEFAULT_QLEARNING_CONFIG,
                params["duration_hours"],
                n_cells=params["n_cells"],
                electrode_area_m2=params["electrode_area_m2"],
                target_conc=params["target_conc"],
                gui_refresh_interval=current_refresh_interval,
            ):
                st.success(f"Started {params['selected_duration']} simulation!")
                st.info(
                    f"Data saving synchronized with {current_refresh_interval}s refresh rate",
                )
                st.rerun()
            else:
                st.error("Simulation already running!")

    with col2:
        if st.button(
            "Stop Simulation",
            disabled=not st.session_state.sim_runner.is_running,
        ):
            if st.session_state.sim_runner.stop_simulation():
                st.success("Stopping simulation...")
                st.rerun()
            else:
                st.error("No simulation is running")

    with col3:
        if st.button("Refresh Status"):
            st.rerun()

    # Simulation status
    if st.session_state.sim_runner.is_running:
        st.markdown(
            '<p class="status-running">Simulation Running...</p>',
            unsafe_allow_html=True,
        )
        st.info("Switch to the Monitor tab to see real-time updates")
    else:
        st.markdown(
            '<p class="status-stopped">Simulation Stopped</p>',
            unsafe_allow_html=True,
        )

    # Configuration preview
    st.subheader("Current Configuration")
    current_refresh = st.session_state.get("current_refresh_interval", 5.0)
    gui_refresh_hours = current_refresh / 3600.0
    min_save_steps = 30
    max_save_steps = 100
    calculated_steps = max(1, int(gui_refresh_hours / 0.1))
    actual_save_steps = max(min_save_steps, min(calculated_steps, max_save_steps))
    save_frequency_hours = actual_save_steps * 0.1

    config_data = {
        "Duration": f"{params['duration_hours']:,} hours ({params['duration_hours'] / 24:.1f} days)",
        "Target Concentration": f"{params['target_conc']} mM",
        "Number of Cells": params["n_cells"],
        "Anode Area": f"{params['anode_area_cm2']:.1f} cm2/cell ({params['anode_area_cm2'] * params['n_cells']:.1f} cm2 total)",
        "Cathode Area": f"{params['cathode_area_cm2']:.1f} cm2/cell ({params['cathode_area_cm2'] * params['n_cells']:.1f} cm2 total)",
        "Sensor Areas": "EIS: 1.0 cm2, QCM: 0.196 cm2 (fixed)",
        "Pre-trained Q-table": "Enabled" if params["use_pretrained"] else "Disabled",
        "GPU Backend": params["gpu_backend"],
        "Data Save Sync": f"Every {save_frequency_hours:.2f} sim hours (GUI: {current_refresh}s)",
    }

    for key, value in config_data.items():
        st.text(f"{key}: {value}")


def render_monitor_tab():
    """Render the real-time monitoring tab."""
    st.header("Real-Time Monitoring")

    # Auto-refresh controls
    col1, col2, col3 = st.columns([2, 1, 3])

    with col1:
        auto_refresh = st.checkbox("Enable Auto-refresh", value=True)

    with col2:
        refresh_interval = st.number_input(
            "Interval (s)",
            min_value=1,
            max_value=60,
            value=5,
            step=1,
            disabled=not auto_refresh,
            key="refresh_interval_input",
        )
        st.session_state.current_refresh_interval = refresh_interval

    with col3:
        if auto_refresh:
            st.info(f"Auto-refreshing every {refresh_interval} seconds")
            if st.session_state.sim_runner.is_running:
                st.success("Data sync enabled with simulation")

    # Implement auto-refresh
    if auto_refresh:
        current_time = time.time()
        if "last_refresh_time" not in st.session_state:
            st.session_state.last_refresh_time = current_time

        if current_time - st.session_state.last_refresh_time >= refresh_interval:
            st.session_state.last_refresh_time = current_time
            st.rerun()

    # Check if simulation is running and show live data
    if (
        st.session_state.sim_runner.is_running
        and st.session_state.sim_runner.current_output_dir
    ):
        st.subheader("Live Simulation Data")

        new_data = st.session_state.sim_runner.get_live_data()
        df = st.session_state.sim_runner.get_buffered_data()

        if df is not None and len(df) > 0:
            actual_hours = (
                df["time_hours"].iloc[-1] if "time_hours" in df.columns else 0
            )
            st.info(
                f"Simulation running: {actual_hours:.1f} hours elapsed, {len(df)} data points",
            )
            st.success(
                f"Memory streaming: {len(new_data)} new points this refresh",
            )
        else:
            st.info("Waiting for simulation data...")
            df = None
    else:
        # Load most recent simulation data
        recent_sims = load_recent_simulations()
        if recent_sims:
            latest_sim = recent_sims[0]
            st.subheader(f"Latest Simulation: {latest_sim['name']}")
            df = load_simulation_data(latest_sim["path"])
        else:
            df = None

    if df is not None:
        # Smart plot updates
        if st.session_state.sim_runner.should_update_plots():
            fig = create_real_time_plots(df)
            st.session_state["cached_plot"] = fig
            st.plotly_chart(fig, use_container_width=True, key="monitor_plots")
        else:
            cached_plot = st.session_state.get("cached_plot")
            if cached_plot is not None:
                st.plotly_chart(
                    cached_plot,
                    use_container_width=True,
                    key="monitor_plots",
                )
            else:
                fig = create_real_time_plots(df)
                st.session_state["cached_plot"] = fig
                st.plotly_chart(fig, use_container_width=True, key="monitor_plots")

        # Current status metrics
        if len(df) > 0:
            latest = df.iloc[-1]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Time", f"{latest['time_hours']:.1f} h")
            with col2:
                st.metric(
                    "Reservoir Conc",
                    f"{latest['reservoir_concentration']:.2f} mM",
                )
            with col3:
                st.metric("Power Output", f"{latest['total_power']:.3f} W")
            with col4:
                st.metric("Current Action", int(latest["q_action"]))
    elif not st.session_state.sim_runner.is_running:
        st.info(
            "No recent simulations found. Start a simulation to see real-time monitoring.",
        )
    else:
        st.info("Waiting for simulation to generate data...")


def render_results_tab():
    """Render the simulation results tab."""
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
            st.json(results.get("performance_metrics", {}))

        with col2:
            st.json(results.get("simulation_info", {}))

    else:
        st.info("No simulation results available. Run a simulation first.")


def render_history_tab():
    """Render the simulation history tab."""
    st.header("Simulation History")

    recent_sims = load_recent_simulations()

    if recent_sims:
        # Create summary table
        df_history = pd.DataFrame(recent_sims)

        # Display table with metrics
        st.dataframe(
            df_history[
                ["name", "duration", "final_conc", "control_effectiveness"]
            ].rename(
                columns={
                    "name": "Simulation",
                    "duration": "Duration (h)",
                    "final_conc": "Final Conc (mM)",
                    "control_effectiveness": "Control Eff (%)",
                },
            ),
            use_container_width=True,
        )

        # Selection for detailed view
        selected_sim = st.selectbox(
            "Select simulation for detailed view:",
            options=recent_sims,
            format_func=lambda x: f"{x['name']} - {x['duration']}h",
        )

        if selected_sim:
            df = load_simulation_data(selected_sim["path"])
            if df is not None:
                st.subheader(f"Detailed View: {selected_sim['name']}")
                fig = create_real_time_plots(df)
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=f"history_plot_{selected_sim['name']}",
                )

                # Download option
                st.download_button(
                    label="Download CSV Data",
                    data=df.to_csv(index=False),
                    file_name=f"{selected_sim['name']}_data.csv",
                    mime="text/csv",
                )
    else:
        st.info("No simulation history found.")


def main() -> None:
    """Main Streamlit application."""
    # Title and header
    st.title("MFC Simulation Control Panel")
    st.markdown("Real-time monitoring and control for Microbial Fuel Cell simulations")

    # Render sidebar and get parameters
    params = render_sidebar()

    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Run Simulation", "Monitor", "Results", "History"],
    )

    with tab1:
        render_simulation_tab(params)

    with tab2:
        render_monitor_tab()

    with tab3:
        render_results_tab()

    with tab4:
        render_history_tab()

    # Footer
    st.markdown("---")
    st.markdown("MFC Simulation Control Panel | Built with Streamlit")

    # Cleanup on app close
    if st.session_state.sim_runner.is_running:
        st.sidebar.warning("Simulation running - will cleanup on stop")


def cleanup_on_exit() -> None:
    """Cleanup function to be called when app exits."""
    try:
        if "sim_runner" in st.session_state:
            if st.session_state.sim_runner.is_running:
                st.session_state.sim_runner.stop_simulation()
    except Exception:
        pass


if __name__ == "__main__":
    import atexit

    atexit.register(cleanup_on_exit)
    main()
