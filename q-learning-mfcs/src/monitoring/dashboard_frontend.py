#!/usr/bin/env python3
"""
Streamlit Dashboard Frontend for MFC Monitoring System with HTTPS Support
Provides secure web interface for simulation monitoring and control.
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from monitoring.ssl_config import (
        SecurityHeaders,
        SSLConfig,
        initialize_ssl_infrastructure,
        load_ssl_config,
        test_ssl_connection,
    )
except ImportError as e:
    st.error(f"Failed to import SSL configuration: {e}")
    st.stop()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global SSL configuration
@st.cache_resource
def get_ssl_config() -> SSLConfig:
    """Load and cache SSL configuration"""
    return load_ssl_config()

@st.cache_resource
def setup_https_session() -> requests.Session:
    """Setup HTTPS session with SSL configuration and security"""
    session = requests.Session()

    # Retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # SSL configuration
    ssl_config = get_ssl_config()
    if ssl_config:
        # For self-signed certificates, disable SSL verification in development
        if ssl_config.domain == "localhost":
            session.verify = False
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        else:
            # Use proper certificate verification in production
            session.verify = True

    return session

# Streamlit page configuration with security
st.set_page_config(
    page_title="MFC Monitoring Dashboard",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,  # Disable for security
        'Report a bug': None,  # Disable for security
        'About': "MFC Monitoring Dashboard v1.2.0 - Secure HTTPS Version"
    }
)

# Security: Hide Streamlit menu and footer
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Security indicator */
.ssl-indicator {
    position: fixed;
    top: 10px;
    right: 10px;
    background: #28a745;
    color: white;
    padding: 5px 10px;
    border-radius: 15px;
    font-size: 12px;
    z-index: 1000;
}

/* Custom styling for secure interface */
.metric-card {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.status-secure { 
    color: #28a745; 
    font-weight: bold;
}

.status-running { 
    color: #17a2b8; 
    font-weight: bold;
}

.status-stopped { 
    color: #dc3545; 
    font-weight: bold;
}

.alert-warning {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    color: #856404;
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}

.alert-success {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

class APIClient:
    """HTTPS API client for secure communication with backend"""

    def __init__(self):
        self.ssl_config = get_ssl_config()
        self.session = setup_https_session()
        self.base_url = self._get_api_base_url()

    def _get_api_base_url(self) -> str:
        """Get API base URL with HTTPS"""
        if self.ssl_config:
            protocol = "https"
            port = self.ssl_config.https_port_api
            host = self.ssl_config.domain
        else:
            protocol = "http"  # Fallback for development
            port = 8000
            host = "localhost"

        return f"{protocol}://{host}:{port}"

    def test_connection(self) -> Tuple[bool, str]:
        """Test connection to API server"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            response.raise_for_status()

            health_data = response.json()
            ssl_enabled = health_data.get("ssl_config", {}).get("enabled", False)

            return True, f"✅ Connected (SSL: {'Enabled' if ssl_enabled else 'Disabled'})"

        except requests.exceptions.SSLError as e:
            return False, f"❌ SSL Connection Failed: {str(e)}"
        except requests.exceptions.ConnectionError as e:
            return False, f"❌ Connection Failed: {str(e)}"
        except requests.exceptions.Timeout:
            return False, "❌ Connection Timeout"
        except Exception as e:
            return False, f"❌ Error: {str(e)}"

    def get_simulation_status(self) -> Optional[Dict]:
        """Get simulation status from API"""
        try:
            response = self.session.get(f"{self.base_url}/simulation/status", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get simulation status: {e}")
            return None

    def start_simulation(self, config: Dict) -> Tuple[bool, str]:
        """Start simulation via API"""
        try:
            response = self.session.post(
                f"{self.base_url}/simulation/start",
                json=config,
                timeout=30
            )
            response.raise_for_status()
            return True, "Simulation started successfully"
        except Exception as e:
            return False, f"Failed to start simulation: {str(e)}"

    def stop_simulation(self) -> Tuple[bool, str]:
        """Stop simulation via API"""
        try:
            response = self.session.post(f"{self.base_url}/simulation/stop", timeout=30)
            response.raise_for_status()
            return True, "Simulation stopped successfully"
        except Exception as e:
            return False, f"Failed to stop simulation: {str(e)}"

    def get_latest_data(self, limit: int = 100) -> Optional[List[Dict]]:
        """Get latest simulation data"""
        try:
            response = self.session.get(
                f"{self.base_url}/data/latest",
                params={"limit": limit},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get latest data: {e}")
            return None

    def get_performance_metrics(self) -> Optional[Dict]:
        """Get performance metrics"""
        try:
            response = self.session.get(f"{self.base_url}/metrics/performance", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return None

@st.cache_data(ttl=60)  # Cache for 1 minute
def get_system_info(api_client: APIClient) -> Optional[Dict]:
    """Get system information with caching"""
    try:
        response = api_client.session.get(f"{api_client.base_url}/system/info", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        return None

def create_real_time_plots(data: List[Dict]) -> Optional[go.Figure]:
    """Create real-time monitoring plots"""
    if not data:
        return None

    # Convert to DataFrame for easier processing
    df = pd.DataFrame(data)

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Substrate Concentration', 'Power Output',
            'Q-Learning Actions', 'Biofilm Growth'
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}]
        ]
    )

    # Substrate concentration
    fig.add_trace(
        go.Scatter(
            x=df['time_hours'],
            y=df['reservoir_concentration'],
            name='Reservoir',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df['time_hours'],
            y=df['outlet_concentration'],
            name='Outlet',
            line=dict(color='red', width=2)
        ),
        row=1, col=1
    )

    # Target line
    fig.add_hline(y=25.0, line_dash="dash", line_color="green", row=1, col=1)

    # Power output
    fig.add_trace(
        go.Scatter(
            x=df['time_hours'],
            y=df['total_power'],
            name='Power',
            line=dict(color='orange', width=2)
        ),
        row=1, col=2
    )

    # Q-learning actions
    fig.add_trace(
        go.Scatter(
            x=df['time_hours'],
            y=df['q_action'],
            mode='markers',
            name='Actions',
            marker=dict(color='purple', size=4)
        ),
        row=2, col=1
    )

    # Biofilm thickness (average)
    biofilm_avg = []
    for thicknesses in df['biofilm_thicknesses']:
        if isinstance(thicknesses, list) and thicknesses:
            biofilm_avg.append(np.mean(thicknesses))
        else:
            biofilm_avg.append(10.0)  # Default

    fig.add_trace(
        go.Scatter(
            x=df['time_hours'],
            y=biofilm_avg,
            name='Avg Thickness',
            line=dict(color='brown', width=2)
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="Secure MFC Monitoring Dashboard - Real-Time Data"
    )

    # Update axes labels
    fig.update_yaxes(title_text="Concentration (mM)", row=1, col=1)
    fig.update_yaxes(title_text="Power (W)", row=1, col=2)
    fig.update_yaxes(title_text="Action ID", row=2, col=1)
    fig.update_yaxes(title_text="Thickness (μm)", row=2, col=2)
    fig.update_xaxes(title_text="Time (hours)", row=2, col=1)
    fig.update_xaxes(title_text="Time (hours)", row=2, col=2)

    return fig

def show_ssl_status(ssl_config: SSLConfig, api_client: APIClient):
    """Display SSL status and security information"""
    st.markdown("### 🔒 Security Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**SSL Configuration**")
        if ssl_config:
            st.markdown('<div class="status-secure">✅ HTTPS Enabled</div>', unsafe_allow_html=True)
            st.text(f"Domain: {ssl_config.domain}")
            st.text(f"Port: {ssl_config.https_port_frontend}")
        else:
            st.markdown('<div class="status-stopped">❌ HTTP Only</div>', unsafe_allow_html=True)
            st.warning("Running in HTTP mode. Enable HTTPS for production.")

    with col2:
        st.markdown("**API Connection**")
        connected, status_msg = api_client.test_connection()
        if connected:
            st.markdown(f'<div class="status-secure">{status_msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="status-stopped">{status_msg}</div>', unsafe_allow_html=True)

    with col3:
        st.markdown("**Security Features**")
        if ssl_config:
            st.text("✅ HSTS Headers" if ssl_config.enable_hsts else "❌ HSTS Disabled")
            st.text("✅ CSP Headers" if ssl_config.enable_csp else "❌ CSP Disabled")
            st.text("✅ Secure Cookies")
        else:
            st.text("❌ Security headers disabled")

def main():
    """Main Streamlit application with HTTPS support"""

    # Initialize SSL configuration and API client
    ssl_config = get_ssl_config()
    api_client = APIClient()

    # SSL status indicator
    if ssl_config and ssl_config.domain != "localhost":
        st.markdown(
            '<div class="ssl-indicator">🔒 HTTPS Secure</div>',
            unsafe_allow_html=True
        )

    # Main title
    st.title("🔋 MFC Monitoring Dashboard")
    if ssl_config:
        st.markdown("**Secure HTTPS Interface** | Real-time monitoring and control")
    else:
        st.markdown("**Development Mode** | Enable HTTPS for production")

    # Sidebar
    st.sidebar.header("🔧 Dashboard Controls")

    # SSL and security status
    with st.sidebar.expander("🔒 Security Status", expanded=False):
        show_ssl_status(ssl_config, api_client)

    # System information
    with st.sidebar.expander("ℹ️ System Information", expanded=False):
        system_info = get_system_info(api_client)
        if system_info:
            st.json(system_info)
        else:
            st.error("Could not retrieve system information")

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚀 Simulation Control",
        "📊 Real-Time Monitor",
        "📈 Performance Metrics",
        "⚙️ Configuration"
    ])

    with tab1:
        st.header("Simulation Control")

        # Connection status
        connected, status_msg = api_client.test_connection()
        if connected:
            st.success(status_msg)
        else:
            st.error(status_msg)
            st.stop()

        # Simulation parameters
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Simulation Parameters")
            duration_hours = st.number_input(
                "Duration (hours)",
                min_value=1,
                max_value=8760,
                value=24,
                step=1
            )
            n_cells = st.number_input(
                "Number of Cells",
                min_value=1,
                max_value=20,
                value=5,
                step=1
            )
            electrode_area = st.number_input(
                "Electrode Area (cm²/cell)",
                min_value=0.1,
                value=10.0,
                step=0.1
            )
            target_conc = st.number_input(
                "Target Concentration (mM)",
                min_value=10.0,
                max_value=40.0,
                value=25.0,
                step=0.1
            )

        with col2:
            st.subheader("Control Parameters")
            use_pretrained = st.checkbox("Use Pre-trained Q-table", value=True)

            if not use_pretrained:
                learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1)
                epsilon_initial = st.slider("Initial Epsilon", 0.1, 1.0, 0.4)
                discount_factor = st.slider("Discount Factor", 0.8, 0.99, 0.95)

        # Control buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("▶️ Start Simulation", type="primary"):
                config = {
                    "duration_hours": duration_hours,
                    "n_cells": n_cells,
                    "electrode_area_m2": electrode_area * 1e-4,  # Convert to m²
                    "target_concentration": target_conc,
                    "use_pretrained": use_pretrained
                }

                if not use_pretrained:
                    config.update({
                        "learning_rate": learning_rate,
                        "epsilon_initial": epsilon_initial,
                        "discount_factor": discount_factor
                    })

                success, message = api_client.start_simulation(config)
                if success:
                    st.success(message)
                    st.experimental_rerun()
                else:
                    st.error(message)

        with col2:
            if st.button("⏹️ Stop Simulation", type="secondary"):
                success, message = api_client.stop_simulation()
                if success:
                    st.success(message)
                    st.experimental_rerun()
                else:
                    st.error(message)

        with col3:
            if st.button("🔄 Refresh Status"):
                st.experimental_rerun()

        # Simulation status
        status = api_client.get_simulation_status()
        if status:
            if status.get("is_running"):
                st.markdown(
                    '<div class="status-running">🟢 Simulation Running</div>',
                    unsafe_allow_html=True
                )

                if status.get("current_time_hours"):
                    progress = (status["current_time_hours"] / status["duration_hours"]) * 100
                    st.progress(progress / 100)
                    st.text(f"Progress: {progress:.1f}% ({status['current_time_hours']:.1f}h / {status['duration_hours']:.1f}h)")
            else:
                st.markdown(
                    '<div class="status-stopped">🔴 Simulation Stopped</div>',
                    unsafe_allow_html=True
                )

    with tab2:
        st.header("Real-Time Monitoring")

        # Auto-refresh controls
        col1, col2 = st.columns([2, 1])

        with col1:
            auto_refresh = st.checkbox("Enable Auto-refresh", value=False)

        with col2:
            refresh_interval = st.number_input(
                "Interval (s)",
                min_value=1,
                max_value=60,
                value=5,
                step=1,
                disabled=not auto_refresh
            )

        # Auto-refresh implementation
        if auto_refresh:
            import time
            time.sleep(refresh_interval)
            st.experimental_rerun()

        # Load and display real-time data
        data = api_client.get_latest_data(limit=200)

        if data:
            st.success(f"📊 Loaded {len(data)} data points")

            # Real-time plots
            fig = create_real_time_plots(data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            # Current status metrics
            if data:
                latest = data[-1]

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Current Time", f"{latest['time_hours']:.1f} h")

                with col2:
                    st.metric(
                        "Reservoir Concentration",
                        f"{latest['reservoir_concentration']:.2f} mM",
                        delta=f"{latest['reservoir_concentration'] - 25:.2f}"
                    )

                with col3:
                    st.metric("Power Output", f"{latest['total_power']:.3f} W")

                with col4:
                    st.metric("Q-Action", int(latest['q_action']))
        else:
            st.info("No simulation data available. Start a simulation to see real-time monitoring.")

    with tab3:
        st.header("Performance Metrics")

        metrics = api_client.get_performance_metrics()

        if metrics:
            # Performance dashboard
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Final Concentration",
                    f"{metrics['final_reservoir_concentration']:.2f} mM",
                    delta=f"{metrics['final_reservoir_concentration'] - 25:.2f}"
                )

            with col2:
                st.metric(
                    "Control Effectiveness",
                    f"{metrics['control_effectiveness_2mM']:.1f}%"
                )

            with col3:
                st.metric(
                    "Mean Power",
                    f"{metrics['mean_power']:.3f} W"
                )

            with col4:
                st.metric(
                    "Substrate Consumed",
                    f"{metrics['total_substrate_added']:.1f} mmol"
                )

            # Additional metrics
            if metrics.get('energy_efficiency'):
                st.subheader("Additional Metrics")
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Energy Efficiency", f"{metrics['energy_efficiency']:.1f}%")

                with col2:
                    st.metric("Stability Score", f"{metrics.get('stability_score', 0):.2f}")

            # Export options
            st.subheader("Data Export")
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("📥 Export CSV"):
                    st.info("Export functionality requires API implementation")

            with col2:
                if st.button("📊 Export Excel"):
                    st.info("Export functionality requires API implementation")

            with col3:
                if st.button("📋 Generate Report"):
                    st.info("Report generation requires API implementation")
        else:
            st.info("No performance metrics available. Run a simulation to see results.")

    with tab4:
        st.header("Configuration & Settings")

        # SSL Configuration
        st.subheader("🔒 SSL Configuration")

        if ssl_config:
            config_data = {
                "Domain": ssl_config.domain,
                "HTTPS Ports": {
                    "API": ssl_config.https_port_api,
                    "Frontend": ssl_config.https_port_frontend,
                    "WebSocket": ssl_config.wss_port_streaming
                },
                "Certificate Files": {
                    "Certificate": ssl_config.cert_file,
                    "Private Key": ssl_config.key_file
                },
                "Security Features": {
                    "HSTS": ssl_config.enable_hsts,
                    "CSP": ssl_config.enable_csp,
                    "Auto-renewal": ssl_config.auto_renew
                }
            }

            st.json(config_data)
        else:
            st.warning("SSL not configured. Running in HTTP mode.")

            if st.button("🔧 Initialize SSL"):
                st.info("SSL initialization requires backend API")

        # Alert Configuration
        st.subheader("🚨 Alert Configuration")
        st.info("Alert configuration panel - requires API implementation")

        # System Settings
        st.subheader("⚙️ System Settings")

        col1, col2 = st.columns(2)

        with col1:
            st.checkbox("Enable Email Notifications", value=False, disabled=True)
            st.selectbox("Log Level", ["INFO", "DEBUG", "WARNING", "ERROR"], disabled=True)

        with col2:
            st.number_input("Data Retention (days)", min_value=1, value=30, disabled=True)
            st.checkbox("Enable Performance Monitoring", value=True, disabled=True)

    # Footer
    st.markdown("---")
    footer_text = "🔬 MFC Monitoring Dashboard v1.2.0"
    if ssl_config:
        footer_text += " | 🔒 HTTPS Secure"
    else:
        footer_text += " | ⚠️ HTTP Development Mode"

    st.markdown(footer_text)

def run_streamlit_https(
    port: Optional[int] = None,
    ssl_config_override: Optional[SSLConfig] = None
):
    """Run Streamlit with HTTPS configuration"""

    # Load SSL configuration
    ssl_config = ssl_config_override or load_ssl_config()

    # Use configured port or default
    if port is None:
        port = ssl_config.https_port_frontend if ssl_config else 8501

    # Streamlit command with SSL
    cmd = [
        "streamlit", "run", __file__,
        "--server.port", str(port),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--server.fileWatcherType", "none",
        "--browser.gatherUsageStats", "false"
    ]

    # Add SSL parameters if certificates exist
    if ssl_config and Path(ssl_config.cert_file).exists() and Path(ssl_config.key_file).exists():
        cmd.extend([
            "--server.sslCertFile", ssl_config.cert_file,
            "--server.sslKeyFile", ssl_config.key_file
        ])

        logger.info(f"Starting HTTPS Streamlit on port {port}")
        logger.info(f"Certificate: {ssl_config.cert_file}")
        logger.info(f"Key: {ssl_config.key_file}")
    else:
        logger.warning("SSL certificates not found, running HTTP Streamlit")

    # Run Streamlit
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        logger.info("Streamlit frontend stopped")

if __name__ == "__main__":
    # When run as script, handle HTTPS startup
    if len(sys.argv) > 1 and sys.argv[1] == "run_https":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else None
        run_streamlit_https(port)
    else:
        # Normal Streamlit execution
        main()
