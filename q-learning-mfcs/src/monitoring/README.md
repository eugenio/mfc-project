# MFC Real-time Monitoring System
## Architecture Overview

The monitoring system consists of four main components:

1. **Dashboard API** (`dashboard_api.py`) - FastAPI-based REST API server
2. **Frontend Dashboard** (`dashboard_frontend.py`) - Streamlit-based web interface
3. **Real-time Streamer** (`realtime_streamer.py`) - WebSocket streaming service
4. **Safety Monitor** (`safety_monitor.py`) - Safety monitoring and emergency response
## Key Features

### 🔍 Real-time Monitoring
- Live system metrics (power, efficiency, temperature, pH, pressure)
- Individual cell monitoring (voltage, current, biofilm thickness)
- Performance trending and analytics
- System health indicators

### ⚡ Safety System
- Configurable safety thresholds for all parameters
- Automated emergency responses (power reduction, flow control, shutdown)
- Safety protocol execution for critical scenarios
- Real-time alert generation and notification

### 📊 Dashboard Interface
- Modern web-based dashboard with real-time updates
- Interactive charts and visualizations
- System control panel (start/stop/emergency stop)
- Alert management and acknowledgment
- Historical data analysis

### 🌐 API & Streaming
- RESTful API for system integration
- WebSocket streaming for real-time data
- Multi-client support with authentication
- Event subscription management
