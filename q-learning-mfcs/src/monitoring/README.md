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
## Quick Start

### Prerequisites
Ensure you have the required dependencies installed:

```bash
# Install required packages
pixi install
# or
pip install fastapi uvicorn streamlit plotly websockets pandas numpy
```

### Starting the System

#### Option 1: Start All Services
```bash
cd q-learning-mfcs/src/monitoring
python start_monitoring.py
```

#### Option 2: Start Individual Components

1. **Start API Server:**
```bash
cd q-learning-mfcs/src
python -m uvicorn monitoring.simple_dashboard_api:app --reload --host 0.0.0.0 --port 8000
```

2. **Start Dashboard Frontend:**
```bash
cd q-learning-mfcs/src
streamlit run monitoring/dashboard_frontend.py --server.port 8501
```

### Access Points

Once running, access the system through:

- **📊 Dashboard UI**: http://localhost:8501
- **📖 API Documentation**: http://localhost:8000/docs
- **🔗 Health Check**: http://localhost:8000/api/health
- **🔄 WebSocket Stream**: ws://localhost:8001/ws
## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Browser   │    │   Streamlit      │    │   FastAPI       │
│   (Dashboard)   │◄──►│   Frontend       │◄──►│   Backend       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                       ┌──────────────────┐    ┌─────────────────┐
                       │   WebSocket      │    │   Safety        │
                       │   Streamer       │◄──►│   Monitor       │
                       └──────────────────┘    └─────────────────┘
                                                        │
                                                ┌─────────────────┐
                                                │   MFC Model     │
                                                │   Integration   │
                                                └─────────────────┘
```
