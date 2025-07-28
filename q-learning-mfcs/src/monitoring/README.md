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
## Safety Monitoring

The safety system monitors critical parameters and responds automatically:

### Default Safety Thresholds
- **Temperature**: Max 45°C (reduces power if exceeded)
- **Pressure**: Max 2.5 bar (stops flow if exceeded)
- **pH Level**: Range 5.5-8.5 (notifies personnel)
- **Voltage**: Min 0.05V, Max 1.2V (reduces power)
- **Current Density**: Max 15 mA/cm² (reduces power)
- **Flow Rate**: Range 10-500 mL/min (notifies personnel)

### Emergency Actions
- **REDUCE_POWER**: Automatically reduces system power output
- **STOP_FLOW**: Stops fluid circulation
- **EMERGENCY_SHUTDOWN**: Complete system shutdown
- **ISOLATE_SYSTEM**: Isolates affected components
- **NOTIFY_PERSONNEL**: Sends alerts to operators

### Safety Protocols
- **Thermal Runaway Protection**: Temperature + current monitoring
- **Pressure Emergency**: Immediate flow shutdown for overpressure
- **System Failure**: Multi-parameter failure detection
- **Biological Contamination**: pH and biofilm monitoring
## API Reference

### Core Endpoints

#### System Status
```http
GET /api/system/status
```
Returns current system operational status.

#### Current Metrics
```http
GET /api/metrics/current
```
Returns real-time system measurements.

#### Control Commands
```http
POST /api/control/command
Content-Type: application/json

{
  "command": "start|stop|pause|resume|emergency_stop",
  "parameters": {}
}
```

#### Active Alerts
```http
GET /api/alerts/active
```
Returns current active safety alerts.

### WebSocket Streaming

Connect to `ws://localhost:8001/ws` for real-time updates:

```javascript
const ws = new WebSocket('ws://localhost:8001/ws');

// Subscribe to metrics updates
ws.send(JSON.stringify({
    type: 'subscribe',
    events: ['metrics_update', 'alert', 'status_change']
}));

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Real-time update:', data);
};
```
## Configuration

### Safety Thresholds
Update safety thresholds via API:

```http
POST /api/config/update
Content-Type: application/json

{
  "section": "safety_thresholds",
  "parameters": {
    "temperature": {"max_value": 50.0, "warning_buffer": 5.0}
  }
}
```

### Dashboard Settings
Customize dashboard appearance and refresh rates in the web interface sidebar.
