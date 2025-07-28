# MFC Real-time Monitoring System
## Architecture Overview

The monitoring system consists of four main components:

1. **Dashboard API** (`dashboard_api.py`) - FastAPI-based REST API server
2. **Frontend Dashboard** (`dashboard_frontend.py`) - Streamlit-based web interface
3. **Real-time Streamer** (`realtime_streamer.py`) - WebSocket streaming service
4. **Safety Monitor** (`safety_monitor.py`) - Safety monitoring and emergency response
