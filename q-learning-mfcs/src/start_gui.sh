#!/bin/bash
# Launch MFC Simulation Streamlit GUI

echo "🚀 Starting MFC Simulation Control Panel..."
echo "📡 The GUI will be available at: http://localhost:8501"
echo "🔧 Backend: Streamlit with real-time monitoring"
echo ""

# Change to source directory
cd "$(dirname "$0")"

# Launch Streamlit using pixi
pixi run streamlit run mfc_streamlit_gui.py --server.port 8501 --server.address localhost

echo "✅ GUI session ended"