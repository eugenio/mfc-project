#!/bin/bash
"""
Enhanced MFC GUI Startup Script

Launches the enhanced Streamlit-based MFC research platform with all
advanced features for scientific community engagement.

Created: 2025-07-31
"""

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUI_SCRIPT="$SCRIPT_DIR/gui/enhanced_mfc_gui.py"
PORT=8502  # Different port from original GUI

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Print banner
echo -e "${PURPLE}"
echo "=========================================="
echo "🔬 Enhanced MFC Research Platform"
echo "=========================================="
echo -e "${NC}"
echo "Advanced Microbial Fuel Cell Analysis & Q-Learning Optimization"
echo "Designed for Scientific Community Engagement"
echo ""

# Check if GUI script exists
if [ ! -f "$GUI_SCRIPT" ]; then
    echo -e "${RED}❌ Error: GUI script not found at $GUI_SCRIPT${NC}"
    exit 1
fi

# Check Python environment
echo -e "${BLUE}🔍 Checking Python environment...${NC}"

# Check if pixi is available
if command -v pixi &> /dev/null; then
    echo -e "${GREEN}✅ Pixi environment manager found${NC}"
    PYTHON_CMD="pixi run python"
    STREAMLIT_CMD="pixi run streamlit"
else
    echo -e "${YELLOW}⚠️  Pixi not found, using system Python${NC}"
    PYTHON_CMD="python"
    STREAMLIT_CMD="streamlit"
fi

# Check if Streamlit is available
if ! $PYTHON_CMD -c "import streamlit" &> /dev/null; then
    echo -e "${RED}❌ Error: Streamlit not found${NC}"
    echo "Please install Streamlit:"
    echo "  pip install streamlit"
    echo "  or"
    echo "  pixi add streamlit"
    exit 1
fi

# Check required dependencies
echo -e "${BLUE}🔍 Checking required dependencies...${NC}"

REQUIRED_PACKAGES=(
    "pandas"
    "numpy" 
    "plotly"
    "scipy"
)

MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! $PYTHON_CMD -c "import $package" &> /dev/null; then
        MISSING_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
    echo -e "${RED}❌ Missing required packages: ${MISSING_PACKAGES[*]}${NC}"
    echo "Please install missing packages:"
    for package in "${MISSING_PACKAGES[@]}"; do
        echo "  pixi add $package"
    done
    exit 1
fi

echo -e "${GREEN}✅ All dependencies satisfied${NC}"

# Check port availability
echo -e "${BLUE}🔍 Checking port $PORT availability...${NC}"

if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null; then
    echo -e "${YELLOW}⚠️  Port $PORT is already in use${NC}"
    echo "Do you want to use a different port? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Enter new port number:"
        read -r PORT
    else
        echo "Attempting to use port $PORT anyway..."
    fi
fi

# Set environment variables for enhanced performance
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Create logs directory if it doesn't exist
LOGS_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOGS_DIR"

# Launch enhanced GUI
echo -e "${GREEN}🚀 Starting Enhanced MFC Research Platform...${NC}"
echo ""
echo -e "${BLUE}📍 GUI will be available at: http://localhost:$PORT${NC}"
echo -e "${BLUE}📁 Working directory: $SCRIPT_DIR${NC}"
echo -e "${BLUE}📝 Logs directory: $LOGS_DIR${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

# Change to script directory
cd "$SCRIPT_DIR"

# Launch Streamlit with enhanced GUI
$STREAMLIT_CMD run "$GUI_SCRIPT" \
    --server.port=$PORT \
    --server.address=localhost \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --theme.primaryColor="#667eea" \
    --theme.backgroundColor="#ffffff" \
    --theme.secondaryBackgroundColor="#f0f2f6" \
    --theme.textColor="#262730" \
    2>&1 | tee "$LOGS_DIR/enhanced_gui_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo -e "${GREEN}✅ Enhanced MFC Research Platform stopped${NC}"