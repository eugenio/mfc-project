#!/bin/bash
# Automatic GPU Detection and Environment Setup Script
#
# This script automatically detects your GPU hardware and sets up the optimal
# pixi environment with the correct ML framework dependencies.

set -e  # Exit on any error

echo "🚀 MFC Project - Automatic Environment Setup"
echo "============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if pixi is installed
if ! command -v pixi &> /dev/null; then
    echo -e "${RED}❌ Pixi not found. Please install pixi first:${NC}"
    echo "   curl -fsSL https://pixi.sh/install.sh | bash"
    exit 1
fi

echo -e "${BLUE}🔍 Detecting GPU hardware...${NC}"

# Run GPU detection
cd "$(dirname "$0")"
python scripts/detect_gpu.py
GPU_EXIT_CODE=$?

# Determine environment based on detection result
case $GPU_EXIT_CODE in
    0)  # CPU/Default
        ENVIRONMENT="default"
        echo -e "${YELLOW}💻 Using CPU-only environment${NC}"
        ;;
    1)  # NVIDIA
        ENVIRONMENT="nvidia"
        echo -e "${GREEN}🎮 Using NVIDIA CUDA environment${NC}"
        ;;
    2)  # AMD
        ENVIRONMENT="amd"
        echo -e "${GREEN}🎮 Using AMD ROCm environment${NC}"
        ;;
    *)
        ENVIRONMENT="default"
        echo -e "${YELLOW}⚠️ Unknown GPU type, falling back to CPU environment${NC}"
        ;;
esac

echo -e "${BLUE}📦 Installing dependencies for $ENVIRONMENT environment...${NC}"

# Install the specific environment
pixi install --environment $ENVIRONMENT

# Verify the installation
echo -e "${BLUE}🔍 Verifying installation...${NC}"
python scripts/install_gpu_deps.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Environment setup completed successfully!${NC}"
    echo ""
    echo -e "${BLUE}🚀 To activate the optimized environment, run:${NC}"
    echo -e "${GREEN}   pixi shell $ENVIRONMENT${NC}"
    echo ""
    echo -e "${BLUE}💡 Then you can test the cathode models:${NC}"
    echo "   python -c \"from cathode_models.platinum_cathode import create_platinum_cathode; print('✅ Cathode models working!')\""
    echo ""
    echo -e "${BLUE}🎯 Or run the GUI:${NC}"
    echo "   streamlit run src/mfc_streamlit_gui.py"
else
    echo -e "${YELLOW}⚠️ Environment setup completed with warnings${NC}"
    echo -e "${BLUE}💡 You can still use the CPU-only environment:${NC}"
    echo -e "${YELLOW}   pixi shell default${NC}"
fi

echo ""
echo -e "${GREEN}🎉 Setup complete! Happy computing! 🧪⚡${NC}"