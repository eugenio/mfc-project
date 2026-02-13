#!/bin/bash
# Start 1-year MFC simulation in detached mode with GPU acceleration
#
# Created: 2025-07-26

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/../data/logs"
mkdir -p "$LOG_DIR"

# Timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/1year_simulation_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/1year_simulation.pid"

echo "🚀 Starting 1-year MFC simulation in detached mode"
echo "📁 Log file: $LOG_FILE"
echo "🆔 PID file: $PID_FILE"

# Check if simulation is already running
if [ -f "$PID_FILE" ]; then
    if ps -p $(cat "$PID_FILE") > /dev/null 2>&1; then
        echo "❌ Simulation already running (PID: $(cat "$PID_FILE"))"
        echo "   Stop it first with: kill $(cat "$PID_FILE")"
        exit 1
    else
        echo "🧹 Removing stale PID file"
        rm "$PID_FILE"
    fi
fi

# Set GPU environment variables
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
export NUMBA_ENABLE_CUDASIM=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=8

echo "🔧 Environment configured for GPU acceleration"

# Start simulation in background
cd "$SCRIPT_DIR"
nohup python run_1year_optimized.py > "$LOG_FILE" 2>&1 &
SIMULATION_PID=$!

# Save PID
echo $SIMULATION_PID > "$PID_FILE"

echo "✅ Simulation started successfully!"
echo "🆔 Process ID: $SIMULATION_PID"
echo "📊 Monitor progress with: tail -f $LOG_FILE"
echo "⏹️  Stop simulation with: kill $SIMULATION_PID"
echo ""
echo "📋 Useful commands:"
echo "   Monitor: tail -f $LOG_FILE"
echo "   Status:  ps -p $SIMULATION_PID"
echo "   Stop:    kill $SIMULATION_PID"
echo "   Results: ls -la ${SCRIPT_DIR}/../data/simulation_data/"
echo ""
echo "⏱️  Estimated completion: 2-4 hours"
echo "🎯 Target: 25.0 mM substrate concentration over 8,784 hours"

# Show initial log output
echo ""
echo "📖 Initial log output:"
echo "===================="
sleep 2
tail -n 20 "$LOG_FILE" 2>/dev/null || echo "Log file not ready yet..."

echo ""
echo "🚀 1-year simulation is now running in the background!"
echo "💡 The simulation will continue even if you close this terminal."