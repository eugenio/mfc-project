# MFC GUI Debug System Guide

## Overview

The MFC Streamlit GUI now includes a comprehensive debug system that provides real-time monitoring of system events, GPU operations, and simulation diagnostics.

## Features

### 1. Debug Mode Toggle
- **Location**: Sidebar under "Settings"
- **Checkbox**: "🐛 Debug Mode"
- **Effect**: Enables verbose console output and shows debug monitor tab

### 2. Debug Console Tab
When debug mode is enabled, a new "🐛 Debug Console" tab appears with:

#### Real-time Debug Messages
- **Timestamped logs** with categories (DEBUG, GPU, SIM, CLEANUP, ERROR)
- **Scrollable text area** showing last 100 messages
- **Message filtering** by log level (ALL, DEBUG, INFO, GPU, ERROR)
- **Auto-refresh option** for real-time monitoring

#### System Information Panel
- **GPU Backend**: Current GPU acceleration backend (ROCM/CUDA/CPU)
- **GPU Device**: Graphics card model and details
- **RAM Usage**: System memory consumption
- **Simulation Status**: Current simulation state
- **Debug Message Count**: Total messages captured

#### GPU Memory Monitoring (ROCm)
- **GPU Memory Used/Free**: VRAM consumption in MB
- **GPU Memory Usage**: Percentage utilization
- **Real-time updates** during simulation

### 3. Debug Control Buttons
- **🔄 Refresh**: Manually refresh debug messages
- **🗑️ Clear Log**: Clear all debug messages
- **🔄 Auto-refresh**: Toggle automatic refresh every 2 seconds
- **🧪 Test Debug Message**: Send test message to verify system

## How to Use

### Basic Usage
1. **Enable Debug Mode**: Check the "🐛 Debug Mode" checkbox in the sidebar
2. **View Debug Console**: Click the "🐛 Debug Console" tab
3. **Monitor Events**: Watch real-time system events and GPU operations

### URL Parameter
You can also enable debug mode directly via URL:
```
http://localhost:8501?debug=true
```

### Advanced Monitoring
1. **Filter Messages**: Use the "Filter Level" dropdown to show specific types of events
2. **Auto-refresh**: Enable auto-refresh for continuous monitoring during simulations
3. **System Monitoring**: Check RAM and GPU memory usage in real-time

## Debug Message Categories

### 🐛 DEBUG
General debug information and system state changes

### 🎮 GPU  
GPU backend detection, initialization, and memory operations

### 🚀 SIM
Simulation lifecycle events (start, stop, progress)

### 🧹 CLEANUP
Resource cleanup and memory management operations

### ❌ ERROR
Error conditions and exceptions

## Benefits

### For Users
- **Real-time monitoring** of simulation progress and system health
- **Issue diagnosis** when simulations fail or perform poorly
- **Performance tracking** of GPU and memory usage
- **Transparency** into what the system is doing

### For Debugging
- **Console output mirroring** in GUI for better visibility
- **Timestamped logs** for precise event tracking
- **Categorized messages** for focused troubleshooting
- **Thread-safe logging** for multi-threaded operations

## Technical Implementation

### DebugLogger Class
- **Thread-safe** message storage using deque and locks
- **Configurable verbosity** with enable/disable functionality
- **Automatic timestamping** with millisecond precision
- **Message categorization** for organized viewing

### GPU Integration
- **Silent initialization** by default to reduce console noise
- **Verbose mode** when debug is enabled
- **Memory monitoring** for ROCm GPUs
- **Cleanup tracking** with detailed resource management logs

### Streamlit Integration
- **Session state management** for debug mode persistence
- **Dynamic tab creation** when debug mode is enabled
- **Real-time updates** with auto-refresh functionality
- **URL parameter support** for direct debug activation

## Example Usage

1. **Start the GUI** normally: `pixi run gui`
2. **Enable debug mode** via checkbox or URL parameter
3. **Run a simulation** and watch the debug console
4. **Monitor GPU usage** during intensive computations
5. **Review logs** for any issues or performance bottlenecks

The debug system provides comprehensive visibility into the MFC simulation system while maintaining clean, noise-free operation when disabled.