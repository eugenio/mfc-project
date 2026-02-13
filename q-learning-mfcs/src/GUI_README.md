# 🔋 MFC Simulation GUI

A comprehensive Streamlit-based graphical interface for Microbial Fuel Cell (MFC) simulation control and monitoring.

## Features

### 🚀 **Simulation Control**

- **Quick Setup**: Pre-configured simulation durations (1 hour to 1 year)
- **Q-Learning Control**: Toggle between pre-trained and fresh Q-learning agents
- **Parameter Tuning**: Adjust target concentrations and learning parameters
- **Real-time Status**: Live simulation monitoring with progress indicators

### 📊 **Real-Time Monitoring**

- **Live Plots**: Substrate concentration, power output, Q-learning actions, biofilm growth
- **Auto-refresh**: Configurable real-time data updates
- **Performance Metrics**: Current concentration, power, and control effectiveness
- **Target Visualization**: Clear indication of 25 mM substrate target

### 📈 **Results Analysis**

- **Performance Dashboard**: Key metrics with delta indicators
- **Detailed Plots**: Interactive Plotly visualizations
- **Control Effectiveness**: Percentage within ±2mM and ±5mM targets
- **Export Options**: Download simulation data as CSV

### 📁 **Simulation History**

- **Recent Runs**: Browse previous simulation results
- **Comparison View**: Compare control effectiveness across runs
- **Data Access**: Load and visualize any historical simulation

## Quick Start

### 1. Launch the GUI

```bash
cd /home/uge/mfc-project/q-learning-mfcs/src
./start_gui.sh
```

### 2. Access the Interface

Open your browser to: **http://localhost:8501**

### 3. Run a Simulation

1. Go to "🚀 Run Simulation" tab
1. Select duration (start with "1 Hour (Quick Test)")
1. Ensure "Use Pre-trained Q-table" is checked ✅
1. Click "▶️ Start Simulation"
1. Switch to "📊 Monitor" tab for real-time updates

## GUI Tabs

### 🚀 Run Simulation

- **Duration Selection**: 1 hour to 1 year options
- **Q-Learning Settings**: Pre-trained vs fresh agent
- **Target Configuration**: Substrate concentration targets
- **Advanced Settings**: GPU backend, save intervals
- **Simulation Control**: Start/stop buttons with status

### 📊 Monitor

- **Real-Time Plots**: 4-panel dashboard showing:
  - Substrate concentration (reservoir vs outlet)
  - Power output over time
  - Q-learning action selection
  - Average biofilm thickness
- **Live Metrics**: Current time, concentration, power, action
- **Auto-refresh**: 10-second update option

### 📈 Results

- **Performance Metrics**: Final concentration, control effectiveness, power output
- **Delta Indicators**: Deviation from target values
- **JSON Export**: Complete results data
- **Simulation Info**: Runtime, backend, configuration

### 📁 History

- **Recent Simulations**: Table of all available runs
- **Key Metrics**: Duration, final concentration, control effectiveness
- **Detailed View**: Select any simulation for full analysis
- **Data Download**: Export historical data as CSV

## Key Features

### ✅ **Pre-trained Q-Learning**

The GUI automatically loads pre-trained Q-learning weights that achieve:

- **Excellent Control**: ~27 mM final concentration (vs 25 mM target)
- **54% effectiveness** within ±2mM tolerance
- **100% effectiveness** within ±5mM tolerance
- **Massive improvement** over untrained agent (100 mM failure)

### 🎯 **Real-Time Monitoring**

- Live substrate concentration tracking
- Power output visualization
- Q-learning action analysis
- Biofilm growth monitoring
- Target deviation alerts

### 📊 **Performance Analytics**

- Control effectiveness percentages
- Substrate consumption rates
- Power generation metrics
- Maintenance schedule calculations

## Technical Details

### Dependencies

- **Streamlit**: Web-based GUI framework
- **Plotly**: Interactive plotting library
- **JAX/ROCm**: GPU acceleration backend
- **Pandas**: Data manipulation
- **Threading**: Background simulation execution

### Data Sources

- **Live Simulations**: Real-time data from active runs
- **Historical Data**: Compressed CSV files from `../data/simulation_data/`
- **Configuration**: Q-learning parameters from checkpoints
- **Results**: JSON summary files with performance metrics

### GPU Support

- **Auto-detection**: NVIDIA CUDA, AMD ROCm, CPU fallback
- **Backend Display**: Shows active acceleration method
- **Performance**: ~8400× speedup vs CPU-only simulation

## Usage Tips

### 🎯 **Best Practices**

1. **Start Small**: Begin with 1-hour test simulations
1. **Use Pre-trained**: Always enable pre-trained Q-table for best control
1. **Monitor Live**: Switch to Monitor tab during simulation
1. **Check History**: Compare results across different configurations

### ⚠️ **Troubleshooting**

- **Slow Loading**: Large simulations may take time to process
- **Missing Data**: Ensure simulation completed successfully
- **Plot Issues**: Try refreshing the browser page
- **Performance**: Close unused browser tabs for better responsiveness

### 🔧 **Advanced Usage**

- **Custom Targets**: Adjust substrate concentration targets
- **Parameter Tuning**: Experiment with Q-learning hyperparameters
- **Backend Selection**: Force specific GPU backends if needed
- **Data Export**: Download results for external analysis

## Architecture

```
mfc_streamlit_gui.py
├── SimulationRunner: Background thread management
├── Real-time Monitoring: Live data updates
├── Data Loading: Historical simulation access
├── Plotting: Interactive visualizations
└── Interface: Multi-tab Streamlit layout
```

## Performance

- **Simulation Speed**: 1-year simulation in ~1 hour (8400× speedup)
- **GUI Responsiveness**: Real-time updates with minimal latency
- **Data Handling**: Efficient compressed CSV loading
- **Memory Usage**: Optimized for long-term monitoring

______________________________________________________________________

**🎉 Ready to use!** Launch the GUI and start monitoring your MFC simulations with professional-grade controls and visualization.
