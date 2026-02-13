# 🐛 Debug Mode for MFC Simulations

The MFC simulation system now supports a debug mode that outputs all simulation files to a temporary directory instead of the normal project directory. This is useful for testing, development, and avoiding clutter in your main data directories.

## 🎯 Purpose

Debug mode is designed for:
- **Testing simulations** without filling up your main data directory
- **Development work** where you need to run many short test simulations
- **Experimentation** with different parameters without cluttering results
- **Continuous Integration** where temporary outputs are preferred

## 🚀 Usage Methods

### 1. Command Line Flag
Add the `--debug` flag to any simulation script:

```bash
# Run a short test simulation in debug mode
python src/mfc_recirculation_control.py --duration 1 --debug

# All files will go to /tmp/mfc_debug_simulation/ instead of data/
```

### 2. Environment Variable
Set the `MFC_DEBUG_MODE` environment variable:

```bash
# Enable debug mode for all simulations
export MFC_DEBUG_MODE=true
python src/mfc_recirculation_control.py --duration 1

# Or for a single command
MFC_DEBUG_MODE=true python src/mfc_recirculation_control.py --duration 1
```

### 3. Streamlit GUI
In the Streamlit GUI, enable debug mode via the sidebar:
1. Expand "Advanced Settings" in the sidebar
2. Check the "Debug Mode" checkbox
3. Start your simulation normally
4. Files will be saved to temporary directory

### 4. Programmatic Control
Enable/disable debug mode in Python code:

```python
from path_config import enable_debug_mode, disable_debug_mode, is_debug_mode

# Enable debug mode
enable_debug_mode()
print(f"Debug mode: {is_debug_mode()}")  # True

# Run your simulation code here...

# Disable debug mode
disable_debug_mode()
```

## 📁 File Locations

### Normal Mode
```
q-learning-mfcs/
├── data/
│   ├── simulation_data/     # CSV, JSON files
│   ├── figures/            # Plots and visualizations
│   ├── logs/               # Simulation logs
│   └── ...
```

### Debug Mode
```
/tmp/mfc_debug_simulation/   # Temporary directory
├── data/
│   ├── simulation_data/     # CSV, JSON files (debug)
│   ├── figures/            # Plots and visualizations (debug)
│   ├── logs/               # Simulation logs (debug)
│   └── ...
```

## 🔍 Features

### Visual Indicators
- **Command Line**: Shows debug message when enabled
- **Streamlit GUI**: Warning message appears when debug mode is active
- **File Paths**: All output paths automatically redirect to temp directory

### Automatic Cleanup
- Files in `/tmp/mfc_debug_simulation/` may be automatically cleaned by the system
- Debug files are temporary and not meant for long-term storage
- Normal project files remain untouched

## 📊 Example Output

### Command Line with Debug Mode
```bash
$ python src/mfc_recirculation_control.py --duration 1 --debug
🐛 DEBUG MODE ENABLED - Output directory: /tmp/mfc_debug_simulation
=== MFC SIMULATION STARTED ===
Duration: 1 hours
Output directory: /tmp/mfc_debug_simulation/data/simulation_data/mfc_simulation_20250728_151030
...
CSV data saved to: mfc_simulation_1h_20250728_151030_data.csv
JSON data saved to: mfc_simulation_1h_20250728_151030_data.json
```

### Streamlit GUI with Debug Mode
When debug mode is enabled in the GUI:
- ✅ Simulation started successfully
- 📊 Data saving synchronized with 5s refresh rate  
- ⚠️ **DEBUG MODE: Files will be saved to temporary directory**

## ⚙️ Configuration

### Environment Variables
- `MFC_DEBUG_MODE`: Set to `true`, `1`, or `yes` to enable debug mode globally

### Path Configuration
The debug mode uses the `path_config.py` module to redirect all file outputs:
- `get_simulation_data_path()` → `/tmp/mfc_debug_simulation/data/simulation_data/`
- `get_figure_path()` → `/tmp/mfc_debug_simulation/data/figures/`
- `get_log_path()` → `/tmp/mfc_debug_simulation/data/logs/`

## 🧪 Testing

Test the debug mode functionality:

```bash
cd src
python test_debug_mode.py
```

This will verify that:
- Debug mode can be enabled/disabled
- File paths switch correctly
- Environment variables work
- No interference with normal mode

## 🚨 Important Notes

1. **Temporary Storage**: Debug files are stored in `/tmp/` and may be cleaned automatically
2. **Not for Production**: Debug mode is for testing only - use normal mode for important simulations
3. **Path Consistency**: All simulation components respect debug mode automatically
4. **No Data Loss**: Normal mode files are never affected by debug mode
5. **System Cleanup**: The system may automatically delete `/tmp/` contents on reboot

## 🔧 Implementation Details

The debug mode is implemented in `src/path_config.py` with:
- Environment variable detection
- Programmatic enable/disable functions
- Automatic temporary directory creation
- Thread-safe path resolution
- Backward compatibility with existing code

All simulation scripts automatically use the path configuration, so no code changes are needed to support debug mode.