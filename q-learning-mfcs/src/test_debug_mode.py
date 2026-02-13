#!/usr/bin/env python3
"""
Test script for debug mode functionality
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from path_config import (
    is_debug_mode, get_current_base_path, enable_debug_mode, disable_debug_mode,
    get_simulation_data_path, get_figure_path, get_log_path
)

def test_debug_mode():
    """Test debug mode functionality"""
    
    print("🧪 Testing Debug Mode Functionality")
    print("=" * 50)
    
    # Test initial state
    print(f"Initial debug mode: {is_debug_mode()}")
    print(f"Initial base path: {get_current_base_path()}")
    print()
    
    # Test normal mode paths
    print("📁 Normal Mode Paths:")
    print(f"  Simulation data: {get_simulation_data_path('test_file.csv')}")
    print(f"  Figures: {get_figure_path('test_plot.png')}")
    print(f"  Logs: {get_log_path('test.log')}")
    print()
    
    # Enable debug mode
    print("🐛 Enabling Debug Mode...")
    enable_debug_mode()
    print(f"Debug mode enabled: {is_debug_mode()}")
    print(f"Debug base path: {get_current_base_path()}")
    print()
    
    # Test debug mode paths
    print("📁 Debug Mode Paths:")
    print(f"  Simulation data: {get_simulation_data_path('test_file.csv')}")
    print(f"  Figures: {get_figure_path('test_plot.png')}")
    print(f"  Logs: {get_log_path('test.log')}")
    print()
    
    # Test environment variable
    print("🌍 Testing Environment Variable...")
    os.environ['MFC_DEBUG_MODE'] = 'true'
    # Need to reload module for environment variable to take effect
    import importlib
    import path_config
    importlib.reload(path_config)
    
    print(f"Environment debug mode: {path_config.is_debug_mode()}")
    print(f"Environment base path: {path_config.get_current_base_path()}")
    print()
    
    # Cleanup
    disable_debug_mode()
    os.environ.pop('MFC_DEBUG_MODE', None)
    
    print("✅ Debug mode test completed!")

if __name__ == "__main__":
    test_debug_mode()