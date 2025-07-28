#!/usr/bin/env python3
"""
Test the improved data loading and error handling
"""

import sys
import os
import time
import tempfile
from pathlib import Path

# Add src to path (go up two directories from tests/gui to reach src)
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src')
sys.path.insert(0, src_path)

from path_config import enable_debug_mode, disable_debug_mode
from mfc_streamlit_gui import SimulationRunner, load_simulation_data
from config.qlearning_config import DEFAULT_QLEARNING_CONFIG

def test_data_loading_fix():
    """Test that data loading handles empty files and errors gracefully"""
    
    print("🧪 Testing Data Loading Fix")
    print("=" * 50)
    
    # Enable debug mode
    enable_debug_mode()
    print("🐛 Debug mode enabled")
    
    try:
        # Create runner
        runner = SimulationRunner()
        print("✅ SimulationRunner created")
        
        # Start a very short simulation
        print("\n🚀 Starting short simulation...")
        success = runner.start_simulation(
            config=DEFAULT_QLEARNING_CONFIG,
            duration_hours=0.002,  # ~7 seconds
            n_cells=3,
            electrode_area_m2=0.001,
            target_conc=25.0,
            gui_refresh_interval=1.0,
            debug_mode=True
        )
        
        if success:
            print("✅ Simulation started successfully")
            
            # Wait a moment for initial data to be saved
            time.sleep(3)
            
            # Test data loading
            if runner.current_output_dir:
                print(f"\n📊 Testing data loading from: {runner.current_output_dir}")
                
                # Try loading data multiple times to see if it improves
                for i in range(5):
                    df = load_simulation_data(runner.current_output_dir)
                    
                    if df is not None:
                        print(f"✅ Attempt {i+1}: Data loaded successfully - {len(df)} rows, {len(df.columns)} columns")
                        print(f"   Columns: {list(df.columns)}")
                        if len(df) > 0:
                            print(f"   Sample data: {df.iloc[0].to_dict()}")
                        break
                    else:
                        print(f"⏳ Attempt {i+1}: No data yet, waiting...")
                        time.sleep(2)
                
                # Check files directly
                csv_files = list(runner.current_output_dir.glob("*.csv.gz"))
                if csv_files:
                    csv_file = csv_files[0]
                    print(f"\n📁 File info: {csv_file.name} ({csv_file.stat().st_size} bytes)")
                    
                    # Try reading directly
                    import gzip
                    import pandas as pd
                    
                    try:
                        with gzip.open(csv_file, 'rt') as f:
                            content = f.read()
                            lines = content.strip().split('\n')
                            print(f"   Lines in file: {len(lines)}")
                            if lines:
                                print(f"   Header: {lines[0]}")
                                if len(lines) > 1:
                                    print(f"   First data row: {lines[1]}")
                                else:
                                    print("   No data rows (headers only)")
                    except Exception as e:
                        print(f"   Error reading file: {e}")
            
            # Stop simulation
            if runner.is_actually_running():
                print("\n⏹️ Stopping simulation...")
                runner.stop_simulation()
                time.sleep(2)
            
        else:
            print("❌ Failed to start simulation")
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        disable_debug_mode()
        print("\n✅ Test completed!")

if __name__ == "__main__":
    test_data_loading_fix()