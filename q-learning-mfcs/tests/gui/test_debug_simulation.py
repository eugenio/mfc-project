#!/usr/bin/env python3
"""
Test debug mode simulation to check for GUI-related bugs
"""

import sys
import os
import time
import tempfile
from pathlib import Path

# Add src to path (go up two directories from tests/gui to reach src)
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src')
sys.path.insert(0, src_path)

from path_config import enable_debug_mode, disable_debug_mode, is_debug_mode, get_current_base_path
from config.qlearning_config import DEFAULT_QLEARNING_CONFIG

def test_debug_simulation():
    """Test a short debug simulation similar to GUI workflow"""
    
    print("🧪 Testing Debug Mode Simulation")
    print("=" * 50)
    
    # Enable debug mode
    print("🐛 Enabling debug mode...")
    enable_debug_mode()
    print(f"Debug mode active: {is_debug_mode()}")
    print(f"Debug output path: {get_current_base_path()}")
    print()
    
    # Test SimulationRunner class like the GUI does
    try:
        print("📱 Testing GUI SimulationRunner class...")
        
        # Import the SimulationRunner from GUI
        from mfc_streamlit_gui import SimulationRunner
        
        # Create runner instance
        runner = SimulationRunner()
        print("✅ SimulationRunner created successfully")
        
        # Test path resolution
        from path_config import get_simulation_data_path
        test_path = get_simulation_data_path("test_debug_sim")
        print(f"Test path resolution: {test_path}")
        
        # Check if we can create directories
        test_dir = Path(test_path)
        test_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Directory creation works: {test_dir.exists()}")
        
        print()
        print("🚀 Starting short debug simulation...")
        
        # Start a very short simulation (30 seconds = 0.5 minutes)
        duration_hours = 0.008  # ~30 seconds
        
        success = runner.start_simulation(
            config=DEFAULT_QLEARNING_CONFIG,
            duration_hours=duration_hours,  
            n_cells=5,
            electrode_area_m2=0.001,  # 10 cm²
            target_conc=25.0,
            gui_refresh_interval=1.0,  # Check every second
            debug_mode=True
        )
        
        if success:
            print("✅ Simulation started successfully")
            
            # Monitor simulation briefly
            max_wait = 60  # Maximum 60 seconds
            start_time = time.time()
            
            while runner.is_running and (time.time() - start_time) < max_wait:
                status = runner.get_status()
                print(f"Status: {status[0]} - {status[1]}")
                
                if status[0] in ['completed', 'error', 'stopped']:
                    break
                    
                time.sleep(2)
            
            # Get final status
            final_status = runner.get_status()
            print(f"Final status: {final_status[0]} - {final_status[1]}")
            
            if final_status[2]:  # output_dir
                output_dir = final_status[2]
                print(f"Output directory: {output_dir}")
                
                # Check if files were created
                if output_dir.exists():
                    files = list(output_dir.glob("*"))
                    print(f"Files created: {len(files)}")
                    for file in files:
                        print(f"  - {file.name} ({file.stat().st_size} bytes)")
                else:
                    print("⚠️  Output directory doesn't exist")
            
            # Stop if still running
            if runner.is_running:
                print("⏹️  Stopping simulation...")
                runner.stop_simulation()
        
        else:
            print("❌ Failed to start simulation")
            
    except Exception as e:
        print(f"❌ Error during simulation test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        disable_debug_mode()
        print(f"Debug mode disabled: {not is_debug_mode()}")
        
        # Clean up temp directory
        temp_dir = Path(tempfile.gettempdir()) / "mfc_debug_simulation"
        if temp_dir.exists():
            try:
                import shutil
                shutil.rmtree(temp_dir)
                print("✅ Temporary debug directory cleaned up")
            except Exception as e:
                print(f"⚠️  Could not clean temp directory: {e}")
    
    print("\n✅ Debug simulation test completed!")

if __name__ == "__main__":
    test_debug_simulation()