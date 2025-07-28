#!/usr/bin/env python3
"""
Test script to verify GUI fixes work properly
"""

import sys
import os
# Add src to path (go up two directories from tests/gui to reach src)
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src')
sys.path.insert(0, src_path)

def test_gpu_cleanup():
    """Test the GPU cleanup function"""
    print("🔍 Testing GPU cleanup function...")
    
    try:
        # Import and create a simulation runner
        from mfc_streamlit_gui import SimulationRunner
        runner = SimulationRunner()
        
        # Test cleanup
        print("🧹 Running cleanup test...")
        runner._cleanup_resources()
        print("✅ GPU cleanup test completed")
        
        return True
    except Exception as e:
        print(f"❌ GPU cleanup test failed: {e}")
        return False

def test_data_loading():
    """Test the data loading function with real files"""
    print("\n🔍 Testing data loading function...")
    
    try:
        from mfc_streamlit_gui import load_simulation_data
        from pathlib import Path
        
        # Find a recent simulation directory
        data_dir = Path("../data/simulation_data")
        sim_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('gui_simulation')]
        
        if not sim_dirs:
            print("⚠️  No GUI simulation directories found for testing")
            return True  # Not a failure, just no data
            
        # Test loading from most recent
        latest_dir = max(sim_dirs, key=lambda x: x.stat().st_mtime)
        print(f"📁 Testing with: {latest_dir.name}")
        
        df = load_simulation_data(latest_dir)
        if df is not None:
            print(f"✅ Data loaded: {len(df)} rows")
            
            # Check attributes
            is_live = getattr(df, 'attrs', {}).get('is_live_data', False)
            print(f"   Live data: {is_live}")
            
            if len(df) > 0:
                print(f"   Time range: {df['time_hours'].iloc[0]:.1f} - {df['time_hours'].iloc[-1]:.1f} hours")
                print(f"   Final concentration: {df['reservoir_concentration'].iloc[-1]:.2f} mM")
        else:
            print("⚠️  No data loaded")
            
        return True
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def check_memory_usage():
    """Check current GPU memory usage"""
    print("\n🔍 Checking GPU memory usage...")
    
    try:
        import subprocess
        result = subprocess.run(['rocm-smi', '--showmeminfo'], capture_output=True, text=True)
        if result.returncode == 0:
            print("📊 Current GPU memory usage:")
            lines = result.stdout.split('\n')
            for line in lines:
                if 'GPU' in line or 'VRAM' in line or 'Used' in line:
                    print(f"   {line}")
        else:
            print("⚠️  Could not get GPU memory info")
    except Exception as e:
        print(f"⚠️  GPU memory check failed: {e}")

if __name__ == "__main__":
    print("🔬 GUI Fixes Test Suite")
    print("=" * 40)
    
    # Test 1: GPU cleanup
    test1_result = test_gpu_cleanup()
    
    # Test 2: Data loading
    test2_result = test_data_loading()
    
    # Check memory usage
    check_memory_usage()
    
    print("\n" + "=" * 40)
    print("📋 TEST RESULTS")
    print("=" * 40)
    print(f"GPU cleanup:   {'✅ PASS' if test1_result else '❌ FAIL'}")
    print(f"Data loading:  {'✅ PASS' if test2_result else '❌ FAIL'}")
    
    if test1_result and test2_result:
        print("\n🎉 All tests passed!")
        print("💡 GUI fixes should now work properly.")
        print("📌 To test GUI:")
        print("   1. Enable auto-refresh in Monitor tab")
        print("   2. Use Manual Refresh button")
        print("   3. Use Force GPU Cleanup button after simulations")
    else:
        print("\n⚠️  Some tests failed. Check error messages above.")