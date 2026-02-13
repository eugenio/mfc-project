#!/usr/bin/env python3
"""
Test script to verify GUI performance optimizations work properly
"""

import sys
import os
import time
import threading
from pathlib import Path

# Add src to path (go up two directories from tests/gui to reach src)
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src')
sys.path.insert(0, src_path)

def test_optimized_data_structures():
    """Test the new optimized data structures"""
    print("🔍 Testing optimized data structures...")
    
    try:
        from mfc_streamlit_gui import SimulationSnapshot, OptimizedDataBuffer
        
        # Test SimulationSnapshot creation
        snapshot = SimulationSnapshot(
            current_time=1.0,
            reservoir_concentration=10.0,
            outlet_concentration=9.5,
            total_power=100.0,
            total_current=50.0,
            system_voltage=2.0,
            flow_rate_ml_h=20.0,
            substrate_efficiency=0.95,
            biofilm_thickness_avg=150.0,
            q_action=1,
            reward=0.8,
            epsilon=0.1,
            step_number=100,
            progress_percent=10.0
        )
        print("  ✅ SimulationSnapshot created successfully")
        
        # Test OptimizedDataBuffer
        buffer = OptimizedDataBuffer(max_gui_points=100)
        buffer.add_snapshot(snapshot)
        
        gui_data = buffer.get_gui_data()
        latest_metrics = buffer.get_latest_metrics()
        
        print(f"  ✅ OptimizedDataBuffer working: {len(gui_data)} fields, latest power: {latest_metrics.get('total_power', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"  ❌ Optimized data structures test failed: {e}")
        return False

def test_adaptive_refresh():
    """Test adaptive refresh rate calculation"""
    print("\n🔍 Testing adaptive refresh functionality...")
    
    try:
        from mfc_streamlit_gui import SimulationRunner
        
        runner = SimulationRunner()
        
        # Test adaptive refresh calculation
        runner._calculate_adaptive_refresh(100.0)  # Fast simulation
        fast_refresh = runner.adaptive_refresh_interval
        
        runner._calculate_adaptive_refresh(5.0)   # Slow simulation
        slow_refresh = runner.adaptive_refresh_interval
        
        print(f"  ✅ Adaptive refresh working: fast={fast_refresh:.1f}s, slow={slow_refresh:.1f}s")
        
        return True
    except Exception as e:
        print(f"  ❌ Adaptive refresh test failed: {e}")
        return False

def test_memory_monitoring():
    """Test memory monitoring functionality"""
    print("\n🔍 Testing memory monitoring...")
    
    try:
        from mfc_streamlit_gui import SimulationRunner
        import psutil
        
        runner = SimulationRunner()
        runner.start_memory_mb = 100.0  # Mock start memory
        
        # Test memory monitoring (should not crash)
        runner._monitor_memory_usage()
        
        print("  ✅ Memory monitoring working without errors")
        return True
    except Exception as e:
        print(f"  ❌ Memory monitoring test failed: {e}")
        return False

def test_parquet_support():
    """Test Parquet file format support"""
    print("\n🔍 Testing Parquet format support...")
    
    try:
        import pandas as pd
        import pyarrow  # This will fail if pyarrow is not installed
        from pathlib import Path
        import tempfile
        
        # Create test data
        test_data = {
            'time_hours': [0.0, 1.0, 2.0],
            'total_power': [100.0, 110.0, 120.0],
            'reservoir_concentration': [10.0, 9.5, 9.0]
        }
        df = pd.DataFrame(test_data)
        
        # Test Parquet save/load
        with tempfile.TemporaryDirectory() as temp_dir:
            parquet_file = Path(temp_dir) / "test_data.parquet"
            
            # Save as Parquet
            df.to_parquet(parquet_file, compression='snappy', index=False)
            
            # Load back
            df_loaded = pd.read_parquet(parquet_file)
            
            # Verify data integrity
            assert len(df_loaded) == 3
            assert df_loaded['total_power'].iloc[0] == 100.0
            
            print(f"  ✅ Parquet support working: saved and loaded {len(df_loaded)} rows")
            
        return True
    except ImportError as e:
        print(f"  ❌ Parquet support test failed (missing dependency): {e}")
        return False
    except Exception as e:
        print(f"  ❌ Parquet support test failed: {e}")
        return False

def test_optimized_simulation_runner():
    """Test the optimized simulation runner initialization"""
    print("\n🔍 Testing optimized simulation runner...")
    
    try:
        from mfc_streamlit_gui import SimulationRunner
        from config.qlearning_config import DEFAULT_QLEARNING_CONFIG
        
        runner = SimulationRunner()
        
        # Test that all new attributes are present
        assert hasattr(runner, 'data_buffer'), "Missing data_buffer attribute"
        assert hasattr(runner, 'adaptive_refresh_interval'), "Missing adaptive_refresh_interval"
        assert hasattr(runner, 'simulation_speed_avg'), "Missing simulation_speed_avg"
        assert hasattr(runner, 'start_memory_mb'), "Missing start_memory_mb"
        
        # Test new methods
        gui_data = runner.get_live_data()
        latest_metrics = runner.get_latest_metrics()
        
        print("  ✅ Optimized SimulationRunner initialized with all new features")
        return True
    except Exception as e:
        print(f"  ❌ Optimized simulation runner test failed: {e}")
        return False

def test_lightweight_data_access():
    """Test that data access is now lightweight"""
    print("\n🔍 Testing lightweight data access...")
    
    try:
        from mfc_streamlit_gui import SimulationRunner, SimulationSnapshot
        import time
        
        runner = SimulationRunner()
        
        # Add many snapshots to simulate a long-running simulation
        start_time = time.time()
        for i in range(1000):
            snapshot = SimulationSnapshot(
                current_time=float(i),
                reservoir_concentration=10.0 - i * 0.001,
                outlet_concentration=9.5 - i * 0.001,
                total_power=100.0 + i * 0.1,
                total_current=50.0,
                system_voltage=2.0,
                flow_rate_ml_h=20.0,
                substrate_efficiency=0.95,
                biofilm_thickness_avg=150.0,
                q_action=1,
                reward=0.8,
                epsilon=0.1,
                step_number=i,
                progress_percent=i / 10.0
            )
            runner.data_buffer.add_snapshot(snapshot)
        
        # Test data access speed
        access_start = time.time()
        gui_data = runner.get_live_data()
        latest_metrics = runner.get_latest_metrics()
        access_time = time.time() - access_start
        
        total_time = time.time() - start_time
        
        print(f"  ✅ Lightweight data access: {access_time*1000:.1f}ms for GUI data, {total_time*1000:.1f}ms total")
        print(f"      Buffer limited to {len(runner.data_buffer.snapshots)} points (max: {runner.data_buffer.max_gui_points})")
        
        return True
    except Exception as e:
        print(f"  ❌ Lightweight data access test failed: {e}")
        return False

def run_all_tests():
    """Run all GUI optimization tests"""
    print("🚀 Running GUI Optimization Tests\n")
    print("=" * 50)
    
    tests = [
        test_optimized_data_structures,
        test_adaptive_refresh,
        test_memory_monitoring,
        test_parquet_support,
        test_optimized_simulation_runner,
        test_lightweight_data_access
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All GUI optimization tests passed!")
        return True
    else:
        print(f"⚠️ {failed} tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)