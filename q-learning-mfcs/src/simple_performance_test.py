#!/usr/bin/env python3
"""
Simple Performance Test: Compare simulation execution times
Test just the core simulation loop performance without complex initialization.

Created: 2025-07-29
"""

import os
import sys
import time
import threading
import queue
from datetime import datetime

# Add source directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_cli_simulation(duration_hours=0.5):
    """Test CLI simulation performance using the main run function"""
    print(f"🚀 Testing CLI mode ({duration_hours}h)...")
    
    start_time = time.time()
    
    try:
        # Import and run the main simulation function
        from mfc_gpu_accelerated import run_gpu_accelerated_simulation
        
        results, output_dir = run_gpu_accelerated_simulation(duration_hours)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Extract performance metrics
        sim_data = results.get('simulation', {})
        n_steps = len(sim_data.get('time_hours', []))
        
        metrics = {
            'mode': 'CLI',
            'duration_hours': duration_hours,
            'total_time': total_time,
            'steps_completed': n_steps,
            'steps_per_second': n_steps / total_time if total_time > 0 else 0,
            'final_power': results.get('performance_metrics', {}).get('final_power', 0),
            'output_dir': str(output_dir)
        }
        
        print(f"✅ CLI completed: {total_time:.2f}s, {metrics['steps_per_second']:.1f} steps/s")
        return metrics
        
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return None

def test_gui_simulation(duration_hours=0.5):
    """Test GUI simulation performance using SimulationRunner"""
    print(f"🚀 Testing GUI mode ({duration_hours}h)...")
    
    start_time = time.time()
    
    try:
        # Import GUI components
        from mfc_streamlit_gui import SimulationRunner
        from config.qlearning_config import QLearningConfig
        
        # Create default config
        config = QLearningConfig()
        
        # Initialize runner
        runner = SimulationRunner()
        
        # Start simulation
        success = runner.start_simulation(
            config=config,
            duration_hours=duration_hours,
            target_conc=25.0,
            gui_refresh_interval=0.5,  # Fast refresh for testing
            debug_mode=False
        )
        
        if not success:
            print("❌ Failed to start GUI simulation")
            return None
        
        # Wait for completion
        results = None
        timeout_start = time.time()
        
        while runner.is_running and (time.time() - timeout_start) < (duration_hours * 3600 + 60):  # Duration + 1 minute timeout
            try:
                result_type, result_data, output_dir = runner.results_queue.get(timeout=1.0)
                if result_type == 'completed':
                    results = result_data
                    break
                elif result_type == 'error':
                    print(f"❌ Simulation error: {result_data}")
                    return None
            except queue.Empty:
                continue
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Clean up
        runner.stop_simulation()
        
        if results is None:
            print("❌ GUI simulation timed out or failed")
            return None
        
        # Extract performance metrics
        sim_data = results.get('simulation', {})
        n_steps = len(sim_data.get('time_hours', []))
        
        metrics = {
            'mode': 'GUI',
            'duration_hours': duration_hours,
            'total_time': total_time,
            'steps_completed': n_steps,
            'steps_per_second': n_steps / total_time if total_time > 0 else 0,
            'final_power': results.get('performance_metrics', {}).get('final_power', 0),
            'output_dir': str(output_dir) if 'output_dir' in locals() else 'unknown'
        }
        
        print(f"✅ GUI completed: {total_time:.2f}s, {metrics['steps_per_second']:.1f} steps/s")
        return metrics
        
    except Exception as e:
        print(f"❌ GUI test failed: {e}")
        return None

def compare_performance():
    """Run performance comparison"""
    print("=" * 60)
    print("🔬 MFC Performance Test: GUI vs CLI")
    print("=" * 60)
    
    # Test parameters
    test_duration = 0.25  # 15 minutes
    
    # Run tests
    print(f"\\n⏱️  Testing {test_duration} hour simulations...")
    
    # CLI test
    cli_metrics = test_cli_simulation(test_duration)
    
    # Wait between tests
    print("\\n⏳ Waiting 5 seconds between tests...")
    time.sleep(5)
    
    # GUI test  
    gui_metrics = test_gui_simulation(test_duration)
    
    # Analysis
    print("\\n" + "=" * 60)
    print("📊 PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    if cli_metrics and gui_metrics:
        # Compare times
        cli_time = cli_metrics['total_time']
        gui_time = gui_metrics['total_time']
        
        overhead_pct = ((gui_time - cli_time) / cli_time) * 100 if cli_time > 0 else 0
        
        # Compare speeds
        cli_speed = cli_metrics['steps_per_second']
        gui_speed = gui_metrics['steps_per_second']
        
        speed_ratio = gui_speed / cli_speed if cli_speed > 0 else 0
        
        print(f"\\n⏱️  Execution Time:")
        print(f"   CLI: {cli_time:.2f} seconds")
        print(f"   GUI: {gui_time:.2f} seconds")
        print(f"   Overhead: {overhead_pct:+.1f}%")
        
        print(f"\\n⚡ Processing Speed:")
        print(f"   CLI: {cli_speed:.1f} steps/second")
        print(f"   GUI: {gui_speed:.1f} steps/second")
        print(f"   Ratio: {speed_ratio:.2f}x")
        
        print(f"\\n📊 Steps Completed:")
        print(f"   CLI: {cli_metrics['steps_completed']:,} steps")
        print(f"   GUI: {gui_metrics['steps_completed']:,} steps")
        
        print(f"\\n⚡ Final Power Output:")
        print(f"   CLI: {cli_metrics['final_power']:.3f} W")
        print(f"   GUI: {gui_metrics['final_power']:.3f} W")
        
        # Verdict
        print(f"\\n🎯 VERDICT:")
        if abs(overhead_pct) < 5:
            print("   ✅ No significant performance difference")
        elif overhead_pct > 10:
            print(f"   ⚠️  GUI shows {overhead_pct:.1f}% overhead - investigate")
        elif overhead_pct < -10:
            print(f"   🔄 GUI appears faster by {-overhead_pct:.1f}% - check measurement")
        else:
            print(f"   ℹ️  Minor difference: {overhead_pct:+.1f}% overhead")
            
    elif cli_metrics:
        print("❌ GUI test failed, only CLI results available")
        print(f"   CLI: {cli_metrics['total_time']:.2f}s, {cli_metrics['steps_per_second']:.1f} steps/s")
    elif gui_metrics:
        print("❌ CLI test failed, only GUI results available")
        print(f"   GUI: {gui_metrics['total_time']:.2f}s, {gui_metrics['steps_per_second']:.1f} steps/s")
    else:
        print("❌ Both tests failed - unable to perform comparison")
    
    print("\\n" + "=" * 60)

if __name__ == "__main__":
    compare_performance()