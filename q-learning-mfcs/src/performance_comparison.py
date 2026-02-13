#!/usr/bin/env python3
"""
Performance Comparison: GUI vs CLI Simulation
Compare simulation performance between GUI and CLI modes.

Created: 2025-07-29
"""

import os
import sys
import time
import json
import threading
import queue
from datetime import datetime
from pathlib import Path
import numpy as np

# Add source directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def benchmark_cli_simulation(duration_hours=1.0, config_path="configs/research_optimization.yaml"):
    """Benchmark CLI simulation performance"""
    from mfc_gpu_accelerated import GPUAcceleratedMFC
    
    print(f"🚀 Starting CLI benchmark ({duration_hours}h simulation)...")
    
    start_time = time.time()
    
    # Initialize simulation
    simulation = GPUAcceleratedMFC(config_file=config_path)
    
    # Set simulation parameters
    simulation.total_hours = duration_hours
    simulation.target_concentration = 25.0
    
    init_time = time.time() - start_time
    
    # Run simulation
    sim_start_time = time.time()
    results = simulation.run_optimization()
    sim_end_time = time.time()
    
    total_time = time.time() - start_time
    sim_time = sim_end_time - sim_start_time
    
    # Collect performance metrics
    metrics = {
        'mode': 'CLI',
        'duration_hours': duration_hours,
        'init_time': init_time,
        'simulation_time': sim_time,
        'total_time': total_time,
        'steps_per_second': len(results.get('times', [])) / sim_time if sim_time > 0 else 0,
        'final_power': results.get('performance_metrics', {}).get('mean_power', 0),
        'memory_usage_mb': 0,  # Would need psutil for this
        'gpu_backend': simulation.gpu_backend if hasattr(simulation, 'gpu_backend') else 'unknown'
    }
    
    # Cleanup
    try:
        simulation.cleanup_gpu_resources()
    except Exception as e:
        print(f"Warning: Cleanup failed: {e}")
    
    print(f"✅ CLI benchmark completed in {total_time:.2f}s")
    return metrics

def benchmark_gui_simulation(duration_hours=1.0, config_path="configs/research_optimization.yaml"):
    """Benchmark GUI simulation performance (without actual GUI)"""
    from mfc_streamlit_gui import SimulationRunner
    from utils.config_loader import load_config
    
    print(f"🚀 Starting GUI backend benchmark ({duration_hours}h simulation)...")
    
    start_time = time.time()
    
    # Load configuration
    config = load_config(config_path)
    
    # Initialize GUI simulation runner
    runner = SimulationRunner()
    
    init_time = time.time() - start_time
    
    # Start simulation in background
    sim_start_time = time.time()
    success = runner.start_simulation(
        config=config,
        duration_hours=duration_hours,
        target_conc=25.0,
        gui_refresh_interval=1.0,  # Faster refresh for benchmark
        debug_mode=False
    )
    
    if not success:
        print("❌ Failed to start GUI simulation")
        return None
    
    # Wait for completion
    print("⏳ Waiting for simulation to complete...")
    results = None
    
    while runner.is_running:
        try:
            result_type, result_data, output_dir = runner.results_queue.get(timeout=0.1)
            if result_type == 'completed':
                results = result_data
                break
            elif result_type == 'error':
                print(f"❌ Simulation error: {result_data}")
                break
        except queue.Empty:
            continue
    
    sim_end_time = time.time()
    total_time = time.time() - start_time
    sim_time = sim_end_time - sim_start_time
    
    # Clean up
    runner.stop_simulation()
    
    if results is None:
        print("❌ No results received from GUI simulation")
        return None
    
    # Collect performance metrics
    metrics = {
        'mode': 'GUI',
        'duration_hours': duration_hours,
        'init_time': init_time,
        'simulation_time': sim_time,
        'total_time': total_time,
        'steps_per_second': len(results.get('times', [])) / sim_time if sim_time > 0 else 0,
        'final_power': results.get('performance_metrics', {}).get('mean_power', 0),
        'memory_usage_mb': 0,  # Would need psutil for this
        'gpu_backend': 'unknown'  # GUI runner doesn't expose this directly
    }
    
    print(f"✅ GUI benchmark completed in {total_time:.2f}s")
    return metrics

def run_performance_comparison():
    """Run comprehensive performance comparison"""
    
    print("=" * 60)
    print("🔬 MFC Performance Comparison: GUI vs CLI")
    print("=" * 60)
    
    # Test configurations
    test_durations = [0.5, 1.0, 2.0]  # Hours
    config_file = "configs/research_optimization.yaml"
    
    results = []
    
    for duration in test_durations:
        print(f"\\n🕐 Testing {duration} hour simulation...")
        
        # CLI benchmark
        try:
            cli_metrics = benchmark_cli_simulation(duration, config_file)
            if cli_metrics:
                results.append(cli_metrics)
                print(f"   CLI: {cli_metrics['simulation_time']:.2f}s ({cli_metrics['steps_per_second']:.1f} steps/s)")
        except Exception as e:
            print(f"   ❌ CLI benchmark failed: {e}")
        
        # Small delay between tests
        time.sleep(2)
        
        # GUI benchmark
        try:
            gui_metrics = benchmark_gui_simulation(duration, config_file)
            if gui_metrics:
                results.append(gui_metrics)
                print(f"   GUI: {gui_metrics['simulation_time']:.2f}s ({gui_metrics['steps_per_second']:.1f} steps/s)")
        except Exception as e:
            print(f"   ❌ GUI benchmark failed: {e}")
            
        # Delay between duration tests
        time.sleep(3)
    
    # Analysis
    print("\\n" + "=" * 60)
    print("📊 PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    if not results:
        print("❌ No successful benchmarks to analyze")
        return
    
    # Group by mode
    cli_results = [r for r in results if r['mode'] == 'CLI']
    gui_results = [r for r in results if r['mode'] == 'GUI']
    
    if cli_results and gui_results:
        # Calculate averages
        cli_avg_time = np.mean([r['simulation_time'] for r in cli_results])
        gui_avg_time = np.mean([r['simulation_time'] for r in gui_results])
        
        cli_avg_steps = np.mean([r['steps_per_second'] for r in cli_results])
        gui_avg_steps = np.mean([r['steps_per_second'] for r in gui_results])
        
        print(f"\\n🏃 Average Simulation Time:")
        print(f"   CLI: {cli_avg_time:.2f}s")
        print(f"   GUI: {gui_avg_time:.2f}s")
        
        if gui_avg_time > 0:
            overhead_pct = ((gui_avg_time - cli_avg_time) / cli_avg_time) * 100
            print(f"   GUI Overhead: {overhead_pct:+.1f}%")
        
        print(f"\\n⚡ Average Processing Speed:")
        print(f"   CLI: {cli_avg_steps:.1f} steps/second")
        print(f"   GUI: {gui_avg_steps:.1f} steps/second")
        
        if cli_avg_steps > 0:
            speed_ratio = gui_avg_steps / cli_avg_steps
            print(f"   GUI/CLI Speed Ratio: {speed_ratio:.2f}x")
    
    # Detailed results table
    print(f"\\n📋 Detailed Results:")
    print(f"{'Mode':<4} {'Duration':<8} {'Init':<6} {'Sim':<8} {'Total':<8} {'Steps/s':<8} {'Power':<8}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['mode']:<4} {r['duration_hours']:<8.1f} {r['init_time']:<6.2f} "
              f"{r['simulation_time']:<8.2f} {r['total_time']:<8.2f} "
              f"{r['steps_per_second']:<8.1f} {r['final_power']:<8.3f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"../data/performance_comparison_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'test_config': config_file,
            'results': results,
            'summary': {
                'cli_avg_simulation_time': cli_avg_time if cli_results else None,
                'gui_avg_simulation_time': gui_avg_time if gui_results else None,
                'gui_overhead_percent': overhead_pct if cli_results and gui_results else None
            }
        }, f, indent=2)
    
    print(f"\\n💾 Results saved to: {results_file}")
    
    # Final verdict
    print(f"\\n🎯 VERDICT:")
    if cli_results and gui_results and abs(overhead_pct) < 5:
        print("   ✅ No significant performance difference between GUI and CLI modes")
    elif cli_results and gui_results and overhead_pct > 10:
        print(f"   ⚠️ GUI mode shows {overhead_pct:.1f}% overhead - investigate threading/data sync")
    elif cli_results and gui_results:
        print(f"   ℹ️ Minor performance difference: {overhead_pct:+.1f}% GUI overhead")
    else:
        print("   ❓ Insufficient data for comparison")

if __name__ == "__main__":
    run_performance_comparison()