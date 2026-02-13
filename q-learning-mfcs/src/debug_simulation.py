#!/usr/bin/env python3
"""
Debug script to test GPU simulation outside of GUI
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mfc_gpu_accelerated import GPUAcceleratedMFC
from config.qlearning_config import DEFAULT_QLEARNING_CONFIG
import time

def test_simulation_steps():
    """Test if simulation steps are working properly"""
    
    print("🔍 Testing GPU simulation steps...")
    
    # Initialize simulation
    config = DEFAULT_QLEARNING_CONFIG
    config.n_cells = 5
    config.electrode_area_per_cell = 10.0 * 1e-4  # 10 cm² in m²
    config.substrate_target_concentration = 25.0
    
    try:
        mfc_sim = GPUAcceleratedMFC(config)
        print("✅ Simulation initialized successfully")
        
        # Test a few timesteps
        dt_hours = 0.1
        print(f"\n🚀 Running 10 test timesteps with dt={dt_hours}h...")
        
        for step in range(10):
            print(f"Step {step+1}/10...", end=" ")
            start_time = time.time()
            
            # Run timestep
            result = mfc_sim.simulate_timestep(dt_hours)
            
            step_time = time.time() - start_time
            
            print(f"✅ Complete in {step_time:.3f}s")
            print(f"   Reservoir: {mfc_sim.reservoir_concentration:.2f} mM")
            print(f"   Power: {result['total_power']:.6f} W")
            print(f"   Action: {result['action']}")
            print(f"   Reward: {result['reward']:.1f}")
            print(f"   Epsilon: {result['epsilon']:.6f}")
            print()
            
            # Check if simulation is stuck
            if step_time > 1.0:  # If a single step takes more than 1 second
                print("⚠️  WARNING: Step taking too long!")
                break
                
        print("🎯 Simulation step test completed successfully!")
        
        # Test timing for different durations
        print(f"\n⏱️  Testing timing for 1 hour simulation...")
        n_steps = int(1.0 / dt_hours)  # 1 hour
        print(f"Expected steps: {n_steps}")
        
        start_time = time.time()
        for step in range(min(n_steps, 100)):  # Cap at 100 steps for testing
            result = mfc_sim.simulate_timestep(dt_hours)
            if step % 20 == 0:  # Progress every 20 steps
                elapsed = time.time() - start_time
                progress = step / min(n_steps, 100) * 100
                print(f"   Progress: {progress:.1f}% - {elapsed:.1f}s elapsed")
        
        total_time = time.time() - start_time
        print(f"✅ 1-hour simulation test completed in {total_time:.2f}s")
        
        # Cleanup
        mfc_sim.cleanup_gpu_resources()
        print("🧹 Resources cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ Simulation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gui_simulation_setup():
    """Test the same setup as GUI simulation"""
    
    print("\n🔍 Testing GUI simulation setup...")
    
    # Same parameters as GUI
    config = DEFAULT_QLEARNING_CONFIG
    duration_hours = 1  # 1 hour test
    n_cells = 5
    electrode_area_m2 = 10.0 * 1e-4  # 10 cm² in m²
    target_conc = 25.0
    gui_refresh_interval = 5.0
    
    # Update config with GUI values
    config.n_cells = n_cells
    config.electrode_area_per_cell = electrode_area_m2
    config.substrate_target_concentration = target_conc
    
    try:
        # Initialize simulation
        mfc_sim = GPUAcceleratedMFC(config)
        print("✅ GUI-style simulation initialized")
        
        # Simulation parameters (same as GUI)
        dt_hours = 0.1
        n_steps = int(duration_hours / dt_hours)
        
        # Calculate save interval (same as GUI)
        gui_refresh_hours = gui_refresh_interval / 3600.0
        save_interval_steps = max(1, int(gui_refresh_hours / dt_hours))
        min_save_steps = 10
        save_interval_steps = min(save_interval_steps, min_save_steps)
        
        print(f"Duration: {duration_hours}h ({n_steps} steps)")
        print(f"Save interval: every {save_interval_steps} steps")
        
        # Progress tracking (same as GUI)
        results = {
            'time_hours': [],
            'reservoir_concentration': [],
            'outlet_concentration': [],
            'total_power': [],
            'biofilm_thicknesses': [],
            'substrate_addition_rate': [],
            'q_action': [],
            'epsilon': [],
            'reward': []
        }
        
        # Run simulation (same as GUI)
        print(f"\n🚀 Running {duration_hours}h GUI-style simulation...")
        start_time = time.time()
        
        for step in range(n_steps):
            current_time = step * dt_hours
            
            # Simulate timestep
            step_results = mfc_sim.simulate_timestep(dt_hours)
            
            # Store results at GUI-synchronized intervals
            if step % save_interval_steps == 0:
                results['time_hours'].append(current_time)
                results['reservoir_concentration'].append(float(mfc_sim.reservoir_concentration))
                results['outlet_concentration'].append(float(mfc_sim.outlet_concentration))
                results['total_power'].append(step_results['total_power'])
                results['biofilm_thicknesses'].append([float(x) for x in mfc_sim.biofilm_thicknesses])
                results['substrate_addition_rate'].append(step_results['substrate_addition'])
                results['q_action'].append(step_results['action'])
                results['epsilon'].append(step_results['epsilon'])
                results['reward'].append(step_results['reward'])
                
                print(f"   Saved data point at t={current_time:.1f}h (step {step})")
            
            # Progress indicator
            if step % (n_steps // 10) == 0:
                progress = step / n_steps * 100
                elapsed = time.time() - start_time
                print(f"   Progress: {progress:.0f}% - {elapsed:.1f}s elapsed")
        
        total_time = time.time() - start_time
        print(f"✅ GUI-style simulation completed in {total_time:.2f}s")
        print(f"📊 Collected {len(results['time_hours'])} data points")
        
        # Cleanup
        mfc_sim.cleanup_gpu_resources()
        print("🧹 Resources cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ GUI simulation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔬 GPU Simulation Debug Test")
    print("=" * 50)
    
    # Test 1: Basic simulation steps
    test1_result = test_simulation_steps()
    
    # Test 2: GUI-style simulation
    test2_result = test_gui_simulation_setup()
    
    print("\n" + "=" * 50)
    print("📋 DEBUG TEST SUMMARY")
    print("=" * 50)
    print(f"Basic simulation steps: {'✅ PASS' if test1_result else '❌ FAIL'}")
    print(f"GUI-style simulation:   {'✅ PASS' if test2_result else '❌ FAIL'}")
    
    if test1_result and test2_result:
        print("\n🎉 All tests passed! The issue might be in GUI threading or display.")
        print("💡 Suggestions:")
        print("   - Check Streamlit session state")
        print("   - Verify thread communication")
        print("   - Check auto-refresh logic")
    else:
        print("\n⚠️  Tests failed! The issue is in the simulation itself.")
        print("💡 Check error messages above for details.")