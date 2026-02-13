#!/usr/bin/env python3
"""
Test script to verify Q-table loading improves substrate control.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mfc_gpu_accelerated import GPUAcceleratedMFC
from config.qlearning_config import DEFAULT_QLEARNING_CONFIG
import numpy as np

def test_qtable_performance():
    """Test that loaded Q-table improves control performance."""
    
    print("🧪 Testing Q-table loading performance...")
    
    # Initialize simulation with pre-trained Q-table
    mfc = GPUAcceleratedMFC(DEFAULT_QLEARNING_CONFIG)
    
    print(f"\n📊 Initial State:")
    print(f"   Reservoir: {mfc.reservoir_concentration:.2f} mM")
    print(f"   Target: {mfc.target_concentration:.2f} mM")
    print(f"   Epsilon: {mfc.epsilon:.4f}")
    
    # Run short simulation (1 hour = 10 timesteps)
    results = []
    target = 25.0
    
    print("\n⏱️  Running 1-hour test simulation...")
    
    for step in range(10):  # 1 hour with 6-minute timesteps
        dt_hours = 0.1
        step_result = mfc.simulate_timestep(dt_hours)
        
        reservoir_conc = float(mfc.reservoir_concentration)
        deviation = abs(reservoir_conc - target)
        
        results.append({
            'step': step,
            'time_min': step * 6,
            'reservoir_conc': reservoir_conc,
            'deviation': deviation,
            'action': step_result['action'],
            'reward': step_result['reward'],
            'substrate_addition': step_result['substrate_addition']
        })
        
        print(f"   Step {step+1:2d}: {reservoir_conc:5.1f} mM | "
              f"Deviation: {deviation:4.1f} mM | "
              f"Action: {step_result['action']} | "
              f"Reward: {step_result['reward']:6.1f}")
    
    # Analyze performance
    final_conc = results[-1]['reservoir_conc']
    mean_deviation = np.mean([r['deviation'] for r in results])
    max_deviation = np.max([r['deviation'] for r in results])
    
    print(f"\n📈 PERFORMANCE SUMMARY:")
    print(f"   Final concentration: {final_conc:.2f} mM")
    print(f"   Mean deviation: {mean_deviation:.2f} mM")
    print(f"   Max deviation: {max_deviation:.2f} mM")
    print(f"   Control quality: {'✅ GOOD' if mean_deviation < 5.0 else '⚠️  POOR'}")
    
    # Check if pre-trained weights are being used effectively
    actions_used = [r['action'] for r in results]
    unique_actions = len(set(actions_used))
    
    print(f"\n🎯 Q-LEARNING ANALYSIS:")
    print(f"   Unique actions used: {unique_actions}/10")
    print(f"   Actions: {actions_used}")
    print(f"   Exploitation level: {'✅ HIGH' if unique_actions <= 3 else '⚠️  RANDOM'}")
    
    return mean_deviation < 5.0 and unique_actions <= 3

if __name__ == "__main__":
    success = test_qtable_performance()
    print(f"\n🎉 Q-table loading test: {'✅ PASSED' if success else '❌ FAILED'}")