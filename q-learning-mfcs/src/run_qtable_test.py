#!/usr/bin/env python3
"""
Quick test to verify Q-table loading in the updated GPU simulation.
"""

from mfc_gpu_accelerated import run_gpu_accelerated_simulation

# Run short 24-hour test with Q-table loading
print("🧪 Running 24-hour test with Q-table loading...")
results, output_dir = run_gpu_accelerated_simulation(24)
print(f"✅ Test complete! Results saved to: {output_dir}")