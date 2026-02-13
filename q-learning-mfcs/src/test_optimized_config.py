"""
Test Optimized Q-learning Configuration

Loads optimized hyperparameters from Bayesian optimization and runs
extended simulations to validate substrate control performance.

Created: 2025-07-26
"""

import os
import sys
import json
import argparse
from typing import Dict, Any

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hyperparameter_optimization import apply_optimized_config
from mfc_recirculation_control import run_mfc_simulation


def load_optimized_config(results_file: str) -> Dict[str, Any]:
    """Load optimized configuration from JSON file."""
    
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return results["best_config"]


def run_validation_simulation(
    best_config: Dict[str, Any],
    duration_hours: int = 1000,
    target_concentration: float = 25.0,
    n_cells: int = 5
) -> Dict[str, Any]:
    """
    Run extended validation simulation with optimized parameters.
    
    Args:
        best_config: Optimized hyperparameters
        duration_hours: Full simulation duration
        target_concentration: Target substrate concentration
        n_cells: Number of MFC cells
        
    Returns:
        Simulation results dictionary
    """
    
    # Apply optimized configuration
    qlearning_config = apply_optimized_config(best_config)
    
    print("="*60)
    print("RUNNING VALIDATION SIMULATION")
    print("="*60)
    print(f"Duration: {duration_hours} hours")
    print(f"Target concentration: {target_concentration} mM")
    print(f"Number of cells: {n_cells}")
    print("\nOptimized hyperparameters:")
    for key, value in best_config.items():
        print(f"  {key}: {value:.4f}")
    print()
    
    # Run simulation
    results = run_mfc_simulation(
        duration_hours=duration_hours,
        output_dir="../data/simulation_data",
        config=qlearning_config,
        n_cells=n_cells,
        initial_substrate_concentration=target_concentration,
        user_suffix="optimized_validation",
        verbose=True
    )
    
    return results


def analyze_validation_results(results: Dict[str, Any], target_concentration: float = 25.0):
    """Analyze and report validation simulation performance."""
    
    reservoir_conc = results.get("reservoir_concentration", [])
    outlet_conc = results.get("outlet_concentration", [])
    power_output = results.get("power_output", [])
    
    if not reservoir_conc:
        print("ERROR: No reservoir concentration data found")
        return
    
    import numpy as np
    
    reservoir_conc = np.array(reservoir_conc)
    outlet_conc = np.array(outlet_conc)
    power_output = np.array(power_output)
    
    # Performance analysis
    print("\n" + "="*60)
    print("VALIDATION RESULTS ANALYSIS")
    print("="*60)
    
    # Substrate control performance
    final_concentration = reservoir_conc[-1]
    mean_concentration = np.mean(reservoir_conc)
    concentration_std = np.std(reservoir_conc)
    max_deviation = np.max(np.abs(reservoir_conc - target_concentration))
    
    print(f"Target concentration: {target_concentration:.1f} mM")
    print(f"Final concentration: {final_concentration:.1f} mM")
    print(f"Mean concentration: {mean_concentration:.1f} ± {concentration_std:.1f} mM")
    print(f"Maximum deviation: {max_deviation:.1f} mM")
    
    # Stability assessment (last 25% of simulation)
    last_quarter_idx = len(reservoir_conc) * 3 // 4
    last_quarter_conc = reservoir_conc[last_quarter_idx:]
    stability_std = np.std(last_quarter_conc)
    stability_mean = np.mean(last_quarter_conc)
    
    print(f"\nStability (last 25% of simulation):")
    print(f"  Mean: {stability_mean:.1f} mM")
    print(f"  Std deviation: {stability_std:.1f} mM")
    print(f"  Coefficient of variation: {stability_std/stability_mean*100:.1f}%")
    
    # Power performance
    if len(power_output) > 0:
        final_power = power_output[-1]
        mean_power = np.mean(power_output)
        max_power = np.max(power_output)
    else:
        final_power = mean_power = max_power = 0.0
    
    print(f"\nPower performance:")
    print(f"  Final power: {final_power:.1f} W")
    print(f"  Mean power: {mean_power:.1f} W")
    print(f"  Maximum power: {max_power:.1f} W")
    
    # Control effectiveness
    within_tolerance_5 = np.sum(np.abs(reservoir_conc - target_concentration) <= 5.0)
    within_tolerance_2 = np.sum(np.abs(reservoir_conc - target_concentration) <= 2.0)
    total_points = len(reservoir_conc)
    
    print(f"\nControl effectiveness:")
    print(f"  Within ±5 mM: {within_tolerance_5/total_points*100:.1f}% of time")
    print(f"  Within ±2 mM: {within_tolerance_2/total_points*100:.1f}% of time")
    
    # Overall assessment
    control_success = (
        abs(final_concentration - target_concentration) <= 5.0 and
        stability_std <= 3.0 and
        within_tolerance_5/total_points >= 0.8
    )
    
    print(f"\nOverall assessment: {'SUCCESS' if control_success else 'NEEDS IMPROVEMENT'}")
    if control_success:
        print("✓ Substrate concentration well controlled")
        print("✓ System demonstrates stability")
        print("✓ Performance meets target criteria")
    else:
        print("✗ Further optimization may be needed")
        if abs(final_concentration - target_concentration) > 5.0:
            print("  - Final concentration exceeds tolerance")
        if stability_std > 3.0:
            print("  - System shows high variability")
        if within_tolerance_5/total_points < 0.8:
            print("  - Control effectiveness below target")


def main():
    """Main function for testing optimized configuration."""
    
    parser = argparse.ArgumentParser(description="Test Optimized Q-learning Configuration")
    parser.add_argument("results_file", 
                       help="Path to optimization results JSON file")
    parser.add_argument("--duration", type=int, default=1000,
                       help="Simulation duration in hours (default: 1000)")
    parser.add_argument("--target", type=float, default=25.0,
                       help="Target substrate concentration in mM (default: 25.0)")
    parser.add_argument("--cells", type=int, default=5,
                       help="Number of MFC cells (default: 5)")
    
    args = parser.parse_args()
    
    try:
        # Load optimized configuration
        best_config = load_optimized_config(args.results_file)
        
        # Run validation simulation
        results = run_validation_simulation(
            best_config=best_config,
            duration_hours=args.duration,
            target_concentration=args.target,
            n_cells=args.cells
        )
        
        # Analyze results
        analyze_validation_results(results, args.target)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()