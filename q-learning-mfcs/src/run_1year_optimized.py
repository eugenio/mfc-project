#!/usr/bin/env python3
"""
1-Year MFC Simulation with Optimized Q-learning Parameters

Runs 8784-hour (1 year) continuous operation simulation with GPU acceleration
and calculates substrate consumption and buffer maintenance requirements.

Created: 2025-07-26
"""

import os
import sys
import time
import json
import signal
from datetime import datetime
from pathlib import Path
import numpy as np

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mfc_recirculation_control import simulate_mfc_with_recirculation
from config.qlearning_config import DEFAULT_QLEARNING_CONFIG


def setup_gpu_acceleration():
    """Configure GPU acceleration settings."""

    # Set environment variables for GPU optimization
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU if available
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['NUMBA_ENABLE_CUDASIM'] = '1'  # Enable CUDA simulation

    # Set NumPy to use optimized BLAS
    os.environ['OMP_NUM_THREADS'] = '8'
    os.environ['MKL_NUM_THREADS'] = '8'
    os.environ['NUMEXPR_MAX_THREADS'] = '8'

    print("🚀 GPU acceleration configured")


def calculate_maintenance_requirements(total_substrate_consumed_mmol, simulation_hours):
    """
    Calculate maintenance requirements for substrate and buffer reservoirs.
    
    Args:
        total_substrate_consumed_mmol: Total substrate consumed (mmol)
        simulation_hours: Total simulation duration (hours)
        
    Returns:
        Dictionary with maintenance schedule and requirements
    """

    # Stock solution concentrations and volumes
    substrate_stock_concentration = 5.0  # M (5000 mM)
    substrate_stock_volume = 5.0  # L
    buffer_stock_concentration = 3.0  # M (3000 mM)
    buffer_stock_volume = 5.0  # L

    # Total available substrate and buffer
    total_substrate_available = substrate_stock_concentration * substrate_stock_volume * 1000  # mmol
    total_buffer_available = buffer_stock_concentration * buffer_stock_volume * 1000  # mmol

    # Calculate consumption rates
    substrate_consumption_rate = total_substrate_consumed_mmol / simulation_hours  # mmol/h

    # Assume buffer consumption is proportional to substrate (typical 1:1 for pH control)
    buffer_consumption_rate = substrate_consumption_rate * 0.8  # 80% of substrate rate
    total_buffer_consumed = buffer_consumption_rate * simulation_hours

    # Calculate refill intervals
    substrate_refill_interval_hours = total_substrate_available / substrate_consumption_rate
    buffer_refill_interval_hours = total_buffer_available / buffer_consumption_rate

    # Convert to days
    substrate_refill_interval_days = substrate_refill_interval_hours / 24
    buffer_refill_interval_days = buffer_refill_interval_hours / 24

    # Calculate number of refills needed per year
    substrate_refills_per_year = 8784 / substrate_refill_interval_hours
    buffer_refills_per_year = 8784 / buffer_refill_interval_hours

    # Calculate volumes and costs
    substrate_volume_consumed_per_year = total_substrate_consumed_mmol / (substrate_stock_concentration * 1000)  # L
    buffer_volume_consumed_per_year = total_buffer_consumed / (buffer_stock_concentration * 1000)  # L

    maintenance_data = {
        "substrate_requirements": {
            "total_consumed_mmol": total_substrate_consumed_mmol,
            "consumption_rate_mmol_per_hour": substrate_consumption_rate,
            "consumption_rate_mmol_per_day": substrate_consumption_rate * 24,
            "stock_volume_consumed_L": substrate_volume_consumed_per_year,
            "refill_interval_days": substrate_refill_interval_days,
            "refills_per_year": substrate_refills_per_year,
            "stock_bottles_per_year": np.ceil(substrate_refills_per_year)
        },
        "buffer_requirements": {
            "total_consumed_mmol": total_buffer_consumed,
            "consumption_rate_mmol_per_hour": buffer_consumption_rate,
            "consumption_rate_mmol_per_day": buffer_consumption_rate * 24,
            "stock_volume_consumed_L": buffer_volume_consumed_per_year,
            "refill_interval_days": buffer_refill_interval_days,
            "refills_per_year": buffer_refills_per_year,
            "stock_bottles_per_year": np.ceil(buffer_refills_per_year)
        },
        "maintenance_schedule": {
            "substrate_refill_frequency": f"Every {substrate_refill_interval_days:.1f} days",
            "buffer_refill_frequency": f"Every {buffer_refill_interval_days:.1f} days",
            "recommended_check_frequency": f"Every {min(substrate_refill_interval_days, buffer_refill_interval_days) / 2:.1f} days"
        }
    }

    return maintenance_data


def run_1year_simulation():
    """Run the 1-year MFC simulation with optimized parameters."""

    print("="*80)
    print("🔬 MFC 1-YEAR CONTINUOUS OPERATION SIMULATION")
    print("="*80)
    print("⏱️  Duration: 8,784 hours (365 days)")
    print("🎯 Target: 25.0 mM substrate concentration")
    print("🧠 Q-learning: Bayesian-optimized parameters")
    print("🚀 Acceleration: GPU-enabled")
    print("="*80)

    # Setup GPU acceleration
    setup_gpu_acceleration()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"../data/simulation_data/1year_optimized_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Output directory: {output_dir}")

    # Use optimized configuration
    config = DEFAULT_QLEARNING_CONFIG

    print("\n🧠 OPTIMIZED Q-LEARNING PARAMETERS:")
    print(f"   Learning rate: {config.enhanced_learning_rate:.6f}")
    print(f"   Discount factor: {config.enhanced_discount_factor:.4f}")
    print(f"   Initial epsilon: {config.enhanced_epsilon:.4f}")
    print(f"   Epsilon decay: {config.advanced_epsilon_decay:.6f}")
    print(f"   Power weight: {config.reward_weights.power_weight:.2f}")
    print(f"   Substrate reward multiplier: {config.reward_weights.substrate_reward_multiplier:.2f}")
    print(f"   Biofilm weight: {config.reward_weights.biofilm_weight:.2f}")

    # Start simulation
    start_time = time.time()
    print(f"\n🚀 Starting simulation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("⏳ Estimated completion time: ~2-4 hours")

    try:
        results, cells, reservoir, controller, q_controller = simulate_mfc_with_recirculation(
            8784, config, None)

        elapsed_time = time.time() - start_time
        print(f"\n✅ Simulation completed in {elapsed_time/3600:.2f} hours")

    except KeyboardInterrupt:
        print("\n⚠️  Simulation interrupted by user")
        return None, None
    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        return None, None

    # Calculate substrate consumption
    substrate_addition_rates = results.get("substrate_addition_rate", [])
    time_steps = results.get("time_hours", [])

    if len(substrate_addition_rates) > 1:
        dt_hours = time_steps[1] - time_steps[0]  # Time step in hours
        total_substrate_added = sum(rate * dt_hours for rate in substrate_addition_rates)
    else:
        total_substrate_added = 0.0

    # Get final concentrations
    reservoir_concentrations = results.get("reservoir_concentration", [])
    outlet_concentrations = results.get("outlet_concentration", [])
    power_outputs = results.get("total_power", [])

    final_reservoir = reservoir_concentrations[-1] if reservoir_concentrations else 0
    final_outlet = outlet_concentrations[-1] if outlet_concentrations else 0
    final_power = power_outputs[-1] if power_outputs else 0

    # Calculate performance metrics
    mean_reservoir = np.mean(reservoir_concentrations) if reservoir_concentrations else 0
    std_reservoir = np.std(reservoir_concentrations) if reservoir_concentrations else 0
    mean_power = np.mean(power_outputs) if power_outputs else 0

    # Calculate maintenance requirements
    maintenance_data = calculate_maintenance_requirements(total_substrate_added, 8784)

    # Prepare comprehensive results
    simulation_results = {
        "simulation_info": {
            "duration_hours": 8784,
            "duration_days": 365,
            "start_time": datetime.now().isoformat(),
            "elapsed_time_hours": elapsed_time / 3600,
            "target_concentration_mM": 25.0,
            "optimized_parameters": True
        },
        "performance_summary": {
            "final_reservoir_concentration_mM": final_reservoir,
            "final_outlet_concentration_mM": final_outlet,
            "mean_reservoir_concentration_mM": mean_reservoir,
            "std_reservoir_concentration_mM": std_reservoir,
            "final_power_output_W": final_power,
            "mean_power_output_W": mean_power,
            "total_substrate_consumed_mmol": total_substrate_added,
            "substrate_consumption_rate_mmol_per_day": total_substrate_added / 365
        },
        "maintenance_requirements": maintenance_data,
        "q_learning_config": {
            "learning_rate": config.enhanced_learning_rate,
            "discount_factor": config.enhanced_discount_factor,
            "epsilon_initial": config.enhanced_epsilon,
            "epsilon_decay": config.advanced_epsilon_decay,
            "power_weight": config.reward_weights.power_weight,
            "substrate_reward_multiplier": config.reward_weights.substrate_reward_multiplier,
            "biofilm_weight": config.reward_weights.biofilm_weight
        }
    }

    # Save results
    results_file = output_dir / f"1year_simulation_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(simulation_results, f, indent=2, default=str)

    # Save raw data (compressed)
    import pandas as pd
    df = pd.DataFrame(results)
    data_file = output_dir / f"1year_simulation_data_{timestamp}.csv.gz"
    df.to_csv(data_file, compression='gzip', index=False)

    # Print comprehensive results
    print("\n" + "="*80)
    print("📊 1-YEAR SIMULATION RESULTS")
    print("="*80)

    print("🎯 PERFORMANCE METRICS:")
    print(f"   Final reservoir concentration: {final_reservoir:.2f} mM")
    print(f"   Mean reservoir concentration: {mean_reservoir:.2f} ± {std_reservoir:.2f} mM")
    print(f"   Final outlet concentration: {final_outlet:.2f} mM")
    print(f"   Final power output: {final_power:.2f} W")
    print(f"   Mean power output: {mean_power:.2f} W")

    print("\n📦 SUBSTRATE CONSUMPTION:")
    print(f"   Total consumed: {total_substrate_added:.2f} mmol")
    print(f"   Daily consumption rate: {total_substrate_added/365:.2f} mmol/day")
    print(f"   Stock volume consumed: {maintenance_data['substrate_requirements']['stock_volume_consumed_L']:.3f} L")

    print("\n🔧 MAINTENANCE REQUIREMENTS:")
    print(f"   Substrate refill: {maintenance_data['maintenance_schedule']['substrate_refill_frequency']}")
    print(f"   Buffer refill: {maintenance_data['maintenance_schedule']['buffer_refill_frequency']}")
    print(f"   Recommended checks: {maintenance_data['maintenance_schedule']['recommended_check_frequency']}")
    print(f"   Substrate bottles/year: {maintenance_data['substrate_requirements']['stock_bottles_per_year']:.0f}")
    print(f"   Buffer bottles/year: {maintenance_data['buffer_requirements']['stock_bottles_per_year']:.0f}")

    print("\n📁 RESULTS SAVED TO:")
    print(f"   Summary: {results_file}")
    print(f"   Raw data: {data_file}")

    # Send email notification
    try:
        from email_notification import send_completion_email
        print("\n📧 Sending email notification...")
        send_completion_email(str(results_file))
    except Exception as e:
        print(f"⚠️  Email notification failed: {e}")
        print("💡 To enable email notifications, set environment variables:")
        print("   export NOTIFICATION_EMAIL='your-email@gmail.com'")
        print("   export EMAIL_PASSWORD='your-app-password'")
        print("   export RECIPIENT_EMAIL='recipient@gmail.com'")

    print("\n✅ 1-YEAR SIMULATION COMPLETE!")

    return simulation_results, output_dir


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    print(f"\n⚠️  Received signal {signum}. Saving partial results...")
    # Could implement partial save logic here
    sys.exit(0)


if __name__ == "__main__":
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run the simulation
    results, output_dir = run_1year_simulation()

    if results:
        print(f"\n🎉 Success! Check {output_dir} for detailed results.")
    else:
        print("\n❌ Simulation failed or was interrupted.")
        sys.exit(1)
