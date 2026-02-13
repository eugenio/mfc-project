#!/usr/bin/env python3
"""
Monitor the progress of the comprehensive MFC simulation.
"""

import glob
import json
import os
import time
from datetime import timedelta


def find_latest_simulation():
    """Find the latest simulation directory."""
    from path_config import get_simulation_data_path

    pattern = get_simulation_data_path("comprehensive_simulation_*")
    simulation_dirs = glob.glob(pattern)

    if not simulation_dirs:
        return None

    # Return the most recent directory
    return max(simulation_dirs, key=os.path.getctime)

def monitor_progress():
    """Monitor simulation progress."""
    print("Looking for running simulation...")

    sim_dir = find_latest_simulation()
    if not sim_dir:
        print("No simulation found. Start the simulation first.")
        return

    progress_file = os.path.join(sim_dir, "simulation_progress.json")
    log_file = os.path.join(sim_dir, "simulation.log")

    print(f"Monitoring simulation in: {sim_dir}")
    print("Press Ctrl+C to stop monitoring (simulation will continue)")
    print("=" * 60)

    last_update = 0

    try:
        while True:
            # Check progress file
            if os.path.exists(progress_file):
                try:
                    with open(progress_file) as f:
                        progress = json.load(f)

                    # Only show update if it's new
                    current_update = progress.get('current_hour', 0)
                    if current_update != last_update:
                        elapsed = timedelta(seconds=progress.get('elapsed_time', 0))
                        remaining = timedelta(seconds=progress.get('estimated_remaining', 0))

                        print(f"Hour {progress['current_hour']:.1f}/100 "
                              f"({progress['progress_percent']:.1f}%) - "
                              f"Power: {progress['current_power']:.3f}W - "
                              f"CE: {progress['current_efficiency']:.2%}")
                        print(f"  Elapsed: {elapsed} - Remaining: {remaining}")
                        print(f"  EIS sensors: {progress['sensor_status']['eis_active']}/5 - "
                              f"QCM sensors: {progress['sensor_status']['qcm_active']}/5")
                        print(f"  Fusion confidence: {progress['sensor_status']['fusion_confidence']:.2f}")
                        print("-" * 40)

                        last_update = current_update

                        # Check if simulation is complete
                        if progress['progress_percent'] >= 100:
                            print("🎉 Simulation completed!")
                            break

                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error reading progress: {e}")

            # Check if simulation is still running by looking at log file modification time
            if os.path.exists(log_file):
                log_mod_time = os.path.getmtime(log_file)
                if time.time() - log_mod_time > 600:  # No update for 10 minutes
                    print("⚠️  Simulation appears to have stopped (no updates for 10 minutes)")
                    break

            time.sleep(30)  # Check every 30 seconds

    except KeyboardInterrupt:
        print("\nMonitoring stopped. Simulation continues in background.")

    # Show final status
    if os.path.exists(progress_file):
        try:
            with open(progress_file) as f:
                final_progress = json.load(f)
            print(f"\nFinal status: {final_progress['progress_percent']:.1f}% complete")
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass

if __name__ == "__main__":
    monitor_progress()
