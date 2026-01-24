#!/usr/bin/env python3
"""Monitor the progress of the comprehensive MFC simulation."""

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


def monitor_progress() -> None:
    """Monitor simulation progress."""
    sim_dir = find_latest_simulation()
    if not sim_dir:
        return

    progress_file = os.path.join(sim_dir, "simulation_progress.json")
    log_file = os.path.join(sim_dir, "simulation.log")

    last_update = 0

    try:
        while True:
            # Check progress file
            if os.path.exists(progress_file):
                try:
                    with open(progress_file) as f:
                        progress = json.load(f)

                    # Only show update if it's new
                    current_update = progress.get("current_hour", 0)
                    if current_update != last_update:
                        timedelta(seconds=progress.get("elapsed_time", 0))
                        timedelta(
                            seconds=progress.get("estimated_remaining", 0),
                        )

                        last_update = current_update

                        # Check if simulation is complete
                        if progress["progress_percent"] >= 100:
                            break

                except (json.JSONDecodeError, KeyError):
                    pass

            # Check if simulation is still running by looking at log file modification time
            if os.path.exists(log_file):
                log_mod_time = os.path.getmtime(log_file)
                if time.time() - log_mod_time > 600:  # No update for 10 minutes
                    break

            time.sleep(30)  # Check every 30 seconds

    except KeyboardInterrupt:
        pass

    # Show final status
    if os.path.exists(progress_file):
        try:
            with open(progress_file) as f:
                json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass


if __name__ == "__main__":
    monitor_progress()
