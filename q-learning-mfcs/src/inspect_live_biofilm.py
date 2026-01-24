#!/usr/bin/env python3
"""Inspect live biofilm growth data from the running Streamlit simulation."""

import os
import sys

import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def inspect_latest_simulation() -> None:
    """Check the latest simulation for biofilm data."""
    # Check a completed GUI simulation (to avoid file corruption from active writes)
    latest_dir = "/home/uge/mfc-project/q-learning-mfcs/data/simulation_data/gui_simulation_20250728_165653"

    import gzip
    from pathlib import Path

    import pandas as pd

    csv_file = Path(latest_dir) / "gui_simulation_data_20250728_165653.csv.gz"

    if csv_file.exists():
        try:
            with gzip.open(csv_file, "rt") as f:
                df = pd.read_csv(f)

            if "biofilm_thicknesses" in df.columns:
                # Parse biofilm data (it's stored as string representation of list)
                biofilm_data = []
                for _idx, row in df.iterrows():
                    try:
                        # Convert string representation to list
                        biofilm_str = row["biofilm_thicknesses"]
                        if isinstance(biofilm_str, str):
                            # Remove brackets and split by comma
                            biofilm_str = biofilm_str.strip("[]")
                            biofilm_values = [
                                float(x.strip()) for x in biofilm_str.split(",")
                            ]
                        else:
                            biofilm_values = biofilm_str
                        biofilm_data.append(biofilm_values)
                    except:
                        biofilm_data.append([0, 0, 0])

                biofilm_array = np.array(biofilm_data)

                # Statistics

                # Per-cell analysis
                if len(biofilm_array.shape) > 1:
                    for cell_idx in range(biofilm_array.shape[1]):
                        biofilm_array[:, cell_idx]

                # Overall statistics

                # Recent changes
                if len(biofilm_array) > 10:
                    biofilm_array[-1, :] - biofilm_array[-10, :]
                    df["time_hours"].iloc[-1] - df["time_hours"].iloc[-10]

            else:
                pass

            # Also check other relevant parameters
            if "substrate_addition_rate" in df.columns:
                pass

            if "reservoir_concentration" in df.columns:
                pass

        except Exception:
            import traceback

            traceback.print_exc()
    else:
        # List available simulations
        sim_dir = Path("/home/uge/mfc-project/q-learning-mfcs/data/simulation_data")
        gui_sims = sorted(
            [
                d
                for d in sim_dir.iterdir()
                if d.is_dir() and d.name.startswith("gui_simulation")
            ],
            key=lambda x: x.name,
            reverse=True,
        )

        if gui_sims:
            for _sim in gui_sims[:5]:
                pass


if __name__ == "__main__":
    inspect_latest_simulation()
