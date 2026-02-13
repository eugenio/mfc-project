"""Data loading utilities for MFC Streamlit GUI.

This module provides functions for loading simulation data from files
and retrieving recent simulation histories.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd
import streamlit as st


def load_simulation_data(data_dir):
    """Load simulation data from directory.

    Args:
        data_dir: Path to the simulation data directory.

    Returns:
        DataFrame with simulation data or None if loading fails.
    """
    data_dir = Path(data_dir)

    # Find compressed CSV file
    csv_files = list(data_dir.glob("*_data_*.csv.gz"))
    if not csv_files:
        return None

    csv_file = csv_files[0]

    try:
        with gzip.open(csv_file, "rt") as f:
            return pd.read_csv(f)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def load_recent_simulations():
    """Load list of recent simulation directories.

    Returns:
        List of dictionaries containing simulation metadata,
        sorted by timestamp (most recent first).
    """
    data_dir = Path("../data/simulation_data")

    if not data_dir.exists():
        return []

    sim_dirs = []
    for subdir in data_dir.iterdir():
        if subdir.is_dir() and subdir.name.startswith(("gpu_", "lactate_", "gui_")):
            # Check if it has results
            json_files = list(subdir.glob("*results*.json"))
            csv_files = list(subdir.glob("*data*.csv.gz"))

            if json_files and csv_files:
                try:
                    with open(json_files[0]) as f:
                        results = json.load(f)

                    sim_dirs.append(
                        {
                            "name": subdir.name,
                            "path": str(subdir),
                            "timestamp": results.get("simulation_info", {}).get(
                                "timestamp",
                                "",
                            ),
                            "duration": results.get("simulation_info", {}).get(
                                "duration_hours",
                                0,
                            ),
                            "final_conc": results.get("performance_metrics", {}).get(
                                "final_reservoir_concentration",
                                0,
                            ),
                            "control_effectiveness": results.get(
                                "performance_metrics",
                                {},
                            ).get("control_effectiveness_2mM", 0),
                        },
                    )
                except Exception:
                    continue

    return sorted(sim_dirs, key=lambda x: x["timestamp"], reverse=True)
