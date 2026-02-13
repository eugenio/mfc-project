#!/usr/bin/env python3
"""Detailed analysis of biofilm growth dynamics from simulation data."""

import os
import sys

import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def analyze_biofilm_dynamics() -> None:
    """Analyze biofilm growth patterns and dynamics."""
    import gzip
    from pathlib import Path

    import pandas as pd

    # Use the completed simulation
    csv_file = Path(
        "/home/uge/mfc-project/q-learning-mfcs/data/simulation_data/gui_simulation_20250728_165653/gui_simulation_data_20250728_165653.csv.gz",
    )

    with gzip.open(csv_file, "rt") as f:
        df = pd.read_csv(f)

    # Parse biofilm data
    biofilm_data = []
    for _idx, row in df.iterrows():
        try:
            biofilm_str = row["biofilm_thicknesses"]
            if isinstance(biofilm_str, str):
                biofilm_str = biofilm_str.strip("[]")
                biofilm_values = [float(x.strip()) for x in biofilm_str.split(",")]
            else:
                biofilm_values = biofilm_str
            biofilm_data.append(biofilm_values)
        except (ValueError, IndexError, TypeError):
            biofilm_data.append([1.0] * 5)  # Default to 5 cells with 1 μm thickness

    biofilm_array = np.array(biofilm_data)
    time_hours = df["time_hours"].values

    # Calculate growth metrics

    # Identify growth phases
    avg_thickness = np.mean(biofilm_array, axis=1)
    growth_rate = np.gradient(avg_thickness, time_hours)

    # Find growth phases
    # Phase 1: Initial lag (0-10h)
    phase1_idx = np.where(time_hours <= 10)[0]
    if len(phase1_idx) > 0:
        np.mean(growth_rate[phase1_idx])

    # Phase 2: Exponential growth (10-50h)
    phase2_idx = np.where((time_hours > 10) & (time_hours <= 50))[0]
    if len(phase2_idx) > 0:
        np.mean(growth_rate[phase2_idx])

    # Phase 3: Linear growth (50-100h)
    phase3_idx = np.where((time_hours > 50) & (time_hours <= 100))[0]
    if len(phase3_idx) > 0:
        np.mean(growth_rate[phase3_idx])

    # Phase 4: Mature phase (100h+)
    phase4_idx = np.where(time_hours > 100)[0]
    if len(phase4_idx) > 0:
        np.mean(growth_rate[phase4_idx])

    # Substrate utilization correlation
    substrate_conc = df["reservoir_concentration"].values
    substrate_addition = df["substrate_addition_rate"].values

    # Calculate correlation between biofilm thickness and substrate metrics
    from scipy.stats import pearsonr

    corr_conc, p_conc = pearsonr(avg_thickness, substrate_conc)
    corr_add, p_add = pearsonr(
        avg_thickness[1:],
        substrate_addition[1:],
    )  # Skip first point

    # Power production relationship
    power = df["total_power"].values
    corr_power, p_power = pearsonr(avg_thickness, power)

    # Calculate power per unit biofilm
    power / (avg_thickness + 1e-6)  # Avoid division by zero

    # Growth variability between cells
    np.std(biofilm_array, axis=1)

    # Growth stability metrics
    np.std(growth_rate)

    # Identify growth anomalies
    anomaly_threshold = np.mean(growth_rate) + 3 * np.std(growth_rate)
    anomalies = np.where(np.abs(growth_rate) > anomaly_threshold)[0]
    if len(anomalies) > 0:
        for _idx in anomalies[:5]:  # Show first 5
            pass

    # Q-learning control effectiveness
    q_actions = df["q_action"].values
    unique_actions, action_counts = np.unique(q_actions, return_counts=True)

    # Biofilm health indicator
    # Healthy growth is steady with moderate thickness
    health_score = []
    for i in range(len(avg_thickness)):
        thickness = avg_thickness[i]
        rate = growth_rate[i] if i < len(growth_rate) else 0

        # Optimal thickness range: 50-150 μm
        thickness_score = 1.0 if 50 <= thickness <= 150 else 0.5

        # Optimal growth rate: 0.5-2.0 μm/h
        rate_score = 1.0 if 0.5 <= rate <= 2.0 else 0.5

        health_score.append((thickness_score + rate_score) / 2)

    np.mean(health_score)


if __name__ == "__main__":
    analyze_biofilm_dynamics()
