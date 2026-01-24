#!/usr/bin/env python3
"""Corrected analysis of substrate utilization performance between unified and non-unified MFC models."""

import numpy as np
import pandas as pd


def load_and_analyze_data():
    """Load and analyze the MFC simulation data with corrected substrate utilization calculation."""
    # Read the data files
    unified_data = pd.read_csv(
        "/home/uge/mfc-project/q-learning-mfcs/simulation_data/mfc_unified_qlearning_20250724_022416.csv",
    )
    non_unified_data = pd.read_csv(
        "/home/uge/mfc-project/q-learning-mfcs/simulation_data/mfc_qlearning_20250724_022231.csv",
    )

    # Calculate proper substrate utilization for unified model using concentration data

    # For unified model, we have inlet_concentration and avg_outlet_concentration
    if (
        "inlet_concentration" in unified_data.columns
        and "avg_outlet_concentration" in unified_data.columns
    ):
        unified_substrate_util_corrected = (
            (
                unified_data["inlet_concentration"]
                - unified_data["avg_outlet_concentration"]
            )
            / unified_data["inlet_concentration"]
            * 100
        )
    else:
        unified_substrate_util_corrected = (
            unified_data["substrate_utilization"] * 100
        )  # Convert to percentage

    # For non-unified model, use the provided substrate_utilization
    non_unified_substrate_util = non_unified_data["substrate_utilization"]

    # Extract substrate utilization metrics

    # Final substrate utilization values
    unified_final_util = unified_substrate_util_corrected.iloc[-1]
    non_unified_final_util = non_unified_substrate_util.iloc[-1]

    if non_unified_final_util != 0:
        pass

    # Peak utilization achieved
    unified_peak_util = unified_substrate_util_corrected.max()
    non_unified_peak_util = non_unified_substrate_util.max()

    # Mean utilization over the simulation
    unified_mean_util = unified_substrate_util_corrected.mean()
    non_unified_mean_util = non_unified_substrate_util.mean()

    if non_unified_mean_util != 0:
        pass

    # Stability analysis (standard deviation)
    unified_std_util = unified_substrate_util_corrected.std()
    non_unified_std_util = non_unified_substrate_util.std()

    # Time to reach steady state analysis
    def find_steady_state_time(data, substrate_util, threshold=0.1, window=1000):
        """Find when substrate utilization reaches steady state."""
        for i in range(window, len(data)):
            recent_window = substrate_util.iloc[i - window : i]
            if recent_window.std() < threshold:
                return data["time_hours"].iloc[i]
        return None

    unified_steady_time = find_steady_state_time(
        unified_data,
        unified_substrate_util_corrected,
    )
    non_unified_steady_time = find_steady_state_time(
        non_unified_data,
        non_unified_substrate_util,
    )

    # Biofilm thickness analysis

    # Get biofilm thickness data (average across all cells)
    unified_biofilm_cols = [col for col in unified_data.columns if "biofilm" in col]
    non_unified_biofilm_cols = [
        col for col in non_unified_data.columns if "biofilm" in col
    ]

    unified_avg_biofilm = unified_data[unified_biofilm_cols].mean(axis=1)
    non_unified_avg_biofilm = non_unified_data[non_unified_biofilm_cols].mean(axis=1)

    # Correlation between biofilm thickness and substrate utilization
    unified_corr = np.corrcoef(unified_avg_biofilm, unified_substrate_util_corrected)[
        0,
        1,
    ]
    non_unified_corr = np.corrcoef(non_unified_avg_biofilm, non_unified_substrate_util)[
        0,
        1,
    ]

    # Performance trends analysis

    # Look at trends in the last 25% of simulation time
    last_quarter_start = int(len(unified_data) * 0.75)

    np.polyfit(
        range(last_quarter_start, len(unified_data)),
        unified_substrate_util_corrected.iloc[last_quarter_start:],
        1,
    )[0]
    np.polyfit(
        range(last_quarter_start, len(non_unified_data)),
        non_unified_substrate_util.iloc[last_quarter_start:],
        1,
    )[0]

    # Power and voltage analysis

    unified_final_power = unified_data["stack_power"].iloc[-1]
    non_unified_final_power = non_unified_data["stack_power"].iloc[-1]
    unified_data["stack_voltage"].iloc[-1]
    non_unified_data["stack_voltage"].iloc[-1]

    # Summary and recommendation

    # Calculate overall performance score
    unified_score = 0
    non_unified_score = 0

    if unified_final_util > non_unified_final_util:
        unified_score += 1
    else:
        non_unified_score += 1

    if unified_peak_util > non_unified_peak_util:
        unified_score += 1
    else:
        non_unified_score += 1

    if unified_mean_util > non_unified_mean_util:
        unified_score += 1
    else:
        non_unified_score += 1

    if unified_std_util < non_unified_std_util:
        unified_score += 1
    else:
        non_unified_score += 1

    if unified_final_power > non_unified_final_power:
        unified_score += 1
    else:
        non_unified_score += 1

    if unified_score > non_unified_score or non_unified_score > unified_score:
        pass
    else:
        pass

    # Additional insights

    # Check if the improvement is significant
    if abs(unified_final_util - non_unified_final_util) > 1:
        pass
    else:
        pass

    # Biofilm analysis
    (
        (unified_avg_biofilm.iloc[-1] - non_unified_avg_biofilm.iloc[-1])
        / non_unified_avg_biofilm.iloc[-1]
        * 100
    )

    # Power analysis
    ((unified_final_power - non_unified_final_power) / non_unified_final_power * 100)

    # Correlation analysis
    if abs(unified_corr - non_unified_corr) > 0.1:
        pass
    else:
        pass

    # Steady state analysis
    if unified_steady_time and non_unified_steady_time:
        abs(unified_steady_time - non_unified_steady_time)

    return {
        "unified_final_util": unified_final_util,
        "non_unified_final_util": non_unified_final_util,
        "unified_peak_util": unified_peak_util,
        "non_unified_peak_util": non_unified_peak_util,
        "unified_mean_util": unified_mean_util,
        "non_unified_mean_util": non_unified_mean_util,
        "unified_std_util": unified_std_util,
        "non_unified_std_util": non_unified_std_util,
        "unified_final_power": unified_final_power,
        "non_unified_final_power": non_unified_final_power,
        "unified_score": unified_score,
        "non_unified_score": non_unified_score,
    }


if __name__ == "__main__":
    results = load_and_analyze_data()
