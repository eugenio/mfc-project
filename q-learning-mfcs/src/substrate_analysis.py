#!/usr/bin/env python3
"""Analysis of substrate utilization performance between unified and non-unified MFC models."""

import numpy as np
import pandas as pd


def load_and_analyze_data() -> None:
    """Load and analyze the MFC simulation data."""
    # Read the data files
    unified_data = pd.read_csv(
        "/home/uge/mfc-project/q-learning-mfcs/simulation_data/mfc_unified_qlearning_20250724_022416.csv",
    )
    non_unified_data = pd.read_csv(
        "/home/uge/mfc-project/q-learning-mfcs/simulation_data/mfc_qlearning_20250724_022231.csv",
    )

    # Extract substrate utilization metrics

    # Final substrate utilization values
    unified_final_util = unified_data["substrate_utilization"].iloc[-1]
    non_unified_final_util = non_unified_data["substrate_utilization"].iloc[-1]

    # Peak utilization achieved
    unified_peak_util = unified_data["substrate_utilization"].max()
    non_unified_peak_util = non_unified_data["substrate_utilization"].max()

    # Mean utilization over the simulation
    unified_mean_util = unified_data["substrate_utilization"].mean()
    non_unified_mean_util = non_unified_data["substrate_utilization"].mean()

    # Stability analysis (standard deviation)
    unified_std_util = unified_data["substrate_utilization"].std()
    non_unified_std_util = non_unified_data["substrate_utilization"].std()

    # Time to reach steady state (when utilization stabilizes)
    # Find when substrate utilization changes become small (< 0.001 change per 100 time steps)
    def find_steady_state_time(data, threshold=0.001, window=1000):
        """Find when substrate utilization reaches steady state."""
        for i in range(window, len(data)):
            recent_window = data["substrate_utilization"].iloc[i - window : i]
            if recent_window.std() < threshold:
                return data["time_hours"].iloc[i]
        return None

    find_steady_state_time(unified_data)
    find_steady_state_time(non_unified_data)

    # Biofilm thickness analysis

    # Get biofilm thickness data (average across all cells)
    unified_biofilm_cols = [col for col in unified_data.columns if "biofilm" in col]
    non_unified_biofilm_cols = [
        col for col in non_unified_data.columns if "biofilm" in col
    ]

    unified_avg_biofilm = unified_data[unified_biofilm_cols].mean(axis=1)
    non_unified_avg_biofilm = non_unified_data[non_unified_biofilm_cols].mean(axis=1)

    # Correlation between biofilm thickness and substrate utilization
    unified_corr = np.corrcoef(
        unified_avg_biofilm,
        unified_data["substrate_utilization"],
    )[0, 1]
    non_unified_corr = np.corrcoef(
        non_unified_avg_biofilm,
        non_unified_data["substrate_utilization"],
    )[0, 1]

    # Performance trends analysis

    # Look at trends in the last 25% of simulation time
    last_quarter_start = int(len(unified_data) * 0.75)

    np.polyfit(
        range(last_quarter_start, len(unified_data)),
        unified_data["substrate_utilization"].iloc[last_quarter_start:],
        1,
    )[0]
    np.polyfit(
        range(last_quarter_start, len(non_unified_data)),
        non_unified_data["substrate_utilization"].iloc[last_quarter_start:],
        1,
    )[0]

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

    if unified_score > non_unified_score or non_unified_score > unified_score:
        pass
    else:
        pass

    # Additional insights

    improvement_pct = (
        (unified_final_util - non_unified_final_util) / non_unified_final_util * 100
    )
    if abs(improvement_pct) > 1:
        pass
    else:
        pass

    (
        (unified_avg_biofilm.iloc[-1] - non_unified_avg_biofilm.iloc[-1])
        / non_unified_avg_biofilm.iloc[-1]
        * 100
    )

    if abs(unified_corr - non_unified_corr) > 0.1:
        pass
    else:
        pass


if __name__ == "__main__":
    load_and_analyze_data()
