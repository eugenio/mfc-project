#!/usr/bin/env python3
"""Visualization analysis of substrate utilization performance between unified and non-unified MFC models."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from path_config import get_figure_path, get_simulation_data_path


def create_analysis_plots():
    """Create visualization plots for the MFC comparison analysis."""
    # Read the data files
    unified_data = pd.read_csv(
        get_simulation_data_path("mfc_unified_qlearning_20250724_022416.csv"),
    )
    non_unified_data = pd.read_csv(
        get_simulation_data_path("mfc_qlearning_20250724_022231.csv"),
    )

    # Calculate corrected substrate utilization for unified model
    unified_substrate_util = (
        (unified_data["inlet_concentration"] - unified_data["avg_outlet_concentration"])
        / unified_data["inlet_concentration"]
        * 100
    )
    non_unified_substrate_util = non_unified_data["substrate_utilization"]

    # Get biofilm data
    unified_biofilm_cols = [col for col in unified_data.columns if "biofilm" in col]
    non_unified_biofilm_cols = [
        col for col in non_unified_data.columns if "biofilm" in col
    ]
    unified_avg_biofilm = unified_data[unified_biofilm_cols].mean(axis=1)
    non_unified_avg_biofilm = non_unified_data[non_unified_biofilm_cols].mean(axis=1)

    # Create time vectors (sample every 100 points for better visualization)
    sample_rate = 100
    unified_time = unified_data["time_hours"][::sample_rate]
    non_unified_time = non_unified_data["time_hours"][::sample_rate]

    # Create the plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "MFC Performance Comparison: Unified vs Non-Unified Models",
        fontsize=16,
        fontweight="bold",
    )

    # Plot 1: Substrate Utilization over Time
    axes[0, 0].plot(
        unified_time,
        unified_substrate_util[::sample_rate],
        label="Unified Model",
        color="blue",
        alpha=0.8,
        linewidth=2,
    )
    axes[0, 0].plot(
        non_unified_time,
        non_unified_substrate_util[::sample_rate],
        label="Non-Unified Model",
        color="red",
        alpha=0.8,
        linewidth=2,
    )
    axes[0, 0].set_xlabel("Time (hours)")
    axes[0, 0].set_ylabel("Substrate Utilization (%)")
    axes[0, 0].set_title("Substrate Utilization Over Time")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Biofilm Thickness over Time
    axes[0, 1].plot(
        unified_time,
        unified_avg_biofilm[::sample_rate],
        label="Unified Model",
        color="blue",
        alpha=0.8,
        linewidth=2,
    )
    axes[0, 1].plot(
        non_unified_time,
        non_unified_avg_biofilm[::sample_rate],
        label="Non-Unified Model",
        color="red",
        alpha=0.8,
        linewidth=2,
    )
    axes[0, 1].set_xlabel("Time (hours)")
    axes[0, 1].set_ylabel("Average Biofilm Thickness")
    axes[0, 1].set_title("Biofilm Development Over Time")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Power Output over Time
    axes[1, 0].plot(
        unified_time,
        unified_data["stack_power"][::sample_rate],
        label="Unified Model",
        color="blue",
        alpha=0.8,
        linewidth=2,
    )
    axes[1, 0].plot(
        non_unified_time,
        non_unified_data["stack_power"][::sample_rate],
        label="Non-Unified Model",
        color="red",
        alpha=0.8,
        linewidth=2,
    )
    axes[1, 0].set_xlabel("Time (hours)")
    axes[1, 0].set_ylabel("Stack Power (W)")
    axes[1, 0].set_title("Power Output Over Time")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Biofilm vs Substrate Utilization Correlation
    # Sample data for scatter plot (every 1000 points)
    scatter_sample = 1000
    axes[1, 1].scatter(
        unified_avg_biofilm[::scatter_sample],
        unified_substrate_util[::scatter_sample],
        alpha=0.6,
        color="blue",
        label="Unified Model",
        s=20,
    )
    axes[1, 1].scatter(
        non_unified_avg_biofilm[::scatter_sample],
        non_unified_substrate_util[::scatter_sample],
        alpha=0.6,
        color="red",
        label="Non-Unified Model",
        s=20,
    )
    axes[1, 1].set_xlabel("Average Biofilm Thickness")
    axes[1, 1].set_ylabel("Substrate Utilization (%)")
    axes[1, 1].set_title("Biofilm Thickness vs Substrate Utilization")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        get_figure_path("mfc_comparison_analysis.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # Print key statistics for the plots

    # Substrate utilization analysis

    # Biofilm development

    # Power trends
    unified_power_trend = np.polyfit(
        range(len(unified_data)),
        unified_data["stack_power"],
        1,
    )[0]
    non_unified_power_trend = np.polyfit(
        range(len(non_unified_data)),
        non_unified_data["stack_power"],
        1,
    )[0]

    # Key performance periods

    # Early performance (first 10%)
    early_cutoff = int(len(unified_data) * 0.1)
    unified_early_util = unified_substrate_util[:early_cutoff].mean()
    non_unified_early_util = non_unified_substrate_util[:early_cutoff].mean()

    # Late performance (last 10%)
    late_cutoff = int(len(unified_data) * 0.9)
    unified_late_util = unified_substrate_util[late_cutoff:].mean()
    non_unified_late_util = non_unified_substrate_util[late_cutoff:].mean()

    return {
        "unified_early_util": unified_early_util,
        "unified_late_util": unified_late_util,
        "non_unified_early_util": non_unified_early_util,
        "non_unified_late_util": non_unified_late_util,
        "unified_power_trend": unified_power_trend,
        "non_unified_power_trend": non_unified_power_trend,
    }


if __name__ == "__main__":
    results = create_analysis_plots()
