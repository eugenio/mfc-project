#!/usr/bin/env python3
"""
Analysis of substrate utilization performance between unified and non-unified MFC models
"""

import numpy as np
import pandas as pd


def load_and_analyze_data():
    """Load and analyze the MFC simulation data"""

    # Read the data files
    print("Loading simulation data...")
    unified_data = pd.read_csv('/home/uge/mfc-project/q-learning-mfcs/simulation_data/mfc_unified_qlearning_20250724_022416.csv')
    non_unified_data = pd.read_csv('/home/uge/mfc-project/q-learning-mfcs/simulation_data/mfc_qlearning_20250724_022231.csv')

    print(f"Unified model data shape: {unified_data.shape}")
    print(f"Non-unified model data shape: {non_unified_data.shape}")

    # Extract substrate utilization metrics
    print("\n=== SUBSTRATE UTILIZATION ANALYSIS ===")

    # Final substrate utilization values
    unified_final_util = unified_data['substrate_utilization'].iloc[-1]
    non_unified_final_util = non_unified_data['substrate_utilization'].iloc[-1]

    print("\n1. FINAL SUBSTRATE UTILIZATION:")
    print(f"   Unified model:     {unified_final_util:.6f}")
    print(f"   Non-unified model: {non_unified_final_util:.6f}")
    print(f"   Difference:        {unified_final_util - non_unified_final_util:.6f}")
    print(f"   Improvement:       {((unified_final_util - non_unified_final_util) / non_unified_final_util * 100):.3f}%")

    # Peak utilization achieved
    unified_peak_util = unified_data['substrate_utilization'].max()
    non_unified_peak_util = non_unified_data['substrate_utilization'].max()

    print("\n2. PEAK SUBSTRATE UTILIZATION:")
    print(f"   Unified model:     {unified_peak_util:.6f}")
    print(f"   Non-unified model: {non_unified_peak_util:.6f}")
    print(f"   Difference:        {unified_peak_util - non_unified_peak_util:.6f}")

    # Mean utilization over the simulation
    unified_mean_util = unified_data['substrate_utilization'].mean()
    non_unified_mean_util = non_unified_data['substrate_utilization'].mean()

    print("\n3. AVERAGE SUBSTRATE UTILIZATION:")
    print(f"   Unified model:     {unified_mean_util:.6f}")
    print(f"   Non-unified model: {non_unified_mean_util:.6f}")
    print(f"   Difference:        {unified_mean_util - non_unified_mean_util:.6f}")
    print(f"   Improvement:       {((unified_mean_util - non_unified_mean_util) / non_unified_mean_util * 100):.3f}%")

    # Stability analysis (standard deviation)
    unified_std_util = unified_data['substrate_utilization'].std()
    non_unified_std_util = non_unified_data['substrate_utilization'].std()

    print("\n4. SUBSTRATE UTILIZATION STABILITY (Standard Deviation):")
    print(f"   Unified model:     {unified_std_util:.6f}")
    print(f"   Non-unified model: {non_unified_std_util:.6f}")
    print(f"   Difference:        {unified_std_util - non_unified_std_util:.6f}")

    # Time to reach steady state (when utilization stabilizes)
    # Find when substrate utilization changes become small (< 0.001 change per 100 time steps)
    def find_steady_state_time(data, threshold=0.001, window=1000):
        """Find when substrate utilization reaches steady state"""
        for i in range(window, len(data)):
            recent_window = data['substrate_utilization'].iloc[i-window:i]
            if recent_window.std() < threshold:
                return data['time_hours'].iloc[i]
        return None

    unified_steady_time = find_steady_state_time(unified_data)
    non_unified_steady_time = find_steady_state_time(non_unified_data)

    print("\n5. TIME TO REACH STEADY STATE:")
    print(f"   Unified model:     {unified_steady_time:.3f} hours" if unified_steady_time else "   Unified model:     Not reached")
    print(f"   Non-unified model: {non_unified_steady_time:.3f} hours" if non_unified_steady_time else "   Non-unified model: Not reached")

    # Biofilm thickness analysis
    print("\n=== BIOFILM THICKNESS ANALYSIS ===")

    # Get biofilm thickness data (average across all cells)
    unified_biofilm_cols = [col for col in unified_data.columns if 'biofilm' in col]
    non_unified_biofilm_cols = [col for col in non_unified_data.columns if 'biofilm' in col]

    unified_avg_biofilm = unified_data[unified_biofilm_cols].mean(axis=1)
    non_unified_avg_biofilm = non_unified_data[non_unified_biofilm_cols].mean(axis=1)

    print("\n6. FINAL AVERAGE BIOFILM THICKNESS:")
    print(f"   Unified model:     {unified_avg_biofilm.iloc[-1]:.6f}")
    print(f"   Non-unified model: {non_unified_avg_biofilm.iloc[-1]:.6f}")
    print(f"   Difference:        {unified_avg_biofilm.iloc[-1] - non_unified_avg_biofilm.iloc[-1]:.6f}")

    # Correlation between biofilm thickness and substrate utilization
    unified_corr = np.corrcoef(unified_avg_biofilm, unified_data['substrate_utilization'])[0,1]
    non_unified_corr = np.corrcoef(non_unified_avg_biofilm, non_unified_data['substrate_utilization'])[0,1]

    print("\n7. BIOFILM-SUBSTRATE UTILIZATION CORRELATION:")
    print(f"   Unified model:     {unified_corr:.6f}")
    print(f"   Non-unified model: {non_unified_corr:.6f}")

    # Performance trends analysis
    print("\n=== PERFORMANCE TRENDS ANALYSIS ===")

    # Look at trends in the last 25% of simulation time
    last_quarter_start = int(len(unified_data) * 0.75)

    unified_late_trend = np.polyfit(range(last_quarter_start, len(unified_data)),
                                   unified_data['substrate_utilization'].iloc[last_quarter_start:], 1)[0]
    non_unified_late_trend = np.polyfit(range(last_quarter_start, len(non_unified_data)),
                                       non_unified_data['substrate_utilization'].iloc[last_quarter_start:], 1)[0]

    print("\n8. LATE-STAGE SUBSTRATE UTILIZATION TREND (last 25% of simulation):")
    print(f"   Unified model slope:     {unified_late_trend:.8f} (per timestep)")
    print(f"   Non-unified model slope: {non_unified_late_trend:.8f} (per timestep)")

    # Summary and recommendation
    print("\n=== SUMMARY AND COMPARISON ===")

    better_final = "Unified" if unified_final_util > non_unified_final_util else "Non-unified"
    better_peak = "Unified" if unified_peak_util > non_unified_peak_util else "Non-unified"
    better_avg = "Unified" if unified_mean_util > non_unified_mean_util else "Non-unified"
    more_stable = "Unified" if unified_std_util < non_unified_std_util else "Non-unified"

    print("\nPerformance Winner Analysis:")
    print(f"- Better final utilization:    {better_final} model")
    print(f"- Better peak utilization:     {better_peak} model")
    print(f"- Better average utilization:  {better_avg} model")
    print(f"- More stable performance:     {more_stable} model")

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

    print("\nOverall Performance Score:")
    print(f"- Unified model:     {unified_score}/4")
    print(f"- Non-unified model: {non_unified_score}/4")

    if unified_score > non_unified_score:
        print("\n🏆 WINNER: Unified model performs better overall")
    elif non_unified_score > unified_score:
        print("\n🏆 WINNER: Non-unified model performs better overall")
    else:
        print("\n🤝 TIE: Both models show comparable performance")

    # Additional insights
    print("\n=== KEY INSIGHTS ===")

    improvement_pct = ((unified_final_util - non_unified_final_util) / non_unified_final_util * 100)
    if abs(improvement_pct) > 1:
        print(f"- Significant substrate utilization difference: {improvement_pct:.1f}%")
    else:
        print("- Substrate utilization performance is very similar between models")

    biofilm_diff_pct = ((unified_avg_biofilm.iloc[-1] - non_unified_avg_biofilm.iloc[-1]) / non_unified_avg_biofilm.iloc[-1] * 100)
    print(f"- Biofilm thickness difference: {biofilm_diff_pct:.1f}%")

    if abs(unified_corr - non_unified_corr) > 0.1:
        print("- Different biofilm-utilization relationships between models")
    else:
        print("- Similar biofilm-utilization relationships in both models")

if __name__ == "__main__":
    load_and_analyze_data()
