#!/usr/bin/env python3
"""
Corrected analysis of substrate utilization performance between unified and non-unified MFC models
"""

import pandas as pd
import numpy as np

def load_and_analyze_data():
    """Load and analyze the MFC simulation data with corrected substrate utilization calculation"""
    
    # Read the data files
    print("Loading simulation data...")
    unified_data = pd.read_csv('/home/uge/mfc-project/q-learning-mfcs/simulation_data/mfc_unified_qlearning_20250724_022416.csv')
    non_unified_data = pd.read_csv('/home/uge/mfc-project/q-learning-mfcs/simulation_data/mfc_qlearning_20250724_022231.csv')
    
    print(f"Unified model data shape: {unified_data.shape}")
    print(f"Non-unified model data shape: {non_unified_data.shape}")
    
    print("\nColumn analysis:")
    print("Unified model columns:", unified_data.columns.tolist())
    print("Non-unified model columns:", non_unified_data.columns.tolist())
    
    # Calculate proper substrate utilization for unified model using concentration data
    print("\n=== SUBSTRATE UTILIZATION CALCULATION ===")
    
    # For unified model, we have inlet_concentration and avg_outlet_concentration
    if 'inlet_concentration' in unified_data.columns and 'avg_outlet_concentration' in unified_data.columns:
        unified_substrate_util_corrected = ((unified_data['inlet_concentration'] - unified_data['avg_outlet_concentration']) / 
                                           unified_data['inlet_concentration'] * 100)
        print("Unified model: Calculated substrate utilization from concentration data")
        print(f"Sample values: {unified_substrate_util_corrected.head().values}")
    else:
        unified_substrate_util_corrected = unified_data['substrate_utilization'] * 100  # Convert to percentage
        print("Unified model: Using provided substrate_utilization column (converted to %)")
    
    # For non-unified model, use the provided substrate_utilization
    non_unified_substrate_util = non_unified_data['substrate_utilization']
    print("Non-unified model: Using provided substrate_utilization column")
    print(f"Sample values: {non_unified_substrate_util.head().values}")
    
    # Extract substrate utilization metrics
    print("\n=== SUBSTRATE UTILIZATION ANALYSIS ===")
    
    # Final substrate utilization values
    unified_final_util = unified_substrate_util_corrected.iloc[-1]
    non_unified_final_util = non_unified_substrate_util.iloc[-1]
    
    print("\n1. FINAL SUBSTRATE UTILIZATION:")
    print(f"   Unified model:     {unified_final_util:.6f}%")
    print(f"   Non-unified model: {non_unified_final_util:.6f}%")
    print(f"   Difference:        {unified_final_util - non_unified_final_util:.6f}%")
    if non_unified_final_util != 0:
        print(f"   Relative change:   {((unified_final_util - non_unified_final_util) / non_unified_final_util * 100):.3f}%")
    
    # Peak utilization achieved
    unified_peak_util = unified_substrate_util_corrected.max()
    non_unified_peak_util = non_unified_substrate_util.max()
    
    print("\n2. PEAK SUBSTRATE UTILIZATION:")
    print(f"   Unified model:     {unified_peak_util:.6f}%")
    print(f"   Non-unified model: {non_unified_peak_util:.6f}%")
    print(f"   Difference:        {unified_peak_util - non_unified_peak_util:.6f}%")
    
    # Mean utilization over the simulation
    unified_mean_util = unified_substrate_util_corrected.mean()
    non_unified_mean_util = non_unified_substrate_util.mean()
    
    print("\n3. AVERAGE SUBSTRATE UTILIZATION:")
    print(f"   Unified model:     {unified_mean_util:.6f}%")
    print(f"   Non-unified model: {non_unified_mean_util:.6f}%")
    print(f"   Difference:        {unified_mean_util - non_unified_mean_util:.6f}%")
    if non_unified_mean_util != 0:
        print(f"   Relative change:   {((unified_mean_util - non_unified_mean_util) / non_unified_mean_util * 100):.3f}%")
    
    # Stability analysis (standard deviation)
    unified_std_util = unified_substrate_util_corrected.std()
    non_unified_std_util = non_unified_substrate_util.std()
    
    print("\n4. SUBSTRATE UTILIZATION STABILITY (Standard Deviation):")
    print(f"   Unified model:     {unified_std_util:.6f}")
    print(f"   Non-unified model: {non_unified_std_util:.6f}")
    print(f"   Difference:        {unified_std_util - non_unified_std_util:.6f}")
    print(f"   More stable:       {'Unified' if unified_std_util < non_unified_std_util else 'Non-unified'}")
    
    # Time to reach steady state analysis
    def find_steady_state_time(data, substrate_util, threshold=0.1, window=1000):
        """Find when substrate utilization reaches steady state"""
        for i in range(window, len(data)):
            recent_window = substrate_util.iloc[i-window:i]
            if recent_window.std() < threshold:
                return data['time_hours'].iloc[i]
        return None
    
    unified_steady_time = find_steady_state_time(unified_data, unified_substrate_util_corrected)
    non_unified_steady_time = find_steady_state_time(non_unified_data, non_unified_substrate_util)
    
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
    print(f"   Relative change:   {((unified_avg_biofilm.iloc[-1] - non_unified_avg_biofilm.iloc[-1]) / non_unified_avg_biofilm.iloc[-1] * 100):.3f}%")
    
    # Correlation between biofilm thickness and substrate utilization
    unified_corr = np.corrcoef(unified_avg_biofilm, unified_substrate_util_corrected)[0,1]
    non_unified_corr = np.corrcoef(non_unified_avg_biofilm, non_unified_substrate_util)[0,1]
    
    print("\n7. BIOFILM-SUBSTRATE UTILIZATION CORRELATION:")
    print(f"   Unified model:     {unified_corr:.6f}")
    print(f"   Non-unified model: {non_unified_corr:.6f}")
    print(f"   Difference:        {unified_corr - non_unified_corr:.6f}")
    
    # Performance trends analysis
    print("\n=== PERFORMANCE TRENDS ANALYSIS ===")
    
    # Look at trends in the last 25% of simulation time
    last_quarter_start = int(len(unified_data) * 0.75)
    
    unified_late_trend = np.polyfit(range(last_quarter_start, len(unified_data)), 
                                   unified_substrate_util_corrected.iloc[last_quarter_start:], 1)[0]
    non_unified_late_trend = np.polyfit(range(last_quarter_start, len(non_unified_data)), 
                                       non_unified_substrate_util.iloc[last_quarter_start:], 1)[0]
    
    print("\n8. LATE-STAGE SUBSTRATE UTILIZATION TREND (last 25% of simulation):")
    print(f"   Unified model slope:     {unified_late_trend:.8f} %/timestep")
    print(f"   Non-unified model slope: {non_unified_late_trend:.8f} %/timestep")
    
    # Power and voltage analysis
    print("\n=== POWER AND VOLTAGE ANALYSIS ===")
    
    unified_final_power = unified_data['stack_power'].iloc[-1]
    non_unified_final_power = non_unified_data['stack_power'].iloc[-1]
    unified_final_voltage = unified_data['stack_voltage'].iloc[-1]
    non_unified_final_voltage = non_unified_data['stack_voltage'].iloc[-1]
    
    print("\n9. FINAL ELECTRICAL PERFORMANCE:")
    print("   Stack Power:")
    print(f"     Unified model:     {unified_final_power:.6f} W")
    print(f"     Non-unified model: {non_unified_final_power:.6f} W")
    print(f"     Difference:        {unified_final_power - non_unified_final_power:.6f} W")
    print(f"     Relative change:   {((unified_final_power - non_unified_final_power) / non_unified_final_power * 100):.3f}%")
    
    print("   Stack Voltage:")
    print(f"     Unified model:     {unified_final_voltage:.6f} V")
    print(f"     Non-unified model: {non_unified_final_voltage:.6f} V")
    print(f"     Difference:        {unified_final_voltage - non_unified_final_voltage:.6f} V")
    print(f"     Relative change:   {((unified_final_voltage - non_unified_final_voltage) / non_unified_final_voltage * 100):.3f}%")
    
    # Summary and recommendation
    print("\n=== SUMMARY AND COMPARISON ===")
    
    better_final = "Unified" if unified_final_util > non_unified_final_util else "Non-unified"
    better_peak = "Unified" if unified_peak_util > non_unified_peak_util else "Non-unified"
    better_avg = "Unified" if unified_mean_util > non_unified_mean_util else "Non-unified"
    more_stable = "Unified" if unified_std_util < non_unified_std_util else "Non-unified"
    better_power = "Unified" if unified_final_power > non_unified_final_power else "Non-unified"
    
    print("\nPerformance Winner Analysis:")
    print(f"- Better final substrate utilization:  {better_final} model")
    print(f"- Better peak substrate utilization:   {better_peak} model") 
    print(f"- Better average substrate utilization: {better_avg} model")
    print(f"- More stable performance:             {more_stable} model")
    print(f"- Better electrical power output:      {better_power} model")
    
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
    
    print("\nOverall Performance Score:")
    print(f"- Unified model:     {unified_score}/5")
    print(f"- Non-unified model: {non_unified_score}/5")
    
    if unified_score > non_unified_score:
        print("\n🏆 WINNER: Unified model performs better overall")
    elif non_unified_score > unified_score:
        print("\n🏆 WINNER: Non-unified model performs better overall")
    else:
        print("\n🤝 TIE: Both models show comparable performance")
    
    # Additional insights
    print("\n=== KEY INSIGHTS ===")
    
    # Check if the improvement is significant
    if abs(unified_final_util - non_unified_final_util) > 1:
        direction = "higher" if unified_final_util > non_unified_final_util else "lower"
        print(f"- Unified model achieves {direction} final substrate utilization by {abs(unified_final_util - non_unified_final_util):.2f} percentage points")
    else:
        print(f"- Final substrate utilization is very similar between models ({abs(unified_final_util - non_unified_final_util):.3f}% difference)")
    
    # Biofilm analysis
    biofilm_diff_pct = ((unified_avg_biofilm.iloc[-1] - non_unified_avg_biofilm.iloc[-1]) / non_unified_avg_biofilm.iloc[-1] * 100)
    print(f"- Unified model has {biofilm_diff_pct:.1f}% {'higher' if biofilm_diff_pct > 0 else 'lower'} biofilm thickness")
    
    # Power analysis  
    power_diff_pct = ((unified_final_power - non_unified_final_power) / non_unified_final_power * 100)
    print(f"- Unified model has {power_diff_pct:.1f}% {'higher' if power_diff_pct > 0 else 'lower'} power output")
    
    # Correlation analysis
    if abs(unified_corr - non_unified_corr) > 0.1:
        print("- Models show different biofilm-utilization relationships")
        print(f"  Unified: {'Positive' if unified_corr > 0 else 'Negative'} correlation ({unified_corr:.3f})")
        print(f"  Non-unified: {'Positive' if non_unified_corr > 0 else 'Negative'} correlation ({non_unified_corr:.3f})")
    else:
        print("- Both models show similar biofilm-utilization relationships")
    
    # Steady state analysis
    if unified_steady_time and non_unified_steady_time:
        faster_steady = "Unified" if unified_steady_time < non_unified_steady_time else "Non-unified"
        time_diff = abs(unified_steady_time - non_unified_steady_time)
        print(f"- {faster_steady} model reaches steady state {time_diff:.1f} hours faster")
    
    return {
        'unified_final_util': unified_final_util,
        'non_unified_final_util': non_unified_final_util,
        'unified_peak_util': unified_peak_util,
        'non_unified_peak_util': non_unified_peak_util,
        'unified_mean_util': unified_mean_util,
        'non_unified_mean_util': non_unified_mean_util,
        'unified_std_util': unified_std_util,
        'non_unified_std_util': non_unified_std_util,
        'unified_final_power': unified_final_power,
        'non_unified_final_power': non_unified_final_power,
        'unified_score': unified_score,
        'non_unified_score': non_unified_score
    }

if __name__ == "__main__":
    results = load_and_analyze_data()