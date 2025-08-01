#!/usr/bin/env python3
"""
Detailed analysis of biofilm growth dynamics from simulation data
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def analyze_biofilm_dynamics():
    """Analyze biofilm growth patterns and dynamics"""
    
    print("🔬 Detailed Biofilm Growth Dynamics Analysis")
    print("=" * 50)
    
    import gzip
    import pandas as pd
    from pathlib import Path
    
    # Use the completed simulation
    csv_file = Path("/home/uge/mfc-project/q-learning-mfcs/data/simulation_data/gui_simulation_20250728_165653/gui_simulation_data_20250728_165653.csv.gz")
    
    with gzip.open(csv_file, 'rt') as f:
        df = pd.read_csv(f)
    
    # Parse biofilm data
    biofilm_data = []
    for idx, row in df.iterrows():
        try:
            biofilm_str = row['biofilm_thicknesses']
            if isinstance(biofilm_str, str):
                biofilm_str = biofilm_str.strip('[]')
                biofilm_values = [float(x.strip()) for x in biofilm_str.split(',')]
            else:
                biofilm_values = biofilm_str
            biofilm_data.append(biofilm_values)
        except (ValueError, IndexError, TypeError):
            biofilm_data.append([1.0] * 5)  # Default to 5 cells with 1 μm thickness
    
    biofilm_array = np.array(biofilm_data)
    time_hours = df['time_hours'].values
    
    # Calculate growth metrics
    print("\n📊 Biofilm Growth Phases:")
    print("-" * 40)
    
    # Identify growth phases
    avg_thickness = np.mean(biofilm_array, axis=1)
    growth_rate = np.gradient(avg_thickness, time_hours)
    
    # Find growth phases
    # Phase 1: Initial lag (0-10h)
    phase1_idx = np.where(time_hours <= 10)[0]
    if len(phase1_idx) > 0:
        phase1_growth = np.mean(growth_rate[phase1_idx])
        print("\n🌱 Initial Phase (0-10h):")
        print(f"   Average growth rate: {phase1_growth:.4f} μm/h")
        print(f"   Final thickness: {avg_thickness[phase1_idx[-1]]:.2f} μm")
    
    # Phase 2: Exponential growth (10-50h)
    phase2_idx = np.where((time_hours > 10) & (time_hours <= 50))[0]
    if len(phase2_idx) > 0:
        phase2_growth = np.mean(growth_rate[phase2_idx])
        print("\n📈 Exponential Phase (10-50h):")
        print(f"   Average growth rate: {phase2_growth:.4f} μm/h")
        print(f"   Thickness increase: {avg_thickness[phase2_idx[-1]] - avg_thickness[phase2_idx[0]]:.2f} μm")
    
    # Phase 3: Linear growth (50-100h)
    phase3_idx = np.where((time_hours > 50) & (time_hours <= 100))[0]
    if len(phase3_idx) > 0:
        phase3_growth = np.mean(growth_rate[phase3_idx])
        print("\n📏 Linear Phase (50-100h):")
        print(f"   Average growth rate: {phase3_growth:.4f} μm/h")
        print(f"   Thickness increase: {avg_thickness[phase3_idx[-1]] - avg_thickness[phase3_idx[0]]:.2f} μm")
    
    # Phase 4: Mature phase (100h+)
    phase4_idx = np.where(time_hours > 100)[0]
    if len(phase4_idx) > 0:
        phase4_growth = np.mean(growth_rate[phase4_idx])
        print("\n🌳 Mature Phase (100h+):")
        print(f"   Average growth rate: {phase4_growth:.4f} μm/h")
        print(f"   Final thickness: {avg_thickness[-1]:.2f} μm")
    
    # Substrate utilization correlation
    print("\n🔗 Biofilm-Substrate Correlation:")
    substrate_conc = df['reservoir_concentration'].values
    substrate_addition = df['substrate_addition_rate'].values
    
    # Calculate correlation between biofilm thickness and substrate metrics
    from scipy.stats import pearsonr
    
    corr_conc, p_conc = pearsonr(avg_thickness, substrate_conc)
    corr_add, p_add = pearsonr(avg_thickness[1:], substrate_addition[1:])  # Skip first point
    
    print(f"   Thickness vs Concentration: r={corr_conc:.3f} (p={p_conc:.3e})")
    print(f"   Thickness vs Addition Rate: r={corr_add:.3f} (p={p_add:.3e})")
    
    # Power production relationship
    power = df['total_power'].values
    corr_power, p_power = pearsonr(avg_thickness, power)
    print("\n⚡ Biofilm-Power Relationship:")
    print(f"   Thickness vs Power: r={corr_power:.3f} (p={p_power:.3e})")
    
    # Calculate power per unit biofilm
    power_per_biofilm = power / (avg_thickness + 1e-6)  # Avoid division by zero
    print(f"   Average power density: {np.mean(power_per_biofilm):.4f} W/μm")
    print(f"   Peak power density: {np.max(power_per_biofilm):.4f} W/μm at {time_hours[np.argmax(power_per_biofilm)]:.1f}h")
    
    # Growth variability between cells
    print("\n📊 Cell-to-Cell Variability:")
    cell_std = np.std(biofilm_array, axis=1)
    print(f"   Initial variability: {cell_std[0]:.4f} μm")
    print(f"   Final variability: {cell_std[-1]:.4f} μm")
    print(f"   Maximum variability: {np.max(cell_std):.4f} μm at {time_hours[np.argmax(cell_std)]:.1f}h")
    
    # Growth stability metrics
    growth_stability = np.std(growth_rate)
    print("\n📉 Growth Stability:")
    print(f"   Growth rate std dev: {growth_stability:.4f} μm/h")
    print(f"   Coefficient of variation: {growth_stability/np.mean(growth_rate):.2%}")
    
    # Identify growth anomalies
    anomaly_threshold = np.mean(growth_rate) + 3 * np.std(growth_rate)
    anomalies = np.where(np.abs(growth_rate) > anomaly_threshold)[0]
    if len(anomalies) > 0:
        print("\n⚠️ Growth Anomalies Detected:")
        for idx in anomalies[:5]:  # Show first 5
            print(f"   Time {time_hours[idx]:.1f}h: growth rate = {growth_rate[idx]:.4f} μm/h")
    
    # Q-learning control effectiveness
    q_actions = df['q_action'].values
    unique_actions, action_counts = np.unique(q_actions, return_counts=True)
    
    print("\n🎮 Q-Learning Control Actions:")
    print(f"   Unique actions used: {len(unique_actions)}")
    print(f"   Most common action: {unique_actions[np.argmax(action_counts)]} ({np.max(action_counts)/len(q_actions)*100:.1f}%)")
    
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
    
    avg_health = np.mean(health_score)
    print(f"\n🏥 Biofilm Health Score: {avg_health:.2%}")
    print(f"   Healthy periods: {sum(h > 0.7 for h in health_score)/len(health_score)*100:.1f}%")

if __name__ == "__main__":
    analyze_biofilm_dynamics()