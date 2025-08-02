#!/usr/bin/env python3
"""
Inspect live biofilm growth data from the running Streamlit simulation
"""

import os
import sys

import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def inspect_latest_simulation():
    """Check the latest simulation for biofilm data"""

    print("🔍 Inspecting Biofilm Growth Data")
    print("=" * 50)

    # Check a completed GUI simulation (to avoid file corruption from active writes)
    latest_dir = "/home/uge/mfc-project/q-learning-mfcs/data/simulation_data/gui_simulation_20250728_165653"

    import gzip
    from pathlib import Path

    import pandas as pd

    csv_file = Path(latest_dir) / "gui_simulation_data_20250728_165653.csv.gz"

    if csv_file.exists():
        print(f"📊 Found simulation data: {csv_file}")

        try:
            with gzip.open(csv_file, 'rt') as f:
                df = pd.read_csv(f)

            print(f"\n📈 Data shape: {df.shape[0]} time points, {df.shape[1]} columns")
            print(f"📋 Columns: {list(df.columns)}")

            if 'biofilm_thicknesses' in df.columns:
                print("\n🦠 Biofilm Growth Analysis:")
                print("-" * 30)

                # Parse biofilm data (it's stored as string representation of list)
                biofilm_data = []
                for idx, row in df.iterrows():
                    try:
                        # Convert string representation to list
                        biofilm_str = row['biofilm_thicknesses']
                        if isinstance(biofilm_str, str):
                            # Remove brackets and split by comma
                            biofilm_str = biofilm_str.strip('[]')
                            biofilm_values = [float(x.strip()) for x in biofilm_str.split(',')]
                        else:
                            biofilm_values = biofilm_str
                        biofilm_data.append(biofilm_values)
                    except:
                        biofilm_data.append([0, 0, 0])

                biofilm_array = np.array(biofilm_data)

                # Statistics
                print(f"📏 Number of cells: {biofilm_array.shape[1] if len(biofilm_array.shape) > 1 else 1}")
                print(f"⏰ Time span: {df['time_hours'].iloc[0]:.1f} - {df['time_hours'].iloc[-1]:.1f} hours")

                # Per-cell analysis
                if len(biofilm_array.shape) > 1:
                    for cell_idx in range(biofilm_array.shape[1]):
                        cell_biofilm = biofilm_array[:, cell_idx]
                        print(f"\n🧫 Cell {cell_idx + 1}:")
                        print(f"   Initial thickness: {cell_biofilm[0]:.3f} μm")
                        print(f"   Current thickness: {cell_biofilm[-1]:.3f} μm")
                        print(f"   Growth: {cell_biofilm[-1] - cell_biofilm[0]:.3f} μm")
                        print(f"   Max thickness: {np.max(cell_biofilm):.3f} μm")
                        print(f"   Growth rate: {(cell_biofilm[-1] - cell_biofilm[0]) / (df['time_hours'].iloc[-1] - df['time_hours'].iloc[0]):.4f} μm/h")

                # Overall statistics
                print("\n📊 Overall Biofilm Statistics:")
                print(f"   Average current thickness: {np.mean(biofilm_array[-1, :]):.3f} μm")
                print(f"   Total biofilm volume (approx): {np.sum(biofilm_array[-1, :]):.3f} μm")

                # Recent changes
                if len(biofilm_array) > 10:
                    recent_growth = biofilm_array[-1, :] - biofilm_array[-10, :]
                    time_diff = df['time_hours'].iloc[-1] - df['time_hours'].iloc[-10]
                    print(f"\n📈 Recent Activity (last {time_diff:.1f} hours):")
                    print(f"   Average growth: {np.mean(recent_growth):.4f} μm")
                    print(f"   Growth rate: {np.mean(recent_growth) / time_diff:.5f} μm/h")

            else:
                print("❌ No biofilm thickness data found in simulation")

            # Also check other relevant parameters
            if 'substrate_addition_rate' in df.columns:
                print("\n💉 Substrate Control:")
                print(f"   Current addition rate: {df['substrate_addition_rate'].iloc[-1]:.2f} mM/h")
                print(f"   Average rate: {df['substrate_addition_rate'].mean():.2f} mM/h")

            if 'reservoir_concentration' in df.columns:
                print("\n🧪 Reservoir Concentration:")
                print(f"   Current: {df['reservoir_concentration'].iloc[-1]:.2f} mM")
                print(f"   Range: {df['reservoir_concentration'].min():.2f} - {df['reservoir_concentration'].max():.2f} mM")

        except Exception as e:
            print(f"❌ Error reading simulation data: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ No simulation data file found")

        # List available simulations
        sim_dir = Path("/home/uge/mfc-project/q-learning-mfcs/data/simulation_data")
        gui_sims = sorted([d for d in sim_dir.iterdir() if d.is_dir() and d.name.startswith("gui_simulation")],
                         key=lambda x: x.name, reverse=True)

        if gui_sims:
            print("\n📁 Available GUI simulations:")
            for sim in gui_sims[:5]:
                print(f"   - {sim.name}")

if __name__ == "__main__":
    inspect_latest_simulation()
