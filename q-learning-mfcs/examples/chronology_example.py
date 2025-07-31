#!/usr/bin/env python3
"""
Example: Using Chronology and Browser Download Features

This example demonstrates how to use the new chronology system and browser download
capabilities with MFC simulations.

Created: 2025-07-31
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulation_helpers import SimulationRunner, quick_simulation_with_chronology
from config.qlearning_config import DEFAULT_QLEARNING_CONFIG
from config.sensor_config import DEFAULT_SENSOR_CONFIG
from config.simulation_chronology import get_chronology_manager


def mock_simulation_function() -> dict:
    """
    Mock simulation function for demonstration.
    Replace this with your actual simulation code.
    """
    print("🔬 Running mock MFC simulation...")
    
    # Simulate some processing time
    time.sleep(2)
    
    # Generate mock results
    duration_hours = 24.0
    time_points = np.linspace(0, duration_hours, 100)
    
    # Mock time series data
    power = 2.5 + 0.5 * np.sin(time_points * 0.5) + np.random.normal(0, 0.1, len(time_points))
    voltage = 0.8 + 0.1 * np.sin(time_points * 0.3) + np.random.normal(0, 0.02, len(time_points))
    current = power / voltage
    
    biofilm_thickness = 10 + 5 * (1 - np.exp(-time_points / 8)) + np.random.normal(0, 0.5, len(time_points))
    
    results = {
        'total_energy': float(np.trapz(power, time_points) * 3600),  # J
        'average_power': float(np.mean(power)),  # W
        'peak_power': float(np.max(power)),  # W
        'coulombic_efficiency': 0.85,
        'final_biofilm_thickness': float(biofilm_thickness[-1]),
        'final_current_density': float(current[-1] / 0.01),  # A/m² (assuming 0.01 m² electrode)
        
        'time_series': {
            'time': time_points.tolist(),
            'power': power.tolist(),
            'voltage': voltage.tolist(),
            'current': current.tolist(),
            'biofilm_thickness': biofilm_thickness.tolist()
        },
        
        'convergence_achieved': True,
        'iterations_completed': 500
    }
    
    print(f"✅ Simulation completed! Average power: {results['average_power']:.3f} W")
    return results


def example_1_quick_simulation():
    """Example 1: Quick simulation with chronology tracking."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Quick Simulation with Chronology")
    print("="*60)
    
    results = quick_simulation_with_chronology(
        simulation_name="Quick Test Simulation",
        simulation_func=mock_simulation_function,
        description="Example simulation to test chronology system",
        tags=["example", "test", "mock"]
    )
    
    print(f"📝 Simulation entry ID: {results['chronology_entry_id']}")
    print(f"📊 Total energy generated: {results.get('total_energy', 'N/A'):.1f} J")
    
    # Show download files
    if results.get('download_files'):
        print("\n📁 Files prepared for download:")
        for file_type, path in results['download_files'].items():
            print(f"  - {file_type}: {path}")


def example_2_full_simulation_runner():
    """Example 2: Full simulation runner with custom parameters."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Full Simulation Runner")
    print("="*60)
    
    runner = SimulationRunner(output_dir="examples/simulation_outputs")
    
    # Custom parameters
    custom_params = {
        'substrate_concentration': 20.0,  # mM
        'temperature': 30.0,  # °C
        'cell_count': 5,
        'bacterial_species': ['S. oneidensis', 'G. sulfurreducens']
    }
    
    results = runner.run_simulation_with_tracking(
        simulation_name="Custom Parameter Study",
        simulation_func=mock_simulation_function,
        description="Testing different substrate concentrations and bacterial mixtures",
        duration_hours=24.0,
        qlearning_config=DEFAULT_QLEARNING_CONFIG,
        sensor_config=DEFAULT_SENSOR_CONFIG,
        parameters=custom_params,
        tags=["parameter-study", "substrate", "mixed-culture"],
        enable_browser_download=True
    )
    
    print(f"📝 Simulation entry ID: {results['chronology_entry_id']}")
    print(f"📊 Peak power: {results.get('peak_power', 'N/A'):.3f} W")
    print(f"⚡ Coulombic efficiency: {results.get('coulombic_efficiency', 'N/A'):.2%}")
    
    # Show metadata
    metadata = results.get('simulation_metadata', {})
    print(f"⏱️  Execution time: {metadata.get('execution_time_seconds', 0):.2f} seconds")


def example_3_chronology_management():
    """Example 3: Managing chronology data."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Chronology Management")
    print("="*60)
    
    manager = get_chronology_manager()
    
    # Show summary
    summary = manager.get_chronology_summary()
    print(f"📊 Total simulations: {summary['total_entries']}")
    print(f"✅ Successful runs: {summary['successful_runs']}")
    print(f"📈 Success rate: {summary['success_rate']:.1f}%")
    print(f"⏰ Total simulation time: {summary['total_simulation_time_hours']:.1f} hours")
    
    # Show recent entries
    recent = manager.chronology.get_recent_entries(3)
    print(f"\n📋 Most recent {len(recent)} simulations:")
    for entry in recent:
        status = "✅" if entry.success else "❌"
        print(f"  {status} {entry.id}: {entry.simulation_name}")
        print(f"     {entry.timestamp} | {entry.duration_hours:.1f}h")
        if entry.tags:
            print(f"     Tags: {', '.join(entry.tags)}")
    
    # Show available tags
    all_entries = manager.chronology.entries
    all_tags = set()
    for entry in all_entries:
        all_tags.update(entry.tags)
    
    if all_tags:
        print(f"\n🏷️  Available tags: {', '.join(sorted(all_tags))}")


def example_4_export_chronology():
    """Example 4: Export chronology to different formats."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Export Chronology")
    print("="*60)
    
    manager = get_chronology_manager()
    
    # Create export directory
    export_dir = Path("examples/exports")
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # Export to YAML
    yaml_path = export_dir / "simulation_chronology_export.yaml"
    manager.export_chronology_yaml(yaml_path)
    print(f"📤 YAML export: {yaml_path}")
    
    # Export to JSON
    json_path = export_dir / "simulation_chronology_export.json"
    manager.export_chronology_json(json_path)
    print(f"📤 JSON export: {json_path}")
    
    print("\n💡 You can import these files later using:")
    print("    python src/chronology_cli.py import --file exports/simulation_chronology_export.yaml")


def example_5_browser_download_demo():
    """Example 5: Demonstrate browser download server."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Browser Download Server")
    print("="*60)
    
    print("🌐 To start the browser download server, run:")
    print("    python src/web_download_server.py --interface fastapi --port 8080")
    print()
    print("Or for Streamlit interface:")
    print("    streamlit run src/web_download_server.py")
    print()
    print("The server will provide:")
    print("  - 📄 Web interface to browse and download simulation files")
    print("  - 🔗 Direct download links for all result files")
    print("  - 📊 JSON API for programmatic access")
    print("  - 🔍 Filtering and search capabilities")


def main():
    """Run all examples."""
    print("🔬 MFC Simulation Chronology & Browser Download Examples")
    print("=" * 70)
    
    try:
        example_1_quick_simulation()
        example_2_full_simulation_runner()
        example_3_chronology_management()
        example_4_export_chronology()
        example_5_browser_download_demo()
        
        print("\n" + "="*70)
        print("🎉 All examples completed successfully!")
        print("="*70)
        
        print("\n💡 Next steps:")
        print("1. Check the chronology file: simulation_chronology.yaml")
        print("2. Explore exports in: examples/exports/")
        print("3. Start the download server to browse results")
        print("4. Use the CLI tool: python src/chronology_cli.py --help")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()