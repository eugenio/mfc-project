#!/usr/bin/env python3
"""Example: Using Chronology and Browser Download Features.

This example demonstrates how to use the new chronology system and browser download
capabilities with MFC simulations.

Created: 2025-07-31
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config.qlearning_config import DEFAULT_QLEARNING_CONFIG
from config.sensor_config import DEFAULT_SENSOR_CONFIG
from config.simulation_chronology import get_chronology_manager
from simulation_helpers import SimulationRunner, quick_simulation_with_chronology


def mock_simulation_function() -> dict:
    """Mock simulation function for demonstration.
    Replace this with your actual simulation code.
    """
    # Simulate some processing time
    time.sleep(2)

    # Generate mock results
    duration_hours = 24.0
    time_points = np.linspace(0, duration_hours, 100)

    # Mock time series data
    power = (
        2.5
        + 0.5 * np.sin(time_points * 0.5)
        + np.random.normal(0, 0.1, len(time_points))
    )
    voltage = (
        0.8
        + 0.1 * np.sin(time_points * 0.3)
        + np.random.normal(0, 0.02, len(time_points))
    )
    current = power / voltage

    biofilm_thickness = (
        10
        + 5 * (1 - np.exp(-time_points / 8))
        + np.random.normal(0, 0.5, len(time_points))
    )

    return {
        "total_energy": float(np.trapezoid(power, time_points) * 3600),  # J
        "average_power": float(np.mean(power)),  # W
        "peak_power": float(np.max(power)),  # W
        "coulombic_efficiency": 0.85,
        "final_biofilm_thickness": float(biofilm_thickness[-1]),
        "final_current_density": float(
            current[-1] / 0.01,
        ),  # A/m² (assuming 0.01 m² electrode)
        "time_series": {
            "time": time_points.tolist(),
            "power": power.tolist(),
            "voltage": voltage.tolist(),
            "current": current.tolist(),
            "biofilm_thickness": biofilm_thickness.tolist(),
        },
        "convergence_achieved": True,
        "iterations_completed": 500,
    }


def example_1_quick_simulation() -> None:
    """Example 1: Quick simulation with chronology tracking."""
    results = quick_simulation_with_chronology(
        simulation_name="Quick Test Simulation",
        simulation_func=mock_simulation_function,
        description="Example simulation to test chronology system",
        tags=["example", "test", "mock"],
    )

    # Show download files
    if results.get("download_files"):
        for _file_type, _path in results["download_files"].items():
            pass


def example_2_full_simulation_runner() -> None:
    """Example 2: Full simulation runner with custom parameters."""
    runner = SimulationRunner(output_dir="examples/simulation_outputs")

    # Custom parameters
    custom_params = {
        "substrate_concentration": 20.0,  # mM
        "temperature": 30.0,  # °C
        "cell_count": 5,
        "bacterial_species": ["S. oneidensis", "G. sulfurreducens"],
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
        enable_browser_download=True,
    )

    # Show metadata
    results.get("simulation_metadata", {})


def example_3_chronology_management() -> None:
    """Example 3: Managing chronology data."""
    manager = get_chronology_manager()

    # Show summary
    manager.get_chronology_summary()

    # Show recent entries
    recent = manager.chronology.get_recent_entries(3)
    for entry in recent:
        if entry.tags:
            pass

    # Show available tags
    all_entries = manager.chronology.entries
    all_tags = set()
    for entry in all_entries:
        all_tags.update(entry.tags)

    if all_tags:
        pass


def example_4_export_chronology() -> None:
    """Example 4: Export chronology to different formats."""
    manager = get_chronology_manager()

    # Create export directory
    export_dir = Path("examples/exports")
    export_dir.mkdir(parents=True, exist_ok=True)

    # Export to YAML
    yaml_path = export_dir / "simulation_chronology_export.yaml"
    manager.export_chronology_yaml(yaml_path)

    # Export to JSON
    json_path = export_dir / "simulation_chronology_export.json"
    manager.export_chronology_json(json_path)


def example_5_browser_download_demo() -> None:
    """Example 5: Demonstrate browser download server."""


def main() -> None:
    """Run all examples."""
    try:
        example_1_quick_simulation()
        example_2_full_simulation_runner()
        example_3_chronology_management()
        example_4_export_chronology()
        example_5_browser_download_demo()

    except Exception:
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
