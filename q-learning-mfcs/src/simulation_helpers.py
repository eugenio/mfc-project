"""
Simulation Helper Functions

Provides utility functions for simulation execution with integrated chronology
tracking and browser-friendly data export capabilities.

Created: 2025-07-31
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from config.qlearning_config import QLearningConfig
from config.sensor_config import SensorConfig
from config.simulation_chronology import get_chronology_manager

logger = logging.getLogger(__name__)


class SimulationRunner:
    """Enhanced simulation runner with chronology tracking."""

    def __init__(self, output_dir: Union[str, Path] = "simulation_outputs"):
        """
        Initialize simulation runner.
        
        Args:
            output_dir: Directory for simulation outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chronology_manager = get_chronology_manager()

    def run_simulation_with_tracking(self,
                                   simulation_name: str,
                                   simulation_func: Callable[[], Dict[str, Any]],
                                   description: str = "",
                                   duration_hours: float = 0.0,
                                   qlearning_config: Optional[QLearningConfig] = None,
                                   sensor_config: Optional[SensorConfig] = None,
                                   parameters: Optional[Dict[str, Any]] = None,
                                   tags: Optional[list] = None,
                                   enable_browser_download: bool = True) -> Dict[str, Any]:
        """
        Run simulation with full chronology tracking and browser download support.
        
        Args:
            simulation_name: Name of the simulation
            simulation_func: Function that runs the simulation and returns results
            description: Description of the simulation
            duration_hours: Expected duration in hours
            qlearning_config: Q-learning configuration
            sensor_config: Sensor configuration
            parameters: Additional parameters
            tags: Tags for categorization
            enable_browser_download: Whether to prepare files for browser download
            
        Returns:
            Enhanced results dictionary with chronology info
        """
        # Create chronology entry
        entry = self.chronology_manager.create_entry(
            simulation_name=simulation_name,
            description=description,
            duration_hours=duration_hours,
            qlearning_config=qlearning_config,
            sensor_config=sensor_config,
            parameters=parameters,
            tags=tags or []
        )

        # Add entry to chronology
        self.chronology_manager.add_simulation_entry(entry)

        logger.info(f"Starting simulation: {simulation_name} (ID: {entry.id})")

        # Run simulation with timing
        start_time = time.time()
        success = True
        error_message = ""
        results = {}

        try:
            results = simulation_func()
            logger.info(f"Simulation completed successfully: {entry.id}")

        except Exception as e:
            success = False
            error_message = str(e)
            logger.error(f"Simulation failed: {entry.id} - {error_message}")
            results = {"error": error_message}

        execution_time = time.time() - start_time

        # Prepare result files for browser download
        result_files = {}
        if enable_browser_download and results:
            result_files = self._prepare_browser_download_files(entry.id, results,
                                                              qlearning_config, sensor_config)

        # Extract results summary
        results_summary = self._extract_results_summary(results)

        # Update chronology entry with results
        self.chronology_manager.update_entry_results(
            entry_id=entry.id,
            results_summary=results_summary,
            result_files=result_files,
            execution_time=execution_time,
            success=success,
            error_message=error_message
        )

        # Add chronology metadata to results
        enhanced_results = {
            **results,
            "chronology_entry_id": entry.id,
            "simulation_metadata": {
                "name": simulation_name,
                "timestamp": entry.timestamp,
                "execution_time_seconds": execution_time,
                "success": success,
                "description": description,
                "tags": tags or []
            },
            "download_files": result_files if enable_browser_download else {}
        }

        return enhanced_results

    def _extract_results_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics for chronology summary."""
        summary = {}

        # Common result fields to extract
        key_fields = [
            'total_energy', 'average_power', 'peak_power',
            'coulombic_efficiency', 'average_coulombic_efficiency',
            'final_biofilm_thickness', 'final_current_density',
            'convergence_achieved', 'iterations_completed'
        ]

        for field in key_fields:
            if field in results:
                summary[field] = results[field]

        # Extract time series summaries
        if 'time_series' in results:
            time_series = results['time_series']
            if isinstance(time_series, dict):
                summary['time_series_length'] = len(time_series.get('time', []))

                # Statistical summaries
                for key in ['power', 'voltage', 'current']:
                    if key in time_series:
                        data = time_series[key]
                        if data:
                            summary[f'{key}_mean'] = float(sum(data) / len(data))
                            summary[f'{key}_max'] = float(max(data))
                            summary[f'{key}_min'] = float(min(data))

        return summary

    def _prepare_browser_download_files(self,
                                      entry_id: str,
                                      results: Dict[str, Any],
                                      qlearning_config: Optional[QLearningConfig] = None,
                                      sensor_config: Optional[SensorConfig] = None) -> Dict[str, str]:
        """
        Prepare files for browser download instead of fixed locations.
        
        Args:
            entry_id: Simulation entry ID
            results: Simulation results
            qlearning_config: Q-learning configuration
            sensor_config: Sensor configuration
            
        Returns:
            Dictionary mapping file types to relative paths
        """
        download_files = {}

        # Create timestamped directory for this simulation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sim_dir = self.output_dir / f"{entry_id}_{timestamp}"
        sim_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Full results JSON
            results_file = sim_dir / "simulation_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            download_files['results_json'] = str(results_file.relative_to(self.output_dir.parent))

            # 2. Configuration files
            if qlearning_config:
                from config.config_io import save_config
                config_file = sim_dir / "qlearning_config.yaml"
                save_config(qlearning_config, config_file, format='yaml')
                download_files['qlearning_config'] = str(config_file.relative_to(self.output_dir.parent))

            if sensor_config:
                from config.config_io import save_config
                sensor_file = sim_dir / "sensor_config.yaml"
                save_config(sensor_config, sensor_file, format='yaml')
                download_files['sensor_config'] = str(sensor_file.relative_to(self.output_dir.parent))

            # 3. Time series CSV export
            if 'time_series' in results:
                csv_file = self._export_time_series_csv(results['time_series'], sim_dir / "time_series.csv")
                if csv_file:
                    download_files['time_series_csv'] = str(csv_file.relative_to(self.output_dir.parent))

            # 4. Summary report
            summary_file = self._create_summary_report(entry_id, results, sim_dir / "summary_report.txt")
            if summary_file:
                download_files['summary_report'] = str(summary_file.relative_to(self.output_dir.parent))

            logger.info(f"Prepared {len(download_files)} download files for entry {entry_id}")

        except Exception as e:
            logger.error(f"Failed to prepare download files: {e}")

        return download_files

    def _export_time_series_csv(self, time_series: Dict[str, Any], output_path: Path) -> Optional[Path]:
        """Export time series data to CSV format."""
        try:
            import pandas as pd

            # Convert time series to DataFrame
            df_data = {}

            for key, values in time_series.items():
                if isinstance(values, list) and values:
                    df_data[key] = values

            if not df_data:
                return None

            df = pd.DataFrame(df_data)
            df.to_csv(output_path, index=False)

            return output_path

        except ImportError:
            # Fallback to manual CSV creation
            try:
                import csv

                # Get all keys and determine max length
                keys = list(time_series.keys())
                max_len = max(len(time_series[key]) for key in keys if isinstance(time_series[key], list))

                with open(output_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(keys)  # Header

                    for i in range(max_len):
                        row = []
                        for key in keys:
                            values = time_series[key]
                            if isinstance(values, list) and i < len(values):
                                row.append(values[i])
                            else:
                                row.append('')
                        writer.writerow(row)

                return output_path

            except Exception as e:
                logger.error(f"Failed to create CSV: {e}")
                return None

        except Exception as e:
            logger.error(f"Failed to export time series CSV: {e}")
            return None

    def _create_summary_report(self, entry_id: str, results: Dict[str, Any], output_path: Path) -> Optional[Path]:
        """Create a human-readable summary report."""
        try:
            with open(output_path, 'w') as f:
                f.write("MFC SIMULATION SUMMARY REPORT\n")
                f.write("=" * 50 + "\n\n")

                f.write(f"Simulation ID: {entry_id}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # Key performance metrics
                f.write("PERFORMANCE METRICS\n")
                f.write("-" * 20 + "\n")

                metrics = [
                    ('Total Energy (J)', 'total_energy'),
                    ('Average Power (W)', 'average_power'),
                    ('Peak Power (W)', 'peak_power'),
                    ('Coulombic Efficiency (%)', 'coulombic_efficiency'),
                    ('Final Biofilm Thickness (μm)', 'final_biofilm_thickness'),
                    ('Final Current Density (A/m²)', 'final_current_density')
                ]

                for label, key in metrics:
                    value = results.get(key, 'N/A')
                    if isinstance(value, (int, float)):
                        f.write(f"{label}: {value:.4f}\n")
                    else:
                        f.write(f"{label}: {value}\n")

                f.write("\n")

                # Time series summary
                if 'time_series' in results:
                    ts = results['time_series']
                    f.write("TIME SERIES DATA\n")
                    f.write("-" * 16 + "\n")
                    f.write(f"Data points: {len(ts.get('time', []))}\n")
                    f.write(f"Available series: {', '.join(ts.keys())}\n\n")

                # Configuration summary
                if 'simulation_metadata' in results:
                    meta = results['simulation_metadata']
                    f.write("SIMULATION METADATA\n")
                    f.write("-" * 19 + "\n")
                    f.write(f"Name: {meta.get('name', 'N/A')}\n")
                    f.write(f"Description: {meta.get('description', 'N/A')}\n")
                    f.write(f"Execution time: {meta.get('execution_time_seconds', 0):.2f} seconds\n")
                    f.write(f"Success: {meta.get('success', False)}\n")
                    if meta.get('tags'):
                        f.write(f"Tags: {', '.join(meta['tags'])}\n")

                f.write("\n" + "=" * 50 + "\n")
                f.write("Report generated by MFC Q-Learning Simulation System\n")

            return output_path

        except Exception as e:
            logger.error(f"Failed to create summary report: {e}")
            return None


def quick_simulation_with_chronology(simulation_name: str,
                                   simulation_func: Callable[[], Dict[str, Any]],
                                   description: str = "",
                                   tags: Optional[list] = None) -> Dict[str, Any]:
    """
    Quick utility function to run simulation with chronology tracking.
    
    Args:
        simulation_name: Name of the simulation
        simulation_func: Function that returns simulation results
        description: Description of the simulation
        tags: Tags for categorization
        
    Returns:
        Enhanced simulation results
    """
    runner = SimulationRunner()
    return runner.run_simulation_with_tracking(
        simulation_name=simulation_name,
        simulation_func=simulation_func,
        description=description,
        tags=tags
    )
