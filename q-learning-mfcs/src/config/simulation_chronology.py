"""
Simulation Chronology Management System

This module provides functionality to save, maintain, and export chronological
records of simulation parameters and results with YAML export/import capabilities.

Created: 2025-07-31
"""

import yaml
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
import logging
import uuid

from .config_io import dataclass_to_dict, convert_values_for_serialization
from .qlearning_config import QLearningConfig
from .sensor_config import SensorConfig

logger = logging.getLogger(__name__)


@dataclass
class SimulationEntry:
    """Single simulation run entry with metadata and parameters."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    simulation_name: str = ""
    description: str = ""
    duration_hours: float = 0.0

    # Configuration parameters
    qlearning_config: Optional[Dict[str, Any]] = None
    sensor_config: Optional[Dict[str, Any]] = None

    # Simulation parameters
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Results summary
    results_summary: Dict[str, Any] = field(default_factory=dict)

    # File paths for detailed data
    result_files: Dict[str, str] = field(default_factory=dict)

    # Tags for categorization
    tags: List[str] = field(default_factory=list)

    # Execution metadata
    execution_time_seconds: float = 0.0
    success: bool = True
    error_message: str = ""


@dataclass
class SimulationChronology:
    """Complete chronological record of simulation runs."""

    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: str = "1.0"

    entries: List[SimulationEntry] = field(default_factory=list)

    # Metadata
    project_name: str = "MFC Q-Learning Simulations"
    author: str = ""
    notes: str = ""

    def add_entry(self, entry: SimulationEntry) -> None:
        """Add a new simulation entry to the chronology."""
        self.entries.append(entry)
        self.last_updated = datetime.now(timezone.utc).isoformat()
        logger.info(f"Added simulation entry: {entry.simulation_name} ({entry.id})")

    def get_entry_by_id(self, entry_id: str) -> Optional[SimulationEntry]:
        """Retrieve entry by ID."""
        for entry in self.entries:
            if entry.id == entry_id:
                return entry
        return None

    def get_entries_by_tag(self, tag: str) -> List[SimulationEntry]:
        """Get all entries with a specific tag."""
        return [entry for entry in self.entries if tag in entry.tags]

    def get_recent_entries(self, count: int = 10) -> List[SimulationEntry]:
        """Get most recent entries."""
        return sorted(self.entries, key=lambda x: x.timestamp, reverse=True)[:count]


class ChronologyManager:
    """Manages simulation chronology with YAML export/import capabilities."""

    def __init__(self, chronology_file: Union[str, Path] = "simulation_chronology.yaml"):
        """
        Initialize chronology manager.
        
        Args:
            chronology_file: Path to chronology YAML file
        """
        self.chronology_file = Path(chronology_file)
        self.chronology = SimulationChronology()

        # Load existing chronology if file exists
        if self.chronology_file.exists():
            self.load_chronology()

    def create_entry(self,
                    simulation_name: str,
                    description: str = "",
                    duration_hours: float = 0.0,
                    qlearning_config: Optional[QLearningConfig] = None,
                    sensor_config: Optional[SensorConfig] = None,
                    parameters: Optional[Dict[str, Any]] = None,
                    tags: Optional[List[str]] = None) -> SimulationEntry:
        """
        Create a new simulation entry.
        
        Args:
            simulation_name: Name of the simulation
            description: Description of the simulation
            duration_hours: Duration of simulation in hours
            qlearning_config: Q-learning configuration
            sensor_config: Sensor configuration  
            parameters: Additional simulation parameters
            tags: Tags for categorization
            
        Returns:
            New simulation entry
        """
        entry = SimulationEntry(
            simulation_name=simulation_name,
            description=description,
            duration_hours=duration_hours,
            parameters=parameters or {},
            tags=tags or []
        )

        # Convert configurations to dictionaries
        if qlearning_config:
            entry.qlearning_config = dataclass_to_dict(qlearning_config)

        if sensor_config:
            entry.sensor_config = dataclass_to_dict(sensor_config)

        return entry

    def add_simulation_entry(self, entry: SimulationEntry) -> None:
        """Add simulation entry to chronology."""
        self.chronology.add_entry(entry)
        self.save_chronology()

    def update_entry_results(self,
                           entry_id: str,
                           results_summary: Dict[str, Any],
                           result_files: Optional[Dict[str, str]] = None,
                           execution_time: Optional[float] = None,
                           success: bool = True,
                           error_message: str = "") -> bool:
        """
        Update simulation entry with results.
        
        Args:
            entry_id: ID of the entry to update
            results_summary: Summary of simulation results
            result_files: Paths to detailed result files
            execution_time: Execution time in seconds
            success: Whether simulation succeeded
            error_message: Error message if failed
            
        Returns:
            True if entry was found and updated
        """
        entry = self.chronology.get_entry_by_id(entry_id)
        if not entry:
            logger.warning(f"Entry not found: {entry_id}")
            return False

        entry.results_summary = results_summary
        entry.result_files = result_files or {}
        entry.success = success
        entry.error_message = error_message

        if execution_time:
            entry.execution_time_seconds = execution_time

        self.chronology.last_updated = datetime.now(timezone.utc).isoformat()
        self.save_chronology()

        logger.info(f"Updated results for entry: {entry_id}")
        return True

    def save_chronology(self) -> None:
        """Save chronology to YAML file."""
        try:
            # Convert to dictionary and serialize properly
            chronology_dict = dataclass_to_dict(self.chronology)
            chronology_dict = convert_values_for_serialization(chronology_dict)

            self.chronology_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.chronology_file, 'w') as f:
                yaml.dump(chronology_dict, f,
                         default_flow_style=False,
                         indent=2,
                         sort_keys=False)

            logger.info(f"Chronology saved to: {self.chronology_file}")

        except Exception as e:
            logger.error(f"Failed to save chronology: {e}")
            raise

    def load_chronology(self) -> None:
        """Load chronology from YAML file."""
        try:
            with open(self.chronology_file, 'r') as f:
                data = yaml.safe_load(f)

            if not data:
                logger.warning("Empty chronology file, using default")
                return

            # Reconstruct chronology object
            self.chronology = SimulationChronology()

            # Set basic metadata
            self.chronology.created_at = data.get('created_at', self.chronology.created_at)
            self.chronology.last_updated = data.get('last_updated', self.chronology.last_updated)
            self.chronology.version = data.get('version', self.chronology.version)
            self.chronology.project_name = data.get('project_name', self.chronology.project_name)
            self.chronology.author = data.get('author', self.chronology.author)
            self.chronology.notes = data.get('notes', self.chronology.notes)

            # Reconstruct entries
            for entry_data in data.get('entries', []):
                entry = SimulationEntry(
                    id=entry_data.get('id', str(uuid.uuid4())[:8]),
                    timestamp=entry_data.get('timestamp', datetime.now(timezone.utc).isoformat()),
                    simulation_name=entry_data.get('simulation_name', ''),
                    description=entry_data.get('description', ''),
                    duration_hours=entry_data.get('duration_hours', 0.0),
                    qlearning_config=entry_data.get('qlearning_config'),
                    sensor_config=entry_data.get('sensor_config'),
                    parameters=entry_data.get('parameters', {}),
                    results_summary=entry_data.get('results_summary', {}),
                    result_files=entry_data.get('result_files', {}),
                    tags=entry_data.get('tags', []),
                    execution_time_seconds=entry_data.get('execution_time_seconds', 0.0),
                    success=entry_data.get('success', True),
                    error_message=entry_data.get('error_message', '')
                )
                self.chronology.entries.append(entry)

            logger.info(f"Loaded chronology with {len(self.chronology.entries)} entries")

        except FileNotFoundError:
            logger.info("No existing chronology file found")
        except Exception as e:
            logger.error(f"Failed to load chronology: {e}")
            raise

    def export_chronology_yaml(self, output_path: Union[str, Path]) -> None:
        """Export chronology to a specific YAML file."""
        output_path = Path(output_path)

        # Convert to dictionary
        chronology_dict = dataclass_to_dict(self.chronology)
        chronology_dict = convert_values_for_serialization(chronology_dict)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            yaml.dump(chronology_dict, f,
                     default_flow_style=False,
                     indent=2,
                     sort_keys=False)

        logger.info(f"Chronology exported to: {output_path}")

    def export_chronology_json(self, output_path: Union[str, Path]) -> None:
        """Export chronology to JSON format."""
        output_path = Path(output_path)

        chronology_dict = dataclass_to_dict(self.chronology)
        chronology_dict = convert_values_for_serialization(chronology_dict)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(chronology_dict, f, indent=2)

        logger.info(f"Chronology exported to JSON: {output_path}")

    def import_chronology_yaml(self, input_path: Union[str, Path]) -> None:
        """Import chronology from YAML file."""
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Chronology file not found: {input_path}")

        # Backup current chronology
        backup_path = self.chronology_file.with_suffix('.backup.yaml')
        if self.chronology_file.exists():
            self.export_chronology_yaml(backup_path)
            logger.info(f"Backed up current chronology to: {backup_path}")

        # Load new chronology
        self.chronology_file = input_path
        self.load_chronology()

        logger.info(f"Imported chronology from: {input_path}")

    def get_chronology_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the chronology."""
        total_entries = len(self.chronology.entries)
        successful_runs = len([e for e in self.chronology.entries if e.success])
        failed_runs = total_entries - successful_runs

        total_simulation_time = sum(e.duration_hours for e in self.chronology.entries)
        total_execution_time = sum(e.execution_time_seconds for e in self.chronology.entries)

        # Get all unique tags
        all_tags = set()
        for entry in self.chronology.entries:
            all_tags.update(entry.tags)

        return {
            'total_entries': total_entries,
            'successful_runs': successful_runs,
            'failed_runs': failed_runs,
            'success_rate': (successful_runs / total_entries * 100) if total_entries > 0 else 0,
            'total_simulation_time_hours': total_simulation_time,
            'total_execution_time_seconds': total_execution_time,
            'unique_tags': sorted(list(all_tags)),
            'created_at': self.chronology.created_at,
            'last_updated': self.chronology.last_updated
        }


# Global chronology manager instance
_chronology_manager: Optional[ChronologyManager] = None


def get_chronology_manager(chronology_file: Union[str, Path] = "simulation_chronology.yaml") -> ChronologyManager:
    """Get or create global chronology manager instance."""
    global _chronology_manager

    if _chronology_manager is None:
        _chronology_manager = ChronologyManager(chronology_file)

    return _chronology_manager
