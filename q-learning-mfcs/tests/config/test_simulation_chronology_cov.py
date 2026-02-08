"""Tests for simulation_chronology module - 98%+ coverage target."""
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from config.simulation_chronology import (
    ChronologyManager,
    SimulationChronology,
    SimulationEntry,
    get_chronology_manager,
)


@pytest.fixture
def tmp_chronology_file(tmp_path):
    return tmp_path / "test_chronology.yaml"


@pytest.fixture
def manager(tmp_chronology_file):
    return ChronologyManager(chronology_file=tmp_chronology_file)


@pytest.fixture
def sample_entry():
    return SimulationEntry(
        id="abc12345",
        simulation_name="test_sim",
        description="A test simulation",
        duration_hours=10.0,
        parameters={"lr": 0.01},
        tags=["test", "demo"],
        execution_time_seconds=120.0,
        success=True,
    )


class TestSimulationEntry:
    def test_default_fields(self):
        entry = SimulationEntry()
        assert len(entry.id) == 8
        assert entry.timestamp is not None
        assert entry.simulation_name == ""
        assert entry.duration_hours == 0.0
        assert entry.parameters == {}
        assert entry.tags == []
        assert entry.success is True
        assert entry.error_message == ""

    def test_custom_fields(self, sample_entry):
        assert sample_entry.id == "abc12345"
        assert sample_entry.simulation_name == "test_sim"
        assert sample_entry.duration_hours == 10.0
        assert "test" in sample_entry.tags


class TestSimulationChronology:
    def test_defaults(self):
        chron = SimulationChronology()
        assert chron.version == "1.0"
        assert chron.entries == []
        assert chron.project_name == "MFC Q-Learning Simulations"

    def test_add_entry(self, sample_entry):
        chron = SimulationChronology()
        chron.add_entry(sample_entry)
        assert len(chron.entries) == 1
        assert chron.entries[0].id == "abc12345"

    def test_get_entry_by_id(self, sample_entry):
        chron = SimulationChronology()
        chron.add_entry(sample_entry)
        found = chron.get_entry_by_id("abc12345")
        assert found is not None
        assert found.simulation_name == "test_sim"

    def test_get_entry_by_id_not_found(self):
        chron = SimulationChronology()
        assert chron.get_entry_by_id("nonexistent") is None

    def test_get_entries_by_tag(self, sample_entry):
        chron = SimulationChronology()
        chron.add_entry(sample_entry)
        entries = chron.get_entries_by_tag("test")
        assert len(entries) == 1
        entries_none = chron.get_entries_by_tag("nonexistent")
        assert len(entries_none) == 0

    def test_get_recent_entries(self):
        chron = SimulationChronology()
        for i in range(15):
            chron.add_entry(SimulationEntry(simulation_name=f"sim_{i}"))
        recent = chron.get_recent_entries(5)
        assert len(recent) == 5
        recent_all = chron.get_recent_entries(20)
        assert len(recent_all) == 15


class TestChronologyManager:
    def test_init_no_existing_file(self, tmp_chronology_file):
        mgr = ChronologyManager(chronology_file=tmp_chronology_file)
        assert isinstance(mgr.chronology, SimulationChronology)

    def test_create_entry_basic(self, manager):
        entry = manager.create_entry(
            simulation_name="my_sim",
            description="desc",
            duration_hours=5.0,
            tags=["tag1"],
        )
        assert entry.simulation_name == "my_sim"
        assert entry.duration_hours == 5.0
        assert "tag1" in entry.tags

    def test_create_entry_with_configs(self, manager):
        mock_ql = MagicMock()
        mock_sensor = MagicMock()
        with patch("config.simulation_chronology.dataclass_to_dict") as mock_dc:
            mock_dc.return_value = {"key": "value"}
            entry = manager.create_entry(
                simulation_name="cfg_sim",
                qlearning_config=mock_ql,
                sensor_config=mock_sensor,
            )
            assert entry.qlearning_config == {"key": "value"}
            assert entry.sensor_config == {"key": "value"}

    def test_create_entry_no_configs(self, manager):
        entry = manager.create_entry(simulation_name="no_cfg")
        assert entry.qlearning_config is None
        assert entry.sensor_config is None

    def test_add_simulation_entry(self, manager, sample_entry):
        manager.add_simulation_entry(sample_entry)
        assert len(manager.chronology.entries) == 1
        assert manager.chronology_file.exists()

    def test_update_entry_results_success(self, manager, sample_entry):
        manager.add_simulation_entry(sample_entry)
        result = manager.update_entry_results(
            entry_id="abc12345",
            results_summary={"metric": 0.95},
            result_files={"data": "/path/to/file"},
            execution_time=300.0,
            success=True,
        )
        assert result is True
        entry = manager.chronology.get_entry_by_id("abc12345")
        assert entry.results_summary == {"metric": 0.95}
        assert entry.execution_time_seconds == 300.0

    def test_update_entry_results_failed(self, manager, sample_entry):
        manager.add_simulation_entry(sample_entry)
        result = manager.update_entry_results(
            entry_id="abc12345",
            results_summary={},
            success=False,
            error_message="Out of memory",
        )
        assert result is True
        entry = manager.chronology.get_entry_by_id("abc12345")
        assert entry.success is False
        assert entry.error_message == "Out of memory"

    def test_update_entry_results_not_found(self, manager):
        result = manager.update_entry_results(
            entry_id="nonexistent",
            results_summary={},
        )
        assert result is False

    def test_update_entry_no_execution_time(self, manager, sample_entry):
        manager.add_simulation_entry(sample_entry)
        result = manager.update_entry_results(
            entry_id="abc12345",
            results_summary={"m": 1},
        )
        assert result is True

    def test_save_and_load_chronology(self, manager, sample_entry):
        manager.add_simulation_entry(sample_entry)
        manager.save_chronology()
        mgr2 = ChronologyManager(chronology_file=manager.chronology_file)
        assert len(mgr2.chronology.entries) == 1
        assert mgr2.chronology.entries[0].simulation_name == "test_sim"

    def test_load_empty_file(self, tmp_path):
        f = tmp_path / "empty.yaml"
        f.write_text("")
        mgr = ChronologyManager(chronology_file=f)
        assert len(mgr.chronology.entries) == 0

    def test_load_chronology_file_not_found(self, tmp_path):
        mgr = ChronologyManager(chronology_file=tmp_path / "nope.yaml")
        assert len(mgr.chronology.entries) == 0

    def test_export_chronology_yaml(self, manager, sample_entry, tmp_path):
        manager.add_simulation_entry(sample_entry)
        out = tmp_path / "export.yaml"
        manager.export_chronology_yaml(out)
        assert out.exists()
        data = yaml.safe_load(out.read_text())
        assert len(data["entries"]) == 1

    def test_export_chronology_json(self, manager, sample_entry, tmp_path):
        manager.add_simulation_entry(sample_entry)
        out = tmp_path / "export.json"
        manager.export_chronology_json(out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert len(data["entries"]) == 1

    def test_import_chronology_yaml(self, manager, sample_entry, tmp_path):
        manager.add_simulation_entry(sample_entry)
        export_file = tmp_path / "to_import.yaml"
        manager.export_chronology_yaml(export_file)
        new_mgr = ChronologyManager(chronology_file=tmp_path / "new.yaml")
        new_mgr.import_chronology_yaml(export_file)
        assert len(new_mgr.chronology.entries) == 1

    def test_import_chronology_yaml_not_found(self, manager, tmp_path):
        with pytest.raises(FileNotFoundError):
            manager.import_chronology_yaml(tmp_path / "nope.yaml")

    def test_import_chronology_with_backup(self, manager, sample_entry, tmp_path):
        manager.add_simulation_entry(sample_entry)
        manager.save_chronology()
        export_file = tmp_path / "import_src.yaml"
        manager.export_chronology_yaml(export_file)
        manager.import_chronology_yaml(export_file)

    def test_get_chronology_summary_empty(self, manager):
        summary = manager.get_chronology_summary()
        assert summary["total_entries"] == 0
        assert summary["success_rate"] == 0

    def test_get_chronology_summary_with_data(self, manager):
        e1 = SimulationEntry(
            simulation_name="s1",
            duration_hours=5.0,
            execution_time_seconds=100.0,
            success=True,
            tags=["a"],
        )
        e2 = SimulationEntry(
            simulation_name="s2",
            duration_hours=3.0,
            execution_time_seconds=50.0,
            success=False,
            tags=["b"],
        )
        manager.add_simulation_entry(e1)
        manager.add_simulation_entry(e2)
        summary = manager.get_chronology_summary()
        assert summary["total_entries"] == 2
        assert summary["successful_runs"] == 1
        assert summary["failed_runs"] == 1
        assert summary["success_rate"] == 50.0
        assert summary["total_simulation_time_hours"] == 8.0
        assert "a" in summary["unique_tags"]
        assert "b" in summary["unique_tags"]

    def test_save_chronology_error(self, tmp_path):
        mgr = ChronologyManager(chronology_file=tmp_path / "ok.yaml")
        mgr.add_simulation_entry(SimulationEntry(simulation_name="s"))
        with patch("builtins.open", side_effect=PermissionError("denied")):
            with pytest.raises(PermissionError):
                mgr.save_chronology()

    def test_load_chronology_bad_yaml(self, tmp_path):
        f = tmp_path / "bad.yaml"
        f.write_text("{{invalid yaml::::")
        with pytest.raises(Exception):
            mgr = ChronologyManager(chronology_file=f)

    def test_load_chronology_file_not_found_path(self, tmp_path):
        mgr = ChronologyManager(chronology_file=tmp_path / "nope.yaml")
        mgr.chronology_file = tmp_path / "does_not_exist_for_real.yaml"
        mgr.load_chronology()
        assert len(mgr.chronology.entries) == 0


class TestGetChronologyManager:
    def test_singleton(self, tmp_path):
        import config.simulation_chronology as mod

        mod._chronology_manager = None
        mgr = get_chronology_manager(tmp_path / "singleton.yaml")
        assert mgr is not None
        mgr2 = get_chronology_manager()
        assert mgr2 is mgr
        mod._chronology_manager = None
