"""Tests for chronology_cli.py - coverage target 98%+."""
import sys
import os
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


@pytest.fixture
def mock_manager():
    """Create a mock chronology manager."""
    mgr = MagicMock()
    entry = MagicMock()
    entry.tags = ["tag1", "tag2"]
    entry.description = "Test simulation"
    entry.results_summary = {"total_energy": 1.5, "average_power": 0.3}
    entry.error_message = "Some error"
    entry.result_files = {"csv": "/tmp/data.csv"}
    entry.qlearning_config = {"lr": 0.1}
    entry.sensor_config = {"eis": True}
    entry.id = "abc123"
    entry.simulation_name = "test_sim"
    entry.timestamp = "2025-01-01"
    entry.success = True
    entry.duration_hours = 10.0
    entry.execution_time_seconds = 5.0
    entry.parameters = {}

    mgr.get_chronology_summary.return_value = {"unique_tags": ["tag1"]}
    mgr.chronology.get_recent_entries.return_value = [entry]
    mgr.chronology.entries = [entry]
    mgr.chronology.get_entry_by_id.return_value = entry
    mgr.chronology.get_entries_by_tag.return_value = [entry]
    return mgr


@pytest.fixture
def mock_entry_no_results():
    """Entry with no results_summary."""
    entry = MagicMock()
    entry.tags = []
    entry.description = ""
    entry.results_summary = {}
    entry.error_message = ""
    entry.result_files = {}
    entry.qlearning_config = None
    entry.sensor_config = None
    entry.id = "def456"
    return entry


class TestCmdSummary:
    def test_summary_with_tags(self, mock_manager):
        with patch("chronology_cli.get_chronology_manager", return_value=mock_manager):
            from chronology_cli import cmd_summary
            args = MagicMock()
            args.chronology_file = "test.yaml"
            cmd_summary(args)

    def test_summary_no_tags(self, mock_manager):
        mock_manager.get_chronology_summary.return_value = {"unique_tags": []}
        with patch("chronology_cli.get_chronology_manager", return_value=mock_manager):
            from chronology_cli import cmd_summary
            args = MagicMock()
            args.chronology_file = "test.yaml"
            cmd_summary(args)


class TestCmdList:
    def test_list_recent(self, mock_manager):
        with patch("chronology_cli.get_chronology_manager", return_value=mock_manager):
            from chronology_cli import cmd_list
            args = MagicMock()
            args.chronology_file = "test.yaml"
            args.recent = 5
            cmd_list(args)

    def test_list_all(self, mock_manager):
        with patch("chronology_cli.get_chronology_manager", return_value=mock_manager):
            from chronology_cli import cmd_list
            args = MagicMock()
            args.chronology_file = "test.yaml"
            args.recent = None
            cmd_list(args)

    def test_list_entry_no_tags_no_desc(self, mock_manager, mock_entry_no_results):
        mock_manager.chronology.entries = [mock_entry_no_results]
        with patch("chronology_cli.get_chronology_manager", return_value=mock_manager):
            from chronology_cli import cmd_list
            args = MagicMock()
            args.chronology_file = "test.yaml"
            args.recent = None
            cmd_list(args)

    def test_list_entry_with_non_numeric_results(self, mock_manager):
        entry = MagicMock()
        entry.tags = []
        entry.description = ""
        entry.results_summary = {"total_energy": "N/A", "label": "test"}
        mock_manager.chronology.entries = [entry]
        with patch("chronology_cli.get_chronology_manager", return_value=mock_manager):
            from chronology_cli import cmd_list
            args = MagicMock()
            args.chronology_file = "test.yaml"
            args.recent = None
            cmd_list(args)


class TestCmdShow:
    def test_show_full_entry(self, mock_manager):
        with patch("chronology_cli.get_chronology_manager", return_value=mock_manager):
            from chronology_cli import cmd_show
            args = MagicMock()
            args.chronology_file = "test.yaml"
            args.entry_id = "abc123"
            cmd_show(args)

    def test_show_not_found(self, mock_manager):
        mock_manager.chronology.get_entry_by_id.return_value = None
        with patch("chronology_cli.get_chronology_manager", return_value=mock_manager):
            from chronology_cli import cmd_show
            args = MagicMock()
            args.chronology_file = "test.yaml"
            args.entry_id = "nonexistent"
            cmd_show(args)

    def test_show_no_optional_fields(self, mock_manager, mock_entry_no_results):
        mock_manager.chronology.get_entry_by_id.return_value = mock_entry_no_results
        with patch("chronology_cli.get_chronology_manager", return_value=mock_manager):
            from chronology_cli import cmd_show
            args = MagicMock()
            args.chronology_file = "test.yaml"
            args.entry_id = "def456"
            cmd_show(args)


class TestCmdExport:
    def test_export_yaml(self, mock_manager):
        with patch("chronology_cli.get_chronology_manager", return_value=mock_manager):
            from chronology_cli import cmd_export
            args = MagicMock()
            args.chronology_file = "test.yaml"
            args.format = "yaml"
            args.output = "/tmp/export.yaml"
            cmd_export(args)
            mock_manager.export_chronology_yaml.assert_called_once()

    def test_export_json(self, mock_manager):
        with patch("chronology_cli.get_chronology_manager", return_value=mock_manager):
            from chronology_cli import cmd_export
            args = MagicMock()
            args.chronology_file = "test.yaml"
            args.format = "json"
            args.output = "/tmp/export.json"
            cmd_export(args)
            mock_manager.export_chronology_json.assert_called_once()

    def test_export_unknown_format(self, mock_manager):
        with patch("chronology_cli.get_chronology_manager", return_value=mock_manager):
            from chronology_cli import cmd_export
            args = MagicMock()
            args.chronology_file = "test.yaml"
            args.format = "csv"
            args.output = "/tmp/export.csv"
            cmd_export(args)


class TestCmdImport:
    def test_import_file_not_found(self, tmp_path):
        from chronology_cli import cmd_import
        args = MagicMock()
        args.file = str(tmp_path / "nonexistent.yaml")
        args.chronology_file = "test.yaml"
        cmd_import(args)

    def test_import_force(self, tmp_path, mock_manager):
        f = tmp_path / "import.yaml"
        f.write_text("data: test")
        with patch("chronology_cli.get_chronology_manager", return_value=mock_manager):
            from chronology_cli import cmd_import
            args = MagicMock()
            args.file = str(f)
            args.force = True
            args.chronology_file = "test.yaml"
            cmd_import(args)
            mock_manager.import_chronology_yaml.assert_called_once()

    def test_import_confirm_yes(self, tmp_path, mock_manager):
        f = tmp_path / "import.yaml"
        f.write_text("data: test")
        with patch("chronology_cli.get_chronology_manager", return_value=mock_manager):
            with patch("builtins.input", return_value="y"):
                from chronology_cli import cmd_import
                args = MagicMock()
                args.file = str(f)
                args.force = False
                args.chronology_file = "test.yaml"
                cmd_import(args)

    def test_import_confirm_no(self, tmp_path):
        f = tmp_path / "import.yaml"
        f.write_text("data: test")
        with patch("builtins.input", return_value="n"):
            from chronology_cli import cmd_import
            args = MagicMock()
            args.file = str(f)
            args.force = False
            args.chronology_file = "test.yaml"
            cmd_import(args)

    def test_import_error_suppressed(self, tmp_path, mock_manager):
        f = tmp_path / "import.yaml"
        f.write_text("data: test")
        mock_manager.import_chronology_yaml.side_effect = Exception("import error")
        with patch("chronology_cli.get_chronology_manager", return_value=mock_manager):
            from chronology_cli import cmd_import
            args = MagicMock()
            args.file = str(f)
            args.force = True
            args.chronology_file = "test.yaml"
            cmd_import(args)


class TestCmdTags:
    def test_tags_specific_tag(self, mock_manager):
        with patch("chronology_cli.get_chronology_manager", return_value=mock_manager):
            from chronology_cli import cmd_tags
            args = MagicMock()
            args.chronology_file = "test.yaml"
            args.tag = "tag1"
            cmd_tags(args)

    def test_tags_all(self, mock_manager):
        with patch("chronology_cli.get_chronology_manager", return_value=mock_manager):
            from chronology_cli import cmd_tags
            args = MagicMock()
            args.chronology_file = "test.yaml"
            args.tag = None
            cmd_tags(args)

    def test_tags_entry_no_description(self, mock_manager, mock_entry_no_results):
        mock_manager.chronology.get_entries_by_tag.return_value = [mock_entry_no_results]
        with patch("chronology_cli.get_chronology_manager", return_value=mock_manager):
            from chronology_cli import cmd_tags
            args = MagicMock()
            args.chronology_file = "test.yaml"
            args.tag = "tag1"
            cmd_tags(args)


class TestMain:
    def test_no_command(self):
        with patch("chronology_cli.get_chronology_manager"):
            with patch("sys.argv", ["chronology_cli.py"]):
                from chronology_cli import main
                main()

    def test_summary_command(self, mock_manager):
        with patch("chronology_cli.get_chronology_manager", return_value=mock_manager):
            with patch("sys.argv", ["chronology_cli.py", "summary"]):
                from chronology_cli import main
                main()

    def test_command_raises_exception(self, mock_manager):
        mock_manager.get_chronology_summary.side_effect = Exception("fail")
        with patch("chronology_cli.get_chronology_manager", return_value=mock_manager):
            with patch("sys.argv", ["chronology_cli.py", "summary"]):
                from chronology_cli import main
                with pytest.raises(SystemExit):
                    main()

    def test_unknown_command(self):
        with patch("chronology_cli.get_chronology_manager"):
            with patch("sys.argv", ["chronology_cli.py", "unknown_cmd"]):
                from chronology_cli import main
                with pytest.raises(SystemExit):
                    main()
