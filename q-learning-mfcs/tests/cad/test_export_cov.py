"""Comprehensive coverage tests for cad.export module."""
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Mock cadquery before importing export (it does lazy import inside functions)
_mock_cq = MagicMock()
sys.modules.setdefault("cadquery", _mock_cq)

import pytest

from cad.cad_config import StackCADConfig
from cad.export import (
    _ensure_dir,
    export_assembly,
    export_component,
    generate_bom,
    write_bom_json,
)


class TestEnsureDir:
    def test_creates_dir(self, tmp_path):
        target = tmp_path / "sub" / "dir"
        result = _ensure_dir(target)
        assert target.exists()
        assert result == target

    def test_existing_dir(self, tmp_path):
        result = _ensure_dir(tmp_path)
        assert result == tmp_path


class TestExportComponent:
    def test_default_formats(self, tmp_path):
        solid = MagicMock()
        paths = export_component(solid, "test_part", tmp_path)
        assert len(paths) == 2
        assert any(str(p).endswith(".step") for p in paths)
        assert any(str(p).endswith(".stl") for p in paths)

    def test_step_only(self, tmp_path):
        solid = MagicMock()
        paths = export_component(solid, "part", tmp_path, formats=["step"])
        assert len(paths) == 1
        assert str(paths[0]).endswith(".step")

    def test_stl_only(self, tmp_path):
        solid = MagicMock()
        paths = export_component(solid, "part", tmp_path, formats=["stl"])
        assert len(paths) == 1

    def test_unsupported_format(self, tmp_path):
        solid = MagicMock()
        with pytest.raises(ValueError, match="Unsupported format"):
            export_component(solid, "part", tmp_path, formats=["obj"])


class TestExportAssembly:
    def test_export(self, tmp_path):
        assembly = MagicMock()
        result = export_assembly(assembly, tmp_path, name="test_asm")
        assert str(result).endswith("test_asm.step")
        assembly.save.assert_called_once()


class TestGenerateBom:
    def test_default_config(self):
        cfg = StackCADConfig()
        bom = generate_bom(cfg)
        assert bom["num_cells"] == 10
        assert "parts" in bom
        assert len(bom["parts"]) >= 14

    def test_bom_parts_structure(self):
        cfg = StackCADConfig()
        bom = generate_bom(cfg)
        for part in bom["parts"]:
            assert "item" in part
            assert "part" in part
            assert "qty" in part
            assert "material" in part
            assert "dimensions_mm" in part

    def test_single_cell(self):
        cfg = StackCADConfig(num_cells=1)
        bom = generate_bom(cfg)
        assert bom["num_cells"] == 1
        anode_frame = next(p for p in bom["parts"] if p["part"] == "Anode Frame")
        assert anode_frame["qty"] == 1

    def test_extended_parts(self):
        cfg = StackCADConfig()
        bom = generate_bom(cfg)
        part_names = [p["part"] for p in bom["parts"]]
        assert "Barb Fitting" in part_names
        assert "Anolyte Reservoir" in part_names
        assert "Peristaltic Pump Head" in part_names
        assert "Silicone Tubing" in part_names
        assert "Support Foot (U-cradle)" in part_names

    def test_bom_title(self):
        cfg = StackCADConfig(num_cells=5)
        bom = generate_bom(cfg)
        assert "5-cell" in bom["title"]

    def test_stack_dimensions_in_bom(self):
        cfg = StackCADConfig()
        bom = generate_bom(cfg)
        assert bom["stack_length_mm"] > 0
        assert "outer_dimensions_mm" in bom


class TestWriteBomJson:
    def test_writes_file(self, tmp_path):
        cfg = StackCADConfig()
        result = write_bom_json(cfg, tmp_path)
        assert result.exists()
        assert result.name == "bom.json"
        data = json.loads(result.read_text())
        assert data["num_cells"] == 10


class TestMain:
    def test_main_function_exists(self):
        """Verify main is importable (actual execution needs full CQ stack)."""
        from cad.export import main
        assert callable(main)
