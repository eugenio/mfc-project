"""Tests for export.py — STEP/STL/BOM export utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cad.cad_config import StackCADConfig
from cad.export import generate_bom, write_bom_json


class TestGenerateBOM:
    def test_default_config(self, default_config: StackCADConfig) -> None:
        bom = generate_bom(default_config)
        assert bom["num_cells"] == 10
        assert len(bom["parts"]) == 19  # 14 original + 5 extended

    def test_part_quantities(self, default_config: StackCADConfig) -> None:
        bom = generate_bom(default_config)
        parts = {p["part"]: p["qty"] for p in bom["parts"]}
        assert parts["Inlet End Plate"] == 1
        assert parts["Outlet End Plate"] == 1
        assert parts["Anode Frame"] == 10
        assert parts["Cathode Frame (liquid)"] == 10
        assert parts["Membrane Gasket"] == 10
        assert parts["Nafion 117 Membrane"] == 10
        assert parts["Face Seal O-Ring"] == 22
        assert parts["Rod Seal O-Ring"] == 60
        assert parts["Tie Rod M10"] == 4
        assert parts["Hex Nut M10"] == 8
        assert parts["Flat Washer M10"] == 8
        assert parts["Ti Current Rod"] == 60
        assert parts["Anode Electrode"] == 10
        assert parts["Cathode Electrode"] == 10

    def test_stack_length_mm(self, default_config: StackCADConfig) -> None:
        bom = generate_bom(default_config)
        assert bom["stack_length_mm"] == pytest.approx(570.0)

    def test_single_cell(self, single_cell_config: StackCADConfig) -> None:
        bom = generate_bom(single_cell_config)
        assert bom["num_cells"] == 1
        parts = {p["part"]: p["qty"] for p in bom["parts"]}
        assert parts["Anode Frame"] == 1

    def test_json_serialisable(self, default_config: StackCADConfig) -> None:
        bom = generate_bom(default_config)
        text = json.dumps(bom)
        assert len(text) > 100


class TestWriteBomJson:
    def test_writes_file(
        self,
        default_config: StackCADConfig,
        tmp_path: Path,
    ) -> None:
        out = write_bom_json(default_config, tmp_path)
        assert out.exists()
        assert out.stat().st_size > 0
        data = json.loads(out.read_text())
        assert data["num_cells"] == 10


# CadQuery-dependent export tests
class TestExportComponent:
    @pytest.fixture(autouse=True)
    def _require_cadquery(self) -> None:
        pytest.importorskip("cadquery", reason="CadQuery not installed")

    def test_export_step_stl(
        self,
        default_config: StackCADConfig,
        tmp_path: Path,
    ) -> None:
        from cad.components.electrode_placeholder import build
        from cad.export import export_component

        solid = build(default_config)
        paths = export_component(solid, "test_electrode", tmp_path)
        assert len(paths) == 2
        for p in paths:
            assert p.exists()
            assert p.stat().st_size > 0
