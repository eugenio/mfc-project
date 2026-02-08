"""Tests for end_plate.py, membrane_gasket.py, tie_rod.py,
current_collector.py, and electrode_placeholder.py components.
"""

from __future__ import annotations

import pytest

from cad.cad_config import StackCADConfig

cq = pytest.importorskip("cadquery", reason="CadQuery not installed")

from cad.components.current_collector import build as build_collector
from cad.components.electrode_placeholder import build as build_electrode
from cad.components.end_plate import build as build_end_plate
from cad.components.membrane_gasket import build as build_gasket
from cad.components.tie_rod import build_nut, build_rod, build_washer


class TestEndPlate:
    def test_inlet_plate_valid(self, default_config: StackCADConfig) -> None:
        result = build_end_plate(default_config, is_inlet=True)
        assert result.val().isValid()

    def test_outlet_plate_valid(self, default_config: StackCADConfig) -> None:
        result = build_end_plate(default_config, is_inlet=False)
        assert result.val().isValid()

    def test_thickness(self, default_config: StackCADConfig) -> None:
        result = build_end_plate(default_config)
        bb = result.val().BoundingBox()
        expected_mm = default_config.end_plate.thickness * 1000
        assert bb.zlen == pytest.approx(expected_mm, rel=0.05)


class TestMembraneGasket:
    def test_valid_solid(self, default_config: StackCADConfig) -> None:
        result = build_gasket(default_config)
        assert result.val().isValid()

    def test_thin(self, default_config: StackCADConfig) -> None:
        result = build_gasket(default_config)
        bb = result.val().BoundingBox()
        expected_mm = default_config.membrane.gasket_thickness * 1000
        assert bb.zlen == pytest.approx(expected_mm, rel=0.05)


class TestTieRod:
    def test_rod_valid(self, default_config: StackCADConfig) -> None:
        assert build_rod(default_config).val().isValid()

    def test_nut_valid(self, default_config: StackCADConfig) -> None:
        assert build_nut(default_config).val().isValid()

    def test_washer_valid(self, default_config: StackCADConfig) -> None:
        assert build_washer(default_config).val().isValid()

    def test_rod_length(self, default_config: StackCADConfig) -> None:
        rod = build_rod(default_config)
        bb = rod.val().BoundingBox()
        expected_mm = default_config.tie_rod_length * 1000
        assert bb.zlen == pytest.approx(expected_mm, rel=0.02)


class TestCurrentCollector:
    def test_valid(self, default_config: StackCADConfig) -> None:
        assert build_collector(default_config).val().isValid()


class TestElectrodePlaceholder:
    def test_valid(self, default_config: StackCADConfig) -> None:
        result = build_electrode(default_config)
        assert result.val().isValid()

    def test_dimensions(self, default_config: StackCADConfig) -> None:
        result = build_electrode(default_config)
        bb = result.val().BoundingBox()
        side_mm = default_config.electrode.side_length * 1000
        thick_mm = default_config.electrode.thickness * 1000
        assert bb.xlen == pytest.approx(side_mm, rel=0.01)
        assert bb.ylen == pytest.approx(side_mm, rel=0.01)
        assert bb.zlen == pytest.approx(thick_mm, rel=0.01)
