"""Tests for anode_frame.py component."""

from __future__ import annotations

import pytest

from cad.cad_config import StackCADConfig

cq = pytest.importorskip("cadquery", reason="CadQuery not installed")

from cad.components.anode_frame import build


class TestAnodeFrame:
    def test_build_returns_solid(self, default_config: StackCADConfig) -> None:
        result = build(default_config)
        assert result is not None
        solid = result.val()
        assert solid.isValid()

    def test_bounding_box_dimensions(self, default_config: StackCADConfig) -> None:
        result = build(default_config)
        bb = result.val().BoundingBox()
        outer_mm = default_config.outer_side * 1000
        depth_mm = default_config.semi_cell.depth * 1000
        assert bb.xlen == pytest.approx(outer_mm, rel=0.05)
        assert bb.ylen == pytest.approx(outer_mm, rel=0.05)
        assert bb.zlen == pytest.approx(depth_mm, rel=0.05)

    def test_volume_less_than_solid_block(
        self,
        default_config: StackCADConfig,
    ) -> None:
        result = build(default_config)
        vol = result.val().Volume()
        outer_mm = default_config.outer_side * 1000
        depth_mm = default_config.semi_cell.depth * 1000
        solid_vol = outer_mm * outer_mm * depth_mm
        # Must be less due to chamber pocket, holes, grooves
        assert vol < solid_vol

    def test_single_cell_config(self, single_cell_config: StackCADConfig) -> None:
        result = build(single_cell_config)
        assert result.val().isValid()
