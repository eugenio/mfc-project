"""Tests for cathode_frame.py and cathode_frame_gas.py components."""

from __future__ import annotations

import pytest

from cad.cad_config import StackCADConfig

cq = pytest.importorskip("cadquery", reason="CadQuery not installed")

from cad.components.cathode_frame import build as build_liquid
from cad.components.cathode_frame_gas import build as build_gas


class TestLiquidCathodeFrame:
    def test_build_returns_valid_solid(
        self,
        default_config: StackCADConfig,
    ) -> None:
        result = build_liquid(default_config)
        assert result.val().isValid()

    def test_bounding_box(self, default_config: StackCADConfig) -> None:
        result = build_liquid(default_config)
        bb = result.val().BoundingBox()
        outer_mm = default_config.outer_side * 1000
        assert bb.xlen == pytest.approx(outer_mm, rel=0.05)
        assert bb.ylen == pytest.approx(outer_mm, rel=0.05)


class TestGasCathodeFrame:
    def test_build_returns_valid_solid(
        self,
        default_config: StackCADConfig,
    ) -> None:
        result = build_gas(default_config)
        assert result.val().isValid()

    def test_deeper_than_standard(self, default_config: StackCADConfig) -> None:
        liquid = build_liquid(default_config)
        gas = build_gas(default_config)
        bb_liquid = liquid.val().BoundingBox()
        bb_gas = gas.val().BoundingBox()
        # Gas variant has extra headspace_depth
        assert bb_gas.zlen > bb_liquid.zlen
