"""Tests for pump_support.py — platform with mounting holes."""

from __future__ import annotations

import pytest

from cad.cad_config import StackCADConfig


class TestPumpSupport:
    @pytest.fixture(autouse=True)
    def _require_cadquery(self) -> None:
        pytest.importorskip("cadquery", reason="CadQuery not installed")

    def test_build_returns_solid(self, default_config: StackCADConfig) -> None:
        from cad.components.pump_support import build

        result = build(default_config)
        assert result is not None
        assert len(result.solids().vals()) >= 1

    def test_platform_dimensions(self, default_config: StackCADConfig) -> None:
        from cad.components.pump_support import build

        result = build(default_config)
        bb = result.val().BoundingBox()
        spec = default_config.pump_support
        assert bb.xlen == pytest.approx(spec.platform_width * 1000, rel=0.15)
        assert bb.ylen == pytest.approx(spec.platform_depth * 1000, rel=0.15)

    def test_has_mounting_holes(self, default_config: StackCADConfig) -> None:
        """Platform volume should be less than bounding box volume (holes drilled)."""
        from cad.components.pump_support import build

        result = build(default_config)
        bb = result.val().BoundingBox()
        bounding_vol = bb.xlen * bb.ylen * bb.zlen
        actual = result.val().Volume()
        assert actual < bounding_vol
