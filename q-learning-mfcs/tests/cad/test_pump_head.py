"""Tests for pump_head.py — peristaltic pump head block."""

from __future__ import annotations

import pytest

from cad.cad_config import StackCADConfig

pytest.importorskip("cadquery", reason="CadQuery not installed")


class TestPumpHead:
    def test_build_returns_solid(self, default_config: StackCADConfig) -> None:
        from cad.components.pump_head import build

        result = build(default_config)
        assert result is not None
        assert len(result.solids().vals()) >= 1

    def test_bounding_box_dimensions(self, default_config: StackCADConfig) -> None:
        """Body dimensions should match spec."""
        from cad.components.pump_head import build

        result = build(default_config)
        bb = result.val().BoundingBox()
        spec = default_config.pump_head
        assert bb.xlen == pytest.approx(spec.body_width * 1000, rel=0.1)
        assert bb.ylen == pytest.approx(spec.body_depth * 1000, rel=0.1)
        assert bb.zlen == pytest.approx(spec.body_height * 1000, rel=0.1)

    def test_has_port_holes(self, default_config: StackCADConfig) -> None:
        """Volume should be less than solid block (ports bored)."""
        from cad.components.pump_head import build

        result = build(default_config)
        spec = default_config.pump_head
        block_vol = (
            spec.body_width * spec.body_depth * spec.body_height * 1e9
        )
        actual_vol = result.val().Volume()
        assert actual_vol < block_vol

    def test_has_mounting_holes(self, default_config: StackCADConfig) -> None:
        """4 corner mounting holes should reduce volume further."""
        from cad.components.pump_head import build

        result = build(default_config)
        spec = default_config.pump_head
        # Volume with just ports would be smaller; with mounting holes even smaller
        block_vol = (
            spec.body_width * spec.body_depth * spec.body_height * 1e9
        )
        actual_vol = result.val().Volume()
        # Should be meaningfully less than the solid block
        assert actual_vol < block_vol * 0.99
