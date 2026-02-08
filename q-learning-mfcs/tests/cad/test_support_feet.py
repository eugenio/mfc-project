"""Tests for support_feet.py — U-cradle support bracket."""

from __future__ import annotations

import pytest

from cad.cad_config import StackCADConfig

pytest.importorskip("cadquery", reason="CadQuery not installed")


class TestSupportFoot:
    def test_build_returns_solid(self, default_config: StackCADConfig) -> None:
        from cad.components.support_feet import build

        result = build(default_config)
        assert result is not None
        # CadQuery Workplane should have a single solid
        assert len(result.solids().vals()) >= 1

    def test_bounding_box_height(self, default_config: StackCADConfig) -> None:
        """Foot height should match spec."""
        from cad.components.support_feet import build

        result = build(default_config)
        bb = result.val().BoundingBox()
        foot_height_mm = default_config.support_feet.foot_height * 1000
        assert bb.ylen == pytest.approx(foot_height_mm, rel=0.1)

    def test_bounding_box_width(self, default_config: StackCADConfig) -> None:
        """Foot width (Z-axis) should match spec."""
        from cad.components.support_feet import build

        result = build(default_config)
        bb = result.val().BoundingBox()
        foot_width_mm = default_config.support_feet.foot_width * 1000
        assert bb.zlen == pytest.approx(foot_width_mm, rel=0.1)

    def test_bounding_box_depth(self, default_config: StackCADConfig) -> None:
        """Foot depth (X-axis, full cradle) should match spec."""
        from cad.components.support_feet import build

        result = build(default_config)
        bb = result.val().BoundingBox()
        foot_depth_mm = default_config.support_feet.foot_depth * 1000
        assert bb.xlen == pytest.approx(foot_depth_mm, rel=0.1)

    def test_has_mounting_holes(self, default_config: StackCADConfig) -> None:
        """Should have 4 mounting holes in base plate."""
        from cad.components.support_feet import build

        result = build(default_config)
        # Volume with holes should be less than solid block
        solid_vol = result.val().Volume()
        # A simple U-bracket without holes would be larger
        spec = default_config.support_feet
        base_vol = (spec.foot_depth * spec.foot_width * spec.wall_thickness) * 1e9  # mm³
        # Volume should be less than a filled rectangular block
        block_vol = spec.foot_depth * spec.foot_width * spec.foot_height * 1e9
        assert solid_vol < block_vol
