"""Tests for barb_fitting.py — hose barb fitting."""

from __future__ import annotations

import math

import pytest

from cad.cad_config import StackCADConfig


class TestBarbFitting:
    def test_build_returns_solid(self, default_config: StackCADConfig) -> None:
        from cad.components.barb_fitting import build

        result = build(default_config)
        assert result is not None
        assert len(result.solids().vals()) >= 1

    def test_bounding_box_length(self, default_config: StackCADConfig) -> None:
        """Total length should be thread + hex + barb."""
        from cad.components.barb_fitting import build

        result = build(default_config)
        bb = result.val().BoundingBox()
        spec = default_config.barb_fitting
        expected_length_mm = (
            spec.thread_length + spec.hex_height + spec.barb_length
        ) * 1000
        # Z-length should match total fitting length
        assert bb.zlen == pytest.approx(expected_length_mm, rel=0.1)

    def test_has_bore(self, default_config: StackCADConfig) -> None:
        """Fitting should be hollow (bored through)."""
        from cad.components.barb_fitting import build

        result = build(default_config)
        spec = default_config.barb_fitting
        # Solid volume should be less than a solid cylinder
        total_len_mm = (spec.thread_length + spec.hex_height + spec.barb_length) * 1000
        max_od_mm = spec.hex_af * 1000  # hex is widest
        max_vol = math.pi * (max_od_mm / 2) ** 2 * total_len_mm
        actual_vol = result.val().Volume()
        assert actual_vol < max_vol

    def test_bounding_box_width(self, default_config: StackCADConfig) -> None:
        """Width/height bounded by hex across-flats."""
        from cad.components.barb_fitting import build

        result = build(default_config)
        bb = result.val().BoundingBox()
        spec = default_config.barb_fitting
        # Hex circumradius = af / cos(30°)
        hex_od_mm = spec.hex_af * 1000 / math.cos(math.radians(30))
        assert bb.xlen <= hex_od_mm * 1.1
