"""Tests for reservoir_lid.py — air-tight lid with ports."""

from __future__ import annotations

import pytest

from cad.cad_config import StackCADConfig


class TestReservoirLid:
    @pytest.fixture(autouse=True)
    def _require_cadquery(self) -> None:
        pytest.importorskip("cadquery", reason="CadQuery not installed")

    def test_build_returns_solid(self, default_config: StackCADConfig) -> None:
        from cad.components.reservoir_lid import build

        result = build(default_config)
        assert result is not None
        assert len(result.solids().vals()) >= 1

    def test_diameter_matches_reservoir(self, default_config: StackCADConfig) -> None:
        from cad.components.reservoir_lid import build

        result = build(default_config)
        bb = result.val().BoundingBox()
        od_mm = default_config.reservoir.outer_diameter * 1000
        assert bb.xlen == pytest.approx(od_mm, rel=0.15)

    def test_has_motor_hole(self, default_config: StackCADConfig) -> None:
        """Lid should be hollow in centre (motor shaft hole)."""
        import math

        from cad.components.reservoir_lid import build

        result = build(default_config)
        od = default_config.reservoir.outer_diameter * 1000
        thick = default_config.reservoir_lid.thickness * 1000
        solid_vol = math.pi * (od / 2) ** 2 * thick
        actual = result.val().Volume()
        assert actual < solid_vol
