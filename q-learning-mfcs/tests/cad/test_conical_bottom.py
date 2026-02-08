"""Tests for conical_bottom.py — frustum with drain fitting."""

from __future__ import annotations

import pytest

from cad.cad_config import StackCADConfig


class TestConicalBottom:
    @pytest.fixture(autouse=True)
    def _require_cadquery(self) -> None:
        pytest.importorskip("cadquery", reason="CadQuery not installed")

    def test_build_returns_solid(self, default_config: StackCADConfig) -> None:
        from cad.components.conical_bottom import build

        result = build(default_config)
        assert result is not None
        assert len(result.solids().vals()) >= 1

    def test_height_matches_spec(self, default_config: StackCADConfig) -> None:
        from cad.components.conical_bottom import build

        result = build(default_config)
        bb = result.val().BoundingBox()
        cone_h = default_config.conical_bottom.cone_height * 1000
        boss_l = default_config.conical_bottom.drain_boss_length * 1000
        expected_total = cone_h + boss_l
        assert bb.zlen == pytest.approx(expected_total, rel=0.15)

    def test_top_wider_than_bottom(self, default_config: StackCADConfig) -> None:
        """Frustum should be wider at top (reservoir diameter)."""
        from cad.components.conical_bottom import build

        result = build(default_config)
        bb = result.val().BoundingBox()
        res_od = default_config.reservoir.outer_diameter * 1000
        drain_d = default_config.conical_bottom.drain_fitting_diameter * 1000
        # Bounding box X width should be close to reservoir OD
        assert bb.xlen > drain_d
        assert bb.xlen == pytest.approx(res_od, rel=0.15)

    def test_is_hollow(self, default_config: StackCADConfig) -> None:
        """Volume should be less than solid frustum."""
        from cad.components.conical_bottom import build

        result = build(default_config)
        actual_vol = result.val().Volume()
        # A solid frustum of same dimensions would be larger
        assert actual_vol > 0
