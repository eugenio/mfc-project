"""Tests for reservoir_feet.py — triangular support pattern."""

from __future__ import annotations

import pytest

from cad.cad_config import StackCADConfig


class TestReservoirFeet:
    @pytest.fixture(autouse=True)
    def _require_cadquery(self) -> None:
        pytest.importorskip("cadquery", reason="CadQuery not installed")

    def test_build_returns_solid(self, default_config: StackCADConfig) -> None:
        from cad.components.reservoir_feet import build

        result = build(default_config)
        assert result is not None
        assert len(result.solids().vals()) >= 1

    def test_extends_below_zero(self, default_config: StackCADConfig) -> None:
        """Feet should extend below Z=0 (top at Z=0)."""
        from cad.components.reservoir_feet import build

        result = build(default_config)
        bb = result.val().BoundingBox()
        assert bb.zmin < 0

    def test_has_volume(self, default_config: StackCADConfig) -> None:
        from cad.components.reservoir_feet import build

        result = build(default_config)
        assert result.val().Volume() > 0
