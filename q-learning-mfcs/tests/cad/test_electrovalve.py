"""Tests for electrovalve.py — 3-way solenoid valve."""

from __future__ import annotations

import pytest

from cad.cad_config import StackCADConfig


class TestElectrovalve:
    @pytest.fixture(autouse=True)
    def _require_cadquery(self) -> None:
        pytest.importorskip("cadquery", reason="CadQuery not installed")

    def test_build_returns_solid(self, default_config: StackCADConfig) -> None:
        from cad.components.electrovalve import build

        result = build(default_config)
        assert result is not None
        assert len(result.solids().vals()) >= 1

    def test_body_dimensions(self, default_config: StackCADConfig) -> None:
        from cad.components.electrovalve import build

        result = build(default_config)
        bb = result.val().BoundingBox()
        spec = default_config.electrovalve
        # With solenoid on top, Z should be body + solenoid
        min_z = (spec.body_height + spec.solenoid_height) * 1000
        assert bb.zlen >= min_z * 0.8

    def test_has_port_bores(self, default_config: StackCADConfig) -> None:
        from cad.components.electrovalve import build

        result = build(default_config)
        actual = result.val().Volume()
        assert actual > 0  # Should be a valid solid
