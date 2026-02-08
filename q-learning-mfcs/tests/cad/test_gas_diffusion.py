"""Tests for gas_diffusion.py — cylindrical diffuser element."""

from __future__ import annotations

import pytest

from cad.cad_config import StackCADConfig


class TestGasDiffusion:
    @pytest.fixture(autouse=True)
    def _require_cadquery(self) -> None:
        pytest.importorskip("cadquery", reason="CadQuery not installed")

    def test_build_returns_solid(self, default_config: StackCADConfig) -> None:
        from cad.components.gas_diffusion import build

        result = build(default_config)
        assert result is not None
        assert len(result.solids().vals()) >= 1

    def test_total_length(self, default_config: StackCADConfig) -> None:
        from cad.components.gas_diffusion import build

        result = build(default_config)
        bb = result.val().BoundingBox()
        spec = default_config.gas_diffusion
        expected = (spec.element_length + spec.boss_length) * 1000
        assert bb.xlen == pytest.approx(expected, rel=0.15)

    def test_diameter(self, default_config: StackCADConfig) -> None:
        from cad.components.gas_diffusion import build

        result = build(default_config)
        bb = result.val().BoundingBox()
        elem_d = default_config.gas_diffusion.element_diameter * 1000
        assert bb.ylen == pytest.approx(elem_d, rel=0.2)
