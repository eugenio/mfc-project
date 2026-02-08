"""Tests for reservoir.py — cylindrical reservoir vessel with roles."""

from __future__ import annotations

import math

import pytest

from cad.cad_config import ReservoirRole, ReservoirSpec, StackCADConfig


class TestReservoirSpec:
    def test_inner_diameter_10l(self) -> None:
        """10 L reservoir with aspect 2.0 should have d ~ 186 mm."""
        spec = ReservoirSpec()
        d_mm = spec.inner_diameter * 1000
        assert d_mm == pytest.approx(186, abs=5)

    def test_inner_height_10l(self) -> None:
        """Height = 2 x diameter ~ 371 mm."""
        spec = ReservoirSpec()
        h_mm = spec.inner_height * 1000
        assert h_mm == pytest.approx(371, abs=10)

    def test_volume_consistency(self) -> None:
        """Computed dimensions should reproduce the target volume."""
        spec = ReservoirSpec()
        vol = math.pi / 4 * spec.inner_diameter**2 * spec.inner_height
        assert vol * 1e3 == pytest.approx(spec.volume_liters, rel=0.01)

    def test_outer_dimensions(self) -> None:
        spec = ReservoirSpec()
        assert spec.outer_diameter > spec.inner_diameter
        assert spec.outer_height > spec.inner_height

    def test_nutrient_reservoir_1l(self) -> None:
        cfg = StackCADConfig()
        spec = cfg.nutrient_reservoir
        assert spec.volume_liters == pytest.approx(1.0)
        vol = math.pi / 4 * spec.inner_diameter**2 * spec.inner_height
        assert vol * 1e3 == pytest.approx(1.0, rel=0.01)

    def test_buffer_reservoir_5l(self) -> None:
        cfg = StackCADConfig()
        spec = cfg.buffer_reservoir
        assert spec.volume_liters == pytest.approx(5.0)


class TestReservoirComponent:
    @pytest.fixture(autouse=True)
    def _require_cadquery(self) -> None:
        pytest.importorskip("cadquery", reason="CadQuery not installed")

    def test_build_returns_solid(self, default_config: StackCADConfig) -> None:
        from cad.components.reservoir import build

        result = build(default_config)
        assert result is not None
        assert len(result.solids().vals()) >= 1

    def test_bounding_box_height(self, default_config: StackCADConfig) -> None:
        from cad.components.reservoir import build

        result = build(default_config)
        bb = result.val().BoundingBox()
        oh_mm = default_config.reservoir.outer_height * 1000
        assert bb.zlen == pytest.approx(oh_mm, rel=0.15)

    def test_hollow_interior(self, default_config: StackCADConfig) -> None:
        """Reservoir should be hollow."""
        from cad.components.reservoir import build

        result = build(default_config)
        spec = default_config.reservoir
        solid_vol = (
            math.pi / 4 * (spec.outer_diameter * 1000) ** 2
            * spec.outer_height * 1000
        )
        actual_vol = result.val().Volume()
        assert actual_vol < solid_vol * 0.8

    def test_build_all_roles(self, default_config: StackCADConfig) -> None:
        """All reservoir roles should build successfully."""
        from cad.components.reservoir import build

        for role in ReservoirRole:
            result = build(default_config, role=role)
            assert result is not None
            assert len(result.solids().vals()) >= 1

    def test_catholyte_has_gas_port(self, default_config: StackCADConfig) -> None:
        """Catholyte reservoir should have additional gas port volume."""
        from cad.components.reservoir import build

        anolyte = build(default_config, role=ReservoirRole.ANOLYTE)
        catholyte = build(default_config, role=ReservoirRole.CATHOLYTE)
        # Catholyte should have slightly different volume due to gas port
        a_vol = anolyte.val().Volume()
        c_vol = catholyte.val().Volume()
        assert a_vol != pytest.approx(c_vol, rel=0.001)
