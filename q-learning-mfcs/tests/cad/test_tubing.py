"""Tests for tubing.py — swept tube segments."""

from __future__ import annotations

import math

import pytest

from cad.cad_config import StackCADConfig, TubingSpec




class TestTubingSpec:
    def test_outer_diameter(self) -> None:
        spec = TubingSpec()
        assert spec.outer_diameter == pytest.approx(0.012)  # 8 + 2*2 mm

    def test_cross_section_area(self) -> None:
        spec = TubingSpec()
        expected = math.pi * (0.004) ** 2  # pi * r²
        assert spec.cross_section_area == pytest.approx(expected)

    def test_dead_volume(self) -> None:
        spec = TubingSpec()
        length = 1.0  # 1 metre
        expected = spec.cross_section_area * 1.0
        assert spec.dead_volume(length) == pytest.approx(expected)


class TestBuildStraight:
    @pytest.fixture(autouse=True)
    def _require_cadquery(self) -> None:
        pytest.importorskip("cadquery", reason="CadQuery not installed")

    def test_returns_solid_and_length(self, default_config: StackCADConfig) -> None:
        from cad.components.tubing import build_straight

        solid, length_m = build_straight(
            start=(0, 0, 0), end=(0, 0, 100), config=default_config,
        )
        assert solid is not None
        assert len(solid.solids().vals()) >= 1
        assert length_m == pytest.approx(0.1)  # 100 mm = 0.1 m

    def test_length_matches_distance(self, default_config: StackCADConfig) -> None:
        from cad.components.tubing import build_straight

        _, length_m = build_straight(
            start=(0, 0, 0), end=(30, 40, 0), config=default_config,
        )
        expected = math.sqrt(30**2 + 40**2) / 1000  # 50 mm = 0.05 m
        assert length_m == pytest.approx(expected)

    def test_tube_is_hollow(self, default_config: StackCADConfig) -> None:
        from cad.components.tubing import build_straight

        solid, _ = build_straight(
            start=(0, 0, 0), end=(0, 0, 100), config=default_config,
        )
        spec = default_config.tubing
        od = spec.outer_diameter * 1000
        solid_cylinder_vol = math.pi * (od / 2) ** 2 * 100
        assert solid.val().Volume() < solid_cylinder_vol


class TestBuildUtube:
    @pytest.fixture(autouse=True)
    def _require_cadquery(self) -> None:
        pytest.importorskip("cadquery", reason="CadQuery not installed")

    def test_returns_solid_and_length(self, default_config: StackCADConfig) -> None:
        from cad.components.tubing import build_utube

        solid, length_m = build_utube(
            port_a=(0, -65, 37.5),
            port_b=(0, -65, 89.5),
            clearance_mm=30,
            normal=(-1, 0, 0),
            config=default_config,
        )
        assert solid is not None
        assert length_m > 0

    def test_utube_length_geometry(self, default_config: StackCADConfig) -> None:
        """U-tube length ≈ 2 * clearance + traverse distance."""
        from cad.components.tubing import build_utube

        clearance = 30  # mm (> 2x OD for robust geometry)
        z1, z2 = 37.5, 89.5
        _, length_m = build_utube(
            port_a=(0, -65, z1),
            port_b=(0, -65, z2),
            clearance_mm=clearance,
            normal=(0, -1, 0),
            config=default_config,
        )
        expected_mm = 2 * clearance + abs(z2 - z1)
        assert length_m == pytest.approx(expected_mm / 1000, rel=0.15)
