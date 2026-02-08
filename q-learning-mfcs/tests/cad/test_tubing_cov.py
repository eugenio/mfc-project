"""Comprehensive coverage tests for cad.components.tubing module."""
import math
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Mock cadquery before import
sys.modules.setdefault("cadquery", MagicMock())

import pytest

from cad.cad_config import StackCADConfig
from cad.components.tubing import _dist, _mm, _solid_cylinder_between, build_straight, build_utube


class TestHelpers:
    def test_mm(self):
        assert _mm(0.1) == pytest.approx(100.0)

    def test_dist_same_point(self):
        assert _dist((0, 0, 0), (0, 0, 0)) == pytest.approx(0.0)

    def test_dist_unit(self):
        assert _dist((0, 0, 0), (1, 0, 0)) == pytest.approx(1.0)

    def test_dist_3d(self):
        assert _dist((0, 0, 0), (3, 4, 0)) == pytest.approx(5.0)

    def test_dist_diagonal(self):
        d = _dist((1, 2, 3), (4, 6, 3))
        assert d == pytest.approx(5.0)


class TestSolidCylinderBetween:
    def test_creates_solid(self):
        result = _solid_cylinder_between((0, 0, 0), (0, 0, 10), 2.0)
        assert result is not None


class TestBuildStraight:
    def test_returns_tuple(self):
        cfg = StackCADConfig()
        solid, length_m = build_straight((0, 0, 0), (0, 0, 100), cfg)
        assert solid is not None
        assert length_m == pytest.approx(0.1)

    def test_short_segment(self):
        cfg = StackCADConfig()
        # Short segment: less than 2x OD -> no bore subtraction
        od_mm = _mm(cfg.tubing.outer_diameter)
        solid, length_m = build_straight((0, 0, 0), (0, 0, od_mm), cfg)
        assert solid is not None
        assert length_m > 0

    def test_long_segment(self):
        cfg = StackCADConfig()
        od_mm = _mm(cfg.tubing.outer_diameter)
        solid, length_m = build_straight((0, 0, 0), (0, 0, od_mm * 5), cfg)
        assert solid is not None


class TestBuildUtube:
    def test_returns_tuple(self):
        cfg = StackCADConfig()
        solid, length_m = build_utube(
            port_a=(0, -65, 50),
            port_b=(0, -65, 100),
            clearance_mm=20.0,
            normal=(0, -1, 0),
            config=cfg,
        )
        assert solid is not None
        assert length_m > 0

    def test_length_calculation(self):
        cfg = StackCADConfig()
        solid, length_m = build_utube(
            port_a=(0, 0, 0),
            port_b=(0, 0, 100),
            clearance_mm=20.0,
            normal=(0, -1, 0),
            config=cfg,
        )
        # length = dist(port_a, exit_a) + dist(exit_a, exit_b) + dist(exit_b, port_b)
        # exit_a = (0, -20, 0), exit_b = (0, -20, 100)
        expected_mm = 20.0 + 100.0 + 20.0
        assert length_m == pytest.approx(expected_mm / 1000.0)
