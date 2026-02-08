"""Tests for stirring_motor.py — motor + shaft + impeller."""

from __future__ import annotations

import pytest

from cad.cad_config import StackCADConfig


class TestStirringMotor:
    @pytest.fixture(autouse=True)
    def _require_cadquery(self) -> None:
        pytest.importorskip("cadquery", reason="CadQuery not installed")

    def test_build_returns_solid(self, default_config: StackCADConfig) -> None:
        from cad.components.stirring_motor import build

        result = build(default_config)
        assert result is not None
        assert len(result.solids().vals()) >= 1

    def test_motor_height(self, default_config: StackCADConfig) -> None:
        from cad.components.stirring_motor import build

        result = build(default_config)
        bb = result.val().BoundingBox()
        motor_h = default_config.stirring_motor.motor_height * 1000
        shaft_l = default_config.stirring_motor.shaft_length * 1000
        expected = motor_h + shaft_l
        assert bb.zlen == pytest.approx(expected, rel=0.2)

    def test_impeller_wider_than_shaft(self, default_config: StackCADConfig) -> None:
        from cad.components.stirring_motor import build

        result = build(default_config)
        bb = result.val().BoundingBox()
        imp_d = default_config.stirring_motor.impeller_diameter * 1000
        shaft_d = default_config.stirring_motor.shaft_diameter * 1000
        assert bb.xlen > shaft_d
