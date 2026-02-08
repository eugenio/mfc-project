"""Tests for parameter_mapping.py — CAD <-> simulator bridge."""

from __future__ import annotations

import math

import pytest

from cad.cad_config import (
    ElectrodeDimensions,
    MembraneDimensions,
    SemiCellDimensions,
    StackCADConfig,
)
from cad.parameter_mapping import (
    cad_to_simulator,
    round_trip_check,
    simulator_to_cad,
)


class TestCADToSimulator:
    def test_default_config_volumes(self, default_config: StackCADConfig) -> None:
        params = cad_to_simulator(default_config)
        # 250 mL = 250e-6 m³ per semi-cell
        assert params["V_a"] == pytest.approx(250e-6)
        assert params["V_c"] == pytest.approx(250e-6)

    def test_membrane_area(self, default_config: StackCADConfig) -> None:
        params = cad_to_simulator(default_config)
        assert params["A_m"] == pytest.approx(0.01)  # 100 cm²

    def test_membrane_thickness(self, default_config: StackCADConfig) -> None:
        params = cad_to_simulator(default_config)
        assert params["d_m"] == pytest.approx(1.78e-4)

    def test_cell_thickness(self, default_config: StackCADConfig) -> None:
        params = cad_to_simulator(default_config)
        assert params["d_cell"] == pytest.approx(0.052)

    def test_num_cells(self, default_config: StackCADConfig) -> None:
        params = cad_to_simulator(default_config)
        assert params["n_cells"] == 10.0

    def test_small_config_matches_sim_defaults(
        self,
        small_config: StackCADConfig,
    ) -> None:
        params = cad_to_simulator(small_config)
        # 5 cm × 5 cm × 1.1 cm = 27.5 mL
        assert params["V_a"] == pytest.approx(27.5e-6)
        assert params["A_m"] == pytest.approx(25e-4)  # 25 cm²


class TestSimulatorToCAD:
    def test_from_sim_defaults(self) -> None:
        params = {
            "V_a": 5.5e-5,
            "A_m": 5.0e-4,
            "d_m": 1.778e-4,
            "n_cells": 5.0,
        }
        cfg = simulator_to_cad(params)
        assert cfg.num_cells == 5
        # side = sqrt(5e-4) ≈ 0.02236 m
        assert cfg.semi_cell.inner_side == pytest.approx(math.sqrt(5e-4))
        # volume back: side² × depth == V_a
        assert cfg.semi_cell.chamber_volume == pytest.approx(5.5e-5)

    def test_override_num_cells(self) -> None:
        params = {"V_a": 5.5e-5, "A_m": 5.0e-4, "d_m": 1.778e-4}
        cfg = simulator_to_cad(params, num_cells=3)
        assert cfg.num_cells == 3

    def test_produces_valid_config(self) -> None:
        params = {"V_a": 250e-6, "A_m": 0.01, "d_m": 1.78e-4}
        cfg = simulator_to_cad(params, num_cells=10)
        warnings = cfg.validate()
        assert warnings == []


class TestRoundTrip:
    def test_default_config_round_trip(
        self,
        default_config: StackCADConfig,
    ) -> None:
        result = round_trip_check(default_config)
        for key, (original, reconstructed) in result.items():
            assert original == pytest.approx(reconstructed), (
                f"Round-trip mismatch for {key}"
            )

    def test_small_config_round_trip(
        self,
        small_config: StackCADConfig,
    ) -> None:
        result = round_trip_check(small_config)
        for key, (original, reconstructed) in result.items():
            assert original == pytest.approx(reconstructed), (
                f"Round-trip mismatch for {key}"
            )
