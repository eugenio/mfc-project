"""Tests for cad_config.py — parametric CAD configuration dataclasses."""

from __future__ import annotations

import pytest

from cad.cad_config import (
    CurrentCollectorSpec,
    ElectrodeDimensions,
    EndPlateSpec,
    MembraneDimensions,
    ORingSpec,
    SemiCellDimensions,
    StackCADConfig,
    TieRodSpec,
)


# ---------------------------------------------------------------------------
# ElectrodeDimensions
# ---------------------------------------------------------------------------
class TestElectrodeDimensions:
    def test_default_values(self) -> None:
        e = ElectrodeDimensions()
        assert e.side_length == pytest.approx(0.10)
        assert e.thickness == pytest.approx(0.005)

    def test_area(self) -> None:
        e = ElectrodeDimensions(side_length=0.10)
        assert e.area == pytest.approx(0.01)  # 100 cm²

    def test_volume(self) -> None:
        e = ElectrodeDimensions(side_length=0.10, thickness=0.005)
        assert e.volume == pytest.approx(5.0e-5)


# ---------------------------------------------------------------------------
# SemiCellDimensions
# ---------------------------------------------------------------------------
class TestSemiCellDimensions:
    def test_outer_side(self) -> None:
        sc = SemiCellDimensions()
        assert sc.outer_side == pytest.approx(0.13)  # 10 + 2*1.5 cm

    def test_chamber_volume_250ml(self) -> None:
        sc = SemiCellDimensions()
        vol_ml = sc.chamber_volume * 1e6  # m³ -> mL
        assert vol_ml == pytest.approx(250.0)

    def test_custom_dimensions(self) -> None:
        sc = SemiCellDimensions(inner_side=0.05, depth=0.011, wall_thickness=0.01)
        assert sc.outer_side == pytest.approx(0.07)
        vol_ml = sc.chamber_volume * 1e6
        assert vol_ml == pytest.approx(27.5)


# ---------------------------------------------------------------------------
# ORingSpec
# ---------------------------------------------------------------------------
class TestORingSpec:
    def test_groove_depth_25pct_compression(self) -> None:
        o = ORingSpec()
        assert o.groove_depth == pytest.approx(0.00353 * 0.75)

    def test_groove_width(self) -> None:
        o = ORingSpec()
        assert o.groove_width == pytest.approx(0.00353 * 1.35)

    def test_compressed_height(self) -> None:
        o = ORingSpec()
        assert o.compressed_height == pytest.approx(o.groove_depth)


# ---------------------------------------------------------------------------
# TieRodSpec
# ---------------------------------------------------------------------------
class TestTieRodSpec:
    def test_inset(self) -> None:
        t = TieRodSpec()
        expected = t.washer_od / 2 + 0.003
        assert t.inset == pytest.approx(expected)

    def test_m10_diameter(self) -> None:
        t = TieRodSpec()
        assert t.diameter == pytest.approx(0.010)


# ---------------------------------------------------------------------------
# CurrentCollectorSpec
# ---------------------------------------------------------------------------
class TestCurrentCollectorSpec:
    def test_three_per_electrode(self) -> None:
        cc = CurrentCollectorSpec()
        assert cc.count_per_electrode == 3

    def test_rod_length(self) -> None:
        cc = CurrentCollectorSpec()
        assert cc.rod_length == pytest.approx(0.070)


# ---------------------------------------------------------------------------
# StackCADConfig — derived properties
# ---------------------------------------------------------------------------
class TestStackCADConfig:
    def test_defaults_create_valid_config(self, default_config: StackCADConfig) -> None:
        warnings = default_config.validate()
        assert warnings == []

    def test_cell_thickness(self, default_config: StackCADConfig) -> None:
        expected = 0.025 + 0.002 + 0.025  # 5.2 cm
        assert default_config.cell_thickness == pytest.approx(expected)

    def test_stack_length(self, default_config: StackCADConfig) -> None:
        expected = 2 * 0.025 + 10 * 0.052  # 57 cm
        assert default_config.stack_length == pytest.approx(expected)

    def test_outer_side(self, default_config: StackCADConfig) -> None:
        assert default_config.outer_side == pytest.approx(0.13)

    def test_four_tie_rod_positions(self, default_config: StackCADConfig) -> None:
        positions = default_config.tie_rod_positions
        assert len(positions) == 4
        for x, y in positions:
            assert abs(x) == pytest.approx(abs(positions[0][0]))
            assert abs(y) == pytest.approx(abs(positions[0][1]))

    def test_three_collector_positions(self, default_config: StackCADConfig) -> None:
        positions = default_config.collector_positions
        assert len(positions) == 3
        for _x, y in positions:
            assert y == pytest.approx(0.0)

    def test_tie_rod_length(self, default_config: StackCADConfig) -> None:
        rod = default_config.tie_rod
        extra = 2 * (rod.nut_height + rod.washer_thickness + 0.005)
        expected = default_config.stack_length + extra
        assert default_config.tie_rod_length == pytest.approx(expected)

    def test_total_volumes(self, default_config: StackCADConfig) -> None:
        assert default_config.total_anode_volume == pytest.approx(10 * 250e-6)
        assert default_config.total_cathode_volume == pytest.approx(10 * 250e-6)

    def test_active_membrane_area(self, default_config: StackCADConfig) -> None:
        assert default_config.active_membrane_area == pytest.approx(0.01)

    def test_single_cell_stack_length(
        self,
        single_cell_config: StackCADConfig,
    ) -> None:
        expected = 2 * 0.025 + 1 * 0.052
        assert single_cell_config.stack_length == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class TestStackCADConfigValidation:
    def test_zero_cells_warns(self) -> None:
        cfg = StackCADConfig(num_cells=0)
        assert any("num_cells" in w for w in cfg.validate())

    def test_mismatched_electrode_warns(self) -> None:
        cfg = StackCADConfig(
            electrode=ElectrodeDimensions(side_length=0.08),
        )
        assert any("inner_side" in w for w in cfg.validate())

    def test_electrode_too_thick_warns(self) -> None:
        cfg = StackCADConfig(
            electrode=ElectrodeDimensions(thickness=0.030),
        )
        assert any("Electrode thicker" in w for w in cfg.validate())

    def test_gasket_thinner_than_membrane_warns(self) -> None:
        cfg = StackCADConfig(
            membrane=MembraneDimensions(
                thickness=0.003,
                gasket_thickness=0.001,
                active_side=0.10,
            ),
        )
        assert any("Gasket thinner" in w for w in cfg.validate())
