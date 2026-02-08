"""Tests for cad_config.py — parametric CAD configuration dataclasses."""

from __future__ import annotations

import pytest

from cad.cad_config import (
    ConicalBottomSpec,
    CurrentCollectorSpec,
    ElectrodeDimensions,
    ElectrovalveSpec,
    EndPlateSpec,
    GasDiffusionSpec,
    MembraneDimensions,
    ORingSpec,
    PumpSupportSpec,
    ReservoirFeetSpec,
    ReservoirLidSpec,
    ReservoirRole,
    ReservoirSpec,
    SemiCellDimensions,
    StackCADConfig,
    StirringMotorSpec,
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

    def test_collector_positions_union(self, default_config: StackCADConfig) -> None:
        """collector_positions is union of anode + cathode (6 total)."""
        positions = default_config.collector_positions
        n = default_config.current_collector.count_per_electrode
        assert len(positions) == 2 * n  # 3 anode + 3 cathode = 6
        expected_y = default_config.semi_cell.inner_side * 0.3
        for _x, y in positions:
            assert y == pytest.approx(expected_y)

    def test_anode_collector_positions_left_half(self, default_config: StackCADConfig) -> None:
        """Anode collectors on left side (negative x)."""
        positions = default_config.anode_collector_positions
        n = default_config.current_collector.count_per_electrode
        assert len(positions) == n
        for x, _y in positions:
            assert x < 0  # left half

    def test_cathode_collector_positions_right_half(self, default_config: StackCADConfig) -> None:
        """Cathode collectors on right side (positive x)."""
        positions = default_config.cathode_collector_positions
        n = default_config.current_collector.count_per_electrode
        assert len(positions) == n
        for x, _y in positions:
            assert x > 0  # right half

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


# ---------------------------------------------------------------------------
# New component specs
# ---------------------------------------------------------------------------
class TestConicalBottomSpec:
    def test_default_values(self) -> None:
        spec = ConicalBottomSpec()
        assert spec.cone_height == pytest.approx(0.030)
        assert spec.drain_fitting_diameter == pytest.approx(0.008)

    def test_frozen(self) -> None:
        spec = ConicalBottomSpec()
        with pytest.raises(AttributeError):
            spec.cone_height = 0.050  # type: ignore[misc]


class TestStirringMotorSpec:
    def test_default_motor_dimensions(self) -> None:
        spec = StirringMotorSpec()
        assert spec.motor_diameter == pytest.approx(0.045)
        assert spec.motor_height == pytest.approx(0.060)

    def test_impeller_params(self) -> None:
        spec = StirringMotorSpec()
        assert spec.impeller_diameter == pytest.approx(0.050)
        assert spec.impeller_turns == pytest.approx(2.0)


class TestGasDiffusionSpec:
    def test_default_values(self) -> None:
        spec = GasDiffusionSpec()
        assert spec.element_diameter == pytest.approx(0.020)
        assert spec.element_length == pytest.approx(0.060)
        assert spec.port_height_from_bottom == pytest.approx(0.050)


class TestReservoirLidSpec:
    def test_default_values(self) -> None:
        spec = ReservoirLidSpec()
        assert spec.thickness == pytest.approx(0.005)
        assert spec.feed_port_count == 2
        assert spec.motor_hole_diameter == pytest.approx(0.012)


class TestReservoirFeetSpec:
    def test_default_values(self) -> None:
        spec = ReservoirFeetSpec()
        assert spec.foot_count == 3
        assert spec.foot_height == pytest.approx(0.020)
        assert spec.foot_width == pytest.approx(0.0375)
        assert spec.foot_depth == pytest.approx(0.0225)


class TestElectrovalveSpec:
    def test_default_values(self) -> None:
        spec = ElectrovalveSpec()
        assert spec.body_width == pytest.approx(0.035)
        assert spec.body_height == pytest.approx(0.055)
        assert spec.port_diameter == pytest.approx(0.008)


class TestPumpSupportSpec:
    def test_default_values(self) -> None:
        spec = PumpSupportSpec()
        assert spec.platform_width == pytest.approx(0.120)
        assert spec.foot_count == 4


class TestReservoirRole:
    def test_enum_values(self) -> None:
        assert ReservoirRole.ANOLYTE.value == "anolyte"
        assert ReservoirRole.CATHOLYTE.value == "catholyte"
        assert ReservoirRole.NUTRIENT.value == "nutrient"
        assert ReservoirRole.BUFFER.value == "buffer"


class TestMultiReservoirConfig:
    def test_backward_compatible_reservoir_property(self) -> None:
        cfg = StackCADConfig()
        assert cfg.reservoir is cfg.anolyte_reservoir

    def test_four_reservoir_specs(self) -> None:
        cfg = StackCADConfig()
        assert cfg.anolyte_reservoir.volume_liters == pytest.approx(10.0)
        assert cfg.catholyte_reservoir.volume_liters == pytest.approx(10.0)
        assert cfg.nutrient_reservoir.volume_liters == pytest.approx(1.0)
        assert cfg.buffer_reservoir.volume_liters == pytest.approx(5.0)

    def test_reservoir_spec_for_role(self) -> None:
        cfg = StackCADConfig()
        for role in ReservoirRole:
            spec = cfg.reservoir_spec_for_role(role)
            assert isinstance(spec, ReservoirSpec)

    def test_new_component_specs_present(self) -> None:
        cfg = StackCADConfig()
        assert isinstance(cfg.conical_bottom, ConicalBottomSpec)
        assert isinstance(cfg.reservoir_lid, ReservoirLidSpec)
        assert isinstance(cfg.reservoir_feet, ReservoirFeetSpec)
        assert isinstance(cfg.stirring_motor, StirringMotorSpec)
        assert isinstance(cfg.gas_diffusion, GasDiffusionSpec)
        assert isinstance(cfg.electrovalve, ElectrovalveSpec)
        assert isinstance(cfg.pump_support, PumpSupportSpec)
