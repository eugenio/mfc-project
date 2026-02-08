"""Comprehensive coverage tests for cad.cad_config module."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from cad.cad_config import (
    BarbFittingSpec,
    CathodeType,
    CurrentCollectorSpec,
    ElectrodeDimensions,
    EndPlateSpec,
    FlowConfiguration,
    GasCathodeDimensions,
    ManifoldSpec,
    MembraneDimensions,
    ORingSpec,
    PortLabelSpec,
    PumpHeadSpec,
    ReservoirSpec,
    RodSealORingSpec,
    SemiCellDimensions,
    StackCADConfig,
    SupportFeetSpec,
    TieRodSpec,
    TubingSpec,
)


class TestElectrodeDimensions:
    def test_defaults(self):
        e = ElectrodeDimensions()
        assert e.side_length == 0.10
        assert e.thickness == 0.005

    def test_area(self):
        e = ElectrodeDimensions(side_length=0.10)
        assert e.area == pytest.approx(0.01)

    def test_volume(self):
        e = ElectrodeDimensions(side_length=0.10, thickness=0.005)
        assert e.volume == pytest.approx(0.01 * 0.005)

    def test_custom_values(self):
        e = ElectrodeDimensions(side_length=0.05, thickness=0.003)
        assert e.area == pytest.approx(0.0025)
        assert e.volume == pytest.approx(0.0025 * 0.003)


class TestSemiCellDimensions:
    def test_defaults(self):
        sc = SemiCellDimensions()
        assert sc.inner_side == 0.10
        assert sc.depth == 0.025
        assert sc.wall_thickness == 0.015

    def test_outer_side(self):
        sc = SemiCellDimensions(inner_side=0.10, wall_thickness=0.015)
        assert sc.outer_side == pytest.approx(0.13)

    def test_chamber_volume(self):
        sc = SemiCellDimensions(inner_side=0.10, depth=0.025)
        assert sc.chamber_volume == pytest.approx(0.01 * 0.025)

    def test_flow_port_defaults(self):
        sc = SemiCellDimensions()
        assert sc.flow_port_diameter == 0.008
        assert sc.flow_port_offset == 0.015


class TestGasCathodeDimensions:
    def test_defaults(self):
        gc = GasCathodeDimensions()
        assert gc.headspace_depth == 0.015
        assert gc.gas_port_diameter == 0.008
        assert gc.gas_port_offset_top == 0.005


class TestMembraneDimensions:
    def test_defaults(self):
        m = MembraneDimensions()
        assert m.thickness == pytest.approx(1.78e-4)
        assert m.gasket_thickness == 0.002
        assert m.active_side == 0.10


class TestORingSpec:
    def test_defaults(self):
        o = ORingSpec()
        assert o.cross_section_diameter == pytest.approx(0.00353)
        assert o.material_shore_a == 70
        assert o.compression_ratio == 0.25

    def test_groove_depth(self):
        o = ORingSpec()
        expected = 0.00353 * (1.0 - 0.25)
        assert o.groove_depth == pytest.approx(expected)

    def test_groove_width(self):
        o = ORingSpec()
        expected = 0.00353 * 1.35
        assert o.groove_width == pytest.approx(expected)

    def test_compressed_height(self):
        o = ORingSpec()
        expected = 0.00353 * (1.0 - 0.25)
        assert o.compressed_height == pytest.approx(expected)


class TestRodSealORingSpec:
    def test_defaults(self):
        r = RodSealORingSpec()
        assert r.cross_section_diameter == pytest.approx(0.00178)
        assert r.compression_ratio == 0.25

    def test_groove_depth(self):
        r = RodSealORingSpec()
        expected = 0.00178 * (1.0 - 0.25)
        assert r.groove_depth == pytest.approx(expected)

    def test_groove_width(self):
        r = RodSealORingSpec()
        expected = 0.00178 * 1.35
        assert r.groove_width == pytest.approx(expected)


class TestTieRodSpec:
    def test_defaults(self):
        t = TieRodSpec()
        assert t.diameter == 0.010
        assert t.clearance_hole_diameter == 0.011
        assert t.material == "SS316"

    def test_inset(self):
        t = TieRodSpec()
        expected = t.washer_od / 2 + 0.003
        assert t.inset == pytest.approx(expected)


class TestCurrentCollectorSpec:
    def test_defaults(self):
        cc = CurrentCollectorSpec()
        assert cc.diameter == 0.006
        assert cc.count_per_electrode == 3
        assert cc.material == "Ti Grade 2"

    def test_rod_length(self):
        cc = CurrentCollectorSpec()
        assert cc.rod_length == 0.070

    def test_seal_oring(self):
        cc = CurrentCollectorSpec()
        assert isinstance(cc.seal_oring, RodSealORingSpec)


class TestEndPlateSpec:
    def test_defaults(self):
        ep = EndPlateSpec()
        assert ep.thickness == 0.025
        assert ep.nut_pocket_depth == 0.012
        assert ep.material == "Polypropylene"


class TestEnums:
    def test_cathode_type_values(self):
        assert CathodeType.LIQUID.value == "liquid"
        assert CathodeType.GAS.value == "gas"

    def test_flow_configuration_values(self):
        assert FlowConfiguration.SERIES.value == "series"
        assert FlowConfiguration.PARALLEL.value == "parallel"


class TestSupportFeetSpec:
    def test_defaults(self):
        sf = SupportFeetSpec()
        assert sf.foot_width == 0.030
        assert sf.foot_height == 0.040
        assert sf.mounting_hole_diameter == 0.006


class TestPortLabelSpec:
    def test_defaults(self):
        pl = PortLabelSpec()
        assert pl.font_size == 0.005
        assert pl.text_depth == 0.001
        assert pl.plate_thickness == 0.002


class TestBarbFittingSpec:
    def test_defaults(self):
        bf = BarbFittingSpec()
        assert bf.barb_od == 0.010
        assert bf.bore_diameter == 0.008
        assert bf.thread_length == 0.010


class TestTubingSpec:
    def test_defaults(self):
        t = TubingSpec()
        assert t.inner_diameter == 0.008
        assert t.wall_thickness == 0.002

    def test_outer_diameter(self):
        t = TubingSpec()
        assert t.outer_diameter == pytest.approx(0.012)

    def test_cross_section_area(self):
        t = TubingSpec()
        expected = math.pi * (0.008 / 2) ** 2
        assert t.cross_section_area == pytest.approx(expected)

    def test_dead_volume(self):
        t = TubingSpec()
        length = 1.0
        expected = t.cross_section_area * length
        assert t.dead_volume(length) == pytest.approx(expected)

    def test_dead_volume_zero(self):
        t = TubingSpec()
        assert t.dead_volume(0.0) == pytest.approx(0.0)


class TestManifoldSpec:
    def test_defaults(self):
        m = ManifoldSpec()
        assert m.header_od == 0.016
        assert m.branch_id == 0.008


class TestReservoirSpec:
    def test_defaults(self):
        r = ReservoirSpec()
        assert r.volume_liters == 10.0
        assert r.aspect_ratio == 2.0
        assert r.num_ports == 3

    def test_inner_diameter(self):
        r = ReservoirSpec()
        vol_m3 = 10.0 * 1e-3
        d_cubed = vol_m3 / (math.pi / 4 * 2.0)
        expected = d_cubed ** (1.0 / 3.0)
        assert r.inner_diameter == pytest.approx(expected)

    def test_inner_height(self):
        r = ReservoirSpec()
        assert r.inner_height == pytest.approx(r.aspect_ratio * r.inner_diameter)

    def test_outer_diameter(self):
        r = ReservoirSpec()
        assert r.outer_diameter == pytest.approx(r.inner_diameter + 2 * 0.003)

    def test_outer_height(self):
        r = ReservoirSpec()
        assert r.outer_height == pytest.approx(r.inner_height + 0.003)


class TestPumpHeadSpec:
    def test_defaults(self):
        p = PumpHeadSpec()
        assert p.body_width == 0.100
        assert p.max_flow_rate == pytest.approx(5.0e-4 / 3600)


class TestStackCADConfig:
    def test_defaults(self):
        cfg = StackCADConfig()
        assert cfg.num_cells == 10
        assert isinstance(cfg.electrode, ElectrodeDimensions)
        assert cfg.flow_config == FlowConfiguration.SERIES

    def test_gasket_membrane_thickness(self):
        cfg = StackCADConfig()
        assert cfg.gasket_membrane_thickness == cfg.membrane.gasket_thickness

    def test_cell_thickness(self):
        cfg = StackCADConfig()
        expected = 0.025 + 0.002 + 0.025
        assert cfg.cell_thickness == pytest.approx(expected)

    def test_stack_length(self):
        cfg = StackCADConfig()
        expected = 2 * 0.025 + 10 * cfg.cell_thickness
        assert cfg.stack_length == pytest.approx(expected)

    def test_outer_side(self):
        cfg = StackCADConfig()
        assert cfg.outer_side == cfg.semi_cell.outer_side

    def test_tie_rod_positions(self):
        cfg = StackCADConfig()
        pos = cfg.tie_rod_positions
        assert len(pos) == 4
        for x, y in pos:
            assert isinstance(x, float)
            assert isinstance(y, float)

    def test_collector_positions(self):
        cfg = StackCADConfig()
        pos = cfg.collector_positions
        assert len(pos) == 3
        for x, y in pos:
            assert isinstance(x, float)
            assert y == 0.0

    def test_tie_rod_length(self):
        cfg = StackCADConfig()
        extra = 2 * (cfg.tie_rod.nut_height + cfg.tie_rod.washer_thickness + 0.005)
        expected = cfg.stack_length + extra
        assert cfg.tie_rod_length == pytest.approx(expected)

    def test_total_anode_volume(self):
        cfg = StackCADConfig()
        expected = 10 * cfg.semi_cell.chamber_volume
        assert cfg.total_anode_volume == pytest.approx(expected)

    def test_total_cathode_volume(self):
        cfg = StackCADConfig()
        assert cfg.total_cathode_volume == pytest.approx(cfg.total_anode_volume)

    def test_active_membrane_area(self):
        cfg = StackCADConfig()
        assert cfg.active_membrane_area == pytest.approx(0.01)

    def test_validate_ok(self):
        cfg = StackCADConfig()
        warnings = cfg.validate()
        assert warnings == []

    def test_validate_num_cells_zero(self):
        cfg = StackCADConfig(num_cells=0)
        warnings = cfg.validate()
        assert any("num_cells" in w for w in warnings)

    def test_validate_mismatched_electrode(self):
        cfg = StackCADConfig(
            electrode=ElectrodeDimensions(side_length=0.05),
        )
        warnings = cfg.validate()
        assert any("electrode side_length" in w for w in warnings)

    def test_validate_mismatched_membrane(self):
        cfg = StackCADConfig(
            membrane=MembraneDimensions(active_side=0.05),
        )
        warnings = cfg.validate()
        assert any("Membrane active_side" in w for w in warnings)

    def test_validate_oring_groove_too_deep(self):
        cfg = StackCADConfig(
            face_oring=ORingSpec(cross_section_diameter=0.10),
        )
        warnings = cfg.validate()
        assert any("O-ring groove" in w for w in warnings)

    def test_validate_electrode_too_thick(self):
        cfg = StackCADConfig(
            electrode=ElectrodeDimensions(thickness=0.10),
        )
        warnings = cfg.validate()
        assert any("Electrode thicker" in w for w in warnings)

    def test_validate_gasket_thinner_than_membrane(self):
        cfg = StackCADConfig(
            membrane=MembraneDimensions(
                thickness=0.01,
                gasket_thickness=0.001,
                active_side=0.10,
            ),
        )
        warnings = cfg.validate()
        assert any("Gasket thinner" in w for w in warnings)

    def test_validate_washer_extends_beyond(self):
        cfg = StackCADConfig(
            tie_rod=TieRodSpec(washer_od=0.10),
        )
        warnings = cfg.validate()
        assert any("washer extends" in w for w in warnings)

    def test_extra_tubing_lengths(self):
        cfg = StackCADConfig()
        assert cfg.anode_tubing_extra_length == 0.10
        assert cfg.cathode_tubing_extra_length == 0.10
        assert cfg.reservoir_tubing_length == 0.50
        assert cfg.pump_tubing_length == 0.30

    def test_utube_and_manifold(self):
        cfg = StackCADConfig()
        assert cfg.utube_clearance == 0.020
        assert cfg.manifold_standoff == 0.040

    def test_single_cell(self):
        cfg = StackCADConfig(num_cells=1)
        assert cfg.stack_length == pytest.approx(2 * 0.025 + cfg.cell_thickness)
