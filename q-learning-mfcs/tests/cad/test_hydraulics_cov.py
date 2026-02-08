"""Comprehensive coverage tests for cad.hydraulics module."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from cad.cad_config import FlowConfiguration, StackCADConfig
from cad.hydraulics import (
    FlowCircuit,
    PortPosition,
    TubingSegment,
    _m,
    _mm,
    _utube_length_mm,
    compute_parallel_flow_path,
    compute_port_positions,
    compute_series_flow_path,
    compute_total_circuit_lengths,
    compute_total_dead_volume,
)


class TestHelpers:
    def test_mm_conversion(self):
        assert _mm(1.0) == pytest.approx(1000.0)
        assert _mm(0.1) == pytest.approx(100.0)

    def test_m_conversion(self):
        assert _m(1000.0) == pytest.approx(1.0)
        assert _m(100.0) == pytest.approx(0.1)

    def test_roundtrip(self):
        assert _m(_mm(0.05)) == pytest.approx(0.05)


class TestDataStructures:
    def test_port_position(self):
        p = PortPosition(x=1.0, y=2.0, z=3.0, normal=(0, 1, 0), label="test")
        assert p.x == 1.0
        assert p.label == "test"

    def test_tubing_segment(self):
        start = PortPosition(x=0, y=0, z=0, normal=(0, 0, 1), label="s")
        end = PortPosition(x=0, y=0, z=10, normal=(0, 0, 1), label="e")
        seg = TubingSegment(
            start=start, end=end, via_points=[], length_m=0.01, label="t"
        )
        assert seg.length_m == 0.01

    def test_flow_circuit(self):
        fc = FlowCircuit(
            segments=[], total_length_m=1.5, dead_volume_m3=0.001, circuit_type="anode"
        )
        assert fc.total_length_m == 1.5
        assert fc.circuit_type == "anode"


class TestUtubeLength:
    def test_basic(self):
        result = _utube_length_mm(20.0, 50.0)
        assert result == pytest.approx(2 * 20.0 + 50.0)

    def test_zero_traverse(self):
        result = _utube_length_mm(10.0, 0.0)
        assert result == pytest.approx(20.0)


class TestComputePortPositions:
    def test_default_config(self):
        cfg = StackCADConfig()
        ports = compute_port_positions(cfg)
        # 10 cells * 4 ports + 2 end plate ports = 42
        assert len(ports) == 42

    def test_single_cell(self):
        cfg = StackCADConfig(num_cells=1)
        ports = compute_port_positions(cfg)
        # 1 cell * 4 ports + 2 end plate ports = 6
        assert len(ports) == 6

    def test_port_labels(self):
        cfg = StackCADConfig(num_cells=2)
        ports = compute_port_positions(cfg)
        labels = [p.label for p in ports]
        assert "anode_0_inlet" in labels
        assert "anode_0_outlet" in labels
        assert "cathode_0_inlet" in labels
        assert "cathode_0_outlet" in labels
        assert "anode_1_inlet" in labels
        assert "end_plate_inlet" in labels
        assert "end_plate_outlet" in labels

    def test_port_normals(self):
        cfg = StackCADConfig(num_cells=1)
        ports = compute_port_positions(cfg)
        anode_in = next(p for p in ports if p.label == "anode_0_inlet")
        assert anode_in.normal == (0.0, -1.0, 0.0)
        anode_out = next(p for p in ports if p.label == "anode_0_outlet")
        assert anode_out.normal == (0.0, 1.0, 0.0)
        cathode_in = next(p for p in ports if p.label == "cathode_0_inlet")
        assert cathode_in.normal == (-1.0, 0.0, 0.0)

    def test_end_plate_ports(self):
        cfg = StackCADConfig(num_cells=1)
        ports = compute_port_positions(cfg)
        ep_in = next(p for p in ports if p.label == "end_plate_inlet")
        assert ep_in.z == 0.0
        assert ep_in.normal == (0.0, 0.0, -1.0)
        ep_out = next(p for p in ports if p.label == "end_plate_outlet")
        assert ep_out.normal == (0.0, 0.0, 1.0)

    def test_z_positions_increase(self):
        cfg = StackCADConfig(num_cells=3)
        ports = compute_port_positions(cfg)
        anode_inlets = sorted(
            [p for p in ports if "anode" in p.label and "inlet" in p.label],
            key=lambda p: p.z,
        )
        for i in range(len(anode_inlets) - 1):
            assert anode_inlets[i + 1].z > anode_inlets[i].z


class TestSeriesFlowPath:
    def test_anode_circuit(self):
        cfg = StackCADConfig(num_cells=3)
        fc = compute_series_flow_path(cfg, "anode")
        assert fc.circuit_type == "anode"
        assert fc.total_length_m > 0
        assert fc.dead_volume_m3 > 0

    def test_cathode_circuit(self):
        cfg = StackCADConfig(num_cells=3)
        fc = compute_series_flow_path(cfg, "cathode")
        assert fc.circuit_type == "cathode"

    def test_segment_count(self):
        cfg = StackCADConfig(num_cells=3)
        fc = compute_series_flow_path(cfg, "anode")
        # end_inlet + (n-1) u-tubes + end_outlet = 1 + 2 + 1 = 4
        assert len(fc.segments) == 4

    def test_single_cell_no_utubes(self):
        cfg = StackCADConfig(num_cells=1)
        fc = compute_series_flow_path(cfg, "anode")
        # end_inlet + end_outlet only = 2
        assert len(fc.segments) == 2

    def test_via_points_on_utubes(self):
        cfg = StackCADConfig(num_cells=3)
        fc = compute_series_flow_path(cfg, "anode")
        utube_segments = [
            s for s in fc.segments if "utube" in s.label
        ]
        for seg in utube_segments:
            assert len(seg.via_points) == 2

    def test_segment_labels(self):
        cfg = StackCADConfig(num_cells=2)
        fc = compute_series_flow_path(cfg, "anode")
        labels = [s.label for s in fc.segments]
        assert "anode_end_inlet" in labels
        assert "anode_utube_0" in labels
        assert "anode_end_outlet" in labels


class TestParallelFlowPath:
    def test_anode_circuit(self):
        cfg = StackCADConfig(num_cells=3)
        fc = compute_parallel_flow_path(cfg, "anode")
        assert fc.circuit_type == "anode"
        assert fc.total_length_m > 0

    def test_cathode_circuit(self):
        cfg = StackCADConfig(num_cells=3)
        fc = compute_parallel_flow_path(cfg, "cathode")
        assert fc.circuit_type == "cathode"

    def test_segment_count(self):
        cfg = StackCADConfig(num_cells=3)
        fc = compute_parallel_flow_path(cfg, "anode")
        # supply header + 3 branches + return header = 5
        assert len(fc.segments) == 5

    def test_header_labels(self):
        cfg = StackCADConfig(num_cells=2)
        fc = compute_parallel_flow_path(cfg, "anode")
        labels = [s.label for s in fc.segments]
        assert "anode_supply_header" in labels
        assert "anode_return_header" in labels
        assert "anode_branch_0" in labels
        assert "anode_branch_1" in labels

    def test_dead_volume(self):
        cfg = StackCADConfig(num_cells=3)
        fc = compute_parallel_flow_path(cfg, "anode")
        assert fc.dead_volume_m3 > 0


class TestTotalCircuitLengths:
    def test_series(self):
        cfg = StackCADConfig(flow_config=FlowConfiguration.SERIES)
        lengths = compute_total_circuit_lengths(cfg)
        assert "anode" in lengths
        assert "cathode" in lengths
        assert "total" in lengths
        assert lengths["total"] == pytest.approx(
            lengths["anode"] + lengths["cathode"]
        )

    def test_parallel(self):
        cfg = StackCADConfig(flow_config=FlowConfiguration.PARALLEL)
        lengths = compute_total_circuit_lengths(cfg)
        assert lengths["total"] > 0


class TestTotalDeadVolume:
    def test_default(self):
        cfg = StackCADConfig()
        dv = compute_total_dead_volume(cfg)
        assert dv > 0

    def test_includes_extras(self):
        cfg = StackCADConfig()
        dv = compute_total_dead_volume(cfg)
        # Should be larger than just circuit tubing
        lengths = compute_total_circuit_lengths(cfg)
        circuit_dv = cfg.tubing.dead_volume(lengths["total"])
        assert dv > circuit_dv
