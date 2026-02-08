"""Tests for hydraulics.py — flow path computation and tubing lengths."""

from __future__ import annotations

import math

import pytest

from cad.cad_config import StackCADConfig
from cad.hydraulics import (
    FlowCircuit,
    PortPosition,
    TubingSegment,
    compute_parallel_flow_path,
    compute_port_positions,
    compute_series_flow_path,
    compute_total_circuit_lengths,
    compute_total_dead_volume,
)


# ---------------------------------------------------------------------------
# PortPosition
# ---------------------------------------------------------------------------
class TestPortPosition:
    def test_fields(self) -> None:
        p = PortPosition(
            x=10.0, y=20.0, z=30.0,
            normal=(0.0, -1.0, 0.0),
            label="anode_0_inlet",
        )
        assert p.x == 10.0
        assert p.y == 20.0
        assert p.z == 30.0
        assert p.normal == (0.0, -1.0, 0.0)
        assert p.label == "anode_0_inlet"

    def test_equality(self) -> None:
        a = PortPosition(x=0, y=0, z=0, normal=(1, 0, 0), label="a")
        b = PortPosition(x=0, y=0, z=0, normal=(1, 0, 0), label="a")
        assert a == b


# ---------------------------------------------------------------------------
# TubingSegment
# ---------------------------------------------------------------------------
class TestTubingSegment:
    def test_fields(self) -> None:
        start = PortPosition(x=0, y=0, z=0, normal=(0, -1, 0), label="start")
        end = PortPosition(x=0, y=0, z=50, normal=(0, 1, 0), label="end")
        seg = TubingSegment(
            start=start,
            end=end,
            via_points=[(0, -20, 0), (0, -20, 50)],
            length_m=0.09,
            label="utube_0",
        )
        assert seg.length_m == pytest.approx(0.09)
        assert len(seg.via_points) == 2


# ---------------------------------------------------------------------------
# FlowCircuit
# ---------------------------------------------------------------------------
class TestFlowCircuit:
    def test_total_length_matches_segments(self) -> None:
        s = PortPosition(x=0, y=0, z=0, normal=(0, -1, 0), label="s")
        e = PortPosition(x=0, y=0, z=1, normal=(0, 1, 0), label="e")
        segs = [
            TubingSegment(start=s, end=e, via_points=[], length_m=0.5, label="a"),
            TubingSegment(start=e, end=s, via_points=[], length_m=0.3, label="b"),
        ]
        fc = FlowCircuit(
            segments=segs,
            total_length_m=0.8,
            dead_volume_m3=1e-5,
            circuit_type="anode",
        )
        assert fc.total_length_m == pytest.approx(0.8)
        assert fc.dead_volume_m3 == pytest.approx(1e-5)
        assert fc.circuit_type == "anode"


# ---------------------------------------------------------------------------
# compute_port_positions
# ---------------------------------------------------------------------------
class TestComputePortPositions:
    def test_default_config_port_count(self, default_config: StackCADConfig) -> None:
        """10-cell stack: 2 ports per cell per circuit × 2 circuits = 40 + 2 end plate ports."""
        ports = compute_port_positions(default_config)
        # Each cell: anode_inlet + anode_outlet + cathode_inlet + cathode_outlet = 4
        # Plus 2 end plate flow ports
        # Total = 10 * 4 + 2 = 42
        assert len(ports) >= 40

    def test_single_cell_ports(self, single_cell_config: StackCADConfig) -> None:
        ports = compute_port_positions(single_cell_config)
        labels = [p.label for p in ports]
        # Should have anode and cathode ports for cell 0
        assert any("anode_0_inlet" in lbl for lbl in labels)
        assert any("anode_0_outlet" in lbl for lbl in labels)
        assert any("cathode_0_inlet" in lbl for lbl in labels)
        assert any("cathode_0_outlet" in lbl for lbl in labels)

    def test_anode_ports_on_x_walls(self, default_config: StackCADConfig) -> None:
        """Anode inlet on -X wall, outlet on +X wall."""
        ports = compute_port_positions(default_config)
        half_outer = default_config.outer_side / 2 * 1000  # mm
        for p in ports:
            if "anode" in p.label and "inlet" in p.label:
                assert p.x == pytest.approx(-half_outer, abs=1)
            elif "anode" in p.label and "outlet" in p.label:
                assert p.x == pytest.approx(half_outer, abs=1)

    def test_cathode_ports_on_y_walls(self, default_config: StackCADConfig) -> None:
        """Cathode inlet on -Y wall, outlet on +Y wall."""
        ports = compute_port_positions(default_config)
        half_outer = default_config.outer_side / 2 * 1000  # mm
        for p in ports:
            if "cathode" in p.label and "inlet" in p.label:
                assert p.y == pytest.approx(-half_outer, abs=1)
            elif "cathode" in p.label and "outlet" in p.label:
                assert p.y == pytest.approx(half_outer, abs=1)

    def test_port_z_positions_monotonic(self, default_config: StackCADConfig) -> None:
        """Port Z positions should increase with cell index."""
        ports = compute_port_positions(default_config)
        anode_inlets = sorted(
            [p for p in ports if "anode" in p.label and "inlet" in p.label],
            key=lambda p: p.z,
        )
        for i in range(len(anode_inlets) - 1):
            assert anode_inlets[i].z < anode_inlets[i + 1].z


# ---------------------------------------------------------------------------
# compute_series_flow_path
# ---------------------------------------------------------------------------
class TestComputeSeriesFlowPath:
    def test_anode_circuit_segments(self, default_config: StackCADConfig) -> None:
        """10-cell anode series: 9 U-tubes + 2 end connections = 11 segments."""
        fc = compute_series_flow_path(default_config, "anode")
        # 9 U-tube segments between cells + 2 end connection segments
        assert len(fc.segments) == default_config.num_cells - 1 + 2
        assert fc.circuit_type == "anode"

    def test_cathode_circuit_segments(self, default_config: StackCADConfig) -> None:
        fc = compute_series_flow_path(default_config, "cathode")
        assert len(fc.segments) == default_config.num_cells - 1 + 2
        assert fc.circuit_type == "cathode"

    def test_total_length_reasonable(self, default_config: StackCADConfig) -> None:
        """Anode series for 10 cells should be roughly 0.8-1.5 m."""
        fc = compute_series_flow_path(default_config, "anode")
        assert 0.5 < fc.total_length_m < 2.0

    def test_dead_volume_positive(self, default_config: StackCADConfig) -> None:
        fc = compute_series_flow_path(default_config, "anode")
        assert fc.dead_volume_m3 > 0

    def test_single_cell_no_utubes(self, single_cell_config: StackCADConfig) -> None:
        """Single cell: 0 U-tubes + 2 end connections = 2 segments."""
        fc = compute_series_flow_path(single_cell_config, "anode")
        assert len(fc.segments) == 2


# ---------------------------------------------------------------------------
# compute_parallel_flow_path
# ---------------------------------------------------------------------------
class TestComputeParallelFlowPath:
    def test_anode_branch_count(self, default_config: StackCADConfig) -> None:
        """Parallel: N branch segments + 2 header segments."""
        fc = compute_parallel_flow_path(default_config, "anode")
        # N branch segments from manifold to cells + 2 manifold/header segments
        assert len(fc.segments) == default_config.num_cells + 2

    def test_has_headers_and_branches(self, default_config: StackCADConfig) -> None:
        """Parallel config should have header and branch segments."""
        fc = compute_parallel_flow_path(default_config, "anode")
        labels = [s.label for s in fc.segments]
        assert any("header" in lbl for lbl in labels)
        assert any("branch" in lbl for lbl in labels)


# ---------------------------------------------------------------------------
# compute_total_circuit_lengths
# ---------------------------------------------------------------------------
class TestComputeTotalCircuitLengths:
    def test_returns_all_keys(self, default_config: StackCADConfig) -> None:
        result = compute_total_circuit_lengths(default_config)
        assert "anode" in result
        assert "cathode" in result
        assert "total" in result

    def test_total_is_sum(self, default_config: StackCADConfig) -> None:
        result = compute_total_circuit_lengths(default_config)
        assert result["total"] == pytest.approx(
            result["anode"] + result["cathode"],
        )

    def test_all_positive(self, default_config: StackCADConfig) -> None:
        result = compute_total_circuit_lengths(default_config)
        for v in result.values():
            assert v > 0


# ---------------------------------------------------------------------------
# compute_total_dead_volume
# ---------------------------------------------------------------------------
class TestComputeTotalDeadVolume:
    def test_positive(self, default_config: StackCADConfig) -> None:
        dv = compute_total_dead_volume(default_config)
        assert dv > 0

    def test_scales_with_cells(self) -> None:
        cfg5 = StackCADConfig(num_cells=5)
        cfg10 = StackCADConfig(num_cells=10)
        dv5 = compute_total_dead_volume(cfg5)
        dv10 = compute_total_dead_volume(cfg10)
        assert dv10 > dv5
