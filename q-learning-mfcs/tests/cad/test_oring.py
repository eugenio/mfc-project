"""Tests for oring.py — ISO 3601 groove geometry helpers."""

from __future__ import annotations

import math

import pytest

from cad.cad_config import ORingSpec, RodSealORingSpec
from cad.components.oring import (
    GrooveCrossSection,
    compute_face_seal_groove,
    compute_rod_seal_groove,
    oring_count_per_cell,
    rectangular_groove_path_length,
    total_oring_count,
)


class TestComputeFaceSealGroove:
    def test_default_spec(self) -> None:
        groove = compute_face_seal_groove(ORingSpec())
        assert groove.width == pytest.approx(0.00353 * 1.35)
        assert groove.depth == pytest.approx(0.00353 * 0.75)
        assert groove.corner_radius == pytest.approx(0.2e-3)

    def test_fill_ratio_in_range(self) -> None:
        groove = compute_face_seal_groove(ORingSpec())
        # ISO 3601 recommends 70-85 % fill
        assert 0.60 <= groove.volume_fill_ratio <= 0.90

    def test_custom_spec(self) -> None:
        spec = ORingSpec(cross_section_diameter=0.005, compression_ratio=0.20)
        groove = compute_face_seal_groove(spec)
        assert groove.depth == pytest.approx(0.005 * 0.80)


class TestComputeRodSealGroove:
    def test_default_rod_seal(self) -> None:
        groove = compute_rod_seal_groove(RodSealORingSpec())
        assert groove.width == pytest.approx(0.00178 * 1.35)
        assert groove.depth == pytest.approx(0.00178 * 0.75)

    def test_fill_ratio_in_range(self) -> None:
        groove = compute_rod_seal_groove(RodSealORingSpec())
        assert 0.60 <= groove.volume_fill_ratio <= 0.90


class TestRectangularGroovePathLength:
    def test_square_path(self) -> None:
        inner = 0.10
        offset = 0.005
        length = rectangular_groove_path_length(inner, offset)
        expected = 4 * inner + 2 * math.pi * offset
        assert length == pytest.approx(expected)

    def test_zero_offset(self) -> None:
        length = rectangular_groove_path_length(0.10, 0.0)
        assert length == pytest.approx(0.40)


class TestORingCounts:
    def test_per_cell(self) -> None:
        counts = oring_count_per_cell(num_collector_rods=3)
        assert counts["face_seal"] == 2
        assert counts["rod_seal"] == 6

    def test_total_10_cell_stack(self) -> None:
        counts = total_oring_count(num_cells=10, num_collector_rods=3)
        assert counts["face_seal"] == 22  # 10*2 + 2 end plates
        assert counts["rod_seal"] == 60  # 10*6
