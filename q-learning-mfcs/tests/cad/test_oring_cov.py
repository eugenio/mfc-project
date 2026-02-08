"""Comprehensive coverage tests for cad.components.oring module."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

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


class TestGrooveCrossSection:
    def test_creation(self):
        g = GrooveCrossSection(width=0.005, depth=0.003, corner_radius=0.0002, volume_fill_ratio=0.75)
        assert g.width == 0.005
        assert g.depth == 0.003
        assert g.corner_radius == 0.0002
        assert g.volume_fill_ratio == 0.75


class TestComputeFaceSealGroove:
    def test_default_spec(self):
        spec = ORingSpec()
        groove = compute_face_seal_groove(spec)
        assert groove.width == pytest.approx(spec.groove_width)
        assert groove.depth == pytest.approx(spec.groove_depth)
        assert groove.corner_radius == pytest.approx(0.2e-3)

    def test_fill_ratio_range(self):
        spec = ORingSpec()
        groove = compute_face_seal_groove(spec)
        assert 0.5 < groove.volume_fill_ratio < 1.0

    def test_fill_ratio_calculation(self):
        spec = ORingSpec()
        groove = compute_face_seal_groove(spec)
        cs = spec.cross_section_diameter
        oring_area = math.pi * (cs / 2) ** 2
        groove_area = groove.width * groove.depth
        expected_fill = oring_area / groove_area
        assert groove.volume_fill_ratio == pytest.approx(expected_fill)

    def test_zero_groove_area(self):
        spec = ORingSpec(cross_section_diameter=0.0, compression_ratio=0.0)
        groove = compute_face_seal_groove(spec)
        assert groove.volume_fill_ratio == 0.0


class TestComputeRodSealGroove:
    def test_default_spec(self):
        spec = RodSealORingSpec()
        groove = compute_rod_seal_groove(spec)
        assert groove.width == pytest.approx(spec.groove_width)
        assert groove.depth == pytest.approx(spec.groove_depth)
        assert groove.corner_radius == pytest.approx(0.1e-3)

    def test_fill_ratio(self):
        spec = RodSealORingSpec()
        groove = compute_rod_seal_groove(spec)
        assert 0.5 < groove.volume_fill_ratio < 1.0

    def test_zero_groove_area(self):
        spec = RodSealORingSpec(cross_section_diameter=0.0, compression_ratio=0.0)
        groove = compute_rod_seal_groove(spec)
        assert groove.volume_fill_ratio == 0.0


class TestRectangularGroovePathLength:
    def test_basic(self):
        length = rectangular_groove_path_length(0.10, 0.005)
        expected = 4 * (0.10 + 2 * 0.005) + 2 * math.pi * 0.005 - 8 * 0.005
        assert length == pytest.approx(expected)

    def test_zero_offset(self):
        length = rectangular_groove_path_length(0.10, 0.0)
        assert length == pytest.approx(4 * 0.10)


class TestOringCountPerCell:
    def test_three_rods(self):
        counts = oring_count_per_cell(3)
        assert counts["face_seal"] == 2
        assert counts["rod_seal"] == 6

    def test_zero_rods(self):
        counts = oring_count_per_cell(0)
        assert counts["face_seal"] == 2
        assert counts["rod_seal"] == 0


class TestTotalOringCount:
    def test_default(self):
        counts = total_oring_count(10, 3)
        assert counts["face_seal"] == 10 * 2 + 2  # cells + end plates
        assert counts["rod_seal"] == 10 * 6

    def test_single_cell(self):
        counts = total_oring_count(1, 3)
        assert counts["face_seal"] == 4  # 1*2 + 2
        assert counts["rod_seal"] == 6

    def test_custom_end_plates(self):
        counts = total_oring_count(5, 3, num_end_plates=4)
        assert counts["face_seal"] == 5 * 2 + 4
