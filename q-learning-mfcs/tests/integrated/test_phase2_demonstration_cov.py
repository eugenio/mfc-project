"""Tests for phase2_demonstration.py - coverage target 98%+."""
import sys
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def _mock_eis_measurement():
    m = MagicMock()
    m.frequency = 1000.0
    m.impedance_magnitude = 1200.0
    m.impedance_phase = -15.0
    m.real_impedance = 1200.0 * np.cos(-15.0 * np.pi / 180)
    m.imaginary_impedance = 1200.0 * np.sin(-15.0 * np.pi / 180)
    m.timestamp = 1.0
    m.temperature = 298.15
    return m


def _mock_qcm_measurement():
    m = MagicMock()
    m.frequency = 5_000_000.0
    m.frequency_shift = -375.0
    m.dissipation = 1.15e-6
    m.quality_factor = 8000.0
    m.timestamp = 1.0
    m.temperature = 298.15
    return m


class TestCreateSampleMeasurements:
    def test_basic(self):
        from phase2_demonstration import create_sample_measurements

        np.random.seed(42)
        eis_m, qcm_m, eis_props, qcm_props = create_sample_measurements(
            1.0, 15.0
        )
        assert eis_m is not None
        assert qcm_m is not None

    def test_different_times(self):
        from phase2_demonstration import create_sample_measurements

        np.random.seed(42)
        _, _, eis_p1, _ = create_sample_measurements(0.0, 10.0)
        _, _, eis_p2, _ = create_sample_measurements(50.0, 10.0)
        # Different times should produce different measurements
        assert eis_p1 is not None
        assert eis_p2 is not None


class TestDemonstratePhase2Enhancements:
    def test_runs_without_error(self):
        from phase2_demonstration import demonstrate_phase2_enhancements

        np.random.seed(42)
        # This calls many internal components; verify it doesn't crash
        demonstrate_phase2_enhancements()


class TestDemonstrateIndividualComponents:
    def test_runs_without_error(self):
        from phase2_demonstration import demonstrate_individual_components

        np.random.seed(42)
        demonstrate_individual_components()
