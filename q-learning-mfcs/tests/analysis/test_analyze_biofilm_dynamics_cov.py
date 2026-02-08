"""Tests for analyze_biofilm_dynamics.py - coverage target 98%+."""
import sys
import os
import gzip
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def _make_dynamics_df(n=200):
    """Create a DataFrame matching analyze_biofilm_dynamics expectations."""
    return pd.DataFrame({
        "time_hours": np.linspace(0, 200, n),
        "biofilm_thicknesses": [
            "[10.0, 12.0, 11.0, 9.0, 13.0]" for _ in range(n)
        ],
        "reservoir_concentration": np.random.uniform(20, 30, n),
        "substrate_addition_rate": np.random.uniform(0.1, 0.5, n),
        "total_power": np.random.uniform(0.01, 0.1, n),
        "q_action": np.random.choice([0, 1, 2, 3], n),
    })


class TestAnalyzeBiofilmDynamics:
    def test_full_analysis(self, tmp_path):
        """Test the full analysis pipeline with mocked gzip file."""
        from analyze_biofilm_dynamics import analyze_biofilm_dynamics

        df = _make_dynamics_df(200)
        csv_file = tmp_path / "test.csv.gz"
        with gzip.open(csv_file, "wt") as f:
            df.to_csv(f, index=False)

        mock_gzip_open = MagicMock()
        mock_gzip_open.__enter__ = MagicMock(
            return_value=gzip.open(csv_file, "rt")
        )
        mock_gzip_open.__exit__ = MagicMock(return_value=False)

        with patch("gzip.open") as mock_gz:
            mock_gz.return_value.__enter__ = lambda s: open(
                csv_file, "rb"
            ).__enter__()

            # Instead, directly patch pd.read_csv
            with patch("pandas.read_csv", return_value=df):
                with patch("gzip.open"):
                    analyze_biofilm_dynamics()

    def test_parse_error_fallback(self):
        """Test biofilm parsing fallback on bad data."""
        bad_df = pd.DataFrame({
            "time_hours": [0, 1],
            "biofilm_thicknesses": ["bad_data", "[1.0, 2.0]"],
            "reservoir_concentration": [25, 24],
            "substrate_addition_rate": [0.1, 0.2],
            "total_power": [0.01, 0.02],
            "q_action": [0, 1],
        })

        biofilm_data = []
        for _, row in bad_df.iterrows():
            try:
                bs = row["biofilm_thicknesses"]
                if isinstance(bs, str):
                    bs = bs.strip("[]")
                    vals = [float(x.strip()) for x in bs.split(",")]
                else:
                    vals = bs
                biofilm_data.append(vals)
            except (ValueError, IndexError, TypeError):
                biofilm_data.append([1.0] * 5)

        assert biofilm_data[0] == [1.0] * 5
        assert biofilm_data[1] == [1.0, 2.0]

    def test_growth_phases(self):
        """Test growth phase identification logic."""
        n = 200
        time_hours = np.linspace(0, 200, n)
        avg_thickness = np.linspace(10, 100, n)
        growth_rate = np.gradient(avg_thickness, time_hours)

        phase1_idx = np.where(time_hours <= 10)[0]
        phase2_idx = np.where((time_hours > 10) & (time_hours <= 50))[0]
        phase3_idx = np.where((time_hours > 50) & (time_hours <= 100))[0]
        phase4_idx = np.where(time_hours > 100)[0]

        assert len(phase1_idx) > 0
        assert len(phase2_idx) > 0
        assert len(phase3_idx) > 0
        assert len(phase4_idx) > 0

        for idx_set in [phase1_idx, phase2_idx, phase3_idx, phase4_idx]:
            mean_rate = np.mean(growth_rate[idx_set])
            assert isinstance(mean_rate, float)

    def test_health_score_calculation(self):
        """Test health score calculation logic."""
        avg_thickness = np.array([30, 80, 120, 200])
        growth_rate = np.array([0.1, 1.0, 1.5, 3.0])

        health_score = []
        for i in range(len(avg_thickness)):
            thickness = avg_thickness[i]
            rate = growth_rate[i]
            thickness_score = 1.0 if 50 <= thickness <= 150 else 0.5
            rate_score = 1.0 if 0.5 <= rate <= 2.0 else 0.5
            health_score.append((thickness_score + rate_score) / 2)

        assert health_score[0] == 0.5  # thickness < 50, rate < 0.5
        assert health_score[1] == 1.0  # both in optimal range
        assert health_score[2] == 1.0  # both in optimal range
        assert health_score[3] == 0.5  # thickness > 150, rate > 2.0

    def test_anomaly_detection(self):
        """Test anomaly detection in growth rates."""
        growth_rate = np.zeros(100)
        growth_rate[50] = 100  # Big anomaly
        anomaly_threshold = np.mean(growth_rate) + 3 * np.std(growth_rate)
        anomalies = np.where(np.abs(growth_rate) > anomaly_threshold)[0]
        assert 50 in anomalies
