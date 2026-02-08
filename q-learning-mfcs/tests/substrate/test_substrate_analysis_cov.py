"""Tests for substrate_analysis.py - coverage target 98%+."""
import sys
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def _make_df(n=2000):
    """Create realistic simulation DataFrame."""
    return pd.DataFrame({
        "time_hours": np.linspace(0, 100, n),
        "substrate_utilization": np.random.uniform(0.5, 0.9, n),
        "stack_power": np.random.uniform(0.3, 0.5, n),
        "stack_voltage": np.random.uniform(2.0, 3.0, n),
        "biofilm_cell_1": np.random.uniform(10, 50, n),
    })


class TestLoadAndAnalyzeData:
    def test_basic_analysis(self):
        from substrate_analysis import load_and_analyze_data

        unified = _make_df()
        non_unified = _make_df()
        with patch("pandas.read_csv", side_effect=[unified, non_unified]):
            load_and_analyze_data()

    def test_steady_state_reached(self):
        from substrate_analysis import load_and_analyze_data

        n = 2000
        unified = _make_df(n)
        non_unified = _make_df(n)
        unified["substrate_utilization"] = 0.8
        non_unified["substrate_utilization"] = 0.7
        with patch("pandas.read_csv", side_effect=[unified, non_unified]):
            load_and_analyze_data()

    def test_large_improvement(self):
        from substrate_analysis import load_and_analyze_data

        n = 2000
        unified = _make_df(n)
        non_unified = _make_df(n)
        unified["substrate_utilization"] = np.full(n, 0.9)
        non_unified["substrate_utilization"] = np.full(n, 0.5)
        with patch("pandas.read_csv", side_effect=[unified, non_unified]):
            load_and_analyze_data()

    def test_similar_correlation(self):
        from substrate_analysis import load_and_analyze_data

        n = 2000
        unified = _make_df(n)
        non_unified = _make_df(n)
        # Use same biofilm data for both to make correlations similar
        non_unified["biofilm_cell_1"] = unified["biofilm_cell_1"].copy()
        non_unified["substrate_utilization"] = unified["substrate_utilization"].copy()
        with patch("pandas.read_csv", side_effect=[unified, non_unified]):
            load_and_analyze_data()
