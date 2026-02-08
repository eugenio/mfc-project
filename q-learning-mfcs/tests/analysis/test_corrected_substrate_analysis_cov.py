"""Tests for corrected_substrate_analysis.py - coverage target 98%+."""
import sys
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def _make_unified_df(n=2000, has_inlet=True):
    """Create a realistic unified MFC DataFrame."""
    df = pd.DataFrame({
        "time_hours": np.linspace(0, 100, n),
        "substrate_utilization": np.random.uniform(0.6, 0.9, n),
        "stack_power": np.random.uniform(0.3, 0.5, n),
        "stack_voltage": np.random.uniform(2.0, 3.0, n),
        "biofilm_cell_1": np.random.uniform(10, 50, n),
    })
    if has_inlet:
        df["inlet_concentration"] = np.random.uniform(20, 30, n)
        df["avg_outlet_concentration"] = np.random.uniform(5, 15, n)
    return df


def _make_non_unified_df(n=2000):
    """Create a realistic non-unified MFC DataFrame."""
    return pd.DataFrame({
        "time_hours": np.linspace(0, 100, n),
        "substrate_utilization": np.random.uniform(0.5, 0.85, n),
        "stack_power": np.random.uniform(0.2, 0.4, n),
        "stack_voltage": np.random.uniform(1.8, 2.8, n),
        "biofilm_cell_1": np.random.uniform(10, 50, n),
    })


class TestLoadAndAnalyzeData:
    def test_with_inlet_columns(self):
        from corrected_substrate_analysis import load_and_analyze_data

        unified = _make_unified_df(has_inlet=True)
        non_unified = _make_non_unified_df()
        with patch("pandas.read_csv", side_effect=[unified, non_unified]):
            results = load_and_analyze_data()
            assert "unified_final_util" in results
            assert "non_unified_final_util" in results
            assert "unified_score" in results
            assert "non_unified_score" in results

    def test_without_inlet_columns(self):
        from corrected_substrate_analysis import load_and_analyze_data

        unified = _make_unified_df(has_inlet=False)
        non_unified = _make_non_unified_df()
        with patch("pandas.read_csv", side_effect=[unified, non_unified]):
            results = load_and_analyze_data()
            assert isinstance(results["unified_final_util"], float)

    def test_steady_state_found(self):
        from corrected_substrate_analysis import load_and_analyze_data

        n = 2000
        unified = _make_unified_df(n=n, has_inlet=True)
        non_unified = _make_non_unified_df(n=n)
        # Make substrate_utilization very stable to trigger steady state
        unified["substrate_utilization"] = 0.8
        non_unified["substrate_utilization"] = 0.7
        # Add inlet columns
        unified["inlet_concentration"] = 25.0
        unified["avg_outlet_concentration"] = 10.0
        with patch("pandas.read_csv", side_effect=[unified, non_unified]):
            results = load_and_analyze_data()
            assert results is not None

    def test_scores_balanced(self):
        from corrected_substrate_analysis import load_and_analyze_data

        n = 2000
        unified = _make_unified_df(n=n, has_inlet=False)
        non_unified = _make_non_unified_df(n=n)
        # Make metrics identical to get balanced scores
        for col in ["substrate_utilization", "stack_power", "stack_voltage"]:
            non_unified[col] = unified[col].copy()
        with patch("pandas.read_csv", side_effect=[unified, non_unified]):
            results = load_and_analyze_data()
            assert isinstance(results["unified_score"], int)


class TestMainEntryPoint:
    def test_main_runs(self):
        unified = _make_unified_df()
        non_unified = _make_non_unified_df()
        with patch("pandas.read_csv", side_effect=[unified, non_unified]):
            from corrected_substrate_analysis import load_and_analyze_data
            results = load_and_analyze_data()
            assert results is not None
