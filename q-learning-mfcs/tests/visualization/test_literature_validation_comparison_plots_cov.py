"""Tests for literature_validation_comparison_plots.py."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import json
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _make_original_df(n=200):
    """Synthetic original recirculation control data."""
    t = np.linspace(0, 1000, n)
    return pd.DataFrame({
        "time_hours": t,
        "reservoir_concentration": np.full(n, 20.0),
        "outlet_concentration": np.full(n, 18.0),
        "total_power": np.random.uniform(0.05, 0.10, n),
        "cell_1_biofilm": np.linspace(1.0, 1.08, n),
        "cell_2_biofilm": np.linspace(1.0, 1.08, n),
        "cell_3_biofilm": np.linspace(1.0, 1.08, n),
        "cell_4_biofilm": np.linspace(1.0, 1.08, n),
        "cell_5_biofilm": np.linspace(1.0, 1.08, n),
    })


def _make_literature_100h_df(n=200):
    """Synthetic literature-validated 100h data."""
    t = np.linspace(0, 100, n)
    return pd.DataFrame({
        "time_hours": t,
        "reservoir_concentration": np.full(n, 20.0),
        "outlet_concentration": np.full(n, 16.25),
        "total_power": np.random.uniform(0.15, 0.25, n),
        "cell_1_biofilm": np.linspace(1.0, 2.5, n),
        "cell_2_biofilm": np.linspace(1.0, 2.5, n),
        "cell_3_biofilm": np.linspace(1.0, 2.5, n),
        "cell_4_biofilm": np.linspace(1.0, 2.5, n),
        "cell_5_biofilm": np.linspace(1.0, 2.5, n),
    })


def _make_literature_1000h_df(n=200):
    """Synthetic literature-validated 1000h data."""
    t = np.linspace(0, 1000, n)
    return pd.DataFrame({
        "time_hours": t,
        "reservoir_concentration": np.full(n, 20.0),
        "outlet_concentration": np.full(n, 16.0),
        "total_power": np.random.uniform(0.15, 0.25, n),
        "cell_1_biofilm": np.linspace(1.0, 3.0, n),
        "cell_2_biofilm": np.linspace(1.0, 3.0, n),
        "cell_3_biofilm": np.linspace(1.0, 3.0, n),
        "cell_4_biofilm": np.linspace(1.0, 3.0, n),
        "cell_5_biofilm": np.linspace(1.0, 3.0, n),
    })


@pytest.fixture
def data_dir(tmp_path):
    """Write three CSV files and one JSON metadata file."""
    orig = _make_original_df()
    lit100 = _make_literature_100h_df()
    lit1000 = _make_literature_1000h_df()

    orig_csv = tmp_path / "mfc_recirculation_control_20250724_040215.csv"
    lit100_csv = (
        tmp_path
        / "mfc_recirculation_control_literature_validated_100h_044346.csv"
    )
    lit1000_csv = (
        tmp_path
        / "mfc_recirculation_control_literature_validated_1000h_044433.csv"
    )

    orig.to_csv(orig_csv, index=False)
    lit100.to_csv(lit100_csv, index=False)
    lit1000.to_csv(lit1000_csv, index=False)

    # Write a JSON metadata file for the lit100h CSV
    meta = {"parameters": {"growth_rate": 0.05}}
    lit100_json = str(lit100_csv).replace(".csv", ".json")
    with open(lit100_json, "w") as f:
        json.dump(meta, f)

    return tmp_path


class TestLoadSimulationData:
    def test_loads_csv_and_metadata(self, data_dir):
        with patch(
            "literature_validation_comparison_plots.get_simulation_data_path",
            side_effect=lambda f: str(data_dir / f),
        ), patch(
            "literature_validation_comparison_plots.glob.glob",
            side_effect=lambda pattern: sorted(
                str(p)
                for p in data_dir.glob(os.path.basename(pattern))
            ),
        ):
            from literature_validation_comparison_plots import (
                load_simulation_data,
            )
            df, meta = load_simulation_data(
                "mfc_recirculation_control_literature_validated_*_044346.csv"
            )

        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert "time_hours" in df.columns
        assert isinstance(meta, dict)
        assert "parameters" in meta

    def test_returns_none_when_no_files(self, tmp_path):
        with patch(
            "literature_validation_comparison_plots.get_simulation_data_path",
            side_effect=lambda f: str(tmp_path / f),
        ), patch(
            "literature_validation_comparison_plots.glob.glob",
            return_value=[],
        ):
            from literature_validation_comparison_plots import (
                load_simulation_data,
            )
            df, meta = load_simulation_data("nonexistent_*.csv")

        assert df is None
        assert meta is None

    def test_handles_missing_json(self, data_dir):
        with patch(
            "literature_validation_comparison_plots.get_simulation_data_path",
            side_effect=lambda f: str(data_dir / f),
        ), patch(
            "literature_validation_comparison_plots.glob.glob",
            side_effect=lambda pattern: sorted(
                str(p)
                for p in data_dir.glob(os.path.basename(pattern))
            ),
        ):
            from literature_validation_comparison_plots import (
                load_simulation_data,
            )
            # The 1000h CSV has no corresponding JSON
            df, meta = load_simulation_data(
                "mfc_recirculation_control_literature_validated_*_044433.csv"
            )

        assert df is not None
        assert meta == {}


class TestCreateLiteratureComparisonPlots:
    def test_returns_filename(self, data_dir):
        def mock_glob(pattern):
            base = os.path.basename(pattern)
            return sorted(str(p) for p in data_dir.glob(base))

        with patch(
            "literature_validation_comparison_plots.get_simulation_data_path",
            side_effect=lambda f: str(data_dir / f),
        ), patch(
            "literature_validation_comparison_plots.glob.glob",
            side_effect=mock_glob,
        ), patch(
            "literature_validation_comparison_plots.get_figure_path",
            side_effect=lambda f: str(data_dir / os.path.basename(f)),
        ), patch.object(plt.Figure, "savefig"):
            from literature_validation_comparison_plots import (
                create_literature_comparison_plots,
            )
            result = create_literature_comparison_plots()

        assert result is not None
        assert isinstance(result, str)
        assert "literature_validation_comparison" in result

    def test_returns_none_when_missing_data(self, tmp_path):
        with patch(
            "literature_validation_comparison_plots.get_simulation_data_path",
            side_effect=lambda f: str(tmp_path / f),
        ), patch(
            "literature_validation_comparison_plots.glob.glob",
            return_value=[],
        ):
            from literature_validation_comparison_plots import (
                create_literature_comparison_plots,
            )
            result = create_literature_comparison_plots()

        assert result is None
