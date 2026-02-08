"""Tests for create_summary_plots.py - summary performance plots."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


@pytest.fixture
def _mock_save(tmp_path):
    with patch(
        "create_summary_plots.get_figure_path",
        side_effect=lambda f: str(tmp_path / f),
    ):
        yield


class TestCreateSummaryPlots:
    def test_runs_without_error(self, _mock_save):
        from create_summary_plots import create_summary_plots
        create_summary_plots()

    def test_creates_figure(self, _mock_save, tmp_path):
        from create_summary_plots import create_summary_plots
        create_summary_plots()
        files = list(tmp_path.glob("*.png"))
        assert len(files) >= 1


class TestCreateTechnicalSummary:
    def test_runs_without_error(self, _mock_save):
        from create_summary_plots import create_technical_summary
        create_technical_summary()


class TestMain:
    def test_main_runs(self, _mock_save):
        from create_summary_plots import main
        main()
