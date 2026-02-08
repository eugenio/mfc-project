"""Tests for energy_sustainability_analysis.py - coverage target 98%+."""
import sys
import os
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


@pytest.fixture(autouse=True)
def mock_path_config():
    with patch("energy_sustainability_analysis.get_figure_path", return_value="/tmp/fig.png"):
        yield


class TestAnalyzeEnergySustainability:
    def test_returns_results_and_scenarios(self):
        from energy_sustainability_analysis import analyze_energy_sustainability
        results, scenarios = analyze_energy_sustainability()
        assert isinstance(results, dict)
        assert isinstance(scenarios, dict)
        assert len(results) == 4
        assert "Ultra-low power mode" in scenarios

    def test_all_controllers_have_metrics(self):
        from energy_sustainability_analysis import analyze_energy_sustainability
        results, _ = analyze_energy_sustainability()
        for name, data in results.items():
            assert "controller_power" in data
            assert "total_consumption" in data
            assert "sustainability_margin" in data
            assert "efficiency" in data
            assert "energy_surplus_24h" in data
            assert "is_sustainable" in data

    def test_sustainability_margins(self):
        from energy_sustainability_analysis import analyze_energy_sustainability
        results, _ = analyze_energy_sustainability()
        # At least one controller should be in the results
        assert len(results) > 0
        # Check margin is numeric for all
        for name, data in results.items():
            assert isinstance(data["sustainability_margin"], float)


class TestCreateSustainabilityVisualization:
    def test_visualization_saved(self):
        with patch("energy_sustainability_analysis.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_plt.subplots.return_value = (mock_fig, ((MagicMock(), MagicMock()), (MagicMock(), MagicMock())))
            from energy_sustainability_analysis import (
                analyze_energy_sustainability,
                create_sustainability_visualization,
            )
            results, scenarios = analyze_energy_sustainability()
            create_sustainability_visualization(results, scenarios)
            mock_plt.savefig.assert_called_once()
            mock_plt.close.assert_called_once()


class TestCreateSustainabilitySummary:
    def test_summary_saved(self):
        with patch("energy_sustainability_analysis.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)
            mock_plt.Rectangle.return_value = MagicMock()
            from energy_sustainability_analysis import create_sustainability_summary
            create_sustainability_summary()
            mock_plt.savefig.assert_called_once()
            mock_plt.close.assert_called_once()


class TestMain:
    def test_main(self):
        with patch("energy_sustainability_analysis.plt") as mock_plt:
            mock_ax = MagicMock()
            mock_fig = MagicMock()
            mock_plt.subplots.side_effect = [
                (mock_fig, ((MagicMock(), MagicMock()), (MagicMock(), MagicMock()))),
                (mock_fig, mock_ax),
            ]
            mock_plt.Rectangle.return_value = MagicMock()
            from energy_sustainability_analysis import main
            main()
