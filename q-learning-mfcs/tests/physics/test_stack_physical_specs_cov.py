"""Tests for stack_physical_specs.py - coverage target 98%+."""
import sys
import os
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


@pytest.fixture(autouse=True)
def mock_path_config():
    with patch("stack_physical_specs.get_figure_path", return_value="/tmp/fig.png"):
        yield


class TestCalculateStackDimensions:
    def test_returns_dict(self):
        from stack_physical_specs import calculate_stack_dimensions
        result = calculate_stack_dimensions()
        assert isinstance(result, dict)
        assert "membrane_area" in result
        assert "stack_length" in result
        assert "stack_volume" in result
        assert "power_density_area" in result

    def test_values_positive(self):
        from stack_physical_specs import calculate_stack_dimensions
        result = calculate_stack_dimensions()
        for key, val in result.items():
            assert val > 0, f"{key} should be positive"


class TestEstimateStackMass:
    def test_basic(self):
        from stack_physical_specs import estimate_stack_mass
        mass = estimate_stack_mass(1e-4)
        assert mass > 0

    def test_zero_volume(self):
        from stack_physical_specs import estimate_stack_mass
        mass = estimate_stack_mass(0)
        assert mass == 0

    def test_large_volume(self):
        from stack_physical_specs import estimate_stack_mass
        mass = estimate_stack_mass(1.0)
        assert mass > 0


class TestCreateStackDiagram:
    def test_diagram_saved(self):
        with patch("stack_physical_specs.plt") as mock_plt:
            mock_ax1 = MagicMock()
            mock_ax2 = MagicMock()
            mock_fig = MagicMock()
            mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
            mock_plt.Rectangle.return_value = MagicMock()
            from stack_physical_specs import create_stack_diagram
            create_stack_diagram()
            mock_plt.savefig.assert_called_once()
            mock_plt.close.assert_called_once()


class TestCreateSpecificationsSheet:
    def test_specs_saved(self):
        with patch("stack_physical_specs.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)
            from stack_physical_specs import create_specifications_sheet
            create_specifications_sheet()
            mock_plt.savefig.assert_called_once()
            mock_plt.close.assert_called_once()


class TestMain:
    def test_main_runs(self):
        with patch("stack_physical_specs.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_plt.subplots.side_effect = [
                (mock_fig, (MagicMock(), MagicMock())),
                (mock_fig, mock_ax),
            ]
            mock_plt.Rectangle.return_value = MagicMock()
            from stack_physical_specs import main
            main()
