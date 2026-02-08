"""Tests for mfc_model.py - Mojo-powered MFC simulation script."""
import sys
import os
from unittest.mock import MagicMock, patch

import numpy as np

# Add source path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


class TestMFCModelScript:
    """Test the mfc_model script execution."""

    def test_script_runs_and_initial_conditions(self):
        """Test script runs end-to-end and initial conditions are correct."""
        mock_mfc = MagicMock()
        mock_mfc.mfc_odes.return_value = [0.0] * 9

        mock_mfc_class = MagicMock(return_value=mock_mfc)

        mock_solution = MagicMock()
        mock_solution.t = np.linspace(0, 100, 500)
        mock_solution.y = np.random.rand(9, 500)

        mock_odes_module = MagicMock()
        mock_odes_module.MFCModel = mock_mfc_class

        with patch.dict(sys.modules, {'odes': mock_odes_module}):
            with patch('matplotlib.pyplot.style'):
                with patch('matplotlib.pyplot.subplots') as mock_subplots:
                    with patch('matplotlib.pyplot.show') as mock_show:
                        with patch(
                            'scipy.integrate.solve_ivp',
                            return_value=mock_solution,
                        ) as mock_ivp:
                            mock_subplots.return_value = (MagicMock(), MagicMock())

                            if 'mfc_model' in sys.modules:
                                del sys.modules['mfc_model']

                            import mfc_model

                            mock_ivp.assert_called_once()
                            mock_show.assert_called_once()

                            # Verify initial conditions
                            assert len(mfc_model.y0) == 9
                            assert mfc_model.t_span == [0, 100]
                            assert mfc_model.constant_i_fc == 1.0
                            assert len(mfc_model.t_eval) == 500
