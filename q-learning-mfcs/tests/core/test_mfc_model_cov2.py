"""Coverage boost tests for mfc_model.py - targeting remaining uncovered paths."""
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


class TestMFCModelConstants:
    def test_all_constants_and_execution(self):
        """Verify y0, t_span, t_eval, constant_i_fc, and solve_ivp call."""
        mock_mfc = MagicMock()
        mock_mfc.mfc_odes.return_value = [0.0] * 9
        mock_class = MagicMock(return_value=mock_mfc)
        mock_odes = MagicMock()
        mock_odes.MFCModel = mock_class

        mock_solution = MagicMock()
        mock_solution.t = np.linspace(0, 100, 500)
        mock_solution.y = np.random.rand(9, 500)

        with patch.dict(sys.modules, {"odes": mock_odes}):
            with patch("matplotlib.pyplot.style"):
                with patch("matplotlib.pyplot.subplots") as mock_sub:
                    with patch("matplotlib.pyplot.show"):
                        with patch(
                            "scipy.integrate.solve_ivp",
                            return_value=mock_solution,
                        ) as mock_ivp:
                            mock_sub.return_value = (MagicMock(), MagicMock())
                            if "mfc_model" in sys.modules:
                                del sys.modules["mfc_model"]
                            import mfc_model

                            assert len(mfc_model.y0) == 9
                            assert mfc_model.t_span == [0, 100]
                            assert mfc_model.constant_i_fc == 1.0
                            assert len(mfc_model.t_eval) == 500
                            mock_ivp.assert_called_once()
                            call_kwargs = mock_ivp.call_args
                            assert call_kwargs.kwargs.get("method") == "RK45"
