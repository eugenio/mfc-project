"""Tests for eis_qcm_biofilm_correlation.py - coverage target 98%+."""
import sys
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


@pytest.fixture(autouse=True)
def mock_path_config():
    with patch("eis_qcm_biofilm_correlation.get_figure_path", return_value="/tmp/f.png"):
        with patch("eis_qcm_biofilm_correlation.get_simulation_data_path", return_value="/tmp/d.csv"):
            yield


class TestEISParameters:
    def test_defaults(self):
        from eis_qcm_biofilm_correlation import EISParameters
        p = EISParameters()
        assert p.freq_min == 100.0
        assert p.freq_max == 1e6
        assert p.n_frequencies == 50

    def test_custom(self):
        from eis_qcm_biofilm_correlation import EISParameters
        p = EISParameters(freq_min=50.0, n_frequencies=20)
        assert p.freq_min == 50.0
        assert p.n_frequencies == 20


class TestEISBiofilmModel:
    def test_randles_no_biofilm(self):
        from eis_qcm_biofilm_correlation import EISParameters, EISBiofilmModel
        p = EISParameters()
        m = EISBiofilmModel(p)
        Z = m.randles_circuit_impedance(1000.0, 10.0, 100.0, 1e-5)
        assert isinstance(Z, complex)

    def test_randles_with_biofilm(self):
        from eis_qcm_biofilm_correlation import EISParameters, EISBiofilmModel
        p = EISParameters()
        m = EISBiofilmModel(p)
        Z = m.randles_circuit_impedance(1000.0, 10.0, 100.0, 1e-5, 50.0, 1e-7)
        assert isinstance(Z, complex)

    def test_thickness_to_impedance(self):
        from eis_qcm_biofilm_correlation import EISParameters, EISBiofilmModel
        p = EISParameters()
        m = EISBiofilmModel(p)
        Z = m.biofilm_thickness_to_impedance(10.0, 1000.0)
        assert isinstance(Z, complex)

    def test_thickness_to_impedance_temp(self):
        from eis_qcm_biofilm_correlation import EISParameters, EISBiofilmModel
        p = EISParameters()
        m = EISBiofilmModel(p)
        Z = m.biofilm_thickness_to_impedance(10.0, 1000.0, temperature_c=35.0)
        assert isinstance(Z, complex)

    def test_impedance_to_thickness_no_cal(self):
        from eis_qcm_biofilm_correlation import EISParameters, EISBiofilmModel
        p = EISParameters()
        m = EISBiofilmModel(p)
        data = {
            "frequencies": np.array([100, 500, 1000, 5000, 10000]),
            "impedances": np.array([120, 110, 105, 100, 95]),
        }
        t = m.impedance_to_biofilm_thickness(data)
        assert isinstance(t, float)
        assert 0 <= t <= 100

    def test_impedance_to_thickness_with_cal(self):
        from eis_qcm_biofilm_correlation import EISParameters, EISBiofilmModel
        p = EISParameters()
        m = EISBiofilmModel(p)
        data = {
            "frequencies": np.array([100, 1000, 10000]),
            "impedances": np.array([120, 105, 95]),
        }
        cal = {
            "impedances": [90, 100, 110, 120],
            "thicknesses": [40, 30, 20, 10],
        }
        t = m.impedance_to_biofilm_thickness(data, cal)
        assert isinstance(t, float)


class TestCombinedEISQCMSensor:
    def test_init(self):
        from eis_qcm_biofilm_correlation import EISParameters, CombinedEISQCMSensor
        s = CombinedEISQCMSensor(EISParameters())
        assert "eis_qcm_linear" in s.calibration_coefficients

    def test_simulate_eis(self):
        from eis_qcm_biofilm_correlation import EISParameters, CombinedEISQCMSensor
        np.random.seed(42)
        s = CombinedEISQCMSensor(EISParameters())
        result = s.simulate_eis_measurement(15.0)
        assert "frequencies" in result
        assert "Z_1kHz" in result

    def test_simulate_eis_no_noise(self):
        from eis_qcm_biofilm_correlation import EISParameters, CombinedEISQCMSensor
        s = CombinedEISQCMSensor(EISParameters())
        result = s.simulate_eis_measurement(15.0, add_noise=False)
        assert len(result["impedances"]) == 50

    def test_correlate_with_qcm_few_points(self):
        from eis_qcm_biofilm_correlation import EISParameters, CombinedEISQCMSensor
        np.random.seed(42)
        s = CombinedEISQCMSensor(EISParameters())
        eis_data = s.simulate_eis_measurement(15.0)
        result = s.correlate_with_qcm(eis_data, 14.5)
        assert "correlation_r2" in result
        assert result["correlation_r2"] == 0.5

    def test_correlate_with_qcm_many_points(self):
        from eis_qcm_biofilm_correlation import EISParameters, CombinedEISQCMSensor
        np.random.seed(42)
        s = CombinedEISQCMSensor(EISParameters())
        for i in range(6):
            s.measurement_history["eis_thickness"].append(10 + i)
            s.measurement_history["qcm_thickness"].append(11 + i)
        eis_data = s.simulate_eis_measurement(15.0)
        result = s.correlate_with_qcm(eis_data, 14.5)
        assert "correlation_r2" in result

    def test_adaptive_high_confidence(self):
        from eis_qcm_biofilm_correlation import EISParameters, CombinedEISQCMSensor
        np.random.seed(42)
        s = CombinedEISQCMSensor(EISParameters())
        for i in range(10):
            s.measurement_history["eis_thickness"].append(10 + i)
            s.measurement_history["qcm_thickness"].append(10 + i * 1.01)
            s.measurement_history["time"].append(i)
            s.measurement_history["correlation_quality"].append(0.9)
        eis_data = s.simulate_eis_measurement(15.0)
        result = s.adaptive_thickness_estimation(eis_data, 15.0, confidence_threshold=0.0)
        assert result["estimation_method"] in ("weighted_combination", "qcm_preferred")

    def test_adaptive_low_confidence(self):
        from eis_qcm_biofilm_correlation import EISParameters, CombinedEISQCMSensor
        np.random.seed(42)
        s = CombinedEISQCMSensor(EISParameters())
        eis_data = s.simulate_eis_measurement(15.0)
        result = s.adaptive_thickness_estimation(eis_data, 15.0, confidence_threshold=0.99)
        assert result["estimation_method"] == "qcm_preferred"

    def test_update_calibration_no_history(self):
        from eis_qcm_biofilm_correlation import EISParameters, CombinedEISQCMSensor
        s = CombinedEISQCMSensor(EISParameters())
        eis_data = {"Z_1kHz": 100.0}
        s.update_calibration(eis_data, 15.0)

    def test_update_calibration_with_history(self):
        from eis_qcm_biofilm_correlation import EISParameters, CombinedEISQCMSensor
        s = CombinedEISQCMSensor(EISParameters())
        s.measurement_history["time"].append(0)
        eis_data = {"Z_1kHz": 100.0}
        s.update_calibration(eis_data, 15.0)
        assert s.calibration_coefficients["eis_qcm_linear"] == 15.0 / 100.0


class TestCreateAnalysis:
    def test_short_analysis(self):
        from eis_qcm_biofilm_correlation import create_eis_qcm_correlation_analysis
        np.random.seed(42)
        results = create_eis_qcm_correlation_analysis(duration_hours=2)
        assert "time_hours" in results
        assert len(results["thickness_eis"]) > 0


class TestVisualization:
    def test_visualization(self):
        with patch("eis_qcm_biofilm_correlation.plt") as mock_plt:
            mock_plt.figure.return_value = MagicMock()
            mock_plt.subplot.return_value = MagicMock()
            from eis_qcm_biofilm_correlation import create_comprehensive_visualization
            np.random.seed(42)
            n = 10
            results = {
                "time_hours": np.linspace(0, 10, n),
                "thickness_true": np.ones(n) * 15,
                "thickness_qcm": np.ones(n) * 14.5,
                "thickness_eis": [14.0] * n,
                "thickness_combined": [14.2] * n,
                "correlation_r2": [0.9] * n,
                "validation_score": [0.9] * n,
                "measurement_confidence": [0.85] * n,
                "eis_impedance_1kHz": [100.0] * n,
                "impedance_change_percent": [-22.0] * n,
            }
            create_comprehensive_visualization(results)
            mock_plt.savefig.assert_called_once()


class TestMainFunc:
    def test_main(self):
        with patch("eis_qcm_biofilm_correlation.plt") as mock_plt:
            mock_plt.figure.return_value = MagicMock()
            mock_plt.subplot.return_value = MagicMock()
            with patch("builtins.open", MagicMock()):
                with patch("eis_qcm_biofilm_correlation.pd.DataFrame") as mock_df:
                    mock_df.return_value = MagicMock()
                    np.random.seed(42)
                    from eis_qcm_biofilm_correlation import main
                    main()
