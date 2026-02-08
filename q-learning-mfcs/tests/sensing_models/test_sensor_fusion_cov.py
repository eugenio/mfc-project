"""Tests for sensor_fusion module - targeting 98%+ coverage."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# Mock gpu_acceleration and config before importing
sys.modules["gpu_acceleration"] = MagicMock()
sys.modules["config"] = MagicMock()
sys.modules["config"].SensorConfig = None
sys.modules["config"].FusionMethod = None

from sensing_models.eis_model import BacterialSpecies, EISMeasurement
from sensing_models.sensor_fusion import (
    FusionMethod,
    FusedMeasurement,
    KalmanFilter,
    SensorCalibration,
    SensorFusion,
)


def _make_eis_measurement(freq=1000.0, mag=500.0, phase=-0.5, timestamp=0.0):
    return EISMeasurement(
        frequency=freq, impedance_magnitude=mag, impedance_phase=phase,
        real_impedance=mag * 0.8, imaginary_impedance=-mag * 0.6,
        timestamp=timestamp, temperature=303.0,
    )


def _make_qcm_measurement(thickness=10.0, quality=0.8):
    m = MagicMock()
    m.thickness_estimate = thickness
    m.quality_factor = 50000
    m.frequency_shift = -50.0
    m.frequency = 5e6
    m.dissipation = 2e-5
    return m


class TestFusionMethod:
    def test_enum_values(self):
        assert FusionMethod.KALMAN_FILTER.value == "kalman_filter"
        assert FusionMethod.WEIGHTED_AVERAGE.value == "weighted_average"
        assert FusionMethod.MAXIMUM_LIKELIHOOD.value == "maximum_likelihood"
        assert FusionMethod.BAYESIAN_FUSION.value == "bayesian_fusion"


class TestKalmanFilter:
    def test_init_default(self):
        kf = KalmanFilter()
        assert kf.n_states == 5
        assert kf.initialized is False

    def test_init_with_config(self):
        config = MagicMock()
        config.fusion.process_noise_covariance = [0.1, 0.5, 1e-6, 0.01, 0.05]
        config.fusion.measurement_noise_covariance = [2.0, 1.0, 1e-5]
        kf = KalmanFilter(config=config)
        assert kf.Q is not None

    def test_initialize_state(self):
        kf = KalmanFilter()
        kf.initialize_state(10.0, 5.0, 0.001)
        assert kf.initialized is True
        assert kf.state[0] == 10.0

    def test_predict(self):
        kf = KalmanFilter()
        kf.initialize_state(10.0, 5.0, 0.001)
        kf.predict()
        assert kf.state[0] >= 0

    def test_update(self):
        kf = KalmanFilter()
        kf.initialize_state(10.0, 5.0, 0.001)
        kf.predict()
        measurements = np.array([11.0, 10.5, 0.001])
        uncertainties = np.array([2.0, 1.0, 0.001])
        kf.update(measurements, uncertainties)
        assert len(kf.innovation_history) == 1
        assert len(kf.residual_history) == 1

    def test_get_state_estimate(self):
        kf = KalmanFilter()
        kf.initialize_state(10.0, 5.0, 0.001)
        state, uncertainties = kf.get_state_estimate()
        assert len(state) == 5
        assert len(uncertainties) == 5

    def test_assess_filter_performance_insufficient(self):
        kf = KalmanFilter()
        result = kf.assess_filter_performance()
        assert result.get("insufficient_data") is True

    def test_assess_filter_performance(self):
        kf = KalmanFilter()
        kf.initialize_state(10.0, 5.0, 0.001)
        for i in range(10):
            kf.predict()
            m = np.array([10.0 + i * 0.1, 10.0 + i * 0.1, 0.001])
            u = np.array([2.0, 1.0, 0.001])
            kf.update(m, u)
        result = kf.assess_filter_performance()
        assert "mean_normalized_innovation" in result or "numerical_issues" in result


class TestSensorCalibration:
    def test_init_default(self):
        cal = SensorCalibration()
        assert cal.eis_reliability == 1.0
        assert cal.qcm_reliability == 1.0

    def test_init_with_config(self):
        config = MagicMock()
        config.eis.species_calibration = {"mixed": {"thickness_slope": -100.0}}
        config.eis.baseline_uncertainty = 3.0
        config.qcm.biofilm_density = 1.2
        config.qcm.mass_sensitivity_factor = 1.1
        config.qcm.baseline_uncertainty = 0.8
        config.fusion.eis_reliability = 0.9
        config.fusion.qcm_reliability = 0.95
        cal = SensorCalibration(config=config)
        assert cal.eis_reliability == 0.9

    def test_update_calibration_insufficient(self):
        cal = SensorCalibration()
        cal.update_calibration([], [])

    def test_update_calibration(self):
        cal = SensorCalibration()
        eis_measurements = [_make_eis_measurement(mag=v) for v in [500, 510, 520]]
        qcm_measurements = [_make_qcm_measurement(thickness=t) for t in [10, 11, 12]]
        cal.update_calibration(eis_measurements, qcm_measurements)
        assert len(cal.agreement_history) == 1

    def test_update_calibration_with_reference(self):
        cal = SensorCalibration()
        eis_measurements = [_make_eis_measurement(mag=v) for v in [500, 510, 520]]
        qcm_measurements = [_make_qcm_measurement(thickness=t) for t in [10, 11, 12]]
        cal.update_calibration(eis_measurements, qcm_measurements, reference_thickness=15.0)
        assert len(cal.calibration_history) == 1

    def test_update_reliability(self):
        cal = SensorCalibration()
        cal.agreement_history = [0.8] * 10
        cal._update_reliability()
        assert cal.eis_reliability > 0

    def test_get_measurement_uncertainty_eis_linear(self):
        cal = SensorCalibration()
        u = cal.get_measurement_uncertainty("eis", 20.0)
        assert u > 0

    def test_get_measurement_uncertainty_eis_quadratic(self):
        cal = SensorCalibration()
        cal.eis_calibration["uncertainty_model"] = "quadratic"
        u = cal.get_measurement_uncertainty("eis", 20.0)
        assert u > 0

    def test_get_measurement_uncertainty_qcm_sqrt(self):
        cal = SensorCalibration()
        u = cal.get_measurement_uncertainty("qcm", 15.0)
        assert u > 0

    def test_get_measurement_uncertainty_qcm_linear(self):
        cal = SensorCalibration()
        cal.qcm_calibration["uncertainty_model"] = "linear"
        u = cal.get_measurement_uncertainty("qcm", 15.0)
        assert u > 0

    def test_get_calibration_status(self):
        cal = SensorCalibration()
        status = cal.get_calibration_status()
        assert "eis_reliability" in status
        assert "needs_calibration" in status


class TestSensorFusion:
    def _make_fusion(self, method=FusionMethod.KALMAN_FILTER):
        with patch("sensing_models.sensor_fusion.get_gpu_accelerator", None):
            return SensorFusion(method=method, use_gpu=False)

    def _fuse_one(self, fusion):
        eis_m = _make_eis_measurement()
        qcm_m = _make_qcm_measurement()
        eis_props = {"thickness_um": 10.0, "conductivity_S_per_m": 0.001, "measurement_quality": 0.9}
        qcm_props = {"thickness_um": 11.0, "measurement_quality": 0.9}
        return fusion.fuse_measurements(eis_m, qcm_m, eis_props, qcm_props, time_hours=0.0)

    def test_init_kalman(self):
        fusion = self._make_fusion(FusionMethod.KALMAN_FILTER)
        assert fusion.kalman_filter is not None

    def test_init_weighted_average(self):
        fusion = self._make_fusion(FusionMethod.WEIGHTED_AVERAGE)
        assert fusion.kalman_filter is None

    def test_init_with_config(self):
        config = MagicMock()
        config.fusion.min_sensor_weight = 0.05
        config.fusion.max_sensor_disagreement = 15.0
        config.fusion.sensor_fault_threshold = 0.2
        config.fusion.process_noise_covariance = [0.1, 0.5, 1e-6, 0.01, 0.05]
        config.fusion.measurement_noise_covariance = [2.0, 1.0, 1e-5]
        config.fusion.eis_reliability = 0.9
        config.fusion.qcm_reliability = 0.95
        with patch("sensing_models.sensor_fusion.get_gpu_accelerator", None):
            fusion = SensorFusion(config=config, use_gpu=False)
            assert fusion.min_sensor_weight == 0.05

    def test_fuse_kalman(self):
        fusion = self._make_fusion(FusionMethod.KALMAN_FILTER)
        result = self._fuse_one(fusion)
        assert isinstance(result, FusedMeasurement)
        assert result.thickness_um > 0

    def test_fuse_weighted_average(self):
        fusion = self._make_fusion(FusionMethod.WEIGHTED_AVERAGE)
        result = self._fuse_one(fusion)
        assert result.thickness_um > 0

    def test_fuse_maximum_likelihood(self):
        fusion = self._make_fusion(FusionMethod.MAXIMUM_LIKELIHOOD)
        result = self._fuse_one(fusion)
        assert result.thickness_um > 0

    def test_fuse_bayesian(self):
        fusion = self._make_fusion(FusionMethod.BAYESIAN_FUSION)
        result = self._fuse_one(fusion)
        assert result.thickness_um > 0

    def test_assess_sensor_status_eis_good(self):
        fusion = self._make_fusion()
        status = fusion._assess_sensor_status(
            "eis", _make_eis_measurement(), {"measurement_quality": 0.9}
        )
        assert status == "good"

    def test_assess_sensor_status_eis_degraded(self):
        fusion = self._make_fusion()
        status = fusion._assess_sensor_status(
            "eis", _make_eis_measurement(), {"measurement_quality": 0.6}
        )
        assert status == "degraded"

    def test_assess_sensor_status_eis_failed(self):
        fusion = self._make_fusion()
        fusion.calibration.eis_reliability = 0.3
        status = fusion._assess_sensor_status(
            "eis", _make_eis_measurement(), {"measurement_quality": 0.3}
        )
        assert status == "failed"

    def test_assess_sensor_status_qcm_good(self):
        fusion = self._make_fusion()
        qcm = _make_qcm_measurement()
        qcm.quality_factor = 50000
        status = fusion._assess_sensor_status(
            "qcm", qcm, {"measurement_quality": 0.9}
        )
        assert status == "good"

    def test_assess_sensor_status_qcm_degraded(self):
        fusion = self._make_fusion()
        qcm = _make_qcm_measurement()
        qcm.quality_factor = 3000
        status = fusion._assess_sensor_status(
            "qcm", qcm, {"measurement_quality": 0.6}
        )
        assert status == "degraded"

    def test_assess_sensor_status_qcm_failed(self):
        fusion = self._make_fusion()
        fusion.calibration.qcm_reliability = 0.3
        qcm = _make_qcm_measurement()
        qcm.quality_factor = 500
        status = fusion._assess_sensor_status(
            "qcm", qcm, {"measurement_quality": 0.3}
        )
        assert status == "failed"

    def test_calculate_agreement(self):
        fusion = self._make_fusion()
        assert fusion._calculate_agreement(10.0, 10.0) == 1.0
        assert fusion._calculate_agreement(0.0, 0.0) == 1.0
        assert fusion._calculate_agreement(10.0, 15.0) > 0

    def test_estimate_biomass_geobacter(self):
        with patch("sensing_models.sensor_fusion.get_gpu_accelerator", None):
            fusion = SensorFusion(species=BacterialSpecies.GEOBACTER, use_gpu=False)
            biomass = fusion._estimate_biomass_from_thickness(20.0)
            assert biomass > 0

    def test_estimate_biomass_shewanella(self):
        with patch("sensing_models.sensor_fusion.get_gpu_accelerator", None):
            fusion = SensorFusion(species=BacterialSpecies.SHEWANELLA, use_gpu=False)
            biomass = fusion._estimate_biomass_from_thickness(20.0)
            assert biomass > 0

    def test_estimate_biomass_mixed(self):
        fusion = self._make_fusion()
        biomass = fusion._estimate_biomass_from_thickness(0.0)
        assert biomass == 0.0

    def test_calculate_fusion_confidence(self):
        fusion = self._make_fusion()
        confidence = fusion._calculate_fusion_confidence(0.8, 0.5, 0.5, "good", "good")
        assert 0.0 <= confidence <= 1.0

    def test_get_fusion_summary(self):
        fusion = self._make_fusion()
        summary = fusion.get_fusion_summary()
        assert "fusion_method" in summary
        assert "gpu_available" in summary

    def test_detect_sensor_faults_insufficient(self):
        fusion = self._make_fusion()
        faults = fusion.detect_sensor_faults()
        assert faults == {"eis_faults": [], "qcm_faults": [], "fusion_faults": []}

    def test_detect_sensor_faults(self):
        fusion = self._make_fusion()
        for i in range(10):
            self._fuse_one(fusion)
        faults = fusion.detect_sensor_faults()
        assert isinstance(faults, dict)

    def test_detect_sensor_faults_failures(self):
        fusion = self._make_fusion()
        # Populate history with failed sensors
        for _ in range(10):
            fm = FusedMeasurement(
                timestamp=0.0, thickness_um=10.0, thickness_uncertainty=2.0,
                biomass_density_g_per_L=1.0, biomass_uncertainty=0.2,
                conductivity_S_per_m=0.001, conductivity_uncertainty=0.0001,
                eis_thickness=10.0, qcm_thickness=10.0, eis_weight=0.5,
                qcm_weight=0.5, sensor_agreement=0.1, fusion_confidence=0.2,
                cross_validation_error=15.0, eis_status="failed", qcm_status="failed",
            )
            fusion.fusion_history.append(fm)
        fusion.calibration.eis_reliability = 0.1
        fusion.calibration.qcm_reliability = 0.1
        faults = fusion.detect_sensor_faults()
        assert "frequent_failures" in faults["eis_faults"]
        assert "low_reliability" in faults["eis_faults"]
        assert "poor_sensor_agreement" in faults["fusion_faults"]

    def test_weighted_average_zero_weights(self):
        fusion = self._make_fusion(FusionMethod.WEIGHTED_AVERAGE)
        fusion.calibration.eis_reliability = 0.0
        fusion.calibration.qcm_reliability = 0.0
        result = self._fuse_one(fusion)
        assert result.thickness_um >= 0

    def test_bayesian_no_above_half_max(self):
        fusion = self._make_fusion(FusionMethod.BAYESIAN_FUSION)
        eis_m = _make_eis_measurement()
        qcm_m = _make_qcm_measurement()
        eis_props = {"thickness_um": 0.001, "conductivity_S_per_m": 0.001, "measurement_quality": 0.9}
        qcm_props = {"thickness_um": 100.0, "measurement_quality": 0.9}
        result = fusion.fuse_measurements(eis_m, qcm_m, eis_props, qcm_props)
        assert result.thickness_um >= 0

    def test_assess_filter_performance_linalg_error(self):
        """Cover LinAlgError branch and empty nis_values in assess_filter_performance."""
        kf = KalmanFilter()
        kf.initialize_state(10.0, 5.0, 0.001)
        for i in range(10):
            kf.predict()
            m = np.array([10.0, 10.0, 0.001])
            u = np.array([2.0, 1.0, 0.001])
            kf.update(m, u)
        # Force LinAlgError in all iterations so nis_values stays empty
        with patch("numpy.linalg.inv", side_effect=np.linalg.LinAlgError("singular")):
            result = kf.assess_filter_performance()
            assert result.get("numerical_issues") is True

    def test_init_with_gpu(self):
        """Cover GPU accelerator init branch (lines 560-561)."""
        mock_acc = MagicMock()
        mock_acc.is_gpu_available.return_value = True
        mock_get = MagicMock(return_value=mock_acc)
        with patch("sensing_models.sensor_fusion.get_gpu_accelerator", mock_get):
            fusion = SensorFusion(method=FusionMethod.KALMAN_FILTER, use_gpu=True)
            assert fusion.gpu_available is True

    def test_bayesian_zero_total_likelihood(self):
        """Cover total_likelihood == 0 branch in Bayesian fusion (line 938).

        Uses np.exp returning 0 to force both likelihoods to zero.
        """
        fusion = self._make_fusion(FusionMethod.BAYESIAN_FUSION)
        eis_m = _make_eis_measurement()
        qcm_m = _make_qcm_measurement()
        eis_props = {"thickness_um": 10.0, "conductivity_S_per_m": 0.001, "measurement_quality": 0.9}
        qcm_props = {"thickness_um": 10.0, "measurement_quality": 0.9}
        original_exp = np.exp
        call_count = [0]
        def patched_exp(x):
            call_count[0] += 1
            result = original_exp(x)
            # Return zeros for all likelihood computations
            if isinstance(result, np.ndarray):
                return np.zeros_like(result)
            return 0.0
        with patch.object(np, "exp", side_effect=patched_exp):
            result = fusion.fuse_measurements(eis_m, qcm_m, eis_props, qcm_props)
            assert result.thickness_um >= 0
