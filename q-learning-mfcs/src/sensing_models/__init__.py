"""Sensing Models Module for MFC Real-Time Monitoring.

This module provides electrochemical and physical sensing models for:
1. Electrochemical Impedance Spectroscopy (EIS) biofilm characterization
2. Quartz Crystal Microbalance (QCM) mass sensing
3. Sensor fusion and cross-validation
4. Real-time biofilm monitoring integration
5. Advanced predictive sensor fusion with anomaly detection

The models are based on literature-validated parameters and enable
accurate biofilm thickness and mass determination for MFC control.

Recommended Usage:
    For new code, use AdvancedSensorFusion which provides predictive
    capabilities and anomaly detection:

        from sensing_models import AdvancedSensorFusion, create_advanced_sensor_fusion

    The basic SensorFusion class is retained for backward compatibility.
"""

from .advanced_sensor_fusion import (
    AdvancedKalmanFilter,
    AdvancedSensorFusion,
    AnomalyDetection,
    BiofimGrowthPattern,
    PredictiveState,
    StatisticalAnomalyDetector,
    create_advanced_sensor_fusion,
)
from .eis_model import BacterialSpecies, EISCircuitModel, EISMeasurement, EISModel
from .qcm_model import QCMMeasurement, QCMModel, SauerbreyModel, ViscoelasticModel
from .sensor_fusion import (
    FusedMeasurement,
    FusionMethod,
    KalmanFilter,
    SensorCalibration,
    SensorFusion,
)

__all__ = [
    "AdvancedKalmanFilter",
    "AdvancedSensorFusion",
    "AnomalyDetection",
    "BacterialSpecies",
    "BiofimGrowthPattern",
    "EISCircuitModel",
    "EISMeasurement",
    "EISModel",
    "FusedMeasurement",
    "FusionMethod",
    "KalmanFilter",
    "PredictiveState",
    "QCMMeasurement",
    "QCMModel",
    "SauerbreyModel",
    "SensorCalibration",
    "SensorFusion",
    "StatisticalAnomalyDetector",
    "ViscoelasticModel",
    "create_advanced_sensor_fusion",
]

__version__ = "1.1.0"
__author__ = "Claude Code with Literature Validation"
