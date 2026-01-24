"""Sensing Models Module for MFC Real-Time Monitoring.

This module provides electrochemical and physical sensing models for:
1. Electrochemical Impedance Spectroscopy (EIS) biofilm characterization
2. Quartz Crystal Microbalance (QCM) mass sensing
3. Sensor fusion and cross-validation
4. Real-time biofilm monitoring integration

The models are based on literature-validated parameters and enable
accurate biofilm thickness and mass determination for MFC control.
"""

from .eis_model import EISCircuitModel, EISMeasurement, EISModel
from .qcm_model import QCMMeasurement, QCMModel, SauerbreyModel, ViscoelasticModel
from .sensor_fusion import SensorCalibration, SensorFusion

__all__ = [
    "EISCircuitModel",
    "EISMeasurement",
    "EISModel",
    "QCMMeasurement",
    "QCMModel",
    "SauerbreyModel",
    "SensorCalibration",
    "SensorFusion",
    "ViscoelasticModel",
]

__version__ = "1.0.0"
__author__ = "Claude Code with Literature Validation"
