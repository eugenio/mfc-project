"""
Comprehensive Sensing Models Module for MFC Real-Time Monitoring

This module provides a complete suite of sensors for MFC monitoring including:
1. Electrochemical Impedance Spectroscopy (EIS) biofilm characterization
2. Quartz Crystal Microbalance (QCM) mass sensing
3. Substrate concentration sensors (UV-Vis, enzymatic, amperometric)
4. Temperature sensors with thermal dynamics (thermocouples, RTDs, thermistors)
5. Conductivity sensors for electrolyte monitoring (2/4-electrode, inductive, toroidal)
6. Dissolved oxygen sensors (Clark electrode, optical, galvanic, paramagnetic)
7. Sensor fusion and cross-validation
8. Comprehensive power consumption and cost analysis

The models are based on literature-validated parameters and enable
accurate real-time monitoring for advanced MFC control systems.
"""

# Existing sensor models
from .eis_model import EISModel, EISMeasurement, EISCircuitModel
from .qcm_model import QCMModel, QCMMeasurement, SauerbreyModel, ViscoelasticModel
from .sensor_fusion import SensorFusion, SensorCalibration

# New comprehensive sensor models
from .substrate_sensors import (
    SubstrateSensor, SubstrateSensorType, SubstrateSensorSpecs, SubstrateMeasurement,
    create_standard_substrate_sensors
)
from .temperature_sensors import (
    TemperatureSensor, TemperatureSensorType, TemperatureSensorSpecs, TemperatureMeasurement,
    ThermalDynamicsModel, create_standard_temperature_sensors
)
from .conductivity_sensors import (
    ConductivitySensor, ConductivitySensorType, ConductivitySensorSpecs, ConductivityMeasurement,
    ElectrolyteModel, create_standard_conductivity_sensors
)
from .oxygen_sensors import (
    OxygenSensor, OxygenSensorType, OxygenSensorSpecs, OxygenMeasurement,
    HenryLawModel, create_standard_oxygen_sensors
)

__all__ = [
    # Original models
    'EISModel', 'EISMeasurement', 'EISCircuitModel',
    'QCMModel', 'QCMMeasurement', 'SauerbreyModel', 'ViscoelasticModel',
    'SensorFusion', 'SensorCalibration',
    
    # Substrate sensors
    'SubstrateSensor', 'SubstrateSensorType', 'SubstrateSensorSpecs', 'SubstrateMeasurement',
    'create_standard_substrate_sensors',
    
    # Temperature sensors
    'TemperatureSensor', 'TemperatureSensorType', 'TemperatureSensorSpecs', 'TemperatureMeasurement',
    'ThermalDynamicsModel', 'create_standard_temperature_sensors',
    
    # Conductivity sensors
    'ConductivitySensor', 'ConductivitySensorType', 'ConductivitySensorSpecs', 'ConductivityMeasurement',
    'ElectrolyteModel', 'create_standard_conductivity_sensors',
    
    # Oxygen sensors
    'OxygenSensor', 'OxygenSensorType', 'OxygenSensorSpecs', 'OxygenMeasurement',
    'HenryLawModel', 'create_standard_oxygen_sensors'
]

__version__ = "2.0.0"
__author__ = "Claude Code with Literature Validation"