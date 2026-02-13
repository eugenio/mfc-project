"""
Controller System Models for MFC Q-Learning Execution

This module provides comprehensive controller models for executing trained Q-learning
models and managing control electronics in MFC systems.

Components:
- Model inference engine for Q-learning execution
- Control electronics models (MCU, ADC, DAC, GPIO)
- Real-time control loops with timing constraints
- Communication interfaces (SPI, I2C, UART, CAN)
- Hardware abstraction layer for MFC control
"""

from .control_electronics import (
    ADCSpecs,
    CommunicationInterface,
    ControlElectronics,
    DACSpecs,
    GPIOSpecs,
    InterfaceType,
    MCUSpecs,
)
from .hardware_abstraction import (
    DeviceDriver,
    HardwareAbstractionLayer,
    MFCControlInterface,
)
from .model_inference import InferenceSpecs, ModelFormat, ModelInferenceEngine
from .real_time_controller import (
    ControllerMode,
    ControlLoop,
    RealTimeController,
    TimingConstraints,
)

__all__ = [
    'ModelInferenceEngine', 'InferenceSpecs', 'ModelFormat',
    'ControlElectronics', 'MCUSpecs', 'ADCSpecs', 'DACSpecs', 'GPIOSpecs',
    'CommunicationInterface', 'InterfaceType',
    'RealTimeController', 'ControlLoop', 'TimingConstraints', 'ControllerMode',
    'HardwareAbstractionLayer', 'DeviceDriver', 'MFCControlInterface'
]
