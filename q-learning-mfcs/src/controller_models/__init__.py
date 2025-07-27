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

from .model_inference import ModelInferenceEngine, InferenceSpecs, ModelFormat
from .control_electronics import (
    ControlElectronics, MCUSpecs, ADCSpecs, DACSpecs, GPIOSpecs,
    CommunicationInterface, InterfaceType
)
from .real_time_controller import (
    RealTimeController, ControlLoop, TimingConstraints, ControllerMode
)
from .hardware_abstraction import (
    HardwareAbstractionLayer, DeviceDriver, MFCControlInterface
)

__all__ = [
    'ModelInferenceEngine', 'InferenceSpecs', 'ModelFormat',
    'ControlElectronics', 'MCUSpecs', 'ADCSpecs', 'DACSpecs', 'GPIOSpecs',
    'CommunicationInterface', 'InterfaceType',
    'RealTimeController', 'ControlLoop', 'TimingConstraints', 'ControllerMode',
    'HardwareAbstractionLayer', 'DeviceDriver', 'MFCControlInterface'
]