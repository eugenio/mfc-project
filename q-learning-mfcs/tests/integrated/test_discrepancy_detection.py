import pytest
import numpy as np
from unittest.mock import Mock
from datetime import datetime

class TestVirtualPhysicalDiscrepancyDetection:
    """Test detection of discrepancies between virtual and physical systems"""
    
    @pytest.fixture
    def discrepancy_detector(self):
        """Create mock discrepancy detection system"""
        class MockDiscrepancyDetector:
            def __init__(self):
                self.alert_conditions = {}
                self.active_alerts = []
            
            def add_alert_condition(self, name, condition_func, severity):
                self.alert_conditions[name] = {
                    'condition': condition_func,
                    'severity': severity
                }
            
            def check_conditions(self, state):
                self.active_alerts = []
                for name, alert_config in self.alert_conditions.items():
                    if alert_config['condition'](state):
                        alert = Mock()
                        alert.condition_name = name
                        alert.severity = Mock()
                        alert.severity.value = alert_config['severity']
                        self.active_alerts.append(alert)
            
            def get_active_alerts(self):
                return self.active_alerts
        
        detector = MockDiscrepancyDetector()
        detector.add_alert_condition(
            'biofilm_discrepancy',
            lambda state: max(state.get('biofilm_thickness_error', [0])) > 5.0,
            'high'
        )
        detector.add_alert_condition(
            'power_discrepancy', 
            lambda state: abs(state.get('power_prediction_error', 0)) > 0.1,
            'medium'
        )
        return detector
    
    @pytest.fixture
    def digital_twin_model(self):
        """Create mock digital twin model"""
        model = Mock()
        model.step_integrated_dynamics.return_value = Mock(
            biofilm_thickness=[20.0, 22.0, 18.0, 25.0, 21.0],
            average_power=0.35
        )
        return model
    
    def test_biofilm_thickness_discrepancy(self, digital_twin_model, discrepancy_detector):
        """Test detection of biofilm thickness discrepancies"""
        # Simulate physical measurements
        physical_measurements = {
            'biofilm_thickness': [35.0, 38.0, 32.0, 40.0, 36.0],  # Higher than model
            'sensor_confidence': [0.9, 0.85, 0.92, 0.88, 0.90]
        }
        
        # Get model predictions
        model_state = digital_twin_model.step_integrated_dynamics(dt=1.0)
        
        # Calculate discrepancies
        thickness_errors = [
            abs(phys - model) for phys, model in 
            zip(physical_measurements['biofilm_thickness'], model_state.biofilm_thickness)
        ]
        
        # Check for discrepancies
        discrepancy_state = {
            'biofilm_thickness_error': thickness_errors,
            'max_thickness_error': max(thickness_errors)
        }
        
        discrepancy_detector.check_conditions(discrepancy_state)
        alerts = discrepancy_detector.get_active_alerts()
        
        # Should detect discrepancy if error > 5 μm
        if max(thickness_errors) > 5.0:
            assert any(alert.condition_name == 'biofilm_discrepancy' for alert in alerts)
    
    def test_power_output_discrepancy(self, digital_twin_model, discrepancy_detector):
        """Test detection of power output discrepancies"""
        # Get model prediction
        model_state = digital_twin_model.step_integrated_dynamics(dt=1.0)
        predicted_power = model_state.average_power
        
        # Simulate measured power (with significant difference)
        measured_power = predicted_power * 1.5  # 50% higher than predicted
        
        power_error = abs(predicted_power - measured_power)
        
        discrepancy_state = {
            'power_prediction_error': power_error,
            'predicted_power': predicted_power,
            'measured_power': measured_power
        }
        
        discrepancy_detector.check_conditions(discrepancy_state)
        alerts = discrepancy_detector.get_active_alerts()
        
        if power_error > 0.1:
            assert any(alert.condition_name == 'power_discrepancy' for alert in alerts)
    
    def test_sensor_fault_detection(self, discrepancy_detector):
        """Test detection of sensor faults causing discrepancies"""
        # Add sensor fault condition
        discrepancy_detector.add_alert_condition(
            'sensor_fault',
            lambda state: any(status == 'failed' for status in state.get('sensor_status', [])),
            'high'
        )
        
        # Simulate sensor fault scenario
        fault_state = {
            'sensor_status': ['good', 'failed', 'good', 'degraded', 'good']
        }
        
        discrepancy_detector.check_conditions(fault_state)
        alerts = discrepancy_detector.get_active_alerts()
        
        assert any(alert.condition_name == 'sensor_fault' for alert in alerts)
