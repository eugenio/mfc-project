"""
Comprehensive Tests for MFC Stability Analysis System

Tests for all components of the long-term stability analysis framework:
- StabilityFramework
- ReliabilityAnalyzer  
- DegradationDetector
- MaintenanceScheduler
- LongTermDataManager
- StabilityVisualizer

Created: 2025-07-28
"""
import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys
import os
    from stability.stability_framework import StabilityFramework, StabilityMetrics
    from stability.reliability_analyzer import ReliabilityAnalyzer, ComponentReliability, FailureMode
    from stability.degradation_detector import (
    from stability.maintenance_scheduler import (
    from stability.data_manager import LongTermDataManager, DataType, DataQuery
    from stability.stability_visualizer import StabilityVisualizer
    STABILITY_IMPORTS_AVAILABLE = True
    STABILITY_IMPORTS_AVAILABLE = False
class TestStabilityFramework(unittest.TestCase):
    """Test the core stability framework."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not STABILITY_IMPORTS_AVAILABLE:
            self.skipTest("Stability analysis components not available")
        
        self.framework = StabilityFramework()
        
        # Create test data
        self.test_data = {
            'timestamp': datetime.now(),
            'cell_voltages': [0.8, 0.75, 0.82],
            'current_densities': [500, 480, 520],
            'power_outputs': [400, 360, 426],
            'biofilm_thickness': [50e-6, 45e-6, 55e-6],
            'membrane_resistance': [0.5, 0.6, 0.52],
            'temperature': 25.0,
            'ph': 7.2,
            'substrate_concentration': 20.0
        }
    
    def test_stability_metrics_calculation(self):
        """Test calculation of stability metrics."""
        metrics = self.framework.calculate_stability_metrics(self.test_data)
        
        self.assertIsInstance(metrics, StabilityMetrics)
        self.assertGreater(metrics.power_stability, 0)
        self.assertGreater(metrics.voltage_stability, 0)
        self.assertGreater(metrics.efficiency_stability, 0)
        self.assertGreaterEqual(metrics.overall_stability_score, 0)
        self.assertLessEqual(metrics.overall_stability_score, 1)
    
    def test_performance_tracking(self):
        """Test performance tracking over time."""
        # Add multiple data points
        for i in range(10):
            data = self.test_data.copy()
            data['timestamp'] = datetime.now() + timedelta(hours=i)
            # Add some variation
            data['power_outputs'] = [p * (1 + 0.1 * np.sin(i/5)) for p in data['power_outputs']]
            self.framework.add_data_point(data)
        
        # Test performance analysis
        analysis = self.framework.analyze_performance_trends(hours=24)
        
        self.assertIn('power_trend', analysis)
        self.assertIn('voltage_trend', analysis)
        self.assertIn('stability_trend', analysis)
    
    def test_component_degradation_analysis(self):
        """Test component degradation analysis."""
        # Add data with degradation trend
        for i in range(20):
            data = self.test_data.copy()
            data['timestamp'] = datetime.now() + timedelta(hours=i)
            # Simulate membrane resistance increase (fouling)
            data['membrane_resistance'] = [r * (1 + 0.02 * i) for r in data['membrane_resistance']]
            self.framework.add_data_point(data)
        
        degradation = self.framework.analyze_component_degradation()
        
        self.assertIn('membrane', degradation)
        self.assertGreater(degradation['membrane']['degradation_rate'], 0)

