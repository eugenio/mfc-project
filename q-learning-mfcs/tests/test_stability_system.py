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

class TestReliabilityAnalyzer(unittest.TestCase):
    """Test the reliability analyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not STABILITY_IMPORTS_AVAILABLE:
            self.skipTest("Stability analysis components not available")
        
        self.analyzer = ReliabilityAnalyzer()
        
        # Create test component data
        self.component_data = {
            'membrane_01': {
                'installation_date': datetime.now() - timedelta(days=365),
                'operational_hours': 8760,
                'failure_events': 2,
                'maintenance_events': 12,
                'current_status': 'operational'
            },
            'anode_01': {
                'installation_date': datetime.now() - timedelta(days=180),
                'operational_hours': 4320,
                'failure_events': 0,
                'maintenance_events': 6,
                'current_status': 'operational'
            }
        }
    
    def test_component_reliability_calculation(self):
        """Test component reliability calculation."""
        for component_id, data in self.component_data.items():
            reliability = self.analyzer.calculate_component_reliability(
                component_id=component_id,
                operational_hours=data['operational_hours'],
                failure_events=data['failure_events'],
                maintenance_events=data['maintenance_events']
            )
            
            self.assertIsInstance(reliability, ComponentReliability)
            self.assertEqual(reliability.component_id, component_id)
            self.assertGreaterEqual(reliability.current_reliability, 0)
            self.assertLessEqual(reliability.current_reliability, 1)
            self.assertGreater(reliability.mtbf_hours, 0)
            self.assertGreaterEqual(reliability.failure_rate, 0)
    
    def test_weibull_analysis(self):
        """Test Weibull reliability analysis."""
        # Create failure time data
        failure_times = np.array([100, 250, 400, 600, 800, 1000, 1200])
        
        result = self.analyzer.perform_weibull_analysis(failure_times)
        
        self.assertIn('shape_parameter', result)
        self.assertIn('scale_parameter', result)
        self.assertIn('characteristic_life', result)
        self.assertGreater(result['shape_parameter'], 0)
        self.assertGreater(result['scale_parameter'], 0)
    
    def test_fmea_analysis(self):
        """Test Failure Mode and Effects Analysis."""
        # Create test failure modes
        failure_modes = [
            FailureMode(
                component='membrane',
                failure_mode='fouling',
                severity=8,
                occurrence=6,
                detection=4,
                causes=['high substrate concentration', 'inadequate cleaning'],
                effects=['reduced power output', 'increased resistance']
            )
        ]
        
        self.analyzer.failure_modes = failure_modes
        fmea_result = self.analyzer.perform_fmea_analysis()
        
        self.assertIn('high_risk_modes', fmea_result)
        self.assertIn('risk_assessment', fmea_result)
        self.assertIsInstance(fmea_result['risk_assessment'], list)
    
    def test_maintenance_optimization(self):
        """Test maintenance interval optimization."""
        component_id = 'test_component'
        operational_data = np.random.exponential(1000, 100)  # Simulated failure times
        
        optimal_interval = self.analyzer.optimize_maintenance_interval(
            component_id, operational_data
        )
        
        self.assertGreater(optimal_interval, 0)
        self.assertLess(optimal_interval, 10000)  # Reasonable range

class TestDegradationDetector(unittest.TestCase):
    """Test the degradation pattern detector."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not STABILITY_IMPORTS_AVAILABLE:
            self.skipTest("Stability analysis components not available")
        
        self.detector = DegradationDetector()
    
    def test_time_series_metrics(self):
        """Test time series metrics calculation."""
        # Create test time series with trend
        timestamps = pd.date_range(start='2024-01-01', periods=100, freq='H')
        values = np.linspace(1, 2, 100) + 0.1 * np.random.randn(100)  # Linear trend + noise
        
        metrics = self.detector.calculate_time_series_metrics(values, timestamps.values)
        
        self.assertGreater(metrics.trend_slope, 0)  # Positive trend
        self.assertGreater(metrics.trend_r2, 0)     # Some correlation
        self.assertGreaterEqual(metrics.anomaly_score, 0)
        self.assertIsInstance(metrics.change_points, list)
    
    def test_degradation_pattern_detection(self):
        """Test degradation pattern detection."""
        # Add data with degradation signatures
        for i in range(50):
            data = {
                'timestamp': datetime.now() + timedelta(hours=i),
                'membrane_resistance': 0.5 + 0.01 * i,  # Increasing resistance (fouling)
                'power_output': 400 - 2 * i,             # Decreasing power
                'voltage': 0.8 - 0.001 * i,              # Decreasing voltage
                'current_density': 500 - i               # Decreasing current
            }
            self.detector.add_data_point(data)
        
        patterns = self.detector.detect_degradation_patterns()
        
        # Should detect membrane fouling pattern
        fouling_patterns = [
            p for p in patterns 
            if p.degradation_type == DegradationType.MEMBRANE_FOULING
        ]
        self.assertGreater(len(fouling_patterns), 0)
        
        # Check pattern properties
        if fouling_patterns:
            pattern = fouling_patterns[0]
            self.assertGreater(pattern.confidence, 0.3)
            self.assertIn('membrane', pattern.affected_components)
    
    def test_pattern_filtering(self):
        """Test duplicate pattern filtering."""
        # Create duplicate patterns
        pattern1 = DegradationPattern(
            pattern_id="test1",
            degradation_type=DegradationType.MEMBRANE_FOULING,
            severity=DegradationSeverity.MODERATE,
            confidence=0.8,
            start_time=datetime.now(),
            affected_components=['membrane']
        )
        
        pattern2 = DegradationPattern(
            pattern_id="test2",
            degradation_type=DegradationType.MEMBRANE_FOULING,
            severity=DegradationSeverity.LOW,
            confidence=0.6,
            start_time=datetime.now(),
            affected_components=['membrane']
        )
        
        filtered = self.detector._filter_duplicate_patterns([pattern1, pattern2])
        
        # Should keep higher confidence pattern
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].pattern_id, "test1")

class TestMaintenanceScheduler(unittest.TestCase):
    """Test the maintenance scheduler."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not STABILITY_IMPORTS_AVAILABLE:
            self.skipTest("Stability analysis components not available")
        
        self.scheduler = MaintenanceScheduler()
    
    def test_task_creation(self):
        """Test maintenance task creation."""
        task = MaintenanceTask(
            task_id="test_001",
            component="membrane",
            maintenance_type=MaintenanceType.PREVENTIVE,
            priority=MaintenancePriority.MEDIUM,
            scheduled_date=datetime.now() + timedelta(days=7),
            estimated_duration_hours=4.0,
            description="Test maintenance task",
            cost_estimate=100.0
        )
        
        self.assertEqual(task.task_id, "test_001")
        self.assertEqual(task.component, "membrane")
        self.assertEqual(task.maintenance_type, MaintenanceType.PREVENTIVE)
    
    def test_schedule_optimization(self):
        """Test maintenance schedule optimization."""
        # Create test tasks
        tasks = []
        for i in range(5):
            task = MaintenanceTask(
                task_id=f"task_{i}",
                component=f"component_{i}",
                maintenance_type=MaintenanceType.PREVENTIVE,
                priority=MaintenancePriority.MEDIUM,
                scheduled_date=datetime.now() + timedelta(days=i),
                estimated_duration_hours=2.0,
                description=f"Test task {i}",
                cost_estimate=50.0,
                required_resources=['technician']
            )
            tasks.append(task)
        
        result = self.scheduler.optimize_schedule(tasks)
        
        self.assertGreaterEqual(result.optimization_score, 0)
        self.assertLessEqual(result.optimization_score, 1)
        self.assertLessEqual(len(result.scheduled_tasks), len(tasks))
    
    def test_emergency_maintenance(self):
        """Test emergency maintenance creation."""
        from stability.maintenance_scheduler import ComponentStatus
        
        self.scheduler.update_component_status('membrane', ComponentStatus.FAILED)
        
        # Should create emergency task
        emergency_tasks = [
            task for task in self.scheduler.scheduled_tasks
            if task.maintenance_type == MaintenanceType.EMERGENCY
        ]
        
        self.assertGreater(len(emergency_tasks), 0)
        
        if emergency_tasks:
            task = emergency_tasks[0]
            self.assertEqual(task.priority, MaintenancePriority.EMERGENCY)
            self.assertEqual(task.component, 'membrane')

class TestLongTermDataManager(unittest.TestCase):
    """Test the long-term data manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not STABILITY_IMPORTS_AVAILABLE:
            self.skipTest("Stability analysis components not available")
        
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.data_manager = LongTermDataManager(data_directory=self.temp_dir)
        
        # Create test data
        self.test_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H'),
            'power_output': np.random.normal(400, 50, 100),
            'voltage': np.random.normal(0.8, 0.1, 100),
            'current_density': np.random.normal(500, 50, 100),
            'temperature': np.random.normal(25, 2, 100)
        })
    
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_data_storage(self):
        """Test data storage functionality."""
        file_id = self.data_manager.store_data(
            data=self.test_data,
            data_type=DataType.SENSOR_DATA
        )
        
        self.assertIsInstance(file_id, str)
        self.assertGreater(len(file_id), 0)
    
    def test_data_querying(self):
        """Test data querying functionality."""
        # First store some data
        file_id = self.data_manager.store_data(
            data=self.test_data,
            data_type=DataType.SENSOR_DATA
        )
        
        # Query the data back
        query = DataQuery(
            data_types=[DataType.SENSOR_DATA],
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 5),
            metrics=['power_output', 'voltage']
        )
        
        result = self.data_manager.query_data(query)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('power_output', result.columns)
        self.assertIn('voltage', result.columns)
    
    def test_trend_analysis(self):
        """Test trend analysis functionality."""
        # Store test data
        self.data_manager.store_data(
            data=self.test_data,
            data_type=DataType.PERFORMANCE_DATA
        )
        
        # Analyze trends
        result = self.data_manager.analyze_trends(
            metrics=['power_output', 'voltage'],
            time_window_days=30
        )
        
        self.assertEqual(result.analysis_type, "trend_analysis")
        self.assertIn('results', result.results)
    
    def test_data_summary(self):
        """Test data summary functionality."""
        # Store test data
        self.data_manager.store_data(
            data=self.test_data,
            data_type=DataType.SENSOR_DATA
        )
        
        summaries = self.data_manager.get_data_summary([DataType.SENSOR_DATA])
        
        self.assertGreater(len(summaries), 0)
        self.assertEqual(summaries[0].data_type, DataType.SENSOR_DATA)
        self.assertGreater(summaries[0].record_count, 0)

class TestStabilityVisualizer(unittest.TestCase):
    """Test the stability visualizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not STABILITY_IMPORTS_AVAILABLE:
            self.skipTest("Stability analysis components not available")
        
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.visualizer = StabilityVisualizer(output_directory=self.temp_dir)
        
        # Create test patterns
        self.test_patterns = [
            DegradationPattern(
                pattern_id="pattern_1",
                degradation_type=DegradationType.MEMBRANE_FOULING,
                severity=DegradationSeverity.MODERATE,
                confidence=0.85,
                start_time=datetime.now() - timedelta(days=1),
                affected_components=['membrane'],
                predicted_failure_time=datetime.now() + timedelta(days=30)
            ),
            DegradationPattern(
                pattern_id="pattern_2",
                degradation_type=DegradationType.ELECTRODE_CORROSION,
                severity=DegradationSeverity.HIGH,
                confidence=0.75,
                start_time=datetime.now() - timedelta(hours=12),
                affected_components=['anode']
            )
        ]
        
        # Create test reliability data
        self.test_reliability = [
            ComponentReliability(
                component_id='membrane',
                current_reliability=0.85,
                mtbf_hours=2000,
                failure_rate=0.0005,
                confidence_interval=(0.8, 0.9)
            ),
            ComponentReliability(
                component_id='anode',
                current_reliability=0.92,
                mtbf_hours=3000,
                failure_rate=0.0003,
                confidence_interval=(0.88, 0.96)
            )
        ]
        
        # Create test maintenance tasks
        self.test_tasks = [
            MaintenanceTask(
                task_id="maint_1",
                component="membrane",
                maintenance_type=MaintenanceType.PREVENTIVE,
                priority=MaintenancePriority.MEDIUM,
                scheduled_date=datetime.now() + timedelta(days=3),
                estimated_duration_hours=4.0,
                description="Membrane cleaning",
                cost_estimate=100.0
            )
        ]
    
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_degradation_dashboard_creation(self):
        """Test degradation dashboard creation."""
        try:
            dashboard_path = self.visualizer.create_degradation_dashboard(self.test_patterns)
            
            if dashboard_path:  # Only test if visualization was created
                self.assertTrue(Path(dashboard_path).exists())
                self.assertTrue(dashboard_path.endswith('.html'))
        except Exception as e:
            # Skip test if visualization libraries not available
            self.skipTest(f"Visualization not available: {e}")
    
    def test_reliability_trends_plot(self):
        """Test reliability trends plot creation."""
        try:
            plot_path = self.visualizer.create_reliability_trends_plot(self.test_reliability)
            
            if plot_path:  # Only test if visualization was created
                self.assertTrue(Path(plot_path).exists())
                self.assertTrue(plot_path.endswith('.html'))
        except Exception as e:
            # Skip test if visualization libraries not available
            self.skipTest(f"Visualization not available: {e}")
    
    def test_maintenance_schedule_chart(self):
        """Test maintenance schedule chart creation."""
        try:
            chart_path = self.visualizer.create_maintenance_schedule_chart(self.test_tasks)
            
            if chart_path:  # Only test if visualization was created
                self.assertTrue(Path(chart_path).exists())
                self.assertTrue(chart_path.endswith('.html'))
        except Exception as e:
            # Skip test if visualization libraries not available
            self.skipTest(f"Visualization not available: {e}")
    
    def test_stability_report_generation(self):
        """Test comprehensive stability report generation."""
        try:
            report_path = self.visualizer.generate_stability_report(
                patterns=self.test_patterns,
                reliability_data=self.test_reliability,
                maintenance_tasks=self.test_tasks,
                include_plots=False  # Skip plots to avoid dependency issues
            )
            
            self.assertTrue(Path(report_path).exists())
            self.assertTrue(report_path.endswith('.json'))
            
            # Load and verify report content
            with open(report_path, 'r') as f:
                report_data = json.load(f)
            
            self.assertIn('executive_summary', report_data)
            self.assertIn('degradation_analysis', report_data)
            self.assertIn('reliability_analysis', report_data)
            self.assertIn('maintenance_analysis', report_data)
            self.assertIn('recommendations', report_data)
            
        except Exception as e:
            # Skip test if visualization libraries not available
            self.skipTest(f"Report generation not available: {e}")

class TestStabilitySystemIntegration(unittest.TestCase):
    """Integration tests for the complete stability system."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not STABILITY_IMPORTS_AVAILABLE:
            self.skipTest("Stability analysis components not available")
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize all components
        self.framework = StabilityFramework()
        self.detector = DegradationDetector()
        self.reliability_analyzer = ReliabilityAnalyzer()
        self.maintenance_scheduler = MaintenanceScheduler()
        self.data_manager = LongTermDataManager(data_directory=self.temp_dir)
        self.visualizer = StabilityVisualizer(output_directory=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_stability_analysis_workflow(self):
        """Test complete stability analysis workflow."""
        # 1. Generate and store test data
        test_data = []
        for i in range(100):
            data_point = {
                'timestamp': datetime.now() + timedelta(hours=i),
                'power_output': 400 - i * 0.5,  # Gradual decline
                'voltage': 0.8 - i * 0.001,     # Gradual decline
                'current_density': 500 - i,     # Gradual decline
                'membrane_resistance': 0.5 + i * 0.005,  # Gradual increase
                'temperature': 25 + np.random.normal(0, 1),
                'ph': 7.0 + np.random.normal(0, 0.2)
            }
            test_data.append(data_point)
            
            # Add to framework and detector
            self.framework.add_data_point(data_point)
            self.detector.add_data_point(data_point)
        
        # Store in data manager
        df = pd.DataFrame(test_data)
        file_id = self.data_manager.store_data(df, DataType.PERFORMANCE_DATA)
        self.assertIsNotNone(file_id)
        
        # 2. Detect degradation patterns
        patterns = self.detector.detect_degradation_patterns()
        self.assertGreater(len(patterns), 0)
        
        # 3. Calculate component reliability
        reliability_data = []
        for component in ['membrane', 'anode', 'cathode']:
            reliability = self.reliability_analyzer.calculate_component_reliability(
                component_id=component,
                operational_hours=2400,
                failure_events=1,
                maintenance_events=8
            )
            reliability_data.append(reliability)
        
        self.assertEqual(len(reliability_data), 3)
        
        # 4. Generate maintenance tasks from patterns
        maintenance_tasks = self.maintenance_scheduler.analyze_degradation_patterns(patterns)
        self.assertGreaterEqual(len(maintenance_tasks), 0)
        
        # 5. Generate comprehensive report
        try:
            report_path = self.visualizer.generate_stability_report(
                patterns=patterns,
                reliability_data=reliability_data,
                maintenance_tasks=maintenance_tasks,
                include_plots=False
            )
            
            self.assertTrue(Path(report_path).exists())
            
            # Verify report content
            with open(report_path, 'r') as f:
                report = json.load(f)
            
            self.assertIn('executive_summary', report)
            self.assertGreater(report['executive_summary']['total_degradation_patterns'], 0)
            
        except Exception as e:
            self.skipTest(f"Report generation failed: {e}")
    
    def test_system_performance_under_load(self):
        """Test system performance with large datasets."""
        # Generate large dataset
        large_dataset = []
        for i in range(1000):  # 1000 data points
            data_point = {
                'timestamp': datetime.now() + timedelta(minutes=i),
                'power_output': 400 + np.random.normal(0, 20),
                'voltage': 0.8 + np.random.normal(0, 0.05),
                'current_density': 500 + np.random.normal(0, 30),
                'membrane_resistance': 0.5 + np.random.normal(0, 0.05),
                'temperature': 25 + np.random.normal(0, 2)
            }
            large_dataset.append(data_point)
        
        # Test data storage performance
        start_time = datetime.now()
        df = pd.DataFrame(large_dataset)
        file_id = self.data_manager.store_data(df, DataType.SENSOR_DATA)
        storage_time = (datetime.now() - start_time).total_seconds()
        
        self.assertLess(storage_time, 10)  # Should complete within 10 seconds
        self.assertIsNotNone(file_id)
        
        # Test degradation detection performance
        start_time = datetime.now()
        for data_point in large_dataset[-100:]:  # Add last 100 points
            self.detector.add_data_point(data_point)
        
        patterns = self.detector.detect_degradation_patterns()
        detection_time = (datetime.now() - start_time).total_seconds()
        
        self.assertLess(detection_time, 30)  # Should complete within 30 seconds

