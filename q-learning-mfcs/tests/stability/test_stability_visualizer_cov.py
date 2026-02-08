"""Tests for stability_visualizer module."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import json
import shutil
import tempfile
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock all the dependencies before importing the module
mock_degradation = MagicMock()
mock_reliability = MagicMock()
mock_maintenance = MagicMock()
mock_data_manager = MagicMock()
mock_plotly_express = MagicMock()
mock_plotly_go = MagicMock()
mock_plotly_offline = MagicMock()
mock_plotly_subplots = MagicMock()


class MockDegradationType(Enum):
    BIOFILM_AGING = "biofilm_aging"
    ELECTRODE_CORROSION = "electrode_corrosion"
    MEMBRANE_FOULING = "membrane_fouling"


class MockDegradationSeverity(Enum):
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    FAILURE = "failure"


class MockMaintenanceType(Enum):
    PREVENTIVE = "preventive"
    CORRECTIVE = "corrective"


class MockMaintenancePriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MockDegradationPattern:
    def __init__(self, severity, components, confidence=0.8,
                 deg_type=None, predicted_failure=None, pattern_id="P001"):
        self.severity = severity
        self.affected_components = components
        self.confidence = confidence
        self.degradation_type = deg_type or MockDegradationType.BIOFILM_AGING
        self.predicted_failure_time = predicted_failure
        self.pattern_id = pattern_id


class MockComponentReliability:
    def __init__(self, component_id, reliability=0.9, mtbf=1000, failure_rate=0.001):
        self.component_id = component_id
        self.current_reliability = reliability
        self.mtbf_hours = mtbf
        self.failure_rate = failure_rate


class MockMaintenanceTask:
    def __init__(self, task_id, component, priority=None, mtype=None,
                 scheduled=None, duration=2.0, cost=100.0, downtime=1.0):
        self.task_id = task_id
        self.component = component
        self.priority = priority or MockMaintenancePriority.MEDIUM
        self.maintenance_type = mtype or MockMaintenanceType.PREVENTIVE
        self.scheduled_date = scheduled or datetime.now() + timedelta(days=7)
        self.estimated_duration_hours = duration
        self.cost_estimate = cost
        self.downtime_impact = downtime


# Patch modules before import
sys.modules['degradation_detector'] = MagicMock()
sys.modules['reliability_analyzer'] = MagicMock()
sys.modules['maintenance_scheduler'] = MagicMock()
sys.modules['data_manager'] = MagicMock()

# Set up the mock classes
sys.modules['degradation_detector'].DegradationPattern = MockDegradationPattern
sys.modules['degradation_detector'].DegradationSeverity = MockDegradationSeverity
sys.modules['degradation_detector'].DegradationType = MockDegradationType
sys.modules['degradation_detector'].DegradationDetector = MagicMock
sys.modules['reliability_analyzer'].ComponentReliability = MockComponentReliability
sys.modules['reliability_analyzer'].ReliabilityAnalyzer = MagicMock
sys.modules['maintenance_scheduler'].MaintenanceScheduler = MagicMock
sys.modules['maintenance_scheduler'].MaintenanceTask = MockMaintenanceTask
sys.modules['data_manager'].LongTermDataManager = MagicMock

# Now mock plotly
mock_fig = MagicMock()
mock_fig.update_layout = MagicMock()
mock_fig.add_trace = MagicMock()
mock_fig.add_vline = MagicMock()

sys.modules['plotly'] = MagicMock()
sys.modules['plotly.express'] = MagicMock()
sys.modules['plotly.graph_objects'] = MagicMock()
sys.modules['plotly.offline'] = MagicMock()
sys.modules['plotly.subplots'] = MagicMock()
sys.modules['plotly.subplots'].make_subplots = MagicMock(return_value=mock_fig)
sys.modules['plotly.express'].timeline = MagicMock(return_value=mock_fig)
sys.modules['plotly.express'].scatter = MagicMock(return_value=mock_fig)
sys.modules['plotly.graph_objects'].Figure = MagicMock(return_value=mock_fig)
sys.modules['plotly.graph_objects'].Pie = MagicMock()
sys.modules['plotly.graph_objects'].Bar = MagicMock()
sys.modules['plotly.graph_objects'].Histogram = MagicMock()
sys.modules['plotly.graph_objects'].Scatter = MagicMock()
sys.modules['plotly.graph_objects'].Box = MagicMock()
sys.modules['plotly.graph_objects'].Heatmap = MagicMock()

from stability.stability_visualizer import StabilityVisualizer, VisualizationConfig


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def visualizer(tmp_dir):
    return StabilityVisualizer(output_directory=tmp_dir)


def _make_patterns(n=3, with_predictions=True):
    patterns = []
    sevs = [MockDegradationSeverity.MINIMAL, MockDegradationSeverity.MODERATE,
            MockDegradationSeverity.CRITICAL]
    for i in range(n):
        pred = datetime.now() + timedelta(days=30) if with_predictions else None
        patterns.append(MockDegradationPattern(
            severity=sevs[i % len(sevs)],
            components=["membrane", "anode"],
            confidence=0.7 + i * 0.1,
            predicted_failure=pred,
            pattern_id=f"P{i:03d}",
        ))
    return patterns


def _make_reliability(n=3):
    return [MockComponentReliability(f"comp_{i}", 0.85 + i * 0.03, 1000 + i * 100)
            for i in range(n)]


def _make_tasks(n=3):
    tasks = []
    for i in range(n):
        tasks.append(MockMaintenanceTask(
            task_id=f"T{i:03d}", component=f"comp_{i}",
            scheduled=datetime.now() + timedelta(days=i * 10),
        ))
    return tasks


class TestVisualizationConfig:
    def test_defaults(self):
        cfg = VisualizationConfig()
        assert cfg.figure_size == (12, 8)
        assert cfg.dpi == 300


class TestStabilityVisualizer:
    def test_init(self, tmp_dir):
        v = StabilityVisualizer(output_directory=tmp_dir)
        assert Path(tmp_dir).exists()
        assert v.config is not None

    def test_init_with_config(self, tmp_dir):
        cfg = VisualizationConfig()
        cfg.dpi = 150
        v = StabilityVisualizer(output_directory=tmp_dir, config=cfg)
        assert v.config.dpi == 150

    def test_severity_to_score(self, visualizer):
        assert visualizer._severity_to_score(MockDegradationSeverity.MINIMAL) == 1
        assert visualizer._severity_to_score(MockDegradationSeverity.CRITICAL) == 5
        assert visualizer._severity_to_score(MockDegradationSeverity.FAILURE) == 6

    def test_create_degradation_dashboard_empty(self, visualizer):
        assert visualizer.create_degradation_dashboard([]) == ""

    def test_create_degradation_dashboard(self, visualizer):
        patterns = _make_patterns()
        result = visualizer.create_degradation_dashboard(patterns)
        assert result != ""

    def test_create_degradation_dashboard_no_predictions(self, visualizer):
        patterns = _make_patterns(with_predictions=False)
        result = visualizer.create_degradation_dashboard(patterns)
        assert result != ""

    def test_create_reliability_trends_empty(self, visualizer):
        assert visualizer.create_reliability_trends_plot([]) == ""

    def test_create_reliability_trends(self, visualizer):
        data = _make_reliability()
        result = visualizer.create_reliability_trends_plot(data)
        assert result != ""

    def test_create_maintenance_schedule_empty(self, visualizer):
        assert visualizer.create_maintenance_schedule_chart([]) == ""

    def test_create_maintenance_schedule(self, visualizer):
        tasks = _make_tasks()
        result = visualizer.create_maintenance_schedule_chart(tasks)
        assert result != ""

    def test_create_component_health_heatmap(self, visualizer):
        patterns = _make_patterns()
        reliability = _make_reliability()
        result = visualizer.create_component_health_heatmap(patterns, reliability)
        assert result != ""

    def test_create_failure_prediction_empty(self, visualizer):
        assert visualizer.create_failure_prediction_plot([]) == ""

    def test_create_failure_prediction_no_predictions(self, visualizer):
        patterns = _make_patterns(with_predictions=False)
        assert visualizer.create_failure_prediction_plot(patterns) == ""

    def test_create_failure_prediction(self, visualizer):
        patterns = _make_patterns(with_predictions=True)
        result = visualizer.create_failure_prediction_plot(patterns)
        assert result != ""

    def test_generate_stability_report_no_plots(self, visualizer):
        patterns = _make_patterns()
        reliability = _make_reliability()
        tasks = _make_tasks()
        result = visualizer.generate_stability_report(
            patterns, reliability, tasks, include_plots=False)
        assert result != ""
        report_data = json.loads(Path(result).read_text())
        assert "executive_summary" in report_data

    def test_generate_stability_report_with_plots(self, visualizer):
        patterns = _make_patterns()
        reliability = _make_reliability()
        tasks = _make_tasks()
        result = visualizer.generate_stability_report(
            patterns, reliability, tasks, include_plots=True)
        assert result != ""

    def test_generate_executive_summary(self, visualizer):
        patterns = _make_patterns()
        reliability = _make_reliability()
        tasks = _make_tasks()
        summary = visualizer._generate_executive_summary(patterns, reliability, tasks)
        assert "total_degradation_patterns" in summary
        assert summary["total_degradation_patterns"] == 3

    def test_generate_executive_summary_empty(self, visualizer):
        summary = visualizer._generate_executive_summary([], [], [])
        assert summary["total_degradation_patterns"] == 0
        assert summary["average_system_reliability"] == 0

    def test_generate_executive_summary_critical(self, visualizer):
        patterns = [MockDegradationPattern(MockDegradationSeverity.CRITICAL, ["m"])]
        summary = visualizer._generate_executive_summary(patterns, [], [])
        assert summary["critical_degradation_patterns"] == 1

    def test_generate_executive_summary_emergency_tasks(self, visualizer):
        task = MockMaintenanceTask("T1", "c", priority=MockMaintenancePriority.EMERGENCY)
        summary = visualizer._generate_executive_summary([], [], [task])
        assert summary["emergency_maintenance_tasks"] == 1

    def test_analyze_degradation_patterns_empty(self, visualizer):
        result = visualizer._analyze_degradation_patterns([])
        assert "error" in result

    def test_analyze_degradation_patterns(self, visualizer):
        patterns = _make_patterns()
        result = visualizer._analyze_degradation_patterns(patterns)
        assert result["total_patterns"] == 3
        assert "patterns_by_type" in result
        assert "confidence_statistics" in result

    def test_analyze_reliability_empty(self, visualizer):
        result = visualizer._analyze_reliability_data([])
        assert "error" in result

    def test_analyze_reliability(self, visualizer):
        data = _make_reliability()
        result = visualizer._analyze_reliability_data(data)
        assert result["components_analyzed"] == 3

    def test_analyze_reliability_low_component(self, visualizer):
        data = [MockComponentReliability("c1", 0.5)]
        result = visualizer._analyze_reliability_data(data)
        assert "c1" in result["low_reliability_components"]

    def test_analyze_maintenance_empty(self, visualizer):
        result = visualizer._analyze_maintenance_schedule([])
        assert "error" in result

    def test_analyze_maintenance(self, visualizer):
        tasks = _make_tasks()
        result = visualizer._analyze_maintenance_schedule(tasks)
        assert result["total_tasks"] == 3

    def test_analyze_maintenance_overdue(self, visualizer):
        task = MockMaintenanceTask("T1", "c",
                                    scheduled=datetime.now() - timedelta(days=5))
        result = visualizer._analyze_maintenance_schedule([task])
        assert result["overdue_tasks"] == 1

    def test_generate_recommendations_critical(self, visualizer):
        patterns = [MockDegradationPattern(MockDegradationSeverity.CRITICAL, ["m"])]
        recs = visualizer._generate_recommendations(patterns, [], [])
        assert any("URGENT" in r for r in recs)

    def test_generate_recommendations_multi_issues(self, visualizer):
        patterns = [
            MockDegradationPattern(MockDegradationSeverity.MODERATE, ["m"],
                                    deg_type=MockDegradationType.BIOFILM_AGING),
            MockDegradationPattern(MockDegradationSeverity.MODERATE, ["m"],
                                    deg_type=MockDegradationType.ELECTRODE_CORROSION),
            MockDegradationPattern(MockDegradationSeverity.MODERATE, ["m"],
                                    deg_type=MockDegradationType.MEMBRANE_FOULING),
        ]
        recs = visualizer._generate_recommendations(patterns, [], [])
        assert any("comprehensive" in r.lower() for r in recs)

    def test_generate_recommendations_low_reliability(self, visualizer):
        rel = [MockComponentReliability("c1", 0.5)]
        recs = visualizer._generate_recommendations([], rel, [])
        assert any("low reliability" in r.lower() for r in recs)

    def test_generate_recommendations_overdue(self, visualizer):
        task = MockMaintenanceTask("T1", "c",
                                    scheduled=datetime.now() - timedelta(days=5))
        recs = visualizer._generate_recommendations([], [], [task])
        assert any("overdue" in r.lower() for r in recs)

    def test_generate_recommendations_many_patterns(self, visualizer):
        patterns = _make_patterns(15)
        recs = visualizer._generate_recommendations(patterns, [], [])
        assert any("frequent monitoring" in r.lower() for r in recs)

    def test_generate_recommendations_stable(self, visualizer):
        recs = visualizer._generate_recommendations([], [], [])
        assert any("stable" in r.lower() for r in recs)

    def test_severity_to_score_unknown(self, visualizer):
        mock_sev = MagicMock()
        assert visualizer._severity_to_score(mock_sev) == 0
