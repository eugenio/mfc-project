"""
Stability Reporting and Visualization Tools for MFC Long-term Analysis

Comprehensive visualization and reporting system for MFC stability studies.
Creates interactive dashboards, reports, and visualizations for degradation
patterns, reliability metrics, and maintenance scheduling.

Created: 2025-07-28
"""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
from data_manager import LongTermDataManager
from degradation_detector import (
    DegradationDetector,
    DegradationPattern,
    DegradationSeverity,
)
from maintenance_scheduler import MaintenanceScheduler, MaintenanceTask
from plotly.subplots import make_subplots
from reliability_analyzer import ComponentReliability, ReliabilityAnalyzer


class VisualizationConfig:
    """Configuration for visualization settings."""
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    color_scheme: str = "viridis"
    show_interactive: bool = True
    save_static: bool = True
    output_format: str = "png"

class StabilityVisualizer:
    """
    Comprehensive stability visualization and reporting system.
    
    Features:
    - Interactive degradation pattern dashboards
    - Reliability trend visualizations
    - Maintenance schedule charts
    - Component health heatmaps
    - Failure prediction plots
    - Comprehensive PDF reports
    """

    def __init__(self,
                 output_directory: str = "../reports/stability_reports",
                 config: Optional[VisualizationConfig] = None):
        """
        Initialize the stability visualizer.
        
        Args:
            output_directory: Directory for saving reports and visualizations
            config: Visualization configuration
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

        self.config = config or VisualizationConfig()

        # Initialize analysis components
        self.degradation_detector = DegradationDetector()
        self.reliability_analyzer = ReliabilityAnalyzer()
        self.maintenance_scheduler = MaintenanceScheduler()
        self.data_manager = LongTermDataManager()

        # Color schemes
        self.severity_colors = {
            'minimal': '#2E8B57',     # Sea Green
            'low': '#FFD700',         # Gold
            'moderate': '#FF8C00',    # Dark Orange
            'high': '#FF4500',        # Orange Red
            'critical': '#DC143C',    # Crimson
            'failure': '#8B0000'      # Dark Red
        }

        self.component_colors = {
            'membrane': '#1f77b4',
            'anode': '#ff7f0e',
            'cathode': '#2ca02c',
            'separator': '#d62728',
            'housing': '#9467bd',
            'system': '#8c564b'
        }

        # Logger
        self.logger = logging.getLogger(__name__)

    def create_degradation_dashboard(self,
                                   patterns: List[DegradationPattern],
                                   time_window_days: int = 30) -> str:
        """
        Create interactive degradation pattern dashboard.
        
        Args:
            patterns: List of degradation patterns
            time_window_days: Time window for analysis
            
        Returns:
            Path to saved dashboard HTML file
        """
        if not patterns:
            self.logger.warning("No degradation patterns to visualize")
            return ""

        # Create subplot structure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Degradation Patterns by Severity',
                'Component Health Status',
                'Pattern Confidence Distribution',
                'Degradation Types by Component',
                'Failure Predictions Timeline',
                'Root Cause Analysis'
            ],
            specs=[
                [{"type": "pie"}, {"type": "bar"}],
                [{"type": "histogram"}, {"type": "bar"}],
                [{"colspan": 2}, None]
            ]
        )

        # 1. Degradation patterns by severity (pie chart)
        severity_counts = {}
        for pattern in patterns:
            severity = pattern.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        fig.add_trace(
            go.Pie(
                labels=list(severity_counts.keys()),
                values=list(severity_counts.values()),
                marker_colors=[self.severity_colors.get(s, '#999999') for s in severity_counts.keys()],
                name="Severity Distribution"
            ),
            row=1, col=1
        )

        # 2. Component health status (bar chart)
        component_status = {}
        for pattern in patterns:
            for component in pattern.affected_components:
                if component not in component_status:
                    component_status[component] = {'count': 0, 'max_severity': 0}

                component_status[component]['count'] += 1
                severity_score = self._severity_to_score(pattern.severity)
                component_status[component]['max_severity'] = max(
                    component_status[component]['max_severity'],
                    severity_score
                )

        components = list(component_status.keys())
        severity_scores = [component_status[c]['max_severity'] for c in components]

        fig.add_trace(
            go.Bar(
                x=components,
                y=severity_scores,
                marker_color=[self.component_colors.get(c, '#999999') for c in components],
                name="Component Health"
            ),
            row=1, col=2
        )

        # 3. Pattern confidence distribution (histogram)
        confidences = [pattern.confidence for pattern in patterns]

        fig.add_trace(
            go.Histogram(
                x=confidences,
                nbinsx=20,
                marker_color='lightblue',
                name="Confidence Distribution"
            ),
            row=2, col=1
        )

        # 4. Degradation types by component (stacked bar)
        degradation_component_matrix = {}
        for pattern in patterns:
            deg_type = pattern.degradation_type.value
            for component in pattern.affected_components:
                if component not in degradation_component_matrix:
                    degradation_component_matrix[component] = {}
                degradation_component_matrix[component][deg_type] = \
                    degradation_component_matrix[component].get(deg_type, 0) + 1

        # Create stacked bar chart
        components = list(degradation_component_matrix.keys())
        all_deg_types = set()
        for comp_data in degradation_component_matrix.values():
            all_deg_types.update(comp_data.keys())

        for deg_type in all_deg_types:
            values = [degradation_component_matrix[comp].get(deg_type, 0) for comp in components]
            fig.add_trace(
                go.Bar(
                    x=components,
                    y=values,
                    name=deg_type.replace('_', ' ').title()
                ),
                row=2, col=2
            )

        # 5. Failure predictions timeline (scatter plot)
        prediction_data = []
        for pattern in patterns:
            if pattern.predicted_failure_time:
                prediction_data.append({
                    'component': pattern.affected_components[0] if pattern.affected_components else 'unknown',
                    'failure_time': pattern.predicted_failure_time,
                    'confidence': pattern.confidence,
                    'severity': pattern.severity.value,
                    'pattern_id': pattern.pattern_id
                })

        if prediction_data:
            pred_df = pd.DataFrame(prediction_data)

            fig.add_trace(
                go.Scatter(
                    x=pred_df['failure_time'],
                    y=pred_df['component'],
                    mode='markers',
                    marker=dict(
                        size=pred_df['confidence'] * 20,
                        color=[self.severity_colors.get(s, '#999999') for s in pred_df['severity']],
                        opacity=0.7
                    ),
                    text=pred_df['pattern_id'],
                    name="Failure Predictions"
                ),
                row=3, col=1
            )

        # Update layout
        fig.update_layout(
            title=f"MFC Degradation Analysis Dashboard - {datetime.now().strftime('%Y-%m-%d')}",
            height=1200,
            showlegend=True,
            template="plotly_white"
        )

        # Save dashboard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dashboard_path = self.output_directory / f"degradation_dashboard_{timestamp}.html"

        pyo.plot(fig, filename=str(dashboard_path), auto_open=False)

        self.logger.info(f"Degradation dashboard saved to {dashboard_path}")
        return str(dashboard_path)

    def create_reliability_trends_plot(self,
                                     reliability_data: List[ComponentReliability],
                                     time_window_days: int = 90) -> str:
        """Create reliability trends visualization."""

        if not reliability_data:
            self.logger.warning("No reliability data to visualize")
            return ""

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Reliability vs Time',
                'MTBF Trends',
                'Failure Rate Distribution',
                'Maintenance Impact'
            ]
        )

        # Process reliability data by component
        component_data = {}
        for rel_data in reliability_data:
            component = rel_data.component_id
            if component not in component_data:
                component_data[component] = {
                    'reliability': [],
                    'mtbf': [],
                    'failure_rate': [],
                    'maintenance_events': []
                }

            component_data[component]['reliability'].append(rel_data.current_reliability)
            component_data[component]['mtbf'].append(rel_data.mtbf_hours)
            component_data[component]['failure_rate'].append(rel_data.failure_rate)

        # Create time series for reliability
        time_points = pd.date_range(
            start=datetime.now() - timedelta(days=time_window_days),
            end=datetime.now(),
            periods=len(reliability_data)
        )

        # 1. Reliability vs Time
        for component, data in component_data.items():
            if data['reliability']:
                fig.add_trace(
                    go.Scatter(
                        x=time_points[:len(data['reliability'])],
                        y=data['reliability'],
                        mode='lines+markers',
                        name=f"{component} Reliability",
                        line=dict(color=self.component_colors.get(component, '#999999'))
                    ),
                    row=1, col=1
                )

        # 2. MTBF Trends
        for component, data in component_data.items():
            if data['mtbf']:
                fig.add_trace(
                    go.Scatter(
                        x=time_points[:len(data['mtbf'])],
                        y=data['mtbf'],
                        mode='lines+markers',
                        name=f"{component} MTBF",
                        line=dict(color=self.component_colors.get(component, '#999999'))
                    ),
                    row=1, col=2
                )

        # 3. Failure Rate Distribution
        all_failure_rates = []
        components_list = []
        for component, data in component_data.items():
            all_failure_rates.extend(data['failure_rate'])
            components_list.extend([component] * len(data['failure_rate']))

        if all_failure_rates:
            fig.add_trace(
                go.Box(
                    y=all_failure_rates,
                    x=components_list,
                    name="Failure Rate Distribution"
                ),
                row=2, col=1
            )

        # 4. Maintenance Impact (placeholder - would need maintenance data)
        fig.add_trace(
            go.Scatter(
                x=[datetime.now()],
                y=[0],
                mode='markers',
                name="Maintenance Events",
                text="No maintenance data available"
            ),
            row=2, col=2
        )

        fig.update_layout(
            title="MFC System Reliability Analysis",
            height=800,
            showlegend=True,
            template="plotly_white"
        )

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.output_directory / f"reliability_trends_{timestamp}.html"

        pyo.plot(fig, filename=str(plot_path), auto_open=False)

        self.logger.info(f"Reliability trends plot saved to {plot_path}")
        return str(plot_path)

    def create_maintenance_schedule_chart(self,
                                        tasks: List[MaintenanceTask],
                                        time_horizon_days: int = 90) -> str:
        """Create maintenance schedule Gantt chart."""

        if not tasks:
            self.logger.warning("No maintenance tasks to visualize")
            return ""

        # Prepare data for Gantt chart
        gantt_data = []

        for task in tasks:
            start_date = task.scheduled_date
            end_date = start_date + timedelta(hours=task.estimated_duration_hours)

            gantt_data.append({
                'Task': task.task_id,
                'Start': start_date,
                'Finish': end_date,
                'Component': task.component,
                'Priority': task.priority.value,
                'Type': task.maintenance_type.value,
                'Duration': task.estimated_duration_hours,
                'Cost': task.cost_estimate
            })

        df = pd.DataFrame(gantt_data)

        # Create Gantt chart
        fig = px.timeline(
            df,
            x_start='Start',
            x_end='Finish',
            y='Component',
            color='Priority',
            hover_data=['Type', 'Duration', 'Cost'],
            title="MFC Maintenance Schedule",
            color_discrete_map={
                'low': '#90EE90',
                'medium': '#FFD700',
                'high': '#FF8C00',
                'critical': '#FF4500',
                'emergency': '#DC143C'
            }
        )

        fig.update_layout(
            height=600,
            xaxis_title="Timeline",
            yaxis_title="Component",
            template="plotly_white"
        )

        # Save chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_path = self.output_directory / f"maintenance_schedule_{timestamp}.html"

        pyo.plot(fig, filename=str(chart_path), auto_open=False)

        self.logger.info(f"Maintenance schedule chart saved to {chart_path}")
        return str(chart_path)

    def create_component_health_heatmap(self,
                                      patterns: List[DegradationPattern],
                                      reliability_data: List[ComponentReliability]) -> str:
        """Create component health status heatmap."""

        # Collect component health data
        components = set()
        for pattern in patterns:
            components.update(pattern.affected_components)

        for rel_data in reliability_data:
            components.add(rel_data.component_id)

        components = sorted(list(components))

        # Create health matrix
        health_metrics = [
            'Degradation Severity',
            'Pattern Confidence',
            'Reliability Score',
            'Maintenance Urgency',
            'Failure Risk'
        ]

        health_matrix = np.zeros((len(components), len(health_metrics)))

        # Fill matrix with normalized health scores
        for i, component in enumerate(components):
            # Degradation severity
            max_severity = 0
            avg_confidence = 0
            pattern_count = 0

            for pattern in patterns:
                if component in pattern.affected_components:
                    max_severity = max(max_severity, self._severity_to_score(pattern.severity))
                    avg_confidence += pattern.confidence
                    pattern_count += 1

            if pattern_count > 0:
                avg_confidence /= pattern_count

            health_matrix[i, 0] = max_severity / 5  # Normalize to 0-1
            health_matrix[i, 1] = 1 - avg_confidence  # Invert confidence (lower is better)

            # Reliability score
            for rel_data in reliability_data:
                if rel_data.component_id == component:
                    health_matrix[i, 2] = 1 - rel_data.current_reliability
                    health_matrix[i, 4] = rel_data.failure_rate * 1000  # Scale for visibility
                    break

            # Maintenance urgency (placeholder)
            health_matrix[i, 3] = max_severity / 5  # Use severity as proxy

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=health_matrix,
                x=health_metrics,
                y=components,
                colorscale='RdYlGn_r',
                colorbar=dict(title="Health Score (Higher = Worse)"),
                text=np.round(health_matrix, 3),
                texttemplate="%{text}",
                textfont={"size": 10}
            )
        )

        fig.update_layout(
            title="MFC Component Health Status Heatmap",
            xaxis_title="Health Metrics",
            yaxis_title="Components",
            height=max(400, len(components) * 50),
            template="plotly_white"
        )

        # Save heatmap
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        heatmap_path = self.output_directory / f"component_health_heatmap_{timestamp}.html"

        pyo.plot(fig, filename=str(heatmap_path), auto_open=False)

        self.logger.info(f"Component health heatmap saved to {heatmap_path}")
        return str(heatmap_path)

    def create_failure_prediction_plot(self, patterns: List[DegradationPattern]) -> str:
        """Create failure prediction timeline plot."""

        predictions = []
        for pattern in patterns:
            if pattern.predicted_failure_time:
                predictions.append({
                    'component': pattern.affected_components[0] if pattern.affected_components else 'unknown',
                    'failure_time': pattern.predicted_failure_time,
                    'confidence': pattern.confidence,
                    'severity': pattern.severity.value,
                    'degradation_type': pattern.degradation_type.value,
                    'pattern_id': pattern.pattern_id
                })

        if not predictions:
            self.logger.warning("No failure predictions to visualize")
            return ""

        df = pd.DataFrame(predictions)

        # Create timeline plot
        fig = px.scatter(
            df,
            x='failure_time',
            y='component',
            size='confidence',
            color='severity',
            hover_data=['degradation_type', 'pattern_id'],
            title="MFC Component Failure Predictions",
            color_discrete_map=self.severity_colors
        )

        # Add vertical line for current time
        fig.add_vline(
            x=datetime.now(),
            line_dash="dash",
            line_color="red",
            annotation_text="Current Time"
        )

        fig.update_layout(
            xaxis_title="Predicted Failure Time",
            yaxis_title="Component",
            height=500,
            template="plotly_white"
        )

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.output_directory / f"failure_predictions_{timestamp}.html"

        pyo.plot(fig, filename=str(plot_path), auto_open=False)

        self.logger.info(f"Failure prediction plot saved to {plot_path}")
        return str(plot_path)

    def generate_stability_report(self,
                                patterns: List[DegradationPattern],
                                reliability_data: List[ComponentReliability],
                                maintenance_tasks: List[MaintenanceTask],
                                include_plots: bool = True) -> str:
        """Generate comprehensive stability report."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_data = {
            'generated_at': datetime.now().isoformat(),
            'report_title': 'MFC Long-term Stability Analysis Report',
            'executive_summary': self._generate_executive_summary(
                patterns, reliability_data, maintenance_tasks
            ),
            'degradation_analysis': self._analyze_degradation_patterns(patterns),
            'reliability_analysis': self._analyze_reliability_data(reliability_data),
            'maintenance_analysis': self._analyze_maintenance_schedule(maintenance_tasks),
            'recommendations': self._generate_recommendations(
                patterns, reliability_data, maintenance_tasks
            ),
            'visualizations': []
        }

        # Generate visualizations if requested
        if include_plots:
            try:
                dashboard_path = self.create_degradation_dashboard(patterns)
                if dashboard_path:
                    report_data['visualizations'].append({
                        'type': 'degradation_dashboard',
                        'path': dashboard_path,
                        'description': 'Interactive degradation pattern dashboard'
                    })

                reliability_path = self.create_reliability_trends_plot(reliability_data)
                if reliability_path:
                    report_data['visualizations'].append({
                        'type': 'reliability_trends',
                        'path': reliability_path,
                        'description': 'Component reliability trends over time'
                    })

                maintenance_path = self.create_maintenance_schedule_chart(maintenance_tasks)
                if maintenance_path:
                    report_data['visualizations'].append({
                        'type': 'maintenance_schedule',
                        'path': maintenance_path,
                        'description': 'Maintenance schedule Gantt chart'
                    })

                heatmap_path = self.create_component_health_heatmap(patterns, reliability_data)
                if heatmap_path:
                    report_data['visualizations'].append({
                        'type': 'health_heatmap',
                        'path': heatmap_path,
                        'description': 'Component health status heatmap'
                    })

                prediction_path = self.create_failure_prediction_plot(patterns)
                if prediction_path:
                    report_data['visualizations'].append({
                        'type': 'failure_predictions',
                        'path': prediction_path,
                        'description': 'Component failure prediction timeline'
                    })

            except Exception as e:
                self.logger.warning(f"Error generating visualizations: {e}")

        # Save report
        report_path = self.output_directory / f"stability_report_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        self.logger.info(f"Stability report generated: {report_path}")
        return str(report_path)

    def _severity_to_score(self, severity: DegradationSeverity) -> int:
        """Convert severity enum to numeric score."""
        severity_scores = {
            DegradationSeverity.MINIMAL: 1,
            DegradationSeverity.LOW: 2,
            DegradationSeverity.MODERATE: 3,
            DegradationSeverity.HIGH: 4,
            DegradationSeverity.CRITICAL: 5,
            DegradationSeverity.FAILURE: 6
        }
        return severity_scores.get(severity, 0)

    def _generate_executive_summary(self,
                                  patterns: List[DegradationPattern],
                                  reliability_data: List[ComponentReliability],
                                  maintenance_tasks: List[MaintenanceTask]) -> Dict[str, Any]:
        """Generate executive summary for stability report."""

        # Count patterns by severity
        severity_counts = {}
        for pattern in patterns:
            severity = pattern.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        # Count maintenance tasks by priority
        priority_counts = {}
        for task in maintenance_tasks:
            priority = task.priority.value
            priority_counts[priority] = priority_counts.get(priority, 0) + 1

        # Calculate average reliability
        avg_reliability = 0
        if reliability_data:
            avg_reliability = np.mean([r.current_reliability for r in reliability_data])

        # Identify critical issues
        critical_patterns = [p for p in patterns if p.severity in [DegradationSeverity.CRITICAL, DegradationSeverity.FAILURE]]
        emergency_tasks = [t for t in maintenance_tasks if t.priority.value == 'emergency']

        return {
            'total_degradation_patterns': len(patterns),
            'critical_degradation_patterns': len(critical_patterns),
            'patterns_by_severity': severity_counts,
            'average_system_reliability': avg_reliability,
            'total_maintenance_tasks': len(maintenance_tasks),
            'emergency_maintenance_tasks': len(emergency_tasks),
            'tasks_by_priority': priority_counts,
            'components_analyzed': len(set(r.component_id for r in reliability_data)),
            'key_concerns': [
                f"{len(critical_patterns)} critical degradation patterns detected" if critical_patterns else None,
                f"{len(emergency_tasks)} emergency maintenance tasks scheduled" if emergency_tasks else None,
                f"Average system reliability: {avg_reliability:.2%}" if avg_reliability < 0.8 else None
            ]
        }

    def _analyze_degradation_patterns(self, patterns: List[DegradationPattern]) -> Dict[str, Any]:
        """Analyze degradation patterns for report."""

        if not patterns:
            return {'error': 'No degradation patterns to analyze'}

        # Group by type
        type_counts = {}
        for pattern in patterns:
            deg_type = pattern.degradation_type.value
            type_counts[deg_type] = type_counts.get(deg_type, 0) + 1

        # Group by component
        component_counts = {}
        for pattern in patterns:
            for component in pattern.affected_components:
                component_counts[component] = component_counts.get(component, 0) + 1

        # Confidence statistics
        confidences = [p.confidence for p in patterns]

        return {
            'total_patterns': len(patterns),
            'patterns_by_type': type_counts,
            'patterns_by_component': component_counts,
            'confidence_statistics': {
                'mean': np.mean(confidences),
                'median': np.median(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            },
            'most_common_degradation_type': max(type_counts, key=type_counts.get) if type_counts else None,
            'most_affected_component': max(component_counts, key=component_counts.get) if component_counts else None
        }

    def _analyze_reliability_data(self, reliability_data: List[ComponentReliability]) -> Dict[str, Any]:
        """Analyze reliability data for report."""

        if not reliability_data:
            return {'error': 'No reliability data to analyze'}

        reliabilities = [r.current_reliability for r in reliability_data]
        mtbf_values = [r.mtbf_hours for r in reliability_data]
        failure_rates = [r.failure_rate for r in reliability_data]

        return {
            'components_analyzed': len(reliability_data),
            'reliability_statistics': {
                'mean': np.mean(reliabilities),
                'median': np.median(reliabilities),
                'min': np.min(reliabilities),
                'max': np.max(reliabilities)
            },
            'mtbf_statistics': {
                'mean_hours': np.mean(mtbf_values),
                'median_hours': np.median(mtbf_values),
                'min_hours': np.min(mtbf_values),
                'max_hours': np.max(mtbf_values)
            },
            'failure_rate_statistics': {
                'mean': np.mean(failure_rates),
                'median': np.median(failure_rates),
                'min': np.min(failure_rates),
                'max': np.max(failure_rates)
            },
            'low_reliability_components': [
                r.component_id for r in reliability_data
                if r.current_reliability < 0.8
            ]
        }

    def _analyze_maintenance_schedule(self, maintenance_tasks: List[MaintenanceTask]) -> Dict[str, Any]:
        """Analyze maintenance schedule for report."""

        if not maintenance_tasks:
            return {'error': 'No maintenance tasks to analyze'}

        # Calculate total cost and downtime
        total_cost = sum(task.cost_estimate for task in maintenance_tasks)
        total_downtime = sum(task.downtime_impact for task in maintenance_tasks)

        # Group by type and priority
        type_counts = {}
        priority_counts = {}

        for task in maintenance_tasks:
            task_type = task.maintenance_type.value
            priority = task.priority.value

            type_counts[task_type] = type_counts.get(task_type, 0) + 1
            priority_counts[priority] = priority_counts.get(priority, 0) + 1

        # Timeline analysis
        now = datetime.now()
        overdue_tasks = [t for t in maintenance_tasks if t.scheduled_date < now]
        upcoming_tasks = [
            t for t in maintenance_tasks
            if now <= t.scheduled_date <= now + timedelta(days=30)
        ]

        return {
            'total_tasks': len(maintenance_tasks),
            'total_estimated_cost': total_cost,
            'total_estimated_downtime_hours': total_downtime,
            'tasks_by_type': type_counts,
            'tasks_by_priority': priority_counts,
            'overdue_tasks': len(overdue_tasks),
            'upcoming_tasks_30_days': len(upcoming_tasks),
            'average_task_cost': total_cost / len(maintenance_tasks) if maintenance_tasks else 0,
            'average_downtime_per_task': total_downtime / len(maintenance_tasks) if maintenance_tasks else 0
        }

    def _generate_recommendations(self,
                                patterns: List[DegradationPattern],
                                reliability_data: List[ComponentReliability],
                                maintenance_tasks: List[MaintenanceTask]) -> List[str]:
        """Generate actionable recommendations."""

        recommendations = []

        # Degradation-based recommendations
        critical_patterns = [p for p in patterns if p.severity in [DegradationSeverity.CRITICAL, DegradationSeverity.FAILURE]]
        if critical_patterns:
            recommendations.append(
                f"URGENT: Address {len(critical_patterns)} critical degradation patterns immediately"
            )

        # Component-specific recommendations
        component_issues = {}
        for pattern in patterns:
            for component in pattern.affected_components:
                if component not in component_issues:
                    component_issues[component] = []
                component_issues[component].append(pattern.degradation_type.value)

        for component, issues in component_issues.items():
            if len(issues) > 2:
                recommendations.append(
                    f"Component '{component}' shows multiple degradation patterns - consider comprehensive inspection"
                )

        # Reliability-based recommendations
        low_reliability_components = [
            r.component_id for r in reliability_data
            if r.current_reliability < 0.8
        ]

        if low_reliability_components:
            recommendations.append(
                f"Monitor components with low reliability: {', '.join(low_reliability_components)}"
            )

        # Maintenance-based recommendations
        overdue_tasks = [
            t for t in maintenance_tasks
            if t.scheduled_date < datetime.now()
        ]

        if overdue_tasks:
            recommendations.append(
                f"Complete {len(overdue_tasks)} overdue maintenance tasks immediately"
            )

        # General recommendations
        if len(patterns) > 10:
            recommendations.append("Consider implementing more frequent monitoring due to high number of degradation patterns")

        if not recommendations:
            recommendations.append("System appears stable - continue regular monitoring")

        return recommendations
