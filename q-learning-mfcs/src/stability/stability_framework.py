"""
Long-term Stability Study Framework for MFC Systems

Comprehensive framework for analyzing long-term system reliability, detecting
degradation patterns, and predicting maintenance requirements.

Features:
- Multi-scale stability analysis (hours to years)
- Component degradation tracking
- Failure prediction algorithms
- Maintenance scheduling optimization
- Performance drift detection
- Reliability metrics calculation

Author: MFC Development Team
Date: 2025-07-28
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import logging

from config.statistical_analysis import StatisticalAnalyzer

class StabilityTimeScale(Enum):
    """Time scales for stability analysis."""
    SHORT_TERM = "short_term"      # Hours to days
    MEDIUM_TERM = "medium_term"    # Days to weeks
    LONG_TERM = "long_term"        # Weeks to months
    ULTRA_LONG = "ultra_long"      # Months to years
class ComponentType(Enum):
    """MFC system components."""
    MEMBRANE = "membrane"
    ANODE = "anode"
    CATHODE = "cathode"
    BIOFILM = "biofilm"
    ELECTRONICS = "electronics"
    PUMPS = "pumps"
    SENSORS = "sensors"
    OVERALL = "overall"

class DegradationMode(Enum):
    """Types of component degradation."""
    FOULING = "fouling"
    CORROSION = "corrosion"
    MECHANICAL_WEAR = "mechanical_wear"
    THERMAL_STRESS = "thermal_stress"
    CHEMICAL_ATTACK = "chemical_attack"
    BIOLOGICAL_DAMAGE = "biological_damage"
    ELECTRICAL_AGING = "electrical_aging"
@dataclass
class StabilityMetrics:
    """Container for stability analysis results."""
    
    # Performance metrics
    power_stability: float          # Coefficient of variation
    efficiency_drift: float         # %/day change
    voltage_degradation: float      # V/day
    current_fluctuation: float      # Coefficient of variation
    
    # Reliability metrics  
    mtbf_hours: float              # Mean time between failures
    availability: float            # Fraction of uptime
    failure_rate: float            # Failures per hour
    
    # Degradation metrics
    membrane_resistance_increase: float    # %/day
    biofilm_thickness_rate: float         # μm/day
    electrode_degradation: float          # %/day
    
    # Maintenance metrics
    cleaning_frequency_days: float
    component_lifetime_days: float
    maintenance_cost_per_day: float
    
    # Statistical measures
    performance_variance: float
    trend_slope: float
    seasonal_component: float
    residual_variance: float
@dataclass
class ComponentHealth:
    """Health status of individual components."""
    
    component: ComponentType
    health_score: float             # 0-100 scale
    remaining_lifetime: float       # Days
    degradation_rate: float         # %/day
    last_maintenance: datetime
    next_maintenance: datetime
    failure_probability: float      # 0-1 probability in next 30 days
    degradation_modes: List[DegradationMode]
    performance_history: List[float] = field(default_factory=list)
    
class StabilityAnalyzer:
    """
    Core stability analysis engine for MFC systems.
    
    Provides comprehensive long-term stability analysis including:
    - Performance drift detection
    - Component degradation tracking
    - Failure prediction
    - Maintenance optimization
    - Reliability metrics calculation
    """
    
    def __init__(self, data_directory: Optional[Path] = None):
        """
        Initialize stability analyzer.
        
        Args:
            data_directory: Directory for storing analysis results
        """
        self.data_dir = data_directory or Path("stability_analysis")
        self.data_dir.mkdir(exist_ok=True)
        
        # Analysis history
        self.stability_history: List[StabilityMetrics] = []
        self.component_health: Dict[ComponentType, ComponentHealth] = {}
        
        # Configuration
        self.analysis_config = {
            "sampling_interval_hours": 1.0,
            "stability_window_hours": 168,  # 1 week
            "degradation_threshold": 0.05,   # 5% performance loss
            "failure_threshold": 0.8,        # 80% degradation
            "maintenance_buffer_days": 7,    # Schedule maintenance 7 days early
        }
        
        # Initialize statistical analyzer
        self.stats_analyzer = StatisticalAnalyzer()
        
        # Initialize component health tracking
        self._initialize_component_health()
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
    def _initialize_component_health(self):
        """Initialize component health tracking."""
        for component in ComponentType:
            self.component_health[component] = ComponentHealth(
                component=component,
                health_score=100.0,
                remaining_lifetime=365.0,  # 1 year default
                degradation_rate=0.0,
                last_maintenance=datetime.now(),
                next_maintenance=datetime.now() + timedelta(days=30),
                failure_probability=0.0,
                degradation_modes=[]
            )
    
    def analyze_stability(self, simulation_data: Dict[str, Any], 
                         time_scale: StabilityTimeScale = StabilityTimeScale.LONG_TERM) -> StabilityMetrics:
        """
        Perform comprehensive stability analysis.
        
        Args:
            simulation_data: Time series data from MFC simulation
            time_scale: Analysis time scale
            
        Returns:
            StabilityMetrics object with analysis results
        """
        self.logger.info(f"Starting stability analysis for {time_scale.value} time scale")
        
        # Extract time series data
        time_points = simulation_data.get('time', [])
        power_data = simulation_data.get('power', [])
        voltage_data = simulation_data.get('voltage', [])
        current_data = simulation_data.get('current', [])
        efficiency_data = simulation_data.get('efficiency', [])
        
        # Calculate performance stability metrics
        power_stability = np.std(power_data) / np.mean(power_data) if power_data else 0.0
        current_fluctuation = np.std(current_data) / np.mean(current_data) if current_data else 0.0
        
        # Calculate degradation trends
        efficiency_drift = self._calculate_drift_rate(time_points, efficiency_data)
        voltage_degradation = self._calculate_drift_rate(time_points, voltage_data)
        
        # Calculate reliability metrics
        failure_times = self._detect_failures(simulation_data)
        mtbf_hours = self._calculate_mtbf(failure_times, max(time_points) if time_points else 1.0)
        availability = self._calculate_availability(simulation_data)
        failure_rate = len(failure_times) / max(time_points) if time_points else 0.0
        
        # Calculate component-specific degradation
        membrane_resistance_increase = self._calculate_membrane_degradation(simulation_data)
        biofilm_thickness_rate = self._calculate_biofilm_growth_rate(simulation_data)
        electrode_degradation = self._calculate_electrode_degradation(simulation_data)
        
        # Calculate maintenance metrics
        cleaning_frequency = self._estimate_cleaning_frequency(simulation_data)
        component_lifetime = self._estimate_component_lifetime(simulation_data)
        maintenance_cost = self._estimate_maintenance_cost(simulation_data)
        
        # Statistical analysis
        performance_variance = np.var(power_data) if power_data else 0.0
        trend_slope = efficiency_drift
        seasonal_component = self._detect_seasonal_patterns(time_points, power_data)
        residual_variance = self._calculate_residual_variance(time_points, power_data)
        
        # Create stability metrics object
        metrics = StabilityMetrics(
            power_stability=power_stability,
            efficiency_drift=efficiency_drift,
            voltage_degradation=voltage_degradation,
            current_fluctuation=current_fluctuation,
            mtbf_hours=mtbf_hours,
            availability=availability,
            failure_rate=failure_rate,
            membrane_resistance_increase=membrane_resistance_increase,
            biofilm_thickness_rate=biofilm_thickness_rate,
            electrode_degradation=electrode_degradation,
            cleaning_frequency_days=cleaning_frequency,
            component_lifetime_days=component_lifetime,
            maintenance_cost_per_day=maintenance_cost,
            performance_variance=performance_variance,
            trend_slope=trend_slope,
            seasonal_component=seasonal_component,
            residual_variance=residual_variance
        )
        
        # Store results
        self.stability_history.append(metrics)
        
        # Update component health
        self._update_component_health(simulation_data, metrics)
        
        self.logger.info("Stability analysis completed")
        return metrics
    
    def _calculate_drift_rate(self, time_points: List[float], 
                             values: List[float]) -> float:
        """Calculate performance drift rate."""
        if len(time_points) < 2 or len(values) < 2:
            return 0.0
        
        try:
            # Linear regression to find trend
            coefficients = np.polyfit(time_points, values, 1)
            drift_rate = coefficients[0]  # Slope
            
            # Convert to percentage per day
            if len(time_points) > 0:
                time_span_hours = max(time_points) - min(time_points)
                time_span_days = time_span_hours / 24.0
                mean_value = np.mean(values)
                
                if mean_value != 0 and time_span_days > 0:
                    drift_rate_pct_per_day = (drift_rate / mean_value) * 100 / time_span_days
                    return drift_rate_pct_per_day
            
            return 0.0
        except Exception as e:
            self.logger.warning(f"Error calculating drift rate: {e}")
            return 0.0
    
    def _detect_failures(self, simulation_data: Dict[str, Any]) -> List[float]:
        """Detect system failures in time series data."""
        failures = []
        
        # Define failure criteria
        power_data = simulation_data.get('power', [])
        voltage_data = simulation_data.get('voltage', [])
        time_points = simulation_data.get('time', [])
        
        if not power_data or not time_points:
            return failures
        
        # Calculate thresholds
        mean_power = np.mean(power_data)
        min_power_threshold = mean_power * 0.1  # 10% of mean power
        min_voltage_threshold = 0.05  # 50 mV minimum voltage
        
        # Detect failures
        for i, (time, power, voltage) in enumerate(zip(time_points, power_data, 
                                                      voltage_data if voltage_data else [0]*len(power_data))):
            if power < min_power_threshold or voltage < min_voltage_threshold:
                failures.append(time)
        
        return failures
    
    def _calculate_mtbf(self, failure_times: List[float], total_time: float) -> float:
        """Calculate Mean Time Between Failures."""
        if len(failure_times) <= 1:
            return total_time  # No failures or only one failure
        
        # Calculate intervals between failures
        intervals = []
        for i in range(1, len(failure_times)):
            intervals.append(failure_times[i] - failure_times[i-1])
        
        return np.mean(intervals) if intervals else total_time
    
    def _calculate_availability(self, simulation_data: Dict[str, Any]) -> float:
        """Calculate system availability (uptime fraction)."""
        # Simplified availability calculation
        failure_times = self._detect_failures(simulation_data)
        time_points = simulation_data.get('time', [])
        
        if not time_points:
            return 1.0
        
        total_time = max(time_points) - min(time_points)
        downtime = len(failure_times) * 1.0  # Assume 1 hour downtime per failure
        
        if total_time <= 0:
            return 1.0
        
        availability = max(0.0, (total_time - downtime) / total_time)
        return min(1.0, availability)
    
    def _calculate_membrane_degradation(self, simulation_data: Dict[str, Any]) -> float:
        """Calculate membrane resistance increase rate."""
        # Simplified membrane degradation model
        time_points = simulation_data.get('time', [])
        
        if not time_points:
            return 0.0
        
        # Assume resistance increases with fouling over time
        time_span_days = (max(time_points) - min(time_points)) / 24.0
        
        if time_span_days <= 0:
            return 0.0
        
        # Typical membrane resistance increase: 1-5% per day
        base_degradation_rate = 0.02  # 2% per day baseline
        
        # Modulate based on operating conditions
        # TODO: Use actual membrane model data when available
        
        return base_degradation_rate
    
    def _calculate_biofilm_growth_rate(self, simulation_data: Dict[str, Any]) -> float:
        """Calculate biofilm thickness growth rate."""
        biofilm_data = simulation_data.get('biofilm_thickness', [])
        time_points = simulation_data.get('time', [])
        
        if len(biofilm_data) < 2 or len(time_points) < 2:
            return 0.1  # Default 0.1 μm/day
        
        # Calculate growth rate from data
        thickness_change = max(biofilm_data) - min(biofilm_data)
        time_span_days = (max(time_points) - min(time_points)) / 24.0
        
        if time_span_days <= 0:
            return 0.1
        
        growth_rate = thickness_change / time_span_days
        return max(0.0, growth_rate)  # Ensure non-negative
    
    def _calculate_electrode_degradation(self, simulation_data: Dict[str, Any]) -> float:
        """Calculate electrode degradation rate."""
        # Simplified electrode degradation model
        # Based on current density and operating time
        
        current_data = simulation_data.get('current', [])
        time_points = simulation_data.get('time', [])
        
        if not current_data or not time_points:
            return 0.01  # Default 1% per day
        
        # Higher current densities lead to faster degradation
        mean_current = np.mean(current_data)
        time_span_days = (max(time_points) - min(time_points)) / 24.0
        
        if time_span_days <= 0:
            return 0.01
        
        # Empirical degradation model
        base_rate = 0.005  # 0.5% per day baseline
        current_factor = mean_current / 5.0  # Normalize to typical 5 A/m²
        
        degradation_rate = base_rate * (1 + current_factor)
        return min(0.1, degradation_rate)  # Cap at 10% per day
    
    def _estimate_cleaning_frequency(self, simulation_data: Dict[str, Any]) -> float:
        """Estimate required cleaning frequency."""
        # Based on fouling rate and performance degradation
        membrane_degradation = self._calculate_membrane_degradation(simulation_data)
        _ = self._calculate_biofilm_growth_rate(simulation_data)  # Used for side effects
        
        # When degradation reaches 5%, cleaning is needed
        degradation_threshold = 0.05  # 5%
        
        cleaning_frequency = degradation_threshold / max(0.001, membrane_degradation)
        return max(1.0, min(365.0, cleaning_frequency))  # Between 1 day and 1 year
    
    def _estimate_component_lifetime(self, simulation_data: Dict[str, Any]) -> float:
        """Estimate component lifetime until replacement."""
        # Based on degradation rates and failure threshold
        electrode_degradation = self._calculate_electrode_degradation(simulation_data)
        membrane_degradation = self._calculate_membrane_degradation(simulation_data)
        
        # Component fails when degradation reaches 80%
        failure_threshold = 0.8  # 80%
        
        # Estimate time to failure for critical components
        electrode_lifetime = failure_threshold / max(0.001, electrode_degradation)
        membrane_lifetime = failure_threshold / max(0.001, membrane_degradation)
        
        # System lifetime is limited by shortest component lifetime
        system_lifetime = min(electrode_lifetime, membrane_lifetime)
        return max(30.0, min(3650.0, system_lifetime))  # Between 30 days and 10 years
    
    def _estimate_maintenance_cost(self, simulation_data: Dict[str, Any]) -> float:
        """Estimate daily maintenance cost."""
        # Simplified cost model
        cleaning_frequency = self._estimate_cleaning_frequency(simulation_data)
        component_lifetime = self._estimate_component_lifetime(simulation_data)
        
        # Cost estimates (example values)
        cleaning_cost = 100.0  # $100 per cleaning
        component_replacement_cost = 1000.0  # $1000 per component set
        
        daily_cleaning_cost = cleaning_cost / cleaning_frequency
        daily_replacement_cost = component_replacement_cost / component_lifetime
        
        total_daily_cost = daily_cleaning_cost + daily_replacement_cost
        return max(0.1, total_daily_cost)
    
    def _detect_seasonal_patterns(self, time_points: List[float], 
                                 values: List[float]) -> float:
        """Detect seasonal patterns in performance data."""
        if len(time_points) < 24 or len(values) < 24:  # Need at least 24 hours
            return 0.0
        
        try:
            # Convert to hourly sampling if needed
            df = pd.DataFrame({'time': time_points, 'value': values})
            df = df.set_index('time').resample('1H').mean().dropna()
            
            if len(df) < 24:
                return 0.0
            
            # Calculate seasonal component (simplified)
            values_array = df['value'].values
            
            # Detect daily patterns
            if len(values_array) >= 24:
                daily_pattern = np.zeros(24)
                for hour in range(24):
                    hour_values = values_array[hour::24]
                    if len(hour_values) > 0:
                        daily_pattern[hour] = np.mean(hour_values)
                
                seasonal_strength = np.std(daily_pattern) / np.mean(values_array)
                return min(1.0, seasonal_strength)
            
            return 0.0
        except Exception as e:
            self.logger.warning(f"Error detecting seasonal patterns: {e}")
            return 0.0
    
    def _calculate_residual_variance(self, time_points: List[float], 
                                   values: List[float]) -> float:
        """Calculate residual variance after removing trend."""
        if len(time_points) < 3 or len(values) < 3:
            return 0.0
        
        try:
            # Remove linear trend
            coefficients = np.polyfit(time_points, values, 1)
            trend = np.polyval(coefficients, time_points)
            residuals = np.array(values) - trend
            
            return np.var(residuals)
        except Exception as e:
            self.logger.warning(f"Error calculating residual variance: {e}")
            return np.var(values) if values else 0.0
    
    def _update_component_health(self, simulation_data: Dict[str, Any], 
                               metrics: StabilityMetrics):
        """Update component health status based on analysis."""
        
        # Update membrane health
        membrane_health = self.component_health[ComponentType.MEMBRANE]
        membrane_health.degradation_rate = metrics.membrane_resistance_increase
        membrane_health.health_score = max(0, 100 - (metrics.membrane_resistance_increase * 100))
        membrane_health.remaining_lifetime = metrics.cleaning_frequency_days
        
        # Update biofilm health
        biofilm_health = self.component_health[ComponentType.BIOFILM]
        biofilm_health.degradation_rate = metrics.biofilm_thickness_rate
        biofilm_health.health_score = max(0, 100 - (metrics.biofilm_thickness_rate * 10))
        
        # Update electrode health
        electrode_health = self.component_health[ComponentType.ANODE]
        electrode_health.degradation_rate = metrics.electrode_degradation
        electrode_health.health_score = max(0, 100 - (metrics.electrode_degradation * 100))
        electrode_health.remaining_lifetime = metrics.component_lifetime_days
        
        # Update overall system health
        overall_health = self.component_health[ComponentType.OVERALL]
        overall_health.health_score = min(
            membrane_health.health_score,
            biofilm_health.health_score,
            electrode_health.health_score
        )
        overall_health.remaining_lifetime = min(
            membrane_health.remaining_lifetime,
            electrode_health.remaining_lifetime
        )
        
    def generate_stability_report(self, metrics: StabilityMetrics, 
                                output_file: Optional[Path] = None) -> str:
        """Generate comprehensive stability analysis report."""
        
        report_lines = []
        report_lines.append("# MFC Long-term Stability Analysis Report")
        report_lines.append(f"Generated: {datetime.now().isoformat()}")
        report_lines.append("")
        
        # Executive summary
        report_lines.append("## Executive Summary")
        report_lines.append(f"- System Availability: **{metrics.availability*100:.1f}%**")
        report_lines.append(f"- Mean Time Between Failures: **{metrics.mtbf_hours:.1f} hours**")
        report_lines.append(f"- Power Stability: **{metrics.power_stability*100:.2f}% variation**")
        report_lines.append(f"- Estimated Component Lifetime: **{metrics.component_lifetime_days:.0f} days**")
        report_lines.append("")
        
        # Performance metrics
        report_lines.append("## Performance Stability")
        report_lines.append(f"- Power Output Stability: {metrics.power_stability*100:.2f}% coefficient of variation")
        report_lines.append(f"- Efficiency Drift: {metrics.efficiency_drift:.3f}%/day")
        report_lines.append(f"- Voltage Degradation: {metrics.voltage_degradation*1000:.2f} mV/day")
        report_lines.append(f"- Current Fluctuation: {metrics.current_fluctuation*100:.2f}% coefficient of variation")
        report_lines.append("")
        
        # Reliability metrics
        report_lines.append("## Reliability Analysis")
        report_lines.append(f"- Mean Time Between Failures: {metrics.mtbf_hours:.1f} hours")
        report_lines.append(f"- System Availability: {metrics.availability*100:.1f}%")
        report_lines.append(f"- Failure Rate: {metrics.failure_rate*1000:.2f} failures per 1000 hours")
        report_lines.append("")
        
        # Component degradation
        report_lines.append("## Component Degradation")
        report_lines.append(f"- Membrane Resistance Increase: {metrics.membrane_resistance_increase*100:.2f}%/day")
        report_lines.append(f"- Biofilm Growth Rate: {metrics.biofilm_thickness_rate:.2f} μm/day")
        report_lines.append(f"- Electrode Degradation: {metrics.electrode_degradation*100:.2f}%/day")
        report_lines.append("")
        
        # Maintenance schedule
        report_lines.append("## Maintenance Requirements")
        report_lines.append(f"- Cleaning Frequency: Every {metrics.cleaning_frequency_days:.1f} days")
        report_lines.append(f"- Component Replacement: Every {metrics.component_lifetime_days:.0f} days")
        report_lines.append(f"- Daily Maintenance Cost: ${metrics.maintenance_cost_per_day:.2f}")
        report_lines.append("")
        
        # Component health summary
        report_lines.append("## Component Health Status")
        for component_type, health in self.component_health.items():
            report_lines.append(f"- **{component_type.value.title()}**: {health.health_score:.1f}% health, "
                              f"{health.remaining_lifetime:.0f} days remaining")
        report_lines.append("")
        
        # Recommendations
        report_lines.append("## Recommendations")
        
        if metrics.power_stability > 0.1:
            report_lines.append("- ⚠️ High power variability detected - check control system stability")
        
        if metrics.efficiency_drift < -0.01:
            report_lines.append("- ⚠️ Efficiency declining - consider system cleaning or optimization")
        
        if metrics.availability < 0.95:
            report_lines.append("- ⚠️ Low availability - investigate failure causes and improve reliability")
        
        if metrics.cleaning_frequency_days < 7:
            report_lines.append("- ⚠️ Frequent cleaning required - consider pretreatment or operating condition optimization")
        
        if not any(metrics.power_stability > 0.1 or metrics.efficiency_drift < -0.01 or 
                  metrics.availability < 0.95 or metrics.cleaning_frequency_days < 7):
            report_lines.append("- ✅ System operating within acceptable stability parameters")
        report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        # Save to file if specified
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Stability report saved to {output_file}")
        
        return report_text
    
    def predict_failures(self, horizon_days: int = 30) -> Dict[ComponentType, float]:
        """
        Predict component failure probabilities.
        
        Args:
            horizon_days: Prediction horizon in days
            
        Returns:
            Dictionary mapping components to failure probabilities
        """
        predictions = {}
        
        for component_type, health in self.component_health.items():
            # Simple failure prediction based on degradation rate and current health
            current_health = health.health_score / 100.0  # Convert to 0-1 scale
            degradation_rate = health.degradation_rate
            
            # Predict health after horizon_days
            future_health = current_health - (degradation_rate * horizon_days / 100.0)
            
            # Convert to failure probability (sigmoid function)
            failure_threshold = 0.2  # 20% health remaining
            if future_health <= failure_threshold:
                failure_prob = 1.0
            else:
                # Sigmoid curve for failure probability
                x = (future_health - failure_threshold) * 10
                failure_prob = max(0.0, min(1.0, 1.0 / (1.0 + np.exp(x))))
            
            predictions[component_type] = failure_prob
        
        return predictions
    
    def get_maintenance_schedule(self, horizon_days: int = 90) -> List[Dict[str, Any]]:
        """
        Generate optimized maintenance schedule.
        
        Args:
            horizon_days: Planning horizon in days
            
        Returns:
            List of maintenance tasks with dates and priorities
        """
        schedule = []
        current_date = datetime.now()
        
        # Get failure predictions
        failure_predictions = self.predict_failures(horizon_days)
        
        for component_type, health in self.component_health.items():
            if component_type == ComponentType.OVERALL:
                continue
            
            # Schedule based on remaining lifetime and failure probability
            failure_prob = failure_predictions.get(component_type, 0.0)
            
            if failure_prob > 0.7:  # High failure risk
                # Schedule immediate maintenance
                maintenance_date = current_date + timedelta(days=1)
                priority = "Critical"
            elif failure_prob > 0.3:  # Medium failure risk
                # Schedule within a week
                maintenance_date = current_date + timedelta(days=7)
                priority = "High"
            elif health.remaining_lifetime < 30:  # Low remaining lifetime
                # Schedule preventive maintenance
                maintenance_date = current_date + timedelta(days=max(1, health.remaining_lifetime * 0.8))
                priority = "Medium"
            else:
                # Schedule routine maintenance
                maintenance_date = health.next_maintenance
                priority = "Low"
            
            # Only include if within planning horizon
            if (maintenance_date - current_date).days <= horizon_days:
                schedule.append({
                    "component": component_type.value,
                    "date": maintenance_date.isoformat(),
                    "priority": priority,
                    "failure_probability": failure_prob,
                    "health_score": health.health_score,
                    "task_type": "inspection" if priority == "Low" else "maintenance",
                    "estimated_duration_hours": 2 if priority == "Low" else 8,
                    "estimated_cost": 50 if priority == "Low" else 200
                })
        
        # Sort by date and priority
        priority_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
        schedule.sort(key=lambda x: (x["date"], priority_order.get(x["priority"], 4)))
        
        return schedule
    
    def export_analysis_data(self, output_file: Path):
        """Export stability analysis data for external analysis."""
        
        export_data = {
            "analysis_metadata": {
                "generated_at": datetime.now().isoformat(),
                "analysis_config": self.analysis_config,
                "total_analyses": len(self.stability_history)
            },
            "stability_history": [
                {
                    "timestamp": datetime.now().isoformat(),  # Would need actual timestamps
                    "metrics": {
                        "power_stability": metrics.power_stability,
                        "efficiency_drift": metrics.efficiency_drift,
                        "voltage_degradation": metrics.voltage_degradation,
                        "current_fluctuation": metrics.current_fluctuation,
                        "mtbf_hours": metrics.mtbf_hours,
                        "availability": metrics.availability,
                        "failure_rate": metrics.failure_rate,
                        "membrane_resistance_increase": metrics.membrane_resistance_increase,
                        "biofilm_thickness_rate": metrics.biofilm_thickness_rate,
                        "electrode_degradation": metrics.electrode_degradation,
                        "cleaning_frequency_days": metrics.cleaning_frequency_days,
                        "component_lifetime_days": metrics.component_lifetime_days,
                        "maintenance_cost_per_day": metrics.maintenance_cost_per_day
                    }
                }
                for metrics in self.stability_history[-10:]  # Last 10 analyses
            ],
            "component_health": {
                component.value: {
                    "health_score": health.health_score,
                    "remaining_lifetime": health.remaining_lifetime,
                    "degradation_rate": health.degradation_rate,
                    "last_maintenance": health.last_maintenance.isoformat(),
                    "next_maintenance": health.next_maintenance.isoformat(),
                    "failure_probability": health.failure_probability,
                    "degradation_modes": [mode.value for mode in health.degradation_modes]
                }
                for component, health in self.component_health.items()
            },
            "maintenance_schedule": self.get_maintenance_schedule()
        }
        
        # Save to JSON file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Analysis data exported to {output_file}")
def run_long_term_stability_study(simulation_hours: int = 8760,  # 1 year
                                 output_dir: Optional[Path] = None) -> StabilityAnalyzer:
    """
    Run a comprehensive long-term stability study.
    
    Args:
        simulation_hours: Duration of stability study in hours
        output_dir: Directory for outputs
        
    Returns:
        StabilityAnalyzer with completed analysis
    """
    output_dir = output_dir or Path("stability_study_results")
    output_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting {simulation_hours}-hour stability study")
    
    # Initialize stability analyzer
    analyzer = StabilityAnalyzer(output_dir)
    
    # Run simulation (simplified - would use actual MFC model)
    # Generate synthetic data for demonstration
    time_points = np.linspace(0, simulation_hours, simulation_hours)
    
    # Simulate gradual degradation
    baseline_power = 5.0  # W
    degradation_rate = 0.0001  # Per hour
    noise_level = 0.1
    
    power_data = []
    voltage_data = []
    current_data = []
    efficiency_data = []
    
    for t in time_points:
        # Add degradation and noise
        degradation_factor = 1.0 - (degradation_rate * t)
        noise = np.random.normal(0, noise_level)
        
        power = baseline_power * degradation_factor * (1 + noise)
        voltage = 0.7 * degradation_factor * (1 + noise * 0.5)
        current = power / voltage if voltage > 0 else 0
        efficiency = 0.8 * degradation_factor * (1 + noise * 0.3)
        
        power_data.append(max(0, power))
        voltage_data.append(max(0, voltage))
        current_data.append(max(0, current))
        efficiency_data.append(max(0, min(1.0, efficiency)))
    
    # Create simulation data dictionary
    simulation_data = {
        'time': time_points.tolist(),
        'power': power_data,
        'voltage': voltage_data,
        'current': current_data,
        'efficiency': efficiency_data,
        'biofilm_thickness': (np.cumsum(np.random.exponential(0.01, len(time_points))) * 0.1).tolist()
    }
    
    # Perform stability analysis
    metrics = analyzer.analyze_stability(simulation_data, StabilityTimeScale.ULTRA_LONG)
    
    # Generate report
    report_file = output_dir / f"stability_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    analyzer.generate_stability_report(metrics, report_file)
    
    # Export analysis data
    data_file = output_dir / f"stability_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    analyzer.export_analysis_data(data_file)
    
    # Get maintenance schedule
    maintenance_schedule = analyzer.get_maintenance_schedule()
    
    logger.info(f"Stability study completed. Results saved to {output_dir}")
    logger.info(f"System availability: {metrics.availability*100:.1f}%")
    logger.info(f"MTBF: {metrics.mtbf_hours:.1f} hours")
    logger.info(f"Upcoming maintenance tasks: {len(maintenance_schedule)}")
    
    return analyzer

if __name__ == "__main__":
    # Run example stability study
    analyzer = run_long_term_stability_study(
        simulation_hours=8760,  # 1 year
        output_dir=Path("stability_analysis_results")
    )
    
    print("Long-term stability study completed!")
    print("Results available in: stability_analysis_results/")