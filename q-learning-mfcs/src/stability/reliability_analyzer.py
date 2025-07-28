"""
Reliability Metrics and Failure Analysis System

Advanced reliability engineering tools for MFC systems including:
- Failure mode and effects analysis (FMEA)
- Reliability block diagrams
- Weibull analysis for component lifetimes
- Fault tree analysis
- Reliability prediction models

Author: MFC Development Team
Date: 2025-07-28
"""
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import field
from enum import Enum
from pathlib import Path
import json
import logging
import scipy.stats as stats
from scipy.optimize import minimize

class FailureMode(Enum):
    """Types of failure modes."""
    CATASTROPHIC = "catastrophic"      # Complete system failure
    DEGRADED = "degraded"             # Performance degradation
    INTERMITTENT = "intermittent"     # Sporadic failures
    WEAR_OUT = "wear_out"             # Age-related failure
    RANDOM = "random"                 # Random failures
    INFANT_MORTALITY = "infant_mortality"  # Early life failures
class SeverityLevel(Enum):
    """Failure severity levels."""
    NEGLIGIBLE = 1
    MARGINAL = 2
    CRITICAL = 3
    CATASTROPHIC = 4

class DetectionLevel(Enum):
    """Failure detection levels."""
    VERY_HIGH = 1    # Almost certain detection
    HIGH = 2         # High detection probability
    MODERATE = 3     # Moderate detection
    LOW = 4          # Low detection probability
    VERY_LOW = 5     # Very unlikely to detect
class ComponentReliability:
    """Reliability data for a component."""
    component_id: str
    mtbf_hours: float                    # Mean time between failures
    failure_rate: float                  # Failures per hour (λ)
    availability: float                  # Fraction of uptime
    maintainability: float               # Mean time to repair (hours)
    confidence_interval: Tuple[float, float]  # 95% CI for MTBF
    weibull_shape: float = 1.0          # Weibull shape parameter (β)
    weibull_scale: float = 1000.0       # Weibull scale parameter (η)
    operating_hours: float = 0.0        # Total operating time
    failure_count: int = 0              # Number of observed failures
    last_failure: Optional[datetime] = None
class FailureModeEffect:
    """FMEA entry for a failure mode."""
    component: str
    failure_mode: str
    failure_cause: str
    local_effect: str
    system_effect: str
    detection_method: str
    occurrence_rating: int              # 1-10 scale
    severity_rating: int                # 1-10 scale  
    detection_rating: int               # 1-10 scale
    rpn: int = 0                       # Risk Priority Number
    recommended_actions: List[str] = field(default_factory=list)
    current_controls: List[str] = field(default_factory=list)
class ReliabilityPrediction:
    """Reliability prediction results."""
    system_mtbf_hours: float
    system_failure_rate: float
    mission_reliability: float          # For specific mission duration
    availability: float
    component_contributions: Dict[str, float]  # Contribution to system failure rate
    critical_components: List[str]      # Components with highest failure rates
    maintenance_requirements: Dict[str, float]  # Maintenance intervals
    confidence_bounds: Dict[str, Tuple[float, float]]
class ReliabilityAnalyzer:
    """
    Comprehensive reliability analysis engine.
    
    Provides advanced reliability engineering capabilities including:
    - Component reliability modeling
    - System reliability calculation
    - Failure mode and effects analysis (FMEA)
    - Weibull analysis for lifetime prediction
    - Fault tree analysis
    - Maintenance optimization
    """
    
    def __init__(self, data_directory: Optional[Path] = None):
        """Initialize reliability analyzer."""
        self.data_dir = data_directory or Path("reliability_analysis")
        self.data_dir.mkdir(exist_ok=True)
        
        # Component reliability data
        self.component_reliability: Dict[str, ComponentReliability] = {}
        
        # FMEA database
        self.fmea_entries: List[FailureModeEffect] = []
        
        # Historical failure data
        self.failure_history: List[Dict[str, Any]] = []
        
        # System configuration
        self.system_config = {
            "redundancy_level": 1,           # Level of redundancy
            "mission_duration_hours": 8760,  # 1 year mission
            "required_availability": 0.95,   # 95% availability target
            "maintenance_strategy": "predictive"  # preventive, corrective, predictive
        }
        
        # Initialize default component models
        self._initialize_component_models()
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def _initialize_component_models(self):
        """Initialize default reliability models for MFC components."""
        
        # Default reliability data based on typical MFC components
        default_components = {
            "membrane": ComponentReliability(
                component_id="membrane",
                mtbf_hours=2000.0,      # ~3 months
                failure_rate=5e-4,      # 0.5 failures per 1000 hours
                availability=0.95,
                maintainability=8.0,    # 8 hours to replace
                confidence_interval=(1500.0, 2500.0),
                weibull_shape=2.0,      # Wear-out failure mode
                weibull_scale=2200.0
            ),
            "anode": ComponentReliability(
                component_id="anode",
                mtbf_hours=5000.0,      # ~7 months
                failure_rate=2e-4,
                availability=0.98,
                maintainability=4.0,
                confidence_interval=(4000.0, 6000.0),
                weibull_shape=1.5,
                weibull_scale=5500.0
            ),
            "cathode": ComponentReliability(
                component_id="cathode",
                mtbf_hours=4000.0,      # ~5.5 months
                failure_rate=2.5e-4,
                availability=0.97,
                maintainability=4.0,
                confidence_interval=(3200.0, 4800.0),
                weibull_shape=1.8,
                weibull_scale=4400.0
            ),
            "pump": ComponentReliability(
                component_id="pump", 
                mtbf_hours=8000.0,      # ~11 months
                failure_rate=1.25e-4,
                availability=0.99,
                maintainability=2.0,
                confidence_interval=(6500.0, 9500.0),
                weibull_shape=1.2,      # Random failures dominant
                weibull_scale=8500.0
            ),
            "electronics": ComponentReliability(
                component_id="electronics",
                mtbf_hours=15000.0,     # ~1.7 years
                failure_rate=6.7e-5,
                availability=0.995,
                maintainability=1.0,
                confidence_interval=(12000.0, 18000.0),
                weibull_shape=1.0,      # Exponential (constant failure rate)
                weibull_scale=15000.0
            ),
            "sensors": ComponentReliability(
                component_id="sensors",
                mtbf_hours=10000.0,     # ~1.1 years
                failure_rate=1e-4,
                availability=0.99,
                maintainability=0.5,    # Quick replacement
                confidence_interval=(8000.0, 12000.0),
                weibull_shape=1.1,
                weibull_scale=10500.0
            )
        }
        
        self.component_reliability = default_components
        
        # Initialize FMEA entries
        self._initialize_fmea_database()
    
    def _initialize_fmea_database(self):
        """Initialize FMEA database with common failure modes."""
        
        fmea_data = [
            FailureModeEffect(
                component="membrane",
                failure_mode="fouling",
                failure_cause="biofilm accumulation, chemical precipitation",
                local_effect="increased resistance, reduced ion transport",
                system_effect="power degradation, efficiency loss",
                detection_method="resistance monitoring, performance tracking",
                occurrence_rating=6,
                severity_rating=4,
                detection_rating=3,
                current_controls=["regular cleaning", "performance monitoring"],
                recommended_actions=["optimize cleaning schedule", "improve pretreatment"]
            ),
            FailureModeEffect(
                component="membrane",
                failure_mode="cracking",
                failure_cause="thermal stress, mechanical stress, aging",
                local_effect="loss of selectivity, leakage",
                system_effect="contamination, performance loss, safety risk",
                detection_method="pressure testing, visual inspection",
                occurrence_rating=3,
                severity_rating=8,
                detection_rating=5,
                current_controls=["temperature control", "pressure limits"],
                recommended_actions=["improve stress analysis", "enhanced monitoring"]
            ),
            FailureModeEffect(
                component="anode",
                failure_mode="corrosion",
                failure_cause="electrochemical attack, pH extremes",
                local_effect="surface degradation, conductivity loss",
                system_effect="power loss, increased overpotential",
                detection_method="visual inspection, resistance measurement",
                occurrence_rating=5,
                severity_rating=5,
                detection_rating=4,
                current_controls=["pH control", "material selection"],
                recommended_actions=["protective coatings", "improved pH control"]
            ),
            FailureModeEffect(
                component="pump",
                failure_mode="seal_failure",
                failure_cause="wear, chemical attack, thermal cycling",
                local_effect="leakage, loss of flow",
                system_effect="contamination, flow disruption",
                detection_method="leak detection, flow monitoring",
                occurrence_rating=4,
                severity_rating=6,
                detection_rating=2,
                current_controls=["seal material selection", "regular inspection"],
                recommended_actions=["upgrade seal materials", "predictive maintenance"]
            ),
            FailureModeEffect(
                component="electronics",
                failure_mode="component_drift",
                failure_cause="aging, temperature, humidity",
                local_effect="calibration drift, noise increase",
                system_effect="control accuracy loss, measurement errors",
                detection_method="calibration checks, performance monitoring",
                occurrence_rating=3,
                severity_rating=4,
                detection_rating=3,
                current_controls=["environmental control", "regular calibration"],
                recommended_actions=["improve environmental stability", "auto-calibration"]
            )
        ]
        
        # Calculate RPN for each entry
        for entry in fmea_data:
            entry.rpn = entry.occurrence_rating * entry.severity_rating * entry.detection_rating
        
        self.fmea_entries = fmea_data
    
    def calculate_system_reliability(self, mission_duration_hours: float) -> ReliabilityPrediction:
        """
        Calculate system-level reliability metrics.
        
        Args:
            mission_duration_hours: Mission duration for reliability calculation
            
        Returns:
            ReliabilityPrediction with system reliability metrics
        """
        self.logger.info(f"Calculating system reliability for {mission_duration_hours} hour mission")
        
        # Calculate system failure rate (sum of component failure rates for series system)
        system_failure_rate = sum(comp.failure_rate for comp in self.component_reliability.values())
        
        # System MTBF
        system_mtbf = 1.0 / system_failure_rate if system_failure_rate > 0 else float('inf')
        
        # Mission reliability (exponential distribution)
        mission_reliability = np.exp(-system_failure_rate * mission_duration_hours)
        
        # System availability (considering maintenance)
        component_availabilities = [comp.availability for comp in self.component_reliability.values()]
        system_availability = np.prod(component_availabilities)  # Series system
        
        # Component contributions to system failure rate
        component_contributions = {
            comp_id: comp.failure_rate / system_failure_rate if system_failure_rate > 0 else 0
            for comp_id, comp in self.component_reliability.items()
        }
        
        # Identify critical components (top contributors)
        sorted_contributions = sorted(component_contributions.items(), 
                                    key=lambda x: x[1], reverse=True)
        critical_components = [comp_id for comp_id, _ in sorted_contributions[:3]]
        
        # Maintenance requirements (optimal intervals)
        maintenance_requirements = self._calculate_maintenance_intervals()
        
        # Confidence bounds
        confidence_bounds = self._calculate_confidence_bounds(mission_duration_hours)
        
        prediction = ReliabilityPrediction(
            system_mtbf_hours=system_mtbf,
            system_failure_rate=system_failure_rate,
            mission_reliability=mission_reliability,
            availability=system_availability,
            component_contributions=component_contributions,
            critical_components=critical_components,
            maintenance_requirements=maintenance_requirements,
            confidence_bounds=confidence_bounds
        )
        
        self.logger.info(f"System MTBF: {system_mtbf:.1f} hours")
        self.logger.info(f"Mission reliability: {mission_reliability*100:.1f}%")
        self.logger.info(f"System availability: {system_availability*100:.1f}%")
        
        return prediction
    
    def perform_weibull_analysis(self, component_id: str, 
                                failure_times: List[float]) -> Dict[str, float]:
        """
        Perform Weibull analysis on component failure data.
        
        Args:
            component_id: Component identifier
            failure_times: List of failure times
            
        Returns:
            Dictionary with Weibull parameters and reliability metrics
        """
        if len(failure_times) < 3:
            self.logger.warning(f"Insufficient failure data for {component_id} Weibull analysis")
            return {"shape": 1.0, "scale": 1000.0, "reliability_1000h": 0.9}
        
        # Fit Weibull distribution
        try:
            # Use scipy.stats.weibull_min for fitting
            shape, loc, scale = stats.weibull_min.fit(failure_times, floc=0)
            
            # Calculate reliability at various time points
            reliability_100h = stats.weibull_min.sf(100, shape, loc, scale)
            reliability_1000h = stats.weibull_min.sf(1000, shape, loc, scale)
            reliability_8760h = stats.weibull_min.sf(8760, shape, loc, scale)  # 1 year
            
            # Calculate characteristic life (η) and MTBF
            characteristic_life = scale
            mtbf = scale * stats.gamma(1 + 1/shape)
            
            # Update component reliability data
            if component_id in self.component_reliability:
                self.component_reliability[component_id].weibull_shape = shape
                self.component_reliability[component_id].weibull_scale = scale
                self.component_reliability[component_id].mtbf_hours = mtbf
                self.component_reliability[component_id].failure_rate = 1.0 / mtbf
            
            results = {
                "shape_parameter": shape,
                "scale_parameter": scale,
                "characteristic_life": characteristic_life,
                "mtbf_hours": mtbf,
                "reliability_100h": reliability_100h,
                "reliability_1000h": reliability_1000h,
                "reliability_8760h": reliability_8760h,
                "b10_life": scale * (-np.log(0.9))**(1/shape),  # 10% failure life
                "b50_life": scale * (-np.log(0.5))**(1/shape)   # 50% failure life (median)
            }
            
            self.logger.info(f"Weibull analysis for {component_id}:")
            self.logger.info(f"  Shape parameter (β): {shape:.2f}")
            self.logger.info(f"  Scale parameter (η): {scale:.1f} hours")
            self.logger.info(f"  MTBF: {mtbf:.1f} hours")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in Weibull analysis for {component_id}: {e}")
            return {"shape": 1.0, "scale": 1000.0, "mtbf_hours": 1000.0}
    
    def analyze_failure_modes(self) -> List[FailureModeEffect]:
        """
        Analyze failure modes and return prioritized list.
        
        Returns:
            List of FMEA entries sorted by Risk Priority Number (RPN)
        """
        # Sort FMEA entries by RPN (highest risk first)
        sorted_fmea = sorted(self.fmea_entries, key=lambda x: x.rpn, reverse=True)
        
        self.logger.info("Failure Mode Analysis Results:")
        self.logger.info("Top 5 highest risk failure modes:")
        
        for i, entry in enumerate(sorted_fmea[:5], 1):
            self.logger.info(f"{i}. {entry.component}/{entry.failure_mode} - RPN: {entry.rpn}")
            self.logger.info(f"   Cause: {entry.failure_cause}")
            self.logger.info(f"   Effect: {entry.system_effect}")
        
        return sorted_fmea
    
    def predict_component_lifetime(self, component_id: str, 
                                 confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Predict component lifetime with confidence intervals.
        
        Args:
            component_id: Component identifier
            confidence_level: Confidence level for predictions
            
        Returns:
            Dictionary with lifetime predictions
        """
        if component_id not in self.component_reliability:
            self.logger.error(f"Component {component_id} not found in reliability database")
            return {}
        
        comp = self.component_reliability[component_id]
        
        # Use Weibull distribution for lifetime prediction
        shape = comp.weibull_shape
        scale = comp.weibull_scale
        
        # Calculate various lifetime percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        lifetime_percentiles = {}
        
        for p in percentiles:
            lifetime = scale * (-np.log(1 - p/100))**(1/shape)
            lifetime_percentiles[f"B{p}_life"] = lifetime
        
        # Calculate confidence bounds
        alpha = 1 - confidence_level
        lower_bound = scale * (-np.log(1 - alpha/2))**(1/shape)
        upper_bound = scale * (-np.log(alpha/2))**(1/shape)
        
        results = {
            "mean_lifetime": comp.mtbf_hours,
            "median_lifetime": lifetime_percentiles["B50_life"],
            "confidence_lower": lower_bound,
            "confidence_upper": upper_bound,
            **lifetime_percentiles
        }
        
        self.logger.info(f"Lifetime prediction for {component_id}:")
        self.logger.info(f"  Mean lifetime: {comp.mtbf_hours:.1f} hours")
        self.logger.info(f"  Median lifetime: {lifetime_percentiles['B50_life']:.1f} hours")
        self.logger.info(f"  {confidence_level*100:.0f}% confidence: "
                   f"{lower_bound:.1f} - {upper_bound:.1f} hours")
        
        return results
    
    def optimize_maintenance_schedule(self, cost_parameters: Dict[str, float]) -> Dict[str, float]:
        """
        Optimize maintenance schedule to minimize total cost.
        
        Args:
            cost_parameters: Dictionary with cost parameters
            
        Returns:
            Dictionary with optimal maintenance intervals
        """
        # Default cost parameters
        default_costs = {
            "preventive_cost_per_hour": 100.0,    # Cost of preventive maintenance
            "corrective_cost_per_hour": 500.0,    # Cost of corrective maintenance
            "downtime_cost_per_hour": 1000.0,     # Cost of system downtime
            "inspection_cost": 50.0               # Cost per inspection
        }
        
        costs = {**default_costs, **cost_parameters}
        
        optimal_intervals = {}
        
        for comp_id, comp in self.component_reliability.items():
            # Use economic optimization model
            # Minimize: C_prev/T + C_corr * F(T) + C_down * unavailability
            
            def cost_function(interval_hours):
                if interval_hours <= 0:
                    return float('inf')
                
                # Preventive maintenance cost
                prev_cost = costs["preventive_cost_per_hour"] / interval_hours
                
                # Corrective maintenance cost (based on failure probability)
                failure_prob = 1 - np.exp(-comp.failure_rate * interval_hours)
                corr_cost = costs["corrective_cost_per_hour"] * failure_prob / interval_hours
                
                # Downtime cost
                downtime_cost = costs["downtime_cost_per_hour"] * (1 - comp.availability)
                
                return prev_cost + corr_cost + downtime_cost
            
            # Find optimal interval
            try:
                result = minimize(cost_function, x0=comp.mtbf_hours * 0.5, 
                                bounds=[(24, comp.mtbf_hours * 2)], method='bounded')
                
                if result.success:
                    optimal_intervals[comp_id] = result.x[0]
                else:
                    # Fallback to rule-of-thumb (MTBF/2)
                    optimal_intervals[comp_id] = comp.mtbf_hours * 0.5
                    
            except Exception as e:
                self.logger.warning(f"Optimization failed for {comp_id}: {e}")
                optimal_intervals[comp_id] = comp.mtbf_hours * 0.5
        
        self.logger.info("Optimal maintenance intervals:")
        for comp_id, interval in optimal_intervals.items():
            self.logger.info(f"  {comp_id}: {interval:.1f} hours ({interval/24:.1f} days)")
        
        return optimal_intervals
    
    def _calculate_maintenance_intervals(self) -> Dict[str, float]:
        """Calculate optimal maintenance intervals for components."""
        # Simple maintenance interval calculation
        intervals = {}
        
        for comp_id, comp in self.component_reliability.items():
            # Use age replacement model
            # Optimal interval minimizes cost rate
            
            # Simple heuristic: maintain at 70% of MTBF
            intervals[comp_id] = comp.mtbf_hours * 0.7
        
        return intervals
    
    def _calculate_confidence_bounds(self, mission_duration: float) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence bounds for reliability predictions."""
        bounds = {}
        
        for comp_id, comp in self.component_reliability.items():
            # Use confidence interval for MTBF to calculate reliability bounds
            lower_mtbf, upper_mtbf = comp.confidence_interval
            
            # Reliability bounds
            lower_reliability = np.exp(-mission_duration / upper_mtbf)
            upper_reliability = np.exp(-mission_duration / lower_mtbf)
            
            bounds[comp_id] = (lower_reliability, upper_reliability)
        
        return bounds
    
    def update_reliability_data(self, component_id: str, failure_time: float):
        """Update component reliability data with new failure information."""
        if component_id not in self.component_reliability:
            self.logger.error(f"Component {component_id} not found")
            return
        
        comp = self.component_reliability[component_id]
        
        # Update failure count and operating hours
        comp.failure_count += 1
        comp.operating_hours += failure_time
        comp.last_failure = datetime.now()
        
        # Recalculate MTBF
        if comp.failure_count > 0:
            comp.mtbf_hours = comp.operating_hours / comp.failure_count
            comp.failure_rate = 1.0 / comp.mtbf_hours
        
        # Record failure event
        self.failure_history.append({
            "component": component_id,
            "failure_time": failure_time,
            "timestamp": datetime.now().isoformat(),
            "cumulative_failures": comp.failure_count
        })
        
        self.logger.info(f"Updated reliability data for {component_id}")
        self.logger.info(f"  New MTBF: {comp.mtbf_hours:.1f} hours")
        self.logger.info(f"  Total failures: {comp.failure_count}")
    
    def generate_reliability_report(self, output_file: Optional[Path] = None) -> str:
        """Generate comprehensive reliability analysis report."""
        
        # Calculate system reliability
        prediction = self.calculate_system_reliability(8760)  # 1 year mission
        
        # Analyze failure modes
        critical_failures = self.analyze_failure_modes()
        
        report_lines = []
        report_lines.append("# MFC System Reliability Analysis Report")
        report_lines.append(f"Generated: {datetime.now().isoformat()}")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("## Executive Summary")
        report_lines.append(f"- **System MTBF**: {prediction.system_mtbf_hours:.1f} hours")
        report_lines.append(f"- **Mission Reliability** (1 year): {prediction.mission_reliability*100:.1f}%")
        report_lines.append(f"- **System Availability**: {prediction.availability*100:.1f}%")
        report_lines.append(f"- **Critical Components**: {', '.join(prediction.critical_components)}")
        report_lines.append("")
        
        # Component Reliability
        report_lines.append("## Component Reliability Analysis")
        report_lines.append("| Component | MTBF (hours) | Failure Rate | Availability | Weibull β | Weibull η |")
        report_lines.append("|-----------|--------------|--------------|--------------|-----------|-----------|")
        
        for comp_id, comp in self.component_reliability.items():
            report_lines.append(f"| {comp_id} | {comp.mtbf_hours:.1f} | {comp.failure_rate:.2e} | "
                               f"{comp.availability:.3f} | {comp.weibull_shape:.2f} | {comp.weibull_scale:.1f} |")
        report_lines.append("")
        
        # Critical Failure Modes
        report_lines.append("## Critical Failure Modes (Top 5 by RPN)")
        report_lines.append("| Component | Failure Mode | RPN | Occurrence | Severity | Detection |")
        report_lines.append("|-----------|--------------|-----|------------|----------|-----------|")
        
        for entry in critical_failures[:5]:
            report_lines.append(f"| {entry.component} | {entry.failure_mode} | {entry.rpn} | "
                               f"{entry.occurrence_rating} | {entry.severity_rating} | {entry.detection_rating} |")
        report_lines.append("")
        
        # Maintenance Recommendations
        report_lines.append("## Maintenance Recommendations")
        for comp_id, interval in prediction.maintenance_requirements.items():
            report_lines.append(f"- **{comp_id}**: Maintain every {interval:.1f} hours ({interval/24:.1f} days)")
        report_lines.append("")
        
        # Component Contributions
        report_lines.append("## Component Failure Rate Contributions")
        for comp_id, contribution in prediction.component_contributions.items():
            report_lines.append(f"- **{comp_id}**: {contribution*100:.1f}% of total system failure rate")
        report_lines.append("")
        
        # Recommendations
        report_lines.append("## Recommendations")
        
        # High RPN failure modes
        high_rpn_modes = [entry for entry in critical_failures if entry.rpn > 100]
        if high_rpn_modes:
            report_lines.append("### High Risk Failure Modes")
            for entry in high_rpn_modes[:3]:
                report_lines.append(f"- **{entry.component}/{entry.failure_mode}** (RPN: {entry.rpn})")
                for action in entry.recommended_actions:
                    report_lines.append(f"  - {action}")
        
        # Critical components
        if prediction.critical_components:
            report_lines.append("### Focus Areas")
            for comp in prediction.critical_components:
                contribution = prediction.component_contributions[comp]
                report_lines.append(f"- **{comp}**: {contribution*100:.1f}% contribution to system failure rate")
                report_lines.append("  - Consider redundancy or improved reliability")
        
        report_text = "\n".join(report_lines)
        
        # Save report if file specified
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Reliability report saved to {output_file}")
        
        return report_text
    
    def export_reliability_data(self, output_file: Path):
        """Export reliability data for external analysis."""
        
        export_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "system_config": self.system_config,
                "total_components": len(self.component_reliability)
            },
            "component_reliability": {
                comp_id: {
                    "mtbf_hours": comp.mtbf_hours,
                    "failure_rate": comp.failure_rate,
                    "availability": comp.availability,
                    "maintainability": comp.maintainability,
                    "weibull_shape": comp.weibull_shape,
                    "weibull_scale": comp.weibull_scale,
                    "operating_hours": comp.operating_hours,
                    "failure_count": comp.failure_count,
                    "confidence_interval": comp.confidence_interval
                }
                for comp_id, comp in self.component_reliability.items()
            },
            "fmea_analysis": [
                {
                    "component": entry.component,
                    "failure_mode": entry.failure_mode,
                    "failure_cause": entry.failure_cause,
                    "system_effect": entry.system_effect,
                    "occurrence_rating": entry.occurrence_rating,
                    "severity_rating": entry.severity_rating,
                    "detection_rating": entry.detection_rating,
                    "rpn": entry.rpn,
                    "recommended_actions": entry.recommended_actions
                }
                for entry in self.fmea_entries
            ],
            "failure_history": self.failure_history,
            "system_prediction": self.calculate_system_reliability(8760).__dict__
        }
        
        # Save to JSON
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Reliability data exported to {output_file}")
def run_reliability_analysis(output_dir: Optional[Path] = None) -> ReliabilityAnalyzer:
    """
    Run comprehensive reliability analysis.
    
    Args:
        output_dir: Directory for output files
        
    Returns:
        ReliabilityAnalyzer with completed analysis
    """
    output_dir = output_dir or Path("reliability_analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting comprehensive reliability analysis")
    
    # Initialize analyzer
    analyzer = ReliabilityAnalyzer(output_dir)
    
    # Generate reliability report
    report_file = output_dir / f"reliability_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    analyzer.generate_reliability_report(report_file)
    
    # Export data
    data_file = output_dir / f"reliability_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    analyzer.export_reliability_data(data_file)
    
    # Calculate system reliability for different mission durations
    durations = [720, 2160, 4380, 8760]  # 1 month, 3 months, 6 months, 1 year
    
    logger.info("Mission reliability predictions:")
    for duration in durations:
        prediction = analyzer.calculate_system_reliability(duration)
        logger.info(f"  {duration/24:.0f} days: {prediction.mission_reliability*100:.1f}%")
    
    logger.info(f"Reliability analysis completed. Results saved to {output_dir}")
    
    return analyzer

if __name__ == "__main__":
    # Run example reliability analysis
    analyzer = run_reliability_analysis(Path("reliability_analysis_results"))
    
    print("Reliability analysis completed!")
    print("Results available in: reliability_analysis_results/")