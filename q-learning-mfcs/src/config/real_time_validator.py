"""
Real-Time Parameter Validation System

This module provides enhanced real-time validation with instant feedback,
contextual help, and scientific reasoning for MFC parameters.

User Story 1.1.2: Real-Time Parameter Validation
Created: 2025-07-31
Last Modified: 2025-07-31
"""

import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from config.literature_database import LITERATURE_DB, ParameterInfo
from config.unit_converter import UNIT_CONVERTER


class ValidationSpeed(Enum):
    """Validation response time categories."""
    INSTANT = "instant"  # <50ms
    FAST = "fast"      # <200ms
    NORMAL = "normal"   # <500ms
    SLOW = "slow"      # >500ms


class ValidationLevel(Enum):
    """Validation severity levels with color coding."""
    VALID = "valid"           # Green
    CAUTION = "caution"       # Yellow/Orange
    WARNING = "warning"       # Orange/Red
    INVALID = "invalid"       # Red
    UNKNOWN = "unknown"       # Gray


@dataclass
class ValidationResult:
    """Enhanced validation result with scientific context."""

    level: ValidationLevel
    message: str
    scientific_reasoning: str
    suggested_ranges: List[Tuple[float, float]]
    confidence_score: float  # 0.0 to 1.0
    uncertainty_bounds: Tuple[float, float]  # Lower and upper uncertainty bounds
    response_time_ms: float
    recommendations: List[str]
    warnings: List[str]
    parameter_info: Optional[ParameterInfo] = None


@dataclass
class ResearchObjective:
    """Research objective context for parameter suggestions."""

    name: str
    description: str
    priority_parameters: List[str]
    target_ranges: Dict[str, Tuple[float, float]]
    scientific_context: str


class RealTimeValidator:
    """Enhanced real-time parameter validation system."""

    def __init__(self):
        """Initialize real-time validator."""
        self.literature_db = LITERATURE_DB
        self.unit_converter = UNIT_CONVERTER
        self.validation_cache: Dict[str, ValidationResult] = {}
        self.research_objectives = self._define_research_objectives()

        # Performance tracking
        self.validation_times: List[float] = []
        self.cache_hits = 0
        self.cache_misses = 0

    def _define_research_objectives(self) -> Dict[str, ResearchObjective]:
        """Define common research objectives for parameter optimization."""

        objectives = {
            "maximum_power": ResearchObjective(
                name="Maximum Power Output",
                description="Optimize for highest electrical power generation",
                priority_parameters=["max_current_density", "anode_potential", "electrode_area"],
                target_ranges={
                    "max_current_density": (5.0, 15.0),  # mA/cm²
                    "anode_potential": (-0.4, -0.2),     # V vs SHE
                    "electrode_area": (0.001, 0.01)      # m²
                },
                scientific_context="Power maximization requires optimal current density and electrode potential balance"
            ),

            "substrate_efficiency": ResearchObjective(
                name="Substrate Utilization Efficiency",
                description="Maximize substrate conversion to electricity",
                priority_parameters=["substrate_concentration", "flow_rate", "biofilm_thickness"],
                target_ranges={
                    "substrate_concentration": (20.0, 50.0),  # mM
                    "flow_rate": (10.0, 25.0),               # mL/h
                    "biofilm_thickness": (50.0, 200.0)       # μm
                },
                scientific_context="Efficient substrate utilization requires balanced concentration, flow, and biofilm development"
            ),

            "stable_operation": ResearchObjective(
                name="Long-Term Stable Operation",
                description="Optimize for consistent long-term performance",
                priority_parameters=["biofilm_conductivity", "growth_rate", "half_saturation"],
                target_ranges={
                    "biofilm_conductivity": (0.005, 0.02),   # S/m
                    "growth_rate": (0.02, 0.08),             # h⁻¹
                    "half_saturation": (3.0, 8.0)            # mM
                },
                scientific_context="Stability requires balanced biofilm properties and moderate growth rates"
            ),

            "learning_optimization": ResearchObjective(
                name="Q-Learning Performance",
                description="Optimize reinforcement learning parameters",
                priority_parameters=["learning_rate", "discount_factor", "exploration_rate"],
                target_ranges={
                    "learning_rate": (0.05, 0.2),      # dimensionless
                    "discount_factor": (0.9, 0.99),    # dimensionless
                    "exploration_rate": (0.1, 0.4)     # dimensionless
                },
                scientific_context="Effective Q-learning requires balanced exploration-exploitation with appropriate learning rates"
            )
        }

        return objectives

    def validate_parameter_realtime(
        self,
        parameter_name: str,
        value: float,
        research_objective: Optional[str] = None,
        use_cache: bool = True
    ) -> ValidationResult:
        """
        Perform real-time parameter validation with enhanced feedback.
        
        Args:
            parameter_name: Name of parameter to validate
            value: Parameter value
            research_objective: Optional research objective context
            use_cache: Whether to use validation cache
            
        Returns:
            Enhanced validation result with scientific context
        """
        start_time = time.time()

        # Check cache first for performance
        cache_key = f"{parameter_name}_{value}_{research_objective}"
        if use_cache and cache_key in self.validation_cache:
            cached_result = self.validation_cache[cache_key]
            self.cache_hits += 1
            return cached_result

        self.cache_misses += 1

        # Get parameter info
        param = self.literature_db.get_parameter(parameter_name)
        if not param:
            return self._create_unknown_parameter_result(parameter_name, value, start_time)

        # Perform basic validation
        basic_validation = self.literature_db.validate_parameter_value(parameter_name, value)

        # Enhance with scientific context
        result = self._enhance_validation_result(
            param, value, basic_validation, research_objective, start_time
        )

        # Cache result for performance
        if use_cache:
            self.validation_cache[cache_key] = result

        # Track performance
        self.validation_times.append(result.response_time_ms)

        return result

    def _create_unknown_parameter_result(
        self,
        parameter_name: str,
        value: float,
        start_time: float
    ) -> ValidationResult:
        """Create result for unknown parameter."""

        response_time = (time.time() - start_time) * 1000

        return ValidationResult(
            level=ValidationLevel.UNKNOWN,
            message=f"Parameter '{parameter_name}' not found in literature database",
            scientific_reasoning="This parameter is not documented in the scientific literature database. Consider using standard MFC parameters.",
            suggested_ranges=[],
            confidence_score=0.0,
            uncertainty_bounds=(value * 0.5, value * 1.5),  # Wide uncertainty for unknown parameters
            response_time_ms=response_time,
            recommendations=[
                "Check parameter name spelling",
                "Consult MFC literature for standard parameters",
                "Use parameters from established research"
            ],
            warnings=["Unknown parameter may lead to invalid results"]
        )

    def _enhance_validation_result(
        self,
        param: ParameterInfo,
        value: float,
        basic_validation: Dict[str, Any],
        research_objective: Optional[str],
        start_time: float
    ) -> ValidationResult:
        """Enhance basic validation with scientific context."""

        # Map basic status to enhanced levels
        status_mapping = {
            'valid': ValidationLevel.VALID,
            'caution': ValidationLevel.CAUTION,
            'invalid': ValidationLevel.INVALID
        }

        level = status_mapping.get(basic_validation['status'], ValidationLevel.UNKNOWN)

        # Generate scientific reasoning
        scientific_reasoning = self._generate_scientific_reasoning(param, value, level)

        # Get suggested ranges
        suggested_ranges = self._get_suggested_ranges(param, research_objective)

        # Calculate confidence score and uncertainty bounds
        confidence_score = self._calculate_confidence_score(param, value, level)
        uncertainty_bounds = self._calculate_uncertainty_bounds(param, value, confidence_score)

        # Generate enhanced recommendations
        recommendations = self._generate_enhanced_recommendations(param, value, level, research_objective)

        # Generate warnings for problematic combinations
        warnings = self._generate_warnings(param, value, level)

        response_time = (time.time() - start_time) * 1000

        return ValidationResult(
            level=level,
            message=basic_validation.get('message', ''),
            scientific_reasoning=scientific_reasoning,
            suggested_ranges=suggested_ranges,
            confidence_score=confidence_score,
            uncertainty_bounds=uncertainty_bounds,
            response_time_ms=response_time,
            recommendations=recommendations,
            warnings=warnings,
            parameter_info=param
        )

    def _generate_scientific_reasoning(
        self,
        param: ParameterInfo,
        value: float,
        level: ValidationLevel
    ) -> str:
        """Generate scientific reasoning for validation result."""

        reasoning_templates = {
            ValidationLevel.VALID: [
                f"The value {value} {param.unit} is within the scientifically validated range for {param.name}.",
                f"This parameter value aligns with peer-reviewed research findings from {len(param.references)} studies.",
                f"Operating at {value} {param.unit} is consistent with optimal {param.category.value} conditions."
            ],

            ValidationLevel.CAUTION: [
                f"The value {value} {param.unit} is outside the recommended range but within documented limits.",
                "While not optimal, this value has been reported in literature under specific conditions.",
                f"Consider the trade-offs: this value may affect {param.category.value} performance."
            ],

            ValidationLevel.INVALID: [
                f"The value {value} {param.unit} exceeds scientifically documented limits for {param.name}.",
                "Operating outside the valid range may lead to system failure or unrealistic results.",
                f"No peer-reviewed studies support operation at this {param.category.value} level."
            ]
        }

        templates = reasoning_templates.get(level, ["Unknown validation level"])
        # Select first template for now, could be randomized or context-based
        base_reasoning = templates[0]

        # Add parameter-specific context
        if param.notes:
            base_reasoning += f" Note: {param.notes}"

        return base_reasoning

    def _get_suggested_ranges(
        self,
        param: ParameterInfo,
        research_objective: Optional[str]
    ) -> List[Tuple[float, float]]:
        """Get suggested parameter ranges based on research objective."""

        ranges = []

        # Always include literature recommended range
        ranges.append(param.recommended_range)

        # Add research objective specific range if available
        if research_objective and research_objective in self.research_objectives:
            objective = self.research_objectives[research_objective]
            if param.name in objective.target_ranges:
                ranges.append(objective.target_ranges[param.name])

        # Add typical operating range (slightly wider than recommended)
        typical_margin = 0.1  # 10% margin
        rec_min, rec_max = param.recommended_range
        range_span = rec_max - rec_min
        typical_range = (
            rec_min - typical_margin * range_span,
            rec_max + typical_margin * range_span
        )
        # Ensure within valid bounds
        typical_range = (
            max(typical_range[0], param.min_value),
            min(typical_range[1], param.max_value)
        )
        ranges.append(typical_range)

        return ranges

    def _calculate_confidence_score(
        self,
        param: ParameterInfo,
        value: float,
        level: ValidationLevel
    ) -> float:
        """Calculate confidence score for validation result."""

        base_scores = {
            ValidationLevel.VALID: 0.95,
            ValidationLevel.CAUTION: 0.75,
            ValidationLevel.WARNING: 0.5,
            ValidationLevel.INVALID: 0.2,
            ValidationLevel.UNKNOWN: 0.0
        }

        base_score = base_scores.get(level, 0.0)

        # Adjust based on literature support
        literature_factor = min(len(param.references) / 3.0, 1.0)  # Normalize to max 3 references

        # Adjust based on how close to recommended range
        rec_min, rec_max = param.recommended_range
        if rec_min <= value <= rec_max:
            range_factor = 1.0
        else:
            # Calculate distance from nearest boundary
            if value < rec_min:
                distance = (rec_min - value) / (rec_min - param.min_value)
            else:
                distance = (value - rec_max) / (param.max_value - rec_max)
            range_factor = max(0.0, 1.0 - distance)

        final_score = base_score * literature_factor * range_factor
        return min(max(final_score, 0.0), 1.0)  # Clamp to [0, 1]

    def _calculate_uncertainty_bounds(
        self,
        param: ParameterInfo,
        value: float,
        confidence_score: float
    ) -> Tuple[float, float]:
        """
        Calculate uncertainty bounds for parameter value.
        
        Args:
            param: Parameter information
            value: Current parameter value
            confidence_score: Validation confidence score
            
        Returns:
            Tuple of (lower_bound, upper_bound) representing uncertainty
        """
        # Base uncertainty as percentage of parameter range
        param_range = param.max_value - param.min_value
        base_uncertainty = param_range * 0.05  # 5% of range as base uncertainty

        # Adjust uncertainty based on confidence score
        # Lower confidence = higher uncertainty
        uncertainty_factor = 1.0 - confidence_score
        adjusted_uncertainty = base_uncertainty * (1.0 + 2.0 * uncertainty_factor)

        # Calculate bounds
        lower_bound = max(value - adjusted_uncertainty, param.min_value)
        upper_bound = min(value + adjusted_uncertainty, param.max_value)

        # Ensure bounds are valid
        if lower_bound > upper_bound:
            # Handle edge case where bounds are invalid
            mid_point = (param.min_value + param.max_value) / 2
            small_range = (param.max_value - param.min_value) * 0.1
            lower_bound = max(mid_point - small_range, param.min_value)
            upper_bound = min(mid_point + small_range, param.max_value)

        return (lower_bound, upper_bound)

    def _generate_enhanced_recommendations(
        self,
        param: ParameterInfo,
        value: float,
        level: ValidationLevel,
        research_objective: Optional[str]
    ) -> List[str]:
        """Generate enhanced recommendations based on validation level and context."""

        recommendations = []

        if level == ValidationLevel.VALID:
            recommendations.append(f"✅ Excellent choice! {param.name} is optimally configured.")
            if research_objective:
                obj = self.research_objectives.get(research_objective)
                if obj and param.name in obj.priority_parameters:
                    recommendations.append(f"🎯 This parameter is critical for {obj.name} objectives.")

        elif level == ValidationLevel.CAUTION:
            rec_min, rec_max = param.recommended_range
            recommendations.append(f"⚠️ Consider using values between {rec_min} and {rec_max} {param.unit} for optimal performance.")

            if research_objective:
                obj = self.research_objectives.get(research_objective)
                if obj and param.name in obj.target_ranges:
                    target_min, target_max = obj.target_ranges[param.name]
                    recommendations.append(f"🎯 For {obj.name}, target range is {target_min}-{target_max} {param.unit}")

        elif level == ValidationLevel.INVALID:
            recommendations.append(f"❌ Use values between {param.min_value} and {param.max_value} {param.unit}")
            recommendations.append(f"📚 Based on {len(param.references)} peer-reviewed studies")
            recommendations.append(f"🔧 Start with typical value: {param.typical_value} {param.unit}")

        # Add parameter-specific recommendations
        if param.category.value == "electrochemical":
            recommendations.append("⚡ Electrochemical parameters strongly affect power output")
        elif param.category.value == "biological":
            recommendations.append("🦠 Biological parameters influence microbial activity")
        elif param.category.value == "qlearning":
            recommendations.append("🧠 Q-learning parameters affect convergence speed")

        return recommendations

    def _generate_warnings(
        self,
        param: ParameterInfo,
        value: float,
        level: ValidationLevel
    ) -> List[str]:
        """Generate warnings for potentially problematic parameter values."""

        warnings = []

        # Critical value warnings
        if level == ValidationLevel.INVALID:
            warnings.append("🚨 Invalid parameter value may cause simulation failure")

            # Specific warnings based on parameter type
            if param.name == "anode_potential" and value > 0:
                warnings.append("⚠️ Positive anode potential is thermodynamically unfavorable")
            elif param.name == "max_current_density" and value > 20:
                warnings.append("⚠️ Extremely high current density may damage biofilm")
            elif param.name == "learning_rate" and value > 0.5:
                warnings.append("⚠️ High learning rate may prevent Q-learning convergence")

        # Combination warnings (simplified for now)
        if param.category.value == "biological" and level != ValidationLevel.VALID:
            warnings.append("🦠 Suboptimal biological parameters may affect microbial viability")

        return warnings

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get validator performance metrics."""

        if not self.validation_times:
            return {
                "avg_response_time_ms": 0.0,
                "max_response_time_ms": 0.0,
                "min_response_time_ms": 0.0,
                "cache_hit_rate": 0.0,
                "total_validations": 0
            }

        return {
            "avg_response_time_ms": sum(self.validation_times) / len(self.validation_times),
            "max_response_time_ms": max(self.validation_times),
            "min_response_time_ms": min(self.validation_times),
            "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0.0,
            "total_validations": len(self.validation_times),
            "fast_validations": sum(1 for t in self.validation_times if t < 200),
            "instant_validations": sum(1 for t in self.validation_times if t < 50)
        }

    def get_research_objectives(self) -> List[str]:
        """Get available research objectives."""
        return list(self.research_objectives.keys())

    def get_research_objective_info(self, objective_name: str) -> Optional[ResearchObjective]:
        """Get information about a specific research objective."""
        return self.research_objectives.get(objective_name)


# Global instance for easy access
REAL_TIME_VALIDATOR = RealTimeValidator()
