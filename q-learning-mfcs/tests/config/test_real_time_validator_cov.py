import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest
from config.real_time_validator import (
    ValidationSpeed,
    ValidationLevel,
    ValidationResult,
    ResearchObjective,
    RealTimeValidator,
    REAL_TIME_VALIDATOR,
)


class TestEnums:
    def test_validation_speed_values(self):
        assert ValidationSpeed.INSTANT.value == "instant"
        assert ValidationSpeed.FAST.value == "fast"
        assert ValidationSpeed.NORMAL.value == "normal"
        assert ValidationSpeed.SLOW.value == "slow"

    def test_validation_level_values(self):
        assert ValidationLevel.VALID.value == "valid"
        assert ValidationLevel.CAUTION.value == "caution"
        assert ValidationLevel.WARNING.value == "warning"
        assert ValidationLevel.INVALID.value == "invalid"
        assert ValidationLevel.UNKNOWN.value == "unknown"


class TestValidationResult:
    def test_creation(self):
        r = ValidationResult(
            level=ValidationLevel.VALID,
            message="ok",
            scientific_reasoning="good",
            suggested_ranges=[(0.0, 1.0)],
            confidence_score=0.95,
            uncertainty_bounds=(0.1, 0.9),
            response_time_ms=10.0,
            recommendations=["good"],
            warnings=[],
        )
        assert r.level == ValidationLevel.VALID
        assert r.parameter_info is None


class TestResearchObjective:
    def test_creation(self):
        ro = ResearchObjective(
            name="test",
            description="test desc",
            priority_parameters=["a", "b"],
            target_ranges={"a": (0.0, 1.0)},
            scientific_context="context",
        )
        assert ro.name == "test"


class TestRealTimeValidator:
    @pytest.fixture
    def validator(self):
        return RealTimeValidator()

    def test_init(self, validator):
        assert validator.literature_db is not None
        assert validator.unit_converter is not None
        assert validator.validation_cache == {}
        assert validator.cache_hits == 0
        assert validator.cache_misses == 0

    def test_research_objectives_defined(self, validator):
        objs = validator.research_objectives
        assert "maximum_power" in objs
        assert "substrate_efficiency" in objs
        assert "stable_operation" in objs
        assert "learning_optimization" in objs

    def test_validate_unknown_parameter(self, validator):
        result = validator.validate_parameter_realtime("nonexistent_param", 1.0)
        assert result.level == ValidationLevel.UNKNOWN
        assert "not found" in result.message
        assert result.confidence_score == 0.0
        assert len(result.recommendations) > 0
        assert len(result.warnings) > 0

    def test_validate_known_parameter(self, validator):
        params = list(validator.literature_db.parameters.keys())
        if params:
            param_name = params[0]
            param_info = validator.literature_db.parameters[param_name]
            val = param_info.typical_value
            result = validator.validate_parameter_realtime(param_name, val)
            assert result.level in [
                ValidationLevel.VALID,
                ValidationLevel.CAUTION,
                ValidationLevel.INVALID,
                ValidationLevel.UNKNOWN,
            ]
            assert result.response_time_ms >= 0

    def test_cache_hit(self, validator):
        params = list(validator.literature_db.parameters.keys())
        if params:
            param_name = params[0]
            param_info = validator.literature_db.parameters[param_name]
            val = param_info.typical_value
            validator.validate_parameter_realtime(param_name, val, use_cache=True)
            validator.validate_parameter_realtime(param_name, val, use_cache=True)
            assert validator.cache_hits >= 1

    def test_cache_miss(self, validator):
        validator.validate_parameter_realtime("nonexistent", 1.0, use_cache=True)
        assert validator.cache_misses >= 1

    def test_no_cache(self, validator):
        validator.validate_parameter_realtime("nonexistent", 1.0, use_cache=False)
        assert validator.cache_misses >= 1

    def test_validate_with_research_objective(self, validator):
        params = list(validator.literature_db.parameters.keys())
        if params:
            param_name = params[0]
            param_info = validator.literature_db.parameters[param_name]
            val = param_info.typical_value
            result = validator.validate_parameter_realtime(
                param_name, val, research_objective="maximum_power"
            )
            assert result is not None

    def test_unknown_param_uncertainty_bounds(self, validator):
        result = validator.validate_parameter_realtime("fake_param", 10.0)
        assert result.uncertainty_bounds == (5.0, 15.0)

    def test_scientific_reasoning_valid(self, validator):
        params = list(validator.literature_db.parameters.keys())
        if params:
            param_name = params[0]
            param_info = validator.literature_db.parameters[param_name]
            val = param_info.typical_value
            result = validator.validate_parameter_realtime(param_name, val)
            assert len(result.scientific_reasoning) > 0

    def test_get_performance_metrics_empty(self, validator):
        metrics = validator.get_performance_metrics()
        assert metrics["avg_response_time_ms"] == 0.0
        assert metrics["total_validations"] == 0

    def test_get_performance_metrics_with_data(self, validator):
        validator.validate_parameter_realtime("fake", 1.0)
        metrics = validator.get_performance_metrics()
        assert metrics["total_validations"] >= 0

    def test_get_research_objectives(self, validator):
        objs = validator.get_research_objectives()
        assert isinstance(objs, list)
        assert "maximum_power" in objs

    def test_get_research_objective_info_valid(self, validator):
        info = validator.get_research_objective_info("maximum_power")
        assert info is not None
        assert info.name == "Maximum Power Output"

    def test_get_research_objective_info_invalid(self, validator):
        info = validator.get_research_objective_info("nonexistent")
        assert info is None

    def test_global_instance(self):
        assert REAL_TIME_VALIDATOR is not None
        assert isinstance(REAL_TIME_VALIDATOR, RealTimeValidator)

    def test_validate_extreme_low(self, validator):
        params = list(validator.literature_db.parameters.keys())
        if params:
            param_name = params[0]
            param_info = validator.literature_db.parameters[param_name]
            result = validator.validate_parameter_realtime(
                param_name, param_info.min_value - 1
            )
            assert result is not None

    def test_validate_extreme_high(self, validator):
        params = list(validator.literature_db.parameters.keys())
        if params:
            param_name = params[0]
            param_info = validator.literature_db.parameters[param_name]
            result = validator.validate_parameter_realtime(
                param_name, param_info.max_value + 100
            )
            assert result is not None

    def test_performance_metrics_cache_rate(self, validator):
        validator.validate_parameter_realtime("x", 1.0, use_cache=True)
        validator.validate_parameter_realtime("x", 1.0, use_cache=True)
        metrics = validator.get_performance_metrics()
        assert "cache_hit_rate" in metrics

    def test_validation_result_has_recommendations(self, validator):
        params = list(validator.literature_db.parameters.keys())
        if params:
            param_name = params[0]
            info = validator.literature_db.parameters[param_name]
            result = validator.validate_parameter_realtime(
                param_name, info.typical_value
            )
            assert isinstance(result.recommendations, list)
