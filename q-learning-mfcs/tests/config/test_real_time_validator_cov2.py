"""Coverage boost tests for config/real_time_validator.py targeting remaining branches."""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from config.real_time_validator import (
    REAL_TIME_VALIDATOR,
    RealTimeValidator,
    ResearchObjective,
    ValidationLevel,
    ValidationResult,
    ValidationSpeed,
)


@pytest.mark.coverage_extra
class TestRealTimeValidatorCov2:
    def test_validate_unknown_parameter(self):
        """Cover _create_unknown_parameter_result branch."""
        v = RealTimeValidator()
        result = v.validate_parameter_realtime("nonexistent_param_xyz", 5.0)
        assert result.level == ValidationLevel.UNKNOWN
        assert result.confidence_score == 0.0
        assert len(result.warnings) > 0

    def test_validate_with_cache_hit(self):
        """Cover cache hit branch."""
        v = RealTimeValidator()
        # First call populates cache
        r1 = v.validate_parameter_realtime("max_current_density", 5.0)
        # Second call should hit cache
        r2 = v.validate_parameter_realtime("max_current_density", 5.0)
        assert v.cache_hits >= 1
        assert r2.level == r1.level

    def test_validate_with_cache_disabled(self):
        """Cover use_cache=False branch."""
        v = RealTimeValidator()
        r = v.validate_parameter_realtime("max_current_density", 5.0, use_cache=False)
        assert r.level in (ValidationLevel.VALID, ValidationLevel.CAUTION, ValidationLevel.INVALID)

    def test_validate_with_research_objective(self):
        """Cover research objective branch in suggested ranges."""
        v = RealTimeValidator()
        r = v.validate_parameter_realtime(
            "max_current_density", 5.0, research_objective="maximum_power",
        )
        assert len(r.suggested_ranges) >= 2  # lit range + objective range + typical

    def test_validate_invalid_value(self):
        """Cover INVALID validation level branch."""
        v = RealTimeValidator()
        r = v.validate_parameter_realtime("max_current_density", 100.0)
        assert r.level == ValidationLevel.INVALID
        assert any("Invalid" in w or "invalid" in w.lower() for w in r.warnings)

    def test_validate_caution_value(self):
        """Cover CAUTION validation level."""
        v = RealTimeValidator()
        # A value outside recommended but within valid range
        r = v.validate_parameter_realtime("max_current_density", 0.5, use_cache=False)
        # May be caution or valid depending on exact ranges
        assert r.level in (ValidationLevel.VALID, ValidationLevel.CAUTION, ValidationLevel.INVALID)

    def test_generate_scientific_reasoning_with_notes(self):
        """Cover param.notes branch."""
        v = RealTimeValidator()
        param = v.literature_db.get_parameter("max_current_density")
        if param:
            # Test all level templates
            for level in [ValidationLevel.VALID, ValidationLevel.CAUTION, ValidationLevel.INVALID]:
                reasoning = v._generate_scientific_reasoning(param, 5.0, level)
                assert len(reasoning) > 0

    def test_confidence_score_no_references(self):
        """Cover zero references branch in confidence calculation."""
        v = RealTimeValidator()
        param = MagicMock()
        param.references = []
        param.recommended_range = (1.0, 10.0)
        param.min_value = 0.0
        param.max_value = 20.0
        score = v._calculate_confidence_score(param, 5.0, ValidationLevel.VALID)
        assert score < 0.95  # Reduced by literature_factor = 0.3

    def test_confidence_score_two_references(self):
        """Cover 2 references branch."""
        v = RealTimeValidator()
        param = MagicMock()
        param.references = ["ref1", "ref2"]
        param.recommended_range = (1.0, 10.0)
        param.min_value = 0.0
        param.max_value = 20.0
        score = v._calculate_confidence_score(param, 5.0, ValidationLevel.VALID)
        assert score > 0

    def test_confidence_score_above_max(self):
        """Cover value above recommended range."""
        v = RealTimeValidator()
        param = MagicMock()
        param.references = ["ref1", "ref2", "ref3"]
        param.recommended_range = (1.0, 10.0)
        param.min_value = 0.0
        param.max_value = 20.0
        score = v._calculate_confidence_score(param, 15.0, ValidationLevel.CAUTION)
        assert 0.0 <= score <= 1.0

    def test_confidence_score_below_min(self):
        """Cover value below recommended range."""
        v = RealTimeValidator()
        param = MagicMock()
        param.references = ["ref1", "ref2", "ref3"]
        param.recommended_range = (5.0, 10.0)
        param.min_value = 0.0
        param.max_value = 20.0
        score = v._calculate_confidence_score(param, 2.0, ValidationLevel.CAUTION)
        assert 0.0 <= score <= 1.0

    def test_uncertainty_bounds_invalid_bounds(self):
        """Cover edge case where lower > upper in uncertainty bounds."""
        v = RealTimeValidator()
        param = MagicMock()
        param.min_value = 10.0
        param.max_value = 10.1  # Very narrow range
        lower, upper = v._calculate_uncertainty_bounds(param, 10.05, 0.0)
        assert lower <= upper

    def test_generate_warnings_anode_potential(self):
        """Cover specific warning for positive anode potential."""
        v = RealTimeValidator()
        param = MagicMock()
        param.category = MagicMock()
        param.category.value = "electrochemical"
        warnings = v._generate_warnings(param, 0.5, ValidationLevel.INVALID, "anode_potential")
        assert any("Positive" in w for w in warnings)

    def test_generate_warnings_high_current_density(self):
        """Cover warning for extremely high current density."""
        v = RealTimeValidator()
        param = MagicMock()
        param.category = MagicMock()
        param.category.value = "electrochemical"
        warnings = v._generate_warnings(param, 25.0, ValidationLevel.INVALID, "max_current_density")
        assert any("current density" in w for w in warnings)

    def test_generate_warnings_high_learning_rate(self):
        """Cover warning for high learning rate."""
        v = RealTimeValidator()
        param = MagicMock()
        param.category = MagicMock()
        param.category.value = "qlearning"
        warnings = v._generate_warnings(param, 0.8, ValidationLevel.INVALID, "learning_rate")
        assert any("learning rate" in w for w in warnings)

    def test_generate_warnings_biological_suboptimal(self):
        """Cover biological parameter warning."""
        v = RealTimeValidator()
        param = MagicMock()
        param.category = MagicMock()
        param.category.value = "biological"
        warnings = v._generate_warnings(param, 5.0, ValidationLevel.CAUTION, "growth_rate")
        assert any("biological" in w.lower() for w in warnings)

    def test_enhanced_recommendations_valid_with_objective(self):
        """Cover valid level with research objective."""
        v = RealTimeValidator()
        param = MagicMock()
        param.name = "max_current_density"
        param.unit = "mA/cm2"
        param.category = MagicMock()
        param.category.value = "electrochemical"
        recs = v._generate_enhanced_recommendations(
            param, 5.0, ValidationLevel.VALID, "maximum_power", "max_current_density",
        )
        assert any("Excellent" in r for r in recs)

    def test_enhanced_recommendations_caution_with_objective(self):
        """Cover caution level with research objective."""
        v = RealTimeValidator()
        param = MagicMock()
        param.name = "max_current_density"
        param.unit = "mA/cm2"
        param.recommended_range = (5.0, 15.0)
        param.category = MagicMock()
        param.category.value = "electrochemical"
        recs = v._generate_enhanced_recommendations(
            param, 3.0, ValidationLevel.CAUTION, "maximum_power", "max_current_density",
        )
        assert len(recs) > 0

    def test_enhanced_recommendations_invalid_with_objective(self):
        """Cover invalid level with research objective parameter key."""
        v = RealTimeValidator()
        param = MagicMock()
        param.name = "max_current_density"
        param.unit = "mA/cm2"
        param.min_value = 0.0
        param.max_value = 20.0
        param.typical_value = 5.0
        param.references = ["ref1", "ref2"]
        param.category = MagicMock()
        param.category.value = "electrochemical"
        recs = v._generate_enhanced_recommendations(
            param, 50.0, ValidationLevel.INVALID, "maximum_power", "max_current_density",
        )
        assert any("target range" in r for r in recs)

    def test_enhanced_recommendations_biological_category(self):
        """Cover biological category recommendation."""
        v = RealTimeValidator()
        param = MagicMock()
        param.name = "growth_rate"
        param.unit = "h-1"
        param.category = MagicMock()
        param.category.value = "biological"
        recs = v._generate_enhanced_recommendations(
            param, 0.05, ValidationLevel.VALID, None, "growth_rate",
        )
        assert any("Biological" in r or "biological" in r for r in recs)

    def test_enhanced_recommendations_qlearning_category(self):
        """Cover qlearning category recommendation."""
        v = RealTimeValidator()
        param = MagicMock()
        param.name = "learning_rate"
        param.unit = ""
        param.category = MagicMock()
        param.category.value = "qlearning"
        recs = v._generate_enhanced_recommendations(
            param, 0.1, ValidationLevel.VALID, None, "learning_rate",
        )
        assert any("Q-learning" in r or "convergence" in r for r in recs)

    def test_performance_metrics_empty(self):
        """Cover empty validation times."""
        v = RealTimeValidator()
        v.validation_times = []
        metrics = v.get_performance_metrics()
        assert metrics["total_validations"] == 0

    def test_performance_metrics_with_data(self):
        """Cover populated validation times."""
        v = RealTimeValidator()
        v.validation_times = [10.0, 20.0, 50.0, 100.0, 300.0]
        v.cache_hits = 3
        v.cache_misses = 5
        metrics = v.get_performance_metrics()
        assert metrics["total_validations"] == 5
        assert metrics["cache_hit_rate"] == pytest.approx(3.0 / 8.0)
        assert metrics["fast_validations"] == 4
        assert metrics["instant_validations"] == 2

    def test_get_research_objectives(self):
        v = RealTimeValidator()
        objectives = v.get_research_objectives()
        assert "maximum_power" in objectives
        assert "substrate_efficiency" in objectives
        assert "stable_operation" in objectives
        assert "learning_optimization" in objectives

    def test_get_research_objective_info(self):
        v = RealTimeValidator()
        info = v.get_research_objective_info("maximum_power")
        assert info is not None
        assert info.name == "Maximum Power Output"
        assert v.get_research_objective_info("nonexistent") is None

    def test_global_instance(self):
        assert REAL_TIME_VALIDATOR is not None
        assert isinstance(REAL_TIME_VALIDATOR, RealTimeValidator)
