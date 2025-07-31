"""
Tests for Real-Time Parameter Validation System

This module tests the enhanced real-time validation functionality,
including performance requirements, uncertainty quantification,
and scientific reasoning.

User Story 1.1.2: Real-Time Parameter Validation
Created: 2025-07-31
Last Modified: 2025-07-31
"""

import unittest
import time

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.real_time_validator import (
    REAL_TIME_VALIDATOR,
    ValidationLevel,
    ValidationResult,
    ResearchObjective
)


class TestRealTimeValidator(unittest.TestCase):
    """Test cases for real-time parameter validation."""

    def setUp(self):
        """Set up test environment."""
        self.validator = REAL_TIME_VALIDATOR
        # Clear cache for consistent testing
        self.validator.validation_cache.clear()
        self.validator.validation_times.clear()
        self.validator.cache_hits = 0
        self.validator.cache_misses = 0

    def test_instant_validation_feedback(self):
        """Test that validation feedback is provided within 200ms."""
        start_time = time.time()

        result = self.validator.validate_parameter_realtime('learning_rate', 0.1)

        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000

        # Check response time requirement
        self.assertLess(response_time_ms, 200, "Validation should complete within 200ms")
        self.assertLess(result.response_time_ms, 200, "Reported response time should be <200ms")

        # Verify result structure
        self.assertIsInstance(result, ValidationResult)
        self.assertIsInstance(result.level, ValidationLevel)
        self.assertIsInstance(result.message, str)
        self.assertIsInstance(result.scientific_reasoning, str)

    def test_color_coded_validation_status(self):
        """Test color-coded validation status levels."""

        # Test valid parameter (should be green/VALID)
        valid_result = self.validator.validate_parameter_realtime('learning_rate', 0.1)
        self.assertEqual(valid_result.level, ValidationLevel.VALID)

        # Test caution parameter (should be yellow/CAUTION)
        caution_result = self.validator.validate_parameter_realtime('learning_rate', 0.01)
        self.assertEqual(caution_result.level, ValidationLevel.CAUTION)

        # Test invalid parameter (should be red/INVALID)
        invalid_result = self.validator.validate_parameter_realtime('learning_rate', 2.0)
        self.assertEqual(invalid_result.level, ValidationLevel.INVALID)

        # Test unknown parameter (should be gray/UNKNOWN)
        unknown_result = self.validator.validate_parameter_realtime('nonexistent_param', 1.0)
        self.assertEqual(unknown_result.level, ValidationLevel.UNKNOWN)

    def test_contextual_help_with_scientific_reasoning(self):
        """Test that validation results include scientific reasoning."""

        result = self.validator.validate_parameter_realtime('anode_potential', -0.3)

        # Check scientific reasoning is provided
        self.assertIsInstance(result.scientific_reasoning, str)
        self.assertGreater(len(result.scientific_reasoning), 20, "Scientific reasoning should be detailed")

        # Should mention scientific concepts
        reasoning_lower = result.scientific_reasoning.lower()
        self.assertTrue(
            any(term in reasoning_lower for term in [
                'scientific', 'research', 'literature', 'study', 'peer-reviewed',
                'documented', 'validated', 'optimal', 'electrochemical'
            ]),
            "Scientific reasoning should contain scientific terminology"
        )

    def test_suggested_parameter_ranges_research_objectives(self):
        """Test parameter suggestions based on research objectives."""

        # Test without research objective
        result_no_obj = self.validator.validate_parameter_realtime('max_current_density', 8.0)
        self.assertGreater(len(result_no_obj.suggested_ranges), 0)

        # Test with maximum power research objective
        result_power = self.validator.validate_parameter_realtime(
            'max_current_density',
            8.0,
            research_objective='maximum_power'
        )
        self.assertGreater(len(result_power.suggested_ranges), len(result_no_obj.suggested_ranges))

        # Verify recommendations include research objective context
        power_recommendations = ' '.join(result_power.recommendations).lower()
        self.assertTrue(
            any(term in power_recommendations for term in ['power', 'maximum', 'target']),
            "Recommendations should include research objective context"
        )

    def test_warning_messages_problematic_combinations(self):
        """Test warning generation for problematic parameter values."""

        # Test extreme values that should generate warnings
        extreme_result = self.validator.validate_parameter_realtime('max_current_density', 50.0)
        self.assertGreater(len(extreme_result.warnings), 0, "Extreme values should generate warnings")

        # Test positive anode potential (thermodynamically unfavorable)
        anode_result = self.validator.validate_parameter_realtime('anode_potential', 0.5)
        self.assertGreater(len(anode_result.warnings), 0)

        # Check warning content
        anode_warnings = ' '.join(anode_result.warnings).lower()
        self.assertTrue(
            any(term in anode_warnings for term in ['positive', 'unfavorable', 'thermodynamic']),
            "Anode potential warnings should mention thermodynamic issues"
        )

    def test_uncertainty_quantification_integration(self):
        """Test integration with uncertainty quantification system."""

        # Test high confidence parameter
        high_conf_result = self.validator.validate_parameter_realtime('learning_rate', 0.1)
        self.assertGreaterEqual(high_conf_result.confidence_score, 0.8)

        # Check uncertainty bounds
        self.assertIsInstance(high_conf_result.uncertainty_bounds, tuple)
        self.assertEqual(len(high_conf_result.uncertainty_bounds), 2)

        lower, upper = high_conf_result.uncertainty_bounds
        self.assertLessEqual(lower, 0.1)
        self.assertGreaterEqual(upper, 0.1)
        self.assertLess(lower, upper)

        # Test low confidence parameter
        low_conf_result = self.validator.validate_parameter_realtime('learning_rate', 1.5)
        self.assertLess(low_conf_result.confidence_score, 0.5)

        # Lower confidence should have wider uncertainty bounds
        low_lower, low_upper = low_conf_result.uncertainty_bounds
        high_range = upper - lower
        low_range = low_upper - low_lower
        self.assertGreater(low_range, high_range, "Lower confidence should have wider uncertainty")

    def test_performance_metrics_tracking(self):
        """Test performance metrics tracking and reporting."""

        # Generate some validation calls
        for i in range(10):
            self.validator.validate_parameter_realtime('learning_rate', 0.1 + i * 0.01)

        metrics = self.validator.get_performance_metrics()

        # Check metrics structure
        expected_keys = [
            'avg_response_time_ms', 'max_response_time_ms', 'min_response_time_ms',
            'cache_hit_rate', 'total_validations', 'fast_validations', 'instant_validations'
        ]

        for key in expected_keys:
            self.assertIn(key, metrics)

        # Check performance targets
        self.assertLessEqual(metrics['avg_response_time_ms'], 200, "Average response time should be ≤200ms")
        self.assertEqual(metrics['total_validations'], 10)
        self.assertGreaterEqual(metrics['fast_validations'], 8, "Most validations should be fast (<200ms)")

    def test_cache_performance(self):
        """Test validation cache performance."""

        # First call should be cache miss
        result1 = self.validator.validate_parameter_realtime('learning_rate', 0.1, use_cache=True)
        self.assertEqual(self.validator.cache_misses, 1)
        self.assertEqual(self.validator.cache_hits, 0)

        # Second identical call should be cache hit
        result2 = self.validator.validate_parameter_realtime('learning_rate', 0.1, use_cache=True)
        self.assertEqual(self.validator.cache_hits, 1)

        # Results should be identical
        self.assertEqual(result1.level, result2.level)
        self.assertEqual(result1.confidence_score, result2.confidence_score)

    def test_research_objectives_functionality(self):
        """Test research objectives management."""

        objectives = self.validator.get_research_objectives()
        self.assertIsInstance(objectives, list)
        self.assertGreater(len(objectives), 0)

        # Test getting objective info
        for obj_name in objectives:
            obj_info = self.validator.get_research_objective_info(obj_name)
            self.assertIsInstance(obj_info, ResearchObjective)
            self.assertIsInstance(obj_info.name, str)
            self.assertIsInstance(obj_info.description, str)
            self.assertIsInstance(obj_info.priority_parameters, list)
            self.assertIsInstance(obj_info.target_ranges, dict)

    def test_validation_result_completeness(self):
        """Test that validation results contain all required information."""

        result = self.validator.validate_parameter_realtime(
            'substrate_concentration',
            25.0,
            research_objective='substrate_efficiency'
        )

        # Check all required fields are present
        self.assertIsInstance(result.level, ValidationLevel)
        self.assertIsInstance(result.message, str)
        self.assertIsInstance(result.scientific_reasoning, str)
        self.assertIsInstance(result.suggested_ranges, list)
        self.assertIsInstance(result.confidence_score, float)
        self.assertIsInstance(result.uncertainty_bounds, tuple)
        self.assertIsInstance(result.response_time_ms, float)
        self.assertIsInstance(result.recommendations, list)
        self.assertIsInstance(result.warnings, list)

        # Check value ranges
        self.assertGreaterEqual(result.confidence_score, 0.0)
        self.assertLessEqual(result.confidence_score, 1.0)
        self.assertGreater(result.response_time_ms, 0.0)

        # Check content quality
        self.assertGreater(len(result.message), 5)
        self.assertGreater(len(result.scientific_reasoning), 20)
        self.assertGreater(len(result.recommendations), 0)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests for real-time validation."""

    def setUp(self):
        """Set up benchmark environment."""
        self.validator = REAL_TIME_VALIDATOR
        self.validator.validation_cache.clear()

    def test_batch_validation_performance(self):
        """Test performance under batch validation load."""

        parameters = [
            ('learning_rate', 0.1),
            ('discount_factor', 0.95),
            ('exploration_rate', 0.3),
            ('max_current_density', 10.0),
            ('substrate_concentration', 25.0)
        ]

        start_time = time.time()

        results = []
        for param_name, value in parameters:
            result = self.validator.validate_parameter_realtime(param_name, value)
            results.append(result)

        end_time = time.time()
        total_time_ms = (end_time - start_time) * 1000
        avg_time_per_validation = total_time_ms / len(parameters)

        # Performance assertions
        self.assertLess(avg_time_per_validation, 200, f"Average validation time should be <200ms, got {avg_time_per_validation:.1f}ms")
        self.assertLess(total_time_ms, 1000, f"Total batch time should be <1s, got {total_time_ms:.1f}ms")

        # Verify all results are valid
        for result in results:
            self.assertIsInstance(result, ValidationResult)
            self.assertLess(result.response_time_ms, 300)  # Individual validation should be fast


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
