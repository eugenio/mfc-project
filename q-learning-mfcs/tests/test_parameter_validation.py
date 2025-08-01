#!/usr/bin/env python3
"""
Comprehensive test suite for Parameter Validation System

Tests for User Stories 1.1.1 and 1.1.2: Literature-Referenced Parameter Input and Real-Time Validation
Created: 2025-07-31
"""

import pytest

# Import the modules to test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.literature_database import (
    LITERATURE_DB,
    ParameterCategory,
    LiteratureReference,
    ParameterInfo
)

from src.config.parameter_bridge import (
    ParameterBridge,
    ParameterMappingRule
)


class TestLiteratureDatabase:
    """Test literature database functionality."""

    def test_literature_database_structure(self):
        """Test that literature database has correct structure."""
        assert isinstance(LITERATURE_DB.parameters, dict)
        assert len(LITERATURE_DB.parameters) > 10  # Should have many parameters

        # Check that we have the expected categories
        categories = {param.category for param in LITERATURE_DB.parameters.values()}
        expected_categories = {ParameterCategory.ELECTROCHEMICAL, ParameterCategory.BIOLOGICAL, ParameterCategory.QLEARNING}
        assert expected_categories.issubset(categories)

    def test_parameter_database_entries(self):
        """Test individual parameter database entries."""
        # Test a known parameter
        param = LITERATURE_DB.get_parameter('anode_potential')
        if param:
            # Check required fields
            assert param.name is not None
            assert param.symbol is not None
            assert param.unit is not None
            assert param.description is not None
            assert param.category in ParameterCategory
            assert isinstance(param.references, list)
            assert len(param.references) > 0

            # Check value types and ranges
            assert isinstance(param.min_value, (int, float))
            assert isinstance(param.max_value, (int, float))
            assert param.min_value < param.max_value
            assert param.recommended_range[0] >= param.min_value
            assert param.recommended_range[1] <= param.max_value

    def test_get_parameter(self):
        """Test getting parameter by name."""
        # Test existing parameter
        param = LITERATURE_DB.get_parameter('learning_rate')
        if param:
            assert param.name == 'Learning Rate'
            assert param.category == ParameterCategory.QLEARNING

        # Test non-existent parameter
        param = LITERATURE_DB.get_parameter('nonexistent_param')
        assert param is None

    def test_get_parameters_by_category(self):
        """Test getting parameters by category."""
        qlearning_params = LITERATURE_DB.get_parameters_by_category(ParameterCategory.QLEARNING)
        assert isinstance(qlearning_params, list)
        assert len(qlearning_params) > 0

        for param in qlearning_params:
            assert param.category == ParameterCategory.QLEARNING

    def test_validate_parameter_value(self):
        """Test parameter value validation."""
        # Test with learning rate parameter
        result = LITERATURE_DB.validate_parameter_value('learning_rate', 0.1)

        assert 'status' in result
        assert result['status'] in ['valid', 'caution', 'invalid']
        assert 'parameter' in result
        assert 'value' in result
        assert 'recommendations' in result

        # Test with invalid value
        result = LITERATURE_DB.validate_parameter_value('learning_rate', -1.0)
        assert result['status'] == 'invalid'

        # Test with unknown parameter
        result = LITERATURE_DB.validate_parameter_value('unknown_param', 1.0)
        assert result['status'] == 'unknown'

    def test_search_parameters(self):
        """Test parameter search functionality."""
        # Search for learning-related parameters
        results = LITERATURE_DB.search_parameters('learning')
        assert isinstance(results, list)
        assert len(results) > 0

        # All results should contain 'learning' in name, symbol, or description
        for param in results:
            search_text = f"{param.name} {param.symbol} {param.description}".lower()
            assert 'learning' in search_text

    def test_get_citation_list(self):
        """Test getting formatted citation list."""
        citations = LITERATURE_DB.get_citation_list()
        assert isinstance(citations, list)
        assert len(citations) > 0

        # Check that citations are formatted strings
        for citation in citations:
            assert isinstance(citation, str)
            assert len(citation) > 20  # Should be meaningful citations


class TestParameterInfo:
    """Test ParameterInfo dataclass functionality."""

    def test_parameter_info_creation(self):
        """Test creating ParameterInfo with all fields."""
        ref = LiteratureReference(
            authors="Test Author",
            title="Test Title",
            journal="Test Journal",
            year=2025,
            volume="1",
            pages="1-10"
        )

        param = ParameterInfo(
            name="Test Parameter",
            symbol="T_p",
            description="A test parameter",
            unit="units",
            typical_value=1.0,
            min_value=0.0,
            max_value=2.0,
            recommended_range=(0.5, 1.5),
            category=ParameterCategory.QLEARNING,
            references=[ref],
            notes="Test notes"
        )

        assert param.name == "Test Parameter"
        assert param.symbol == "T_p"
        assert param.category == ParameterCategory.QLEARNING
        assert len(param.references) == 1

    def test_parameter_validation_methods(self):
        """Test parameter validation methods."""
        param = ParameterInfo(
            name="Test",
            symbol="T",
            description="Test",
            unit="test",
            typical_value=10.0,
            min_value=0.0,
            max_value=20.0,
            recommended_range=(5.0, 15.0),
            category=ParameterCategory.QLEARNING,
            references=[]
        )

        # Test valid value
        assert param.is_valid_value(10.0)
        assert param.is_within_recommended_range(10.0)
        assert param.get_validation_status(10.0) == "valid"

        # Test caution value (valid but outside recommended)
        assert param.is_valid_value(2.0)
        assert not param.is_within_recommended_range(2.0)
        assert param.get_validation_status(2.0) == "caution"

        # Test invalid value
        assert not param.is_valid_value(-1.0)
        assert param.get_validation_status(-1.0) == "invalid"


class TestLiteratureReference:
    """Test LiteratureReference dataclass functionality."""

    def test_literature_reference_creation(self):
        """Test creating LiteratureReference."""
        ref = LiteratureReference(
            authors="Smith, J., Doe, A.",
            title="Advanced MFC Research",
            journal="Journal of Bioelectrochemistry",
            year=2024,
            volume="10",
            pages="123-145",
            doi="10.1000/test.doi"
        )

        assert ref.authors == "Smith, J., Doe, A."
        assert ref.year == 2024
        assert ref.doi == "10.1000/test.doi"

    def test_format_citation_apa(self):
        """Test APA citation formatting."""
        ref = LiteratureReference(
            authors="Smith, J.",
            title="Test Title",
            journal="Test Journal",
            year=2024,
            volume="1",
            pages="1-10"
        )

        citation = ref.format_citation("apa")
        assert "Smith, J. (2024)" in citation
        assert "Test Title" in citation
        assert "Test Journal" in citation

    def test_format_citation_bibtex(self):
        """Test BibTeX citation formatting."""
        ref = LiteratureReference(
            authors="Smith, J.",
            title="Test Title",
            journal="Test Journal",
            year=2024,
            volume="1",
            pages="1-10"
        )

        citation = ref.format_citation("bibtex")
        assert "@article{" in citation
        assert "author = {Smith, J.}" in citation
        assert "year = {2024}" in citation


class TestParameterBridge:
    """Test ParameterBridge functionality."""

    @pytest.fixture
    def bridge(self):
        """Create a ParameterBridge instance."""
        return ParameterBridge()

    def test_parameter_bridge_initialization(self, bridge):
        """Test ParameterBridge initialization."""
        assert bridge.literature_db is not None
        assert hasattr(bridge, 'mapping_rules')
        assert isinstance(bridge.mapping_rules, dict)

    def test_mapping_rules_structure(self, bridge):
        """Test that mapping rules have correct structure."""
        for rule_name, rule in bridge.mapping_rules.items():
            assert isinstance(rule, ParameterMappingRule)
            assert hasattr(rule, 'literature_name')
            assert hasattr(rule, 'config_attribute')
            assert hasattr(rule, 'conversion_factor')
            assert hasattr(rule, 'validation_required')


class TestParameterMappingRule:
    """Test ParameterMappingRule dataclass."""

    def test_mapping_rule_creation(self):
        """Test creating ParameterMappingRule."""
        rule = ParameterMappingRule(
            literature_name='test_param',
            config_attribute='test_attr',
            conversion_factor=2.0,
            validation_required=True,
            notes='Test notes'
        )

        assert rule.literature_name == 'test_param'
        assert rule.config_attribute == 'test_attr'
        assert rule.conversion_factor == 2.0
        assert rule.validation_required
        assert rule.notes == 'Test notes'


class TestIntegrationScenarios:
    """Test integration scenarios for parameter validation system."""

    @pytest.fixture
    def bridge(self):
        """Create a ParameterBridge instance."""
        return ParameterBridge()

    def test_mfc_parameter_validation_scenario(self, bridge):
        """Test realistic MFC parameter validation scenario."""
        # Test typical MFC parameters that should exist in the database
        test_parameters = ['learning_rate', 'discount_factor', 'exploration_rate']

        for param_name in test_parameters:
            param = LITERATURE_DB.get_parameter(param_name)
            if param:  # Only test if parameter exists
                # Test valid value
                mid_value = (param.min_value + param.max_value) / 2
                result = LITERATURE_DB.validate_parameter_value(param_name, mid_value)
                assert result['status'] in ['valid', 'caution']

                # Test invalid value
                invalid_value = param.max_value + 10
                result = LITERATURE_DB.validate_parameter_value(param_name, invalid_value)
                assert result['status'] in ['invalid', 'caution']

    def test_parameter_boundary_validation(self):
        """Test validation at parameter boundaries."""
        param = LITERATURE_DB.get_parameter('learning_rate')
        if param:
            # Test at minimum boundary
            result = LITERATURE_DB.validate_parameter_value('learning_rate', param.min_value)
            assert result['status'] in ['valid', 'caution']

            # Test at maximum boundary
            result = LITERATURE_DB.validate_parameter_value('learning_rate', param.max_value)
            assert result['status'] in ['valid', 'caution']

            # Test below minimum
            result = LITERATURE_DB.validate_parameter_value('learning_rate', param.min_value - 0.1)
            assert result['status'] in ['invalid', 'caution']

            # Test above maximum
            result = LITERATURE_DB.validate_parameter_value('learning_rate', param.max_value + 0.1)
            assert result['status'] in ['invalid', 'caution']

    def test_real_time_validation_performance(self):
        """Test that validation meets reasonable performance requirements."""
        import time

        # Test validation performance
        validation_times = []

        for i in range(10):  # 10 validation calls
            start_time = time.time()
            result = LITERATURE_DB.validate_parameter_value('learning_rate', 0.1)
            end_time = time.time()

            validation_time = (end_time - start_time) * 1000  # Convert to ms
            validation_times.append(validation_time)
            assert result is not None

        # Check that average validation time is reasonable (< 50ms)
        avg_time = sum(validation_times) / len(validation_times)
        assert avg_time < 50, f"Average validation time {avg_time:.2f}ms too slow"

    def test_parameter_categories_completeness(self):
        """Test that parameters have proper category assignments."""
        all_categories = set()
        for param in LITERATURE_DB.parameters.values():
            all_categories.add(param.category)

        # Check that we have the main categories
        expected_categories = {ParameterCategory.ELECTROCHEMICAL, ParameterCategory.BIOLOGICAL, ParameterCategory.QLEARNING}
        intersection = expected_categories.intersection(all_categories)
        assert len(intersection) > 0, "Should have at least some expected categories"

    def test_scientific_unit_validation(self):
        """Test that parameters have appropriate scientific units."""
        # Expected unit patterns for scientific validation
        expected_units = ['V', 'mV', 'A', 'mA', 'µA', 'Ω', 'kΩ', 'MΩ',
                         'mM', 'M', 'g/L', 'mg/L', 's', 'min', 'h', 'day', '°C', 'K', 'dimensionless']

        units_found = set()
        for param in LITERATURE_DB.parameters.values():
            if param.unit:
                units_found.add(param.unit)

        # Should have found some scientific units
        assert len(units_found) > 3, "Should have diverse scientific units"

        # Check that some common scientific units are present
        common_units = {'V', 'mM', 'dimensionless', 'h⁻¹', 'S/m'}
        found_common = common_units.intersection(units_found)
        assert len(found_common) > 0, f"Should have some common units, found: {units_found}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
