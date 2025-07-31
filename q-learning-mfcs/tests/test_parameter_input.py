"""
Tests for Literature-Referenced Parameter Input System

Tests the parameter input component with literature validation
for User Story 1.1.1: Literature-Referenced Parameter Input

Created: 2025-07-31
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.literature_database import (
    LITERATURE_DB, 
    ParameterCategory, 
    LiteratureReference
)
from gui.parameter_input import ParameterInputComponent


class TestLiteratureDatabase:
    """Test cases for literature database functionality."""
    
    def test_database_initialization(self):
        """Test that literature database initializes properly."""
        db = LITERATURE_DB
        assert db is not None
        assert len(db.parameters) > 0
        
    def test_parameter_retrieval(self):
        """Test parameter retrieval by name."""
        db = LITERATURE_DB
        
        # Test existing parameter
        anode_param = db.get_parameter('anode_potential')
        assert anode_param is not None
        assert anode_param.name == 'Anode Potential'
        assert anode_param.symbol == 'E_an'
        assert anode_param.unit == 'V vs SHE'
        
        # Test non-existing parameter
        fake_param = db.get_parameter('nonexistent_parameter')
        assert fake_param is None
        
    def test_parameter_categories(self):
        """Test parameter categorization."""
        db = LITERATURE_DB
        
        # Test category retrieval
        electrochemical_params = db.get_parameters_by_category(ParameterCategory.ELECTROCHEMICAL)
        assert len(electrochemical_params) > 0
        
        biological_params = db.get_parameters_by_category(ParameterCategory.BIOLOGICAL)
        assert len(biological_params) > 0
        
        qlearning_params = db.get_parameters_by_category(ParameterCategory.QLEARNING)
        assert len(qlearning_params) > 0
        
    def test_parameter_validation(self):
        """Test parameter value validation."""
        db = LITERATURE_DB
        
        # Test valid parameter value
        validation = db.validate_parameter_value('anode_potential', -0.3)
        assert validation['status'] == 'valid'
        
        # Test caution parameter value (outside recommended but within valid range)
        validation = db.validate_parameter_value('anode_potential', -0.5)
        assert validation['status'] == 'caution'
        
        # Test invalid parameter value
        validation = db.validate_parameter_value('anode_potential', -0.7)
        assert validation['status'] == 'invalid'
        
        # Test unknown parameter
        validation = db.validate_parameter_value('unknown_param', 1.0)
        assert validation['status'] == 'unknown'
        
    def test_literature_references(self):
        """Test literature reference functionality."""
        db = LITERATURE_DB
        
        anode_param = db.get_parameter('anode_potential')
        assert len(anode_param.references) > 0
        
        ref = anode_param.references[0]
        assert isinstance(ref, LiteratureReference)
        assert ref.authors is not None
        assert ref.title is not None
        assert ref.journal is not None
        assert ref.year > 2000  # Reasonable year check
        
    def test_citation_formatting(self):
        """Test citation formatting in different styles."""
        db = LITERATURE_DB
        
        anode_param = db.get_parameter('anode_potential')
        ref = anode_param.references[0]
        
        # Test APA formatting
        apa_citation = ref.format_citation('apa')
        assert ref.authors in apa_citation
        assert str(ref.year) in apa_citation
        assert ref.title in apa_citation
        
        # Test BibTeX formatting
        bibtex_citation = ref.format_citation('bibtex')
        assert '@article' in bibtex_citation
        assert 'author' in bibtex_citation
        assert 'title' in bibtex_citation
        
    def test_parameter_search(self):
        """Test parameter search functionality."""
        db = LITERATURE_DB
        
        # Search by name
        results = db.search_parameters('anode')
        assert len(results) > 0
        assert any('anode' in param.name.lower() for param in results)
        
        # Search by symbol
        results = db.search_parameters('α')
        learning_rate_found = any(param.symbol == 'α' for param in results)
        assert learning_rate_found
        
        # Search by description
        results = db.search_parameters('conductivity')
        assert len(results) > 0
        
    def test_parameter_ranges(self):
        """Test parameter range validation methods."""
        db = LITERATURE_DB
        
        param = db.get_parameter('learning_rate')
        assert param is not None
        
        # Test recommended range check
        assert param.is_within_recommended_range(0.1)
        assert not param.is_within_recommended_range(0.5)  # Outside recommended
        
        # Test valid range check
        assert param.is_valid_value(0.1)
        assert param.is_valid_value(0.5)  # Valid but not recommended
        assert not param.is_valid_value(2.0)  # Invalid
        
    def test_citation_list_generation(self):
        """Test generation of complete citation list."""
        db = LITERATURE_DB
        
        citations = db.get_citation_list('apa')
        assert len(citations) > 0
        assert all(isinstance(citation, str) for citation in citations)
        
        # Check for no duplicates
        assert len(citations) == len(set(citations))


class TestParameterInputComponent:
    """Test cases for parameter input component."""
    
    def test_component_initialization(self):
        """Test parameter input component initialization."""
        component = ParameterInputComponent()
        assert component is not None
        assert component.literature_db is not None
        
    def test_parameter_range_visualization(self):
        """Test parameter range visualization creation."""
        component = ParameterInputComponent()
        
        # This would require Streamlit session state, so we'll test the basic structure
        _ = component.create_parameter_range_visualization('anode_potential')
        # In a real Streamlit environment, this would return a plotly figure
        # For testing purposes, we'll just ensure the method exists and doesn't crash
        
    def test_validation_status_messages(self):
        """Test validation status message generation."""
        db = LITERATURE_DB
        
        # Test different validation statuses
        valid_result = db.validate_parameter_value('anode_potential', -0.3)
        assert 'within recommended range' in valid_result['message']
        
        caution_result = db.validate_parameter_value('anode_potential', -0.5)
        assert 'outside recommended range' in caution_result['message']
        
        invalid_result = db.validate_parameter_value('anode_potential', -0.7)
        assert 'outside valid range' in invalid_result['message']


class TestIntegrationWithQLearningConfig:
    """Test integration with existing Q-learning configuration."""
    
    def test_qlearning_parameter_consistency(self):
        """Test that Q-learning parameters match existing config."""
        db = LITERATURE_DB
        
        # Test that Q-learning parameters exist in database
        learning_rate_param = db.get_parameter('learning_rate')
        assert learning_rate_param is not None
        assert learning_rate_param.category == ParameterCategory.QLEARNING
        
        discount_factor_param = db.get_parameter('discount_factor')
        assert discount_factor_param is not None
        assert discount_factor_param.category == ParameterCategory.QLEARNING
        
        exploration_rate_param = db.get_parameter('exploration_rate')
        assert exploration_rate_param is not None
        assert exploration_rate_param.category == ParameterCategory.QLEARNING
        
    def test_electrochemical_parameter_ranges(self):
        """Test that electrochemical parameters have reasonable ranges."""
        db = LITERATURE_DB
        
        anode_param = db.get_parameter('anode_potential')
        assert anode_param.min_value < anode_param.max_value
        assert anode_param.recommended_range[0] >= anode_param.min_value
        assert anode_param.recommended_range[1] <= anode_param.max_value
        
        cathode_param = db.get_parameter('cathode_potential')
        assert cathode_param.min_value < cathode_param.max_value
        assert cathode_param.typical_value >= cathode_param.min_value
        assert cathode_param.typical_value <= cathode_param.max_value
        
    def test_biological_parameter_validation(self):
        """Test biological parameter validation ranges."""
        db = LITERATURE_DB
        
        current_density_param = db.get_parameter('max_current_density')
        assert current_density_param is not None
        assert current_density_param.unit == 'mA/cm²'
        assert current_density_param.min_value > 0
        
        growth_rate_param = db.get_parameter('growth_rate')
        assert growth_rate_param is not None
        assert growth_rate_param.unit == 'h⁻¹'
        assert growth_rate_param.min_value > 0


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])