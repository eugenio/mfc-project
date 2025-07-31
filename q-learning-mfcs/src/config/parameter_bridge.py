"""
Parameter Bridge for Literature Database and Q-Learning Configuration Integration

This module provides integration between the literature-referenced parameter database
and the existing Q-learning configuration system, ensuring seamless compatibility
while maintaining scientific rigor.

User Story 1.1.1: Integration with existing qlearning_config.py system
Created: 2025-07-31
Last Modified: 2025-07-31
"""

from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import copy

from .qlearning_config import (
    QLearningConfig, 
    DEFAULT_QLEARNING_CONFIG
)
from .literature_database import LITERATURE_DB


@dataclass
class ParameterMappingRule:
    """Rule for mapping literature parameters to configuration parameters."""
    
    literature_name: str  # Name in literature database
    config_attribute: str  # Attribute path in configuration (e.g., 'learning_rate' or 'reward_weights.power_weight')
    conversion_factor: float = 1.0  # Conversion factor if units differ
    validation_required: bool = True  # Whether to validate against literature ranges
    notes: Optional[str] = None


class ParameterBridge:
    """Bridge between literature database and Q-learning configuration."""
    
    def __init__(self):
        """Initialize parameter bridge with mapping rules."""
        self.literature_db = LITERATURE_DB
        self.mapping_rules = self._define_mapping_rules()
        
    def _define_mapping_rules(self) -> Dict[str, ParameterMappingRule]:
        """Define mapping rules between literature parameters and config attributes."""
        
        rules = {
            # Q-Learning Parameters
            'learning_rate': ParameterMappingRule(
                literature_name='learning_rate',
                config_attribute='learning_rate',
                notes='Direct mapping for Q-learning alpha parameter'
            ),
            'discount_factor': ParameterMappingRule(
                literature_name='discount_factor',
                config_attribute='discount_factor',
                notes='Direct mapping for Q-learning gamma parameter'
            ),
            'exploration_rate': ParameterMappingRule(
                literature_name='exploration_rate',
                config_attribute='epsilon',
                notes='Maps to epsilon in epsilon-greedy exploration'
            ),
            
            # Electrochemical Parameters
            'electrode_area': ParameterMappingRule(
                literature_name='electrode_area',
                config_attribute='anode_area_per_cell',
                notes='Maps to anode area per cell'
            ),
            
            # Biological Parameters
            'max_current_density': ParameterMappingRule(
                literature_name='max_current_density',
                config_attribute='biofilm_physics.max_current_density',
                conversion_factor=1e-3,  # Convert mA/cm² to A/m²
                notes='Convert from mA/cm² to A/m² for internal calculations'
            ),
            'biofilm_conductivity': ParameterMappingRule(
                literature_name='biofilm_conductivity',
                config_attribute='biofilm_physics.biofilm_conductivity',
                notes='Direct mapping for biofilm electrical conductivity'
            ),
            'growth_rate': ParameterMappingRule(
                literature_name='growth_rate',
                config_attribute='biofilm_physics.max_specific_growth_rate',
                notes='Maximum specific growth rate for biofilm'
            ),
            'half_saturation': ParameterMappingRule(
                literature_name='half_saturation',
                config_attribute='biofilm_physics.half_saturation_constant',
                notes='Monod half-saturation constant'
            ),
            'biofilm_thickness': ParameterMappingRule(
                literature_name='biofilm_thickness',
                config_attribute='optimal_biofilm_thickness',
                conversion_factor=1e-6,  # Convert μm to m
                notes='Convert from μm to m for calculations'
            ),
            'biofilm_density': ParameterMappingRule(
                literature_name='biofilm_density',
                config_attribute='biofilm_physics.biofilm_density',
                notes='Wet density of bacterial biofilm'
            ),
            
            # Substrate Parameters
            'substrate_concentration': ParameterMappingRule(
                literature_name='substrate_concentration',
                config_attribute='substrate_target_concentration',
                notes='Target substrate concentration'
            ),
            'flow_rate': ParameterMappingRule(
                literature_name='flow_rate',
                config_attribute='stability_target_flow_rate',
                notes='Target flow rate for stability'
            )
        }
        
        return rules
    
    def validate_parameter_with_literature(self, config_key: str, value: float) -> Dict[str, Any]:
        """
        Validate a configuration parameter against literature ranges.
        
        Args:
            config_key: Configuration parameter key
            value: Parameter value to validate
            
        Returns:
            Validation result dictionary
        """
        if config_key not in self.mapping_rules:
            return {
                'status': 'unmapped',
                'message': f'Parameter {config_key} not mapped to literature database',
                'recommendations': []
            }
        
        rule = self.mapping_rules[config_key]
        
        # Convert value if needed
        literature_value = value / rule.conversion_factor
        
        # Validate against literature database
        validation = self.literature_db.validate_parameter_value(
            rule.literature_name, 
            literature_value
        )
        
        # Add conversion information to validation result
        if rule.conversion_factor != 1.0:
            validation['converted_value'] = literature_value
            validation['conversion_factor'] = rule.conversion_factor
            validation['original_value'] = value
        
        validation['mapping_rule'] = rule
        
        return validation
    
    def create_literature_validated_config(self, parameter_values: Dict[str, float]) -> Tuple[QLearningConfig, Dict[str, Any]]:
        """
        Create a Q-learning configuration from literature-validated parameters.
        
        Args:
            parameter_values: Dictionary of literature parameter values
            
        Returns:
            Tuple of (validated config, validation results)
        """
        # Start with default configuration
        config = copy.deepcopy(DEFAULT_QLEARNING_CONFIG)
        validation_results = {}
        
        # Apply parameter values with validation
        for lit_param_name, value in parameter_values.items():
            # Find mapping rule for this literature parameter
            mapping_rule = None
            
            for key, rule in self.mapping_rules.items():
                if rule.literature_name == lit_param_name:
                    mapping_rule = rule
                    break
            
            if not mapping_rule:
                validation_results[lit_param_name] = {
                    'status': 'unmapped',
                    'message': f'No mapping rule found for literature parameter {lit_param_name}'
                }
                continue
            
            # Validate parameter
            if mapping_rule.validation_required:
                validation = self.literature_db.validate_parameter_value(lit_param_name, value)
                validation_results[lit_param_name] = validation
                
                # Only apply valid parameters
                if validation['status'] not in ['valid', 'caution']:
                    continue
            
            # Apply parameter to configuration
            config_value = value * mapping_rule.conversion_factor
            self._set_nested_attribute(config, mapping_rule.config_attribute, config_value)
        
        return config, validation_results
    
    def _set_nested_attribute(self, obj: Any, attr_path: str, value: Any):
        """Set nested attribute using dot notation."""
        attrs = attr_path.split('.')
        current_obj = obj
        
        # Navigate to the parent object
        for attr in attrs[:-1]:
            current_obj = getattr(current_obj, attr)
        
        # Set the final attribute
        setattr(current_obj, attrs[-1], value)
    
    def _get_nested_attribute(self, obj: Any, attr_path: str) -> Any:
        """Get nested attribute using dot notation."""
        attrs = attr_path.split('.')
        current_obj = obj
        
        for attr in attrs:
            current_obj = getattr(current_obj, attr)
        
        return current_obj
    
    def extract_literature_parameters_from_config(self, config: QLearningConfig) -> Dict[str, float]:
        """
        Extract literature parameter values from an existing configuration.
        
        Args:
            config: Q-learning configuration
            
        Returns:
            Dictionary of literature parameter values
        """
        literature_params = {}
        
        for config_key, rule in self.mapping_rules.items():
            try:
                config_value = self._get_nested_attribute(config, rule.config_attribute)
                literature_value = config_value / rule.conversion_factor
                literature_params[rule.literature_name] = literature_value
            except AttributeError:
                # Attribute doesn't exist in this configuration
                continue
        
        return literature_params
    
    def get_parameter_mapping_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about parameter mappings for documentation.
        
        Returns:
            Dictionary with mapping information
        """
        mapping_info = {}
        
        for config_key, rule in self.mapping_rules.items():
            lit_param = self.literature_db.get_parameter(rule.literature_name)
            
            mapping_info[config_key] = {
                'literature_parameter': {
                    'name': lit_param.name if lit_param else rule.literature_name,
                    'symbol': lit_param.symbol if lit_param else '',
                    'unit': lit_param.unit if lit_param else '',
                    'category': lit_param.category.value if lit_param else ''
                },
                'config_attribute': rule.config_attribute,
                'conversion_factor': rule.conversion_factor,
                'notes': rule.notes
            }
        
        return mapping_info
    
    def validate_entire_config(self, config: QLearningConfig) -> Dict[str, Any]:
        """
        Validate an entire configuration against literature parameters.
        
        Args:
            config: Q-learning configuration to validate
            
        Returns:
            Complete validation report
        """
        literature_params = self.extract_literature_parameters_from_config(config)
        validation_report: Dict[str, Any] = {
            'overall_status': 'valid',
            'parameter_results': {},
            'summary': {
                'total_parameters': 0,
                'valid_parameters': 0,
                'caution_parameters': 0,
                'invalid_parameters': 0,
                'unmapped_parameters': 0
            }
        }
        
        for lit_param_name, value in literature_params.items():
            validation = self.literature_db.validate_parameter_value(lit_param_name, value)
            validation_report['parameter_results'][lit_param_name] = validation
            
            # Update summary
            validation_report['summary']['total_parameters'] += 1
            status = validation['status']
            
            if status == 'valid':
                validation_report['summary']['valid_parameters'] += 1
            elif status == 'caution':
                validation_report['summary']['caution_parameters'] += 1
            elif status == 'invalid':
                validation_report['summary']['invalid_parameters'] += 1
                validation_report['overall_status'] = 'invalid'
            else:
                validation_report['summary']['unmapped_parameters'] += 1
        
        # Set overall status
        if validation_report['summary']['invalid_parameters'] > 0:
            validation_report['overall_status'] = 'invalid'
        elif validation_report['summary']['caution_parameters'] > 0:
            validation_report['overall_status'] = 'caution'
        
        return validation_report
    
    def suggest_parameter_improvements(self, config: QLearningConfig) -> List[Dict[str, Any]]:
        """
        Suggest improvements to configuration parameters based on literature.
        
        Args:
            config: Current configuration
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        validation_report = self.validate_entire_config(config)
        
        for param_name, validation in validation_report['parameter_results'].items():
            if validation['status'] in ['caution', 'invalid']:
                lit_param = self.literature_db.get_parameter(param_name)
                if lit_param:
                    suggestion = {
                        'parameter': param_name,
                        'current_status': validation['status'],
                        'current_value': validation['value'],
                        'recommended_range': lit_param.recommended_range,
                        'typical_value': lit_param.typical_value,
                        'suggestion': f"Consider using values between {lit_param.recommended_range[0]} and {lit_param.recommended_range[1]} {lit_param.unit}",
                        'literature_references': [ref.format_citation('apa') for ref in lit_param.references]
                    }
                    suggestions.append(suggestion)
        
        return suggestions


# Global instance for easy access
PARAMETER_BRIDGE = ParameterBridge()