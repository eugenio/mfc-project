"""
Comprehensive Configuration Management System

This module provides a centralized configuration management system for the MFC project,
including loading, validation, merging, and dynamic configuration updates.

Classes:
- ConfigManager: Central configuration management system
- ConfigProfile: Configuration profile management
- ConfigValidator: Configuration validation utilities
- ConfigMerger: Configuration merging and inheritance utilities

Features:
- YAML/JSON configuration file support
- Configuration validation and type checking
- Configuration profiles and inheritance
- Dynamic configuration updates
- Configuration versioning and migration
- Environment variable substitution
- Configuration caching and performance optimization

Literature References:
1. Fowler, M. (2013). "Refactoring: Improving the Design of Existing Code"
2. Hunt, A., & Thomas, D. (1999). "The Pragmatic Programmer"
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, TypeVar
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging
from abc import ABC, abstractmethod

# Import all configuration classes
from .control_config import ControlSystemConfig
from .visualization_config import VisualizationConfig

# Type variable for generic configuration handling
T = TypeVar('T')


class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass


class ConfigurationValidationError(ConfigurationError):
    """Exception for configuration validation errors."""
    pass


class ConfigurationLoadError(ConfigurationError):
    """Exception for configuration loading errors."""
    pass


@dataclass
class ConfigProfile:
    """Configuration profile containing all system configurations."""

    # Core configurations
    biological: Optional[Dict[str, Any]] = None
    control: Optional[ControlSystemConfig] = None
    visualization: Optional[VisualizationConfig] = None

    # Profile metadata
    profile_name: str = "default"
    profile_version: str = "1.0.0"
    description: str = "Default configuration profile"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Profile inheritance
    inherits_from: Optional[str] = None  # Parent profile name
    override_keys: List[str] = field(default_factory=list)  # Keys that override parent

    # Environment and context
    environment: str = "development"  # development, testing, production
    context: Dict[str, Any] = field(default_factory=dict)  # Additional context

    def update_timestamp(self):
        """Update the profile timestamp."""
        self.updated_at = datetime.now()


class ConfigValidator(ABC):
    """Abstract base class for configuration validators."""

    @abstractmethod
    def validate(self, config: Any) -> bool:
        """
        Validate configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            bool: True if valid
            
        Raises:
            ConfigurationValidationError: If configuration is invalid
        """
        pass


class BiologicalConfigValidator(ConfigValidator):
    """Validator for biological configurations."""

    def validate(self, config: Dict[str, Any]) -> bool:
        """Validate biological configuration dictionary."""
        if not isinstance(config, dict):
            raise ConfigurationValidationError("Biological config must be a dictionary")

        # Validate species configurations
        if 'species_configs' in config:
            for species_name, species_config in config['species_configs'].items():
                if not isinstance(species_config, dict):
                    raise ConfigurationValidationError(f"Species config for {species_name} must be a dictionary")

                # Check required fields
                required_fields = ['max_growth_rate', 'electron_transport_efficiency']
                for field in required_fields:
                    if field not in species_config:
                        raise ConfigurationValidationError(f"Missing required field {field} in species {species_name}")

        # Validate substrate configurations
        if 'substrate_configs' in config:
            for substrate_name, substrate_config in config['substrate_configs'].items():
                if not isinstance(substrate_config, dict):
                    raise ConfigurationValidationError(f"Substrate config for {substrate_name} must be a dictionary")

        return True


class ControlConfigValidator(ConfigValidator):
    """Validator for control system configurations."""

    def validate(self, config: ControlSystemConfig) -> bool:
        """Validate control system configuration."""
        from .control_config import validate_control_config
        return validate_control_config(config)


class VisualizationConfigValidator(ConfigValidator):
    """Validator for visualization configurations."""

    def validate(self, config: VisualizationConfig) -> bool:
        """Validate visualization configuration."""
        from .visualization_config import validate_visualization_config
        return validate_visualization_config(config)


class ConfigMerger:
    """Utility class for merging configurations."""

    @staticmethod
    def merge_dicts(base: Dict[str, Any], override: Dict[str, Any],
                   deep_merge: bool = True) -> Dict[str, Any]:
        """
        Merge two dictionaries with optional deep merging.
        
        Args:
            base: Base dictionary
            override: Override dictionary
            deep_merge: Whether to perform deep merging
            
        Returns:
            Merged dictionary
        """
        if not deep_merge:
            result = base.copy()
            result.update(override)
            return result

        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigMerger.merge_dicts(result[key], value, deep_merge=True)
            else:
                result[key] = value

        return result

    @staticmethod
    def merge_profiles(base_profile: ConfigProfile,
                      override_profile: ConfigProfile) -> ConfigProfile:
        """
        Merge two configuration profiles.
        
        Args:
            base_profile: Base profile
            override_profile: Override profile
            
        Returns:
            Merged profile
        """
        merged = ConfigProfile()

        # Merge biological configurations
        if base_profile.biological and override_profile.biological:
            merged.biological = ConfigMerger.merge_dicts(
                base_profile.biological, override_profile.biological)
        else:
            merged.biological = override_profile.biological or base_profile.biological

        # Use override for other configurations (can be extended for deep merging)
        merged.control = override_profile.control or base_profile.control
        merged.visualization = override_profile.visualization or base_profile.visualization

        # Update metadata
        merged.profile_name = override_profile.profile_name
        merged.profile_version = override_profile.profile_version
        merged.description = override_profile.description
        merged.environment = override_profile.environment
        merged.context = ConfigMerger.merge_dicts(
            base_profile.context, override_profile.context)

        merged.update_timestamp()

        return merged


class ConfigManager:
    """Central configuration management system."""

    def __init__(self, config_directory: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_directory: Directory containing configuration files
        """
        self.config_directory = Path(config_directory or "configs")
        self.profiles: Dict[str, ConfigProfile] = {}
        self.validators: Dict[str, ConfigValidator] = {
            'biological': BiologicalConfigValidator(),
            'control': ControlConfigValidator(),
            'visualization': VisualizationConfigValidator()
        }
        self.current_profile: Optional[str] = None
        self.cache_enabled = True
        self.logger = logging.getLogger(__name__)

        # Ensure config directory exists
        self.config_directory.mkdir(parents=True, exist_ok=True)

        # Load default profiles
        self._load_default_profiles()

    def _load_default_profiles(self):
        """Load default configuration profiles."""
        try:
            # Load existing profiles from config directory
            for config_file in self.config_directory.glob("*.yaml"):
                self.load_profile_from_file(config_file.stem, str(config_file))

            for config_file in self.config_directory.glob("*.json"):
                self.load_profile_from_file(config_file.stem, str(config_file))

        except Exception as e:
            self.logger.warning(f"Failed to load some default profiles: {e}")

    def load_profile_from_file(self, profile_name: str,
                              file_path: str) -> ConfigProfile:
        """
        Load configuration profile from file.
        
        Args:
            profile_name: Name of the profile
            file_path: Path to configuration file
            
        Returns:
            Loaded configuration profile
            
        Raises:
            ConfigurationLoadError: If loading fails
        """
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                raise ConfigurationLoadError(f"Configuration file not found: {file_path}")

            # Load based on file extension
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                with open(file_path, 'r') as f:
                    config_data = yaml.safe_load(f)
            elif file_path.suffix.lower() == '.json':
                with open(file_path, 'r') as f:
                    config_data = json.load(f)
            else:
                raise ConfigurationLoadError(f"Unsupported file format: {file_path.suffix}")

            # Create profile from loaded data
            profile = self._create_profile_from_dict(profile_name, config_data)

            # Validate profile
            self.validate_profile(profile)

            # Cache profile
            self.profiles[profile_name] = profile

            self.logger.info(f"Loaded configuration profile: {profile_name}")
            return profile

        except Exception as e:
            raise ConfigurationLoadError(f"Failed to load profile {profile_name}: {e}")

    def _create_profile_from_dict(self, profile_name: str,
                                 config_data: Dict[str, Any]) -> ConfigProfile:
        """Create configuration profile from dictionary."""
        profile = ConfigProfile(profile_name=profile_name)

        # Load biological configuration
        if 'biological' in config_data:
            profile.biological = config_data['biological']

        # Load control configuration
        if 'control' in config_data:
            control_data = config_data['control']

            # Handle different ways control config might be structured
            if isinstance(control_data, dict):
                profile.control = self._dict_to_control_config(control_data)
            else:
                profile.control = control_data

        # Load visualization configuration
        if 'visualization' in config_data:
            viz_data = config_data['visualization']

            if isinstance(viz_data, dict):
                profile.visualization = self._dict_to_visualization_config(viz_data)
            else:
                profile.visualization = viz_data

        # Load metadata
        if 'metadata' in config_data:
            metadata = config_data['metadata']
            profile.description = metadata.get('description', profile.description)
            profile.profile_version = metadata.get('version', profile.profile_version)
            profile.environment = metadata.get('environment', profile.environment)
            profile.inherits_from = metadata.get('inherits_from')

        return profile

    def _dict_to_control_config(self, control_data: Dict[str, Any]) -> ControlSystemConfig:
        """Convert dictionary to ControlSystemConfig."""
        # This is a simplified conversion - can be enhanced for full dataclass conversion
        from .control_config import ControlSystemConfig

        config = ControlSystemConfig()

        # Update fields that exist in the data
        for key, value in control_data.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    def _dict_to_visualization_config(self, viz_data: Dict[str, Any]) -> VisualizationConfig:
        """Convert dictionary to VisualizationConfig."""
        from .visualization_config import VisualizationConfig

        config = VisualizationConfig()

        # Update fields that exist in the data
        for key, value in viz_data.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    def save_profile_to_file(self, profile_name: str,
                           file_path: str, format: str = "yaml") -> None:
        """
        Save configuration profile to file.
        
        Args:
            profile_name: Name of profile to save
            file_path: Output file path
            format: File format ("yaml" or "json")
            
        Raises:
            ConfigurationError: If profile doesn't exist or saving fails
        """
        if profile_name not in self.profiles:
            raise ConfigurationError(f"Profile not found: {profile_name}")

        profile = self.profiles[profile_name]

        # Convert profile to dictionary
        config_data = {
            'biological': profile.biological,
            'control': asdict(profile.control) if profile.control else None,
            'visualization': asdict(profile.visualization) if profile.visualization else None,
            'metadata': {
                'description': profile.description,
                'version': profile.profile_version,
                'environment': profile.environment,
                'inherits_from': profile.inherits_from,
                'created_at': profile.created_at.isoformat(),
                'updated_at': profile.updated_at.isoformat()
            }
        }

        # Remove None values
        config_data = {k: v for k, v in config_data.items() if v is not None}

        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            if format.lower() == "yaml":
                with open(file_path, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
            elif format.lower() == "json":
                with open(file_path, 'w') as f:
                    json.dump(config_data, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format}")

            self.logger.info(f"Saved profile {profile_name} to {file_path}")

        except Exception as e:
            raise ConfigurationError(f"Failed to save profile {profile_name}: {e}")

    def validate_profile(self, profile: ConfigProfile) -> bool:
        """
        Validate configuration profile.
        
        Args:
            profile: Profile to validate
            
        Returns:
            bool: True if valid
            
        Raises:
            ConfigurationValidationError: If validation fails
        """
        try:
            # Validate biological configuration
            if profile.biological:
                self.validators['biological'].validate(profile.biological)

            # Validate control configuration
            if profile.control:
                self.validators['control'].validate(profile.control)

            # Validate visualization configuration
            if profile.visualization:
                self.validators['visualization'].validate(profile.visualization)

            return True

        except Exception as e:
            raise ConfigurationValidationError(f"Profile validation failed: {e}")

    def get_profile(self, profile_name: str) -> ConfigProfile:
        """
        Get configuration profile by name.
        
        Args:
            profile_name: Name of profile to retrieve
            
        Returns:
            Configuration profile
            
        Raises:
            ConfigurationError: If profile doesn't exist
        """
        if profile_name not in self.profiles:
            raise ConfigurationError(f"Profile not found: {profile_name}")

        profile = self.profiles[profile_name]

        # Handle profile inheritance
        if profile.inherits_from:
            base_profile = self.get_profile(profile.inherits_from)
            return ConfigMerger.merge_profiles(base_profile, profile)

        return profile

    def set_current_profile(self, profile_name: str) -> None:
        """
        Set the current active profile.
        
        Args:
            profile_name: Name of profile to activate
            
        Raises:
            ConfigurationError: If profile doesn't exist
        """
        if profile_name not in self.profiles:
            raise ConfigurationError(f"Profile not found: {profile_name}")

        self.current_profile = profile_name
        self.logger.info(f"Set current profile to: {profile_name}")

    def get_current_profile(self) -> Optional[ConfigProfile]:
        """
        Get the current active profile.
        
        Returns:
            Current profile or None if not set
        """
        if self.current_profile:
            return self.get_profile(self.current_profile)
        return None

    def create_profile(self, profile_name: str,
                      biological: Optional[Dict[str, Any]] = None,
                      control: Optional[ControlSystemConfig] = None,
                      visualization: Optional[VisualizationConfig] = None,
                      inherits_from: Optional[str] = None) -> ConfigProfile:
        """
        Create a new configuration profile.
        
        Args:
            profile_name: Name of the new profile
            biological: Biological configuration
            control: Control system configuration
            visualization: Visualization configuration
            inherits_from: Parent profile name for inheritance
            
        Returns:
            Created profile
        """
        profile = ConfigProfile(
            profile_name=profile_name,
            biological=biological,
            control=control,
            visualization=visualization,
            inherits_from=inherits_from
        )

        # Validate the profile
        self.validate_profile(profile)

        # Store the profile
        self.profiles[profile_name] = profile

        self.logger.info(f"Created new profile: {profile_name}")
        return profile

    def delete_profile(self, profile_name: str) -> None:
        """
        Delete a configuration profile.
        
        Args:
            profile_name: Name of profile to delete
            
        Raises:
            ConfigurationError: If profile doesn't exist or is currently active
        """
        if profile_name not in self.profiles:
            raise ConfigurationError(f"Profile not found: {profile_name}")

        if self.current_profile == profile_name:
            raise ConfigurationError(f"Cannot delete active profile: {profile_name}")

        del self.profiles[profile_name]
        self.logger.info(f"Deleted profile: {profile_name}")

    def list_profiles(self) -> List[str]:
        """
        Get list of available profile names.
        
        Returns:
            List of profile names
        """
        return list(self.profiles.keys())

    def get_configuration(self, config_type: str,
                         profile_name: Optional[str] = None) -> Any:
        """
        Get specific configuration from profile.
        
        Args:
            config_type: Type of configuration ("biological", "control", "visualization")
            profile_name: Profile name (uses current if None)
            
        Returns:
            Requested configuration
            
        Raises:
            ConfigurationError: If profile or configuration doesn't exist
        """
        if profile_name is None:
            if self.current_profile is None:
                raise ConfigurationError("No current profile set")
            profile_name = self.current_profile

        profile = self.get_profile(profile_name)

        if config_type == "biological":
            return profile.biological
        elif config_type == "control":
            return profile.control
        elif config_type == "visualization":
            return profile.visualization
        else:
            raise ConfigurationError(f"Unknown configuration type: {config_type}")

    def update_configuration(self, config_type: str, config: Any,
                           profile_name: Optional[str] = None) -> None:
        """
        Update specific configuration in profile.
        
        Args:
            config_type: Type of configuration to update
            config: New configuration
            profile_name: Profile name (uses current if None)
            
        Raises:
            ConfigurationError: If profile doesn't exist
        """
        if profile_name is None:
            if self.current_profile is None:
                raise ConfigurationError("No current profile set")
            profile_name = self.current_profile

        if profile_name not in self.profiles:
            raise ConfigurationError(f"Profile not found: {profile_name}")

        profile = self.profiles[profile_name]

        if config_type == "biological":
            profile.biological = config
        elif config_type == "control":
            profile.control = config
        elif config_type == "visualization":
            profile.visualization = config
        else:
            raise ConfigurationError(f"Unknown configuration type: {config_type}")

        # Validate updated profile
        self.validate_profile(profile)

        profile.update_timestamp()
        self.logger.info(f"Updated {config_type} configuration in profile {profile_name}")


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_directory: Optional[str] = None) -> ConfigManager:
    """
    Get global configuration manager instance.
    
    Args:
        config_directory: Configuration directory (only used on first call)
        
    Returns:
        ConfigManager instance
    """
    global _config_manager

    if _config_manager is None:
        _config_manager = ConfigManager(config_directory)

    return _config_manager


def reset_config_manager() -> None:
    """Reset global configuration manager (mainly for testing)."""
    global _config_manager
    _config_manager = None
