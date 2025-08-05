"""Model Registry - MLOps Model Management System.

This module provides a comprehensive model registry for managing machine learning
models with versioning, metadata, lineage tracking, and comparison capabilities.

Key Features:
- Semantic versioning for models (major.minor.patch)
- Model storage and retrieval with pickle serialization
- Comprehensive metadata management with validation
- Model lineage tracking for reproducibility
- Model comparison and analysis tools
- Registry backup and restoration
- Thread-safe operations for concurrent access

Author: TDD Agent 1
"""
import json
import pickle
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import re

class ModelRegistryError(Exception):
    """Base exception for model registry operations."""
    pass


class ModelVersionError(ModelRegistryError):
    """Exception raised for version-related errors."""
    pass


class ModelNotFoundError(ModelRegistryError):
    """Exception raised when a model is not found."""
    pass

class ModelRegistry:
    """Comprehensive model registry for MLOps workflows.
    
    This class provides a complete solution for managing machine learning models
    throughout their lifecycle, including versioning, metadata management,
    lineage tracking, and model comparison capabilities.
    
    Attributes:
        registry_path: Path to the registry storage directory
        _lock: Thread lock for concurrent operations
        _registry_data: In-memory registry metadata
    """
    
    def __init__(self, registry_path: Union[str, Path]):
        """Initialize the model registry.
        
        Args:
            registry_path: Path where the registry will store models and metadata
            
        Raises:
            OSError: If registry directory cannot be created
        """
        self.registry_path = Path(registry_path)
        self._lock = threading.Lock()
        self._registry_data: Dict[str, Any] = {}
        
        # Initialize registry directory structure
        self._initialize_registry()
        
    def _initialize_registry(self) -> None:
        """Initialize the registry directory structure and metadata."""
        # Create directory structure
        self.registry_path.mkdir(parents=True, exist_ok=True)
        (self.registry_path / "models").mkdir(exist_ok=True)
        (self.registry_path / "metadata").mkdir(exist_ok=True)
        (self.registry_path / "lineage").mkdir(exist_ok=True)
        
        # Initialize or load registry metadata
        registry_file = self.registry_path / "registry.json"
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                self._registry_data = json.load(f)
        else:
            self._registry_data = {
                "created_at": datetime.now().isoformat(),
                "version": "1.0.0",
                "models": {}
            }
            self._save_registry_metadata()
    
    def _save_registry_metadata(self) -> None:
        """Save registry metadata to disk."""
        registry_file = self.registry_path / "registry.json"
        with open(registry_file, 'w') as f:
            json.dump(self._registry_data, f, indent=2)
    
    def _validate_metadata(self, metadata: Dict[str, Any]) -> None:
        """Validate model metadata before registration.
        
        Args:
            metadata: Model metadata dictionary
            
        Raises:
            ValueError: If metadata is invalid
        """
        required_fields = ["name", "algorithm"]
        
        for field in required_fields:
            if field not in metadata:
                raise ValueError(f"Missing required metadata field: {field}")
            
        if not metadata["name"] or not isinstance(metadata["name"], str):
            raise ValueError("Model name must be a non-empty string")
            
        if not metadata["algorithm"] or not isinstance(metadata["algorithm"], str):
            raise ValueError("Algorithm must be a non-empty string")
    
    def _parse_version(self, version: str) -> tuple[int, int, int]:
        """Parse semantic version string into components.
        
        Args:
            version: Version string in format "major.minor.patch"
            
        Returns:
            Tuple of (major, minor, patch) integers
            
        Raises:
            ModelVersionError: If version format is invalid
        """
        if not re.match(r'^\d+\.\d+\.\d+$', version):
            raise ModelVersionError(f"Invalid version format: {version}")
        
        major, minor, patch = map(int, version.split('.'))
        return major, minor, patch
    
    def _increment_version(self, current_version: str, increment_type: str) -> str:
        """Increment version based on type.
        
        Args:
            current_version: Current version string
            increment_type: Type of increment ('major', 'minor', 'patch')
            
        Returns:
            New version string
            
        Raises:
            ModelVersionError: If increment type is invalid
        """
        major, minor, patch = self._parse_version(current_version)
        
        if increment_type == "major":
            return f"{major + 1}.0.0"
        elif increment_type == "minor":
            return f"{major}.{minor + 1}.0"
        elif increment_type == "patch":
            return f"{major}.{minor}.{patch + 1}"
        else:
            raise ModelVersionError(f"Invalid increment type: {increment_type}")
    
    def _get_next_version(self, model_name: str, increment_type: str = "patch") -> str:
        """Get the next version for a model.
        
        Args:
            model_name: Name of the model
            increment_type: Type of version increment
            
        Returns:
            Next version string
        """
        if model_name not in self._registry_data["models"]:
            return "1.0.0"
        
        versions = list(self._registry_data["models"][model_name].keys())
        if not versions:
            return "1.0.0"
        
        # Sort versions to get the latest
        versions.sort(key=lambda v: tuple(map(int, v.split('.'))))
        latest_version = versions[-1]
        
        return self._increment_version(latest_version, increment_type)
    
    def register_model(
        self,
        model: Any,
        metadata: Dict[str, Any],
        increment_type: str = "patch",
        parent_model: Optional[str] = None,
        parent_version: Optional[str] = None
    ) -> str:
        """Register a new model or model version.
        
        Args:
            model: The ML model object to register
            metadata: Model metadata dictionary
            increment_type: Version increment type ('major', 'minor', 'patch')
            parent_model: Name of parent model for lineage tracking
            parent_version: Version of parent model for lineage tracking
            
        Returns:
            Version string of the registered model
            
        Raises:
            ValueError: If metadata is invalid
        """
        with self._lock:
            # Validate metadata
            self._validate_metadata(metadata)
            
            model_name = metadata["name"]
            version = self._get_next_version(model_name, increment_type)
            
            # Create model directory
            model_dir = self.registry_path / "models" / model_name / version
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_path = model_dir / "model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Prepare and save metadata
            full_metadata = metadata.copy()
            full_metadata.update({
                "version": version,
                "registration_timestamp": datetime.now().isoformat(),
                "model_path": str(model_path),
                "size_bytes": model_path.stat().st_size
            })
            
            metadata_dir = self.registry_path / "metadata" / model_name
            metadata_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = metadata_dir / f"{version}.json"
            
            with open(metadata_path, 'w') as f:
                json.dump(full_metadata, f, indent=2)
            
            # Update registry data
            if model_name not in self._registry_data["models"]:
                self._registry_data["models"][model_name] = {}
            
            self._registry_data["models"][model_name][version] = {
                "metadata_path": str(metadata_path),
                "registered_at": full_metadata["registration_timestamp"]
            }
            
            # Handle lineage tracking
            if parent_model and parent_version:
                self._record_lineage(model_name, version, parent_model, parent_version)
            
            self._save_registry_metadata()
            
            return version
    
    def _record_lineage(
        self,
        model_name: str,
        version: str,
        parent_model: str,
        parent_version: str
    ) -> None:
        """Record model lineage information.
        
        Args:
            model_name: Name of the child model
            version: Version of the child model
            parent_model: Name of the parent model
            parent_version: Version of the parent model
        """
        lineage_data = {
            "parent_model": parent_model,
            "parent_version": parent_version,
            "lineage_timestamp": datetime.now().isoformat()
        }
        
        lineage_dir = self.registry_path / "lineage" / model_name
        lineage_dir.mkdir(parents=True, exist_ok=True)
        lineage_path = lineage_dir / f"{version}.json"
        
        with open(lineage_path, 'w') as f:
            json.dump(lineage_data, f, indent=2)
    
    def get_model(self, model_name: str, version: str) -> Any:
        """Retrieve a model from the registry.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            
        Returns:
            The loaded model object
            
        Raises:
            FileNotFoundError: If model is not found
        """
        model_path = self.registry_path / "models" / model_name / version / "model.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model {model_name} version {version} not found")
        
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    def get_metadata(self, model_name: str, version: str) -> Dict[str, Any]:
        """Retrieve model metadata.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            
        Returns:
            Model metadata dictionary
            
        Raises:
            FileNotFoundError: If metadata is not found
        """
        metadata_path = self.registry_path / "metadata" / model_name / f"{version}.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata for {model_name} version {version} not found")
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def model_exists(self, model_name: str, version: str) -> bool:
        """Check if a model exists in the registry.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            
        Returns:
            True if model exists, False otherwise
        """
        return (self.registry_path / "models" / model_name / version / "model.pkl").exists()
    
    def list_model_versions(self, model_name: str) -> List[str]:
        """List all versions of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of version strings
        """
        model_dir = self.registry_path / "models" / model_name
        if not model_dir.exists():
            return []
        
        versions = [d.name for d in model_dir.iterdir() if d.is_dir()]
        versions.sort(key=lambda v: tuple(map(int, v.split('.'))))
        return versions
    
    def get_latest_version(self, model_name: str) -> str:
        """Get the latest version of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Latest version string
            
        Raises:
            ModelNotFoundError: If no versions exist
        """
        versions = self.list_model_versions(model_name)
        if not versions:
            raise ModelNotFoundError(f"No versions found for model {model_name}")
        
        return versions[-1]  # Already sorted
    
    def get_model_lineage(self, model_name: str, version: str) -> Dict[str, Any]:
        """Get lineage information for a model.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            
        Returns:
            Lineage information dictionary
            
        Raises:
            FileNotFoundError: If lineage data is not found
        """
        lineage_path = self.registry_path / "lineage" / model_name / f"{version}.json"
        
        if not lineage_path.exists():
            raise FileNotFoundError(f"Lineage data for {model_name} version {version} not found")
        
        with open(lineage_path, 'r') as f:
            return json.load(f)
    
    def compare_models(
        self,
        model1_name: str,
        model1_version: str,
        model2_name: str,
        model2_version: str
    ) -> Dict[str, Any]:
        """Compare two models.
        
        Args:
            model1_name: Name of first model
            model1_version: Version of first model
            model2_name: Name of second model
            model2_version: Version of second model
            
        Returns:
            Comparison results dictionary
        """
        metadata1 = self.get_metadata(model1_name, model1_version)
        metadata2 = self.get_metadata(model2_name, model2_version)
        
        # Compare performance metrics if available
        performance_comparison = {}
        if "performance_metrics" in metadata1 and "performance_metrics" in metadata2:
            metrics1 = metadata1["performance_metrics"]
            metrics2 = metadata2["performance_metrics"]
            
            for metric in set(metrics1.keys()) | set(metrics2.keys()):
                performance_comparison[metric] = {
                    "model1": metrics1.get(metric),
                    "model2": metrics2.get(metric)
                }
        
        # Metadata differences
        metadata_diff = {}
        all_keys = set(metadata1.keys()) | set(metadata2.keys())
        
        for key in all_keys:
            val1 = metadata1.get(key)
            val2 = metadata2.get(key)
            
            if val1 != val2:
                metadata_diff[key] = {
                    "model1": val1,
                    "model2": val2
                }
        
        return {
            "comparison_timestamp": datetime.now().isoformat(),
            "model1": {"name": model1_name, "version": model1_version},
            "model2": {"name": model2_name, "version": model2_version},
            "metadata_diff": metadata_diff,
            "performance_comparison": performance_comparison
        }
    
    def delete_model(self, model_name: str, version: str) -> None:
        """Delete a model from the registry.
        
        Args:
            model_name: Name of the model
            version: Version of the model
        """
        with self._lock:
            # Remove model files
            model_dir = self.registry_path / "models" / model_name / version
            if model_dir.exists():
                shutil.rmtree(model_dir)
            
            # Remove metadata
            metadata_path = self.registry_path / "metadata" / model_name / f"{version}.json"
            if metadata_path.exists():
                metadata_path.unlink()
            
            # Remove lineage data
            lineage_path = self.registry_path / "lineage" / model_name / f"{version}.json"
            if lineage_path.exists():
                lineage_path.unlink()
            
            # Update registry data
            if (model_name in self._registry_data["models"] and 
                version in self._registry_data["models"][model_name]):
                del self._registry_data["models"][model_name][version]
                
                # Remove model entry if no versions left
                if not self._registry_data["models"][model_name]:
                    del self._registry_data["models"][model_name]
            
            self._save_registry_metadata()
    
    def search_models_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """Search models by tag.
        
        Args:
            tag: Tag to search for
            
        Returns:
            List of model metadata dictionaries
        """
        results = []
        
        for model_name in self._registry_data["models"]:
            for version in self._registry_data["models"][model_name]:
                try:
                    metadata = self.get_metadata(model_name, version)
                    if "tags" in metadata and tag in metadata["tags"]:
                        results.append(metadata)
                except FileNotFoundError:
                    continue
        
        return results
    
    def export_registry(self, export_path: Path) -> None:
        """Export registry data to a specified path.
        
        Args:
            export_path: Path where to export the registry
        """
        export_path.mkdir(parents=True, exist_ok=True)
        
        export_data = {
            "registry_metadata": self._registry_data,
            "export_timestamp": datetime.now().isoformat(),
            "models": {}
        }
        
        # Export model metadata
        for model_name in self._registry_data["models"]:
            export_data["models"][model_name] = {}
            for version in self._registry_data["models"][model_name]:
                try:
                    metadata = self.get_metadata(model_name, version)
                    export_data["models"][model_name][version] = metadata
                except FileNotFoundError:
                    continue
        
        export_file = export_path / "registry_export.json"
        with open(export_file, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def backup_registry(self, backup_path: Path) -> None:
        """Create a backup of the entire registry.
        
        Args:
            backup_path: Path where to create the backup
        """
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Copy all registry files
        for item in ["models", "metadata", "lineage"]:
            src = self.registry_path / item
            dst = backup_path / item
            if src.exists():
                shutil.copytree(src, dst, dirs_exist_ok=True)
        
        # Copy registry metadata
        src_registry = self.registry_path / "registry.json"
        dst_registry = backup_path / "registry.json"
        if src_registry.exists():
            shutil.copy(src_registry, dst_registry)
    
    @classmethod
    def restore_from_backup(cls, backup_path: Path, new_registry_path: Path) -> 'ModelRegistry':
        """Restore a registry from a backup.
        
        Args:
            backup_path: Path to the backup
            new_registry_path: Path where to restore the registry
            
        Returns:
            New ModelRegistry instance
        """
        new_registry_path.mkdir(parents=True, exist_ok=True)
        
        # Copy all backup files
        for item in ["models", "metadata", "lineage", "registry.json"]:
            src = backup_path / item
            dst = new_registry_path / item
            if src.exists():
                if src.is_dir():
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy(src, dst)
        
        return cls(new_registry_path)