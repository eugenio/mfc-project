"""Extra coverage tests for model_registry.py.

Targets remaining uncovered lines:
  471 - delete_model removes lineage file
  501-502 - search_models_by_tag FileNotFoundError handling
  527-528 - export_registry FileNotFoundError handling
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from mlops.model_registry import (
class TestDeleteModelWithLineage:
    """Cover line 471: deleting a model that has lineage data."""

    def test_delete_model_removes_lineage(self, registry):
        # Register parent
        registry.register_model({"w": 1}, {"name": "parent", "algorithm": "lr"})
        # Register child with lineage
        registry.register_model(
            {"w": 2},
            {"name": "child", "algorithm": "lr"},
            parent_model="parent",
            parent_version="1.0.0",
        )
        # Verify lineage exists
        lineage_path = registry.registry_path / "lineage" / "child" / "1.0.0.json"
        assert lineage_path.exists()

        # Delete child model - should also remove lineage
        registry.delete_model("child", "1.0.0")
        assert not lineage_path.exists()

class TestSearchModelsByTagMissingMetadata:
    """Cover lines 501-502: FileNotFoundError during tag search."""

    def test_search_tag_with_missing_metadata_file(self, registry):
        # Register a model
        registry.register_model(
            {"w": 1}, {"name": "m", "algorithm": "lr", "tags": ["prod"]}
        )
        # Delete the metadata file but keep registry data
        metadata_path = registry.registry_path / "metadata" / "m" / "1.0.0.json"
        metadata_path.unlink()

        # Search should not crash, just skip the missing model
        results = registry.search_models_by_tag("prod")
        assert results == []

    def test_search_tag_with_mixed_present_and_missing(self, registry):
        # Register two models
        registry.register_model(
            {"w": 1}, {"name": "m1", "algorithm": "lr", "tags": ["prod"]}
        )
        registry.register_model(
            {"w": 2}, {"name": "m2", "algorithm": "rf", "tags": ["prod"]}
        )
        # Delete m1 metadata only
        (registry.registry_path / "metadata" / "m1" / "1.0.0.json").unlink()

        results = registry.search_models_by_tag("prod")
        assert len(results) == 1
        assert results[0]["name"] == "m2"

class TestExportRegistryMissingMetadata:
    """Cover lines 527-528: FileNotFoundError during export."""

    def test_export_with_missing_metadata_file(self, registry, tmp_dir):
        # Register a model
        registry.register_model({"w": 1}, {"name": "m", "algorithm": "lr"})
        # Delete metadata file
        (registry.registry_path / "metadata" / "m" / "1.0.0.json").unlink()

        export_path = Path(tmp_dir) / "export"
        registry.export_registry(export_path)

        # Export should succeed but model entry should be empty
        export_file = export_path / "registry_export.json"
        assert export_file.exists()
        with open(export_file) as f:
            data = json.load(f)
        assert data["models"]["m"] == {}

    def test_export_with_mixed_present_and_missing(self, registry, tmp_dir):
        registry.register_model({"w": 1}, {"name": "m", "algorithm": "lr"})
        registry.register_model({"w": 2}, {"name": "m", "algorithm": "lr"})

        # Delete first version metadata only
        (registry.registry_path / "metadata" / "m" / "1.0.0.json").unlink()

        export_path = Path(tmp_dir) / "export2"
        registry.export_registry(export_path)
        with open(export_path / "registry_export.json") as f:
            data = json.load(f)
        # Only version 1.0.1 should be exported
        assert "1.0.1" in data["models"]["m"]
        assert "1.0.0" not in data["models"]["m"]
@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def registry(tmp_dir):
    return ModelRegistry(registry_path=tmp_dir)

