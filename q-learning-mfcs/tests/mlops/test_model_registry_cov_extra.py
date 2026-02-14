"""Extra coverage tests for model_registry.py.

Targets remaining uncovered lines:
  471 - delete_model removes lineage file
  501-502 - search_models_by_tag FileNotFoundError
  527-528 - export_registry FileNotFoundError
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from mlops.model_registry import ModelRegistry


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def registry(tmp_dir):
    return ModelRegistry(registry_path=tmp_dir)


@pytest.mark.coverage_extra
class TestDeleteModelLineage:
    """Cover line 471."""

    def test_removes_lineage(self, registry):
        registry.register_model({"w": 1}, {"name": "parent", "algorithm": "lr"})
        registry.register_model(
            {"w": 2}, {"name": "child", "algorithm": "lr"},
            parent_model="parent", parent_version="1.0.0",
        )
        lp = registry.registry_path / "lineage" / "child" / "1.0.0.json"
        assert lp.exists()
        registry.delete_model("child", "1.0.0")
        assert not lp.exists()


@pytest.mark.coverage_extra
class TestSearchTagMissing:
    """Cover lines 501-502."""

    def test_missing_metadata(self, registry):
        registry.register_model(
            {"w": 1}, {"name": "m", "algorithm": "lr", "tags": ["prod"]}
        )
        (registry.registry_path / "metadata" / "m" / "1.0.0.json").unlink()
        assert registry.search_models_by_tag("prod") == []

    def test_mixed(self, registry):
        registry.register_model(
            {"w": 1}, {"name": "m1", "algorithm": "lr", "tags": ["prod"]}
        )
        registry.register_model(
            {"w": 2}, {"name": "m2", "algorithm": "rf", "tags": ["prod"]}
        )
        (registry.registry_path / "metadata" / "m1" / "1.0.0.json").unlink()
        results = registry.search_models_by_tag("prod")
        assert len(results) == 1
        assert results[0]["name"] == "m2"


@pytest.mark.coverage_extra
class TestExportMissing:
    """Cover lines 527-528."""

    def test_missing_metadata(self, registry, tmp_dir):
        registry.register_model({"w": 1}, {"name": "m", "algorithm": "lr"})
        (registry.registry_path / "metadata" / "m" / "1.0.0.json").unlink()
        ep = Path(tmp_dir) / "export"
        registry.export_registry(ep)
        with open(ep / "registry_export.json") as f:
            data = json.load(f)
        assert data["models"]["m"] == {}

    def test_mixed(self, registry, tmp_dir):
        registry.register_model({"w": 1}, {"name": "m", "algorithm": "lr"})
        registry.register_model({"w": 2}, {"name": "m", "algorithm": "lr"})
        (registry.registry_path / "metadata" / "m" / "1.0.0.json").unlink()
        ep = Path(tmp_dir) / "export2"
        registry.export_registry(ep)
        with open(ep / "registry_export.json") as f:
            data = json.load(f)
        assert "1.0.1" in data["models"]["m"]
        assert "1.0.0" not in data["models"]["m"]
