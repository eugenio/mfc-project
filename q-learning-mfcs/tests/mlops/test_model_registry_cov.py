"""Tests for model_registry module."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import shutil
import tempfile
from pathlib import Path

import pytest

from mlops.model_registry import (
    ModelNotFoundError,
    ModelRegistry,
    ModelRegistryError,
    ModelVersionError,
)


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def registry(tmp_dir):
    return ModelRegistry(registry_path=tmp_dir)


class TestModelRegistry:
    def test_init_creates_dirs(self, tmp_dir):
        r = ModelRegistry(tmp_dir)
        assert (Path(tmp_dir) / "models").exists()
        assert (Path(tmp_dir) / "metadata").exists()
        assert (Path(tmp_dir) / "lineage").exists()

    def test_init_loads_existing(self, tmp_dir):
        r1 = ModelRegistry(tmp_dir)
        r1.register_model({"w": 1}, {"name": "m", "algorithm": "lr"})
        r2 = ModelRegistry(tmp_dir)
        assert "m" in r2._registry_data["models"]

    def test_validate_metadata_missing_name(self, registry):
        with pytest.raises(ValueError):
            registry._validate_metadata({"algorithm": "lr"})

    def test_validate_metadata_missing_algo(self, registry):
        with pytest.raises(ValueError):
            registry._validate_metadata({"name": "m"})

    def test_validate_metadata_empty_name(self, registry):
        with pytest.raises(ValueError):
            registry._validate_metadata({"name": "", "algorithm": "lr"})

    def test_validate_metadata_empty_algo(self, registry):
        with pytest.raises(ValueError):
            registry._validate_metadata({"name": "m", "algorithm": ""})

    def test_validate_metadata_nonstring_name(self, registry):
        with pytest.raises(ValueError):
            registry._validate_metadata({"name": 123, "algorithm": "lr"})

    def test_validate_metadata_nonstring_algo(self, registry):
        with pytest.raises(ValueError):
            registry._validate_metadata({"name": "m", "algorithm": 123})

    def test_parse_version(self, registry):
        assert registry._parse_version("1.2.3") == (1, 2, 3)

    def test_parse_version_invalid(self, registry):
        with pytest.raises(ModelVersionError):
            registry._parse_version("bad")

    def test_increment_version_major(self, registry):
        assert registry._increment_version("1.2.3", "major") == "2.0.0"

    def test_increment_version_minor(self, registry):
        assert registry._increment_version("1.2.3", "minor") == "1.3.0"

    def test_increment_version_patch(self, registry):
        assert registry._increment_version("1.2.3", "patch") == "1.2.4"

    def test_increment_version_invalid(self, registry):
        with pytest.raises(ModelVersionError):
            registry._increment_version("1.0.0", "bad")

    def test_get_next_version_new(self, registry):
        assert registry._get_next_version("new_model") == "1.0.0"

    def test_get_next_version_existing(self, registry):
        registry.register_model({"w": 1}, {"name": "m", "algorithm": "lr"})
        assert registry._get_next_version("m") == "1.0.1"

    def test_get_next_version_empty_versions(self, registry):
        registry._registry_data["models"]["m"] = {}
        assert registry._get_next_version("m") == "1.0.0"

    def test_register_model(self, registry):
        v = registry.register_model({"w": 1}, {"name": "m", "algorithm": "lr"})
        assert v == "1.0.0"

    def test_register_model_major(self, registry):
        registry.register_model({"w": 1}, {"name": "m", "algorithm": "lr"})
        v = registry.register_model({"w": 2}, {"name": "m", "algorithm": "lr"},
                                     increment_type="major")
        assert v == "2.0.0"

    def test_register_model_with_lineage(self, registry):
        registry.register_model({"w": 1}, {"name": "parent", "algorithm": "lr"})
        v = registry.register_model({"w": 2}, {"name": "child", "algorithm": "lr"},
                                     parent_model="parent", parent_version="1.0.0")
        assert v == "1.0.0"

    def test_get_model(self, registry):
        registry.register_model({"w": 42}, {"name": "m", "algorithm": "lr"})
        model = registry.get_model("m", "1.0.0")
        assert model["w"] == 42

    def test_get_model_not_found(self, registry):
        with pytest.raises(FileNotFoundError):
            registry.get_model("missing", "1.0.0")

    def test_get_metadata(self, registry):
        registry.register_model({"w": 1}, {"name": "m", "algorithm": "lr"})
        meta = registry.get_metadata("m", "1.0.0")
        assert meta["name"] == "m"
        assert "version" in meta

    def test_get_metadata_not_found(self, registry):
        with pytest.raises(FileNotFoundError):
            registry.get_metadata("missing", "1.0.0")

    def test_model_exists(self, registry):
        registry.register_model({"w": 1}, {"name": "m", "algorithm": "lr"})
        assert registry.model_exists("m", "1.0.0") is True
        assert registry.model_exists("m", "9.9.9") is False

    def test_list_model_versions(self, registry):
        registry.register_model({"w": 1}, {"name": "m", "algorithm": "lr"})
        registry.register_model({"w": 2}, {"name": "m", "algorithm": "lr"})
        versions = registry.list_model_versions("m")
        assert len(versions) == 2
        assert versions[0] == "1.0.0"

    def test_list_model_versions_nonexistent(self, registry):
        assert registry.list_model_versions("missing") == []

    def test_get_latest_version(self, registry):
        registry.register_model({"w": 1}, {"name": "m", "algorithm": "lr"})
        registry.register_model({"w": 2}, {"name": "m", "algorithm": "lr"})
        assert registry.get_latest_version("m") == "1.0.1"

    def test_get_latest_version_not_found(self, registry):
        with pytest.raises(ModelNotFoundError):
            registry.get_latest_version("missing")

    def test_get_model_lineage(self, registry):
        registry.register_model({"w": 1}, {"name": "p", "algorithm": "lr"})
        registry.register_model({"w": 2}, {"name": "c", "algorithm": "lr"},
                                 parent_model="p", parent_version="1.0.0")
        lineage = registry.get_model_lineage("c", "1.0.0")
        assert lineage["parent_model"] == "p"

    def test_get_model_lineage_not_found(self, registry):
        with pytest.raises(FileNotFoundError):
            registry.get_model_lineage("missing", "1.0.0")

    def test_compare_models(self, registry):
        registry.register_model({"w": 1}, {"name": "m1", "algorithm": "lr"})
        registry.register_model({"w": 2}, {"name": "m2", "algorithm": "rf"})
        result = registry.compare_models("m1", "1.0.0", "m2", "1.0.0")
        assert "metadata_diff" in result

    def test_compare_models_with_perf_metrics(self, registry):
        registry.register_model({"w": 1}, {"name": "m1", "algorithm": "lr",
                                            "performance_metrics": {"acc": 0.9}})
        registry.register_model({"w": 2}, {"name": "m2", "algorithm": "lr",
                                            "performance_metrics": {"acc": 0.8}})
        result = registry.compare_models("m1", "1.0.0", "m2", "1.0.0")
        assert "performance_comparison" in result

    def test_delete_model(self, registry):
        registry.register_model({"w": 1}, {"name": "m", "algorithm": "lr"})
        registry.delete_model("m", "1.0.0")
        assert not registry.model_exists("m", "1.0.0")
        assert "m" not in registry._registry_data["models"]

    def test_delete_model_keeps_other_versions(self, registry):
        registry.register_model({"w": 1}, {"name": "m", "algorithm": "lr"})
        registry.register_model({"w": 2}, {"name": "m", "algorithm": "lr"})
        registry.delete_model("m", "1.0.0")
        assert registry.model_exists("m", "1.0.1")

    def test_delete_model_nonexistent(self, registry):
        registry.delete_model("missing", "1.0.0")  # should not raise

    def test_search_models_by_tag(self, registry):
        registry.register_model({"w": 1}, {"name": "m1", "algorithm": "lr",
                                            "tags": ["prod"]})
        registry.register_model({"w": 2}, {"name": "m2", "algorithm": "rf",
                                            "tags": ["dev"]})
        results = registry.search_models_by_tag("prod")
        assert len(results) == 1

    def test_search_models_by_tag_no_match(self, registry):
        registry.register_model({"w": 1}, {"name": "m", "algorithm": "lr"})
        assert len(registry.search_models_by_tag("missing")) == 0

    def test_export_registry(self, registry, tmp_dir):
        registry.register_model({"w": 1}, {"name": "m", "algorithm": "lr"})
        export_path = Path(tmp_dir) / "export"
        registry.export_registry(export_path)
        assert (export_path / "registry_export.json").exists()

    def test_backup_and_restore(self, registry, tmp_dir):
        registry.register_model({"w": 1}, {"name": "m", "algorithm": "lr"})
        backup_path = Path(tmp_dir) / "backup"
        registry.backup_registry(backup_path)
        restore_path = Path(tmp_dir) / "restored"
        restored = ModelRegistry.restore_from_backup(backup_path, restore_path)
        assert restored.model_exists("m", "1.0.0")
