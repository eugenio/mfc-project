"""Comprehensive coverage tests for path_config module."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

import path_config


class TestDebugMode:
    def setup_method(self):
        """Reset debug state before each test."""
        path_config._debug_state["enabled"] = False
        os.environ.pop("MFC_DEBUG_MODE", None)

    def teardown_method(self):
        path_config._debug_state["enabled"] = False
        os.environ.pop("MFC_DEBUG_MODE", None)

    def test_default_not_debug(self):
        assert path_config.is_debug_mode() is False

    def test_enable_debug_mode(self):
        path_config.enable_debug_mode()
        assert path_config.is_debug_mode() is True

    def test_disable_debug_mode(self):
        path_config.enable_debug_mode()
        path_config.disable_debug_mode()
        assert path_config.is_debug_mode() is False

    def test_env_debug_true(self):
        os.environ["MFC_DEBUG_MODE"] = "true"
        assert path_config.is_debug_mode() is True

    def test_env_debug_1(self):
        os.environ["MFC_DEBUG_MODE"] = "1"
        assert path_config.is_debug_mode() is True

    def test_env_debug_yes(self):
        os.environ["MFC_DEBUG_MODE"] = "yes"
        assert path_config.is_debug_mode() is True

    def test_env_debug_false(self):
        os.environ["MFC_DEBUG_MODE"] = "false"
        assert path_config.is_debug_mode() is False

    def test_check_env_debug_mode(self):
        os.environ["MFC_DEBUG_MODE"] = "TRUE"
        assert path_config._check_env_debug_mode() is True


class TestGetCurrentBasePath:
    def setup_method(self):
        path_config._debug_state["enabled"] = False
        os.environ.pop("MFC_DEBUG_MODE", None)

    def teardown_method(self):
        path_config._debug_state["enabled"] = False
        os.environ.pop("MFC_DEBUG_MODE", None)

    def test_normal_mode(self):
        result = path_config.get_current_base_path()
        assert result == path_config.PROJECT_ROOT

    def test_debug_mode(self):
        path_config.enable_debug_mode()
        result = path_config.get_current_base_path()
        assert result == path_config.DEBUG_BASE_PATH


class TestPathFunctions:
    def setup_method(self):
        path_config._debug_state["enabled"] = False
        os.environ.pop("MFC_DEBUG_MODE", None)

    def teardown_method(self):
        path_config._debug_state["enabled"] = False
        os.environ.pop("MFC_DEBUG_MODE", None)

    def test_get_figure_path(self):
        result = path_config.get_figure_path("test.png")
        assert result.endswith("test.png")
        assert "figures" in result

    def test_get_simulation_data_path(self):
        result = path_config.get_simulation_data_path("data.json")
        assert result.endswith("data.json")
        assert "simulation_data" in result

    def test_get_log_path(self):
        result = path_config.get_log_path("log.txt")
        assert result.endswith("log.txt")
        assert "logs" in result

    def test_get_model_path(self):
        result = path_config.get_model_path("model.pkl")
        assert result.endswith("model.pkl")
        assert "q_learning_models" in result

    def test_get_report_path(self):
        result = path_config.get_report_path("report.pdf")
        assert result.endswith("report.pdf")
        assert "reports" in result

    def test_get_cad_model_path(self):
        result = path_config.get_cad_model_path("part.step")
        assert result.endswith("part.step")
        assert "cad_models" in result

    def test_get_cad_model_path_with_subdir(self):
        result = path_config.get_cad_model_path("part.step", subdir="components")
        assert result.endswith("part.step")
        assert "components" in result

    def test_get_cad_model_path_empty_subdir(self):
        result = path_config.get_cad_model_path("part.step", subdir="")
        assert result.endswith("part.step")


class TestDebugPaths:
    def setup_method(self):
        path_config._debug_state["enabled"] = False
        os.environ.pop("MFC_DEBUG_MODE", None)

    def teardown_method(self):
        path_config._debug_state["enabled"] = False
        os.environ.pop("MFC_DEBUG_MODE", None)

    def test_debug_paths_redirect(self):
        path_config.enable_debug_mode()
        result = path_config.get_figure_path("test.png")
        assert str(path_config.DEBUG_BASE_PATH) in result

    def test_ensure_debug_directories(self):
        path_config.enable_debug_mode()
        path_config._ensure_debug_directories()
        base = path_config.DEBUG_BASE_PATH
        assert (base / "data" / "figures").exists()

    def test_ensure_debug_dirs_not_debug(self):
        """No directories created when not in debug mode."""
        path_config._ensure_debug_directories()


class TestConstants:
    def test_project_root(self):
        assert path_config.PROJECT_ROOT.exists()

    def test_output_subdirs(self):
        assert len(path_config._OUTPUT_SUBDIRS) >= 5

    def test_standard_dirs_exist(self):
        assert path_config.FIGURES_DIR.exists()
        assert path_config.SIMULATION_DATA_DIR.exists()
        assert path_config.LOGS_DIR.exists()
        assert path_config.MODELS_DIR.exists()
