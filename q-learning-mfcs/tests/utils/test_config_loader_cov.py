"""Tests for utils/config_loader.py - targeting 98%+ coverage."""
import os
import sys
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from utils.config_loader import (
    get_gitlab_config,
    load_env_file,
    setup_gitlab_config,
    validate_gitlab_config,
)


class TestLoadEnvFile:
    """Tests for load_env_file function."""

    def test_load_env_file_not_found(self, tmp_path):
        result = load_env_file(str(tmp_path / "nonexistent.env"))
        assert result == {}

    def test_load_env_file_empty(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("")
        result = load_env_file(str(env_file))
        assert result == {}

    def test_load_env_file_comments_and_blanks(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("# This is a comment\n\n# Another comment\n")
        result = load_env_file(str(env_file))
        assert result == {}

    def test_load_env_file_key_value(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("KEY1=value1\nKEY2=value2\n")
        result = load_env_file(str(env_file))
        assert result == {"KEY1": "value1", "KEY2": "value2"}

    def test_load_env_file_double_quoted_values(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text('KEY1="quoted value"\n')
        result = load_env_file(str(env_file))
        assert result == {"KEY1": "quoted value"}

    def test_load_env_file_single_quoted_values(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("KEY1='single quoted'\n")
        result = load_env_file(str(env_file))
        assert result == {"KEY1": "single quoted"}

    def test_load_env_file_line_without_equals(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("KEY1=value1\nINVALID_LINE\nKEY2=value2\n")
        result = load_env_file(str(env_file))
        assert result == {"KEY1": "value1", "KEY2": "value2"}

    def test_load_env_file_value_with_equals(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("KEY1=value=with=equals\n")
        result = load_env_file(str(env_file))
        assert result == {"KEY1": "value=with=equals"}

    def test_load_env_file_whitespace_trimming(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("  KEY1  =  value1  \n")
        result = load_env_file(str(env_file))
        assert result == {"KEY1": "value1"}

    def test_load_env_file_default_path_not_existing(self):
        with patch.object(Path, "exists", return_value=False):
            result = load_env_file(None)
            assert result == {}

    def test_load_env_file_read_error(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("KEY=value\n")
        with patch("builtins.open", side_effect=PermissionError("denied")):
            result = load_env_file(str(env_file))
            assert result == {}


class TestSetupGitlabConfig:
    """Tests for setup_gitlab_config."""

    def test_setup_with_env_vars(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "GITLAB_TOKEN=test_token\nGITLAB_PROJECT_ID=123\nGITLAB_URL=https://gl.test\n"
        )
        with patch("utils.config_loader.load_env_file", return_value={
            "GITLAB_TOKEN": "test_token",
            "GITLAB_PROJECT_ID": "123",
            "GITLAB_URL": "https://gl.test",
        }):
            result = setup_gitlab_config()
            assert result is True

    def test_setup_with_no_env_vars(self):
        with patch("utils.config_loader.load_env_file", return_value={}):
            result = setup_gitlab_config()
            assert result is False

    def test_setup_partial_env_vars(self):
        with patch("utils.config_loader.load_env_file", return_value={
            "GITLAB_TOKEN": "tok123",
        }):
            result = setup_gitlab_config()
            assert result is True


class TestGetGitlabConfig:
    """Tests for get_gitlab_config."""

    def test_get_config_from_env(self):
        with patch.dict(os.environ, {
            "GITLAB_TOKEN": "test_token",
            "GITLAB_PROJECT_ID": "42",
            "GITLAB_URL": "https://custom.gitlab.com",
        }):
            config = get_gitlab_config()
            assert config["token"] == "test_token"
            assert config["project_id"] == "42"
            assert config["url"] == "https://custom.gitlab.com"

    def test_get_config_defaults(self):
        env = os.environ.copy()
        env.pop("GITLAB_TOKEN", None)
        env.pop("GITLAB_PROJECT_ID", None)
        env.pop("GITLAB_URL", None)
        with patch.dict(os.environ, env, clear=True):
            config = get_gitlab_config()
            assert config["token"] is None
            assert config["project_id"] is None
            assert config["url"] == "https://gitlab.com"


class TestValidateGitlabConfig:
    """Tests for validate_gitlab_config."""

    def test_validate_valid_config(self):
        with patch("utils.config_loader.get_gitlab_config", return_value={
            "token": "tok",
            "project_id": "42",
            "url": "https://gl.com",
        }):
            assert validate_gitlab_config() is True

    def test_validate_missing_token(self):
        with patch("utils.config_loader.get_gitlab_config", return_value={
            "token": None,
            "project_id": "42",
            "url": "https://gl.com",
        }):
            assert validate_gitlab_config() is False

    def test_validate_missing_project_id(self):
        with patch("utils.config_loader.get_gitlab_config", return_value={
            "token": "tok",
            "project_id": None,
            "url": "https://gl.com",
        }):
            assert validate_gitlab_config() is False

    def test_validate_all_missing(self):
        with patch("utils.config_loader.get_gitlab_config", return_value={
            "token": None,
            "project_id": None,
            "url": "https://gl.com",
        }):
            assert validate_gitlab_config() is False
