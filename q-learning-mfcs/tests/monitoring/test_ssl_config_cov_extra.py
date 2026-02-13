"""Extra coverage tests for monitoring/ssl_config.py - targeting 99%+.

Covers missing lines:
- 466-467: dev config file exists path in load_ssl_config
- 471: env var fallback for config_file
- 535: save_ssl_config default path from env var
"""
import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import the module directly to avoid triggering monitoring/__init__.py
_src = os.path.join(os.path.dirname(__file__), "..", "..", "src")
_spec = importlib.util.spec_from_file_location(
    "monitoring.ssl_config",
    os.path.join(_src, "monitoring", "ssl_config.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["monitoring.ssl_config"] = _mod
_spec.loader.exec_module(_mod)

SSLConfig = _mod.SSLConfig
load_ssl_config = _mod.load_ssl_config
save_ssl_config = _mod.save_ssl_config
class TestLoadSSLConfigDevFile:
    """Cover lines 465-467: dev config file exists path."""

    def test_load_detects_dev_config_file(self, tmp_path):
        """When ssl_config_dev.json exists beside the module, it should be used."""
        dev_data = {"domain": "devfile.local", "use_letsencrypt": False}
        dev_file = tmp_path / "ssl_config_dev.json"
        dev_file.write_text(json.dumps(dev_data))

        # Patch Path(__file__).parent to point to tmp_path so dev_config_file
        # resolves to our temp file.
        mock_parent = tmp_path

        # We need to make the dev_config_file path resolve to our file.
        # The source does: Path(__file__).parent / "ssl_config_dev.json"
        # We patch Path to intercept that specific call.
        original_path = Path

        class PatchedPath(type(Path())):
            pass

        with patch.object(_mod, "Path", wraps=Path) as mock_path_cls:
            # Create a mock that returns tmp_path as parent
            mock_file_path = MagicMock()
            mock_file_path.parent = tmp_path

            def path_side_effect(*args, **kwargs):
                if args and args[0] == _mod.__file__:
                    return mock_file_path
                return original_path(*args, **kwargs)

            mock_path_cls.side_effect = path_side_effect

            # Clear env vars that would override, and ensure no dev mode
            with patch.dict(os.environ, {}, clear=True):
                config = load_ssl_config(config_file=None)
                assert config.domain == "devfile.local"

    def test_load_env_var_fallback_config_path(self, tmp_path):
        """Cover line 471: when config_file is None and no dev file,
        fall back to MFC_SSL_CONFIG env var."""
        config_data = {"domain": "envvar.local", "use_letsencrypt": False}
        config_file = tmp_path / "custom_ssl.json"
        config_file.write_text(json.dumps(config_data))

        # Ensure dev config file does NOT exist
        with patch.dict(os.environ, {"MFC_SSL_CONFIG": str(config_file)}, clear=True):
            # Make dev_config_file.exists() return False
            original_path = Path

            def path_side_effect(*args, **kwargs):
                if args and args[0] == _mod.__file__:
                    mock_file_path = MagicMock()
                    mock_file_path.parent.__truediv__ = lambda self, x: original_path(
                        "/nonexistent_dir"
                    ) / x
                    return mock_file_path
                return original_path(*args, **kwargs)

            with patch.object(_mod, "Path", side_effect=path_side_effect):
                config = load_ssl_config(config_file=None)
                assert config.domain == "envvar.local"

class TestSaveSSLConfigDefaultPath:
    """Cover line 535: save_ssl_config with config_file=None uses env var."""

    def test_save_default_path_from_env(self, tmp_path):
        config_file = str(tmp_path / "saved_ssl.json")
        config = SSLConfig(domain="saved.local")

        with patch.dict(os.environ, {"MFC_SSL_CONFIG": config_file}):
            result = save_ssl_config(config, config_file=None)

        assert result is True
        saved = json.loads(Path(config_file).read_text())
        assert saved["domain"] == "saved.local"

    def test_save_default_path_no_env(self, tmp_path):
        """When MFC_SSL_CONFIG is not set, uses /etc/mfc/ssl-config.json.
        This will fail due to permissions, which triggers the except path."""
        config = SSLConfig(domain="noenv.local")

        with patch.dict(os.environ, {}, clear=True):
            # This will fail (can't write to /etc/mfc/) and return False
            result = save_ssl_config(config, config_file=None)
            assert result is False