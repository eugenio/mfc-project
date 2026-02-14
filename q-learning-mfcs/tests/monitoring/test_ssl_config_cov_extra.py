"""Extra coverage tests for ssl_config.py - lines 466-467, 471, 535."""
import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
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
@pytest.mark.coverage_extra
class TestLoadSSLConfigDevFile:
    """Cover lines 465-467: dev config file exists path."""

    def test_load_detects_dev_config_file(self, tmp_path):
        dev_data = {
            "domain": "devfile.local",
            "use_letsencrypt": False,
        }
        dev_file = tmp_path / "ssl_config_dev.json"
        dev_file.write_text(json.dumps(dev_data))

        original_path = Path
        mock_file_path = MagicMock()
        mock_file_path.parent = tmp_path

        def path_side_effect(*args, **kwargs):
            if args and args[0] == _mod.__file__:
                return mock_file_path
            return original_path(*args, **kwargs)

        with patch.object(
            _mod, "Path", side_effect=path_side_effect,
        ), patch.dict(os.environ, {}, clear=True):
            config = load_ssl_config(config_file=None)
            assert config.domain == "devfile.local"

    def test_load_env_var_fallback_config_path(self, tmp_path):
        config_data = {
            "domain": "envvar.local",
            "use_letsencrypt": False,
        }
        config_file = tmp_path / "custom_ssl.json"
        config_file.write_text(json.dumps(config_data))

        original_path = Path

        def path_side_effect(*args, **kwargs):
            if args and args[0] == _mod.__file__:
                mock_fp = MagicMock()
                nonexist = original_path("/nonexistent_dir_xyz")
                mock_fp.parent.__truediv__ = (
                    lambda self, x: nonexist / x
                )
                return mock_fp
            return original_path(*args, **kwargs)

        env = {"MFC_SSL_CONFIG": str(config_file)}
        with patch.object(
            _mod, "Path", side_effect=path_side_effect,
        ), patch.dict(os.environ, env, clear=True):
            config = load_ssl_config(config_file=None)
            assert config.domain == "envvar.local"

@pytest.mark.coverage_extra
class TestSaveSSLConfigDefaultPath:
    """Cover line 535: save default path from env var."""

    def test_save_default_path_from_env(self, tmp_path):
        config_file = str(tmp_path / "saved_ssl.json")
        config = SSLConfig(domain="saved.local")

        with patch.dict(
            os.environ, {"MFC_SSL_CONFIG": config_file},
        ):
            result = save_ssl_config(config, config_file=None)

        assert result is True
        saved = json.loads(Path(config_file).read_text())
        assert saved["domain"] == "saved.local"

    def test_save_default_path_no_env(self):
        config = SSLConfig(domain="noenv.local")

        with patch.dict(os.environ, {}, clear=True):
            result = save_ssl_config(config, config_file=None)
            assert result is False