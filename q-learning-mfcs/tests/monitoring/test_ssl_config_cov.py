"""Tests for monitoring/ssl_config.py - targeting 98%+ coverage."""
import importlib.util
import json
import os
import ssl
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

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

is_development_mode = _mod.is_development_mode
SSLConfig = _mod.SSLConfig
CertificateManager = _mod.CertificateManager
SecurityHeaders = _mod.SecurityHeaders
SSLContextManager = _mod.SSLContextManager
load_ssl_config = _mod.load_ssl_config
save_ssl_config = _mod.save_ssl_config
initialize_ssl_infrastructure = _mod.initialize_ssl_infrastructure
check_ssl_connection = _mod.test_ssl_connection
setup_development_ssl = _mod.setup_development_ssl


class TestIsDevelopmentMode:
    def test_not_development_mode(self):
        with patch.dict(os.environ, {}, clear=True):
            assert is_development_mode() is False

    def test_development_mode_development(self):
        with patch.dict(os.environ, {"MFC_SSL_MODE": "development"}):
            assert is_development_mode() is True

    def test_development_mode_dev(self):
        with patch.dict(os.environ, {"MFC_SSL_MODE": "dev"}):
            assert is_development_mode() is True

    def test_development_mode_uppercase(self):
        with patch.dict(os.environ, {"MFC_SSL_MODE": "DEVELOPMENT"}):
            assert is_development_mode() is True

    def test_development_mode_other(self):
        with patch.dict(os.environ, {"MFC_SSL_MODE": "production"}):
            assert is_development_mode() is False


class TestSSLConfig:
    def test_defaults(self):
        config = SSLConfig()
        assert config.cert_file == "/etc/ssl/certs/mfc-monitoring.crt"
        assert config.key_file == "/etc/ssl/private/mfc-monitoring.key"
        assert config.use_letsencrypt is True
        assert config.domain == "localhost"
        assert config.ssl_version == "TLSv1_2"
        assert config.enable_hsts is True
        assert config.auto_renew is True

    def test_to_dict(self):
        config = SSLConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["domain"] == "localhost"

    def test_from_dict(self):
        data = {"domain": "example.com", "use_letsencrypt": False}
        config = SSLConfig.from_dict(data)
        assert config.domain == "example.com"
        assert config.use_letsencrypt is False

    def test_create_development_config(self, tmp_path):
        config = SSLConfig.create_development_config(project_root=tmp_path)
        assert "ssl_certificates" in config.cert_file
        assert config.use_letsencrypt is False
        assert config.staging is True
        assert config.hsts_max_age == 86400

    def test_create_development_config_no_root(self):
        config = SSLConfig.create_development_config()
        assert config.use_letsencrypt is False


class TestCertificateManager:
    def test_check_certificate_exists_true(self, tmp_path):
        cert = tmp_path / "cert.crt"
        key = tmp_path / "key.pem"
        cert.write_text("cert")
        key.write_text("key")
        config = SSLConfig(cert_file=str(cert), key_file=str(key))
        mgr = CertificateManager(config)
        assert mgr.check_certificate_exists() is True

    def test_check_certificate_exists_false(self):
        config = SSLConfig(cert_file="/nonexistent/cert", key_file="/nonexistent/key")
        mgr = CertificateManager(config)
        assert mgr.check_certificate_exists() is False

    def test_check_certificate_validity_file_not_found(self):
        config = SSLConfig(cert_file="/nonexistent/cert")
        mgr = CertificateManager(config)
        valid, exp = mgr.check_certificate_validity()
        assert valid is False
        assert exp is None

    def test_check_certificate_validity_openssl_success(self, tmp_path):
        cert = tmp_path / "cert.crt"
        cert.write_text("cert")
        config = SSLConfig(cert_file=str(cert), renewal_days_before=30)
        mgr = CertificateManager(config)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "notAfter=Dec 31 23:59:59 2030 GMT\n"
        with patch("subprocess.run", return_value=mock_result):
            valid, exp = mgr.check_certificate_validity()
            assert valid is True
            assert exp is not None

    def test_check_certificate_validity_openssl_bad_date(self, tmp_path):
        cert = tmp_path / "cert.crt"
        cert.write_text("cert")
        config = SSLConfig(cert_file=str(cert))
        mgr = CertificateManager(config)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "notAfter=BADDATE\n"
        with patch("subprocess.run", return_value=mock_result):
            valid, exp = mgr.check_certificate_validity()
            assert valid is False
            assert exp is None

    def test_check_certificate_validity_openssl_fail(self, tmp_path):
        cert = tmp_path / "cert.crt"
        cert.write_text("cert")
        config = SSLConfig(cert_file=str(cert))
        mgr = CertificateManager(config)
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        with patch("subprocess.run", return_value=mock_result):
            valid, exp = mgr.check_certificate_validity()
            assert valid is False

    def test_generate_self_signed_certificate_success(self, tmp_path):
        config = SSLConfig(cert_file=str(tmp_path / "cert.crt"), key_file=str(tmp_path / "key.pem"))
        mgr = CertificateManager(config)
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result), patch("os.chmod"):
            assert mgr.generate_self_signed_certificate() is True

    def test_generate_self_signed_key_fail(self, tmp_path):
        config = SSLConfig(cert_file=str(tmp_path / "cert.crt"), key_file=str(tmp_path / "key.pem"))
        mgr = CertificateManager(config)
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "error"
        with patch("subprocess.run", return_value=mock_result):
            assert mgr.generate_self_signed_certificate() is False

    def test_generate_self_signed_cert_fail(self, tmp_path):
        config = SSLConfig(cert_file=str(tmp_path / "cert.crt"), key_file=str(tmp_path / "key.pem"))
        mgr = CertificateManager(config)
        key_ok = MagicMock(returncode=0)
        cert_fail = MagicMock(returncode=1, stderr="error")
        with patch("subprocess.run", side_effect=[key_ok, cert_fail]):
            assert mgr.generate_self_signed_certificate() is False

    def test_generate_self_signed_exception(self, tmp_path):
        config = SSLConfig(cert_file=str(tmp_path / "c"), key_file=str(tmp_path / "k"))
        mgr = CertificateManager(config)
        with patch("subprocess.run", side_effect=Exception("fail")):
            assert mgr.generate_self_signed_certificate() is False

    def test_request_letsencrypt_disabled(self):
        config = SSLConfig(use_letsencrypt=False)
        mgr = CertificateManager(config)
        assert mgr.request_letsencrypt_certificate() is False

    def test_request_letsencrypt_no_certbot(self):
        config = SSLConfig(use_letsencrypt=True)
        mgr = CertificateManager(config)
        mock_result = MagicMock(returncode=1)
        with patch("subprocess.run", return_value=mock_result):
            assert mgr.request_letsencrypt_certificate() is False

    def test_request_letsencrypt_success(self):
        config = SSLConfig(use_letsencrypt=True, domain="test.com", staging=True)
        mgr = CertificateManager(config)
        check_ok = MagicMock(returncode=0)
        certbot_ok = MagicMock(returncode=0)
        with patch("subprocess.run", side_effect=[check_ok, certbot_ok]):
            assert mgr.request_letsencrypt_certificate() is True
            assert "letsencrypt" in config.cert_file

    def test_request_letsencrypt_fail(self):
        config = SSLConfig(use_letsencrypt=True, domain="test.com")
        mgr = CertificateManager(config)
        check_ok = MagicMock(returncode=0)
        certbot_fail = MagicMock(returncode=1, stderr="error")
        with patch("subprocess.run", side_effect=[check_ok, certbot_fail]):
            assert mgr.request_letsencrypt_certificate() is False

    def test_request_letsencrypt_exception(self):
        config = SSLConfig(use_letsencrypt=True)
        mgr = CertificateManager(config)
        with patch("subprocess.run", side_effect=Exception("fail")):
            assert mgr.request_letsencrypt_certificate() is False

    def test_setup_auto_renewal_success(self):
        config = SSLConfig()
        mgr = CertificateManager(config)
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result), \
             patch.object(Path, "write_text"), patch.object(Path, "chmod"):
            assert mgr.setup_auto_renewal() is True

    def test_setup_auto_renewal_fail(self):
        config = SSLConfig()
        mgr = CertificateManager(config)
        mock_result = MagicMock(returncode=1, stderr="error")
        with patch("subprocess.run", return_value=mock_result), \
             patch.object(Path, "write_text"), patch.object(Path, "chmod"):
            assert mgr.setup_auto_renewal() is False

    def test_setup_auto_renewal_exception(self):
        config = SSLConfig()
        mgr = CertificateManager(config)
        with patch.object(Path, "write_text", side_effect=Exception("fail")):
            assert mgr.setup_auto_renewal() is False


class TestSecurityHeaders:
    def test_all_enabled(self):
        config = SSLConfig(enable_hsts=True, enable_csp=True)
        headers = SecurityHeaders.get_security_headers(config)
        assert "Strict-Transport-Security" in headers
        assert "Content-Security-Policy" in headers
        assert "X-Content-Type-Options" in headers

    def test_hsts_disabled(self):
        config = SSLConfig(enable_hsts=False, enable_csp=True)
        headers = SecurityHeaders.get_security_headers(config)
        assert "Strict-Transport-Security" not in headers

    def test_csp_disabled(self):
        config = SSLConfig(enable_hsts=True, enable_csp=False)
        headers = SecurityHeaders.get_security_headers(config)
        assert "Content-Security-Policy" not in headers


class TestSSLContextManager:
    def test_create_ssl_context(self):
        config = SSLConfig()
        mgr = SSLContextManager(config)
        mock_ctx = MagicMock()
        with patch("ssl.SSLContext", return_value=mock_ctx):
            ctx = mgr.create_ssl_context()
            mock_ctx.load_cert_chain.assert_called_once()
            mock_ctx.set_ciphers.assert_called_once()

    def test_create_ssl_context_failure(self):
        config = SSLConfig()
        mgr = SSLContextManager(config)
        mock_ctx = MagicMock()
        mock_ctx.load_cert_chain.side_effect = Exception("fail")
        with patch("ssl.SSLContext", return_value=mock_ctx):
            with pytest.raises(Exception):
                mgr.create_ssl_context()

    def test_get_uvicorn_ssl_config(self):
        config = SSLConfig()
        mgr = SSLContextManager(config)
        uv = mgr.get_uvicorn_ssl_config()
        assert uv["ssl_keyfile"] == config.key_file
        assert uv["ssl_certfile"] == config.cert_file


class TestLoadSSLConfig:
    def test_load_from_file(self, tmp_path):
        f = tmp_path / "ssl.json"
        f.write_text(json.dumps({"domain": "test.com", "use_letsencrypt": False}))
        config = load_ssl_config(str(f))
        assert config.domain == "test.com"

    def test_load_invalid_json(self, tmp_path):
        f = tmp_path / "ssl.json"
        f.write_text("not json")
        config = load_ssl_config(str(f))
        assert config.domain == "localhost"

    def test_load_development_mode(self):
        with patch.dict(os.environ, {"MFC_SSL_MODE": "development"}), \
             patch.object(Path, "mkdir"):
            config = load_ssl_config()
            assert config.use_letsencrypt is False

    def test_load_env_overrides(self, tmp_path):
        f = tmp_path / "ssl.json"
        f.write_text(json.dumps({"domain": "default.com"}))
        with patch.dict(os.environ, {
            "MFC_SSL_DOMAIN": "override.com",
            "MFC_HTTPS_API_PORT": "9443",
            "MFC_SSL_USE_LETSENCRYPT": "true",
        }):
            config = load_ssl_config(str(f))
            assert config.domain == "override.com"
            assert config.https_port_api == 9443
            assert config.use_letsencrypt is True


class TestSaveSSLConfig:
    def test_save_success(self, tmp_path):
        f = tmp_path / "ssl.json"
        config = SSLConfig(domain="saved.com")
        assert save_ssl_config(config, str(f)) is True
        saved = json.loads(f.read_text())
        assert saved["domain"] == "saved.com"

    def test_save_failure(self):
        config = SSLConfig()
        with patch("builtins.open", side_effect=PermissionError("denied")):
            assert save_ssl_config(config, "/x/y/z.json") is False


class TestInitializeSSLInfrastructure:
    def test_cert_exists_valid(self):
        mock_mgr = MagicMock()
        mock_mgr.check_certificate_exists.return_value = True
        mock_mgr.check_certificate_validity.return_value = (True, datetime(2030, 1, 1))
        config = SSLConfig(use_letsencrypt=False, auto_renew=False)
        with patch.object(_mod, "CertificateManager", return_value=mock_mgr), \
             patch.object(_mod, "save_ssl_config", return_value=True):
            success, _ = initialize_ssl_infrastructure(config)
            assert success is True

    def test_cert_not_exists_self_signed(self):
        mock_mgr = MagicMock()
        mock_mgr.check_certificate_exists.return_value = False
        mock_mgr.generate_self_signed_certificate.return_value = True
        config = SSLConfig(use_letsencrypt=False, auto_renew=False)
        with patch.object(_mod, "CertificateManager", return_value=mock_mgr), \
             patch.object(_mod, "save_ssl_config", return_value=True):
            success, _ = initialize_ssl_infrastructure(config)
            assert success is True

    def test_letsencrypt_fallback(self):
        mock_mgr = MagicMock()
        mock_mgr.check_certificate_exists.return_value = False
        mock_mgr.request_letsencrypt_certificate.return_value = False
        mock_mgr.generate_self_signed_certificate.return_value = True
        config = SSLConfig(use_letsencrypt=True, domain="test.com", auto_renew=False)
        with patch.object(_mod, "CertificateManager", return_value=mock_mgr), \
             patch.object(_mod, "save_ssl_config", return_value=True):
            success, _ = initialize_ssl_infrastructure(config)
            assert success is True

    def test_total_failure(self):
        mock_mgr = MagicMock()
        mock_mgr.check_certificate_exists.return_value = False
        mock_mgr.generate_self_signed_certificate.return_value = False
        config = SSLConfig(use_letsencrypt=False, auto_renew=False)
        with patch.object(_mod, "CertificateManager", return_value=mock_mgr):
            success, _ = initialize_ssl_infrastructure(config)
            assert success is False

    def test_auto_renewal(self):
        mock_mgr = MagicMock()
        mock_mgr.check_certificate_exists.return_value = True
        mock_mgr.check_certificate_validity.return_value = (True, datetime(2030, 1, 1))
        config = SSLConfig(use_letsencrypt=True, auto_renew=True)
        with patch.object(_mod, "CertificateManager", return_value=mock_mgr), \
             patch.object(_mod, "save_ssl_config", return_value=True):
            success, _ = initialize_ssl_infrastructure(config)
            assert success is True
            mock_mgr.setup_auto_renewal.assert_called_once()

    def test_no_config(self):
        mock_config = SSLConfig()
        mock_mgr = MagicMock()
        mock_mgr.check_certificate_exists.return_value = True
        mock_mgr.check_certificate_validity.return_value = (True, datetime(2030, 1, 1))
        with patch.object(_mod, "load_ssl_config", return_value=mock_config), \
             patch.object(_mod, "CertificateManager", return_value=mock_mgr), \
             patch.object(_mod, "save_ssl_config", return_value=True):
            success, _ = initialize_ssl_infrastructure()
            assert success is True


class TestCheckSSLConnection:
    def test_success(self):
        mock_ssock = MagicMock()
        mock_ssock.version.return_value = "TLSv1.3"
        mock_sock = MagicMock()
        mock_sock.__enter__ = MagicMock(return_value=mock_sock)
        mock_sock.__exit__ = MagicMock(return_value=False)
        mock_ctx = MagicMock()
        mock_ctx.wrap_socket.return_value.__enter__ = MagicMock(return_value=mock_ssock)
        mock_ctx.wrap_socket.return_value.__exit__ = MagicMock(return_value=False)
        with patch("ssl.create_default_context", return_value=mock_ctx), \
             patch("socket.create_connection", return_value=mock_sock):
            assert check_ssl_connection("localhost", 8443) is True

    def test_failure(self):
        with patch("socket.create_connection", side_effect=Exception("fail")):
            assert check_ssl_connection("localhost", 8443) is False


class TestSetupDevelopmentSSL:
    def test_success(self, tmp_path):
        mock_mgr = MagicMock()
        mock_mgr.generate_self_signed_certificate.return_value = True
        with patch.object(Path, "mkdir"), \
             patch.object(_mod, "CertificateManager", return_value=mock_mgr), \
             patch.object(_mod, "save_ssl_config", return_value=True):
            assert setup_development_ssl(tmp_path) is True

    def test_cert_gen_failure(self, tmp_path):
        mock_mgr = MagicMock()
        mock_mgr.generate_self_signed_certificate.return_value = False
        with patch.object(Path, "mkdir"), \
             patch.object(_mod, "CertificateManager", return_value=mock_mgr):
            assert setup_development_ssl(tmp_path) is False

    def test_save_failure(self, tmp_path):
        mock_mgr = MagicMock()
        mock_mgr.generate_self_signed_certificate.return_value = True
        with patch.object(Path, "mkdir"), \
             patch.object(_mod, "CertificateManager", return_value=mock_mgr), \
             patch.object(_mod, "save_ssl_config", return_value=False):
            assert setup_development_ssl(tmp_path) is False

    def test_exception(self, tmp_path):
        with patch.object(_mod, "SSLConfig") as MockConfig:
            MockConfig.create_development_config.side_effect = Exception("fail")
            assert setup_development_ssl(tmp_path) is False

    def test_no_project_root(self):
        mock_mgr = MagicMock()
        mock_mgr.generate_self_signed_certificate.return_value = True
        with patch.object(Path, "mkdir"), \
             patch.object(_mod, "CertificateManager", return_value=mock_mgr), \
             patch.object(_mod, "save_ssl_config", return_value=True):
            assert setup_development_ssl(None) is True
