"""Coverage tests for cert_manager.py - lines 44-349."""
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, mock_open

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'monitoring'),
)

from pathlib import Path


# Create a real base class to replace the mocked CertificateManager
class _FakeCertMgr:
    def __init__(self, config):
        self.config = config
        self.cert_dir = Path("/etc/ssl/certs")
        self.key_dir = Path("/etc/ssl/private")
        self.letsencrypt_dir = Path("/etc/letsencrypt")

    def check_certificate_exists(self):
        return False

    def check_certificate_validity(self):
        return (False, None)

    def generate_self_signed_certificate(self):
        return True


mock_ssl = MagicMock()
mock_ssl.CertificateManager = _FakeCertMgr
mock_ssl.SSLConfig = MagicMock

with patch.dict(sys.modules, {
    'monitoring.ssl_config': mock_ssl,
    'ssl_config': mock_ssl,
}):
    import cert_manager as _cm_mod
    from cert_manager import (
        EnhancedCertificateManager,
        print_certificate_report,
    )


def _make_config():
    cfg = MagicMock()
    cfg.domain = "localhost"
    cfg.cert_file = "/tmp/cert.pem"
    cfg.key_file = "/tmp/key.pem"
    cfg.renewal_days_before = 30
    cfg.use_letsencrypt = False
    cfg.auto_renew = True
    return cfg


def _make_mgr():
    """Create an EnhancedCertificateManager with fake base class."""
    with patch('pathlib.Path.exists', return_value=False):
        return EnhancedCertificateManager(_make_config())


@pytest.mark.coverage_extra
class TestLoadNotificationConfig:
    def test_default(self):
        mgr = _make_mgr()
        assert mgr.notification_config["enabled"] is False

    def test_from_file(self):
        import json
        data = {"enabled": True, "to_emails": ["a@b.com"]}
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(data))):
            mgr = EnhancedCertificateManager(_make_config())
        assert mgr.notification_config["enabled"] is True

    def test_file_error(self):
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', side_effect=Exception("err")):
            mgr = EnhancedCertificateManager(_make_config())
        assert mgr.notification_config["enabled"] is False


@pytest.mark.coverage_extra
class TestCheckCertificateExpiry:
    def test_no_expiry(self):
        mgr = _make_mgr()
        mgr.check_certificate_validity = MagicMock(return_value=(False, None))
        needs, expiry, days = mgr.check_certificate_expiry()
        assert needs is True
        assert expiry is None

    def test_valid(self):
        mgr = _make_mgr()
        future = datetime.now() + timedelta(days=90)
        mgr.check_certificate_validity = MagicMock(return_value=(True, future))
        needs, _, days = mgr.check_certificate_expiry()
        assert needs is False
        assert days > 30

    def test_expiring(self):
        mgr = _make_mgr()
        future = datetime.now() + timedelta(days=5)
        mgr.check_certificate_validity = MagicMock(return_value=(True, future))
        needs, _, days = mgr.check_certificate_expiry()
        assert needs is True


@pytest.mark.coverage_extra
class TestSendNotification:
    def test_disabled(self):
        mgr = _make_mgr()
        mgr.notification_config = {"enabled": False}
        mgr.send_notification("test", "msg")

    def test_no_emails(self):
        mgr = _make_mgr()
        mgr.notification_config = {"enabled": True, "to_emails": []}
        mgr.send_notification("test", "msg")

    def test_success_with_tls(self):
        mgr = _make_mgr()
        mgr.notification_config = {
            "enabled": True, "to_emails": ["a@b.com"],
            "from_email": "x@y.com", "smtp_server": "localhost",
            "smtp_port": 587, "use_tls": True,
            "smtp_username": "u", "smtp_password": "p",
        }
        with patch.object(_cm_mod.smtplib, 'SMTP') as ms:
            srv = MagicMock()
            ms.return_value = srv
            mgr.send_notification("test", "msg", is_critical=True)
            srv.starttls.assert_called_once()
            srv.login.assert_called_once()

    def test_no_tls_no_auth(self):
        mgr = _make_mgr()
        mgr.notification_config = {
            "enabled": True, "to_emails": ["a@b.com"],
            "from_email": "x@y.com", "smtp_server": "localhost",
            "smtp_port": 25, "use_tls": False, "smtp_username": "",
        }
        with patch.object(_cm_mod.smtplib, 'SMTP') as ms:
            srv = MagicMock()
            ms.return_value = srv
            mgr.send_notification("test", "msg")
            srv.starttls.assert_not_called()

    def test_failure(self):
        mgr = _make_mgr()
        mgr.notification_config = {
            "enabled": True, "to_emails": ["a@b.com"],
            "from_email": "x@y.com", "smtp_server": "localhost",
            "smtp_port": 587, "use_tls": False,
        }
        with patch.object(_cm_mod.smtplib, 'SMTP', side_effect=Exception("fail")):
            mgr.send_notification("test", "msg")


@pytest.mark.coverage_extra
class TestMonitorCertificate:
    def test_no_cert(self):
        mgr = _make_mgr()
        mgr.check_certificate_exists = MagicMock(return_value=False)
        r = mgr.monitor_certificate()
        assert not r["certificate_exists"]

    def test_healthy(self):
        mgr = _make_mgr()
        mgr.check_certificate_exists = MagicMock(return_value=True)
        mgr.check_certificate_expiry = MagicMock(
            return_value=(False, datetime.now() + timedelta(days=90), 90),
        )
        mgr._check_cron_job_exists = MagicMock(return_value=True)
        r = mgr.monitor_certificate()
        assert r["certificate_valid"]

    def test_urgent_renewal(self):
        mgr = _make_mgr()
        mgr.check_certificate_exists = MagicMock(return_value=True)
        mgr.check_certificate_expiry = MagicMock(
            return_value=(True, datetime.now() + timedelta(days=3), 3),
        )
        mgr._check_cron_job_exists = MagicMock(return_value=False)
        r = mgr.monitor_certificate()
        assert r["needs_renewal"]

    def test_renewal_soon(self):
        mgr = _make_mgr()
        mgr.check_certificate_exists = MagicMock(return_value=True)
        mgr.check_certificate_expiry = MagicMock(
            return_value=(True, datetime.now() + timedelta(days=20), 20),
        )
        mgr._check_cron_job_exists = MagicMock(return_value=True)
        r = mgr.monitor_certificate()
        assert r["needs_renewal"]


@pytest.mark.coverage_extra
class TestCheckCronJobExists:
    def test_with_crontab_found(self):
        mgr = _make_mgr()
        mock_cron = MagicMock()
        job = MagicMock()
        job.command = "python cert_manager.py # mfc-cert-renewal"
        mock_cron.__iter__ = MagicMock(return_value=iter([job]))
        mock_ct = MagicMock()
        mock_ct.CronTab = MagicMock(return_value=mock_cron)
        with patch.dict(sys.modules, {'crontab': mock_ct}):
            result = mgr._check_cron_job_exists()
        assert result is True

    def test_fallback_found(self):
        mgr = _make_mgr()
        with patch.dict(sys.modules, {'crontab': None}):
            mr = MagicMock(returncode=0, stdout="mfc-cert-renewal")
            with patch('subprocess.run', return_value=mr):
                assert mgr._check_cron_job_exists() is True

    def test_fallback_not_found(self):
        mgr = _make_mgr()
        with patch.dict(sys.modules, {'crontab': None}):
            mr = MagicMock(returncode=0, stdout="other-job")
            with patch('subprocess.run', return_value=mr):
                assert mgr._check_cron_job_exists() is False

    def test_exception(self):
        mgr = _make_mgr()
        with patch.dict(sys.modules, {'crontab': None}):
            with patch('subprocess.run', side_effect=Exception("err")):
                assert mgr._check_cron_job_exists() is False


@pytest.mark.coverage_extra
class TestRenewCertificateIfNeeded:
    def test_not_needed(self):
        mgr = _make_mgr()
        mgr.send_notification = MagicMock()
        mgr.check_certificate_expiry = MagicMock(
            return_value=(False, datetime.now() + timedelta(days=90), 90),
        )
        ok, msg = mgr.renew_certificate_if_needed()
        assert ok is True

    def test_letsencrypt_success(self):
        mgr = _make_mgr()
        mgr.send_notification = MagicMock()
        mgr.config.use_letsencrypt = True
        mgr.check_certificate_expiry = MagicMock(
            return_value=(True, datetime.now() + timedelta(days=5), 5),
        )
        with patch('subprocess.run', return_value=MagicMock(returncode=0)):
            ok, _ = mgr.renew_certificate_if_needed()
        assert ok is True

    def test_letsencrypt_fail(self):
        mgr = _make_mgr()
        mgr.send_notification = MagicMock()
        mgr.config.use_letsencrypt = True
        mgr.check_certificate_expiry = MagicMock(
            return_value=(True, datetime.now() + timedelta(days=5), 5),
        )
        with patch('subprocess.run', return_value=MagicMock(returncode=1, stderr="e")):
            ok, _ = mgr.renew_certificate_if_needed()
        assert ok is False

    def test_letsencrypt_timeout(self):
        import subprocess
        mgr = _make_mgr()
        mgr.send_notification = MagicMock()
        mgr.config.use_letsencrypt = True
        mgr.check_certificate_expiry = MagicMock(
            return_value=(True, datetime.now() + timedelta(days=5), 5),
        )
        with patch('subprocess.run', side_effect=subprocess.TimeoutExpired("c", 300)):
            ok, _ = mgr.renew_certificate_if_needed()
        assert ok is False

    def test_self_signed_success(self):
        mgr = _make_mgr()
        mgr.send_notification = MagicMock()
        mgr.config.use_letsencrypt = False
        mgr.check_certificate_expiry = MagicMock(
            return_value=(True, datetime.now() + timedelta(days=5), 5),
        )
        mgr.generate_self_signed_certificate = MagicMock(return_value=True)
        ok, _ = mgr.renew_certificate_if_needed()
        assert ok is True

    def test_self_signed_failure(self):
        mgr = _make_mgr()
        mgr.send_notification = MagicMock()
        mgr.config.use_letsencrypt = False
        mgr.check_certificate_expiry = MagicMock(
            return_value=(True, datetime.now() + timedelta(days=5), 5),
        )
        mgr.generate_self_signed_certificate = MagicMock(return_value=False)
        ok, _ = mgr.renew_certificate_if_needed()
        assert ok is False


@pytest.mark.coverage_extra
class TestCronManagement:
    def test_setup_success(self):
        mgr = _make_mgr()
        with patch('subprocess.run', return_value=MagicMock(returncode=0)):
            assert mgr.setup_monitoring_cron() is True

    def test_setup_failure(self):
        mgr = _make_mgr()
        with patch('subprocess.run', return_value=MagicMock(returncode=1, stderr="e")):
            assert mgr.setup_monitoring_cron() is False

    def test_setup_exception(self):
        mgr = _make_mgr()
        with patch('subprocess.run', side_effect=Exception("err")):
            assert mgr.setup_monitoring_cron() is False

    def test_remove_success(self):
        mgr = _make_mgr()
        with patch('subprocess.run', return_value=MagicMock(returncode=0)):
            assert mgr.remove_monitoring_cron() is True

    def test_remove_failure(self):
        mgr = _make_mgr()
        with patch('subprocess.run', return_value=MagicMock(returncode=1, stderr="e")):
            assert mgr.remove_monitoring_cron() is False

    def test_remove_exception(self):
        mgr = _make_mgr()
        with patch('subprocess.run', side_effect=Exception("err")):
            assert mgr.remove_monitoring_cron() is False


@pytest.mark.coverage_extra
class TestPrintCertificateReport:
    def test_with_expiry_valid(self):
        report = {
            "certificate_exists": True, "certificate_valid": True,
            "auto_renewal_configured": True, "expiry_date": "2026-01-01",
            "needs_renewal": False, "recommendations": ["healthy"],
        }
        print_certificate_report(report)

    def test_needs_renewal(self):
        report = {
            "certificate_exists": True, "certificate_valid": False,
            "auto_renewal_configured": False, "expiry_date": "2025-01-01",
            "needs_renewal": True, "recommendations": [],
        }
        print_certificate_report(report)

    def test_no_expiry(self):
        report = {
            "certificate_exists": False, "certificate_valid": False,
            "auto_renewal_configured": False, "expiry_date": None,
            "needs_renewal": False, "recommendations": ["Generate cert"],
        }
        print_certificate_report(report)
