"""Extra coverage tests for monitoring/cert_manager.py - targeting 99%+.

Covers remaining uncovered paths:
- EnhancedCertificateManager.monitor_certificate with no auto-renewal but valid cert
- main() with --renew-if-needed that actually needs renewal and succeeds
- main() with --renew-if-needed that needs renewal and fails
- main() with --quiet flag on various paths
- main() default action (no arguments)
- print_certificate_report edge cases
- setup_notification_config with various input combinations
- _check_cron_job_exists with crontab module having no matching jobs
- check_certificate_expiry edge case: days == renewal_days_before exactly
"""
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

_src = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.insert(0, _src)
sys.path.insert(0, os.path.join(_src, "monitoring"))


class _FakeCertMgr:
    """Fake base CertificateManager to avoid real ssl_config import."""

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
mock_ssl.load_ssl_config = MagicMock()
mock_ssl.initialize_ssl_infrastructure = MagicMock()

with patch.dict(sys.modules, {
    "monitoring.ssl_config": mock_ssl,
    "ssl_config": mock_ssl,
}):
    _spec = importlib.util.spec_from_file_location(
        "monitoring.cert_manager",
        os.path.join(_src, "monitoring", "cert_manager.py"),
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["monitoring.cert_manager"] = _mod
    _spec.loader.exec_module(_mod)

EnhancedCertificateManager = _mod.EnhancedCertificateManager
print_certificate_report = _mod.print_certificate_report
setup_notification_config = _mod.setup_notification_config
main_func = _mod.main


def _make_config(**overrides):
    cfg = MagicMock()
    cfg.domain = overrides.get("domain", "localhost")
    cfg.cert_file = overrides.get("cert_file", "/tmp/cert.pem")
    cfg.key_file = overrides.get("key_file", "/tmp/key.pem")
    cfg.renewal_days_before = overrides.get("renewal_days_before", 30)
    cfg.use_letsencrypt = overrides.get("use_letsencrypt", False)
    cfg.email = overrides.get("email", "admin@example.com")
    cfg.staging = overrides.get("staging", False)
    cfg.auto_renew = overrides.get("auto_renew", True)
    return cfg


def _make_mgr(**config_overrides):
    with patch("pathlib.Path.exists", return_value=False):
        return EnhancedCertificateManager(_make_config(**config_overrides))


@pytest.mark.coverage_extra
class TestMonitorCertificateEdgeCases:
    def test_monitor_valid_no_auto_renewal(self):
        """Cover: cert valid, auto-renewal not configured."""
        mgr = _make_mgr()
        mgr.check_certificate_exists = MagicMock(return_value=True)
        future = datetime.now() + timedelta(days=60)
        mgr.check_certificate_expiry = MagicMock(
            return_value=(False, future, 60)
        )
        mgr._check_cron_job_exists = MagicMock(return_value=False)
        report = mgr.monitor_certificate()
        assert report["certificate_valid"] is True
        assert report["auto_renewal_configured"] is False
        assert any("Auto-renewal" in r for r in report["recommendations"])
        assert any("healthy" in r for r in report["recommendations"])

    def test_monitor_cert_exists_but_needs_renewal_expired(self):
        """Cover: cert exists, needs renewal, days_until_expiry=0."""
        mgr = _make_mgr()
        mgr.check_certificate_exists = MagicMock(return_value=True)
        past = datetime.now() - timedelta(days=1)
        mgr.check_certificate_expiry = MagicMock(
            return_value=(True, past, -1)
        )
        mgr._check_cron_job_exists = MagicMock(return_value=True)
        report = mgr.monitor_certificate()
        assert report["needs_renewal"] is True
        # days_until_expiry < 0 means expired
        assert report["days_until_expiry"] == -1
        # Should have URGENT recommendation since days <= 7
        assert any("URGENT" in r for r in report["recommendations"])


@pytest.mark.coverage_extra
class TestCheckCertificateExpiryEdge:
    def test_expiry_exactly_at_threshold(self):
        """Cover: days_until_expiry == renewal_days_before exactly."""
        mgr = _make_mgr(renewal_days_before=30)
        future = datetime.now() + timedelta(days=30)
        mgr.check_certificate_validity = MagicMock(return_value=(True, future))
        needs, expiry, days = mgr.check_certificate_expiry()
        # 30 days until expiry, renewal_days_before=30 -> needs_renewal=True
        assert needs is True
        assert days <= 30


@pytest.mark.coverage_extra
class TestCheckCronJobExistsEdgeCases:
    def test_crontab_import_error_subprocess_no_crontab(self):
        """Cover: crontab ImportError, subprocess crontab -l returns empty."""
        mgr = _make_mgr()
        mr = MagicMock(returncode=0, stdout="")
        with patch.dict(sys.modules, {"crontab": None}):
            with patch.object(_mod, "subprocess") as mock_sub:
                mock_sub.run.return_value = mr
                result = mgr._check_cron_job_exists()
        assert result is False

    def test_crontab_module_no_jobs(self):
        """Cover: crontab module available, no jobs at all."""
        mgr = _make_mgr()
        mock_cron = MagicMock()
        mock_cron.__iter__ = MagicMock(return_value=iter([]))
        mock_ct = MagicMock()
        mock_ct.CronTab = MagicMock(return_value=mock_cron)
        with patch.dict(sys.modules, {"crontab": mock_ct}):
            result = mgr._check_cron_job_exists()
        assert result is False

    def test_crontab_module_job_no_command(self):
        """Cover: crontab job exists but command is None."""
        mgr = _make_mgr()
        job = MagicMock()
        job.command = None
        mock_cron = MagicMock()
        mock_cron.__iter__ = MagicMock(return_value=iter([job]))
        mock_ct = MagicMock()
        mock_ct.CronTab = MagicMock(return_value=mock_cron)
        with patch.dict(sys.modules, {"crontab": mock_ct}):
            result = mgr._check_cron_job_exists()
        assert result is False


@pytest.mark.coverage_extra
class TestRenewCertificateIfNeededEdgeCases:
    def test_renew_letsencrypt_exception_generic(self):
        """Cover: LE renewal raises generic Exception."""
        mgr = _make_mgr(use_letsencrypt=True)
        mgr.send_notification = MagicMock()
        mgr.check_certificate_expiry = MagicMock(
            return_value=(True, datetime.now() + timedelta(days=5), 5)
        )
        with patch.object(_mod, "subprocess") as mock_sub:
            mock_sub.run.side_effect = RuntimeError("unexpected")
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired
            ok, msg = mgr.renew_certificate_if_needed()
        assert ok is False
        assert "error" in msg.lower() or "Error" in msg


@pytest.mark.coverage_extra
class TestPrintCertificateReportEdgeCases:
    def test_report_all_false_with_recommendations(self):
        """Cover: all fields False, multiple recommendations."""
        report = {
            "certificate_exists": False,
            "certificate_valid": False,
            "auto_renewal_configured": False,
            "expiry_date": None,
            "needs_renewal": False,
            "recommendations": ["Generate cert", "Setup auto-renewal"],
        }
        print_certificate_report(report)

    def test_report_valid_not_renewal(self):
        """Cover: expiry_date present, needs_renewal=False."""
        report = {
            "certificate_exists": True,
            "certificate_valid": True,
            "auto_renewal_configured": True,
            "expiry_date": "2027-06-15",
            "needs_renewal": False,
            "recommendations": [],
        }
        print_certificate_report(report)


@pytest.mark.coverage_extra
class TestSetupNotificationConfigEdgeCases:
    def test_enabled_default_smtp(self):
        """Cover: enabled with all defaults (empty inputs)."""
        inputs = iter([
            "yes",  # enable
            "",     # smtp server (default localhost)
            "",     # port (default 587)
            "",     # username (empty)
            "",     # from email (default)
            "",     # to emails (empty)
            "",     # use tls (default yes)
        ])
        with patch("builtins.input", side_effect=lambda _: next(inputs)), \
             patch.object(_mod, "Path") as mp, \
             patch("builtins.open", mock_open()):
            mp.return_value.parent.mkdir = MagicMock()
            result = setup_notification_config()
        assert result is True

    def test_not_enabled(self):
        """Cover: user says no to enable."""
        with patch("builtins.input", return_value="no"), \
             patch.object(_mod, "Path") as mp, \
             patch("builtins.open", mock_open()):
            mp.return_value.parent.mkdir = MagicMock()
            result = setup_notification_config()
        assert result is True


@pytest.mark.coverage_extra
class TestMainFunctionEdgeCases:
    def _make_args(self, **overrides):
        defaults = dict(
            quiet=False, domain=None, email=None,
            staging=False, setup_notifications=False,
            test_notifications=False, init=False,
            setup_cron=False, remove_cron=False,
            monitor=False, renew=False, renew_if_needed=False,
        )
        defaults.update(overrides)
        args = MagicMock()
        for k, v in defaults.items():
            setattr(args, k, v)
        return args

    def test_main_quiet_mode(self):
        """Cover: --quiet flag sets logging level."""
        with patch.object(_mod, "argparse") as mock_ap, \
             patch.object(_mod, "load_ssl_config") as mock_lsc, \
             patch.object(_mod, "EnhancedCertificateManager") as mock_ecm, \
             patch.object(_mod, "print_certificate_report"):
            mock_lsc.return_value = _make_config()
            mock_ap.ArgumentParser.return_value.parse_args.return_value = (
                self._make_args(quiet=True)
            )
            mock_ecm.return_value.monitor_certificate.return_value = {
                "certificate_exists": True,
                "needs_renewal": False,
                "days_until_expiry": 90,
            }
            main_func()

    def test_main_domain_email_override(self):
        """Cover: --domain and --email override ssl_config."""
        with patch.object(_mod, "argparse") as mock_ap, \
             patch.object(_mod, "load_ssl_config") as mock_lsc, \
             patch.object(_mod, "EnhancedCertificateManager") as mock_ecm, \
             patch.object(_mod, "print_certificate_report"):
            cfg = _make_config()
            mock_lsc.return_value = cfg
            mock_ap.ArgumentParser.return_value.parse_args.return_value = (
                self._make_args(domain="custom.com", email="custom@test.com", staging=True)
            )
            mock_ecm.return_value.monitor_certificate.return_value = {
                "certificate_exists": True,
                "needs_renewal": False,
                "days_until_expiry": 90,
            }
            main_func()
            assert cfg.domain == "custom.com"
            assert cfg.email == "custom@test.com"
            assert cfg.staging is True

    def test_main_renew_if_needed_needs_and_succeeds(self):
        """Cover: --renew-if-needed where cert needs renewal and succeeds."""
        with patch.object(_mod, "argparse") as mock_ap, \
             patch.object(_mod, "load_ssl_config") as mock_lsc, \
             patch.object(_mod, "EnhancedCertificateManager") as mock_ecm:
            mock_lsc.return_value = _make_config()
            mock_ap.ArgumentParser.return_value.parse_args.return_value = (
                self._make_args(renew_if_needed=True)
            )
            mock_ecm.return_value.check_certificate_expiry.return_value = (
                True, datetime.now() + timedelta(days=5), 5
            )
            mock_ecm.return_value.renew_certificate_if_needed.return_value = (
                True, "Renewed"
            )
            main_func()

    def test_main_renew_if_needed_needs_and_fails(self):
        """Cover: --renew-if-needed where cert needs renewal and fails."""
        with patch.object(_mod, "argparse") as mock_ap, \
             patch.object(_mod, "load_ssl_config") as mock_lsc, \
             patch.object(_mod, "EnhancedCertificateManager") as mock_ecm:
            mock_lsc.return_value = _make_config()
            mock_ap.ArgumentParser.return_value.parse_args.return_value = (
                self._make_args(renew_if_needed=True)
            )
            mock_ecm.return_value.check_certificate_expiry.return_value = (
                True, datetime.now() + timedelta(days=5), 5
            )
            mock_ecm.return_value.renew_certificate_if_needed.return_value = (
                False, "Failed"
            )
            with pytest.raises(SystemExit):
                main_func()

    def test_main_monitor_quiet_ok(self):
        """Cover: --monitor --quiet with valid cert."""
        with patch.object(_mod, "argparse") as mock_ap, \
             patch.object(_mod, "load_ssl_config") as mock_lsc, \
             patch.object(_mod, "EnhancedCertificateManager") as mock_ecm, \
             patch.object(_mod, "print_certificate_report") as mock_print:
            mock_lsc.return_value = _make_config()
            mock_ap.ArgumentParser.return_value.parse_args.return_value = (
                self._make_args(monitor=True, quiet=True)
            )
            mock_ecm.return_value.monitor_certificate.return_value = {
                "certificate_exists": True,
                "needs_renewal": False,
                "days_until_expiry": 90,
            }
            main_func()
            # quiet=True means print_certificate_report should not be called
            mock_print.assert_not_called()

    def test_main_monitor_needs_renewal_days_gt_7(self):
        """Cover: --monitor with needs_renewal=True but days > 7 (exit code 1)."""
        with patch.object(_mod, "argparse") as mock_ap, \
             patch.object(_mod, "load_ssl_config") as mock_lsc, \
             patch.object(_mod, "EnhancedCertificateManager") as mock_ecm, \
             patch.object(_mod, "print_certificate_report"):
            mock_lsc.return_value = _make_config()
            mock_ap.ArgumentParser.return_value.parse_args.return_value = (
                self._make_args(monitor=True)
            )
            mock_ecm.return_value.monitor_certificate.return_value = {
                "certificate_exists": True,
                "needs_renewal": True,
                "days_until_expiry": 15,
            }
            with pytest.raises(SystemExit) as exc_info:
                main_func()
            assert exc_info.value.code == 1

    def test_main_init_success(self):
        """Cover: --init success path."""
        with patch.object(_mod, "argparse") as mock_ap, \
             patch.object(_mod, "load_ssl_config") as mock_lsc, \
             patch.object(_mod, "initialize_ssl_infrastructure") as mock_init, \
             patch.object(_mod, "EnhancedCertificateManager"):
            mock_lsc.return_value = _make_config()
            mock_ap.ArgumentParser.return_value.parse_args.return_value = (
                self._make_args(init=True)
            )
            mock_init.return_value = (True, _make_config())
            main_func()

    def test_main_init_failure(self):
        """Cover: --init failure path."""
        with patch.object(_mod, "argparse") as mock_ap, \
             patch.object(_mod, "load_ssl_config") as mock_lsc, \
             patch.object(_mod, "initialize_ssl_infrastructure") as mock_init, \
             patch.object(_mod, "EnhancedCertificateManager"):
            mock_lsc.return_value = _make_config()
            mock_ap.ArgumentParser.return_value.parse_args.return_value = (
                self._make_args(init=True)
            )
            mock_init.return_value = (False, None)
            with pytest.raises(SystemExit):
                main_func()

    def test_main_setup_cron_success(self):
        """Cover: --setup-cron success."""
        with patch.object(_mod, "argparse") as mock_ap, \
             patch.object(_mod, "load_ssl_config") as mock_lsc, \
             patch.object(_mod, "EnhancedCertificateManager") as mock_ecm:
            mock_lsc.return_value = _make_config()
            mock_ap.ArgumentParser.return_value.parse_args.return_value = (
                self._make_args(setup_cron=True)
            )
            mock_ecm.return_value.setup_monitoring_cron.return_value = True
            main_func()

    def test_main_remove_cron_success(self):
        """Cover: --remove-cron success."""
        with patch.object(_mod, "argparse") as mock_ap, \
             patch.object(_mod, "load_ssl_config") as mock_lsc, \
             patch.object(_mod, "EnhancedCertificateManager") as mock_ecm:
            mock_lsc.return_value = _make_config()
            mock_ap.ArgumentParser.return_value.parse_args.return_value = (
                self._make_args(remove_cron=True)
            )
            mock_ecm.return_value.remove_monitoring_cron.return_value = True
            main_func()

    def test_main_setup_notifications(self):
        """Cover: --setup-notifications."""
        with patch.object(_mod, "argparse") as mock_ap, \
             patch.object(_mod, "load_ssl_config") as mock_lsc, \
             patch.object(_mod, "EnhancedCertificateManager"), \
             patch.object(_mod, "setup_notification_config") as mock_snc:
            mock_lsc.return_value = _make_config()
            mock_ap.ArgumentParser.return_value.parse_args.return_value = (
                self._make_args(setup_notifications=True)
            )
            main_func()
            mock_snc.assert_called_once()

    def test_main_test_notifications(self):
        """Cover: --test-notifications."""
        with patch.object(_mod, "argparse") as mock_ap, \
             patch.object(_mod, "load_ssl_config") as mock_lsc, \
             patch.object(_mod, "EnhancedCertificateManager") as mock_ecm:
            mock_lsc.return_value = _make_config()
            mock_ap.ArgumentParser.return_value.parse_args.return_value = (
                self._make_args(test_notifications=True)
            )
            main_func()
            mock_ecm.return_value.send_notification.assert_called_once()

    def test_main_renew_force_success(self):
        """Cover: --renew (force) success."""
        with patch.object(_mod, "argparse") as mock_ap, \
             patch.object(_mod, "load_ssl_config") as mock_lsc, \
             patch.object(_mod, "EnhancedCertificateManager") as mock_ecm:
            mock_lsc.return_value = _make_config()
            mock_ap.ArgumentParser.return_value.parse_args.return_value = (
                self._make_args(renew=True)
            )
            mock_ecm.return_value.renew_certificate_if_needed.return_value = (
                True, "OK"
            )
            main_func()

    def test_main_renew_if_needed_not_needed_quiet(self):
        """Cover: --renew-if-needed --quiet, cert doesn't need renewal."""
        with patch.object(_mod, "argparse") as mock_ap, \
             patch.object(_mod, "load_ssl_config") as mock_lsc, \
             patch.object(_mod, "EnhancedCertificateManager") as mock_ecm:
            mock_lsc.return_value = _make_config()
            mock_ap.ArgumentParser.return_value.parse_args.return_value = (
                self._make_args(renew_if_needed=True, quiet=True)
            )
            mock_ecm.return_value.check_certificate_expiry.return_value = (
                False, datetime.now() + timedelta(days=90), 90
            )
            main_func()


@pytest.mark.coverage_extra
class TestSendNotificationEdgeCases:
    def test_send_notification_no_smtp_username(self):
        """Cover: smtp_username is empty, no login called."""
        mgr = _make_mgr()
        mgr.notification_config = {
            "enabled": True,
            "to_emails": ["admin@test.com"],
            "from_email": "mfc@localhost",
            "smtp_server": "localhost",
            "smtp_port": 25,
            "use_tls": False,
            "smtp_username": "",
        }
        with patch.object(_mod.smtplib, "SMTP") as mock_smtp:
            srv = MagicMock()
            mock_smtp.return_value = srv
            mgr.send_notification("Test", "Body")
            srv.login.assert_not_called()
            srv.send_message.assert_called_once()
            srv.quit.assert_called_once()
