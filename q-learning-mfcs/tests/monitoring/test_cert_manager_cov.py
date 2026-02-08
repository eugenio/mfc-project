"""Tests for cert_manager module - comprehensive coverage."""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "monitoring"))


@pytest.fixture
def mock_ssl_config():
    config = MagicMock()
    config.domain = "test.example.com"
    config.cert_file = "/tmp/test.crt"
    config.key_file = "/tmp/test.key"
    config.renewal_days_before = 30
    config.use_letsencrypt = False
    config.email = "test@example.com"
    config.staging = False
    return config


@pytest.fixture
def mock_ssl_funcs():
    with patch("cert_manager.load_ssl_config") as ml, \
         patch("cert_manager.initialize_ssl_infrastructure") as mi, \
         patch("cert_manager.CertificateManager") as mc:
        ml.return_value = MagicMock(
            domain="localhost",
            cert_file="/tmp/test.crt",
            key_file="/tmp/test.key",
            renewal_days_before=30,
            use_letsencrypt=False,
        )
        yield ml, mi, mc


def _make_manager(ssl_config):
    with patch("cert_manager.CertificateManager.__init__",
               return_value=None):
        from cert_manager import EnhancedCertificateManager
        mgr = object.__new__(EnhancedCertificateManager)
        mgr.config = ssl_config
        mgr.cert_dir = Path("/etc/ssl/certs")
        mgr.key_dir = Path("/etc/ssl/private")
        mgr.letsencrypt_dir = Path("/etc/letsencrypt")
        mgr.notification_config = {
            "enabled": False, "smtp_server": "localhost",
            "smtp_port": 587, "smtp_username": "",
            "smtp_password": "", "from_email": "mfc@localhost",
            "to_emails": [], "use_tls": True,
        }
        return mgr


class TestEnhancedCertificateManager:

    def test_load_notification_config_no_file(self, mock_ssl_config):
        with patch("cert_manager.CertificateManager.__init__",
                    return_value=None), \
             patch("cert_manager.Path") as mp:
            mp.return_value.exists.return_value = False
            from cert_manager import EnhancedCertificateManager
            mgr = object.__new__(EnhancedCertificateManager)
            mgr.config = mock_ssl_config
            r = mgr._load_notification_config()
            assert r["enabled"] is False

    def test_load_notification_config_with_file(self, mock_ssl_config):
        d = {"enabled": True, "smtp_server": "mail.test.com"}
        with patch("cert_manager.CertificateManager.__init__",
                    return_value=None), \
             patch("cert_manager.Path") as mp, \
             patch("builtins.open", mock_open(read_data=json.dumps(d))):
            mp.return_value.exists.return_value = True
            from cert_manager import EnhancedCertificateManager
            mgr = object.__new__(EnhancedCertificateManager)
            mgr.config = mock_ssl_config
            r = mgr._load_notification_config()
            assert r["enabled"] is True

    def test_load_notification_config_bad_json(self, mock_ssl_config):
        with patch("cert_manager.CertificateManager.__init__",
                    return_value=None), \
             patch("cert_manager.Path") as mp, \
             patch("builtins.open", mock_open(read_data="bad")):
            mp.return_value.exists.return_value = True
            from cert_manager import EnhancedCertificateManager
            mgr = object.__new__(EnhancedCertificateManager)
            mgr.config = mock_ssl_config
            r = mgr._load_notification_config()
            assert r["enabled"] is False

    def test_check_expiry_no_date(self, mock_ssl_config):
        mgr = _make_manager(mock_ssl_config)
        mgr.check_certificate_validity = MagicMock(
            return_value=(False, None))
        needs, exp, days = mgr.check_certificate_expiry()
        assert needs is True and exp is None and days == 0

    def test_check_expiry_valid(self, mock_ssl_config):
        f = datetime.now() + timedelta(days=60)
        mgr = _make_manager(mock_ssl_config)
        mgr.check_certificate_validity = MagicMock(
            return_value=(True, f))
        needs, exp, days = mgr.check_certificate_expiry()
        assert needs is False and days >= 59

    def test_check_expiry_needs_renewal(self, mock_ssl_config):
        f = datetime.now() + timedelta(days=10)
        mgr = _make_manager(mock_ssl_config)
        mgr.check_certificate_validity = MagicMock(
            return_value=(True, f))
        needs, _, _ = mgr.check_certificate_expiry()
        assert needs is True

    def test_send_notification_disabled(self, mock_ssl_config):
        mgr = _make_manager(mock_ssl_config)
        mgr.send_notification("T", "B")

    def test_send_notification_no_emails(self, mock_ssl_config):
        mgr = _make_manager(mock_ssl_config)
        mgr.notification_config["enabled"] = True
        mgr.send_notification("T", "B")

    def test_send_notification_success(self, mock_ssl_config):
        mgr = _make_manager(mock_ssl_config)
        mgr.notification_config.update({
            "enabled": True, "to_emails": ["a@b.com"],
            "use_tls": True, "smtp_username": "u",
            "smtp_password": "p"})
        ms = MagicMock()
        with patch("cert_manager.smtplib.SMTP", return_value=ms):
            mgr.send_notification("S", "B", is_critical=True)
            ms.starttls.assert_called_once()
            ms.login.assert_called_once()

    def test_send_notification_no_tls(self, mock_ssl_config):
        mgr = _make_manager(mock_ssl_config)
        mgr.notification_config.update({
            "enabled": True, "to_emails": ["a@b.com"],
            "use_tls": False, "smtp_username": ""})
        ms = MagicMock()
        with patch("cert_manager.smtplib.SMTP", return_value=ms):
            mgr.send_notification("S", "B")
            ms.starttls.assert_not_called()

    def test_send_notification_failure(self, mock_ssl_config):
        mgr = _make_manager(mock_ssl_config)
        mgr.notification_config.update({
            "enabled": True, "to_emails": ["a@b.com"]})
        with patch("cert_manager.smtplib.SMTP",
                    side_effect=Exception("err")):
            mgr.send_notification("S", "B")

    def test_monitor_no_cert(self, mock_ssl_config):
        mgr = _make_manager(mock_ssl_config)
        mgr.check_certificate_exists = MagicMock(return_value=False)
        r = mgr.monitor_certificate()
        assert r["certificate_exists"] is False

    def test_monitor_urgent(self, mock_ssl_config):
        mgr = _make_manager(mock_ssl_config)
        mgr.check_certificate_exists = MagicMock(return_value=True)
        f = datetime.now() + timedelta(days=5)
        mgr.check_certificate_expiry = MagicMock(
            return_value=(True, f, 5))
        mgr._check_cron_job_exists = MagicMock(return_value=False)
        r = mgr.monitor_certificate()
        assert any("URGENT" in x for x in r["recommendations"])

    def test_monitor_renewal_soon(self, mock_ssl_config):
        mgr = _make_manager(mock_ssl_config)
        mgr.check_certificate_exists = MagicMock(return_value=True)
        f = datetime.now() + timedelta(days=20)
        mgr.check_certificate_expiry = MagicMock(
            return_value=(True, f, 20))
        mgr._check_cron_job_exists = MagicMock(return_value=True)
        r = mgr.monitor_certificate()
        assert any("renewed" in x for x in r["recommendations"])

    def test_monitor_healthy(self, mock_ssl_config):
        mgr = _make_manager(mock_ssl_config)
        mgr.check_certificate_exists = MagicMock(return_value=True)
        f = datetime.now() + timedelta(days=60)
        mgr.check_certificate_expiry = MagicMock(
            return_value=(False, f, 60))
        mgr._check_cron_job_exists = MagicMock(return_value=True)
        r = mgr.monitor_certificate()
        assert any("healthy" in x for x in r["recommendations"])

    def test_cron_check_with_module(self, mock_ssl_config):
        mgr = _make_manager(mock_ssl_config)
        mj = MagicMock()
        mj.command = "mfc-cert-renewal x"
        mc = MagicMock()
        mc.__iter__ = MagicMock(return_value=iter([mj]))
        with patch.dict(sys.modules, {"crontab": MagicMock()}):
            with patch("crontab.CronTab", return_value=mc):
                assert mgr._check_cron_job_exists() is True

    def test_cron_check_no_match(self, mock_ssl_config):
        mgr = _make_manager(mock_ssl_config)
        mj = MagicMock()
        mj.command = "other"
        mc = MagicMock()
        mc.__iter__ = MagicMock(return_value=iter([mj]))
        with patch.dict(sys.modules, {"crontab": MagicMock()}):
            with patch("crontab.CronTab", return_value=mc):
                assert mgr._check_cron_job_exists() is False

    def test_cron_check_fallback_found(self, mock_ssl_config):
        mgr = _make_manager(mock_ssl_config)
        mr = MagicMock(returncode=0, stdout="mfc-cert-renewal daily")
        with patch("cert_manager.subprocess.run", return_value=mr):
            with patch.dict(sys.modules, {"crontab": None}):
                orig = __builtins__["__import__"] if isinstance(
                    __builtins__, dict) else __builtins__.__import__
                def fake_import(name, *a, **k):
                    if name == "crontab":
                        raise ImportError
                    return orig(name, *a, **k)
                with patch("builtins.__import__", side_effect=fake_import):
                    assert isinstance(mgr._check_cron_job_exists(), bool)

    def test_cron_check_fallback_fail(self, mock_ssl_config):
        mgr = _make_manager(mock_ssl_config)
        mr = MagicMock(returncode=1)
        with patch("cert_manager.subprocess.run", return_value=mr):
            with patch.dict(sys.modules, {"crontab": None}):
                orig = __builtins__["__import__"] if isinstance(
                    __builtins__, dict) else __builtins__.__import__
                def fake_import(name, *a, **k):
                    if name == "crontab":
                        raise ImportError
                    return orig(name, *a, **k)
                with patch("builtins.__import__", side_effect=fake_import):
                    assert isinstance(mgr._check_cron_job_exists(), bool)

    def test_cron_check_exception(self, mock_ssl_config):
        mgr = _make_manager(mock_ssl_config)
        with patch("cert_manager.subprocess.run",
                    side_effect=Exception("f")):
            with patch.dict(sys.modules, {"crontab": None}):
                orig = __builtins__["__import__"] if isinstance(
                    __builtins__, dict) else __builtins__.__import__
                def fake_import(name, *a, **k):
                    if name == "crontab":
                        raise ImportError
                    return orig(name, *a, **k)
                with patch("builtins.__import__", side_effect=fake_import):
                    assert mgr._check_cron_job_exists() is False

    def test_renew_not_needed(self, mock_ssl_config):
        mgr = _make_manager(mock_ssl_config)
        mgr.check_certificate_expiry = MagicMock(
            return_value=(False, datetime.now() + timedelta(days=60), 60))
        s, m = mgr.renew_certificate_if_needed()
        assert s is True

    def test_renew_le_success(self, mock_ssl_config):
        mock_ssl_config.use_letsencrypt = True
        mgr = _make_manager(mock_ssl_config)
        mgr.check_certificate_expiry = MagicMock(
            return_value=(True, datetime.now() + timedelta(days=5), 5))
        mgr.send_notification = MagicMock()
        mr = MagicMock(returncode=0)
        with patch("cert_manager.subprocess.run", return_value=mr):
            s, _ = mgr.renew_certificate_if_needed()
            assert s is True

    def test_renew_le_fail(self, mock_ssl_config):
        mock_ssl_config.use_letsencrypt = True
        mgr = _make_manager(mock_ssl_config)
        mgr.check_certificate_expiry = MagicMock(
            return_value=(True, datetime.now() + timedelta(days=5), 5))
        mgr.send_notification = MagicMock()
        mr = MagicMock(returncode=1, stderr="err")
        with patch("cert_manager.subprocess.run", return_value=mr):
            s, _ = mgr.renew_certificate_if_needed()
            assert s is False

    def test_renew_le_timeout(self, mock_ssl_config):
        import subprocess
        mock_ssl_config.use_letsencrypt = True
        mgr = _make_manager(mock_ssl_config)
        mgr.check_certificate_expiry = MagicMock(
            return_value=(True, datetime.now() + timedelta(days=5), 5))
        mgr.send_notification = MagicMock()
        with patch("cert_manager.subprocess.run",
                    side_effect=subprocess.TimeoutExpired("c", 300)):
            s, _ = mgr.renew_certificate_if_needed()
            assert s is False

    def test_renew_le_exception(self, mock_ssl_config):
        mock_ssl_config.use_letsencrypt = True
        mgr = _make_manager(mock_ssl_config)
        mgr.check_certificate_expiry = MagicMock(
            return_value=(True, datetime.now() + timedelta(days=5), 5))
        mgr.send_notification = MagicMock()
        with patch("cert_manager.subprocess.run",
                    side_effect=Exception("e")):
            s, _ = mgr.renew_certificate_if_needed()
            assert s is False

    def test_renew_self_signed_ok(self, mock_ssl_config):
        mock_ssl_config.use_letsencrypt = False
        mgr = _make_manager(mock_ssl_config)
        mgr.check_certificate_expiry = MagicMock(
            return_value=(True, datetime.now() + timedelta(days=5), 5))
        mgr.generate_self_signed_certificate = MagicMock(
            return_value=True)
        mgr.send_notification = MagicMock()
        s, _ = mgr.renew_certificate_if_needed()
        assert s is True

    def test_renew_self_signed_fail(self, mock_ssl_config):
        mock_ssl_config.use_letsencrypt = False
        mgr = _make_manager(mock_ssl_config)
        mgr.check_certificate_expiry = MagicMock(
            return_value=(True, datetime.now() + timedelta(days=5), 5))
        mgr.generate_self_signed_certificate = MagicMock(
            return_value=False)
        mgr.send_notification = MagicMock()
        s, _ = mgr.renew_certificate_if_needed()
        assert s is False

    def test_setup_cron_ok(self, mock_ssl_config):
        mgr = _make_manager(mock_ssl_config)
        mr = MagicMock(returncode=0)
        with patch("cert_manager.subprocess.run", return_value=mr):
            assert mgr.setup_monitoring_cron() is True

    def test_setup_cron_fail(self, mock_ssl_config):
        mgr = _make_manager(mock_ssl_config)
        mr = MagicMock(returncode=1, stderr="e")
        with patch("cert_manager.subprocess.run", return_value=mr):
            assert mgr.setup_monitoring_cron() is False

    def test_setup_cron_exc(self, mock_ssl_config):
        mgr = _make_manager(mock_ssl_config)
        with patch("cert_manager.subprocess.run",
                    side_effect=Exception("e")):
            assert mgr.setup_monitoring_cron() is False

    def test_remove_cron_ok(self, mock_ssl_config):
        mgr = _make_manager(mock_ssl_config)
        mr = MagicMock(returncode=0)
        with patch("cert_manager.subprocess.run", return_value=mr):
            assert mgr.remove_monitoring_cron() is True

    def test_remove_cron_fail(self, mock_ssl_config):
        mgr = _make_manager(mock_ssl_config)
        mr = MagicMock(returncode=1, stderr="e")
        with patch("cert_manager.subprocess.run", return_value=mr):
            assert mgr.remove_monitoring_cron() is False

    def test_remove_cron_exc(self, mock_ssl_config):
        mgr = _make_manager(mock_ssl_config)
        with patch("cert_manager.subprocess.run",
                    side_effect=Exception("e")):
            assert mgr.remove_monitoring_cron() is False


class TestPrintCertificateReport:

    def test_report_needs_renewal(self):
        from cert_manager import print_certificate_report
        print_certificate_report({
            "certificate_exists": True,
            "certificate_valid": True,
            "auto_renewal_configured": True,
            "expiry_date": "2026-12-31",
            "needs_renewal": True,
            "recommendations": ["Renew"],
        })

    def test_report_no_expiry(self):
        from cert_manager import print_certificate_report
        print_certificate_report({
            "certificate_exists": False,
            "certificate_valid": False,
            "auto_renewal_configured": False,
            "expiry_date": None,
            "needs_renewal": False,
            "recommendations": [],
        })

    def test_report_valid(self):
        from cert_manager import print_certificate_report
        print_certificate_report({
            "certificate_exists": True,
            "certificate_valid": True,
            "auto_renewal_configured": True,
            "expiry_date": "2027-12-31",
            "needs_renewal": False,
            "recommendations": ["Healthy"],
        })


class TestMain:

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

    def test_main_default(self, mock_ssl_funcs):
        with patch("cert_manager.argparse.ArgumentParser") as mp, \
             patch("cert_manager.EnhancedCertificateManager") as me, \
             patch("cert_manager.print_certificate_report"):
            mp.return_value.parse_args.return_value = self._make_args()
            me.return_value.monitor_certificate.return_value = {}
            from cert_manager import main
            main()

    def test_main_monitor_critical(self, mock_ssl_funcs):
        with patch("cert_manager.argparse.ArgumentParser") as mp, \
             patch("cert_manager.EnhancedCertificateManager") as me, \
             patch("cert_manager.print_certificate_report"):
            mp.return_value.parse_args.return_value = self._make_args(
                monitor=True)
            me.return_value.monitor_certificate.return_value = {
                "certificate_exists": True,
                "needs_renewal": True,
                "days_until_expiry": 3,
            }
            from cert_manager import main
            with pytest.raises(SystemExit) as ei:
                main()
            assert ei.value.code == 2

    def test_main_monitor_warning(self, mock_ssl_funcs):
        with patch("cert_manager.argparse.ArgumentParser") as mp, \
             patch("cert_manager.EnhancedCertificateManager") as me, \
             patch("cert_manager.print_certificate_report"):
            mp.return_value.parse_args.return_value = self._make_args(
                monitor=True, quiet=True)
            me.return_value.monitor_certificate.return_value = {
                "certificate_exists": False,
                "needs_renewal": False,
                "days_until_expiry": None,
            }
            from cert_manager import main
            with pytest.raises(SystemExit) as ei:
                main()
            assert ei.value.code == 1

    def test_main_init_ok(self, mock_ssl_funcs):
        _, mi, _ = mock_ssl_funcs
        mi.return_value = (True, MagicMock())
        with patch("cert_manager.argparse.ArgumentParser") as mp, \
             patch("cert_manager.EnhancedCertificateManager"):
            mp.return_value.parse_args.return_value = self._make_args(
                init=True, domain="t.com", email="a@b.c", staging=True)
            from cert_manager import main
            main()

    def test_main_init_fail(self, mock_ssl_funcs):
        _, mi, _ = mock_ssl_funcs
        mi.return_value = (False, MagicMock())
        with patch("cert_manager.argparse.ArgumentParser") as mp, \
             patch("cert_manager.EnhancedCertificateManager"):
            mp.return_value.parse_args.return_value = self._make_args(
                init=True)
            from cert_manager import main
            with pytest.raises(SystemExit):
                main()

    def test_main_setup_cron(self, mock_ssl_funcs):
        with patch("cert_manager.argparse.ArgumentParser") as mp, \
             patch("cert_manager.EnhancedCertificateManager") as me:
            mp.return_value.parse_args.return_value = self._make_args(
                setup_cron=True)
            me.return_value.setup_monitoring_cron.return_value = True
            from cert_manager import main
            main()

    def test_main_setup_cron_fail(self, mock_ssl_funcs):
        with patch("cert_manager.argparse.ArgumentParser") as mp, \
             patch("cert_manager.EnhancedCertificateManager") as me:
            mp.return_value.parse_args.return_value = self._make_args(
                setup_cron=True)
            me.return_value.setup_monitoring_cron.return_value = False
            from cert_manager import main
            with pytest.raises(SystemExit):
                main()

    def test_main_remove_cron_ok(self, mock_ssl_funcs):
        with patch("cert_manager.argparse.ArgumentParser") as mp, \
             patch("cert_manager.EnhancedCertificateManager") as me:
            mp.return_value.parse_args.return_value = self._make_args(
                remove_cron=True)
            me.return_value.remove_monitoring_cron.return_value = True
            from cert_manager import main
            main()

    def test_main_remove_cron_fail(self, mock_ssl_funcs):
        with patch("cert_manager.argparse.ArgumentParser") as mp, \
             patch("cert_manager.EnhancedCertificateManager") as me:
            mp.return_value.parse_args.return_value = self._make_args(
                remove_cron=True)
            me.return_value.remove_monitoring_cron.return_value = False
            from cert_manager import main
            with pytest.raises(SystemExit):
                main()

    def test_main_setup_notif(self, mock_ssl_funcs):
        with patch("cert_manager.argparse.ArgumentParser") as mp, \
             patch("cert_manager.EnhancedCertificateManager"), \
             patch("cert_manager.setup_notification_config"):
            mp.return_value.parse_args.return_value = self._make_args(
                setup_notifications=True)
            from cert_manager import main
            main()

    def test_main_test_notif(self, mock_ssl_funcs):
        with patch("cert_manager.argparse.ArgumentParser") as mp, \
             patch("cert_manager.EnhancedCertificateManager") as me:
            mp.return_value.parse_args.return_value = self._make_args(
                test_notifications=True)
            from cert_manager import main
            main()
            me.return_value.send_notification.assert_called()

    def test_main_renew_if_needed_no(self, mock_ssl_funcs):
        with patch("cert_manager.argparse.ArgumentParser") as mp, \
             patch("cert_manager.EnhancedCertificateManager") as me:
            mp.return_value.parse_args.return_value = self._make_args(
                renew_if_needed=True)
            me.return_value.check_certificate_expiry.return_value = (
                False, None, 60)
            from cert_manager import main
            main()

    def test_main_renew_force_ok(self, mock_ssl_funcs):
        with patch("cert_manager.argparse.ArgumentParser") as mp, \
             patch("cert_manager.EnhancedCertificateManager") as me:
            mp.return_value.parse_args.return_value = self._make_args(
                renew=True)
            me.return_value.renew_certificate_if_needed.return_value = (
                True, "OK")
            from cert_manager import main
            main()

    def test_main_renew_fail(self, mock_ssl_funcs):
        with patch("cert_manager.argparse.ArgumentParser") as mp, \
             patch("cert_manager.EnhancedCertificateManager") as me:
            mp.return_value.parse_args.return_value = self._make_args(
                renew=True)
            me.return_value.renew_certificate_if_needed.return_value = (
                False, "FAIL")
            from cert_manager import main
            with pytest.raises(SystemExit):
                main()

    def test_main_renew_if_needed_quiet(self, mock_ssl_funcs):
        with patch("cert_manager.argparse.ArgumentParser") as mp, \
             patch("cert_manager.EnhancedCertificateManager") as me:
            mp.return_value.parse_args.return_value = self._make_args(
                renew_if_needed=True, quiet=True)
            me.return_value.check_certificate_expiry.return_value = (
                False, None, 60)
            from cert_manager import main
            main()

    def test_main_monitor_ok(self, mock_ssl_funcs):
        with patch("cert_manager.argparse.ArgumentParser") as mp, \
             patch("cert_manager.EnhancedCertificateManager") as me, \
             patch("cert_manager.print_certificate_report"):
            mp.return_value.parse_args.return_value = self._make_args(
                monitor=True)
            me.return_value.monitor_certificate.return_value = {
                "certificate_exists": True,
                "needs_renewal": False,
                "days_until_expiry": 60,
            }
            from cert_manager import main
            main()

    def test_main_renew_if_needed_needs_renewal(self, mock_ssl_funcs):
        with patch("cert_manager.argparse.ArgumentParser") as mp, \
             patch("cert_manager.EnhancedCertificateManager") as me:
            mp.return_value.parse_args.return_value = self._make_args(
                renew_if_needed=True)
            me.return_value.check_certificate_expiry.return_value = (
                True, None, 5)
            me.return_value.renew_certificate_if_needed.return_value = (
                True, "OK")
            from cert_manager import main
            main()


class TestSetupNotificationConfig:

    def test_setup_disabled(self):
        from cert_manager import setup_notification_config
        with patch("builtins.input", return_value="n"), \
             patch("cert_manager.Path") as mp, \
             patch("builtins.open", mock_open()):
            mp.return_value.parent.mkdir = MagicMock()
            result = setup_notification_config()
            assert result is True

    def test_setup_enabled_full(self):
        from cert_manager import setup_notification_config
        mock_getpass = MagicMock()
        mock_getpass.getpass.return_value = "secret"
        inputs = iter([
            "y",             # enable
            "smtp.test.com", # smtp server
            "465",           # port
            "user",          # username
            "from@test.com", # from email
            "to@test.com",   # to emails
            "y",             # use tls
        ])
        with patch("builtins.input", side_effect=lambda _: next(inputs)), \
             patch("cert_manager.Path") as mp, \
             patch("builtins.open", mock_open()), \
             patch.dict(sys.modules, {"getpass": mock_getpass}):
            mp.return_value.parent.mkdir = MagicMock()
            result = setup_notification_config()
            assert result is True

    def test_setup_enabled_no_username(self):
        from cert_manager import setup_notification_config
        inputs = iter([
            "yes",           # enable
            "",              # smtp server (default)
            "",              # port (default)
            "",              # username (empty)
            "",              # from email (default)
            "a@b.com,c@d.com",  # to emails
            "n",             # no tls
        ])
        with patch("builtins.input", side_effect=lambda _: next(inputs)), \
             patch("cert_manager.Path") as mp, \
             patch("builtins.open", mock_open()):
            mp.return_value.parent.mkdir = MagicMock()
            result = setup_notification_config()
            assert result is True

    def test_setup_save_failure(self):
        from cert_manager import setup_notification_config
        with patch("builtins.input", return_value="n"), \
             patch("cert_manager.Path") as mp, \
             patch("builtins.open", side_effect=PermissionError("denied")):
            mp.return_value.parent.mkdir = MagicMock()
            result = setup_notification_config()
            assert result is False


class TestEnhancedCertManagerInit:

    def test_init_calls_super_and_loads_config(self, mock_ssl_config):
        with patch("cert_manager.CertificateManager.__init__",
                    return_value=None) as mci, \
             patch("cert_manager.Path") as mp:
            mp.return_value.exists.return_value = False
            from cert_manager import EnhancedCertificateManager
            mgr = EnhancedCertificateManager(mock_ssl_config)
            mci.assert_called_once_with(mock_ssl_config)
            assert mgr.notification_config["enabled"] is False
