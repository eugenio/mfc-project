"""Tests for notifications/linux_handler.py - coverage target 98%+."""
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from notifications.linux_handler import LinuxNotificationHandler
from notifications.base import NotificationConfig, NotificationLevel


class TestLinuxHandlerInit:
    def test_init_default(self):
        h = LinuxNotificationHandler()
        assert h.app_name == "MFC Monitor"

    def test_init_custom(self):
        h = LinuxNotificationHandler("App")
        assert h.app_name == "App"


class TestIsAvailable:
    @patch("platform.system", return_value="Windows")
    def test_not_linux(self, m):
        h = LinuxNotificationHandler()
        assert h.is_available() is False

    def test_cached(self):
        h = LinuxNotificationHandler()
        h._is_available = True
        assert h.is_available() is True

    @patch("platform.system", return_value="Linux")
    def test_no_display(self, m):
        h = LinuxNotificationHandler()
        env = {k: v for k, v in os.environ.items()
               if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
        with patch.dict(os.environ, env, clear=True):
            h._is_available = None
            assert h.is_available() is False

    @patch("platform.system", return_value="Linux")
    def test_with_display(self, m):
        h = LinuxNotificationHandler()
        h._is_available = None
        with patch.dict(os.environ, {"DISPLAY": ":0"}):
            with patch.object(h, "_check_notify_send", return_value=True):
                with patch.object(h, "_check_dbus", return_value=False):
                    with patch.object(h, "_check_plyer", return_value=False):
                        assert h.is_available() is True


class TestCheckMethods:
    @patch("shutil.which", return_value="/usr/bin/notify-send")
    def test_notify_send_ok(self, m):
        h = LinuxNotificationHandler()
        assert h._check_notify_send() is True

    @patch("shutil.which", return_value=None)
    def test_notify_send_no(self, m):
        h = LinuxNotificationHandler()
        assert h._check_notify_send() is False

    def test_notify_send_cached(self):
        h = LinuxNotificationHandler()
        h._notify_send_available = True
        assert h._check_notify_send() is True

    def test_dbus_ok(self):
        h = LinuxNotificationHandler()
        md = MagicMock()
        md.SessionBus.return_value.get_object.return_value = MagicMock()
        md.Interface.return_value.GetCapabilities.return_value = []
        with patch.dict("sys.modules", {"dbus": md}):
            h._dbus_available = None
            assert h._check_dbus() is True

    def test_dbus_fail(self):
        h = LinuxNotificationHandler()
        h._dbus_available = None
        with patch("builtins.__import__", side_effect=ImportError):
            assert h._check_dbus() is False

    def test_dbus_cached(self):
        h = LinuxNotificationHandler()
        h._dbus_available = False
        assert h._check_dbus() is False

    def test_plyer_ok(self):
        h = LinuxNotificationHandler()
        mp = MagicMock()
        mp.notification = MagicMock()
        with patch.dict("sys.modules", {"plyer": mp}):
            assert h._check_plyer() is True

    def test_plyer_no(self):
        h = LinuxNotificationHandler()
        with patch("builtins.__import__", side_effect=ImportError):
            assert h._check_plyer() is False

    @patch("shutil.which", return_value="/usr/bin/paplay")
    def test_pulse_ok(self, m):
        h = LinuxNotificationHandler()
        assert h._check_pulseaudio() is True

    @patch("shutil.which", return_value=None)
    def test_pulse_no(self, m):
        h = LinuxNotificationHandler()
        assert h._check_pulseaudio() is False


class TestSendNotification:
    def test_invalid(self):
        h = LinuxNotificationHandler()
        cfg = NotificationConfig(title="", message="")
        assert h.send_notification(cfg) is False

    def test_via_dbus(self):
        h = LinuxNotificationHandler()
        with patch.object(h, "_check_dbus", return_value=True):
            with patch.object(h, "_send_dbus_notification", return_value=True):
                cfg = NotificationConfig(title="T", message="M", sound_enabled=False)
                assert h.send_notification(cfg) is True

    def test_via_notify_send(self):
        h = LinuxNotificationHandler()
        with patch.object(h, "_check_dbus", return_value=False):
            with patch.object(h, "_check_notify_send", return_value=True):
                with patch.object(h, "_send_notify_send", return_value=True):
                    cfg = NotificationConfig(title="T", message="M", sound_enabled=False)
                    assert h.send_notification(cfg) is True

    def test_via_plyer(self):
        h = LinuxNotificationHandler()
        with patch.object(h, "_check_dbus", return_value=False):
            with patch.object(h, "_check_notify_send", return_value=False):
                with patch.object(h, "_check_plyer", return_value=True):
                    with patch.object(h, "_send_plyer_notification", return_value=True):
                        cfg = NotificationConfig(title="T", message="M", sound_enabled=False)
                        assert h.send_notification(cfg) is True

    def test_fallback(self):
        h = LinuxNotificationHandler()
        with patch.object(h, "_check_dbus", return_value=False):
            with patch.object(h, "_check_notify_send", return_value=False):
                with patch.object(h, "_check_plyer", return_value=False):
                    with patch.object(h, "log_fallback"):
                        cfg = NotificationConfig(title="T", message="M", sound_enabled=False)
                        assert h.send_notification(cfg) is True

    def test_with_sound(self):
        h = LinuxNotificationHandler()
        with patch.object(h, "_check_dbus", return_value=True):
            with patch.object(h, "_send_dbus_notification", return_value=True):
                with patch.object(h, "play_sound"):
                    cfg = NotificationConfig(title="T", message="M", sound_enabled=True)
                    assert h.send_notification(cfg) is True


class TestDbusNotification:
    def test_basic(self):
        h = LinuxNotificationHandler()
        md = MagicMock()
        md.SessionBus.return_value.get_object.return_value = MagicMock()
        mi = MagicMock()
        mi.Notify.return_value = 1
        md.Interface.return_value = mi
        md.Byte = lambda x: x
        md.String = lambda x: x
        md.Boolean = lambda x: x
        with patch.dict("sys.modules", {"dbus": md}):
            cfg = NotificationConfig(title="T", message="M", sound_enabled=False)
            assert h._send_dbus_notification(cfg) is True

    def test_with_icon_actions(self):
        h = LinuxNotificationHandler()
        md = MagicMock()
        md.SessionBus.return_value.get_object.return_value = MagicMock()
        mi = MagicMock()
        mi.Notify.return_value = 1
        md.Interface.return_value = mi
        md.Byte = lambda x: x
        md.String = lambda x: x
        md.Boolean = lambda x: x
        with patch.dict("sys.modules", {"dbus": md}):
            with patch("pathlib.Path.exists", return_value=True):
                cfg = NotificationConfig(
                    title="T", message="M",
                    icon_path=Path("/tmp/i.png"),
                    actions={"ok": "OK"},
                    timeout_ms=3000,
                    sound_enabled=True,
                )
                assert h._send_dbus_notification(cfg) is True

    def test_error(self):
        h = LinuxNotificationHandler()
        with patch("builtins.__import__", side_effect=Exception):
            cfg = NotificationConfig(title="T", message="M")
            assert h._send_dbus_notification(cfg) is False


class TestNotifySend:
    @patch("subprocess.run", return_value=MagicMock(returncode=0))
    def test_success(self, m):
        h = LinuxNotificationHandler()
        cfg = NotificationConfig(title="T", message="M", timeout_ms=5000)
        assert h._send_notify_send(cfg) is True

    @patch("subprocess.run", return_value=MagicMock(returncode=0))
    def test_with_icon(self, m):
        h = LinuxNotificationHandler()
        with patch("pathlib.Path.exists", return_value=True):
            cfg = NotificationConfig(
                title="T", message="M",
                icon_path=Path("/tmp/i.png"),
                level=NotificationLevel.CRITICAL,
            )
            assert h._send_notify_send(cfg) is True

    @patch("subprocess.run", return_value=MagicMock(returncode=1, stderr="err"))
    def test_fail(self, m):
        h = LinuxNotificationHandler()
        cfg = NotificationConfig(title="T", message="M")
        assert h._send_notify_send(cfg) is False

    @patch("subprocess.run", side_effect=Exception)
    def test_error(self, m):
        h = LinuxNotificationHandler()
        cfg = NotificationConfig(title="T", message="M")
        assert h._send_notify_send(cfg) is False


class TestPlyerNotification:
    def test_success(self):
        h = LinuxNotificationHandler()
        mp = MagicMock()
        with patch.dict("sys.modules", {"plyer": mp}):
            cfg = NotificationConfig(title="T", message="M", timeout_ms=5000)
            assert h._send_plyer_notification(cfg) is True

    def test_no_timeout(self):
        h = LinuxNotificationHandler()
        mp = MagicMock()
        with patch.dict("sys.modules", {"plyer": mp}):
            cfg = NotificationConfig(title="T", message="M", timeout_ms=None)
            assert h._send_plyer_notification(cfg) is True

    def test_error(self):
        h = LinuxNotificationHandler()
        with patch("builtins.__import__", side_effect=Exception):
            cfg = NotificationConfig(title="T", message="M")
            assert h._send_plyer_notification(cfg) is False


class TestGetCapabilities:
    def test_basic(self):
        h = LinuxNotificationHandler()
        with patch.object(h, "_check_pulseaudio", return_value=True):
            with patch.object(h, "_check_dbus", return_value=False):
                caps = h.get_capabilities()
                assert caps["sound"] is True
                assert caps["icons"] is True

    def test_with_dbus(self):
        h = LinuxNotificationHandler()
        md = MagicMock()
        md.SessionBus.return_value.get_object.return_value = MagicMock()
        mi = MagicMock()
        mi.GetCapabilities.return_value = ["actions", "body-markup"]
        md.Interface.return_value = mi
        with patch.dict("sys.modules", {"dbus": md}):
            with patch.object(h, "_check_pulseaudio", return_value=False):
                with patch.object(h, "_check_dbus", return_value=True):
                    caps = h.get_capabilities()
                    assert caps["actions"] is True


class TestPlaySounds:
    @patch("shutil.which", return_value="/usr/bin/paplay")
    @patch("subprocess.run")
    def test_custom_ok(self, mr, mw):
        h = LinuxNotificationHandler()
        h._pulseaudio_available = None
        assert h._play_custom_sound(Path("/tmp/s.wav")) is True

    @patch("shutil.which", return_value=None)
    def test_custom_no_pulse(self, m):
        h = LinuxNotificationHandler()
        h._pulseaudio_available = None
        assert h._play_custom_sound(Path("/tmp/s.wav")) is False

    @patch("shutil.which", return_value="/usr/bin/paplay")
    @patch("subprocess.run", side_effect=Exception)
    def test_custom_error(self, mr, mw):
        h = LinuxNotificationHandler()
        h._pulseaudio_available = None
        assert h._play_custom_sound(Path("/tmp/s.wav")) is False

    @patch("shutil.which", return_value="/usr/bin/paplay")
    @patch("subprocess.run")
    def test_system_ok(self, mr, mw):
        h = LinuxNotificationHandler()
        h._pulseaudio_available = None
        assert h._play_system_sound(NotificationLevel.INFO) is True

    @patch("shutil.which", return_value=None)
    def test_system_no_pulse(self, m):
        h = LinuxNotificationHandler()
        h._pulseaudio_available = None
        assert h._play_system_sound(NotificationLevel.INFO) is False

    @patch("shutil.which", return_value="/usr/bin/paplay")
    @patch("subprocess.run")
    def test_system_alsa_fallback(self, mr, mw):
        import subprocess as sp
        mr.side_effect = [sp.CalledProcessError(1, "paplay"), MagicMock()]
        h = LinuxNotificationHandler()
        h._pulseaudio_available = None
        with patch("pathlib.Path.exists", return_value=True):
            assert h._play_system_sound(NotificationLevel.INFO) is True

    @patch("shutil.which", return_value="/usr/bin/paplay")
    @patch("subprocess.run")
    def test_system_alsa_no_file(self, mr, mw):
        import subprocess as sp
        mr.side_effect = sp.CalledProcessError(1, "paplay")
        h = LinuxNotificationHandler()
        h._pulseaudio_available = None
        with patch("pathlib.Path.exists", return_value=False):
            assert h._play_system_sound(NotificationLevel.INFO) is False

    @patch("shutil.which", return_value="/usr/bin/paplay")
    @patch("subprocess.run", side_effect=Exception)
    def test_system_error(self, mr, mw):
        h = LinuxNotificationHandler()
        h._pulseaudio_available = None
        assert h._play_system_sound(NotificationLevel.INFO) is False
