"""Tests for notifications/base.py - coverage target 98%+."""
import sys
import os
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from notifications.base import (
    FallbackHandler,
    NotificationConfig,
    NotificationHandler,
    NotificationLevel,
)


class TestNotificationLevel:
    def test_all_levels(self):
        assert NotificationLevel.INFO.value == "info"
        assert NotificationLevel.WARNING.value == "warning"
        assert NotificationLevel.CRITICAL.value == "critical"
        assert NotificationLevel.SUCCESS.value == "success"


class TestNotificationConfig:
    def test_default_config(self):
        cfg = NotificationConfig(title="T", message="M")
        assert cfg.title == "T"
        assert cfg.message == "M"
        assert cfg.level == NotificationLevel.INFO
        assert cfg.timeout_ms == 5000
        assert cfg.icon_path is None
        assert cfg.sound_enabled is True
        assert cfg.sound_path is None
        assert cfg.actions is None
        assert cfg.app_name == "MFC Monitor"
        assert cfg.category == "mfc.alert"

    def test_custom_config(self):
        cfg = NotificationConfig(
            title="Alert",
            message="Msg",
            level=NotificationLevel.CRITICAL,
            timeout_ms=None,
            icon_path=Path("/tmp/icon.png"),
            sound_enabled=False,
            sound_path=Path("/tmp/sound.wav"),
            actions={"ok": "OK"},
            app_name="Test",
            category="test.cat",
        )
        assert cfg.timeout_ms is None
        assert cfg.actions == {"ok": "OK"}


class ConcreteHandler(NotificationHandler):
    """Concrete handler for testing abstract base."""

    def is_available(self):
        return True

    def send_notification(self, config):
        return True

    def get_capabilities(self):
        return {"actions": False}

    def _play_custom_sound(self, sound_path):
        return True

    def _play_system_sound(self, level):
        return True


class TestNotificationHandler:
    def test_init(self):
        h = ConcreteHandler("MyApp")
        assert h.app_name == "MyApp"
        assert h._is_available is None

    def test_default_app_name(self):
        h = ConcreteHandler()
        assert h.app_name == "MFC Monitor"

    def test_validate_config_valid(self):
        h = ConcreteHandler()
        cfg = NotificationConfig(title="T", message="M")
        assert h.validate_config(cfg) is True

    def test_validate_config_no_title(self):
        h = ConcreteHandler()
        cfg = NotificationConfig(title="", message="M")
        assert h.validate_config(cfg) is False

    def test_validate_config_no_message(self):
        h = ConcreteHandler()
        cfg = NotificationConfig(title="T", message="")
        assert h.validate_config(cfg) is False

    def test_play_sound_custom_exists(self, tmp_path):
        h = ConcreteHandler()
        sound_file = tmp_path / "test.wav"
        sound_file.write_text("audio data")
        result = h.play_sound(sound_file, NotificationLevel.INFO)
        assert result is True

    def test_play_sound_custom_not_exists(self):
        h = ConcreteHandler()
        result = h.play_sound(Path("/nonexistent/sound.wav"), NotificationLevel.INFO)
        assert result is True  # falls back to system

    def test_play_sound_none(self):
        h = ConcreteHandler()
        result = h.play_sound(None, NotificationLevel.WARNING)
        assert result is True

    def test_play_sound_exception(self):
        h = ConcreteHandler()
        h._play_system_sound = MagicMock(side_effect=Exception("fail"))
        result = h.play_sound(None, NotificationLevel.INFO)
        assert result is False

    def test_log_fallback_all_levels(self):
        h = ConcreteHandler()
        for level in NotificationLevel:
            cfg = NotificationConfig(title="T", message="M", level=level)
            h.log_fallback(cfg)  # should not raise

    def test_log_fallback_unknown_level(self):
        h = ConcreteHandler()
        cfg = NotificationConfig(title="T", message="M")
        cfg.level = MagicMock(value="unknown")
        h.log_fallback(cfg)


class TestFallbackHandler:
    def test_is_available(self):
        h = FallbackHandler()
        assert h.is_available() is True

    def test_send_notification_success(self, capsys):
        h = FallbackHandler()
        cfg = NotificationConfig(
            title="Test", message="Hello", sound_enabled=False
        )
        result = h.send_notification(cfg)
        assert result is True
        captured = capsys.readouterr()
        assert "Test" in captured.out

    def test_send_notification_invalid(self):
        h = FallbackHandler()
        cfg = NotificationConfig(title="", message="")
        result = h.send_notification(cfg)
        assert result is False

    def test_send_notification_with_timeout(self, capsys):
        h = FallbackHandler()
        cfg = NotificationConfig(
            title="T", message="M", timeout_ms=3000, sound_enabled=False
        )
        result = h.send_notification(cfg)
        assert result is True
        captured = capsys.readouterr()
        assert "3.0s" in captured.out

    def test_send_notification_no_timeout(self, capsys):
        h = FallbackHandler()
        cfg = NotificationConfig(
            title="T", message="M", timeout_ms=None, sound_enabled=False
        )
        result = h.send_notification(cfg)
        assert result is True
        captured = capsys.readouterr()
        assert "Auto-dismiss" not in captured.out

    def test_send_notification_with_sound(self):
        h = FallbackHandler()
        h.play_sound = MagicMock(return_value=True)
        cfg = NotificationConfig(
            title="T", message="M", sound_enabled=True
        )
        result = h.send_notification(cfg)
        assert result is True
        h.play_sound.assert_called_once()

    def test_send_notification_all_levels(self, capsys):
        h = FallbackHandler()
        for level in NotificationLevel:
            cfg = NotificationConfig(
                title="T", message="M", level=level, sound_enabled=False
            )
            assert h.send_notification(cfg) is True

    def test_get_capabilities(self):
        h = FallbackHandler()
        caps = h.get_capabilities()
        assert caps["actions"] is False
        assert caps["sound"] is True
        assert caps["icons"] is False

    @patch("platform.system", return_value="Linux")
    @patch("subprocess.run")
    def test_play_custom_sound_linux(self, mock_run, mock_sys):
        h = FallbackHandler()
        result = h._play_custom_sound(Path("/tmp/sound.wav"))
        assert result is True

    @patch("platform.system", return_value="Darwin")
    @patch("subprocess.run")
    def test_play_custom_sound_macos(self, mock_run, mock_sys):
        h = FallbackHandler()
        result = h._play_custom_sound(Path("/tmp/sound.wav"))
        assert result is True

    @patch("platform.system", return_value="Windows")
    def test_play_custom_sound_windows(self, mock_sys):
        mock_winsound = MagicMock()
        mock_winsound.SND_FILENAME = 1
        mock_winsound.SND_NOWAIT = 2
        with patch.dict("sys.modules", {"winsound": mock_winsound}):
            h = FallbackHandler()
            result = h._play_custom_sound(Path("/tmp/sound.wav"))
            assert result is True

    @patch("platform.system", return_value="FreeBSD")
    def test_play_custom_sound_unsupported(self, mock_sys):
        h = FallbackHandler()
        result = h._play_custom_sound(Path("/tmp/sound.wav"))
        assert result is False

    @patch("platform.system", return_value="Linux")
    @patch("subprocess.run", side_effect=Exception("fail"))
    def test_play_custom_sound_error(self, mock_run, mock_sys):
        h = FallbackHandler()
        result = h._play_custom_sound(Path("/tmp/sound.wav"))
        assert result is False

    @patch("platform.system", return_value="Linux")
    @patch("subprocess.run")
    def test_play_system_sound_linux(self, mock_run, mock_sys):
        with patch("pathlib.Path.exists", return_value=True):
            h = FallbackHandler()
            result = h._play_system_sound(NotificationLevel.INFO)
            assert result is True

    @patch("platform.system", return_value="Linux")
    def test_play_system_sound_linux_no_files(self, mock_sys):
        with patch("pathlib.Path.exists", return_value=False):
            h = FallbackHandler()
            result = h._play_system_sound(NotificationLevel.INFO)
            assert result is False

    @patch("platform.system", return_value="Darwin")
    @patch("subprocess.run")
    def test_play_system_sound_macos(self, mock_run, mock_sys):
        h = FallbackHandler()
        result = h._play_system_sound(NotificationLevel.WARNING)
        assert result is True

    @patch("platform.system", return_value="Windows")
    def test_play_system_sound_windows(self, mock_sys):
        mock_winsound = MagicMock()
        mock_winsound.MB_OK = 0
        mock_winsound.MB_ICONEXCLAMATION = 1
        mock_winsound.MB_ICONHAND = 2
        with patch.dict("sys.modules", {"winsound": mock_winsound}):
            h = FallbackHandler()
            result = h._play_system_sound(NotificationLevel.CRITICAL)
            assert result is True

    @patch("platform.system", return_value="Darwin")
    @patch("subprocess.run", side_effect=Exception("err"))
    def test_play_system_sound_error(self, mock_run, mock_sys):
        h = FallbackHandler()
        result = h._play_system_sound(NotificationLevel.INFO)
        assert result is False
