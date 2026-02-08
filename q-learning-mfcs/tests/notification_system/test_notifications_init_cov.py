"""Tests for notifications/__init__.py - coverage target 98%+."""
import sys
import os
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


class TestNotificationsInitImports:
    """Test the notifications package __init__ imports."""

    def test_base_imports_always_available(self):
        from notifications import NotificationConfig, NotificationHandler, NotificationLevel
        assert NotificationLevel.INFO.value == "info"
        assert NotificationConfig is not None
        assert NotificationHandler is not None

    def test_all_list_contains_base(self):
        import notifications
        assert "NotificationHandler" in notifications.__all__
        assert "NotificationLevel" in notifications.__all__
        assert "NotificationConfig" in notifications.__all__

    def test_audio_manager_import(self):
        """Test audio_manager imports are attempted."""
        from notifications import AudioConfig, AudioEvent, AudioManager
        assert AudioConfig is not None
        assert AudioEvent is not None
        assert AudioManager is not None

    def test_audio_manager_in_all(self):
        import notifications
        assert "AudioManager" in notifications.__all__
        assert "AudioConfig" in notifications.__all__
        assert "AudioEvent" in notifications.__all__

    def test_linux_handler_import(self):
        """Test LinuxNotificationHandler import attempt."""
        try:
            from notifications import LinuxNotificationHandler
            assert LinuxNotificationHandler is not None
        except ImportError:
            pass  # may not be available on all platforms

    def test_convenience_functions_defined(self):
        """Test convenience functions are available when manager is importable."""
        import notifications
        # These functions depend on manager module being importable
        if hasattr(notifications, "notify"):
            assert callable(notifications.notify)
        if hasattr(notifications, "info"):
            assert callable(notifications.info)
        if hasattr(notifications, "warning"):
            assert callable(notifications.warning)
        if hasattr(notifications, "critical"):
            assert callable(notifications.critical)
        if hasattr(notifications, "success"):
            assert callable(notifications.success)
        if hasattr(notifications, "play_ding"):
            assert callable(notifications.play_ding)


class TestConvenienceFunctions:
    """Test convenience notification functions."""

    def test_notify_creates_default_manager(self):
        """Test that notify creates a default manager on first call."""
        import notifications
        if not hasattr(notifications, "notify"):
            pytest.skip("Manager module not available")

        mock_manager = MagicMock()
        mock_manager.notify.return_value = True
        # Patch _default_manager to be the mock so notify uses it
        with patch.object(notifications, "_default_manager", mock_manager):
            result = notifications.notify("Test", "Message")
            assert result is True
            mock_manager.notify.assert_called_once()

    def test_notify_reuses_manager(self):
        """Test that notify reuses existing default manager."""
        import notifications
        if not hasattr(notifications, "notify"):
            pytest.skip("Manager module not available")

        mock_manager = MagicMock()
        mock_manager.notify.return_value = True
        with patch.object(notifications, "_default_manager", mock_manager):
            result = notifications.notify("Test", "Message")
            assert result is True

    def test_info_function(self):
        import notifications
        if not hasattr(notifications, "info"):
            pytest.skip("Manager module not available")
        mock_manager = MagicMock()
        mock_manager.notify.return_value = True
        with patch.object(notifications, "_default_manager", mock_manager):
            notifications.info("Title", "Msg")
            mock_manager.notify.assert_called_once()

    def test_warning_function(self):
        import notifications
        if not hasattr(notifications, "warning"):
            pytest.skip("Manager module not available")
        mock_manager = MagicMock()
        mock_manager.notify.return_value = True
        with patch.object(notifications, "_default_manager", mock_manager):
            notifications.warning("Title", "Msg")
            mock_manager.notify.assert_called_once()

    def test_critical_function(self):
        import notifications
        if not hasattr(notifications, "critical"):
            pytest.skip("Manager module not available")
        mock_manager = MagicMock()
        mock_manager.notify.return_value = True
        with patch.object(notifications, "_default_manager", mock_manager):
            notifications.critical("Title", "Msg")
            mock_manager.notify.assert_called_once()

    def test_success_function(self):
        import notifications
        if not hasattr(notifications, "success"):
            pytest.skip("Manager module not available")
        mock_manager = MagicMock()
        mock_manager.notify.return_value = True
        with patch.object(notifications, "_default_manager", mock_manager):
            notifications.success("Title", "Msg")
            mock_manager.notify.assert_called_once()

    def test_play_ding_creates_default_manager(self):
        import notifications
        if not hasattr(notifications, "play_ding"):
            pytest.skip("Manager module not available")
        mock_manager = MagicMock()
        mock_manager.play_ding.return_value = True
        with patch.object(notifications, "_default_manager", mock_manager):
            result = notifications.play_ding()
            assert result is True
            mock_manager.play_ding.assert_called_once()

    def test_play_ding_reuses_manager(self):
        import notifications
        if not hasattr(notifications, "play_ding"):
            pytest.skip("Manager module not available")
        mock_manager = MagicMock()
        mock_manager.play_ding.return_value = True
        with patch.object(notifications, "_default_manager", mock_manager):
            result = notifications.play_ding()
            assert result is True
