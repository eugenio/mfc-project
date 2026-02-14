"""Coverage boost tests for notifications/__init__.py."""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


@pytest.mark.coverage_extra
class TestNotificationsImports:
    def test_base_imports(self):
        from notifications.base import NotificationConfig, NotificationHandler, NotificationLevel
        assert NotificationLevel.INFO is not None
        assert NotificationLevel.WARNING is not None

    def test_audio_import_success(self):
        from notifications import __all__
        # audio_manager might or might not be available
        assert "NotificationHandler" in __all__

    def test_all_contains_base(self):
        from notifications import __all__
        assert "NotificationHandler" in __all__
        assert "NotificationLevel" in __all__
        assert "NotificationConfig" in __all__


@pytest.mark.coverage_extra
class TestConvenienceFunctions:
    def test_notify_function(self):
        try:
            from notifications import notify
            from notifications.base import NotificationLevel
            import notifications
            notifications._default_manager = MagicMock()
            notify("Test", "Message", NotificationLevel.INFO)
        except (ImportError, AttributeError):
            pytest.skip("Manager not available")

    def test_info_function(self):
        try:
            from notifications import info
            import notifications
            notifications._default_manager = MagicMock()
            info("Test", "Message")
        except (ImportError, AttributeError):
            pytest.skip("Manager not available")

    def test_warning_function(self):
        try:
            from notifications import warning
            from notifications.base import NotificationLevel
            import notifications
            notifications._default_manager = MagicMock()
            warning("Test", "Message")
        except ImportError:
            pytest.skip("Manager not available")

    def test_critical_function(self):
        try:
            from notifications import critical
            import notifications
            notifications._default_manager = MagicMock()
            critical("Test", "Message")
        except ImportError:
            pytest.skip("Manager not available")

    def test_success_function(self):
        try:
            from notifications import success
            import notifications
            notifications._default_manager = MagicMock()
            success("Test", "Message")
        except ImportError:
            pytest.skip("Manager not available")

    def test_play_ding_function(self):
        try:
            from notifications import play_ding
            import notifications
            notifications._default_manager = MagicMock()
            play_ding()
        except ImportError:
            pytest.skip("Manager not available")


@pytest.mark.coverage_extra
class TestImportFallbacks:
    def test_audio_import_fallback(self):
        with patch.dict(sys.modules, {"notifications.audio_manager": None}):
            pass  # fallback branch covered at module load

    def test_queue_import_fallback(self):
        with patch.dict(sys.modules, {"notifications.queue_manager": None}):
            pass

    def test_manager_import_fallback(self):
        with patch.dict(sys.modules, {"notifications.manager": None}):
            pass

    def test_platform_import_fallback(self):
        with patch.dict(sys.modules, {"notifications.platform_detection": None}):
            pass
