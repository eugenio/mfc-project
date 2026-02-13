"""
Base notification handler interface and common types.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class NotificationLevel(Enum):
    """Notification urgency levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    SUCCESS = "success"


@dataclass
class NotificationConfig:
    """Configuration for notification appearance and behavior."""
    title: str
    message: str
    level: NotificationLevel = NotificationLevel.INFO
    timeout_ms: int | None = 5000  # None for persistent
    icon_path: Path | None = None
    sound_enabled: bool = True
    sound_path: Path | None = None
    actions: dict[str, str] | None = None  # {"action_id": "label"}
    app_name: str = "MFC Monitor"
    category: str = "mfc.alert"


class NotificationHandler(ABC):
    """Base class for platform-specific notification handlers."""

    def __init__(self, app_name: str = "MFC Monitor"):
        self.app_name = app_name
        self.logger = logging.getLogger(self.__class__.__name__)
        self._is_available = None

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this notification handler can be used on the current system."""
        pass

    @abstractmethod
    def send_notification(self, config: NotificationConfig) -> bool:
        """
        Send a notification using platform-specific mechanism.
        
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> dict[str, bool]:
        """
        Get capabilities supported by this handler.
        
        Returns:
            dict: Capabilities like {"actions": True, "sound": True, "icons": True}
        """
        pass

    def play_sound(self, sound_path: Path | None = None, level: NotificationLevel = NotificationLevel.INFO) -> bool:
        """
        Play notification sound.
        
        Args:
            sound_path: Custom sound file path
            level: Notification level for default sound selection
            
        Returns:
            bool: True if sound played successfully
        """
        try:
            if sound_path and sound_path.exists():
                return self._play_custom_sound(sound_path)
            else:
                return self._play_system_sound(level)
        except Exception as e:
            self.logger.error(f"Failed to play sound: {e}")
            return False

    @abstractmethod
    def _play_custom_sound(self, sound_path: Path) -> bool:
        """Play a custom sound file."""
        pass

    @abstractmethod
    def _play_system_sound(self, level: NotificationLevel) -> bool:
        """Play system sound for notification level."""
        pass

    def validate_config(self, config: NotificationConfig) -> bool:
        """Validate notification configuration for this handler."""
        if not config.title or not config.message:
            self.logger.error("Title and message are required")
            return False
        return True

    def log_fallback(self, config: NotificationConfig) -> None:
        """Log notification as fallback when GUI notifications fail."""
        level_map = {
            NotificationLevel.INFO: logging.INFO,
            NotificationLevel.WARNING: logging.WARNING,
            NotificationLevel.CRITICAL: logging.CRITICAL,
            NotificationLevel.SUCCESS: logging.INFO
        }

        log_level = level_map.get(config.level, logging.INFO)
        message = f"[{config.level.value.upper()}] {config.title}: {config.message}"
        self.logger.log(log_level, message)


class FallbackHandler(NotificationHandler):
    """Fallback handler that logs notifications to console and logger."""

    def is_available(self) -> bool:
        """Fallback is always available."""
        return True

    def send_notification(self, config: NotificationConfig) -> bool:
        """Send notification via console output and logging."""
        if not self.validate_config(config):
            return False

        # Console output with formatting
        icon_map = {
            NotificationLevel.INFO: "ℹ️",
            NotificationLevel.WARNING: "⚠️",
            NotificationLevel.CRITICAL: "🚨",
            NotificationLevel.SUCCESS: "✅"
        }

        icon = icon_map.get(config.level, "📢")
        print(f"\n{icon} {config.title}")
        print(f"   {config.message}")
        if config.timeout_ms:
            print(f"   (Auto-dismiss in {config.timeout_ms/1000:.1f}s)")
        print()

        # Log the notification
        self.log_fallback(config)

        # Play sound if enabled
        if config.sound_enabled:
            self.play_sound(config.sound_path, config.level)

        return True

    def get_capabilities(self) -> dict[str, bool]:
        """Get fallback handler capabilities."""
        return {
            "actions": False,
            "sound": True,
            "icons": False,
            "persistence": False,
            "urgency": False
        }

    def _play_custom_sound(self, sound_path: Path) -> bool:
        """Attempt to play custom sound via system commands."""
        import platform
        import subprocess

        system = platform.system()
        try:
            if system == "Linux":
                subprocess.run(["paplay", str(sound_path)], check=True, capture_output=True)
            elif system == "Darwin":  # macOS
                subprocess.run(["afplay", str(sound_path)], check=True, capture_output=True)
            elif system == "Windows":
                # Use winsound for Windows
                import winsound
                winsound.PlaySound(str(sound_path), winsound.SND_FILENAME | winsound.SND_NOWAIT)
            else:
                return False
            return True
        except Exception as e:
            self.logger.debug(f"Failed to play custom sound: {e}")
            return False

    def _play_system_sound(self, level: NotificationLevel) -> bool:
        """Play system notification sound."""
        import platform
        import subprocess

        system = platform.system()
        try:
            if system == "Linux":
                # Use paplay with system sounds
                sound_map = {
                    NotificationLevel.INFO: "/usr/share/sounds/alsa/Front_Left.wav",
                    NotificationLevel.WARNING: "/usr/share/sounds/alsa/Front_Right.wav",
                    NotificationLevel.CRITICAL: "/usr/share/sounds/alsa/Rear_Center.wav",
                    NotificationLevel.SUCCESS: "/usr/share/sounds/alsa/Front_Center.wav"
                }
                sound_file = sound_map.get(level)
                if sound_file and Path(sound_file).exists():
                    subprocess.run(["paplay", sound_file], check=True, capture_output=True)
                    return True

            elif system == "Darwin":  # macOS
                sound_map = {
                    NotificationLevel.INFO: "Glass",
                    NotificationLevel.WARNING: "Ping",
                    NotificationLevel.CRITICAL: "Basso",
                    NotificationLevel.SUCCESS: "Blow"
                }
                sound_name = sound_map.get(level, "Glass")
                subprocess.run(["afplay", f"/System/Library/Sounds/{sound_name}.aiff"],
                             check=True, capture_output=True)
                return True

            elif system == "Windows":
                import winsound
                sound_map = {
                    NotificationLevel.INFO: winsound.MB_OK,
                    NotificationLevel.WARNING: winsound.MB_ICONEXCLAMATION,
                    NotificationLevel.CRITICAL: winsound.MB_ICONHAND,
                    NotificationLevel.SUCCESS: winsound.MB_OK
                }
                sound_type = sound_map.get(level, winsound.MB_OK)
                winsound.MessageBeep(sound_type)
                return True

        except Exception as e:
            self.logger.debug(f"Failed to play system sound: {e}")

        return False
