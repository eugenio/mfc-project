"""
Linux/GNOME Desktop Notification Handler.

Implements notifications using:
1. libnotify (notify-send command)
2. D-Bus interface (python-dbus)
3. Plyer library fallback
4. PulseAudio for sound
"""
import logging
import shutil
import subprocess
from pathlib import Path

from .base import NotificationConfig, NotificationHandler, NotificationLevel

logger = logging.getLogger(__name__)
class LinuxNotificationHandler(NotificationHandler):
    """Linux desktop notification handler using libnotify and D-Bus."""

    def __init__(self, app_name: str = "MFC Monitor"):
        super().__init__(app_name)
        self._dbus_available = None
        self._notify_send_available = None
        self._pulseaudio_available = None

    def is_available(self) -> bool:
        """Check if Linux notifications are available."""
        if self._is_available is None:
            # Check if we're on Linux with desktop environment
            import platform
            if platform.system() != "Linux":
                self._is_available = False
                return False

            # Check for display
            import os
            if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
                self.logger.debug("No display environment found")
                self._is_available = False
                return False

            # Check for notification methods
            self._is_available = (self._check_notify_send() or
                                self._check_dbus() or
                                self._check_plyer())

        return self._is_available

    def _check_notify_send(self) -> bool:
        """Check if notify-send is available."""
        if self._notify_send_available is None:
            self._notify_send_available = shutil.which("notify-send") is not None
        return self._notify_send_available

    def _check_dbus(self) -> bool:
        """Check if D-Bus notifications are available."""
        if self._dbus_available is None:
            try:
                import dbus
                bus = dbus.SessionBus()
                proxy = bus.get_object('org.freedesktop.Notifications',
                                     '/org/freedesktop/Notifications')
                interface = dbus.Interface(proxy, 'org.freedesktop.Notifications')
                # Test the interface
                interface.GetCapabilities()
                self._dbus_available = True
            except Exception as e:
                self.logger.debug(f"D-Bus not available: {e}")
                self._dbus_available = False
        return self._dbus_available

    def _check_plyer(self) -> bool:
        """Check if plyer is available as fallback."""
        try:
            import plyer
            return hasattr(plyer, 'notification')
        except ImportError:
            return False

    def _check_pulseaudio(self) -> bool:
        """Check if PulseAudio is available for sound."""
        if self._pulseaudio_available is None:
            self._pulseaudio_available = shutil.which("paplay") is not None
        return self._pulseaudio_available

    def send_notification(self, config: NotificationConfig) -> bool:
        """Send notification using the best available method."""
        if not self.validate_config(config):
            return False

        # Try methods in order of preference
        success = False

        if self._check_dbus():
            success = self._send_dbus_notification(config)
            if success:
                self.logger.debug("Sent notification via D-Bus")

        if not success and self._check_notify_send():
            success = self._send_notify_send(config)
            if success:
                self.logger.debug("Sent notification via notify-send")

        if not success and self._check_plyer():
            success = self._send_plyer_notification(config)
            if success:
                self.logger.debug("Sent notification via plyer")

        if not success:
            self.logger.warning("All notification methods failed, using fallback")
            self.log_fallback(config)
            success = True

        # Play sound if enabled and notification was sent
        if success and config.sound_enabled:
            self.play_sound(config.sound_path, config.level)

        return success

    def _send_dbus_notification(self, config: NotificationConfig) -> bool:
        """Send notification via D-Bus interface."""
        try:
            import dbus

            bus = dbus.SessionBus()
            proxy = bus.get_object('org.freedesktop.Notifications',
                                 '/org/freedesktop/Notifications')
            interface = dbus.Interface(proxy, 'org.freedesktop.Notifications')

            # Map urgency levels
            urgency_map = {
                NotificationLevel.INFO: 1,      # Normal
                NotificationLevel.WARNING: 1,   # Normal
                NotificationLevel.CRITICAL: 2,  # Critical
                NotificationLevel.SUCCESS: 0    # Low
            }

            # Prepare hints
            hints = {
                'urgency': dbus.Byte(urgency_map.get(config.level, 1)),
                'category': dbus.String(config.category),
                'desktop-entry': dbus.String(self.app_name.lower().replace(' ', '-'))
            }

            if config.sound_enabled:
                hints['sound-name'] = dbus.String('message-new-instant')
            else:
                hints['suppress-sound'] = dbus.Boolean(True)

            # Set icon
            icon = ""
            if config.icon_path and config.icon_path.exists():
                icon = str(config.icon_path)
            else:
                # Use standard icons based on level
                icon_map = {
                    NotificationLevel.INFO: "dialog-information",
                    NotificationLevel.WARNING: "dialog-warning",
                    NotificationLevel.CRITICAL: "dialog-error",
                    NotificationLevel.SUCCESS: "dialog-information"
                }
                icon = icon_map.get(config.level, "dialog-information")

            # Prepare actions
            actions = []
            if config.actions:
                for action_id, label in config.actions.items():
                    actions.extend([action_id, label])

            # Send notification
            timeout = config.timeout_ms if config.timeout_ms else -1

            notification_id = interface.Notify(
                self.app_name,           # app_name
                0,                       # replaces_id
                icon,                    # app_icon
                config.title,            # summary
                config.message,          # body
                actions,                 # actions
                hints,                   # hints
                timeout                  # timeout
            )

            self.logger.debug(f"D-Bus notification sent with ID: {notification_id}")
            return True

        except Exception as e:
            self.logger.error(f"D-Bus notification failed: {e}")
            return False

    def _send_notify_send(self, config: NotificationConfig) -> bool:
        """Send notification via notify-send command."""
        try:
            cmd = ["notify-send"]

            # Add urgency
            urgency_map = {
                NotificationLevel.INFO: "normal",
                NotificationLevel.WARNING: "normal",
                NotificationLevel.CRITICAL: "critical",
                NotificationLevel.SUCCESS: "low"
            }
            urgency = urgency_map.get(config.level, "normal")
            cmd.extend(["-u", urgency])

            # Add timeout
            if config.timeout_ms:
                cmd.extend(["-t", str(config.timeout_ms)])

            # Add app name
            cmd.extend(["-a", self.app_name])

            # Add category
            cmd.extend(["-c", config.category])

            # Add icon
            if config.icon_path and config.icon_path.exists():
                cmd.extend(["-i", str(config.icon_path)])
            else:
                icon_map = {
                    NotificationLevel.INFO: "dialog-information",
                    NotificationLevel.WARNING: "dialog-warning",
                    NotificationLevel.CRITICAL: "dialog-error",
                    NotificationLevel.SUCCESS: "dialog-information"
                }
                icon = icon_map.get(config.level, "dialog-information")
                cmd.extend(["-i", icon])

            # Add title and message
            cmd.extend([config.title, config.message])

            # Execute command
            result = subprocess.run(cmd,
                                  capture_output=True,
                                  text=True,
                                  timeout=10)

            if result.returncode == 0:
                self.logger.debug("notify-send notification sent successfully")
                return True
            else:
                self.logger.error(f"notify-send failed: {result.stderr}")
                return False

        except Exception as e:
            self.logger.error(f"notify-send notification failed: {e}")
            return False

    def _send_plyer_notification(self, config: NotificationConfig) -> bool:
        """Send notification via plyer library."""
        try:
            import plyer

            # Plyer has limited options
            timeout = (config.timeout_ms // 1000) if config.timeout_ms else 10

            plyer.notification.notify(
                title=config.title,
                message=config.message,
                app_name=self.app_name,
                timeout=timeout,
                app_icon=str(config.icon_path) if config.icon_path else None
            )

            self.logger.debug("Plyer notification sent successfully")
            return True

        except Exception as e:
            self.logger.error(f"Plyer notification failed: {e}")
            return False

    def get_capabilities(self) -> dict[str, bool]:
        """Get Linux notification capabilities."""
        caps = {
            "actions": False,
            "sound": self._check_pulseaudio(),
            "icons": True,
            "persistence": True,
            "urgency": True,
            "markup": True
        }

        # Check if D-Bus supports actions
        if self._check_dbus():
            try:
                import dbus
                bus = dbus.SessionBus()
                proxy = bus.get_object('org.freedesktop.Notifications',
                                     '/org/freedesktop/Notifications')
                interface = dbus.Interface(proxy, 'org.freedesktop.Notifications')
                capabilities = interface.GetCapabilities()
                caps["actions"] = "actions" in capabilities
                caps["markup"] = "body-markup" in capabilities
            except Exception as e:
                self.logger.debug(f"Could not get D-Bus capabilities: {e}")

        return caps

    def _play_custom_sound(self, sound_path: Path) -> bool:
        """Play custom sound using PulseAudio."""
        if not self._check_pulseaudio():
            return False

        try:
            subprocess.run(["paplay", str(sound_path)],
                         check=True,
                         capture_output=True,
                         timeout=5)
            return True
        except Exception as e:
            self.logger.debug(f"Failed to play custom sound: {e}")
            return False

    def _play_system_sound(self, level: NotificationLevel) -> bool:
        """Play system notification sound."""
        if not self._check_pulseaudio():
            return False

        # Try freedesktop sound theme first
        sound_map = {
            NotificationLevel.INFO: "message-new-instant",
            NotificationLevel.WARNING: "dialog-warning",
            NotificationLevel.CRITICAL: "dialog-error",
            NotificationLevel.SUCCESS: "complete"
        }

        sound_name = sound_map.get(level, "message-new-instant")

        try:
            # Try themed sound first
            subprocess.run(["paplay", f"/usr/share/sounds/freedesktop/stereo/{sound_name}.oga"],
                         check=True, capture_output=True, timeout=5)
            return True
        except subprocess.CalledProcessError:
            # Fall back to ALSA sounds
            alsa_sound_map = {
                NotificationLevel.INFO: "/usr/share/sounds/alsa/Front_Left.wav",
                NotificationLevel.WARNING: "/usr/share/sounds/alsa/Front_Right.wav",
                NotificationLevel.CRITICAL: "/usr/share/sounds/alsa/Rear_Center.wav",
                NotificationLevel.SUCCESS: "/usr/share/sounds/alsa/Front_Center.wav"
            }

            sound_file = alsa_sound_map.get(level)
            if sound_file and Path(sound_file).exists():
                try:
                    subprocess.run(["paplay", sound_file],
                                 check=True, capture_output=True, timeout=5)
                    return True
                except Exception as e:
                    self.logger.debug(f"ALSA sound failed: {e}")
        except Exception as e:
            self.logger.debug(f"Themed sound failed: {e}")

        return False
