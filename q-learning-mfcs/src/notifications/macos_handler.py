"""
macOS Desktop Notification Handler.

Implements notifications using:
1. osascript with AppleScript
2. terminal-notifier command line tool
3. macOS Notification Center
4. System sounds via afplay
"""
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Optional

from .base import NotificationHandler, NotificationConfig, NotificationLevel

logger = logging.getLogger(__name__)
class MacOSNotificationHandler(NotificationHandler):
    """macOS desktop notification handler using osascript and terminal-notifier."""
    
    def __init__(self, app_name: str = "MFC Monitor"):
        super().__init__(app_name)
        self._osascript_available = None
        self._terminal_notifier_available = None
        self._afplay_available = None
        
    def is_available(self) -> bool:
        """Check if macOS notifications are available."""
        if self._is_available is None:
            import platform
            if platform.system() != "Darwin":
                self._is_available = False
                return False
                
            # Check for notification methods
            self._is_available = (self._check_osascript() or 
                                self._check_terminal_notifier())
                                
        return self._is_available
    
    def _check_osascript(self) -> bool:
        """Check if osascript is available."""
        if self._osascript_available is None:
            self._osascript_available = shutil.which("osascript") is not None
        return self._osascript_available
    
    def _check_terminal_notifier(self) -> bool:
        """Check if terminal-notifier is available."""
        if self._terminal_notifier_available is None:
            self._terminal_notifier_available = shutil.which("terminal-notifier") is not None
        return self._terminal_notifier_available
    
    def _check_afplay(self) -> bool:
        """Check if afplay is available for sound."""
        if self._afplay_available is None:
            self._afplay_available = shutil.which("afplay") is not None
        return self._afplay_available
    
    def send_notification(self, config: NotificationConfig) -> bool:
        """Send notification using the best available method."""
        if not self.validate_config(config):
            return False
            
        success = False
        
        # Try methods in order of preference
        if self._check_terminal_notifier():
            success = self._send_terminal_notifier(config)
            if success:
                self.logger.debug("Sent notification via terminal-notifier")
        
        if not success and self._check_osascript():
            success = self._send_osascript_notification(config)
            if success:
                self.logger.debug("Sent notification via osascript")
        
        if not success:
            self.logger.warning("All macOS notification methods failed, using fallback")
            self.log_fallback(config)
            success = True
        
        # Play sound if enabled and notification was sent
        if success and config.sound_enabled:
            self.play_sound(config.sound_path, config.level)
            
        return success
    
    def _send_terminal_notifier(self, config: NotificationConfig) -> bool:
        """Send notification via terminal-notifier."""
        try:
            cmd = ["terminal-notifier"]
            
            # Basic parameters
            cmd.extend(["-title", config.title])
            cmd.extend(["-message", config.message])
            cmd.extend(["-sender", f"com.{self.app_name.lower().replace(' ', '')}.app"])
            
            # Add subtitle for app name
            cmd.extend(["-subtitle", self.app_name])
            
            # Add sound
            if config.sound_enabled:
                sound_map = {
                    NotificationLevel.INFO: "Glass",
                    NotificationLevel.WARNING: "Ping",
                    NotificationLevel.CRITICAL: "Basso",
                    NotificationLevel.SUCCESS: "Blow"
                }
                sound = sound_map.get(config.level, "Glass")
                cmd.extend(["-sound", sound])
            
            # Add icon if provided
            if config.icon_path and config.icon_path.exists():
                cmd.extend(["-appIcon", str(config.icon_path)])
            
            # Add actions if provided
            if config.actions:
                # terminal-notifier supports limited actions
                action_list = list(config.actions.values())
                if len(action_list) > 0:
                    cmd.extend(["-actions", ",".join(action_list[:3])])  # Max 3 actions
            
            # Add timeout (terminal-notifier doesn't support timeout directly)
            # but we can add group for better management
            cmd.extend(["-group", f"{self.app_name}-{config.level.value}"])
            
            # Execute command
            result = subprocess.run(cmd, 
                                  capture_output=True, 
                                  text=True,
                                  timeout=10)
            
            if result.returncode == 0:
                self.logger.debug("terminal-notifier notification sent successfully")
                return True
            else:
                self.logger.error(f"terminal-notifier failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"terminal-notifier notification failed: {e}")
            return False
    
    def _send_osascript_notification(self, config: NotificationConfig) -> bool:
        """Send notification via osascript AppleScript."""
        try:
            # Build AppleScript
            script_parts = []
            
            # Basic notification
            script_parts.append(f'display notification "{config.message}"')
            script_parts.append(f'with title "{config.title}"')
            script_parts.append(f'subtitle "{self.app_name}"')
            
            # Add sound
            if config.sound_enabled:
                sound_map = {
                    NotificationLevel.INFO: "Glass",
                    NotificationLevel.WARNING: "Ping", 
                    NotificationLevel.CRITICAL: "Basso",
                    NotificationLevel.SUCCESS: "Blow"
                }
                sound = sound_map.get(config.level, "Glass")
                script_parts.append(f'sound name "{sound}"')
            
            # Combine script
            applescript = " ".join(script_parts)
            
            # Execute osascript
            result = subprocess.run(["osascript", "-e", applescript],
                                  capture_output=True,
                                  text=True,
                                  timeout=10)
            
            if result.returncode == 0:
                self.logger.debug("osascript notification sent successfully")
                return True
            else:
                self.logger.error(f"osascript failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"osascript notification failed: {e}")
            return False
    
    def get_capabilities(self) -> dict[str, bool]:
        """Get macOS notification capabilities."""
        caps = {
            "actions": self._check_terminal_notifier(),  # Only terminal-notifier supports actions
            "sound": self._check_afplay(),
            "icons": self._check_terminal_notifier(),  # Only terminal-notifier supports custom icons
            "persistence": True,
            "urgency": False,  # macOS doesn't have urgency levels
            "markup": False
        }
        
        return caps
    
    def _play_custom_sound(self, sound_path: Path) -> bool:
        """Play custom sound using afplay."""
        if not self._check_afplay():
            return False
            
        try:
            subprocess.run(["afplay", str(sound_path)], 
                         check=True, 
                         capture_output=True,
                         timeout=5)
            return True
        except Exception as e:
            self.logger.debug(f"Failed to play custom sound: {e}")
            return False
    
    def _play_system_sound(self, level: NotificationLevel) -> bool:
        """Play system notification sound."""
        if not self._check_afplay():
            return False
            
        # Map notification levels to system sounds
        sound_map = {
            NotificationLevel.INFO: "Glass",
            NotificationLevel.WARNING: "Ping",
            NotificationLevel.CRITICAL: "Basso", 
            NotificationLevel.SUCCESS: "Blow"
        }
        
        sound_name = sound_map.get(level, "Glass")
        sound_path = f"/System/Library/Sounds/{sound_name}.aiff"
        
        try:
            subprocess.run(["afplay", sound_path],
                         check=True, 
                         capture_output=True,
                         timeout=5)
            return True
        except Exception as e:
            self.logger.debug(f"Failed to play system sound: {e}")
            return False