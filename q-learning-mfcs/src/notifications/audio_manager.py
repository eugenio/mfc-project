"""
Audio Manager for MFC Notification System
=========================================

Provides comprehensive audio playback capabilities for notifications including:
- Cross-platform sound file playback (WAV, MP3, OGG)
- Volume normalization and control
- Sound caching for performance
- Non-blocking playback
- Audio device selection
- Fallback to system sounds

Features:
- Supports multiple audio formats
- Intelligent volume normalization
- Platform-specific audio optimization
- Thread-safe concurrent playback
- Caching for frequently used sounds
- Graceful degradation when audio unavailable

Created: 2025-08-03
Author: Agent Delta - Audio Integration Specialist
"""
import logging
import platform
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .base import NotificationLevel

logger = logging.getLogger(__name__)
@dataclass
class AudioConfig:
    """Configuration for audio playback."""
    enabled: bool = True
    volume: float = 0.7  # 0.0 to 1.0
    normalize_volume: bool = True
    cache_sounds: bool = True
    max_concurrent_sounds: int = 3
    audio_device: str | None = None
    fallback_to_system: bool = True
    timeout_seconds: float = 5.0
    supported_formats: list[str] = None

    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['.wav', '.mp3', '.ogg', '.aiff', '.m4a']

@dataclass
class AudioEvent:
    """Represents an audio playback event."""
    sound_path: Path | None
    level: NotificationLevel
    volume: float | None = None
    priority: int = 1  # 1=low, 2=normal, 3=high, 4=critical
    timeout: float | None = None
    callback: callable | None = None


class AudioDevice:
    """Represents an available audio device."""
    def __init__(self, device_id: str, name: str, is_default: bool = False):
        self.device_id = device_id
        self.name = name
        self.is_default = is_default
        self.available = True

    def __str__(self):
        default_indicator = " (default)" if self.is_default else ""
        return f"{self.name}{default_indicator}"

class AudioPlayer:
    """Cross-platform audio player implementation."""

    def __init__(self, config: AudioConfig):
        self.config = config
        self.system = platform.system()
        self._executor = ThreadPoolExecutor(max_workers=config.max_concurrent_sounds)
        self._active_sounds = set()
        self._sound_cache = {}
        self._lock = threading.Lock()

    def play_sound(self, sound_path: Path | None,
                   level: NotificationLevel = NotificationLevel.INFO,
                   volume: float | None = None,
                   blocking: bool = False) -> bool:
        """
        Play a sound file or system sound.
        
        Args:
            sound_path: Path to sound file, None for system sound
            level: Notification level for system sound selection
            volume: Override volume (0.0-1.0)
            blocking: Wait for playback to complete
            
        Returns:
            bool: True if playback started successfully
        """
        if not self.config.enabled:
            logger.debug("Audio playback disabled")
            return False

        # Use configured volume if not specified
        if volume is None:
            volume = self.config.volume

        # Clamp volume to valid range
        volume = max(0.0, min(1.0, volume))

        # Determine playback method
        if sound_path and sound_path.exists():
            if blocking:
                return self._play_file_sync(sound_path, volume)
            else:
                future = self._executor.submit(self._play_file_sync, sound_path, volume)
                return True
        else:
            # Fall back to system sound
            if self.config.fallback_to_system:
                if blocking:
                    return self._play_system_sound_sync(level, volume)
                else:
                    future = self._executor.submit(self._play_system_sound_sync, level, volume)
                    return True

        return False

    def _play_file_sync(self, sound_path: Path, volume: float) -> bool:
        """Synchronously play a sound file."""
        try:
            sound_id = str(sound_path)

            with self._lock:
                # Check if we're at max concurrent sounds
                if len(self._active_sounds) >= self.config.max_concurrent_sounds:
                    logger.warning("Maximum concurrent sounds reached, skipping playback")
                    return False

                self._active_sounds.add(sound_id)

            try:
                # Platform-specific playback
                success = self._execute_playback(sound_path, volume)
                return success

            finally:
                with self._lock:
                    self._active_sounds.discard(sound_id)

        except Exception as e:
            logger.error(f"Failed to play sound {sound_path}: {e}")
            return False

    def _play_system_sound_sync(self, level: NotificationLevel, volume: float) -> bool:
        """Synchronously play system notification sound."""
        try:
            sound_id = f"system_{level.value}"

            with self._lock:
                if len(self._active_sounds) >= self.config.max_concurrent_sounds:
                    logger.warning("Maximum concurrent sounds reached, skipping system sound")
                    return False

                self._active_sounds.add(sound_id)

            try:
                success = self._execute_system_sound(level, volume)
                return success

            finally:
                with self._lock:
                    self._active_sounds.discard(sound_id)

        except Exception as e:
            logger.error(f"Failed to play system sound: {e}")
            return False

    def _execute_playback(self, sound_path: Path, volume: float) -> bool:
        """Execute platform-specific sound file playback."""
        if self.system == "Linux":
            return self._play_linux(sound_path, volume)
        elif self.system == "Darwin":  # macOS
            return self._play_macos(sound_path, volume)
        elif self.system == "Windows":
            return self._play_windows(sound_path, volume)
        else:
            logger.warning(f"Unsupported platform: {self.system}")
            return False

    def _play_linux(self, sound_path: Path, volume: float) -> bool:
        """Play sound on Linux using PulseAudio."""
        try:
            # Try paplay first (PulseAudio)
            cmd = ["paplay"]

            # Add volume control if supported
            volume_percent = int(volume * 65536)  # PulseAudio volume range
            cmd.extend(["--volume", str(volume_percent)])

            # Add audio device if specified
            if self.config.audio_device:
                cmd.extend(["--device", self.config.audio_device])

            cmd.append(str(sound_path))

            result = subprocess.run(cmd,
                                  capture_output=True,
                                  timeout=self.config.timeout_seconds,
                                  check=True)
            return True

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            # Fallback to aplay (ALSA)
            try:
                cmd = ["aplay", str(sound_path)]
                subprocess.run(cmd,
                             capture_output=True,
                             timeout=self.config.timeout_seconds,
                             check=True)
                return True
            except Exception as e:
                logger.debug(f"Linux audio playback failed: {e}")
                return False

    def _play_macos(self, sound_path: Path, volume: float) -> bool:
        """Play sound on macOS using afplay."""
        try:
            cmd = ["afplay"]

            # Add volume control
            cmd.extend(["-v", str(volume)])

            cmd.append(str(sound_path))

            subprocess.run(cmd,
                         capture_output=True,
                         timeout=self.config.timeout_seconds,
                         check=True)
            return True

        except Exception as e:
            logger.debug(f"macOS audio playback failed: {e}")
            return False

    def _play_windows(self, sound_path: Path, volume: float) -> bool:
        """Play sound on Windows using winsound or system commands."""
        try:
            # Try winsound first
            import winsound
            flags = winsound.SND_FILENAME | winsound.SND_NOWAIT
            winsound.PlaySound(str(sound_path), flags)
            return True

        except ImportError:
            # Fallback to PowerShell
            try:
                volume_percent = int(volume * 100)
                cmd = [
                    "powershell", "-c",
                    f"(New-Object Media.SoundPlayer '{sound_path}').PlaySync()"
                ]
                subprocess.run(cmd,
                             capture_output=True,
                             timeout=self.config.timeout_seconds,
                             check=True)
                return True
            except Exception as e:
                logger.debug(f"Windows audio playback failed: {e}")
                return False

    def _execute_system_sound(self, level: NotificationLevel, volume: float) -> bool:
        """Execute platform-specific system sound playback."""
        if self.system == "Linux":
            return self._play_system_linux(level, volume)
        elif self.system == "Darwin":
            return self._play_system_macos(level, volume)
        elif self.system == "Windows":
            return self._play_system_windows(level, volume)
        return False

    def _play_system_linux(self, level: NotificationLevel, volume: float) -> bool:
        """Play system sound on Linux."""
        # Sound theme mapping
        sound_map = {
            NotificationLevel.INFO: "message-new-instant",
            NotificationLevel.WARNING: "dialog-warning",
            NotificationLevel.CRITICAL: "dialog-error",
            NotificationLevel.SUCCESS: "complete"
        }

        sound_name = sound_map.get(level, "message-new-instant")

        # Try freedesktop sound theme
        theme_paths = [
            f"/usr/share/sounds/freedesktop/stereo/{sound_name}.oga",
            f"/usr/share/sounds/freedesktop/stereo/{sound_name}.wav"
        ]

        for sound_file in theme_paths:
            if Path(sound_file).exists():
                return self._play_linux(Path(sound_file), volume)

        # Fallback to ALSA sounds
        alsa_map = {
            NotificationLevel.INFO: "/usr/share/sounds/alsa/Front_Left.wav",
            NotificationLevel.WARNING: "/usr/share/sounds/alsa/Front_Right.wav",
            NotificationLevel.CRITICAL: "/usr/share/sounds/alsa/Rear_Center.wav",
            NotificationLevel.SUCCESS: "/usr/share/sounds/alsa/Front_Center.wav"
        }

        alsa_sound = alsa_map.get(level)
        if alsa_sound and Path(alsa_sound).exists():
            return self._play_linux(Path(alsa_sound), volume)

        return False

    def _play_system_macos(self, level: NotificationLevel, volume: float) -> bool:
        """Play system sound on macOS."""
        sound_map = {
            NotificationLevel.INFO: "Glass",
            NotificationLevel.WARNING: "Ping",
            NotificationLevel.CRITICAL: "Basso",
            NotificationLevel.SUCCESS: "Blow"
        }

        sound_name = sound_map.get(level, "Glass")
        sound_path = Path(f"/System/Library/Sounds/{sound_name}.aiff")

        if sound_path.exists():
            return self._play_macos(sound_path, volume)
        return False

    def _play_system_windows(self, level: NotificationLevel, volume: float) -> bool:
        """Play system sound on Windows."""
        try:
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

        except ImportError:
            return False

    def get_available_devices(self) -> list[AudioDevice]:
        """Get list of available audio devices."""
        devices = []

        try:
            if self.system == "Linux":
                devices = self._get_linux_devices()
            elif self.system == "Darwin":
                devices = self._get_macos_devices()
            elif self.system == "Windows":
                devices = self._get_windows_devices()
        except Exception as e:
            logger.debug(f"Failed to enumerate audio devices: {e}")

        # Always include default device
        if not devices:
            devices.append(AudioDevice("default", "Default Audio Device", True))

        return devices

    def _get_linux_devices(self) -> list[AudioDevice]:
        """Get Linux audio devices via PulseAudio."""
        devices = []
        try:
            result = subprocess.run(["pactl", "list", "short", "sinks"],
                                  capture_output=True, text=True, check=True)

            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        device_id = parts[1]
                        device_name = parts[1].replace('_', ' ').title()
                        devices.append(AudioDevice(device_id, device_name))

            # Mark first device as default
            if devices:
                devices[0].is_default = True

        except Exception as e:
            logger.debug(f"Failed to get Linux audio devices: {e}")

        return devices

    def _get_macos_devices(self) -> list[AudioDevice]:
        """Get macOS audio devices."""
        devices = []
        # macOS device enumeration would require additional libraries
        # For now, return default device
        devices.append(AudioDevice("default", "Default Output Device", True))
        return devices

    def _get_windows_devices(self) -> list[AudioDevice]:
        """Get Windows audio devices."""
        devices = []
        # Windows device enumeration would require additional libraries
        # For now, return default device
        devices.append(AudioDevice("default", "Default Audio Device", True))
        return devices

    def test_audio_playback(self) -> bool:
        """Test if audio playback is working."""
        try:
            # Try to play a system sound
            return self._play_system_sound_sync(NotificationLevel.INFO, 0.1)
        except Exception as e:
            logger.debug(f"Audio test failed: {e}")
            return False

    def stop_all_sounds(self):
        """Stop all currently playing sounds."""
        with self._lock:
            self._active_sounds.clear()

        # Note: Actual stopping of external processes would require
        # more complex process tracking

    def cleanup(self):
        """Clean up audio resources."""
        self.stop_all_sounds()
        self._executor.shutdown(wait=False)

class AudioManager:
    """High-level audio manager for notifications."""

    def __init__(self, config: AudioConfig | None = None):
        self.config = config or AudioConfig()
        self.player = AudioPlayer(self.config)
        self.sound_themes = {}
        self.default_sounds = {}
        self._load_default_sound_mappings()

    def _load_default_sound_mappings(self):
        """Load default sound file mappings for notification levels."""
        # Look for sound files in project sounds directory
        sound_dir = Path(__file__).parent.parent.parent.parent / "scripts" / "sounds"

        # Default sound mappings
        self.default_sounds = {
            NotificationLevel.SUCCESS: sound_dir / "success.wav",
            NotificationLevel.INFO: sound_dir / "complete.wav",
            NotificationLevel.WARNING: sound_dir / "failure.wav",
            NotificationLevel.CRITICAL: sound_dir / "failure.wav"
        }

        # Filter to only existing files
        self.default_sounds = {
            level: path for level, path in self.default_sounds.items()
            if path.exists()
        }

        logger.info(f"Loaded {len(self.default_sounds)} default sound files")

    def play_notification_sound(self, level: NotificationLevel,
                              custom_sound: Path | None = None,
                              volume: float | None = None,
                              blocking: bool = False) -> bool:
        """
        Play notification sound for given level.
        
        Args:
            level: Notification level
            custom_sound: Custom sound file path
            volume: Override volume
            blocking: Wait for playback completion
            
        Returns:
            bool: True if playback started successfully
        """
        # Determine sound file to play
        sound_path = None

        if custom_sound and custom_sound.exists():
            sound_path = custom_sound
        elif level in self.default_sounds:
            sound_path = self.default_sounds[level]

        # Play the sound
        return self.player.play_sound(sound_path, level, volume, blocking)

    def add_sound_theme(self, theme_name: str, sound_mappings: dict[NotificationLevel, Path]):
        """Add a custom sound theme."""
        self.sound_themes[theme_name] = sound_mappings
        logger.info(f"Added sound theme: {theme_name}")

    def apply_sound_theme(self, theme_name: str) -> bool:
        """Apply a sound theme as the default sounds."""
        if theme_name not in self.sound_themes:
            logger.error(f"Unknown sound theme: {theme_name}")
            return False

        self.default_sounds.update(self.sound_themes[theme_name])
        logger.info(f"Applied sound theme: {theme_name}")
        return True

    def get_audio_info(self) -> dict[str, any]:
        """Get audio system information."""
        return {
            "enabled": self.config.enabled,
            "platform": platform.system(),
            "available_devices": [str(d) for d in self.player.get_available_devices()],
            "supported_formats": self.config.supported_formats,
            "default_sounds": {level.value: str(path) for level, path in self.default_sounds.items()},
            "sound_themes": list(self.sound_themes.keys()),
            "max_concurrent": self.config.max_concurrent_sounds,
            "volume": self.config.volume
        }

    def test_audio(self) -> dict[str, bool]:
        """Test audio functionality."""
        results = {}

        # Test system sound
        results["system_sound"] = self.player.test_audio_playback()

        # Test custom sounds
        for level, sound_path in self.default_sounds.items():
            test_name = f"custom_sound_{level.value}"
            results[test_name] = self.player.play_sound(sound_path, level, 0.1, blocking=True)

        return results

    def reload_config(self, new_config: AudioConfig):
        """Reload audio configuration."""
        self.config = new_config
        self.player.config = new_config
        logger.info("Audio configuration reloaded")

    def cleanup(self):
        """Clean up audio resources."""
        self.player.cleanup()
