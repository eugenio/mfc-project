#!/usr/bin/env python3
"""
Audio Notifier - Plays sounds for TDD events
Supports custom sound files with system sound fallback
"""
import platform
import subprocess
import sys
from pathlib import Path


class AudioNotifier:
    def __init__(self, sounds_dir: str = None):
        self.system = platform.system()
        # Default to sounds directory relative to script location
        if sounds_dir is None:
            script_dir = Path(__file__).parent
            self.sounds_dir = script_dir / "sounds"
        else:
            self.sounds_dir = Path(sounds_dir)

        # Supported audio formats by platform
        self.supported_formats = {
            "Darwin": [".wav", ".aiff", ".mp3"],  # macOS
            "Linux": [".wav", ".mp3"],  # Linux
            "Windows": [".wav", ".mp3"]  # Windows
        }

    def play_sound(self, sound_type: str) -> bool:
        """Play notification sound based on type"""

        if sound_type == "success":
            return self._play_success_sound()
        elif sound_type == "failure":
            return self._play_failure_sound()
        elif sound_type == "completion":
            return self._play_completion_sound()
        else:
            print(f"❌ Unknown sound type: {sound_type}")
            return False

    def _play_success_sound(self) -> bool:
        """Play success sound (TDD cycle passed)"""
        return self._play_custom_or_system_sound("success")

    def _play_failure_sound(self) -> bool:
        """Play failure sound (tests failed)"""
        return self._play_custom_or_system_sound("failure")

    def _play_completion_sound(self) -> bool:
        """Play completion sound (feature finished)"""
        return self._play_custom_or_system_sound("completion")

    def _find_custom_sound_file(self, sound_type: str) -> Path:
        """Find custom sound file for given type"""
        if not self.sounds_dir.exists():
            return None

        formats = self.supported_formats.get(self.system, [".wav"])

        for ext in formats:
            sound_file = self.sounds_dir / f"{sound_type}{ext}"
            if sound_file.exists():
                return sound_file

        return None

    def _play_custom_or_system_sound(self, sound_type: str) -> bool:
        """Try custom sound first, fall back to system sound"""
        custom_file = self._find_custom_sound_file(sound_type)

        if custom_file:
            return self._play_custom_sound_file(custom_file, sound_type)
        else:
            return self._play_system_sound(sound_type)

    def _play_custom_sound_file(self, sound_file: Path, sound_type: str) -> bool:
        """Play custom sound file using appropriate system command"""
        try:
            if self.system == "Darwin":  # macOS
                subprocess.run(["afplay", str(sound_file)], check=True)
            elif self.system == "Linux":
                subprocess.run(["paplay", str(sound_file)], check=True)
            elif self.system == "Windows":
                import winsound
                winsound.PlaySound(str(sound_file), winsound.SND_FILENAME)

            print(f"🔊 Played custom {sound_type} sound: {sound_file.name}")
            return True

        except Exception as e:
            print(f"⚠️  Custom sound failed: {e}")
            print("🔄 Falling back to system sound")
            return self._play_system_sound(sound_type)

    def _play_system_sound(self, sound_type: str) -> bool:
        """Play system sound based on OS"""
        try:
            if self.system == "Darwin":  # macOS
                if sound_type == "success":
                    subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"])
                elif sound_type == "failure":
                    subprocess.run(["afplay", "/System/Library/Sounds/Basso.aiff"])
                elif sound_type == "completion":
                    subprocess.run(["afplay", "/System/Library/Sounds/Blow.aiff"])

            elif self.system == "Linux":
                if sound_type == "success":
                    subprocess.run(["paplay", "/usr/share/sounds/alsa/Front_Left.wav"])
                elif sound_type == "failure":
                    subprocess.run(["paplay", "/usr/share/sounds/alsa/Front_Right.wav"])
                elif sound_type == "completion":
                    subprocess.run(["paplay", "/usr/share/sounds/alsa/Rear_Left.wav"])

            elif self.system == "Windows":
                import winsound
                if sound_type == "success":
                    winsound.MessageBeep(winsound.MB_OK)
                elif sound_type == "failure":
                    winsound.MessageBeep(winsound.MB_ICONHAND)
                elif sound_type == "completion":
                    winsound.MessageBeep(winsound.MB_ICONASTERISK)

            print(f"🔊 Played {sound_type} sound")
            return True

        except Exception as e:
            print(f"❌ Failed to play sound: {e}")
            return False

def main():
    """CLI interface for Audio Notifier"""
    if len(sys.argv) < 2:
        print("Usage: audio_notifier.py <sound_type> [sounds_dir]")
        print("Sound types: success, failure, completion")
        print("sounds_dir: Optional custom sounds directory")
        return 1

    sound_type = sys.argv[1]
    sounds_dir = sys.argv[2] if len(sys.argv) > 2 else None

    notifier = AudioNotifier(sounds_dir)

    return 0 if notifier.play_sound(sound_type) else 1

if __name__ == "__main__":
    sys.exit(main())
