"""Audio Notifier - Plays sounds for TDD events."""

import platform
import subprocess
import sys


class AudioNotifier:
    def __init__(self) -> None:
        self.system = platform.system()

    def play_sound(self, sound_type: str) -> bool:
        """Play notification sound based on type."""
        if sound_type == "success":
            return self._play_success_sound()
        if sound_type == "failure":
            return self._play_failure_sound()
        if sound_type == "completion":
            return self._play_completion_sound()
        return False

    def _play_success_sound(self) -> bool:
        """Play success sound (TDD cycle passed)."""
        return self._play_system_sound("success")

    def _play_failure_sound(self) -> bool:
        """Play failure sound (tests failed)."""
        return self._play_system_sound("failure")

    def _play_completion_sound(self) -> bool:
        """Play completion sound (feature finished)."""
        return self._play_system_sound("completion")

    def _play_system_sound(self, sound_type: str) -> bool:
        """Play system sound based on OS."""
        try:
            if self.system == "Darwin":  # macOS
                if sound_type == "success":
                    subprocess.run(
                        ["afplay", "/System/Library/Sounds/Glass.aiff"],
                        check=False,
                    )
                elif sound_type == "failure":
                    subprocess.run(
                        ["afplay", "/System/Library/Sounds/Basso.aiff"],
                        check=False,
                    )
                elif sound_type == "completion":
                    subprocess.run(
                        ["afplay", "/System/Library/Sounds/Blow.aiff"],
                        check=False,
                    )

            elif self.system == "Linux":
                if sound_type == "success":
                    subprocess.run(
                        ["paplay", "/usr/share/sounds/alsa/Front_Left.wav"],
                        check=False,
                    )
                elif sound_type == "failure":
                    subprocess.run(
                        ["paplay", "/usr/share/sounds/alsa/Front_Right.wav"],
                        check=False,
                    )
                elif sound_type == "completion":
                    subprocess.run(
                        ["paplay", "/usr/share/sounds/alsa/Rear_Left.wav"],
                        check=False,
                    )

            elif self.system == "Windows":
                import winsound

                if sound_type == "success":
                    winsound.MessageBeep(winsound.MB_OK)
                elif sound_type == "failure":
                    winsound.MessageBeep(winsound.MB_ICONHAND)
                elif sound_type == "completion":
                    winsound.MessageBeep(winsound.MB_ICONASTERISK)

            return True

        except Exception:
            return False


def main() -> int:
    """CLI interface for Audio Notifier."""
    if len(sys.argv) < 2:
        return 1

    notifier = AudioNotifier()
    sound_type = sys.argv[1]

    return 0 if notifier.play_sound(sound_type) else 1


if __name__ == "__main__":
    sys.exit(main())
