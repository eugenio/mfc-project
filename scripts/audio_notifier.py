"""
Audio Notifier - Plays sounds for TDD events
"""
import sys
import subprocess
import platform
from pathlib import Path
class AudioNotifier:
    def __init__(self):
        self.system = platform.system()
        
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
        return self._play_system_sound("success")
        
    def _play_failure_sound(self) -> bool:
        """Play failure sound (tests failed)"""
        return self._play_system_sound("failure")
        
    def _play_completion_sound(self) -> bool:
        """Play completion sound (feature finished)"""
        return self._play_system_sound("completion")
        
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
        print("Usage: audio_notifier.py <sound_type>")
        print("Sound types: success, failure, completion")
        return 1
        
    notifier = AudioNotifier()
    sound_type = sys.argv[1]
    
    return 0 if notifier.play_sound(sound_type) else 1
if __name__ == "__main__":
    sys.exit(main())