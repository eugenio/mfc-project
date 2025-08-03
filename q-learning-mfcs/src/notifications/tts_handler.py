"""
Text-to-Speech (TTS) Notification Handler.

Implements TTS notifications using pyttsx3 with:
1. Cross-platform voice synthesis
2. Platform-specific voice selection and configuration
3. Audio toggle (TTS vs ding sound fallback)
4. Asynchronous playback with queue management
5. Error handling and graceful degradation
"""
import asyncio
import logging
import platform
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from .base import NotificationHandler, NotificationConfig, NotificationLevel

logger = logging.getLogger(__name__)
class TTSMode(Enum):
    """TTS operation modes."""
    DISABLED = "disabled"
    TTS_ONLY = "tts_only"
    SOUND_ONLY = "sound_only"
    TTS_WITH_FALLBACK = "tts_with_fallback"

@dataclass
class TTSConfig:
    """Configuration for TTS behavior."""
    mode: TTSMode = TTSMode.TTS_WITH_FALLBACK
    rate: int = 200  # Words per minute
    volume: float = 0.9  # 0.0 to 1.0
    voice_id: Optional[str] = None  # Platform-specific voice ID
    language: str = "en"
    max_length: int = 500  # Maximum characters to speak
    timeout_seconds: float = 30.0  # Speech timeout
    queue_enabled: bool = True
    preprocessing_enabled: bool = True
    custom_pronunciations: Dict[str, str] = field(default_factory=dict)


@dataclass
class TTSQueueItem:
    """Item in the TTS queue."""
    text: str
    config: TTSConfig
    priority: int = 0  # Higher priority first
    timestamp: float = field(default_factory=time.time)

class TTSNotificationHandler(NotificationHandler):
    """Cross-platform TTS notification handler with intelligent fallback."""
    
    def __init__(self, app_name: str = "MFC Monitor", tts_config: Optional[TTSConfig] = None):
        super().__init__(app_name)
        self.tts_config = tts_config or TTSConfig()
        self._engine = None
        self._engine_lock = threading.Lock()
        self._tts_queue = queue.PriorityQueue()
        self._worker_thread = None
        self._worker_running = False
        self._platform = platform.system()
        
        # Platform-specific initialization
        self._initialize_platform_specifics()
        
        # Start TTS worker if queue is enabled
        if self.tts_config.queue_enabled:
            self._start_worker()
    
    def __del__(self):
        """Cleanup resources."""
        self._stop_worker()
        self._cleanup_engine()
    
    def _initialize_platform_specifics(self) -> None:
        """Initialize platform-specific TTS settings."""
        self._platform_voices = {}
        self._platform_config = {}
        
        if self._platform == "Windows":
            self._initialize_windows()
        elif self._platform == "Darwin":  # macOS
            self._initialize_macos()
        elif self._platform == "Linux":
            self._initialize_linux()
        else:
            self.logger.warning(f"Unsupported platform: {self._platform}")
    
    def _initialize_windows(self) -> None:
        """Initialize Windows SAPI5 voices."""
        try:
            import pyttsx3
            temp_engine = pyttsx3.init('sapi5')
            voices = temp_engine.getProperty('voices')
            
            for voice in voices or []:
                voice_info = {
                    'id': voice.id,
                    'name': voice.name,
                    'gender': getattr(voice, 'gender', 'unknown'),
                    'age': getattr(voice, 'age', 'unknown'),
                    'languages': getattr(voice, 'languages', [])
                }
                self._platform_voices[voice.id] = voice_info
                
            temp_engine.stop()
            del temp_engine
            
            # Set default voice preferences
            self._platform_config = {
                'preferred_voices': ['Microsoft Zira Desktop', 'Microsoft David Desktop'],
                'engine_name': 'sapi5'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Windows TTS: {e}")
    
    def _initialize_macos(self) -> None:
        """Initialize macOS NSSpeechSynthesizer voices."""
        try:
            import pyttsx3
            temp_engine = pyttsx3.init('nsss')
            voices = temp_engine.getProperty('voices')
            
            for voice in voices or []:
                voice_info = {
                    'id': voice.id,
                    'name': voice.name,
                    'languages': getattr(voice, 'languages', [])
                }
                self._platform_voices[voice.id] = voice_info
                
            temp_engine.stop()
            del temp_engine
            
            # Set default voice preferences
            self._platform_config = {
                'preferred_voices': ['com.apple.speech.synthesis.voice.Alex',
                                   'com.apple.speech.synthesis.voice.Samantha'],
                'engine_name': 'nsss'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to initialize macOS TTS: {e}")
    
    def _initialize_linux(self) -> None:
        """Initialize Linux eSpeak/Festival voices."""
        try:
            import pyttsx3
            temp_engine = pyttsx3.init('espeak')
            voices = temp_engine.getProperty('voices')
            
            for voice in voices or []:
                voice_info = {
                    'id': voice.id,
                    'name': voice.name,
                    'languages': getattr(voice, 'languages', [])
                }
                self._platform_voices[voice.id] = voice_info
                
            temp_engine.stop()
            del temp_engine
            
            # Set default voice preferences
            self._platform_config = {
                'preferred_voices': ['english', 'en', 'en-us'],
                'engine_name': 'espeak'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Linux TTS: {e}")
    
    def _get_tts_engine(self) -> Optional[Any]:
        """Get or create TTS engine with thread safety."""
        with self._engine_lock:
            if self._engine is None:
                try:
                    import pyttsx3
                    
                    # Try platform-specific engine first
                    engine_name = self._platform_config.get('engine_name')
                    if engine_name:
                        self._engine = pyttsx3.init(engine_name)
                    else:
                        self._engine = pyttsx3.init()
                    
                    # Configure engine properties
                    self._configure_engine()
                    
                except Exception as e:
                    self.logger.error(f"Failed to initialize TTS engine: {e}")
                    return None
            
            return self._engine
    
    def _configure_engine(self) -> None:
        """Configure TTS engine properties."""
        if not self._engine:
            return
            
        try:
            # Set rate
            self._engine.setProperty('rate', self.tts_config.rate)
            
            # Set volume
            self._engine.setProperty('volume', self.tts_config.volume)
            
            # Set voice
            if self.tts_config.voice_id:
                voices = self._engine.getProperty('voices')
                for voice in voices or []:
                    if voice.id == self.tts_config.voice_id:
                        self._engine.setProperty('voice', voice.id)
                        break
            else:
                # Use preferred voice for platform
                self._set_preferred_voice()
                
        except Exception as e:
            self.logger.error(f"Failed to configure TTS engine: {e}")
    
    def _set_preferred_voice(self) -> None:
        """Set preferred voice based on platform and language."""
        if not self._engine:
            return
            
        try:
            voices = self._engine.getProperty('voices')
            preferred_voices = self._platform_config.get('preferred_voices', [])
            
            # Try to find preferred voice
            for preferred in preferred_voices:
                for voice in voices or []:
                    if preferred.lower() in voice.name.lower() or preferred == voice.id:
                        self._engine.setProperty('voice', voice.id)
                        self.logger.debug(f"Set voice to: {voice.name}")
                        return
            
            # Fallback: use first available voice
            if voices:
                self._engine.setProperty('voice', voices[0].id)
                self.logger.debug(f"Using fallback voice: {voices[0].name}")
                
        except Exception as e:
            self.logger.error(f"Failed to set preferred voice: {e}")
    
    def _cleanup_engine(self) -> None:
        """Cleanup TTS engine."""
        with self._engine_lock:
            if self._engine:
                try:
                    self._engine.stop()
                except Exception as e:
                    self.logger.debug(f"Error stopping TTS engine: {e}")
                finally:
                    self._engine = None
    
    def _start_worker(self) -> None:
        """Start TTS worker thread."""
        if self._worker_thread and self._worker_thread.is_alive():
            return
            
        self._worker_running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        self.logger.debug("TTS worker thread started")
    
    def _stop_worker(self) -> None:
        """Stop TTS worker thread."""
        self._worker_running = False
        if self._worker_thread and self._worker_thread.is_alive():
            # Add poison pill to wake up worker
            try:
                self._tts_queue.put((999, TTSQueueItem("", self.tts_config)), timeout=1)
            except queue.Full:
                pass
            
            self._worker_thread.join(timeout=5)
            if self._worker_thread.is_alive():
                self.logger.warning("TTS worker thread did not stop gracefully")
    
    def _worker_loop(self) -> None:
        """TTS worker thread main loop."""
        while self._worker_running:
            try:
                # Get next item from queue
                priority, item = self._tts_queue.get(timeout=1)
                
                if not self._worker_running:
                    break
                
                # Skip empty text (poison pill)
                if not item.text.strip():
                    continue
                
                # Speak the text
                self._speak_text_sync(item.text, item.config)
                
                # Mark task as done
                self._tts_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in TTS worker: {e}")
    
    def _preprocess_text(self, text: str, config: TTSConfig) -> str:
        """Preprocess text for better TTS pronunciation."""
        if not config.preprocessing_enabled:
            return text
        
        # Apply custom pronunciations
        for original, replacement in config.custom_pronunciations.items():
            text = text.replace(original, replacement)
        
        # Common technical term replacements
        tech_replacements = {
            'MFC': 'M F C',
            'API': 'A P I',
            'CPU': 'C P U',
            'GPU': 'G P U',
            'RAM': 'R A M',
            'URL': 'U R L',
            'HTTP': 'H T T P',
            'HTTPS': 'H T T P S',
            'SSL': 'S S L',
            'TLS': 'T L S',
            'GUI': 'G U I',
            'CLI': 'C L I',
            'TDD': 'T D D',
            'CI/CD': 'C I C D',
            'SQL': 'S Q L',
            'NoSQL': 'No S Q L',
            'JSON': 'J SON',
            'XML': 'X M L',
            'CSV': 'C S V',
            'PDF': 'P D F'
        }
        
        for original, replacement in tech_replacements.items():
            text = re.sub(rf'\b{re.escape(original)}\b', replacement, text, flags=re.IGNORECASE)
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        # Limit length
        if len(text) > config.max_length:
            text = text[:config.max_length - 3] + "..."
        
        return text
    
    def _speak_text_sync(self, text: str, config: TTSConfig) -> bool:
        """Speak text synchronously."""
        engine = self._get_tts_engine()
        if not engine:
            return False
        
        try:
            # Preprocess text
            processed_text = self._preprocess_text(text, config)
            
            # Speak text
            engine.say(processed_text)
            engine.runAndWait()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to speak text: {e}")
            return False
    
    def speak_text_async(self, text: str, priority: int = 0) -> bool:
        """Add text to TTS queue for asynchronous speaking."""
        if self.tts_config.mode == TTSMode.DISABLED:
            return False
        
        if not self.tts_config.queue_enabled:
            return self._speak_text_sync(text, self.tts_config)
        
        try:
            item = TTSQueueItem(text, self.tts_config, priority)
            self._tts_queue.put((-priority, item), timeout=1)  # Negative for max heap
            return True
        except queue.Full:
            self.logger.warning("TTS queue is full, dropping message")
            return False
    
    def clear_queue(self) -> None:
        """Clear all pending TTS messages."""
        while not self._tts_queue.empty():
            try:
                self._tts_queue.get_nowait()
                self._tts_queue.task_done()
            except queue.Empty:
                break
    
    def is_available(self) -> bool:
        """Check if TTS handler is available."""
        if self._is_available is None:
            try:
                import pyttsx3
                
                # Try to initialize engine
                test_engine = self._get_tts_engine()
                self._is_available = test_engine is not None
                
            except ImportError:
                self.logger.error("pyttsx3 not available")
                self._is_available = False
            except Exception as e:
                self.logger.error(f"TTS not available: {e}")
                self._is_available = False
        
        return self._is_available
    
    def send_notification(self, config: NotificationConfig) -> bool:
        """Send notification via TTS."""
        if not self.validate_config(config):
            return False
        
        # Handle different TTS modes
        if self.tts_config.mode == TTSMode.DISABLED:
            return self._fallback_to_sound(config)
        
        # Prepare text to speak
        text_parts = []
        if config.title:
            text_parts.append(config.title)
        if config.message:
            text_parts.append(config.message)
        
        full_text = ". ".join(text_parts)
        
        # Determine priority based on notification level
        priority_map = {
            NotificationLevel.CRITICAL: 3,
            NotificationLevel.WARNING: 2,
            NotificationLevel.SUCCESS: 1,
            NotificationLevel.INFO: 0
        }
        priority = priority_map.get(config.level, 0)
        
        success = False
        
        # Try TTS first
        if self.tts_config.mode in [TTSMode.TTS_ONLY, TTSMode.TTS_WITH_FALLBACK]:
            success = self.speak_text_async(full_text, priority)
        
        # Fallback to sound if TTS failed and fallback is enabled
        if not success and self.tts_config.mode == TTSMode.TTS_WITH_FALLBACK:
            success = self._fallback_to_sound(config)
        
        # Sound only mode
        if self.tts_config.mode == TTSMode.SOUND_ONLY:
            success = self._fallback_to_sound(config)
        
        return success
    
    def _fallback_to_sound(self, config: NotificationConfig) -> bool:
        """Fallback to sound notification."""
        if config.sound_enabled:
            return self.play_sound(config.sound_path, config.level)
        else:
            # Final fallback to console
            self.log_fallback(config)
            return True
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Get TTS handler capabilities."""
        return {
            "actions": False,
            "sound": True,
            "icons": False,
            "persistence": False,
            "urgency": True,
            "tts": True,
            "queue": self.tts_config.queue_enabled,
            "preprocessing": self.tts_config.preprocessing_enabled
        }
    
    def get_available_voices(self) -> Dict[str, Dict[str, Any]]:
        """Get available TTS voices."""
        return self._platform_voices.copy()
    
    def set_voice(self, voice_id: str) -> bool:
        """Set TTS voice by ID."""
        try:
            engine = self._get_tts_engine()
            if not engine:
                return False
            
            voices = engine.getProperty('voices')
            for voice in voices or []:
                if voice.id == voice_id:
                    engine.setProperty('voice', voice_id)
                    self.tts_config.voice_id = voice_id
                    return True
            
            return False
        except Exception as e:
            self.logger.error(f"Failed to set voice: {e}")
            return False
    
    def set_rate(self, rate: int) -> bool:
        """Set TTS speech rate."""
        try:
            engine = self._get_tts_engine()
            if not engine:
                return False
            
            engine.setProperty('rate', rate)
            self.tts_config.rate = rate
            return True
        except Exception as e:
            self.logger.error(f"Failed to set rate: {e}")
            return False
    
    def set_volume(self, volume: float) -> bool:
        """Set TTS volume (0.0 to 1.0)."""
        try:
            volume = max(0.0, min(1.0, volume))  # Clamp to valid range
            engine = self._get_tts_engine()
            if not engine:
                return False
            
            engine.setProperty('volume', volume)
            self.tts_config.volume = volume
            return True
        except Exception as e:
            self.logger.error(f"Failed to set volume: {e}")
            return False
    
    def _play_custom_sound(self, sound_path: Path) -> bool:
        """Play custom sound file."""
        try:
            if self._platform == "Linux":
                import subprocess
                subprocess.run(["paplay", str(sound_path)], check=True, capture_output=True)
            elif self._platform == "Darwin":  # macOS
                import subprocess
                subprocess.run(["afplay", str(sound_path)], check=True, capture_output=True)
            elif self._platform == "Windows":
                import winsound
                winsound.PlaySound(str(sound_path), winsound.SND_FILENAME | winsound.SND_NOWAIT)
            else:
                return False
            return True
        except Exception as e:
            self.logger.debug(f"Failed to play custom sound: {e}")
            return False
    
    def _play_system_sound(self, level: NotificationLevel) -> bool:
        """Play system sound for notification level."""
        try:
            if self._platform == "Linux":
                import subprocess
                sound_map = {
                    NotificationLevel.INFO: "message-new-instant",
                    NotificationLevel.WARNING: "dialog-warning",
                    NotificationLevel.CRITICAL: "dialog-error",
                    NotificationLevel.SUCCESS: "complete"
                }
                sound_name = sound_map.get(level, "message-new-instant")
                subprocess.run(["paplay", f"/usr/share/sounds/freedesktop/stereo/{sound_name}.oga"],
                             check=True, capture_output=True)
            elif self._platform == "Darwin":  # macOS
                import subprocess
                sound_map = {
                    NotificationLevel.INFO: "Glass",
                    NotificationLevel.WARNING: "Ping",
                    NotificationLevel.CRITICAL: "Basso",
                    NotificationLevel.SUCCESS: "Blow"
                }
                sound_name = sound_map.get(level, "Glass")
                subprocess.run(["afplay", f"/System/Library/Sounds/{sound_name}.aiff"],
                             check=True, capture_output=True)
            elif self._platform == "Windows":
                import winsound
                sound_map = {
                    NotificationLevel.INFO: winsound.MB_OK,
                    NotificationLevel.WARNING: winsound.MB_ICONEXCLAMATION,
                    NotificationLevel.CRITICAL: winsound.MB_ICONHAND,
                    NotificationLevel.SUCCESS: winsound.MB_OK
                }
                sound_type = sound_map.get(level, winsound.MB_OK)
                winsound.MessageBeep(sound_type)
            else:
                return False
            return True
        except Exception as e:
            self.logger.debug(f"Failed to play system sound: {e}")
            return False