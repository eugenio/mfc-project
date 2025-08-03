"""
TTS notification handler with pyttsx3 support and ding sound fallback.
"""
import logging
import threading
import queue
from pathlib import Path
from typing import Optional, Dict, Any, List
from enum import Enum
import time

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

from .base import NotificationHandler, NotificationConfig, NotificationLevel, FallbackHandler

logger = logging.getLogger(__name__)
class TTSMode(Enum):
    """TTS operation modes."""
    TTS_ONLY = "tts_only"
    SOUND_ONLY = "sound_only"
    TTS_WITH_FALLBACK = "tts_with_fallback"
    DISABLED = "disabled"

class TTSEngine:
    """Wrapper for pyttsx3 engine with platform-specific configuration."""
    
    def __init__(self):
        self.engine = None
        self._lock = threading.Lock()
        self._is_initialized = False
        
    def initialize(self) -> bool:
        """Initialize TTS engine."""
        if not PYTTSX3_AVAILABLE:
            logger.warning("pyttsx3 not available")
            return False
            
        try:
            with self._lock:
                self.engine = pyttsx3.init()
                self._configure_engine()
                self._is_initialized = True
                return True
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            return False
            
    def _configure_engine(self):
        """Configure engine with default settings."""
        if not self.engine:
            return
            
        # Set default properties
        self.engine.setProperty('rate', 180)  # Words per minute
        self.engine.setProperty('volume', 0.8)  # 0-1 range
        
    def speak(self, text: str, wait: bool = True) -> bool:
        """Speak text using TTS engine."""
        if not self._is_initialized or not self.engine:
            return False
            
        try:
            with self._lock:
                self.engine.say(text)
                if wait:
                    self.engine.runAndWait()
                else:
                    self.engine.startLoop(False)
                    self.engine.iterate()
                    self.engine.endLoop()
                return True
        except Exception as e:
            logger.error(f"TTS speak failed: {e}")
            return False
            
    def stop(self):
        """Stop current speech."""
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass
                
    def set_voice(self, voice_id: Optional[str] = None):
        """Set TTS voice."""
        if not self.engine:
            return
            
        voices = self.engine.getProperty('voices')
        if voice_id and any(v.id == voice_id for v in voices):
            self.engine.setProperty('voice', voice_id)
        elif voices:
            # Use first available voice
            self.engine.setProperty('voice', voices[0].id)
            
    def set_rate(self, rate: int):
        """Set speech rate (words per minute)."""
        if self.engine:
            self.engine.setProperty('rate', rate)
            
    def set_volume(self, volume: float):
        """Set volume (0.0 to 1.0)."""
        if self.engine:
            self.engine.setProperty('volume', max(0.0, min(1.0, volume)))
            
    def get_voices(self) -> List[Dict[str, str]]:
        """Get available voices."""
        if not self.engine:
            return []
            
        voices = self.engine.getProperty('voices')
        return [{'id': v.id, 'name': getattr(v, 'name', v.id)} for v in voices]
        
    def cleanup(self):
        """Clean up TTS engine resources."""
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass
            self._is_initialized = False

class TTSNotificationHandler(FallbackHandler):
    """Notification handler with TTS support and ding sound fallback."""
    
    def __init__(self, mode: TTSMode = TTSMode.TTS_WITH_FALLBACK):
        super().__init__()
        self.mode = mode
        self.tts_engine = TTSEngine()
        self.tts_available = False
        self._notification_queue = queue.Queue()
        self._worker_thread = None
        self._stop_event = threading.Event()
        
        # TTS configuration
        self.tts_config = {
            'rate': 180,
            'volume': 0.8,
            'voice_id': None,
            'text_limit': 500,  # Max characters for TTS
        }
        
        # Initialize components
        self._initialize_tts()
        if self.mode != TTSMode.DISABLED:
            self._start_worker()
            
    def _initialize_tts(self):
        """Initialize TTS engine if mode requires it."""
        if self.mode in [TTSMode.TTS_ONLY, TTSMode.TTS_WITH_FALLBACK]:
            self.tts_available = self.tts_engine.initialize()
            if self.tts_available:
                self.tts_engine.set_rate(self.tts_config['rate'])
                self.tts_engine.set_volume(self.tts_config['volume'])
                if self.tts_config['voice_id']:
                    self.tts_engine.set_voice(self.tts_config['voice_id'])
                logger.info("TTS engine initialized successfully")
            else:
                logger.warning("TTS engine initialization failed")
                if self.mode == TTSMode.TTS_ONLY:
                    logger.error("TTS_ONLY mode but TTS unavailable")
                    
    def _start_worker(self):
        """Start background worker thread for processing notifications."""
        self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._worker_thread.start()
        
    def _process_queue(self):
        """Process notification queue in background thread."""
        while not self._stop_event.is_set():
            try:
                # Wait for notification with timeout
                notification = self._notification_queue.get(timeout=0.5)
                self._process_notification(notification)
                self._notification_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
                
    def _process_notification(self, config: NotificationConfig):
        """Process a single notification."""
        # Determine what to speak
        speak_text = self._prepare_speech_text(config)
        
        # Handle based on mode
        if self.mode == TTSMode.TTS_ONLY:
            if self.tts_available:
                self._speak_notification(speak_text, config.level)
            else:
                logger.error("TTS unavailable in TTS_ONLY mode")
                
        elif self.mode == TTSMode.SOUND_ONLY:
            self._play_notification_sound(config)
            
        elif self.mode == TTSMode.TTS_WITH_FALLBACK:
            if self.tts_available:
                success = self._speak_notification(speak_text, config.level)
                if not success:
                    self._play_notification_sound(config)
            else:
                self._play_notification_sound(config)
                
    def _prepare_speech_text(self, config: NotificationConfig) -> str:
        """Prepare text for speech synthesis."""
        # Combine title and message
        text = f"{config.title}. {config.message}"
        
        # Truncate if too long
        if len(text) > self.tts_config['text_limit']:
            text = text[:self.tts_config['text_limit']] + "..."
            
        # Clean text for speech
        text = self._clean_text_for_speech(text)
        
        return text
        
    def _clean_text_for_speech(self, text: str) -> str:
        """Clean text for better speech synthesis."""
        # Replace common abbreviations
        replacements = {
            'MFC': 'M F C',
            'CPU': 'C P U',
            'GPU': 'G P U',
            'API': 'A P I',
            'RAM': 'ram',
            'pH': 'P H',
        }
        
        for abbr, spoken in replacements.items():
            text = text.replace(abbr, spoken)
            
        return text
        
    def _speak_notification(self, text: str, level: NotificationLevel) -> bool:
        """Speak notification text with level-appropriate settings."""
        if not self.tts_available or not text:
            return False
            
        try:
            # Adjust speech parameters based on level
            original_rate = self.tts_config['rate']
            original_volume = self.tts_config['volume']
            
            if level == NotificationLevel.CRITICAL:
                self.tts_engine.set_rate(int(original_rate * 1.2))
                self.tts_engine.set_volume(min(1.0, original_volume * 1.2))
            elif level == NotificationLevel.WARNING:
                self.tts_engine.set_rate(int(original_rate * 1.1))
                self.tts_engine.set_volume(min(1.0, original_volume * 1.1))
                
            # Speak the text
            success = self.tts_engine.speak(text)
            
            # Restore original settings
            self.tts_engine.set_rate(original_rate)
            self.tts_engine.set_volume(original_volume)
            
            return success
            
        except Exception as e:
            logger.error(f"TTS speak error: {e}")
            return False
            
    def _play_notification_sound(self, config: NotificationConfig):
        """Play notification sound as fallback or primary method."""
        # Use parent class sound playing capability
        if config.sound_enabled:
            self.play_sound(config.sound_path, config.level)
            
    def is_available(self) -> bool:
        """Check if handler is available."""
        if self.mode == TTSMode.DISABLED:
            return False
        elif self.mode == TTSMode.TTS_ONLY:
            return self.tts_available
        else:
            return True  # Sound fallback always available
            
    def send_notification(self, config: NotificationConfig) -> bool:
        """Send notification with TTS or sound."""
        if self.mode == TTSMode.DISABLED:
            return False
            
        if not self.validate_config(config):
            return False
            
        try:
            # Add to queue for processing
            self._notification_queue.put(config)
            return True
        except Exception as e:
            logger.error(f"Failed to queue notification: {e}")
            return False
            
    def get_capabilities(self) -> Dict[str, bool]:
        """Get handler capabilities."""
        return {
            "actions": False,
            "sound": True,
            "icons": False,
            "persistence": False,
            "urgency": False,
            "tts": self.tts_available,
            "voice_selection": self.tts_available and bool(self.tts_engine.get_voices()),
        }
        
    def configure_tts(self, **kwargs):
        """Configure TTS settings."""
        if 'rate' in kwargs:
            self.tts_config['rate'] = kwargs['rate']
            if self.tts_engine:
                self.tts_engine.set_rate(kwargs['rate'])
                
        if 'volume' in kwargs:
            self.tts_config['volume'] = kwargs['volume']
            if self.tts_engine:
                self.tts_engine.set_volume(kwargs['volume'])
                
        if 'voice_id' in kwargs:
            self.tts_config['voice_id'] = kwargs['voice_id']
            if self.tts_engine:
                self.tts_engine.set_voice(kwargs['voice_id'])
                
        if 'text_limit' in kwargs:
            self.tts_config['text_limit'] = kwargs['text_limit']
            
    def get_voices(self) -> List[Dict[str, str]]:
        """Get available TTS voices."""
        if self.tts_available:
            return self.tts_engine.get_voices()
        return []
        
    def set_mode(self, mode: TTSMode):
        """Change TTS mode."""
        self.mode = mode
        if mode in [TTSMode.TTS_ONLY, TTSMode.TTS_WITH_FALLBACK] and not self.tts_available:
            self._initialize_tts()
            
    def cleanup(self):
        """Clean up resources."""
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=2)
        if self.tts_engine:
            self.tts_engine.cleanup()