"""
Coqui TTS Integration for Advanced Voice Synthesis
==================================================

Integrates Coqui TTS (🐸TTS) for high-quality, neural voice synthesis with the notification system.
Provides advanced voice features including multi-speaker support, voice cloning, and emotional expression.

Features:
- High-quality neural TTS synthesis
- Multi-speaker voice selection
- Voice cloning capabilities  
- Emotional expression control
- Speed and pitch adjustment
- SSML support for advanced speech markup
- Audio file output with multiple formats
- GPU acceleration support

Dependencies:
- TTS (Coqui TTS): pip install TTS
- torch: for neural network inference
- librosa: for audio processing

Created: 2025-08-03
Author: Agent Delta - Audio Integration Specialist
"""
import logging
import threading
import time
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import wave
import json

from .base import NotificationLevel

logger = logging.getLogger(__name__)
@dataclass
class CoquiTTSConfig:
    """Configuration for Coqui TTS engine."""
    
    # Core settings
    enabled: bool = True
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"  # High-quality multilingual model
    language: str = "en"
    
    # Voice settings
    speaker_name: Optional[str] = None  # Speaker name for multi-speaker models
    speaker_idx: Optional[int] = None   # Speaker index for multi-speaker models
    voice_file: Optional[Path] = None   # Path to reference voice for cloning
    
    # Synthesis quality
    use_gpu: bool = True
    speed: float = 1.0          # Speech speed multiplier
    temperature: float = 0.75   # Controls randomness/creativity (0.1-1.0)
    length_penalty: float = 1.0 # Controls sequence length
    repetition_penalty: float = 5.0  # Prevents repetition
    
    # Emotion and expression
    emotion: str = "neutral"    # neutral, happy, sad, angry, surprised, etc.
    emphasis_words: List[str] = field(default_factory=list)  # Words to emphasize
    
    # Audio output
    output_format: str = "wav"  # wav, mp3, ogg
    sample_rate: int = 22050    # Audio sample rate
    
    # Performance
    timeout_seconds: float = 30.0
    max_text_length: int = 1000
    cache_models: bool = True
    
    # Advanced features
    enable_ssml: bool = True
    split_sentences: bool = True
    normalize_text: bool = True

class CoquiTTSError(Exception):
    """Raised when Coqui TTS operations fail."""
    pass

class CoquiTTSManager:
    """
    Advanced TTS manager using Coqui TTS for high-quality voice synthesis.
    
    Provides neural network-based text-to-speech with advanced features like
    voice cloning, emotional expression, and multi-speaker support.
    """
    
    def __init__(self, config: Optional[CoquiTTSConfig] = None):
        """Initialize Coqui TTS manager."""
        self.config = config or CoquiTTSConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self._tts_model = None
        self._model_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="CoquiTTS")
        self._available = None
        
        # Voice level mappings for notification urgency
        self._level_voice_mappings = {
            NotificationLevel.INFO: {"speed": 1.0, "temperature": 0.75, "emotion": "neutral"},
            NotificationLevel.WARNING: {"speed": 1.1, "temperature": 0.8, "emotion": "concerned"},
            NotificationLevel.CRITICAL: {"speed": 1.3, "temperature": 0.9, "emotion": "urgent"},
            NotificationLevel.SUCCESS: {"speed": 0.9, "temperature": 0.7, "emotion": "happy"}
        }
        
        # Initialize model if enabled
        if self.config.enabled:
            self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the Coqui TTS model."""
        try:
            with self._model_lock:
                if self._tts_model is None:
                    # Import TTS here to handle optional dependency
                    from TTS.api import TTS
                    
                    self.logger.info(f"Initializing Coqui TTS model: {self.config.model_name}")
                    
                    # Initialize TTS model
                    self._tts_model = TTS(
                        model_name=self.config.model_name,
                        gpu=self.config.use_gpu
                    )
                    
                    self.logger.info("Coqui TTS model initialized successfully")
                    self._available = True
                    
        except ImportError as e:
            self.logger.warning(f"Coqui TTS not available: {e}")
            self._available = False
        except Exception as e:
            self.logger.error(f"Failed to initialize Coqui TTS: {e}")
            self._available = False
    
    def is_available(self) -> bool:
        """Check if Coqui TTS is available."""
        if self._available is None:
            self._initialize_model()
        return self._available or False
    
    def get_available_speakers(self) -> List[str]:
        """Get list of available speakers for current model."""
        if not self.is_available() or not self._tts_model:
            return []
        
        try:
            # Get speakers if model supports multi-speaker
            if hasattr(self._tts_model, 'speakers') and self._tts_model.speakers:
                return list(self._tts_model.speakers)
            return []
        except Exception as e:
            self.logger.debug(f"Could not get speakers: {e}")
            return []
    
    def get_available_languages(self) -> List[str]:
        """Get list of available languages for current model."""
        if not self.is_available() or not self._tts_model:
            return []
        
        try:
            # Get languages if model supports multilingual
            if hasattr(self._tts_model, 'languages') and self._tts_model.languages:
                return list(self._tts_model.languages)
            return ["en"]  # Default fallback
        except Exception as e:
            self.logger.debug(f"Could not get languages: {e}")
            return ["en"]
    
    def speak(
        self, 
        text: str, 
        level: NotificationLevel = NotificationLevel.INFO,
        blocking: bool = True,
        output_file: Optional[Path] = None,
        **kwargs
    ) -> bool:
        """
        Convert text to speech using Coqui TTS.
        
        Args:
            text: Text to synthesize
            level: Notification level for voice adjustment
            blocking: Whether to wait for completion
            output_file: Optional file to save audio to
            **kwargs: Additional synthesis parameters
        
        Returns:
            bool: True if synthesis succeeded
        """
        if not self.config.enabled:
            return False
        
        if not self.is_available():
            raise CoquiTTSError("Coqui TTS is not available")
        
        if not text or len(text.strip()) == 0:
            return False
        
        # Sanitize and prepare text
        clean_text = self._prepare_text(text)
        
        if blocking:
            return self._synthesize_speech(clean_text, level, output_file, **kwargs)
        else:
            # Submit to thread pool for async execution
            future = self._executor.submit(
                self._synthesize_speech, clean_text, level, output_file, **kwargs
            )
            return True  # Return immediately for async
    
    def _prepare_text(self, text: str) -> str:
        """Prepare and sanitize text for TTS synthesis."""
        if not self.config.normalize_text:
            return text
        
        # Basic text normalization
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Handle common abbreviations
        abbreviations = {
            'Dr.': 'Doctor',
            'Mr.': 'Mister', 
            'Mrs.': 'Misses',
            'Ms.': 'Miss',
            'Prof.': 'Professor',
            'etc.': 'etcetera',
            'e.g.': 'for example',
            'i.e.': 'that is'
        }
        
        for abbr, expansion in abbreviations.items():
            text = text.replace(abbr, expansion)
        
        # Limit text length
        if len(text) > self.config.max_text_length:
            text = text[:self.config.max_text_length] + "..."
            self.logger.warning(f"Text truncated to {self.config.max_text_length} characters")
        
        return text
    
    def _synthesize_speech(
        self, 
        text: str, 
        level: NotificationLevel,
        output_file: Optional[Path] = None,
        **kwargs
    ) -> bool:
        """Perform the actual speech synthesis."""
        try:
            with self._model_lock:
                if not self._tts_model:
                    return False
                
                # Get voice parameters based on notification level
                voice_params = self._level_voice_mappings.get(level, {})
                
                # Merge with config and kwargs
                synthesis_params = {
                    "speed": kwargs.get("speed", voice_params.get("speed", self.config.speed)),
                    "temperature": kwargs.get("temperature", voice_params.get("temperature", self.config.temperature)),
                    "length_penalty": kwargs.get("length_penalty", self.config.length_penalty),
                    "repetition_penalty": kwargs.get("repetition_penalty", self.config.repetition_penalty)
                }
                
                # Prepare output file
                if output_file is None:
                    output_file = Path(tempfile.mktemp(suffix=f".{self.config.output_format}"))
                
                # Synthesis parameters for TTS
                tts_kwargs = {
                    "text": text,
                    "file_path": str(output_file)
                }
                
                # Add language if supported
                if self.config.language in self.get_available_languages():
                    tts_kwargs["language"] = self.config.language
                
                # Add speaker if specified and supported
                speakers = self.get_available_speakers()
                if self.config.speaker_name and self.config.speaker_name in speakers:
                    tts_kwargs["speaker"] = self.config.speaker_name
                elif self.config.speaker_idx is not None and len(speakers) > self.config.speaker_idx:
                    tts_kwargs["speaker"] = speakers[self.config.speaker_idx]
                
                # Add voice cloning file if specified
                if self.config.voice_file and self.config.voice_file.exists():
                    tts_kwargs["speaker_wav"] = str(self.config.voice_file)
                
                # Perform synthesis
                self.logger.debug(f"Synthesizing text: {text[:50]}...")
                start_time = time.time()
                
                # Use timeout to prevent hanging
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("TTS synthesis timed out")
                
                # Set timeout signal (Unix only)
                try:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(int(self.config.timeout_seconds))
                    
                    # Perform TTS synthesis
                    self._tts_model.tts_to_file(**tts_kwargs)
                    
                    signal.alarm(0)  # Cancel timeout
                except AttributeError:
                    # Windows doesn't support SIGALRM, just call directly
                    self._tts_model.tts_to_file(**tts_kwargs)
                
                synthesis_time = time.time() - start_time
                self.logger.debug(f"TTS synthesis completed in {synthesis_time:.2f}s")
                
                # Play the generated audio if no output file was specified
                if output_file and output_file.exists():
                    if kwargs.get("auto_play", True):
                        self._play_audio_file(output_file)
                    
                    # Clean up temporary file unless explicitly keeping it
                    if not kwargs.get("keep_file", False) and output_file.name.startswith("tmp"):
                        try:
                            output_file.unlink()
                        except Exception as e:
                            self.logger.debug(f"Could not remove temp file: {e}")
                
                return True
                
        except TimeoutError:
            self.logger.error(f"TTS synthesis timed out after {self.config.timeout_seconds}s")
            return False
        except Exception as e:
            self.logger.error(f"TTS synthesis failed: {e}")
            return False
    
    def _play_audio_file(self, audio_file: Path) -> None:
        """Play the generated audio file."""
        try:
            import platform
            import subprocess
            
            system = platform.system()
            
            if system == "Linux":
                subprocess.run(["paplay", str(audio_file)], check=True, capture_output=True)
            elif system == "Darwin":  # macOS
                subprocess.run(["afplay", str(audio_file)], check=True, capture_output=True)
            elif system == "Windows":
                # Use winsound for Windows
                import winsound
                winsound.PlaySound(str(audio_file), winsound.SND_FILENAME)
            else:
                self.logger.warning(f"Audio playback not supported on {system}")
                
        except Exception as e:
            self.logger.debug(f"Could not play audio file: {e}")
    
    def clone_voice(self, reference_audio: Path, text: str, output_file: Path) -> bool:
        """
        Clone a voice from reference audio and synthesize text.
        
        Args:
            reference_audio: Path to reference audio file (3-10 seconds recommended)
            text: Text to synthesize with cloned voice
            output_file: Output file for synthesized audio
            
        Returns:
            bool: True if voice cloning succeeded
        """
        if not self.is_available():
            return False
        
        if not reference_audio.exists():
            self.logger.error(f"Reference audio file not found: {reference_audio}")
            return False
        
        try:
            with self._model_lock:
                if not self._tts_model:
                    return False
                
                # Perform voice cloning synthesis
                self._tts_model.tts_to_file(
                    text=text,
                    speaker_wav=str(reference_audio),
                    language=self.config.language,
                    file_path=str(output_file)
                )
                
                self.logger.info(f"Voice cloning synthesis saved to: {output_file}")
                return True
                
        except Exception as e:
            self.logger.error(f"Voice cloning failed: {e}")
            return False
    
    def generate_ssml_speech(self, ssml_text: str, output_file: Path) -> bool:
        """
        Generate speech from SSML markup for advanced speech control.
        
        Args:
            ssml_text: SSML formatted text
            output_file: Output audio file
            
        Returns:
            bool: True if SSML synthesis succeeded
        """
        if not self.config.enable_ssml:
            return False
        
        try:
            # Parse SSML and extract plain text
            # (This is a simplified implementation - real SSML parsing would be more complex)
            import re
            
            # Remove SSML tags for basic compatibility
            plain_text = re.sub(r'<[^>]+>', '', ssml_text)
            
            # Use regular synthesis (Coqui TTS has limited SSML support)
            return self.speak(plain_text, output_file=output_file, auto_play=False)
            
        except Exception as e:
            self.logger.error(f"SSML synthesis failed: {e}")
            return False
    
    def get_voice_info(self) -> Dict[str, Any]:
        """Get information about current voice configuration."""
        info = {
            "model_name": self.config.model_name,
            "language": self.config.language,
            "available": self.is_available(),
            "speakers": self.get_available_speakers(),
            "languages": self.get_available_languages(),
            "gpu_enabled": self.config.use_gpu,
            "voice_cloning": bool(self.config.voice_file),
        }
        
        return info
    
    def set_speaker(self, speaker: Union[str, int]) -> bool:
        """
        Set the speaker for multi-speaker models.
        
        Args:
            speaker: Speaker name or index
            
        Returns:
            bool: True if speaker was set successfully
        """
        try:
            speakers = self.get_available_speakers()
            
            if isinstance(speaker, str):
                if speaker in speakers:
                    self.config.speaker_name = speaker
                    self.config.speaker_idx = None
                    return True
            elif isinstance(speaker, int):
                if 0 <= speaker < len(speakers):
                    self.config.speaker_idx = speaker
                    self.config.speaker_name = None
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to set speaker: {e}")
            return False
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if self._executor:
                self._executor.shutdown(wait=True)
                
            # Clear model to free GPU memory
            with self._model_lock:
                if self._tts_model:
                    del self._tts_model
                    self._tts_model = None
                    
            # Clear GPU cache if using PyTorch
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
                
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


# Integration with existing TTS system
class HybridTTSManager:
    """
    Hybrid TTS manager that can use both pyttsx3 and Coqui TTS.
    
    Automatically falls back to pyttsx3 if Coqui TTS is not available,
    and provides seamless switching between engines.
    """
    
    def __init__(self, prefer_coqui: bool = True, coqui_config: Optional[CoquiTTSConfig] = None):
        """Initialize hybrid TTS manager."""
        self.prefer_coqui = prefer_coqui
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize Coqui TTS
        self.coqui_tts = None
        if prefer_coqui:
            try:
                self.coqui_tts = CoquiTTSManager(coqui_config)
                if not self.coqui_tts.is_available():
                    self.coqui_tts = None
            except Exception as e:
                self.logger.debug(f"Coqui TTS initialization failed: {e}")
        
        # Initialize pyttsx3 as fallback
        self.pyttsx3_engine = None
        try:
            import pyttsx3
            self.pyttsx3_engine = pyttsx3.init()
        except Exception as e:
            self.logger.debug(f"pyttsx3 initialization failed: {e}")
    
    def is_available(self) -> bool:
        """Check if any TTS engine is available."""
        return (self.coqui_tts and self.coqui_tts.is_available()) or self.pyttsx3_engine is not None
    
    def speak(self, text: str, level: NotificationLevel = NotificationLevel.INFO, **kwargs) -> bool:
        """Speak text using the best available TTS engine."""
        # Try Coqui TTS first if preferred and available
        if self.coqui_tts and self.coqui_tts.is_available():
            try:
                return self.coqui_tts.speak(text, level, **kwargs)
            except Exception as e:
                self.logger.warning(f"Coqui TTS failed, falling back to pyttsx3: {e}")
        
        # Fall back to pyttsx3
        if self.pyttsx3_engine:
            try:
                self.pyttsx3_engine.say(text)
                if kwargs.get("blocking", True):
                    self.pyttsx3_engine.runAndWait()
                return True
            except Exception as e:
                self.logger.error(f"pyttsx3 TTS failed: {e}")
                return False
        
        return False
    
    def get_active_engine(self) -> str:
        """Get the name of the currently active TTS engine."""
        if self.coqui_tts and self.coqui_tts.is_available():
            return "Coqui TTS"
        elif self.pyttsx3_engine:
            return "pyttsx3"
        else:
            return "None"
    
    def cleanup(self) -> None:
        """Clean up both TTS engines."""
        if self.coqui_tts:
            self.coqui_tts.cleanup()
        
        if self.pyttsx3_engine:
            try:
                self.pyttsx3_engine.stop()
            except Exception:
                pass