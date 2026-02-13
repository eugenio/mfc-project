# TTS Engine Configuration Guide

## Overview

The MFC notification system now supports multiple Text-to-Speech (TTS) engines:

1. **pyttsx3** - Lightweight, cross-platform TTS (default)
2. **Coqui TTS** - Advanced neural network-based TTS with high-quality voices
3. **Hybrid** - Automatically selects the best available engine

## Configuration Options

### 1. Engine Selection

When creating a `TTSNotificationHandler`, you can specify the engine type:

```python
from notifications.tts_handler import TTSNotificationHandler, TTSMode, TTSEngineType

# Use pyttsx3 (default, lightweight)
handler = TTSNotificationHandler(
    mode=TTSMode.TTS_WITH_FALLBACK,
    engine_type=TTSEngineType.PYTTSX3
)

# Use Coqui TTS (high quality, requires installation)
handler = TTSNotificationHandler(
    mode=TTSMode.TTS_WITH_FALLBACK,
    engine_type=TTSEngineType.COQUI
)

# Use Hybrid (automatic selection)
handler = TTSNotificationHandler(
    mode=TTSMode.TTS_WITH_FALLBACK,
    engine_type=TTSEngineType.HYBRID
)
```

### 2. Environment Variables

The hybrid engine respects the following environment variable:

- `TTS_PREFER_COQUI` - Set to `true` to prefer Coqui TTS when available (default: `true`)

```bash
# Prefer Coqui TTS
export TTS_PREFER_COQUI=true

# Prefer pyttsx3
export TTS_PREFER_COQUI=false
```

### 3. Coqui TTS Configuration

For advanced Coqui TTS configuration:

```python
from notifications.coqui_tts_manager import CoquiTTSConfig
from notifications.tts_handler import TTSNotificationHandler, TTSEngineType

# Custom Coqui configuration
coqui_config = CoquiTTSConfig(
    model_name="tts_models/en/ljspeech/tacotron2-DDC",  # Specific model
    language="en",
    speed=1.2,  # Faster speech
    temperature=0.8,  # More natural variation
    use_gpu=True  # Enable GPU acceleration
)

handler = TTSNotificationHandler(
    engine_type=TTSEngineType.COQUI,
    coqui_config=coqui_config
)
```

## Installation

### Basic Installation (pyttsx3 only)

The default installation includes pyttsx3:

```bash
pixi install
```

### Advanced Installation (with Coqui TTS)

To use Coqui TTS, install with the advanced-tts feature:

```bash
pixi install -e tts-dev
```

Or add to an existing environment:

```bash
pixi add --feature advanced-tts TTS librosa
```

## Engine Comparison

| Feature | pyttsx3 | Coqui TTS |
|---------|---------|-----------|
| **Quality** | Good | Excellent |
| **Speed** | Fast | Slower (neural network) |
| **Installation** | Easy | Complex |
| **Dependencies** | Minimal | Heavy (PyTorch) |
| **GPU Support** | No | Yes |
| **Voice Options** | System voices | Many models |
| **Offline** | Yes | Yes |
| **Memory Usage** | Low | High |

## Usage Examples

### Simple Notification

```python
from notifications.tts_handler import TTSNotificationHandler
from notifications.base import NotificationConfig, NotificationLevel

# Create handler (uses hybrid by default)
handler = TTSNotificationHandler()

# Send notification
config = NotificationConfig(
    title="Task Complete",
    message="The simulation has finished successfully",
    level=NotificationLevel.SUCCESS
)
handler.send_notification(config)
```

### Advanced Usage with Voice Selection

```python
# Coqui TTS with specific voice
from notifications.coqui_tts_manager import CoquiTTSManager, CoquiTTSConfig

config = CoquiTTSConfig(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    language="es",  # Spanish
    speaker_name="Esperanza"  # Specific speaker
)

manager = CoquiTTSManager(config)
manager.speak("La simulación ha terminado", NotificationLevel.SUCCESS)
```

### Testing Available Engines

```bash
# Test pyttsx3
pixi run tts-test-pyttsx3

# Test Coqui TTS (requires advanced-tts feature)
pixi run -e tts-dev tts-test-coqui

# Test hybrid engine
pixi run tts-test-hybrid

# Get Coqui TTS info
pixi run -e tts-dev tts-coqui-info
```

## Troubleshooting

### Coqui TTS Not Available

If you see "Coqui TTS not available" warnings:

1. Install the advanced-tts feature: `pixi install -e tts-dev`
2. Check GPU drivers if using GPU acceleration
3. Verify Python version compatibility (3.8-3.11 recommended)

### Voice Not Found

For pyttsx3:
- List available voices: `pixi run tts-list-voices`
- System voices depend on OS (espeak on Linux, SAPI on Windows, NSSpeechSynthesizer on macOS)

For Coqui TTS:
- Check available models at: https://github.com/coqui-ai/TTS#released-models
- Some models require specific languages or speakers

### Performance Issues

- Use pyttsx3 for faster response times
- Disable GPU for Coqui TTS if experiencing issues: `use_gpu=False`
- Reduce text length for faster synthesis
- Use the hybrid engine to automatically select the best option

## Contributing

To add support for additional TTS engines:

1. Create a new manager class following the pattern in `coqui_tts_manager.py`
2. Add the engine type to `TTSEngineType` enum
3. Update `_initialize_tts_engine()` in `tts_handler.py`
4. Add tests and documentation
5. Submit a merge request

## Future Enhancements

- Voice cloning support (already implemented in CoquiTTSManager)
- SSML (Speech Synthesis Markup Language) support
- Real-time voice conversion
- Custom voice training integration
- Cloud TTS service integration (Amazon Polly, Google Cloud TTS, etc.)