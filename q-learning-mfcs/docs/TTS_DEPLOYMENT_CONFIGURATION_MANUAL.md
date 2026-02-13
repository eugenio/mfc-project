# TTS Deployment and Configuration Manual
## Table of Contents

1. [Installation Guide](#installation-guide)
## Overview

This manual provides comprehensive instructions for deploying and configuring the TTS (Text-to-Speech) notification system across different environments. It covers installation, configuration, optimization, and maintenance procedures for production, development, and testing environments.
## Installation Guide

### Prerequisites

#### System Requirements

**Minimum Requirements:**
- Python 3.8+
- 4GB RAM
- 1GB free disk space
- Audio output device

**Recommended Requirements:**
- Python 3.11+
- 8GB RAM (16GB for Coqui TTS with GPU)
- 5GB free disk space
- GPU with 4GB+ VRAM (for Coqui TTS)
- High-quality audio output

#### Platform-Specific Prerequisites

**Linux (Ubuntu/Debian):**
```bash
# Install system dependencies
sudo apt update
sudo apt install -y \
    python3-dev \
    python3-pip \
    espeak \
    espeak-data \
    libespeak1 \
    libespeak-dev \
    festival \
    pulseaudio \
    alsa-utils \
    portaudio19-dev \
    build-essential

# For Coqui TTS with GPU support
sudo apt install -y \
    nvidia-driver-470 \
    nvidia-cuda-toolkit \
    libcudnn8 \
    libcudnn8-dev
```

**CentOS/RHEL/Fedora:**
```bash
# Install system dependencies
sudo dnf install -y \
    python3-devel \
    python3-pip \
    espeak \
    festival \
    pulseaudio \
    alsa-lib-devel \
    portaudio-devel \
    gcc \
    gcc-c++

# For GPU support
sudo dnf install -y \
    nvidia-driver \
    cuda \
    cudnn
```

**macOS:**
```bash
# Install via Homebrew
brew install portaudio
brew install espeak

# For development
xcode-select --install
```

**Windows:**
```powershell
# Install chocolatey packages
choco install python3
choco install git
choco install microsoft-visual-cpp-build-tools

# SAPI TTS is built-in on Windows
```

### Basic Installation

#### Option 1: Pixi Environment (Recommended)

```bash
# Clone repository
git clone <repository-url>
cd mfc-project

# Install basic TTS support
pixi install

# Test basic installation
pixi run tts-test-simple
```

#### Option 2: Advanced Installation with Coqui TTS

```bash
# Install with advanced TTS features
pixi install -e tts-dev

# Verify Coqui TTS installation
pixi run -e tts-dev tts-coqui-info

# Test Coqui TTS
pixi run -e tts-dev tts-test-coqui
```

#### Option 3: Manual Installation

```bash
# Create virtual environment
python -m venv tts_env
source tts_env/bin/activate  # Linux/macOS
# or
tts_env\Scripts\activate     # Windows

# Install basic dependencies
pip install pyttsx3 pygame numpy

# Install advanced dependencies (optional)
pip install TTS torch librosa soundfile

# Install project
pip install -e .
```

### Installation Verification

```bash
# Verify pyttsx3 installation
python -c "import pyttsx3; print('pyttsx3 OK')"

# Verify Coqui TTS installation (if installed)
python -c "from TTS.api import TTS; print('Coqui TTS OK')"

# Test audio system
python -c "
import pygame
pygame.mixer.init()
print('Audio system OK')
"

# Run comprehensive test
pixi run tts-test-all-modes
```
## Environment Configuration

### Development Environment

#### Configuration File: `config/dev_tts.yaml`

```yaml
# Development TTS Configuration
tts:
  mode: "tts_with_fallback"
  engine: "pyttsx3"  # Fast startup for development
  
  pyttsx3:
    rate: 200
    volume: 0.7
    voice_id: null  # Use system default
    
  # Coqui TTS disabled in dev for faster startup
  coqui:
    enabled: false
    
  # Development-specific settings
  debug_mode: true
  log_level: "DEBUG"
  timeout_seconds: 10.0
  
  # Performance settings for development
  text_limit: 200
  queue_size: 50
  worker_threads: 1
```

#### Environment Variables

```bash
# .env.development
export TTS_MODE="tts_with_fallback"
export TTS_ENGINE="pyttsx3"
export TTS_DEBUG="true"
export TTS_LOG_LEVEL="DEBUG"
export TTS_RATE="200"
export TTS_VOLUME="0.7"
export TTS_TIMEOUT="10"
```

#### Development Setup Script

```bash
#!/bin/bash
# setup_dev_tts.sh

echo "Setting up TTS for development environment..."

# Load development environment
export TTS_MODE="tts_with_fallback"
export TTS_ENGINE="pyttsx3"
export TTS_DEBUG="true"

# Install development dependencies
pixi install

# Configure development audio (Linux)
if command -v pactl &> /dev/null; then
    # Ensure PulseAudio is running
    pulseaudio --start --log-target=journal
    
    # Set reasonable volume
    pactl set-sink-volume @DEFAULT_SINK@ 50%
fi

# Test installation
echo "Testing TTS installation..."
pixi run tts-test-simple

echo "Development TTS setup complete!"
```

### Production Environment

#### Configuration File: `config/prod_tts.yaml`

```yaml
# Production TTS Configuration
tts:
  mode: "tts_with_fallback"
  engine: "hybrid"  # Intelligent selection
  
  pyttsx3:
    rate: 180
    volume: 0.8
    voice_id: "english-us-female"
    
  coqui:
    enabled: true
    model_name: "tts_models/multilingual/multi-dataset/xtts_v2"
    language: "en"
    use_gpu: true
    cache_models: true
    timeout_seconds: 30.0
    
    # Production quality settings
    speed: 1.0
    temperature: 0.75
    length_penalty: 1.0
    repetition_penalty: 5.0
    
    # Output settings
    output_format: "wav"
    sample_rate: 22050
    
  # Production performance settings
  debug_mode: false
  log_level: "INFO"
  text_limit: 500
  queue_size: 200
  worker_threads: 4
  
  # Monitoring
  enable_metrics: true
  metrics_port: 9090
```

#### Environment Variables

```bash
# .env.production
export TTS_MODE="tts_with_fallback"
export TTS_ENGINE="hybrid"
export TTS_DEBUG="false"
export TTS_LOG_LEVEL="INFO"
export TTS_PREFER_COQUI="true"
export TTS_USE_GPU="true"
export TTS_CACHE_MODELS="true"
export TTS_ENABLE_METRICS="true"
```

#### Production Deployment Script

```bash
#!/bin/bash
# deploy_prod_tts.sh

echo "Deploying TTS for production environment..."

# Set production environment
export NODE_ENV="production"
export TTS_MODE="tts_with_fallback"
export TTS_ENGINE="hybrid"
export TTS_DEBUG="false"

# Install production dependencies
pixi install -e tts-dev

# Configure system audio (Linux)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Install and configure audio system
    sudo systemctl enable pulseaudio
    sudo systemctl start pulseaudio
    
    # Configure audio device
    sudo usermod -a -G audio $USER
    
    # Set audio levels
    amixer set Master 80%
fi

# Test GPU availability for Coqui TTS
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected for Coqui TTS"
    export TTS_USE_GPU="true"
else
    echo "No GPU detected, using CPU for Coqui TTS"
    export TTS_USE_GPU="false"
fi

# Pre-load models
echo "Pre-loading TTS models..."
python -c "
from notifications.coqui_tts_manager import CoquiTTSManager, CoquiTTSConfig
config = CoquiTTSConfig(cache_models=True)
manager = CoquiTTSManager(config)
print('Models pre-loaded successfully')
"

# Health check
echo "Running production health checks..."
pixi run tts-test-hybrid

echo "Production TTS deployment complete!"
```

### CI/CD Environment

#### Configuration File: `config/ci_tts.yaml`

```yaml
# CI/CD TTS Configuration
tts:
  mode: "disabled"  # No audio in CI
  engine: "mock"    # Mock engine for testing
  
  # Test-specific settings
  debug_mode: true
  log_level: "DEBUG"
  enable_mocks: true
  
  # CI performance settings
  timeout_seconds: 5.0
  text_limit: 100
  queue_size: 10
  worker_threads: 1
```

#### CI/CD Pipeline Configuration

```yaml
# .github/workflows/tts-ci.yml
name: TTS System CI

on: [push, pull_request]

jobs:
  test-tts:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install system dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y espeak espeak-data libespeak1
        # Install virtual audio for testing
        sudo apt-get install -y pulseaudio pavucontrol
        
    - name: Setup audio environment
      run: |
        # Start PulseAudio in system mode for CI
        pulseaudio --start --system --disallow-exit
        
    - name: Install TTS dependencies
      run: |
        pip install pixi
        pixi install
        
    - name: Run TTS tests
      env:
        TTS_MODE: "disabled"
        TTS_DEBUG: "true"
        CI: "true"
      run: |
        # Run unit tests
        pixi run test-tts-handler
        
        # Run integration tests with mocks
        cd q-learning-mfcs/tests/notification_system
        python run_tts_tests.py --ci --coverage
        
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage-tts.xml
```
## Engine Configuration

### pyttsx3 Engine Configuration

#### Basic Configuration

```python
# config/pyttsx3_config.py
from notifications.tts_handler import TTSNotificationHandler, TTSEngineType

def configure_pyttsx3():
    """Configure pyttsx3 engine for optimal performance."""
    handler = TTSNotificationHandler(engine_type=TTSEngineType.PYTTSX3)
    
    # Get available voices
    voices = handler.get_voices()
    
    # Configure based on platform
    import platform
    system = platform.system()
    
    if system == "Linux":
        # Prefer espeak voices
        preferred_voices = ["english", "en", "en-us"]
        voice_id = None
        for voice in voices:
            if any(pref in voice['name'].lower() for pref in preferred_voices):
                voice_id = voice['id']
                break
                
    elif system == "Windows":
        # Prefer SAPI voices
        preferred_voices = ["Zira", "David", "Mark"]
        voice_id = None
        for voice in voices:
            if any(pref in voice['name'] for pref in preferred_voices):
                voice_id = voice['id']
                break
                
    elif system == "Darwin":  # macOS
        # Prefer high-quality voices
        preferred_voices = ["Alex", "Samantha", "Victoria"]
        voice_id = None
        for voice in voices:
            if any(pref in voice['name'] for pref in preferred_voices):
                voice_id = voice['id']
                break
    
    # Apply configuration
    handler.configure_tts(
        voice_id=voice_id,
        rate=180,  # Comfortable speaking rate
        volume=0.8,  # Clear audible volume
        text_limit=400  # Reasonable text length
    )
    
    return handler
```

#### Platform-Specific Configuration

**Linux Configuration:**
```bash
# Configure espeak voices
sudo apt install espeak-data-*

# List available voices
espeak --voices

# Test voice quality
espeak "This is a test of the espeak voice system"

# Configure ALSA (if needed)
cat > ~/.asoundrc << EOF
pcm.!default {
    type pulse
}
ctl.!default {
    type pulse
}
EOF
```

**Windows Configuration:**
```powershell
# List SAPI voices
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }

# Test SAPI voice
$synth.Speak("This is a test of the Windows SAPI voice system")
```

**macOS Configuration:**
```bash
# List available voices
say -v ?

# Test voice quality
say -v Alex "This is a test of the macOS voice system"

# Configure audio output
sudo dscl . -create /Users/$USER RealName "TTS User"
```

### Coqui TTS Engine Configuration

#### Model Selection Guide

```python
# config/coqui_models.py
from notifications.coqui_tts_manager import CoquiTTSConfig

# Model configurations for different use cases
MODEL_CONFIGS = {
    "fast": CoquiTTSConfig(
        model_name="tts_models/en/ljspeech/tacotron2-DDC",
        language="en",
        use_gpu=True,
        speed=1.2,
        temperature=0.7,
        timeout_seconds=10.0
    ),
    
    "quality": CoquiTTSConfig(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        language="en",
        use_gpu=True,
        speed=1.0,
        temperature=0.75,
        timeout_seconds=30.0,
        enable_ssml=True
    ),
    
    "multilingual": CoquiTTSConfig(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        language="auto",  # Auto-detect language
        use_gpu=True,
        speed=1.0,
        temperature=0.8,
        timeout_seconds=30.0
    ),
    
    "voice_cloning": CoquiTTSConfig(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        language="en",
        use_gpu=True,
        speed=1.0,
        temperature=0.75,
        # voice_file will be set per request
    )
}

def get_config(use_case: str) -> CoquiTTSConfig:
    """Get configuration for specific use case."""
    return MODEL_CONFIGS.get(use_case, MODEL_CONFIGS["quality"])
```

#### GPU Configuration

```bash
#!/bin/bash
# configure_gpu_tts.sh

echo "Configuring GPU for Coqui TTS..."

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected"
    nvidia-smi
    
    # Check GPU memory
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo "GPU Memory: ${GPU_MEMORY}MB"
    
    if [ "$GPU_MEMORY" -lt 4000 ]; then
        echo "Warning: GPU memory < 4GB, may need CPU fallback"
        export TTS_USE_GPU="false"
    else
        export TTS_USE_GPU="true"
    fi
    
    # Set CUDA environment
    export CUDA_VISIBLE_DEVICES=0
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    
elif command -v rocm-smi &> /dev/null; then
    echo "AMD GPU detected"
    rocm-smi
    export TTS_USE_GPU="true"
    export HIP_VISIBLE_DEVICES=0
    
else
    echo "No GPU detected, using CPU"
    export TTS_USE_GPU="false"
fi

# Test GPU configuration
python - << EOF
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
EOF
```

#### Model Caching Configuration

```python
# config/model_cache.py
import os
from pathlib import Path

def setup_model_cache():
    """Setup model caching for Coqui TTS."""
    
    # Set cache directory
    cache_dir = Path.home() / ".cache" / "tts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Environment variables for TTS caching
    os.environ["TTS_CACHE"] = str(cache_dir)
    os.environ["TORCH_HOME"] = str(cache_dir / "torch")
    
    print(f"TTS cache directory: {cache_dir}")
    
    # Pre-download common models
    from TTS.api import TTS
    
    models_to_cache = [
        "tts_models/en/ljspeech/tacotron2-DDC",
        "tts_models/multilingual/multi-dataset/xtts_v2"
    ]
    
    for model_name in models_to_cache:
        try:
            print(f"Caching model: {model_name}")
            tts = TTS(model_name)
            del tts  # Free memory
            print(f"Model cached successfully: {model_name}")
        except Exception as e:
            print(f"Failed to cache model {model_name}: {e}")

# Run setup
if __name__ == "__main__":
    setup_model_cache()
```

### Hybrid Engine Configuration

```python
# config/hybrid_config.py
import os
from notifications.tts_handler import TTSNotificationHandler, TTSEngineType
from notifications.coqui_tts_manager import CoquiTTSConfig

def configure_hybrid_engine():
    """Configure hybrid TTS engine with intelligent fallback."""
    
    # Environment-based preference
    prefer_coqui = os.getenv('TTS_PREFER_COQUI', 'true').lower() == 'true'
    use_gpu = os.getenv('TTS_USE_GPU', 'true').lower() == 'true'
    
    # Coqui TTS configuration
    coqui_config = CoquiTTSConfig(
        enabled=prefer_coqui,
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        language="en",
        use_gpu=use_gpu,
        speed=1.0,
        temperature=0.75,
        cache_models=True,
        timeout_seconds=20.0,
        max_text_length=500
    )
    
    # Initialize hybrid handler
    handler = TTSNotificationHandler(
        engine_type=TTSEngineType.HYBRID,
        coqui_config=coqui_config
    )
    
    # Configure fallback pyttsx3 settings
    handler.configure_tts(
        rate=180,
        volume=0.8,
        text_limit=400
    )
    
    return handler

def test_hybrid_configuration():
    """Test hybrid engine configuration."""
    handler = configure_hybrid_engine()
    
    # Test availability
    if handler.is_available():
        print("Hybrid TTS engine configured successfully")
        
        # Get capabilities
        capabilities = handler.get_capabilities()
        print(f"TTS available: {capabilities['tts']}")
        print(f"Voice selection: {capabilities['voice_selection']}")
        
        # Test notification
        from notifications.base import NotificationConfig, NotificationLevel
        config = NotificationConfig(
            title="Configuration Test",
            message="Hybrid TTS engine is working correctly",
            level=NotificationLevel.SUCCESS
        )
        
        success = handler.send_notification(config)
        print(f"Test notification: {'Success' if success else 'Failed'}")
        
    else:
        print("Hybrid TTS engine configuration failed")
        
    return handler

if __name__ == "__main__":
    test_hybrid_configuration()
```
## Performance Tuning

### Memory Optimization

```python
# config/memory_optimization.py
import gc
import torch
from notifications.coqui_tts_manager import CoquiTTSManager, CoquiTTSConfig

class OptimizedCoquiTTSManager(CoquiTTSManager):
    """Memory-optimized Coqui TTS manager."""
    
    def __init__(self, config: CoquiTTSConfig):
        # Enable memory optimization
        config.cache_models = False  # Don't cache for memory-constrained systems
        super().__init__(config)
        
    def speak(self, *args, **kwargs):
        """Speak with memory cleanup."""
        try:
            result = super().speak(*args, **kwargs)
            
            # Cleanup after synthesis
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            gc.collect()
            
            return result
            
        except Exception as e:
            # Emergency memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            raise

def configure_memory_optimized_tts():
    """Configure TTS for memory-constrained environments."""
    config = CoquiTTSConfig(
        # Use smaller, faster model
        model_name="tts_models/en/ljspeech/tacotron2-DDC",
        
        # Reduce batch processing
        use_gpu=False,  # Use CPU to save GPU memory
        
        # Shorter timeout to prevent hanging
        timeout_seconds=15.0,
        
        # Limit text length
        max_text_length=200,
        
        # Disable caching
        cache_models=False
    )
    
    return OptimizedCoquiTTSManager(config)
```

### Latency Optimization

```python
# config/latency_optimization.py
from notifications.tts_handler import TTSNotificationHandler, TTSEngineType

def configure_low_latency_tts():
    """Configure TTS for minimal latency."""
    
    # Use pyttsx3 for fastest response
    handler = TTSNotificationHandler(engine_type=TTSEngineType.PYTTSX3)
    
    # Optimize for speed
    handler.configure_tts(
        rate=250,  # Fast speech
        volume=0.8,
        text_limit=150,  # Short messages
    )
    
    return handler

def configure_preloaded_tts():
    """Configure TTS with preloaded models for faster synthesis."""
    from notifications.coqui_tts_manager import CoquiTTSConfig
    
    config = CoquiTTSConfig(
        # Use fast model
        model_name="tts_models/en/ljspeech/tacotron2-DDC",
        
        # Preload and cache
        cache_models=True,
        
        # Optimize synthesis parameters
        speed=1.2,
        temperature=0.7,
        
        # Shorter timeout
        timeout_seconds=10.0
    )
    
    handler = TTSNotificationHandler(
        engine_type=TTSEngineType.COQUI,
        coqui_config=config
    )
    
    # Warm up the model
    from notifications.base import NotificationConfig, NotificationLevel
    warmup_config = NotificationConfig(
        title="Warmup",
        message="Model warmup",
        level=NotificationLevel.INFO
    )
    
    handler.send_notification(warmup_config)
    
    return handler
```

### Throughput Optimization

```python
# config/throughput_optimization.py
import asyncio
from concurrent.futures import ThreadPoolExecutor
from notifications.tts_handler import TTSNotificationHandler, TTSEngineType

class HighThroughputTTSManager:
    """High-throughput TTS manager with connection pooling."""
    
    def __init__(self, pool_size: int = 4):
        self.pool_size = pool_size
        self.handlers = []
        self.executor = ThreadPoolExecutor(max_workers=pool_size)
        
        # Create handler pool
        for i in range(pool_size):
            handler = TTSNotificationHandler(
                engine_type=TTSEngineType.PYTTSX3  # Fast for high throughput
            )
            handler.configure_tts(
                rate=220,
                volume=0.8,
                text_limit=200
            )
            self.handlers.append(handler)
    
    async def send_notification_async(self, config):
        """Send notification asynchronously using handler pool."""
        import random
        
        # Select random handler from pool
        handler = random.choice(self.handlers)
        
        # Submit to thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            handler.send_notification,
            config
        )
        
        return result
    
    async def send_batch_notifications(self, configs):
        """Send multiple notifications concurrently."""
        tasks = [
            self.send_notification_async(config)
            for config in configs
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        success_count = sum(1 for r in results if r is True)
        error_count = len(results) - success_count
        
        return {
            'total': len(configs),
            'success': success_count,
            'errors': error_count,
            'results': results
        }
    
    def cleanup(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)
        for handler in self.handlers:
            handler.cleanup()

# Usage example
async def high_throughput_example():
    manager = HighThroughputTTSManager(pool_size=8)
    
    # Create batch of notifications
    from notifications.base import NotificationConfig, NotificationLevel
    configs = [
        NotificationConfig(
            title=f"Alert {i}",
            message=f"High throughput notification {i}",
            level=NotificationLevel.INFO
        )
        for i in range(100)
    ]
    
    # Send all notifications
    results = await manager.send_batch_notifications(configs)
    print(f"Processed {results['success']}/{results['total']} notifications")
    
    manager.cleanup()

# Run example
# asyncio.run(high_throughput_example())
```
## Security Configuration

### Input Sanitization

```python
# config/security_config.py
import re
import html
from typing import str

class TTSSecurityConfig:
    """Security configuration for TTS input sanitization."""
    
    # Dangerous patterns to filter
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # JavaScript
        r'<iframe[^>]*>.*?</iframe>',  # Iframes
        r'javascript:',               # JavaScript URLs
        r'data:',                     # Data URLs
        r'vbscript:',                # VBScript URLs
        r'on\w+\s*=',                # Event handlers
    ]
    
    # Maximum text lengths
    MAX_TITLE_LENGTH = 100
    MAX_MESSAGE_LENGTH = 1000
    MAX_TOTAL_LENGTH = 1100
    
    # Allowed characters regex
    ALLOWED_CHARS = re.compile(r'^[a-zA-Z0-9\s\.\,\!\?\:\;\-\(\)\[\]\'\"]+$')
    
    @classmethod
    def sanitize_text(cls, text: str) -> str:
        """Sanitize text input for TTS synthesis."""
        if not text:
            return ""
        
        # Remove HTML entities
        text = html.unescape(text)
        
        # Remove dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Validate characters
        if not cls.ALLOWED_CHARS.match(text):
            # Remove disallowed characters
            text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\:\;\-\(\)\[\]\'\"]+', '', text)
        
        return text
    
    @classmethod
    def validate_config(cls, config) -> bool:
        """Validate notification configuration."""
        # Check title length
        if len(config.title) > cls.MAX_TITLE_LENGTH:
            return False
        
        # Check message length
        if len(config.message) > cls.MAX_MESSAGE_LENGTH:
            return False
        
        # Check total length
        total_length = len(config.title) + len(config.message)
        if total_length > cls.MAX_TOTAL_LENGTH:
            return False
        
        return True
    
    @classmethod
    def sanitize_config(cls, config):
        """Sanitize notification configuration."""
        # Sanitize title
        config.title = cls.sanitize_text(config.title)
        if len(config.title) > cls.MAX_TITLE_LENGTH:
            config.title = config.title[:cls.MAX_TITLE_LENGTH] + "..."
        
        # Sanitize message
        config.message = cls.sanitize_text(config.message)
        if len(config.message) > cls.MAX_MESSAGE_LENGTH:
            config.message = config.message[:cls.MAX_MESSAGE_LENGTH] + "..."
        
        return config

# Apply security to TTS handler
def create_secure_tts_handler():
    """Create TTS handler with security features."""
    from notifications.tts_handler import TTSNotificationHandler
    
    class SecureTTSHandler(TTSNotificationHandler):
        def send_notification(self, config):
            # Apply security sanitization
            config = TTSSecurityConfig.sanitize_config(config)
            
            # Validate configuration
            if not TTSSecurityConfig.validate_config(config):
                raise ValueError("Invalid notification configuration")
            
            return super().send_notification(config)
    
    return SecureTTSHandler()
```

### Access Control

```python
# config/access_control.py
import hashlib
import time
from typing import Dict, Set
from dataclasses import dataclass

@dataclass
class AccessToken:
    """Access token for TTS operations."""
    token_id: str
    permissions: Set[str]
    expires_at: float
    rate_limit: int  # requests per minute

class TTSAccessControl:
    """Access control system for TTS operations."""
    
    def __init__(self):
        self.tokens: Dict[str, AccessToken] = {}
        self.request_counts: Dict[str, list] = {}
    
    def create_token(self, permissions: Set[str], expires_in: int = 3600, rate_limit: int = 100) -> str:
        """Create access token."""
        token_id = hashlib.sha256(f"{time.time()}{permissions}".encode()).hexdigest()[:32]
        
        token = AccessToken(
            token_id=token_id,
            permissions=permissions,
            expires_at=time.time() + expires_in,
            rate_limit=rate_limit
        )
        
        self.tokens[token_id] = token
        self.request_counts[token_id] = []
        
        return token_id
    
    def validate_token(self, token_id: str, required_permission: str) -> bool:
        """Validate access token."""
        if token_id not in self.tokens:
            return False
        
        token = self.tokens[token_id]
        
        # Check expiration
        if time.time() > token.expires_at:
            del self.tokens[token_id]
            return False
        
        # Check permission
        if required_permission not in token.permissions:
            return False
        
        # Check rate limit
        now = time.time()
        minute_ago = now - 60
        
        # Remove old requests
        self.request_counts[token_id] = [
            req_time for req_time in self.request_counts[token_id]
            if req_time > minute_ago
        ]
        
        # Check rate limit
        if len(self.request_counts[token_id]) >= token.rate_limit:
            return False
        
        # Record this request
        self.request_counts[token_id].append(now)
        
        return True

# Secure TTS handler with access control
class AccessControlledTTSHandler:
    """TTS handler with access control."""
    
    def __init__(self):
        from notifications.tts_handler import TTSNotificationHandler
        self.handler = TTSNotificationHandler()
        self.access_control = TTSAccessControl()
    
    def send_notification(self, config, token_id: str):
        """Send notification with access control."""
        if not self.access_control.validate_token(token_id, "tts.send"):
            raise PermissionError("Invalid or expired token")
        
        return self.handler.send_notification(config)
    
    def configure_tts(self, token_id: str, **kwargs):
        """Configure TTS with access control."""
        if not self.access_control.validate_token(token_id, "tts.configure"):
            raise PermissionError("Invalid or expired token")
        
        return self.handler.configure_tts(**kwargs)
    
    def create_access_token(self, permissions: Set[str], **kwargs) -> str:
        """Create access token."""
        return self.access_control.create_token(permissions, **kwargs)

# Usage example
def setup_secure_tts():
    handler = AccessControlledTTSHandler()
    
    # Create token with specific permissions
    token = handler.create_access_token(
        permissions={"tts.send", "tts.configure"},
        expires_in=3600,  # 1 hour
        rate_limit=60     # 60 requests per minute
    )
    
    print(f"Access token: {token}")
    return handler, token
```
## Monitoring and Logging

### Comprehensive Logging Configuration

```python
# config/logging_config.py
import logging
import logging.handlers
import json
from pathlib import Path
from datetime import datetime

def setup_tts_logging(log_level: str = "INFO", log_dir: str = "logs"):
    """Setup comprehensive logging for TTS system."""
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger('tts')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_path / "tts.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # JSON handler for structured logging
    json_handler = logging.handlers.RotatingFileHandler(
        log_path / "tts.json",
        maxBytes=10*1024*1024,
        backupCount=5
    )
    
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_entry = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'function': record.funcName,
                'line': record.lineno,
                'message': record.getMessage(),
                'module': record.module,
                'pathname': record.pathname
            }
            
            # Add exception info if present
            if record.exc_info:
                log_entry['exception'] = self.formatException(record.exc_info)
            
            return json.dumps(log_entry)
    
    json_handler.setFormatter(JSONFormatter())
    logger.addHandler(json_handler)
    
    # Performance metrics handler
    metrics_handler = logging.handlers.RotatingFileHandler(
        log_path / "tts_metrics.log",
        maxBytes=5*1024*1024,
        backupCount=3
    )
    
    metrics_formatter = logging.Formatter(
        '%(asctime)s - METRICS - %(message)s'
    )
    metrics_handler.setFormatter(metrics_formatter)
    
    # Create metrics logger
    metrics_logger = logging.getLogger('tts.metrics')
    metrics_logger.setLevel(logging.INFO)
    metrics_logger.addHandler(metrics_handler)
    
    return logger
```

### Performance Monitoring

```python
# config/monitoring.py
import time
import psutil
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import json

@dataclass
class TTSMetrics:
    """TTS performance metrics."""
    timestamp: float
    operation: str
    duration: float
    success: bool
    engine: str
    text_length: int
    memory_usage_mb: float
    cpu_percent: float
    gpu_memory_mb: Optional[float] = None

class TTSMonitor:
    """Performance monitoring for TTS system."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics: deque = deque(maxlen=window_size)
        self.counters = defaultdict(int)
        self.timing_stats = defaultdict(list)
        self._lock = threading.Lock()
        
    def record_operation(
        self,
        operation: str,
        duration: float,
        success: bool,
        engine: str,
        text_length: int
    ):
        """Record TTS operation metrics."""
        
        # Get system metrics
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()
        
        # Get GPU memory if available
        gpu_memory_mb = None
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        except ImportError:
            pass
        
        # Create metrics record
        metrics = TTSMetrics(
            timestamp=time.time(),
            operation=operation,
            duration=duration,
            success=success,
            engine=engine,
            text_length=text_length,
            memory_usage_mb=memory_mb,
            cpu_percent=cpu_percent,
            gpu_memory_mb=gpu_memory_mb
        )
        
        with self._lock:
            self.metrics.append(metrics)
            
            # Update counters
            self.counters[f"{operation}.total"] += 1
            if success:
                self.counters[f"{operation}.success"] += 1
            else:
                self.counters[f"{operation}.failure"] += 1
            
            # Update timing stats
            self.timing_stats[operation].append(duration)
            if len(self.timing_stats[operation]) > 100:
                self.timing_stats[operation].pop(0)
    
    def get_statistics(self) -> Dict:
        """Get performance statistics."""
        with self._lock:
            if not self.metrics:
                return {}
            
            stats = {
                'total_operations': len(self.metrics),
                'time_window': {
                    'start': self.metrics[0].timestamp,
                    'end': self.metrics[-1].timestamp,
                    'duration': self.metrics[-1].timestamp - self.metrics[0].timestamp
                },
                'counters': dict(self.counters),
                'performance': {}
            }
            
            # Calculate performance statistics
            for operation, timings in self.timing_stats.items():
                if timings:
                    stats['performance'][operation] = {
                        'count': len(timings),
                        'avg_duration': sum(timings) / len(timings),
                        'min_duration': min(timings),
                        'max_duration': max(timings),
                        'p95_duration': sorted(timings)[int(len(timings) * 0.95)] if len(timings) > 20 else max(timings)
                    }
            
            # System resource statistics
            recent_metrics = list(self.metrics)[-100:]  # Last 100 operations
            if recent_metrics:
                avg_memory = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
                avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
                
                stats['resources'] = {
                    'avg_memory_mb': avg_memory,
                    'avg_cpu_percent': avg_cpu
                }
                
                # GPU statistics if available
                gpu_metrics = [m.gpu_memory_mb for m in recent_metrics if m.gpu_memory_mb is not None]
                if gpu_metrics:
                    stats['resources']['avg_gpu_memory_mb'] = sum(gpu_metrics) / len(gpu_metrics)
            
            return stats
    
    def export_metrics(self, filename: str):
        """Export metrics to JSON file."""
        with self._lock:
            metrics_data = [asdict(m) for m in self.metrics]
        
        with open(filename, 'w') as f:
            json.dump({
                'metrics': metrics_data,
                'statistics': self.get_statistics()
            }, f, indent=2)

# Monitoring decorator
def monitor_tts_operation(monitor: TTSMonitor, operation: str, engine: str):
    """Decorator to monitor TTS operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = False
            text_length = 0
            
            try:
                # Extract text length if available
                if args and hasattr(args[0], 'message'):
                    text_length = len(args[0].message)
                elif 'text' in kwargs:
                    text_length = len(kwargs['text'])
                
                result = func(*args, **kwargs)
                success = bool(result)
                return result
                
            except Exception as e:
                success = False
                raise
                
            finally:
                duration = time.time() - start_time
                monitor.record_operation(
                    operation=operation,
                    duration=duration,
                    success=success,
                    engine=engine,
                    text_length=text_length
                )
        
        return wrapper
    return decorator

# Usage example
monitor = TTSMonitor()

# Apply monitoring to TTS handler
class MonitoredTTSHandler:
    def __init__(self):
        from notifications.tts_handler import TTSNotificationHandler
        self.handler = TTSNotificationHandler()
        self.monitor = monitor
    
    @monitor_tts_operation(monitor, "send_notification", "hybrid")
    def send_notification(self, config):
        return self.handler.send_notification(config)
    
    def get_performance_stats(self):
        return self.monitor.get_statistics()
    
    def export_performance_data(self, filename: str):
        self.monitor.export_metrics(filename)
```
## Maintenance Procedures

### Health Check System

```bash
#!/bin/bash
# tts_health_check.sh

echo "TTS System Health Check"
echo "======================"

# Check Python environment
echo "1. Python Environment:"
python --version
echo "TTS modules installed:"
python -c "
try:
    import pyttsx3
    print('  ✓ pyttsx3 available')
except ImportError:
    print('  ✗ pyttsx3 not available')

try:
    from TTS.api import TTS
    print('  ✓ Coqui TTS available')
except ImportError:
    print('  ✗ Coqui TTS not available')
"

# Check audio system
echo -e "\n2. Audio System:"
if command -v pactl &> /dev/null; then
    echo "  ✓ PulseAudio detected"
    pactl info | grep "Server Name"
    pactl list short sinks | head -3
elif command -v amixer &> /dev/null; then
    echo "  ✓ ALSA detected"
    amixer get Master | grep "Playback"
else
    echo "  ⚠ No audio system detected"
fi

# Check GPU availability
echo -e "\n3. GPU Status:"
if command -v nvidia-smi &> /dev/null; then
    echo "  ✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
elif command -v rocm-smi &> /dev/null; then
    echo "  ✓ AMD GPU detected"
    rocm-smi --showmeminfo vram
else
    echo "  ⚠ No GPU detected"
fi

# Test TTS functionality
echo -e "\n4. TTS Functionality:"
cd q-learning-mfcs

# Test pyttsx3
echo "  Testing pyttsx3:"
timeout 10s python -c "
from src.notifications.tts_handler import TTSNotificationHandler, TTSEngineType
from src.notifications.base import NotificationConfig, NotificationLevel
handler = TTSNotificationHandler(engine_type=TTSEngineType.PYTTSX3)
if handler.is_available():
    config = NotificationConfig('Health Check', 'pyttsx3 is working', NotificationLevel.INFO)
    success = handler.send_notification(config)
    print('    ✓ pyttsx3 working' if success else '    ✗ pyttsx3 failed')
else:
    print('    ✗ pyttsx3 not available')
" 2>/dev/null || echo "    ✗ pyttsx3 test failed"

# Test Coqui TTS (if available)
echo "  Testing Coqui TTS:"
timeout 30s python -c "
from src.notifications.tts_handler import TTSNotificationHandler, TTSEngineType
from src.notifications.base import NotificationConfig, NotificationLevel
try:
    handler = TTSNotificationHandler(engine_type=TTSEngineType.COQUI)
    if handler.is_available():
        config = NotificationConfig('Health Check', 'Coqui TTS is working', NotificationLevel.INFO)
        success = handler.send_notification(config)
        print('    ✓ Coqui TTS working' if success else '    ✗ Coqui TTS failed')
    else:
        print('    ✗ Coqui TTS not available')
except Exception as e:
    print(f'    ✗ Coqui TTS error: {e}')
" 2>/dev/null || echo "    ✗ Coqui TTS test failed"

# Test hybrid engine
echo "  Testing Hybrid Engine:"
timeout 15s python -c "
from src.notifications.tts_handler import TTSNotificationHandler, TTSEngineType
from src.notifications.base import NotificationConfig, NotificationLevel
handler = TTSNotificationHandler(engine_type=TTSEngineType.HYBRID)
if handler.is_available():
    config = NotificationConfig('Health Check', 'Hybrid engine is working', NotificationLevel.INFO)
    success = handler.send_notification(config)
    print('    ✓ Hybrid engine working' if success else '    ✗ Hybrid engine failed')
else:
    print('    ✗ Hybrid engine not available')
" 2>/dev/null || echo "    ✗ Hybrid engine test failed"

# Performance test
echo -e "\n5. Performance Test:"
python -c "
import time
from src.notifications.tts_handler import TTSNotificationHandler
from src.notifications.base import NotificationConfig, NotificationLevel

handler = TTSNotificationHandler()
config = NotificationConfig('Performance Test', 'Measuring TTS response time', NotificationLevel.INFO)

start_time = time.time()
success = handler.send_notification(config)
end_time = time.time()

duration = end_time - start_time
print(f'  Response time: {duration:.2f}s')
if duration < 2.0:
    print('  ✓ Good performance')
elif duration < 5.0:
    print('  ⚠ Acceptable performance')
else:
    print('  ✗ Poor performance')
"

echo -e "\nHealth check complete!"
```

### Automated Maintenance Script

```bash
#!/bin/bash
# tts_maintenance.sh

echo "TTS System Maintenance"
echo "====================="

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Clean up old logs
log "Cleaning up old log files..."
find logs/ -name "*.log" -mtime +30 -delete 2>/dev/null || true
find logs/ -name "*.json" -mtime +30 -delete 2>/dev/null || true

# Clean up temporary files
log "Cleaning up temporary files..."
find /tmp -name "tmp*tts*" -mtime +1 -delete 2>/dev/null || true
find /tmp -name "*.wav" -mtime +1 -delete 2>/dev/null || true

# Update TTS models (if configured)
if [ "$TTS_AUTO_UPDATE" = "true" ]; then
    log "Updating TTS models..."
    python -c "
from TTS.api import TTS
import os

models_to_update = [
    'tts_models/en/ljspeech/tacotron2-DDC',
    'tts_models/multilingual/multi-dataset/xtts_v2'
]

for model_name in models_to_update:
    try:
        print(f'Updating model: {model_name}')
        tts = TTS(model_name)
        del tts
        print(f'Model updated: {model_name}')
    except Exception as e:
        print(f'Failed to update model {model_name}: {e}')
" 2>/dev/null || log "Model update failed"
fi

# Clean GPU memory cache
if command -v nvidia-smi &> /dev/null; then
    log "Cleaning GPU memory cache..."
    python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('GPU cache cleared')
" 2>/dev/null || true
fi

# Rotate metrics files
log "Rotating metrics files..."
if [ -f "logs/tts_metrics.log" ]; then
    cp logs/tts_metrics.log logs/tts_metrics_$(date +%Y%m%d).log
    > logs/tts_metrics.log
fi

# Check disk space
log "Checking disk space..."
DISK_USAGE=$(df . | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 80 ]; then
    log "WARNING: Disk usage is ${DISK_USAGE}%"
    # Clean up more aggressively
    find logs/ -name "*.log" -mtime +7 -delete 2>/dev/null || true
fi

# Generate maintenance report
log "Generating maintenance report..."
cat > logs/maintenance_report_$(date +%Y%m%d).txt << EOF
TTS System Maintenance Report
============================
Date: $(date)
Disk Usage: ${DISK_USAGE}%
Log Files Cleaned: $(find logs/ -name "*.log.*" | wc -l)
Temp Files Cleaned: $(find /tmp -name "*tts*" 2>/dev/null | wc -l)

System Status:
$(bash tts_health_check.sh)
EOF

log "Maintenance complete!"
```

This comprehensive deployment and configuration manual provides everything needed to successfully deploy, configure, and maintain the TTS notification system across different environments and use cases.