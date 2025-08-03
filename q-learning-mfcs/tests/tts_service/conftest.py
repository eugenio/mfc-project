"""
TTS Service Test Configuration and Fixtures
==========================================

Shared pytest fixtures and configuration for TTS service testing.
Provides mock objects, test data, and reusable test components.

Created: 2025-08-03
Author: Agent Gamma - TTS Service Implementation Lead
"""
import pytest
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, MagicMock
import tempfile
import json

# Import models and components
from src.tts_service.models import (
    TTSRequest, TTSResponse, VoiceInfo, TTSStatus, TTSPriority,
    TTSEngineType, TTSOutputFormat, HealthResponse, QueueStatus
)


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_tts_request() -> TTSRequest:
    """Sample TTS request for testing."""
    return TTSRequest(
        text="Hello, this is a test message for the TTS service.",
        voice_id="en-US-neural-1",
        engine=TTSEngineType.HYBRID,
        priority=TTSPriority.NORMAL,
        speed=1.0,
        pitch=1.0,
        volume=0.8,
        output_format=TTSOutputFormat.WAV,
        client_id="test-client-001"
    )


@pytest.fixture
def urgent_tts_request() -> TTSRequest:
    """Urgent priority TTS request for testing."""
    return TTSRequest(
        text="URGENT: System alert - immediate attention required!",
        engine=TTSEngineType.PYTTSX3,
        priority=TTSPriority.URGENT,
        speed=1.2,
        volume=1.0,
        client_id="alert-system"
    )


@pytest.fixture
def batch_tts_requests() -> List[TTSRequest]:
    """List of TTS requests for batch testing."""
    return [
        TTSRequest(text=f"Test message {i}", client_id=f"batch-client-{i}")
        for i in range(5)
    ]


@pytest.fixture
def sample_voice_info() -> VoiceInfo:
    """Sample voice information for testing."""
    return VoiceInfo(
        id="en-US-neural-1",
        name="English (US) Neural Female",
        language="en-US",
        gender="female",
        engine="coqui",
        quality="high",
        sample_rate=22050,
        is_neural=True,
        supports_ssml=True
    )


@pytest.fixture
def sample_voices() -> List[VoiceInfo]:
    """List of sample voices for testing."""
    return [
        VoiceInfo(
            id="en-US-male-1",
            name="English (US) Male",
            language="en-US",
            gender="male",
            engine="pyttsx3",
            quality="medium",
            is_neural=False,
            supports_ssml=False
        ),
        VoiceInfo(
            id="en-US-female-1",
            name="English (US) Female",
            language="en-US",
            gender="female",
            engine="pyttsx3",
            quality="medium",
            is_neural=False,
            supports_ssml=False
        ),
        VoiceInfo(
            id="en-US-neural-1",
            name="English (US) Neural Female",
            language="en-US",
            gender="female",
            engine="coqui",
            quality="high",
            sample_rate=22050,
            is_neural=True,
            supports_ssml=True
        )
    ]


@pytest.fixture
def mock_audio_data() -> bytes:
    """Mock audio data for testing."""
    # Create a simple WAV-like header for testing
    return b"RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x08\x00\x00"


@pytest.fixture
def temp_audio_file(tmp_path) -> Path:
    """Temporary audio file for testing."""
    audio_file = tmp_path / "test_audio.wav"
    audio_file.write_bytes(b"fake wav data")
    return audio_file


@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    """Temporary output directory for testing."""
    output_dir = tmp_path / "tts_output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def mock_tts_engine():
    """Mock TTS engine for testing."""
    engine = Mock()
    engine.is_available.return_value = True
    engine.is_healthy.return_value = True
    engine.get_voices.return_value = [
        VoiceInfo(
            id="test-voice-1",
            name="Test Voice 1",
            language="en-US",
            engine="mock",
            quality="high"
        )
    ]
    engine.synthesize.return_value = b"mock audio data"
    engine.synthesize_async = AsyncMock(return_value=b"mock audio data")
    return engine


@pytest.fixture
def mock_pyttsx3_engine():
    """Mock pyttsx3 engine for testing."""
    engine = Mock()
    engine.is_available.return_value = True
    engine.initialize.return_value = True
    engine.speak.return_value = True
    engine.get_voices.return_value = [
        {"id": "voice1", "name": "Voice 1"},
        {"id": "voice2", "name": "Voice 2"}
    ]
    engine.set_voice.return_value = None
    engine.set_rate.return_value = None
    engine.set_volume.return_value = None
    return engine


@pytest.fixture
def mock_coqui_tts():
    """Mock Coqui TTS manager for testing."""
    manager = Mock()
    manager.is_available.return_value = True
    manager.speak.return_value = True
    manager.get_available_speakers.return_value = ["Speaker1", "Speaker2"]
    manager.get_available_languages.return_value = ["en", "es", "fr"]
    manager.get_voice_info.return_value = {
        "model_name": "test-model",
        "available": True,
        "speakers": ["Speaker1", "Speaker2"],
        "languages": ["en", "es", "fr"]
    }
    return manager


@pytest.fixture
def mock_queue_processor():
    """Mock queue processor for testing."""
    processor = AsyncMock()
    processor.is_running = True
    processor.queue_size = 0
    processor.processed_count = 0
    processor.failed_count = 0
    processor.add_request = AsyncMock(return_value="test-request-id")
    processor.get_status = AsyncMock(return_value=QueueStatus(
        total_requests=0,
        pending_requests=0,
        processing_requests=0,
        completed_requests=0,
        failed_requests=0,
        average_processing_time=0.5,
        queue_throughput=10.0,
        estimated_wait_time=0.0,
        priority_breakdown={"normal": 0, "high": 0, "urgent": 0, "low": 0}
    ))
    return processor


@pytest.fixture
def mock_engine_manager():
    """Mock engine manager for testing."""
    manager = Mock()
    manager.get_available_engines.return_value = ["pyttsx3", "coqui"]
    manager.get_best_engine.return_value = "coqui"
    manager.get_engine.return_value = Mock()
    manager.get_all_voices.return_value = [
        VoiceInfo(id="voice1", name="Voice 1", language="en-US", engine="pyttsx3")
    ]
    manager.is_healthy.return_value = True
    return manager


@pytest.fixture
def sample_health_response() -> HealthResponse:
    """Sample health response for testing."""
    return HealthResponse(
        version="1.0.0",
        status="healthy",
        uptime_seconds=3600.0,
        engines=[],
        healthy_engines=2,
        total_engines=2,
        queue_status=QueueStatus(
            total_requests=10,
            pending_requests=0,
            processing_requests=1,
            completed_requests=9,
            failed_requests=0,
            average_processing_time=1.2,
            queue_throughput=5.0,
            estimated_wait_time=0.2,
            priority_breakdown={"normal": 8, "high": 2, "urgent": 0, "low": 0}
        ),
        queue_health="healthy",
        memory_usage_mb=150.5,
        cpu_usage_percent=15.2,
        total_requests_processed=100,
        requests_per_minute=5.0,
        error_rate_percent=1.0
    )


@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Test configuration for TTS service."""
    return {
        "host": "localhost",
        "port": 8001,  # Use different port for testing
        "workers": 1,
        "max_queue_size": 100,
        "max_concurrent_requests": 2,
        "request_timeout_seconds": 30.0,
        "enabled_engines": ["pyttsx3", "coqui"],
        "default_engine": "hybrid",
        "max_text_length": 1000,
        "log_level": "DEBUG"
    }


@pytest.fixture
def mock_fastapi_app():
    """Mock FastAPI application for testing."""
    from fastapi import FastAPI
    app = FastAPI(title="Test TTS Service")
    return app


@pytest.fixture
def mock_request_id() -> str:
    """Mock request ID for testing."""
    return "test-req-12345-abcde"


@pytest.fixture
def sample_tts_response(mock_request_id, sample_voice_info) -> TTSResponse:
    """Sample TTS response for testing."""
    return TTSResponse(
        request_id=mock_request_id,
        status=TTSStatus.COMPLETED,
        audio_data=b"mock audio data",
        audio_format="wav",
        audio_duration=2.5,
        voice_used=sample_voice_info,
        engine_used="coqui",
        processing_time=1.2,
        created_at=datetime.now()
    )


@pytest.fixture
def invalid_tts_requests() -> List[Dict[str, Any]]:
    """Invalid TTS request data for validation testing."""
    return [
        {"text": ""},  # Empty text
        {"text": " " * 10},  # Whitespace only
        {"text": "a" * 10000},  # Too long
        {"speed": -1.0},  # Invalid speed
        {"volume": 2.0},  # Invalid volume
        {"priority": "invalid"},  # Invalid priority
        {"engine": "nonexistent"},  # Invalid engine
        {"callback_url": "not-a-url"},  # Invalid URL
    ]


@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically cleanup temporary files after each test."""
    yield
    # Cleanup code runs after each test
    temp_files = Path("/tmp").glob("tts_test_*")
    for file in temp_files:
        try:
            if file.is_file():
                file.unlink()
            elif file.is_dir():
                import shutil
                shutil.rmtree(file)
        except Exception:
            pass  # Ignore cleanup errors


# Mock patches for external dependencies
@pytest.fixture
def mock_pyttsx3_module():
    """Mock the pyttsx3 module."""
    with pytest.MonkeyPatch.context() as m:
        mock_module = Mock()
        mock_engine = Mock()
        mock_module.init.return_value = mock_engine
        m.setattr("pyttsx3", mock_module)
        yield mock_engine


@pytest.fixture
def mock_coqui_module():
    """Mock the Coqui TTS module."""
    with pytest.MonkeyPatch.context() as m:
        mock_tts = Mock()
        mock_api = Mock()
        mock_api.TTS = mock_tts
        m.setattr("TTS", mock_api)
        m.setattr("TTS.api", mock_api)
        yield mock_tts


@pytest.fixture
def performance_test_config():
    """Configuration for performance tests."""
    return {
        "concurrent_requests": 10,
        "total_requests": 100,
        "max_response_time": 5.0,  # seconds
        "max_memory_usage": 500.0,  # MB
        "min_throughput": 5.0,  # requests per second
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests") 
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "tts: TTS service tests")
    config.addinivalue_line("markers", "api: API endpoint tests")
    config.addinivalue_line("markers", "engine: Engine tests")
    config.addinivalue_line("markers", "queue: Queue processing tests")


# Custom test utilities
class TTSTestClient:
    """Test client for TTS service API testing."""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.session = None
    
    async def post_synthesize(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock POST /synthesize endpoint."""
        # This will be implemented when the actual API server is ready
        pass
    
    async def get_health(self) -> Dict[str, Any]:
        """Mock GET /health endpoint."""
        pass
    
    async def get_voices(self) -> List[Dict[str, Any]]:
        """Mock GET /voices endpoint."""
        pass


@pytest.fixture
def tts_test_client():
    """TTS service test client."""
    return TTSTestClient()