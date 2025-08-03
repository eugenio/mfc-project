"""
TTS Engine Manager Tests
=======================

Comprehensive test suite for the TTS Engine Manager implementation.
Tests the abstraction layer that manages multiple TTS engines and provides
a unified interface for voice synthesis.

Test Categories:
- Engine initialization and health checking
- Voice management and selection
- Engine abstraction and switching
- Error handling and fallback mechanisms
- Performance and resource management

Created: 2025-08-03
Author: Agent Gamma - TTS Service Implementation Lead
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any
from pathlib import Path

# These imports will fail initially (TDD - Red phase)
from src.tts_service.engine_manager import TTSEngineManager, TTSEngineAdapter
from src.tts_service.models import (
    VoiceInfo, TTSEngineType, TTSRequest, TTSResponse, TTSStatus
)


@pytest.mark.tts
@pytest.mark.engine
@pytest.mark.unit
class TestTTSEngineManager:
    """Test TTS Engine Manager functionality."""
    
    def test_engine_manager_creation(self):
        """Test creating TTS engine manager instance."""
        manager = TTSEngineManager()
        assert manager is not None
        assert hasattr(manager, 'engines')
        assert hasattr(manager, 'available_engines')
        
    def test_engine_manager_with_config(self, test_config):
        """Test creating engine manager with configuration."""
        manager = TTSEngineManager(config=test_config)
        assert manager.config == test_config
        
    def test_initialize_engines_success(self, mock_pyttsx3_engine, mock_coqui_tts):
        """Test successful initialization of all engines."""
        with patch('src.tts_service.engine_manager.Pyttsx3Engine', return_value=mock_pyttsx3_engine), \
             patch('src.tts_service.engine_manager.CoquiTTSManager', return_value=mock_coqui_tts):
            
            manager = TTSEngineManager()
            manager.initialize_engines()
            
            assert len(manager.engines) > 0
            assert TTSEngineType.PYTTSX3 in manager.engines
            
    def test_initialize_engines_partial_failure(self, mock_pyttsx3_engine):
        """Test engine initialization with some engines failing."""
        mock_pyttsx3_engine.is_available.return_value = True
        
        with patch('src.tts_service.engine_manager.Pyttsx3Engine', return_value=mock_pyttsx3_engine), \
             patch('src.tts_service.engine_manager.CoquiTTSManager', side_effect=Exception("Coqui unavailable")):
            
            manager = TTSEngineManager()
            manager.initialize_engines()
            
            # Should have pyttsx3 but not coqui
            assert TTSEngineType.PYTTSX3 in manager.engines
            assert TTSEngineType.COQUI not in manager.engines
            
    def test_get_available_engines(self, mock_pyttsx3_engine, mock_coqui_tts):
        """Test getting list of available engines."""
        mock_pyttsx3_engine.is_available.return_value = True
        mock_coqui_tts.is_available.return_value = True
        
        with patch('src.tts_service.engine_manager.Pyttsx3Engine', return_value=mock_pyttsx3_engine), \
             patch('src.tts_service.engine_manager.CoquiTTSManager', return_value=mock_coqui_tts):
            
            manager = TTSEngineManager()
            manager.initialize_engines()
            
            available = manager.get_available_engines()
            assert isinstance(available, list)
            assert len(available) >= 1
            
    def test_get_best_engine_auto_selection(self, mock_pyttsx3_engine, mock_coqui_tts):
        """Test automatic best engine selection."""
        mock_pyttsx3_engine.is_available.return_value = True
        mock_coqui_tts.is_available.return_value = True
        
        with patch('src.tts_service.engine_manager.Pyttsx3Engine', return_value=mock_pyttsx3_engine), \
             patch('src.tts_service.engine_manager.CoquiTTSManager', return_value=mock_coqui_tts):
            
            manager = TTSEngineManager()
            manager.initialize_engines()
            
            best_engine = manager.get_best_engine()
            assert best_engine in [TTSEngineType.PYTTSX3, TTSEngineType.COQUI, TTSEngineType.HYBRID]
            
    def test_get_engine_by_type(self, mock_pyttsx3_engine):
        """Test getting specific engine by type."""
        with patch('src.tts_service.engine_manager.Pyttsx3Engine', return_value=mock_pyttsx3_engine):
            manager = TTSEngineManager()
            manager.initialize_engines()
            
            engine = manager.get_engine(TTSEngineType.PYTTSX3)
            assert engine is not None
            assert engine == mock_pyttsx3_engine
            
    def test_get_nonexistent_engine(self):
        """Test getting engine that doesn't exist."""
        manager = TTSEngineManager()
        
        with pytest.raises(KeyError):
            manager.get_engine("nonexistent_engine")
            
    def test_health_check_all_engines(self, mock_pyttsx3_engine, mock_coqui_tts):
        """Test health checking all engines."""
        mock_pyttsx3_engine.is_available.return_value = True
        mock_coqui_tts.is_available.return_value = False  # Unhealthy
        
        with patch('src.tts_service.engine_manager.Pyttsx3Engine', return_value=mock_pyttsx3_engine), \
             patch('src.tts_service.engine_manager.CoquiTTSManager', return_value=mock_coqui_tts):
            
            manager = TTSEngineManager()
            manager.initialize_engines()
            
            health_status = manager.check_engine_health()
            assert isinstance(health_status, dict)
            assert TTSEngineType.PYTTSX3 in health_status
            assert health_status[TTSEngineType.PYTTSX3] is True
            
    @pytest.mark.asyncio
    async def test_synthesize_text_success(self, mock_pyttsx3_engine, sample_tts_request):
        """Test successful text synthesis."""
        mock_pyttsx3_engine.is_available.return_value = True
        mock_pyttsx3_engine.synthesize_async = AsyncMock(return_value=b"mock audio data")
        
        with patch('src.tts_service.engine_manager.Pyttsx3Engine', return_value=mock_pyttsx3_engine):
            manager = TTSEngineManager()
            manager.initialize_engines()
            
            result = await manager.synthesize_async(sample_tts_request)
            assert isinstance(result, TTSResponse)
            assert result.status == TTSStatus.COMPLETED
            assert result.audio_data == b"mock audio data"
            
    @pytest.mark.asyncio
    async def test_synthesize_engine_fallback(self, mock_pyttsx3_engine, mock_coqui_tts, sample_tts_request):
        """Test engine fallback when primary engine fails."""
        # Coqui fails, should fallback to pyttsx3
        mock_coqui_tts.synthesize_async = AsyncMock(side_effect=Exception("Coqui failed"))
        mock_pyttsx3_engine.synthesize_async = AsyncMock(return_value=b"fallback audio")
        
        with patch('src.tts_service.engine_manager.CoquiTTSManager', return_value=mock_coqui_tts), \
             patch('src.tts_service.engine_manager.Pyttsx3Engine', return_value=mock_pyttsx3_engine):
            
            manager = TTSEngineManager()
            manager.initialize_engines()
            
            # Request Coqui specifically
            sample_tts_request.engine = TTSEngineType.COQUI
            result = await manager.synthesize_async(sample_tts_request)
            
            # Should fallback to pyttsx3
            assert result.status == TTSStatus.COMPLETED
            assert result.engine_used == TTSEngineType.PYTTSX3.value
            
    @pytest.mark.asyncio
    async def test_synthesize_all_engines_fail(self, sample_tts_request):
        """Test synthesis when all engines fail."""
        mock_failing_engine = Mock()
        mock_failing_engine.synthesize_async = AsyncMock(side_effect=Exception("Engine failed"))
        
        with patch('src.tts_service.engine_manager.Pyttsx3Engine', return_value=mock_failing_engine):
            manager = TTSEngineManager()
            manager.initialize_engines()
            
            result = await manager.synthesize_async(sample_tts_request)
            assert result.status == TTSStatus.FAILED
            assert result.error_message is not None
            
    def test_get_all_voices(self, sample_voices, mock_pyttsx3_engine, mock_coqui_tts):
        """Test getting all available voices from all engines."""
        pyttsx3_voices = [v for v in sample_voices if v.engine == "pyttsx3"]
        coqui_voices = [v for v in sample_voices if v.engine == "coqui"]
        
        mock_pyttsx3_engine.get_voices.return_value = pyttsx3_voices
        mock_coqui_tts.get_voices.return_value = coqui_voices
        
        with patch('src.tts_service.engine_manager.Pyttsx3Engine', return_value=mock_pyttsx3_engine), \
             patch('src.tts_service.engine_manager.CoquiTTSManager', return_value=mock_coqui_tts):
            
            manager = TTSEngineManager()
            manager.initialize_engines()
            
            all_voices = manager.get_all_voices()
            assert isinstance(all_voices, list)
            assert len(all_voices) == len(pyttsx3_voices) + len(coqui_voices)
            
    def test_get_voices_by_language(self, sample_voices, mock_pyttsx3_engine):
        """Test filtering voices by language."""
        en_voices = [v for v in sample_voices if v.language == "en-US"]
        mock_pyttsx3_engine.get_voices.return_value = en_voices
        
        with patch('src.tts_service.engine_manager.Pyttsx3Engine', return_value=mock_pyttsx3_engine):
            manager = TTSEngineManager()
            manager.initialize_engines()
            
            filtered_voices = manager.get_voices_by_language("en-US")
            assert all(v.language == "en-US" for v in filtered_voices)
            
    def test_get_voices_by_engine(self, sample_voices, mock_pyttsx3_engine):
        """Test filtering voices by engine."""
        pyttsx3_voices = [v for v in sample_voices if v.engine == "pyttsx3"]
        mock_pyttsx3_engine.get_voices.return_value = pyttsx3_voices
        
        with patch('src.tts_service.engine_manager.Pyttsx3Engine', return_value=mock_pyttsx3_engine):
            manager = TTSEngineManager()
            manager.initialize_engines()
            
            filtered_voices = manager.get_voices_by_engine(TTSEngineType.PYTTSX3)
            assert all(v.engine == "pyttsx3" for v in filtered_voices)
            
    def test_voice_selection_with_preferences(self, sample_voices, mock_engine_manager):
        """Test voice selection with user preferences."""
        manager = TTSEngineManager()
        
        # Test voice selection logic
        preferred_voice = manager.select_best_voice(
            language="en-US",
            gender="female",
            quality="high"
        )
        
        # Should select the best matching voice
        assert preferred_voice is not None
        
    def test_engine_performance_tracking(self, mock_pyttsx3_engine):
        """Test performance tracking for engines."""
        with patch('src.tts_service.engine_manager.Pyttsx3Engine', return_value=mock_pyttsx3_engine):
            manager = TTSEngineManager()
            manager.initialize_engines()
            
            # Track performance metrics
            manager.record_synthesis_time(TTSEngineType.PYTTSX3, 1.5)
            manager.record_synthesis_time(TTSEngineType.PYTTSX3, 2.0)
            
            stats = manager.get_engine_statistics(TTSEngineType.PYTTSX3)
            assert stats['total_requests'] == 2
            assert stats['average_time'] == 1.75
            
    def test_engine_resource_monitoring(self, mock_pyttsx3_engine):
        """Test monitoring engine resource usage."""
        with patch('src.tts_service.engine_manager.Pyttsx3Engine', return_value=mock_pyttsx3_engine):
            manager = TTSEngineManager()
            manager.initialize_engines()
            
            resource_usage = manager.get_resource_usage()
            assert 'memory_mb' in resource_usage
            assert 'cpu_percent' in resource_usage
            
    def test_cleanup_engines(self, mock_pyttsx3_engine, mock_coqui_tts):
        """Test proper cleanup of engine resources."""
        with patch('src.tts_service.engine_manager.Pyttsx3Engine', return_value=mock_pyttsx3_engine), \
             patch('src.tts_service.engine_manager.CoquiTTSManager', return_value=mock_coqui_tts):
            
            manager = TTSEngineManager()
            manager.initialize_engines()
            
            manager.cleanup()
            
            # Verify cleanup was called on all engines
            mock_pyttsx3_engine.cleanup.assert_called_once()
            mock_coqui_tts.cleanup.assert_called_once()


@pytest.mark.tts
@pytest.mark.engine
@pytest.mark.unit
class TestTTSEngineAdapter:
    """Test TTS Engine Adapter functionality."""
    
    def test_adapter_creation(self, mock_pyttsx3_engine):
        """Test creating engine adapter."""
        adapter = TTSEngineAdapter(mock_pyttsx3_engine, TTSEngineType.PYTTSX3)
        assert adapter.engine == mock_pyttsx3_engine
        assert adapter.engine_type == TTSEngineType.PYTTSX3
        
    def test_adapter_availability_check(self, mock_pyttsx3_engine):
        """Test adapter availability checking."""
        mock_pyttsx3_engine.is_available.return_value = True
        
        adapter = TTSEngineAdapter(mock_pyttsx3_engine, TTSEngineType.PYTTSX3)
        assert adapter.is_available() is True
        
    def test_adapter_health_check(self, mock_pyttsx3_engine):
        """Test adapter health checking."""
        mock_pyttsx3_engine.is_available.return_value = True
        
        adapter = TTSEngineAdapter(mock_pyttsx3_engine, TTSEngineType.PYTTSX3)
        health = adapter.check_health()
        
        assert isinstance(health, dict)
        assert 'is_available' in health
        assert 'engine_type' in health
        
    @pytest.mark.asyncio
    async def test_adapter_synthesis(self, mock_pyttsx3_engine, sample_tts_request):
        """Test adapter synthesis functionality."""
        mock_pyttsx3_engine.synthesize_async = AsyncMock(return_value=b"test audio")
        
        adapter = TTSEngineAdapter(mock_pyttsx3_engine, TTSEngineType.PYTTSX3)
        result = await adapter.synthesize_async(sample_tts_request)
        
        assert result == b"test audio"
        mock_pyttsx3_engine.synthesize_async.assert_called_once_with(sample_tts_request)
        
    def test_adapter_voice_management(self, mock_pyttsx3_engine, sample_voices):
        """Test adapter voice management."""
        pyttsx3_voices = [v for v in sample_voices if v.engine == "pyttsx3"]
        mock_pyttsx3_engine.get_voices.return_value = pyttsx3_voices
        
        adapter = TTSEngineAdapter(mock_pyttsx3_engine, TTSEngineType.PYTTSX3)
        voices = adapter.get_voices()
        
        assert voices == pyttsx3_voices
        mock_pyttsx3_engine.get_voices.assert_called_once()
        
    def test_adapter_error_handling(self, mock_pyttsx3_engine, sample_tts_request):
        """Test adapter error handling."""
        mock_pyttsx3_engine.synthesize_async = AsyncMock(side_effect=Exception("Engine error"))
        
        adapter = TTSEngineAdapter(mock_pyttsx3_engine, TTSEngineType.PYTTSX3)
        
        with pytest.raises(Exception):
            asyncio.run(adapter.synthesize_async(sample_tts_request))


@pytest.mark.tts
@pytest.mark.engine
@pytest.mark.integration
class TestEngineIntegration:
    """Test engine integration with real TTS engines (mocked)."""
    
    @pytest.mark.asyncio
    async def test_pyttsx3_integration(self, mock_pyttsx3_module, sample_tts_request):
        """Test integration with pyttsx3 engine."""
        with patch('src.tts_service.engine_manager.pyttsx3', mock_pyttsx3_module):
            manager = TTSEngineManager()
            manager.initialize_engines()
            
            # Should have pyttsx3 engine available
            assert TTSEngineType.PYTTSX3 in manager.engines
            
            # Test synthesis
            result = await manager.synthesize_async(sample_tts_request)
            assert result.status in [TTSStatus.COMPLETED, TTSStatus.FAILED]
            
    @pytest.mark.asyncio
    async def test_coqui_integration(self, mock_coqui_module, sample_tts_request):
        """Test integration with Coqui TTS engine."""
        with patch('src.tts_service.engine_manager.TTS', mock_coqui_module):
            manager = TTSEngineManager()
            manager.initialize_engines()
            
            # May or may not have Coqui depending on availability
            result = await manager.synthesize_async(sample_tts_request)
            assert result.status in [TTSStatus.COMPLETED, TTSStatus.FAILED]
            
    @pytest.mark.asyncio
    async def test_hybrid_engine_selection(self, mock_pyttsx3_module, mock_coqui_module, sample_tts_request):
        """Test hybrid engine selection logic."""
        with patch('src.tts_service.engine_manager.pyttsx3', mock_pyttsx3_module), \
             patch('src.tts_service.engine_manager.TTS', mock_coqui_module):
            
            manager = TTSEngineManager()
            manager.initialize_engines()
            
            # Request hybrid engine
            sample_tts_request.engine = TTSEngineType.HYBRID
            result = await manager.synthesize_async(sample_tts_request)
            
            # Should select the best available engine
            assert result.engine_used in ["pyttsx3", "coqui", "hybrid"]
            
    def test_engine_failover_scenario(self, mock_pyttsx3_module):
        """Test engine failover when primary engine becomes unavailable."""
        with patch('src.tts_service.engine_manager.pyttsx3', mock_pyttsx3_module):
            manager = TTSEngineManager()
            manager.initialize_engines()
            
            # Simulate engine failure
            original_engine = manager.engines[TTSEngineType.PYTTSX3]
            original_engine.is_available = Mock(return_value=False)
            
            # Health check should detect the failure
            health = manager.check_engine_health()
            assert health[TTSEngineType.PYTTSX3] is False
            
    def test_voice_caching(self, mock_pyttsx3_engine, sample_voices):
        """Test voice information caching for performance."""
        mock_pyttsx3_engine.get_voices.return_value = sample_voices
        
        with patch('src.tts_service.engine_manager.Pyttsx3Engine', return_value=mock_pyttsx3_engine):
            manager = TTSEngineManager()
            manager.initialize_engines()
            
            # First call should fetch voices
            voices1 = manager.get_all_voices()
            
            # Second call should use cache
            voices2 = manager.get_all_voices()
            
            assert voices1 == voices2
            # Should have called get_voices only once due to caching
            mock_pyttsx3_engine.get_voices.assert_called_once()


@pytest.mark.tts
@pytest.mark.engine
@pytest.mark.performance
class TestEnginePerformance:
    """Test engine performance and resource usage."""
    
    @pytest.mark.slow
    def test_engine_initialization_time(self):
        """Test engine initialization performance."""
        import time
        
        start_time = time.time()
        manager = TTSEngineManager()
        manager.initialize_engines()
        end_time = time.time()
        
        initialization_time = end_time - start_time
        assert initialization_time < 5.0  # Should initialize within 5 seconds
        
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_synthesis_performance(self, mock_pyttsx3_engine, sample_tts_request):
        """Test synthesis performance under load."""
        mock_pyttsx3_engine.synthesize_async = AsyncMock(return_value=b"test audio")
        
        with patch('src.tts_service.engine_manager.Pyttsx3Engine', return_value=mock_pyttsx3_engine):
            manager = TTSEngineManager()
            manager.initialize_engines()
            
            # Perform multiple synthesis operations
            import time
            start_time = time.time()
            
            tasks = []
            for _ in range(10):
                task = manager.synthesize_async(sample_tts_request)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            total_time = end_time - start_time
            assert total_time < 10.0  # Should complete within 10 seconds
            assert all(r.status == TTSStatus.COMPLETED for r in results)
            
    def test_memory_usage_monitoring(self, mock_pyttsx3_engine):
        """Test monitoring memory usage of engines."""
        with patch('src.tts_service.engine_manager.Pyttsx3Engine', return_value=mock_pyttsx3_engine):
            manager = TTSEngineManager()
            manager.initialize_engines()
            
            # Get initial memory usage
            initial_usage = manager.get_resource_usage()
            
            # Simulate heavy usage
            for _ in range(100):
                manager.record_synthesis_time(TTSEngineType.PYTTSX3, 1.0)
            
            # Memory usage should be tracked
            current_usage = manager.get_resource_usage()
            assert 'memory_mb' in current_usage
            assert current_usage['memory_mb'] >= 0