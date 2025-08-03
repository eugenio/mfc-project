"""
TTS Service API Server Tests
===========================

Comprehensive test suite for the TTS Service FastAPI server implementation.
These tests follow TDD principles - they are written BEFORE the implementation
and define the expected behavior of the API endpoints.

Test Categories:
- API endpoint functionality
- Request/response validation
- Error handling and status codes
- Authentication and rate limiting
- Health checks and monitoring
- Concurrent request handling

Created: 2025-08-03
Author: Agent Gamma - TTS Service Implementation Lead
"""
import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import status
import json

# These imports will fail initially (TDD - Red phase)
# They define what we need to implement
from src.tts_service.api_server import TTSAPIServer, create_app
from src.tts_service.models import (
    TTSRequest, TTSResponse, TTSStatus, VoiceInfo,
    HealthResponse, QueueStatus, TTSError, BatchTTSRequest
)


@pytest.mark.tts
@pytest.mark.api
@pytest.mark.unit
class TestTTSAPIServerCreation:
    """Test TTS API server creation and initialization."""
    
    def test_create_api_server_instance(self):
        """Test creating TTS API server instance."""
        server = TTSAPIServer()
        assert server is not None
        assert hasattr(server, 'app')  # Should have FastAPI app
        assert hasattr(server, 'engine_manager')  # Should have engine manager
        assert hasattr(server, 'queue_processor')  # Should have queue processor
        
    def test_create_api_server_with_config(self, test_config):
        """Test creating API server with custom configuration."""
        server = TTSAPIServer(config=test_config)
        assert server.config['host'] == test_config['host']
        assert server.config['port'] == test_config['port']
        assert server.config['max_queue_size'] == test_config['max_queue_size']
        
    def test_create_fastapi_app(self):
        """Test creating FastAPI application."""
        app = create_app()
        assert app is not None
        assert app.title == "TTS Service API"
        assert app.version.startswith("1.")
        
    def test_server_has_required_routes(self):
        """Test that server has all required API routes."""
        app = create_app()
        route_paths = [route.path for route in app.routes]
        
        required_routes = [
            "/health",
            "/synthesize",
            "/synthesize/batch",
            "/voices",
            "/queue/status",
            "/engines",
            "/engines/{engine_id}/status"
        ]
        
        for route in required_routes:
            assert route in route_paths or any(route in path for path in route_paths)


@pytest.mark.tts
@pytest.mark.api
@pytest.mark.integration
class TestHealthCheckEndpoint:
    """Test /health endpoint functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)
        
    def test_health_check_returns_200(self, client, mock_engine_manager, mock_queue_processor):
        """Test health check endpoint returns 200 OK."""
        with patch('src.tts_service.api_server.engine_manager', mock_engine_manager), \
             patch('src.tts_service.api_server.queue_processor', mock_queue_processor):
            response = client.get("/health")
            assert response.status_code == status.HTTP_200_OK
            
    def test_health_check_response_structure(self, client, sample_health_response):
        """Test health check response has correct structure."""
        with patch('src.tts_service.api_server.get_health_status', return_value=sample_health_response):
            response = client.get("/health")
            data = response.json()
            
            # Check required fields
            assert "service_name" in data
            assert "version" in data
            assert "status" in data
            assert "uptime_seconds" in data
            assert "engines" in data
            assert "queue_status" in data
            assert "memory_usage_mb" in data
            assert "total_requests_processed" in data
            
    def test_health_check_with_unhealthy_engines(self, client):
        """Test health check when engines are unhealthy."""
        mock_health = HealthResponse(
            version="1.0.0",
            status="degraded",
            uptime_seconds=3600.0,
            engines=[],
            healthy_engines=0,
            total_engines=2,
            queue_status=QueueStatus(
                total_requests=0, pending_requests=0, processing_requests=0,
                completed_requests=0, failed_requests=0, average_processing_time=0.0,
                queue_throughput=0.0, estimated_wait_time=0.0, priority_breakdown={}
            ),
            queue_health="healthy",
            memory_usage_mb=150.0,
            cpu_usage_percent=50.0,
            total_requests_processed=0,
            requests_per_minute=0.0,
            error_rate_percent=100.0
        )
        
        with patch('src.tts_service.api_server.get_health_status', return_value=mock_health):
            response = client.get("/health")
            data = response.json()
            assert data["status"] == "degraded"
            assert data["healthy_engines"] == 0


@pytest.mark.tts
@pytest.mark.api
@pytest.mark.integration
class TestSynthesizeEndpoint:
    """Test /synthesize endpoint functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)
        
    def test_synthesize_valid_request(self, client, sample_tts_request):
        """Test synthesize endpoint with valid request."""
        request_data = sample_tts_request.dict()
        
        with patch('src.tts_service.api_server.process_tts_request') as mock_process:
            mock_response = TTSResponse(
                request_id="test-123",
                status=TTSStatus.COMPLETED,
                audio_data=b"mock audio",
                audio_format="wav",
                processing_time=1.0
            )
            mock_process.return_value = mock_response
            
            response = client.post("/synthesize", json=request_data)
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert data["request_id"] == "test-123"
            assert data["status"] == "completed"
            
    def test_synthesize_invalid_request(self, client):
        """Test synthesize endpoint with invalid request data."""
        invalid_data = {"text": ""}  # Empty text
        
        response = client.post("/synthesize", json=invalid_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
    def test_synthesize_missing_text(self, client):
        """Test synthesize endpoint with missing text field."""
        invalid_data = {"voice_id": "test-voice"}  # Missing required text
        
        response = client.post("/synthesize", json=invalid_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
    def test_synthesize_text_too_long(self, client):
        """Test synthesize endpoint with text exceeding length limit."""
        long_text = "a" * 10000  # Exceeds max length
        request_data = {"text": long_text}
        
        response = client.post("/synthesize", json=request_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
    def test_synthesize_with_priority(self, client, urgent_tts_request):
        """Test synthesize endpoint with urgent priority."""
        request_data = urgent_tts_request.dict()
        
        with patch('src.tts_service.api_server.process_tts_request') as mock_process:
            mock_response = TTSResponse(
                request_id="urgent-123",
                status=TTSStatus.COMPLETED,
                processing_time=0.5  # Should be faster for urgent
            )
            mock_process.return_value = mock_response
            
            response = client.post("/synthesize", json=request_data)
            assert response.status_code == status.HTTP_200_OK
            
            # Verify priority was processed
            call_args = mock_process.call_args[0][0]
            assert call_args.priority == "urgent"
            
    def test_synthesize_async_processing(self, client, sample_tts_request):
        """Test synthesize endpoint with async processing."""
        request_data = sample_tts_request.dict()
        request_data["async_processing"] = True
        
        with patch('src.tts_service.api_server.queue_tts_request') as mock_queue:
            mock_queue.return_value = "queued-request-123"
            
            response = client.post("/synthesize", json=request_data)
            assert response.status_code == status.HTTP_202_ACCEPTED
            
            data = response.json()
            assert data["request_id"] == "queued-request-123"
            assert data["status"] == "pending"
            
    def test_synthesize_engine_unavailable(self, client, sample_tts_request):
        """Test synthesize when requested engine is unavailable."""
        request_data = sample_tts_request.dict()
        
        with patch('src.tts_service.api_server.process_tts_request') as mock_process:
            mock_process.side_effect = Exception("Engine unavailable")
            
            response = client.post("/synthesize", json=request_data)
            assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
            
    def test_synthesize_rate_limiting(self, client, sample_tts_request):
        """Test rate limiting on synthesize endpoint."""
        request_data = sample_tts_request.dict()
        
        # Make many rapid requests to trigger rate limiting
        responses = []
        for _ in range(100):  # Exceed rate limit
            response = client.post("/synthesize", json=request_data)
            responses.append(response)
            
        # At least one should be rate limited
        rate_limited = any(r.status_code == status.HTTP_429_TOO_MANY_REQUESTS for r in responses)
        assert rate_limited


@pytest.mark.tts
@pytest.mark.api
@pytest.mark.integration
class TestBatchSynthesizeEndpoint:
    """Test /synthesize/batch endpoint functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)
        
    def test_batch_synthesize_valid_request(self, client, batch_tts_requests):
        """Test batch synthesize with valid requests."""
        batch_data = {
            "requests": [req.dict() for req in batch_tts_requests],
            "batch_id": "test-batch-123"
        }
        
        with patch('src.tts_service.api_server.process_batch_tts_request') as mock_process:
            mock_response = {
                "batch_id": "test-batch-123",
                "total_requests": len(batch_tts_requests),
                "status": "processing"
            }
            mock_process.return_value = mock_response
            
            response = client.post("/synthesize/batch", json=batch_data)
            assert response.status_code == status.HTTP_202_ACCEPTED
            
            data = response.json()
            assert data["batch_id"] == "test-batch-123"
            assert data["total_requests"] == len(batch_tts_requests)
            
    def test_batch_synthesize_empty_requests(self, client):
        """Test batch synthesize with empty request list."""
        batch_data = {"requests": []}
        
        response = client.post("/synthesize/batch", json=batch_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
    def test_batch_synthesize_too_many_requests(self, client):
        """Test batch synthesize with too many requests."""
        # Create more than the allowed limit
        large_batch = {
            "requests": [{"text": f"Message {i}"} for i in range(200)]  # Exceeds limit
        }
        
        response = client.post("/synthesize/batch", json=large_batch)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.tts
@pytest.mark.api
@pytest.mark.integration
class TestVoicesEndpoint:
    """Test /voices endpoint functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)
        
    def test_get_voices_returns_list(self, client, sample_voices):
        """Test voices endpoint returns list of available voices."""
        with patch('src.tts_service.api_server.get_available_voices', return_value=sample_voices):
            response = client.get("/voices")
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == len(sample_voices)
            
    def test_get_voices_structure(self, client, sample_voices):
        """Test voices endpoint response structure."""
        with patch('src.tts_service.api_server.get_available_voices', return_value=sample_voices):
            response = client.get("/voices")
            data = response.json()
            
            for voice in data:
                assert "id" in voice
                assert "name" in voice
                assert "language" in voice
                assert "engine" in voice
                
    def test_get_voices_filtered_by_language(self, client, sample_voices):
        """Test voices endpoint with language filter."""
        with patch('src.tts_service.api_server.get_available_voices') as mock_get:
            mock_get.return_value = [v for v in sample_voices if v.language == "en-US"]
            
            response = client.get("/voices?language=en-US")
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            for voice in data:
                assert voice["language"] == "en-US"
                
    def test_get_voices_filtered_by_engine(self, client, sample_voices):
        """Test voices endpoint with engine filter."""
        with patch('src.tts_service.api_server.get_available_voices') as mock_get:
            mock_get.return_value = [v for v in sample_voices if v.engine == "coqui"]
            
            response = client.get("/voices?engine=coqui")
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            for voice in data:
                assert voice["engine"] == "coqui"


@pytest.mark.tts
@pytest.mark.api
@pytest.mark.integration
class TestQueueStatusEndpoint:
    """Test /queue/status endpoint functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)
        
    def test_queue_status_returns_metrics(self, client, mock_queue_processor):
        """Test queue status endpoint returns processing metrics."""
        with patch('src.tts_service.api_server.queue_processor', mock_queue_processor):
            response = client.get("/queue/status")
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert "total_requests" in data
            assert "pending_requests" in data
            assert "processing_requests" in data
            assert "average_processing_time" in data
            assert "queue_throughput" in data
            
    def test_queue_status_with_active_requests(self, client):
        """Test queue status with active processing requests."""
        mock_status = QueueStatus(
            total_requests=10,
            pending_requests=3,
            processing_requests=2,
            completed_requests=5,
            failed_requests=0,
            average_processing_time=1.5,
            queue_throughput=8.0,
            estimated_wait_time=2.25,
            priority_breakdown={"normal": 8, "high": 2}
        )
        
        with patch('src.tts_service.api_server.get_queue_status', return_value=mock_status):
            response = client.get("/queue/status")
            data = response.json()
            
            assert data["total_requests"] == 10
            assert data["pending_requests"] == 3
            assert data["processing_requests"] == 2
            assert data["estimated_wait_time"] == 2.25


@pytest.mark.tts
@pytest.mark.api
@pytest.mark.integration
class TestEnginesEndpoint:
    """Test /engines endpoint functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)
        
    def test_get_engines_list(self, client, mock_engine_manager):
        """Test engines endpoint returns list of available engines."""
        mock_engine_manager.get_engine_status.return_value = [
            {"engine_id": "pyttsx3", "is_available": True, "is_healthy": True},
            {"engine_id": "coqui", "is_available": True, "is_healthy": False}
        ]
        
        with patch('src.tts_service.api_server.engine_manager', mock_engine_manager):
            response = client.get("/engines")
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 2
            
    def test_get_specific_engine_status(self, client, mock_engine_manager):
        """Test getting status of specific engine."""
        mock_status = {
            "engine_id": "coqui",
            "engine_type": "neural",
            "is_available": True,
            "is_healthy": True,
            "total_requests": 42,
            "average_latency": 1.2
        }
        
        mock_engine_manager.get_engine_status.return_value = mock_status
        
        with patch('src.tts_service.api_server.engine_manager', mock_engine_manager):
            response = client.get("/engines/coqui/status")
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert data["engine_id"] == "coqui"
            assert data["is_available"] is True
            
    def test_get_nonexistent_engine_status(self, client, mock_engine_manager):
        """Test getting status of nonexistent engine."""
        mock_engine_manager.get_engine_status.side_effect = KeyError("Engine not found")
        
        with patch('src.tts_service.api_server.engine_manager', mock_engine_manager):
            response = client.get("/engines/nonexistent/status")
            assert response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.tts
@pytest.mark.api
@pytest.mark.performance
class TestAPIPerformance:
    """Test API performance and concurrent request handling."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)
        
    @pytest.mark.slow
    def test_concurrent_synthesize_requests(self, client, sample_tts_request, performance_test_config):
        """Test handling multiple concurrent synthesize requests."""
        import threading
        import time
        
        request_data = sample_tts_request.dict()
        results = []
        
        def make_request():
            with patch('src.tts_service.api_server.process_tts_request') as mock_process:
                mock_response = TTSResponse(
                    request_id=f"perf-test-{threading.current_thread().ident}",
                    status=TTSStatus.COMPLETED,
                    processing_time=0.1
                )
                mock_process.return_value = mock_response
                
                start_time = time.time()
                response = client.post("/synthesize", json=request_data)
                end_time = time.time()
                
                results.append({
                    "status_code": response.status_code,
                    "response_time": end_time - start_time
                })
        
        # Launch concurrent requests
        threads = []
        concurrent_requests = performance_test_config["concurrent_requests"]
        
        for _ in range(concurrent_requests):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Analyze results
        successful_requests = [r for r in results if r["status_code"] == 200]
        assert len(successful_requests) >= concurrent_requests * 0.9  # 90% success rate
        
        avg_response_time = sum(r["response_time"] for r in successful_requests) / len(successful_requests)
        assert avg_response_time < performance_test_config["max_response_time"]
        
    def test_health_check_performance(self, client):
        """Test health check endpoint performance."""
        import time
        
        # Make multiple health check requests
        response_times = []
        for _ in range(10):
            start_time = time.time()
            response = client.get("/health")
            end_time = time.time()
            
            assert response.status_code == 200
            response_times.append(end_time - start_time)
        
        # Health checks should be very fast
        avg_response_time = sum(response_times) / len(response_times)
        assert avg_response_time < 0.1  # 100ms
        
    def test_voices_endpoint_caching(self, client):
        """Test that voices endpoint uses caching for performance."""
        import time
        
        # First request (should populate cache)
        start_time = time.time()
        response1 = client.get("/voices")
        first_request_time = time.time() - start_time
        
        # Second request (should use cache)
        start_time = time.time()
        response2 = client.get("/voices")
        second_request_time = time.time() - start_time
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # Second request should be significantly faster due to caching
        assert second_request_time < first_request_time * 0.5


@pytest.mark.tts
@pytest.mark.api
@pytest.mark.integration
class TestErrorHandling:
    """Test API error handling and edge cases."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)
        
    def test_malformed_json_request(self, client):
        """Test handling of malformed JSON in requests."""
        response = client.post(
            "/synthesize",
            data="{ invalid json }",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
    def test_unsupported_content_type(self, client):
        """Test handling of unsupported content types."""
        response = client.post(
            "/synthesize",
            data="text=hello",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        # Should still work as FastAPI can handle form data
        assert response.status_code in [200, 422]  # Either processed or validation error
        
    def test_request_timeout_handling(self, client, sample_tts_request):
        """Test handling of request timeouts."""
        request_data = sample_tts_request.dict()
        
        with patch('src.tts_service.api_server.process_tts_request') as mock_process:
            import asyncio
            mock_process.side_effect = asyncio.TimeoutError("Request timeout")
            
            response = client.post("/synthesize", json=request_data)
            assert response.status_code == status.HTTP_408_REQUEST_TIMEOUT
            
    def test_internal_server_error_handling(self, client, sample_tts_request):
        """Test handling of internal server errors."""
        request_data = sample_tts_request.dict()
        
        with patch('src.tts_service.api_server.process_tts_request') as mock_process:
            mock_process.side_effect = Exception("Unexpected error")
            
            response = client.post("/synthesize", json=request_data)
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            
            data = response.json()
            assert "error" in data
            
    def test_404_for_unknown_endpoints(self, client):
        """Test 404 response for unknown endpoints."""
        response = client.get("/unknown/endpoint")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        
    def test_method_not_allowed(self, client):
        """Test 405 response for wrong HTTP methods."""
        response = client.delete("/synthesize")  # DELETE not allowed
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED