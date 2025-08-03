"""
TTS Queue Processor Tests
========================

Comprehensive test suite for the TTS Queue Processor implementation.
Tests the async request processing queue with priority handling and
concurrent processing capabilities.

Test Categories:
- Queue operations (add, process, complete)
- Priority handling and ordering
- Concurrent request processing
- Queue status and metrics
- Error handling and recovery
- Performance and load testing

Created: 2025-08-03
Author: Agent Gamma - TTS Service Implementation Lead
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any
import time

# These imports will fail initially (TDD - Red phase)
from src.tts_service.queue_processor import TTSQueueProcessor, QueuedRequest
from src.tts_service.models import (
    TTSRequest, TTSResponse, TTSStatus, TTSPriority,
    QueueStatus, TTSEngineType
)


@pytest.mark.tts
@pytest.mark.queue
@pytest.mark.unit
class TestTTSQueueProcessor:
    """Test TTS Queue Processor functionality."""
    
    def test_queue_processor_creation(self):
        """Test creating queue processor instance."""
        processor = TTSQueueProcessor()
        assert processor is not None
        assert hasattr(processor, 'queue')
        assert hasattr(processor, 'processing_requests')
        assert hasattr(processor, 'completed_requests')
        
    def test_queue_processor_with_config(self, test_config):
        """Test creating queue processor with configuration."""
        processor = TTSQueueProcessor(
            max_queue_size=test_config['max_queue_size'],
            max_concurrent=test_config['max_concurrent_requests']
        )
        assert processor.max_queue_size == test_config['max_queue_size']
        assert processor.max_concurrent == test_config['max_concurrent_requests']
        
    @pytest.mark.asyncio
    async def test_start_stop_processor(self):
        """Test starting and stopping queue processor."""
        processor = TTSQueueProcessor()
        
        # Initially not running
        assert not processor.is_running
        
        # Start processor
        await processor.start()
        assert processor.is_running
        
        # Stop processor
        await processor.stop()
        assert not processor.is_running
        
    @pytest.mark.asyncio
    async def test_add_request_to_queue(self, sample_tts_request):
        """Test adding request to processing queue."""
        processor = TTSQueueProcessor()
        
        request_id = await processor.add_request(sample_tts_request)
        assert request_id is not None
        assert processor.queue_size == 1
        
    @pytest.mark.asyncio
    async def test_add_request_with_priority(self, urgent_tts_request):
        """Test adding high priority request to queue."""
        processor = TTSQueueProcessor()
        
        request_id = await processor.add_request(urgent_tts_request)
        assert request_id is not None
        
        # Should be at front of queue due to priority
        queued_request = processor._get_next_request()
        assert queued_request.request.priority == TTSPriority.URGENT
        
    @pytest.mark.asyncio
    async def test_queue_size_limit(self, sample_tts_request):
        """Test queue size limiting."""
        processor = TTSQueueProcessor(max_queue_size=2)
        
        # Add requests up to limit
        await processor.add_request(sample_tts_request)
        await processor.add_request(sample_tts_request)
        
        # Adding another should raise exception or return error
        with pytest.raises(Exception):
            await processor.add_request(sample_tts_request)
            
    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """Test that requests are processed in priority order."""
        processor = TTSQueueProcessor()
        
        # Add requests with different priorities
        low_req = TTSRequest(text="Low priority", priority=TTSPriority.LOW)
        high_req = TTSRequest(text="High priority", priority=TTSPriority.HIGH)
        urgent_req = TTSRequest(text="Urgent priority", priority=TTSPriority.URGENT)
        normal_req = TTSRequest(text="Normal priority", priority=TTSPriority.NORMAL)
        
        # Add in random order
        await processor.add_request(low_req)
        await processor.add_request(normal_req)
        await processor.add_request(urgent_req)
        await processor.add_request(high_req)
        
        # Should be processed in priority order: URGENT, HIGH, NORMAL, LOW
        first = processor._get_next_request()
        assert first.request.priority == TTSPriority.URGENT
        
        second = processor._get_next_request()
        assert second.request.priority == TTSPriority.HIGH
        
    @pytest.mark.asyncio
    async def test_concurrent_processing_limit(self, mock_engine_manager):
        """Test concurrent processing limit enforcement."""
        processor = TTSQueueProcessor(max_concurrent=2)
        mock_engine_manager.synthesize_async = AsyncMock(return_value=TTSResponse(
            request_id="test", status=TTSStatus.COMPLETED
        ))
        
        processor.engine_manager = mock_engine_manager
        await processor.start()
        
        # Add more requests than concurrent limit
        requests = [TTSRequest(text=f"Test {i}") for i in range(5)]
        for req in requests:
            await processor.add_request(req)
        
        # Wait briefly for processing to start
        await asyncio.sleep(0.1)
        
        # Should have at most 2 requests processing concurrently
        assert len(processor.processing_requests) <= 2
        
        await processor.stop()
        
    @pytest.mark.asyncio
    async def test_request_processing_success(self, sample_tts_request, mock_engine_manager):
        """Test successful request processing."""
        mock_response = TTSResponse(
            request_id="test-123",
            status=TTSStatus.COMPLETED,
            audio_data=b"test audio",
            processing_time=1.0
        )
        mock_engine_manager.synthesize_async = AsyncMock(return_value=mock_response)
        
        processor = TTSQueueProcessor()
        processor.engine_manager = mock_engine_manager
        await processor.start()
        
        request_id = await processor.add_request(sample_tts_request)
        
        # Wait for processing to complete
        await asyncio.sleep(0.2)
        
        # Request should be completed
        assert processor.queue_size == 0
        assert len(processor.completed_requests) == 1
        
        await processor.stop()
        
    @pytest.mark.asyncio
    async def test_request_processing_failure(self, sample_tts_request, mock_engine_manager):
        """Test failed request processing."""
        mock_engine_manager.synthesize_async = AsyncMock(
            side_effect=Exception("Synthesis failed")
        )
        
        processor = TTSQueueProcessor()
        processor.engine_manager = mock_engine_manager
        await processor.start()
        
        request_id = await processor.add_request(sample_tts_request)
        
        # Wait for processing to complete
        await asyncio.sleep(0.2)
        
        # Request should be failed
        assert processor.queue_size == 0
        assert len(processor.failed_requests) == 1
        
        await processor.stop()
        
    @pytest.mark.asyncio
    async def test_get_request_status(self, sample_tts_request, mock_engine_manager):
        """Test getting status of specific request."""
        processor = TTSQueueProcessor()
        processor.engine_manager = mock_engine_manager
        
        request_id = await processor.add_request(sample_tts_request)
        
        status = await processor.get_request_status(request_id)
        assert status.status == TTSStatus.PENDING
        assert status.request_id == request_id
        
    @pytest.mark.asyncio
    async def test_cancel_request(self, sample_tts_request):
        """Test cancelling pending request."""
        processor = TTSQueueProcessor()
        
        request_id = await processor.add_request(sample_tts_request)
        
        # Cancel the request
        success = await processor.cancel_request(request_id)
        assert success is True
        
        # Request should be removed from queue
        assert processor.queue_size == 0
        
    @pytest.mark.asyncio
    async def test_cancel_processing_request(self, sample_tts_request, mock_engine_manager):
        """Test cancelling request that's currently processing."""
        # Mock a slow synthesis operation
        mock_engine_manager.synthesize_async = AsyncMock()
        mock_engine_manager.synthesize_async.side_effect = lambda x: asyncio.sleep(5)
        
        processor = TTSQueueProcessor()
        processor.engine_manager = mock_engine_manager
        await processor.start()
        
        request_id = await processor.add_request(sample_tts_request)
        
        # Wait for processing to start
        await asyncio.sleep(0.1)
        
        # Try to cancel
        success = await processor.cancel_request(request_id)
        
        # Should successfully cancel even if processing
        assert success is True
        
        await processor.stop()
        
    @pytest.mark.asyncio
    async def test_queue_status_metrics(self, mock_engine_manager):
        """Test queue status and metrics reporting."""
        processor = TTSQueueProcessor()
        processor.engine_manager = mock_engine_manager
        
        # Add some requests
        for i in range(5):
            req = TTSRequest(text=f"Test {i}")
            await processor.add_request(req)
        
        status = await processor.get_queue_status()
        
        assert isinstance(status, QueueStatus)
        assert status.total_requests == 5
        assert status.pending_requests == 5
        assert status.processing_requests == 0
        assert status.completed_requests == 0
        
    @pytest.mark.asyncio
    async def test_queue_throughput_calculation(self, mock_engine_manager):
        """Test queue throughput calculation."""
        mock_response = TTSResponse(
            request_id="test",
            status=TTSStatus.COMPLETED,
            processing_time=0.5
        )
        mock_engine_manager.synthesize_async = AsyncMock(return_value=mock_response)
        
        processor = TTSQueueProcessor()
        processor.engine_manager = mock_engine_manager
        await processor.start()
        
        # Process several requests
        start_time = time.time()
        for i in range(10):
            req = TTSRequest(text=f"Test {i}")
            await processor.add_request(req)
        
        # Wait for processing
        await asyncio.sleep(1.0)
        
        status = await processor.get_queue_status()
        
        # Should have some throughput metrics
        assert status.queue_throughput >= 0
        assert status.average_processing_time >= 0
        
        await processor.stop()
        
    @pytest.mark.asyncio
    async def test_estimated_wait_time(self, mock_engine_manager):
        """Test estimated wait time calculation."""
        # Mock slow synthesis
        mock_response = TTSResponse(
            request_id="test",
            status=TTSStatus.COMPLETED,
            processing_time=2.0
        )
        mock_engine_manager.synthesize_async = AsyncMock(return_value=mock_response)
        
        processor = TTSQueueProcessor(max_concurrent=1)  # Single thread
        processor.engine_manager = mock_engine_manager
        
        # Add multiple requests
        for i in range(5):
            req = TTSRequest(text=f"Test {i}")
            await processor.add_request(req)
        
        status = await processor.get_queue_status()
        
        # Should have realistic wait time estimate
        assert status.estimated_wait_time > 0
        assert status.estimated_wait_time < 100  # Reasonable upper bound
        
    @pytest.mark.asyncio
    async def test_priority_breakdown(self):
        """Test priority breakdown in queue status."""
        processor = TTSQueueProcessor()
        
        # Add requests with different priorities
        await processor.add_request(TTSRequest(text="Low", priority=TTSPriority.LOW))
        await processor.add_request(TTSRequest(text="Normal", priority=TTSPriority.NORMAL))
        await processor.add_request(TTSRequest(text="High", priority=TTSPriority.HIGH))
        await processor.add_request(TTSRequest(text="Urgent", priority=TTSPriority.URGENT))
        await processor.add_request(TTSRequest(text="Normal2", priority=TTSPriority.NORMAL))
        
        status = await processor.get_queue_status()
        
        expected_breakdown = {
            "low": 1,
            "normal": 2,
            "high": 1,
            "urgent": 1
        }
        
        assert status.priority_breakdown == expected_breakdown
        
    @pytest.mark.asyncio
    async def test_request_timeout_handling(self, sample_tts_request, mock_engine_manager):
        """Test handling of request timeouts."""
        # Mock synthesis that never completes
        mock_engine_manager.synthesize_async = AsyncMock()
        mock_engine_manager.synthesize_async.side_effect = lambda x: asyncio.sleep(1000)
        
        processor = TTSQueueProcessor(request_timeout=0.5)  # Short timeout
        processor.engine_manager = mock_engine_manager
        await processor.start()
        
        request_id = await processor.add_request(sample_tts_request)
        
        # Wait for timeout
        await asyncio.sleep(1.0)
        
        # Request should be failed due to timeout
        assert len(processor.failed_requests) == 1
        
        await processor.stop()
        
    @pytest.mark.asyncio
    async def test_queue_cleanup_old_requests(self, mock_engine_manager):
        """Test cleanup of old completed/failed requests."""
        processor = TTSQueueProcessor(max_completed_requests=2)
        
        # Add and complete several requests
        mock_response = TTSResponse(
            request_id="test",
            status=TTSStatus.COMPLETED
        )
        mock_engine_manager.synthesize_async = AsyncMock(return_value=mock_response)
        processor.engine_manager = mock_engine_manager
        
        await processor.start()
        
        # Add many requests
        for i in range(5):
            req = TTSRequest(text=f"Test {i}")
            await processor.add_request(req)
        
        # Wait for processing
        await asyncio.sleep(0.5)
        
        # Should have cleaned up old completed requests
        assert len(processor.completed_requests) <= 2
        
        await processor.stop()


@pytest.mark.tts
@pytest.mark.queue
@pytest.mark.unit
class TestQueuedRequest:
    """Test QueuedRequest data structure."""
    
    def test_queued_request_creation(self, sample_tts_request):
        """Test creating queued request."""
        queued = QueuedRequest(
            request_id="test-123",
            request=sample_tts_request,
            created_at=datetime.now()
        )
        
        assert queued.request_id == "test-123"
        assert queued.request == sample_tts_request
        assert queued.status == TTSStatus.PENDING
        
    def test_queued_request_priority_comparison(self):
        """Test priority-based comparison of queued requests."""
        high_req = QueuedRequest(
            request_id="high",
            request=TTSRequest(text="High", priority=TTSPriority.HIGH),
            created_at=datetime.now()
        )
        
        low_req = QueuedRequest(
            request_id="low", 
            request=TTSRequest(text="Low", priority=TTSPriority.LOW),
            created_at=datetime.now()
        )
        
        # High priority should be "less than" low priority (for priority queue)
        assert high_req < low_req
        
    def test_queued_request_timestamp_comparison(self):
        """Test timestamp-based comparison for same priority."""
        now = datetime.now()
        earlier = now - timedelta(seconds=10)
        
        req1 = QueuedRequest(
            request_id="req1",
            request=TTSRequest(text="First", priority=TTSPriority.NORMAL),
            created_at=earlier
        )
        
        req2 = QueuedRequest(
            request_id="req2",
            request=TTSRequest(text="Second", priority=TTSPriority.NORMAL),
            created_at=now
        )
        
        # Earlier request should be processed first
        assert req1 < req2
        
    def test_queued_request_update_status(self, sample_tts_request):
        """Test updating request status."""
        queued = QueuedRequest(
            request_id="test",
            request=sample_tts_request,
            created_at=datetime.now()
        )
        
        # Update to processing
        queued.status = TTSStatus.PROCESSING
        queued.started_at = datetime.now()
        
        assert queued.status == TTSStatus.PROCESSING
        assert queued.started_at is not None
        
        # Update to completed
        queued.status = TTSStatus.COMPLETED
        queued.completed_at = datetime.now()
        
        assert queued.status == TTSStatus.COMPLETED
        assert queued.completed_at is not None


@pytest.mark.tts
@pytest.mark.queue
@pytest.mark.integration
class TestQueueIntegration:
    """Test queue processor integration with engine manager."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_processing(self, sample_tts_request, mock_engine_manager):
        """Test complete end-to-end request processing."""
        mock_response = TTSResponse(
            request_id="e2e-test",
            status=TTSStatus.COMPLETED,
            audio_data=b"test audio data",
            processing_time=1.0
        )
        mock_engine_manager.synthesize_async = AsyncMock(return_value=mock_response)
        
        processor = TTSQueueProcessor()
        processor.engine_manager = mock_engine_manager
        await processor.start()
        
        # Add request
        request_id = await processor.add_request(sample_tts_request)
        
        # Wait for processing
        await asyncio.sleep(0.3)
        
        # Get final status
        final_status = await processor.get_request_status(request_id)
        
        assert final_status.status == TTSStatus.COMPLETED
        assert final_status.audio_data == b"test audio data"
        
        await processor.stop()
        
    @pytest.mark.asyncio
    async def test_batch_processing(self, batch_tts_requests, mock_engine_manager):
        """Test processing batch of requests."""
        mock_response = TTSResponse(
            request_id="batch-test",
            status=TTSStatus.COMPLETED,
            processing_time=0.5
        )
        mock_engine_manager.synthesize_async = AsyncMock(return_value=mock_response)
        
        processor = TTSQueueProcessor(max_concurrent=3)
        processor.engine_manager = mock_engine_manager
        await processor.start()
        
        # Add all requests
        request_ids = []
        for req in batch_tts_requests:
            request_id = await processor.add_request(req)
            request_ids.append(request_id)
        
        # Wait for all to complete
        await asyncio.sleep(2.0)
        
        # All should be completed
        for request_id in request_ids:
            status = await processor.get_request_status(request_id)
            assert status.status == TTSStatus.COMPLETED
        
        await processor.stop()
        
    @pytest.mark.asyncio
    async def test_mixed_priority_processing(self, mock_engine_manager):
        """Test processing mixed priority requests."""
        mock_response = TTSResponse(
            request_id="mixed-test",
            status=TTSStatus.COMPLETED,
            processing_time=0.1
        )
        mock_engine_manager.synthesize_async = AsyncMock(return_value=mock_response)
        
        processor = TTSQueueProcessor(max_concurrent=1)  # Sequential processing
        processor.engine_manager = mock_engine_manager
        await processor.start()
        
        # Add requests in non-priority order
        low_id = await processor.add_request(TTSRequest(text="Low", priority=TTSPriority.LOW))
        urgent_id = await processor.add_request(TTSRequest(text="Urgent", priority=TTSPriority.URGENT))
        normal_id = await processor.add_request(TTSRequest(text="Normal", priority=TTSPriority.NORMAL))
        
        # Wait for processing
        await asyncio.sleep(1.0)
        
        # Get completion order from processor history
        completed = list(processor.completed_requests.values())
        completed.sort(key=lambda x: x.completed_at)
        
        # Should process in priority order: URGENT, NORMAL, LOW
        assert completed[0].request.priority == TTSPriority.URGENT
        assert completed[1].request.priority == TTSPriority.NORMAL
        assert completed[2].request.priority == TTSPriority.LOW
        
        await processor.stop()


@pytest.mark.tts
@pytest.mark.queue
@pytest.mark.performance
class TestQueuePerformance:
    """Test queue processor performance and scalability."""
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_high_throughput_processing(self, mock_engine_manager, performance_test_config):
        """Test high throughput request processing."""
        # Fast mock synthesis
        mock_response = TTSResponse(
            request_id="perf-test",
            status=TTSStatus.COMPLETED,
            processing_time=0.01  # Very fast
        )
        mock_engine_manager.synthesize_async = AsyncMock(return_value=mock_response)
        
        processor = TTSQueueProcessor(max_concurrent=5)
        processor.engine_manager = mock_engine_manager
        await processor.start()
        
        # Add many requests
        num_requests = performance_test_config["total_requests"]
        start_time = time.time()
        
        for i in range(num_requests):
            req = TTSRequest(text=f"Performance test {i}")
            await processor.add_request(req)
        
        # Wait for all to complete
        while processor.queue_size > 0 or len(processor.processing_requests) > 0:
            await asyncio.sleep(0.1)
        
        end_time = time.time()
        total_time = end_time - start_time
        throughput = num_requests / total_time
        
        min_throughput = performance_test_config["min_throughput"]
        assert throughput >= min_throughput
        
        await processor.stop()
        
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, mock_engine_manager):
        """Test memory usage under heavy load."""
        import psutil
        import os
        
        mock_response = TTSResponse(
            request_id="memory-test",
            status=TTSStatus.COMPLETED,
            processing_time=0.1
        )
        mock_engine_manager.synthesize_async = AsyncMock(return_value=mock_response)
        
        processor = TTSQueueProcessor(max_concurrent=2)
        processor.engine_manager = mock_engine_manager
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        await processor.start()
        
        # Add many requests
        for i in range(1000):
            req = TTSRequest(text=f"Memory test request {i}")
            await processor.add_request(req)
            
            # Check memory periodically
            if i % 100 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_increase = current_memory - initial_memory
                
                # Memory shouldn't grow excessively
                assert memory_increase < 500  # Less than 500MB increase
        
        await processor.stop()
        
    @pytest.mark.asyncio
    async def test_queue_scaling_with_load(self, mock_engine_manager):
        """Test queue performance scales with concurrent workers."""
        mock_response = TTSResponse(
            request_id="scaling-test",
            status=TTSStatus.COMPLETED,
            processing_time=0.1
        )
        mock_engine_manager.synthesize_async = AsyncMock(return_value=mock_response)
        
        # Test with different concurrency levels
        results = {}
        
        for concurrency in [1, 2, 4]:
            processor = TTSQueueProcessor(max_concurrent=concurrency)
            processor.engine_manager = mock_engine_manager
            await processor.start()
            
            # Add fixed number of requests
            num_requests = 20
            start_time = time.time()
            
            for i in range(num_requests):
                req = TTSRequest(text=f"Scaling test {i}")
                await processor.add_request(req)
            
            # Wait for completion
            while processor.queue_size > 0 or len(processor.processing_requests) > 0:
                await asyncio.sleep(0.05)
            
            end_time = time.time()
            total_time = end_time - start_time
            results[concurrency] = total_time
            
            await processor.stop()
        
        # Higher concurrency should be faster (up to a point)
        assert results[2] <= results[1]  # 2 workers should be faster than 1
        
    @pytest.mark.asyncio
    async def test_priority_queue_performance(self):
        """Test priority queue performance with many requests."""
        processor = TTSQueueProcessor()
        
        # Add many requests with random priorities
        import random
        priorities = list(TTSPriority)
        
        start_time = time.time()
        
        for i in range(1000):
            priority = random.choice(priorities)
            req = TTSRequest(text=f"Priority test {i}", priority=priority)
            await processor.add_request(req)
        
        # Time to get next request (should be fast even with 1000 items)
        next_start = time.time()
        next_request = processor._get_next_request()
        next_time = time.time() - next_start
        
        assert next_time < 0.01  # Should be very fast
        assert next_request.request.priority == TTSPriority.URGENT  # Highest priority
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Adding 1000 requests should be fast
        assert total_time < 1.0