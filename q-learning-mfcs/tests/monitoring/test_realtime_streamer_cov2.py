"""Coverage boost tests for realtime_streamer.py - targeting uncovered lines."""
import asyncio
import json
import os
import sys
import types
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock heavy deps before any imports from monitoring
_mock_pd = MagicMock()
sys.modules.setdefault("pandas", _mock_pd)

_mock_ws = MagicMock()
_mock_ws.exceptions = MagicMock()
_mock_ws.exceptions.ConnectionClosed = type("ConnectionClosed", (Exception,), {})
_mock_ws.serve = AsyncMock()
sys.modules.setdefault("websockets", _mock_ws)
sys.modules.setdefault("websockets.server", MagicMock())
sys.modules.setdefault("websockets.exceptions", _mock_ws.exceptions)

# Mock ssl_config
_mock_ssl = MagicMock()
_mock_ssl.SSLConfig = MagicMock
_mock_ssl.SSLContextManager = MagicMock
_mock_ssl.initialize_ssl_infrastructure = MagicMock(return_value=(True, MagicMock()))
_mock_ssl.load_ssl_config = MagicMock(return_value=None)
_mock_ssl.test_ssl_connection = MagicMock(return_value=True)
sys.modules.setdefault("monitoring.ssl_config", _mock_ssl)

# Use a real ModuleType with __path__ so Python treats it as a package.
_monitoring_pkg = types.ModuleType("monitoring")
_monitoring_pkg.__path__ = [os.path.join(os.path.dirname(__file__), "..", "..", "src", "monitoring")]
sys.modules.setdefault("monitoring", _monitoring_pkg)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Now import the module under test directly using importlib
import importlib
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "monitoring.realtime_streamer",
    os.path.join(os.path.dirname(__file__), "..", "..", "src", "monitoring", "realtime_streamer.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

DataStreamManager = _mod.DataStreamManager
WSSSecurity = _mod.WSSSecurity
signal_handler = _mod.signal_handler


@pytest.fixture
def stream_manager():
    return DataStreamManager(ssl_config=None)


@pytest.mark.coverage_extra
class TestDataStreamManager:
    def test_init_no_ssl(self, stream_manager):
        assert stream_manager.ssl_config is None
        assert stream_manager.clients == set()
        assert stream_manager.data_cache == []
        assert stream_manager.streaming_active is False

    def test_init_with_ssl(self):
        mock_cfg = MagicMock()
        with patch.object(_mod, "SSLContextManager", return_value=MagicMock()):
            mgr = DataStreamManager(ssl_config=mock_cfg)
        assert mgr.ssl_config is mock_cfg
        assert mgr.ssl_context_manager is not None

    @pytest.mark.asyncio
    async def test_register_client_empty_cache(self, stream_manager):
        ws = AsyncMock()
        ws.remote_address = ("127.0.0.1", 12345)
        await stream_manager.register_client(ws)
        assert ws in stream_manager.clients

    @pytest.mark.asyncio
    async def test_register_client_with_cache(self, stream_manager):
        stream_manager.data_cache = [{"val": i} for i in range(10)]
        ws = AsyncMock()
        ws.remote_address = ("127.0.0.1", 12345)
        await stream_manager.register_client(ws)
        assert ws in stream_manager.clients

    @pytest.mark.asyncio
    async def test_unregister_client(self, stream_manager):
        ws = AsyncMock()
        ws.remote_address = ("127.0.0.1", 12345)
        stream_manager.clients.add(ws)
        await stream_manager.unregister_client(ws)
        assert ws not in stream_manager.clients

    @pytest.mark.asyncio
    async def test_send_to_client_ok(self, stream_manager):
        ws = AsyncMock()
        ws.remote_address = ("127.0.0.1", 12345)
        stream_manager.clients.add(ws)
        await stream_manager.send_to_client(ws, {"type": "test"})
        ws.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_connection_closed(self, stream_manager):
        ws = AsyncMock()
        ws.remote_address = ("127.0.0.1", 12345)
        ws.send.side_effect = _mock_ws.exceptions.ConnectionClosed()
        stream_manager.clients.add(ws)
        await stream_manager.send_to_client(ws, {"type": "test"})
        assert ws not in stream_manager.clients

    @pytest.mark.asyncio
    async def test_send_generic_error(self, stream_manager):
        ws = AsyncMock()
        ws.remote_address = ("127.0.0.1", 12345)
        ws.send.side_effect = RuntimeError("fail")
        stream_manager.clients.add(ws)
        await stream_manager.send_to_client(ws, {"type": "test"})
        assert ws not in stream_manager.clients

    @pytest.mark.asyncio
    async def test_broadcast_no_clients(self, stream_manager):
        await stream_manager.broadcast_to_clients({"type": "test"})

    @pytest.mark.asyncio
    async def test_broadcast_with_clients(self, stream_manager):
        ws1 = AsyncMock()
        ws1.remote_address = ("127.0.0.1", 1)
        ws2 = AsyncMock()
        ws2.remote_address = ("127.0.0.1", 2)
        stream_manager.clients = {ws1, ws2}
        await stream_manager.broadcast_to_clients({"type": "test"})

    def test_cache_empty_list(self, stream_manager):
        stream_manager.update_data_cache([])
        assert stream_manager.data_cache == []

    def test_cache_add(self, stream_manager):
        stream_manager.update_data_cache([{"a": 1}, {"b": 2}])
        assert len(stream_manager.data_cache) == 2

    def test_cache_overflow(self, stream_manager):
        stream_manager.cache_size_limit = 5
        stream_manager.update_data_cache([{"i": i} for i in range(10)])
        assert len(stream_manager.data_cache) == 5

    def test_load_simulation_data_no_dir(self, stream_manager):
        result = stream_manager.load_simulation_data()
        assert result is None

    def test_start_streaming(self, stream_manager):
        stream_manager.start_streaming()
        assert stream_manager.streaming_active is True

    def test_start_streaming_already_active(self, stream_manager):
        stream_manager.streaming_active = True
        stream_manager.start_streaming()
        assert stream_manager.streaming_active is True

    def test_stop_streaming(self, stream_manager):
        stream_manager.streaming_active = True
        stream_manager.stop_streaming()
        assert stream_manager.streaming_active is False
        assert stream_manager._stop_event.is_set()

    def test_stop_not_active(self, stream_manager):
        stream_manager.streaming_active = False
        stream_manager.stop_streaming()


@pytest.mark.coverage_extra
class TestHandleClientMessage:
    @pytest.mark.asyncio
    async def test_ping(self, stream_manager):
        ws = AsyncMock()
        ws.remote_address = ("127.0.0.1", 1)
        stream_manager.clients.add(ws)
        await stream_manager.handle_client_message(ws, json.dumps({"type": "ping"}))

    @pytest.mark.asyncio
    async def test_request_data(self, stream_manager):
        ws = AsyncMock()
        ws.remote_address = ("127.0.0.1", 1)
        stream_manager.clients.add(ws)
        stream_manager.data_cache = [{"v": i} for i in range(200)]
        msg = json.dumps({"type": "request_data", "limit": 50})
        await stream_manager.handle_client_message(ws, msg)

    @pytest.mark.asyncio
    async def test_subscribe(self, stream_manager):
        ws = AsyncMock()
        ws.remote_address = ("127.0.0.1", 1)
        stream_manager.clients.add(ws)
        msg = json.dumps({"type": "subscribe", "streams": ["power"]})
        await stream_manager.handle_client_message(ws, msg)

    @pytest.mark.asyncio
    async def test_unknown_type(self, stream_manager):
        ws = AsyncMock()
        ws.remote_address = ("127.0.0.1", 1)
        stream_manager.clients.add(ws)
        msg = json.dumps({"type": "unknown_xyz"})
        await stream_manager.handle_client_message(ws, msg)

    @pytest.mark.asyncio
    async def test_invalid_json(self, stream_manager):
        ws = AsyncMock()
        ws.remote_address = ("127.0.0.1", 1)
        stream_manager.clients.add(ws)
        await stream_manager.handle_client_message(ws, "not json{{{")

    @pytest.mark.asyncio
    async def test_generic_exception(self, stream_manager):
        ws = AsyncMock()
        ws.remote_address = ("127.0.0.1", 1)
        stream_manager.clients.add(ws)
        # Trigger generic exception path by making json.loads return something
        # that causes message_type lookup to fail
        with patch.object(_mod, "json") as mock_json:
            mock_json.loads.side_effect = RuntimeError("boom")
            mock_json.JSONDecodeError = json.JSONDecodeError
            await stream_manager.handle_client_message(ws, '{"type":"x"}')


@pytest.mark.coverage_extra
class TestWSSSecurity:
    def test_validate_origin_empty(self):
        assert WSSSecurity.validate_origin("", ["http://localhost"]) is False

    def test_validate_origin_exact_match(self):
        assert WSSSecurity.validate_origin("http://localhost", ["http://localhost"]) is True

    def test_validate_origin_endswith(self):
        assert WSSSecurity.validate_origin("https://sub.example.com", [".example.com"]) is True

    def test_validate_origin_no_match(self):
        assert WSSSecurity.validate_origin("http://evil.com", ["http://localhost"]) is False


@pytest.mark.coverage_extra
class TestSignalHandler:
    def test_signal_handler_runs(self):
        signal_handler(2, None)
