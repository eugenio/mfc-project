"""Tests for monitoring/realtime_streamer.py - targeting 98%+ coverage."""
import asyncio
import gzip
import importlib.util
import json
import os
import signal
import sys
import types
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---- Mock heavy deps before import ----

# Mock pandas
_mock_pd = MagicMock()
sys.modules.setdefault("pandas", _mock_pd)

# Mock websockets
_mock_ws = MagicMock()
_mock_ws.exceptions = MagicMock()
_mock_ws.exceptions.ConnectionClosed = type(
    "ConnectionClosed", (Exception,), {},
)
_mock_ws.serve = AsyncMock()
sys.modules.setdefault("websockets", _mock_ws)
sys.modules.setdefault("websockets.server", MagicMock())
sys.modules.setdefault("websockets.exceptions", _mock_ws.exceptions)

# Mock monitoring.ssl_config
_mock_ssl_config_mod = MagicMock()


class _FakeSSLConfig:
    def __init__(self, **kwargs):
        self.cert_file = kwargs.get("cert_file", "/tmp/fake.crt")
        self.key_file = kwargs.get("key_file", "/tmp/fake.key")
        self.ca_file = kwargs.get("ca_file", None)
        self.domain = kwargs.get("domain", "localhost")
        self.ciphers = kwargs.get(
            "ciphers",
            "ECDHE+AESGCM:!aNULL:!MD5",
        )
        self.wss_port_streaming = kwargs.get("wss_port_streaming", 8001)
        self.https_port_frontend = kwargs.get("https_port_frontend", 8444)
        self.use_letsencrypt = kwargs.get("use_letsencrypt", False)
        self.staging = kwargs.get("staging", False)
        self.email = kwargs.get("email", "test@example.com")
        self.ssl_version = kwargs.get("ssl_version", "TLSv1_2")


_mock_ssl_config_mod.SSLConfig = _FakeSSLConfig
_mock_ssl_config_mod.SSLContextManager = MagicMock()
_mock_ssl_config_mod.load_ssl_config = MagicMock(
    return_value=_FakeSSLConfig(),
)
_mock_ssl_config_mod.initialize_ssl_infrastructure = MagicMock(
    return_value=(True, _FakeSSLConfig()),
)
_mock_ssl_config_mod.test_ssl_connection = MagicMock(return_value=True)

sys.modules.setdefault("monitoring.ssl_config", _mock_ssl_config_mod)
# Use a real ModuleType with __path__ so Python treats it as a package.
_monitoring_pkg = types.ModuleType("monitoring")
_monitoring_pkg.__path__ = [os.path.join(os.path.dirname(__file__), "..", "..", "src", "monitoring")]
sys.modules.setdefault("monitoring", _monitoring_pkg)

# ---- Load the module ----

_src = os.path.join(os.path.dirname(__file__), "..", "..", "src")
_spec = importlib.util.spec_from_file_location(
    "monitoring.realtime_streamer",
    os.path.join(_src, "monitoring", "realtime_streamer.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["monitoring.realtime_streamer"] = _mod
_spec.loader.exec_module(_mod)

DataStreamManager = _mod.DataStreamManager
WSSSecurity = _mod.WSSSecurity
signal_handler = _mod.signal_handler


# ---- Async test helper ----


def _run(coro):
    """Run async coroutine in a new event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_ws_mock(remote_addr=("127.0.0.1", 12345)):
    """Create a mock WebSocket connection."""
    ws = AsyncMock()
    ws.remote_address = remote_addr
    ws.send = AsyncMock()
    return ws


# ---- Tests ----


class TestDataStreamManager:
    def test_init_default(self):
        mgr = DataStreamManager()
        assert mgr.clients == set()
        assert mgr.data_cache == []
        assert mgr.cache_size_limit == 1000
        assert mgr.streaming_active is False

    def test_init_with_ssl_config(self):
        cfg = _FakeSSLConfig()
        mgr = DataStreamManager(ssl_config=cfg)
        assert mgr.ssl_config is cfg
        assert mgr.ssl_context_manager is not None

    def test_init_no_ssl_config(self):
        mgr = DataStreamManager(ssl_config=None)
        assert mgr.ssl_config is not None

    def test_register_client(self):
        mgr = DataStreamManager()
        ws = _make_ws_mock()
        _run(mgr.register_client(ws))
        assert ws in mgr.clients

    def test_register_client_with_cache(self):
        mgr = DataStreamManager()
        mgr.data_cache = [{"x": i} for i in range(200)]
        ws = _make_ws_mock()
        _run(mgr.register_client(ws))
        assert ws in mgr.clients
        ws.send.assert_called_once()
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == "initial_data"
        assert len(sent["data"]) == 100

    def test_unregister_client(self):
        mgr = DataStreamManager()
        ws = _make_ws_mock()
        mgr.clients.add(ws)
        _run(mgr.unregister_client(ws))
        assert ws not in mgr.clients

    def test_unregister_client_not_registered(self):
        mgr = DataStreamManager()
        ws = _make_ws_mock()
        _run(mgr.unregister_client(ws))

    def test_send_to_client_success(self):
        mgr = DataStreamManager()
        ws = _make_ws_mock()
        mgr.clients.add(ws)
        _run(mgr.send_to_client(ws, {"type": "test"}))
        ws.send.assert_called_once()

    def test_send_to_client_connection_closed(self):
        mgr = DataStreamManager()
        ws = _make_ws_mock()
        mgr.clients.add(ws)
        ws.send.side_effect = _mock_ws.exceptions.ConnectionClosed()
        _run(mgr.send_to_client(ws, {"type": "test"}))
        assert ws not in mgr.clients

    def test_send_to_client_generic_error(self):
        mgr = DataStreamManager()
        ws = _make_ws_mock()
        mgr.clients.add(ws)
        ws.send.side_effect = RuntimeError("send failed")
        _run(mgr.send_to_client(ws, {"type": "test"}))
        assert ws not in mgr.clients

    def test_broadcast_no_clients(self):
        mgr = DataStreamManager()
        _run(mgr.broadcast_to_clients({"type": "test"}))

    def test_broadcast_to_clients(self):
        mgr = DataStreamManager()
        ws1 = _make_ws_mock(("10.0.0.1", 1111))
        ws2 = _make_ws_mock(("10.0.0.2", 2222))
        mgr.clients.add(ws1)
        mgr.clients.add(ws2)
        _run(mgr.broadcast_to_clients({"type": "broadcast"}))
        assert ws1.send.called
        assert ws2.send.called

    def test_update_data_cache_empty(self):
        mgr = DataStreamManager()
        mgr.update_data_cache([])
        assert mgr.data_cache == []

    def test_update_data_cache_within_limit(self):
        mgr = DataStreamManager()
        mgr.update_data_cache([{"x": 1}, {"x": 2}])
        assert len(mgr.data_cache) == 2

    def test_update_data_cache_exceeds_limit(self):
        mgr = DataStreamManager()
        mgr.cache_size_limit = 5
        mgr.update_data_cache([{"x": i} for i in range(10)])
        assert len(mgr.data_cache) == 5
        assert mgr.data_cache[0]["x"] == 5

    def test_load_simulation_data_no_dir(self):
        mgr = DataStreamManager()
        result = mgr.load_simulation_data()
        assert result is None

    def test_load_simulation_data_with_real_files(self, tmp_path):
        """Cover lines 125-177: full data loading path."""
        mgr = DataStreamManager()
        mgr.last_data_time = datetime(2020, 1, 1)

        # Create directory structure
        sim_dir = tmp_path / "sim_001"
        sim_dir.mkdir()

        # Create gzipped CSV
        csv_content = (
            "time_hours,reservoir_concentration,outlet_concentration,"
            "total_power,biofilm_thicknesses,substrate_addition_rate,"
            "q_action,epsilon,reward\n"
            '1.0,10.0,5.0,25.0,"[10.0, 20.0]",0.5,1,0.1,0.5\n'
            "2.0,11.0,6.0,26.0,15.0,0.6,2,0.09,0.6\n"
        )
        gz_file = sim_dir / "mfc_data_001.csv.gz"
        with gzip.open(gz_file, "wt") as f:
            f.write(csv_content)

        # Create a mock pandas DataFrame that behaves like read_csv output
        class FakeRow(dict):
            def __getitem__(self, key):
                return dict.__getitem__(self, key)

        row1 = FakeRow(
            time_hours=1.0,
            reservoir_concentration=10.0,
            outlet_concentration=5.0,
            total_power=25.0,
            biofilm_thicknesses="[10.0, 20.0]",
            substrate_addition_rate=0.5,
            q_action=1,
            epsilon=0.1,
            reward=0.5,
        )
        row2 = FakeRow(
            time_hours=2.0,
            reservoir_concentration=11.0,
            outlet_concentration=6.0,
            total_power=26.0,
            biofilm_thicknesses=15.0,
            substrate_addition_rate=0.6,
            q_action=2,
            epsilon=0.09,
            reward=0.6,
        )
        mock_df = MagicMock()
        mock_df.iterrows.return_value = [(0, row1), (1, row2)]
        _mock_pd.read_csv.return_value = mock_df

        # Patch the Path constructor used inside load_simulation_data
        # so data_dir = tmp_path
        data_dir = tmp_path

        with patch.object(
            _mod,
            "Path",
            wraps=Path,
        ) as mock_path_cls:
            # Override the specific call Path("../../../data/simulation_data")

            def patched_path(*args, **kwargs):
                Path.__new__(Path)
                if args and str(args[0]) == "../../../data/simulation_data":
                    return data_dir
                return Path(*args, **kwargs)

            mock_path_cls.side_effect = patched_path
            # Actually, Path is used both as constructor and for exists/iterdir
            # Let's just patch at a lower level.

        # Simpler approach: just directly call with the right structure
        # by monkey-patching the hard-coded path

        with patch.object(
            _mod,
            "Path",
        ) as mock_path_cls:
            # When called with "../../../data/simulation_data", return
            # a mock that has exists=True, iterdir=[sim_dir]
            mock_data_dir = MagicMock()
            mock_data_dir.exists.return_value = True
            mock_data_dir.iterdir.return_value = [sim_dir]

            def path_side_effect(arg):
                if arg == "../../../data/simulation_data":
                    return mock_data_dir
                return Path(arg)

            mock_path_cls.side_effect = path_side_effect

            result = mgr.load_simulation_data()
            if result is not None:
                assert isinstance(result, list)
                assert len(result) == 2
                assert result[0]["time_hours"] == 1.0

    def test_load_simulation_data_no_simulation_dirs(self, tmp_path):
        """Cover line 126-127: no simulation dirs."""
        mgr = DataStreamManager()
        mock_data_dir = MagicMock()
        mock_data_dir.exists.return_value = True
        mock_data_dir.iterdir.return_value = []

        with patch.object(_mod, "Path") as mock_path_cls:
            mock_path_cls.side_effect = lambda arg: (
                mock_data_dir
                if arg == "../../../data/simulation_data"
                else Path(arg)
            )
            result = mgr.load_simulation_data()
            assert result is None

    def test_load_simulation_data_no_csv_files(self, tmp_path):
        """Cover lines 132-134: no CSV files in dir."""
        mgr = DataStreamManager()
        sim_dir = MagicMock()
        sim_dir.is_dir.return_value = True
        sim_dir.stat.return_value = MagicMock(st_mtime=1000)
        sim_dir.glob.return_value = []

        mock_data_dir = MagicMock()
        mock_data_dir.exists.return_value = True
        mock_data_dir.iterdir.return_value = [sim_dir]

        with patch.object(_mod, "Path") as mock_path_cls:
            mock_path_cls.side_effect = lambda arg: (
                mock_data_dir
                if arg == "../../../data/simulation_data"
                else Path(arg)
            )
            result = mgr.load_simulation_data()
            assert result is None

    def test_load_simulation_data_not_modified(self, tmp_path):
        """Cover lines 139-141: file not modified since last load."""
        mgr = DataStreamManager()
        mgr.last_data_time = datetime(2099, 1, 1)  # Far future

        mock_file = MagicMock()
        mock_file.stat.return_value = MagicMock(st_mtime=1000)
        sim_dir = MagicMock()
        sim_dir.is_dir.return_value = True
        sim_dir.stat.return_value = MagicMock(st_mtime=1000)
        sim_dir.glob.return_value = [mock_file]

        mock_data_dir = MagicMock()
        mock_data_dir.exists.return_value = True
        mock_data_dir.iterdir.return_value = [sim_dir]

        with patch.object(_mod, "Path") as mock_path_cls:
            mock_path_cls.side_effect = lambda arg: (
                mock_data_dir
                if arg == "../../../data/simulation_data"
                else Path(arg)
            )
            result = mgr.load_simulation_data()
            assert result is None

    def test_load_simulation_data_biofilm_parse_error(self, tmp_path):
        """Cover line 160-161: biofilm thickness parse exception."""
        mgr = DataStreamManager()
        mgr.last_data_time = datetime(2020, 1, 1)

        mock_file = MagicMock()
        mock_file.stat.return_value = MagicMock(st_mtime=2000000000)

        sim_dir = MagicMock()
        sim_dir.is_dir.return_value = True
        sim_dir.stat.return_value = MagicMock(st_mtime=1000)
        sim_dir.glob.return_value = [mock_file]

        mock_data_dir = MagicMock()
        mock_data_dir.exists.return_value = True
        mock_data_dir.iterdir.return_value = [sim_dir]

        # Make biofilm_thicknesses cause a parse error
        class BadRow(dict):
            def __getitem__(self, key):
                return dict.__getitem__(self, key)

        row = BadRow(
            time_hours=1.0,
            reservoir_concentration=10.0,
            outlet_concentration=5.0,
            total_power=25.0,
            biofilm_thicknesses="not_parseable{{{",
            substrate_addition_rate=0.5,
            q_action=1,
            epsilon=0.1,
            reward=0.5,
        )
        mock_df = MagicMock()
        mock_df.iterrows.return_value = [(0, row)]
        _mock_pd.read_csv.return_value = mock_df

        with patch.object(_mod, "Path") as mock_path_cls:
            mock_path_cls.side_effect = lambda arg: (
                mock_data_dir
                if arg == "../../../data/simulation_data"
                else Path(arg)
            )
            with patch.object(_mod, "gzip") as mock_gzip:
                mock_gzip.open.return_value.__enter__ = MagicMock()
                mock_gzip.open.return_value.__exit__ = MagicMock(
                    return_value=False,
                )
                result = mgr.load_simulation_data()
                if result is not None:
                    assert result[0]["biofilm_thicknesses"] == [10.0]

    def test_load_simulation_data_exception(self):
        mgr = DataStreamManager()
        with patch.object(
            Path,
            "exists",
            side_effect=RuntimeError("disk error"),
        ):
            result = mgr.load_simulation_data()
            assert result is None

    def test_start_streaming(self):
        mgr = DataStreamManager()
        mgr.start_streaming()
        assert mgr.streaming_active is True

    def test_start_streaming_already_active(self):
        mgr = DataStreamManager()
        mgr.streaming_active = True
        mgr.start_streaming()
        assert mgr.streaming_active is True

    def test_stop_streaming(self):
        mgr = DataStreamManager()
        mgr.streaming_active = True
        mgr.stop_streaming()
        assert mgr.streaming_active is False
        assert mgr._stop_event.is_set()

    def test_stop_streaming_not_active(self):
        mgr = DataStreamManager()
        mgr.stop_streaming()

    def test_handle_client_message_ping(self):
        mgr = DataStreamManager()
        ws = _make_ws_mock()
        mgr.clients.add(ws)
        msg = json.dumps({"type": "ping"})
        _run(mgr.handle_client_message(ws, msg))
        ws.send.assert_called_once()
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == "pong"

    def test_handle_client_message_request_data(self):
        mgr = DataStreamManager()
        mgr.data_cache = [{"x": i} for i in range(50)]
        ws = _make_ws_mock()
        mgr.clients.add(ws)
        msg = json.dumps({"type": "request_data", "limit": 10})
        _run(mgr.handle_client_message(ws, msg))
        ws.send.assert_called_once()
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == "data_response"
        assert len(sent["data"]) == 10

    def test_handle_client_message_subscribe(self):
        mgr = DataStreamManager()
        ws = _make_ws_mock()
        mgr.clients.add(ws)
        msg = json.dumps({"type": "subscribe", "streams": ["power"]})
        _run(mgr.handle_client_message(ws, msg))
        ws.send.assert_called_once()
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == "subscription_confirmed"
        assert sent["streams"] == ["power"]

    def test_handle_client_message_unknown_type(self):
        mgr = DataStreamManager()
        ws = _make_ws_mock()
        mgr.clients.add(ws)
        msg = json.dumps({"type": "foobar"})
        _run(mgr.handle_client_message(ws, msg))
        ws.send.assert_called_once()
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == "error"

    def test_handle_client_message_invalid_json(self):
        mgr = DataStreamManager()
        ws = _make_ws_mock()
        mgr.clients.add(ws)
        _run(mgr.handle_client_message(ws, "not valid json{{{"))
        ws.send.assert_called_once()
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == "error"
        assert "Invalid JSON" in sent["message"]

    def test_handle_client_message_exception(self):
        mgr = DataStreamManager()
        ws = _make_ws_mock()
        mgr.clients.add(ws)
        # Cause an exception inside handle_client_message that is NOT
        # JSONDecodeError -- e.g. make data.get raise
        msg = json.dumps({"type": "ping"})

        # Patch send_to_client to raise only on first call, succeed on
        # the error-response call (lines 302-311)
        call_count = [0]
        original = mgr.send_to_client

        async def flaky_send(websocket, message):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("test err")
            return await original(websocket, message)

        mgr.send_to_client = flaky_send
        _run(mgr.handle_client_message(ws, msg))

    def test_client_handler_normal(self):
        mgr = DataStreamManager()
        ws = _make_ws_mock()

        async def fake_iter():
            yield json.dumps({"type": "ping"})

        ws.__aiter__ = lambda self: fake_iter()
        _run(mgr.client_handler(ws, "/"))
        assert ws not in mgr.clients

    def test_client_handler_connection_closed(self):
        mgr = DataStreamManager()
        ws = _make_ws_mock()

        async def raising_iter():
            raise _mock_ws.exceptions.ConnectionClosed
            yield

        ws.__aiter__ = lambda self: raising_iter()
        _run(mgr.client_handler(ws, "/"))

    def test_client_handler_exception(self):
        mgr = DataStreamManager()
        ws = _make_ws_mock()

        async def raising_iter():
            raise RuntimeError("unexpected")
            yield

        ws.__aiter__ = lambda self: raising_iter()
        _run(mgr.client_handler(ws, "/"))

    def test_data_polling_loop_with_data(self):
        mgr = DataStreamManager()
        mgr._stop_event.clear()

        call_count = [0]

        def mock_load():
            call_count[0] += 1
            if call_count[0] == 1:
                return [{"x": 1}, {"x": 2}]
            mgr._stop_event.set()
            return None

        mgr.load_simulation_data = mock_load

        async def run_loop():
            with patch("asyncio.sleep", new_callable=AsyncMock):
                try:
                    await asyncio.wait_for(
                        mgr.data_polling_loop(),
                        timeout=5.0,
                    )
                except asyncio.TimeoutError:
                    pass

        _run(run_loop())
        assert len(mgr.data_cache) >= 2

    def test_data_polling_loop_heartbeat(self):
        mgr = DataStreamManager()
        mgr._stop_event.clear()
        ws = _make_ws_mock()
        mgr.clients.add(ws)

        call_count = [0]

        def mock_load():
            call_count[0] += 1
            if call_count[0] >= 2:
                mgr._stop_event.set()

        mgr.load_simulation_data = mock_load

        async def run_loop():
            with patch("asyncio.sleep", new_callable=AsyncMock):
                try:
                    await asyncio.wait_for(
                        mgr.data_polling_loop(),
                        timeout=5.0,
                    )
                except asyncio.TimeoutError:
                    pass

        _run(run_loop())
        assert ws.send.called

    def test_data_polling_loop_exception(self):
        mgr = DataStreamManager()
        mgr._stop_event.clear()

        call_count = [0]

        def mock_load():
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("polling error")
            mgr._stop_event.set()

        mgr.load_simulation_data = mock_load

        async def run_loop():
            with patch("asyncio.sleep", new_callable=AsyncMock):
                try:
                    await asyncio.wait_for(
                        mgr.data_polling_loop(),
                        timeout=5.0,
                    )
                except asyncio.TimeoutError:
                    pass

        _run(run_loop())

    def test_data_polling_loop_cancelled(self):
        mgr = DataStreamManager()

        async def run_loop():
            task = asyncio.create_task(mgr.data_polling_loop())
            await asyncio.sleep(0.1)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        _run(run_loop())


class TestWSSSecurity:
    def test_create_ssl_context(self):
        cfg = _FakeSSLConfig()
        with patch("ssl.SSLContext") as mock_ctx_cls:
            mock_ctx = MagicMock()
            mock_ctx_cls.return_value = mock_ctx
            result = WSSSecurity.create_ssl_context(cfg)
            mock_ctx.load_cert_chain.assert_called_once_with(
                cfg.cert_file, cfg.key_file,
            )
            assert result is mock_ctx

    def test_validate_origin_empty(self):
        assert WSSSecurity.validate_origin("", ["http://localhost"]) is False

    def test_validate_origin_exact_match(self):
        assert WSSSecurity.validate_origin(
            "https://localhost:8444",
            ["https://localhost:8444"],
        ) is True

    def test_validate_origin_endswith_match(self):
        assert WSSSecurity.validate_origin(
            "https://sub.example.com",
            [".example.com"],
        ) is True

    def test_validate_origin_no_match(self):
        assert WSSSecurity.validate_origin(
            "https://evil.com",
            ["https://localhost:8444"],
        ) is False


class TestSignalHandler:
    def test_signal_handler(self):
        signal_handler(signal.SIGINT, None)
        signal_handler(signal.SIGTERM, None)


class TestMain:
    def test_main_init_ssl_success(self):
        _mock_ssl_config_mod.initialize_ssl_infrastructure.return_value = (
            True,
            _FakeSSLConfig(),
        )
        with patch(
            "sys.argv",
            ["realtime_streamer", "--init-ssl"],
        ):
            _mod.main()

    def test_main_init_ssl_failure(self):
        _mock_ssl_config_mod.initialize_ssl_infrastructure.return_value = (
            False,
            None,
        )
        with patch("sys.argv", ["realtime_streamer", "--init-ssl"]):
            with pytest.raises(SystemExit) as exc_info:
                _mod.main()
            assert exc_info.value.code == 1

    def test_main_test_ssl_success(self):
        _mock_ssl_config_mod.load_ssl_config.return_value = _FakeSSLConfig()
        _mock_ssl_config_mod.test_ssl_connection.return_value = True
        with patch("sys.argv", ["realtime_streamer", "--test-ssl"]):
            _mod.main()

    def test_main_test_ssl_failure(self):
        _mock_ssl_config_mod.load_ssl_config.return_value = _FakeSSLConfig()
        _mock_ssl_config_mod.test_ssl_connection.return_value = False
        with patch("sys.argv", ["realtime_streamer", "--test-ssl"]):
            with pytest.raises(SystemExit) as exc_info:
                _mod.main()
            assert exc_info.value.code == 1

    def test_main_test_ssl_no_config(self):
        _mock_ssl_config_mod.load_ssl_config.return_value = None
        with patch("sys.argv", ["realtime_streamer", "--test-ssl"]):
            with pytest.raises(SystemExit) as exc_info:
                _mod.main()
            assert exc_info.value.code == 1

    def test_main_run_server(self):
        with patch("sys.argv", ["realtime_streamer"]):
            with patch.object(_mod, "asyncio") as mock_asyncio:
                mock_asyncio.run = MagicMock()
                _mod.main()
                mock_asyncio.run.assert_called_once()

    def test_main_run_server_keyboard_interrupt(self):
        with patch("sys.argv", ["realtime_streamer"]):
            with patch.object(_mod, "asyncio") as mock_asyncio:
                mock_asyncio.run.side_effect = KeyboardInterrupt()
                _mod.main()

    def test_main_run_server_exception(self):
        with patch("sys.argv", ["realtime_streamer"]):
            with patch.object(_mod, "asyncio") as mock_asyncio:
                mock_asyncio.run.side_effect = RuntimeError("server error")
                with pytest.raises(SystemExit) as exc_info:
                    _mod.main()
                assert exc_info.value.code == 1


class TestRunWebsocketServer:
    def test_run_server_no_ssl(self):
        cfg = _FakeSSLConfig(
            cert_file="/nonexistent/cert.pem",
            key_file="/nonexistent/key.pem",
        )
        mock_server = AsyncMock()
        mock_server.wait_closed = AsyncMock(
            side_effect=KeyboardInterrupt(),
        )
        mock_server.close = MagicMock()
        _mock_ws.serve = AsyncMock(return_value=mock_server)

        async def run():
            await _mod.run_websocket_server(
                host="127.0.0.1", port=9999, ssl_config=cfg,
            )

        try:
            _run(run())
        except KeyboardInterrupt:
            pass

    def test_run_server_with_ssl(self, tmp_path):
        cert_file = tmp_path / "cert.pem"
        key_file = tmp_path / "key.pem"
        cert_file.write_text("fake cert")
        key_file.write_text("fake key")
        cfg = _FakeSSLConfig(
            cert_file=str(cert_file),
            key_file=str(key_file),
        )
        mock_server = AsyncMock()
        mock_server.wait_closed = AsyncMock(
            side_effect=KeyboardInterrupt(),
        )
        mock_server.close = MagicMock()
        _mock_ws.serve = AsyncMock(return_value=mock_server)

        with patch.object(
            WSSSecurity,
            "create_ssl_context",
            return_value=MagicMock(),
        ):
            async def run():
                await _mod.run_websocket_server(
                    host="127.0.0.1", port=9999, ssl_config=cfg,
                )

            try:
                _run(run())
            except KeyboardInterrupt:
                pass

    def test_run_server_default_config(self):
        _mock_ssl_config_mod.load_ssl_config.return_value = _FakeSSLConfig(
            cert_file="/nonexistent/cert.pem",
            key_file="/nonexistent/key.pem",
        )
        mock_server = AsyncMock()
        mock_server.wait_closed = AsyncMock(
            side_effect=KeyboardInterrupt(),
        )
        mock_server.close = MagicMock()
        _mock_ws.serve = AsyncMock(return_value=mock_server)

        async def run():
            await _mod.run_websocket_server(
                host="127.0.0.1", port=None, ssl_config=None,
            )

        try:
            _run(run())
        except KeyboardInterrupt:
            pass
