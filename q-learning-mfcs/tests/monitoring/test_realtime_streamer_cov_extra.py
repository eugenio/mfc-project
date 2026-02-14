import importlib.util
import os
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

# Snapshot sys.modules before mocking
_original_modules = dict(sys.modules)

# Save original sys.modules state for restoration after import
_saved_modules = {}
for _k in list(sys.modules):
    if _k == "monitoring" or _k.startswith("monitoring."):
        _saved_modules[_k] = sys.modules[_k]

# Mock heavy deps before import (use setdefault to avoid clobbering real modules)
_mock_pd = MagicMock()
sys.modules.setdefault("pandas", _mock_pd)

_mock_ws = MagicMock()
_mock_ws.exceptions = MagicMock()
_mock_ws.exceptions.ConnectionClosed = type("ConnectionClosed", (Exception,), {})
_mock_ws.serve = AsyncMock()
sys.modules.setdefault("websockets", _mock_ws)
sys.modules.setdefault("websockets.server", MagicMock())
sys.modules.setdefault("websockets.exceptions", _mock_ws.exceptions)

# Mock monitoring.ssl_config
_mock_ssl_config_mod = MagicMock()


class _FakeSSLConfig:
    """Minimal fake SSLConfig for testing."""

    def __init__(self, **kwargs):
        self.cert_file = kwargs.get("cert_file", "/tmp/fake.crt")
        self.key_file = kwargs.get("key_file", "/tmp/fake.key")
        self.ca_file = kwargs.get("ca_file", None)
        self.domain = kwargs.get("domain", "localhost")
        self.ciphers = kwargs.get("ciphers", "ECDHE+AESGCM:!aNULL:!MD5")
        self.wss_port_streaming = kwargs.get("wss_port_streaming", 8001)
        self.https_port_frontend = kwargs.get("https_port_frontend", 8444)


_mock_ssl_config_mod.SSLConfig = _FakeSSLConfig
_mock_ssl_config_mod.SSLContextManager = MagicMock()
_mock_ssl_config_mod.load_ssl_config = MagicMock(return_value=_FakeSSLConfig())
_mock_ssl_config_mod.initialize_ssl_infrastructure = MagicMock(
    return_value=(True, _FakeSSLConfig()),
)
_mock_ssl_config_mod.test_ssl_connection = MagicMock(return_value=True)

# Temporarily set up monitoring namespace for our importlib load
for _k in list(sys.modules):
    if _k == "monitoring" or _k.startswith("monitoring."):
        del sys.modules[_k]
sys.modules["monitoring.ssl_config"] = _mock_ssl_config_mod
_monitoring_pkg = types.ModuleType("monitoring")
_monitoring_pkg.__path__ = [
    os.path.join(os.path.dirname(__file__), "..", "..", "src", "monitoring"),
]
sys.modules["monitoring"] = _monitoring_pkg

# Load the module under test
_src = os.path.join(os.path.dirname(__file__), "..", "..", "src")
_spec = importlib.util.spec_from_file_location(
    "monitoring.realtime_streamer",
    os.path.join(_src, "monitoring", "realtime_streamer.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["monitoring.realtime_streamer"] = _mod
_spec.loader.exec_module(_mod)

# Restore original monitoring modules so other tests are not affected
for _k in list(sys.modules):
    if (_k == "monitoring" or _k.startswith("monitoring.")) and _k != "monitoring.realtime_streamer":
        del sys.modules[_k]
for _k, _v in _saved_modules.items():
    sys.modules[_k] = _v

# --- Restore sys.modules to prevent mock leakage ---
for _mock_key in list(sys.modules):
    if _mock_key not in _original_modules:
        if isinstance(sys.modules[_mock_key], (MagicMock, types.ModuleType)):
            del sys.modules[_mock_key]
    elif isinstance(sys.modules[_mock_key], MagicMock):
        sys.modules[_mock_key] = _original_modules[_mock_key]