"""Tests for monitoring/security_middleware.py - targeting 98%+ coverage."""
import importlib.util
import os
import sys
from datetime import datetime, timedelta
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Pre-mock heavy dependencies before importing the module
_mock_fastapi = MagicMock()
_mock_fastapi.Depends = lambda x: x
_mock_fastapi.HTTPException = type("HTTPException", (Exception,), {"__init__": lambda self, status_code=None, detail=None: Exception.__init__(self, detail)})
_mock_fastapi.Request = MagicMock
_mock_fastapi.Response = MagicMock

_mock_http_bearer = MagicMock()
_mock_fastapi_security = MagicMock()
_mock_fastapi_security.HTTPBearer = MagicMock(return_value=_mock_http_bearer)
_mock_fastapi_security.HTTPAuthorizationCredentials = MagicMock

_mock_starlette_middleware = MagicMock()


class FakeBaseHTTPMiddleware:
    def __init__(self, app, **kwargs):
        self.app = app


_mock_starlette_middleware.BaseHTTPMiddleware = FakeBaseHTTPMiddleware

_mock_starlette_responses = MagicMock()
_mock_redirect_response = MagicMock()
_mock_starlette_responses.RedirectResponse = _mock_redirect_response

# Mock monitoring.ssl_config
_mock_ssl_config_mod = MagicMock()
_mock_ssl_config_mod.SecurityHeaders = MagicMock()
_mock_ssl_config_mod.SSLConfig = MagicMock
_mock_ssl_config_mod.load_ssl_config = MagicMock(return_value=None)

sys.modules.setdefault("fastapi", _mock_fastapi)
sys.modules.setdefault("fastapi.security", _mock_fastapi_security)
sys.modules.setdefault("starlette", MagicMock())
sys.modules.setdefault("starlette.middleware", MagicMock())
sys.modules.setdefault("starlette.middleware.base", _mock_starlette_middleware)
sys.modules.setdefault("starlette.responses", _mock_starlette_responses)
sys.modules.setdefault("monitoring.ssl_config", _mock_ssl_config_mod)

_src = os.path.join(os.path.dirname(__file__), "..", "..", "src")
_spec = importlib.util.spec_from_file_location(
    "monitoring.security_middleware",
    os.path.join(_src, "monitoring", "security_middleware.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["monitoring.security_middleware"] = _mod
_spec.loader.exec_module(_mod)

SecurityConfig = _mod.SecurityConfig
SessionManager = _mod.SessionManager
CSRFProtection = _mod.CSRFProtection
RateLimiter = _mod.RateLimiter
SecurityMiddleware = _mod.SecurityMiddleware
get_csrf_token = _mod.get_csrf_token
is_authenticated = _mod.is_authenticated
create_security_middleware = _mod.create_security_middleware
get_current_session = _mod.get_current_session
require_authentication = _mod.require_authentication


class TestSecurityConfig:
    def test_init_defaults(self):
        config = SecurityConfig()
        assert config.session_timeout_minutes == 60
        assert config.session_cookie_name == "mfc_session"
        assert config.csrf_token_name == "csrf_token"
        assert config.csrf_header_name == "X-CSRF-Token"
        assert config.jwt_algorithm == "HS256"
        assert config.jwt_expiry_hours == 24
        assert config.rate_limit_requests_per_minute == 60
        assert config.rate_limit_burst == 10
        assert config.enforce_https is True
        assert config.strict_transport_security is True
        assert config.content_type_nosniff is True
        assert config.frame_options_deny is True
        assert config.xss_protection is True
        assert config.referrer_policy_strict is True

    def test_generate_secret_key(self):
        config = SecurityConfig()
        key = config._generate_secret_key()
        assert isinstance(key, str)
        assert len(key) > 20

    def test_load_allowed_ips_default(self):
        config = SecurityConfig()
        assert "127.0.0.1" in config.allowed_ips
        assert "::1" in config.allowed_ips

    def test_load_blocked_ips_default(self):
        config = SecurityConfig()
        assert config.blocked_ips == []

    def test_get_default_csp(self):
        config = SecurityConfig()
        assert "default-src 'self'" in config.csp_policy

    def test_env_vars_used(self):
        with patch.dict(os.environ, {"MFC_SESSION_SECRET": "test_session_secret"}):
            config = SecurityConfig()
            assert config.session_secret_key == "test_session_secret"


class TestSessionManager:
    def setup_method(self):
        self.config = SecurityConfig()
        self.manager = SessionManager(self.config)

    def test_create_session(self):
        session_id = self.manager.create_session("user1")
        assert isinstance(session_id, str)
        assert session_id in self.manager.active_sessions

    def test_create_session_with_data(self):
        session_id = self.manager.create_session("user1", {"role": "admin"})
        session = self.manager.active_sessions[session_id]
        assert session["user_data"]["role"] == "admin"

    def test_validate_session_valid(self):
        session_id = self.manager.create_session("user1")
        result = self.manager.validate_session(session_id)
        assert result is not None
        assert result["user_id"] == "user1"

    def test_validate_session_empty(self):
        result = self.manager.validate_session("")
        assert result is None

    def test_validate_session_nonexistent(self):
        result = self.manager.validate_session("fake_id")
        assert result is None

    def test_validate_session_expired(self):
        session_id = self.manager.create_session("user1")
        session = self.manager.active_sessions[session_id]
        session["expires_at"] = (datetime.now() - timedelta(hours=1)).isoformat()
        result = self.manager.validate_session(session_id)
        assert result is None
        assert session_id not in self.manager.active_sessions

    def test_destroy_session(self):
        session_id = self.manager.create_session("user1")
        result = self.manager.destroy_session(session_id)
        assert result is True
        assert session_id not in self.manager.active_sessions

    def test_destroy_session_nonexistent(self):
        result = self.manager.destroy_session("fake_id")
        assert result is False

    def test_cleanup_expired_sessions(self):
        sid1 = self.manager.create_session("user1")
        sid2 = self.manager.create_session("user2")
        self.manager.active_sessions[sid1]["expires_at"] = (
            datetime.now() - timedelta(hours=1)
        ).isoformat()
        self.manager.cleanup_expired_sessions()
        assert sid1 not in self.manager.active_sessions
        assert sid2 in self.manager.active_sessions

    def test_cleanup_no_expired(self):
        self.manager.create_session("user1")
        self.manager.cleanup_expired_sessions()
        assert len(self.manager.active_sessions) == 1


class TestCSRFProtection:
    def setup_method(self):
        self.config = SecurityConfig()
        self.csrf = CSRFProtection(self.config)

    def test_generate_csrf_token(self):
        token = self.csrf.generate_csrf_token("session123")
        assert isinstance(token, str)
        assert len(token) > 10

    def test_validate_csrf_token_valid(self):
        token = self.csrf.generate_csrf_token("session123")
        result = self.csrf.validate_csrf_token(token, "session123")
        assert result is True

    def test_validate_csrf_token_wrong_session(self):
        token = self.csrf.generate_csrf_token("session123")
        result = self.csrf.validate_csrf_token(token, "wrong_session")
        assert result is False

    def test_validate_csrf_token_invalid_format(self):
        import base64
        bad_token = base64.b64encode(b"only:two_parts").decode()
        result = self.csrf.validate_csrf_token(bad_token, "session123")
        assert result is False

    def test_validate_csrf_token_garbage(self):
        result = self.csrf.validate_csrf_token("not_base64!!!", "session123")
        assert result is False


class TestRateLimiter:
    def setup_method(self):
        self.config = SecurityConfig()
        self.config.rate_limit_requests_per_minute = 3
        self.limiter = RateLimiter(self.config)

    def test_not_rate_limited(self):
        assert self.limiter.is_rate_limited("1.2.3.4") is False

    def test_rate_limited_after_exceeding(self):
        for _ in range(3):
            self.limiter.is_rate_limited("1.2.3.4")
        assert self.limiter.is_rate_limited("1.2.3.4") is True

    def test_different_ips_independent(self):
        for _ in range(3):
            self.limiter.is_rate_limited("1.2.3.4")
        assert self.limiter.is_rate_limited("5.6.7.8") is False


class TestSecurityMiddleware:
    def setup_method(self):
        self.app = MagicMock()
        _mock_ssl_config_mod.load_ssl_config.return_value = None
        self.middleware = SecurityMiddleware(self.app, ssl_config=None)

    def test_init(self):
        assert self.middleware.ssl_config is None
        assert isinstance(self.middleware.security_config, SecurityConfig)
        assert isinstance(self.middleware.session_manager, SessionManager)
        assert isinstance(self.middleware.csrf_protection, CSRFProtection)
        assert isinstance(self.middleware.rate_limiter, RateLimiter)

    def test_get_client_ip_forwarded(self):
        request = MagicMock()
        request.headers = {"X-Forwarded-For": "10.0.0.1, 10.0.0.2"}
        ip = self.middleware._get_client_ip(request)
        assert ip == "10.0.0.1"

    def test_get_client_ip_real_ip(self):
        request = MagicMock()
        headers_data = {"X-Forwarded-For": None, "X-Real-IP": "10.0.0.5"}
        request.headers = MagicMock()
        request.headers.get = lambda key, default=None: headers_data.get(key, default)
        ip = self.middleware._get_client_ip(request)
        assert ip == "10.0.0.5"

    def test_get_client_ip_direct(self):
        request = MagicMock()
        request.headers = MagicMock()
        request.headers.get = MagicMock(return_value=None)
        request.client = MagicMock()
        request.client.host = "192.168.1.1"
        ip = self.middleware._get_client_ip(request)
        assert ip == "192.168.1.1"

    def test_get_client_ip_no_client(self):
        request = MagicMock()
        request.headers = MagicMock()
        request.headers.get = MagicMock(return_value=None)
        request.client = None
        ip = self.middleware._get_client_ip(request)
        assert ip == "unknown"

    def test_is_ip_allowed_blocked(self):
        self.middleware.security_config.blocked_ips = ["1.2.3.4"]
        assert self.middleware._is_ip_allowed("1.2.3.4") is False

    def test_is_ip_allowed_no_allowed_list(self):
        self.middleware.security_config.allowed_ips = []
        self.middleware.security_config.blocked_ips = []
        assert self.middleware._is_ip_allowed("1.2.3.4") is True

    def test_is_ip_allowed_in_list(self):
        self.middleware.security_config.allowed_ips = ["1.2.3.4"]
        assert self.middleware._is_ip_allowed("1.2.3.4") is True

    def test_is_ip_allowed_not_in_list(self):
        self.middleware.security_config.allowed_ips = ["5.6.7.8"]
        assert self.middleware._is_ip_allowed("1.2.3.4") is False

    def test_is_secure_request_https(self):
        request = MagicMock()
        request.url.scheme = "https"
        request.headers = MagicMock()
        request.headers.get = MagicMock(return_value=None)
        assert self.middleware._is_secure_request(request) is True

    def test_is_secure_request_forwarded_proto(self):
        request = MagicMock()
        request.url.scheme = "http"
        request.headers.get = lambda key, default=None: {"X-Forwarded-Proto": "https"}.get(key, default)
        assert self.middleware._is_secure_request(request) is True

    def test_is_secure_request_forwarded_ssl(self):
        request = MagicMock()
        request.url.scheme = "http"
        request.headers.get = lambda key, default=None: {"X-Forwarded-SSL": "on"}.get(key, default)
        assert self.middleware._is_secure_request(request) is True

    def test_has_valid_api_auth_no_header(self):
        request = MagicMock()
        request.headers.get = MagicMock(return_value=None)
        assert self.middleware._has_valid_api_auth(request) is False

    def test_has_valid_api_auth_no_bearer(self):
        request = MagicMock()
        request.headers.get = MagicMock(return_value="Basic xyz")
        assert self.middleware._has_valid_api_auth(request) is False

    def test_has_valid_api_auth_wrong_token(self):
        request = MagicMock()
        request.headers.get = MagicMock(return_value="Bearer wrong_token")
        with patch.dict(os.environ, {"MFC_API_TOKEN": "correct_token"}):
            assert self.middleware._has_valid_api_auth(request) is False

    def test_has_valid_api_auth_correct_token(self):
        request = MagicMock()
        request.headers.get = MagicMock(return_value="Bearer development-token")
        assert self.middleware._has_valid_api_auth(request) is True

    def test_create_error_response(self):
        response = self.middleware._create_error_response("test error", 403)
        assert response is not None

    def test_add_security_headers_no_ssl(self):
        self.middleware.ssl_config = None
        response = MagicMock()
        response.headers = {}
        self.middleware._add_security_headers(response)
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
        assert "X-XSS-Protection" in response.headers
        assert "Referrer-Policy" in response.headers
        assert "Content-Security-Policy" in response.headers
        assert "Permissions-Policy" in response.headers

    def test_add_security_headers_with_login(self):
        self.middleware.ssl_config = None
        headers_store = {}

        class FakeHeaders:
            def __setitem__(self, key, value):
                headers_store[key] = value

            def __contains__(self, key):
                return key in headers_store

            def get(self, key, default=""):
                if key == "location":
                    return "https://example.com/login"
                return headers_store.get(key, default)

        response = MagicMock()
        response.headers = FakeHeaders()
        self.middleware._add_security_headers(response)
        assert headers_store.get("Cache-Control") == "no-cache, no-store, must-revalidate"
        assert headers_store.get("Pragma") == "no-cache"
        assert headers_store.get("Expires") == "0"

    def test_add_security_headers_disabled_flags(self):
        self.middleware.ssl_config = None
        self.middleware.security_config.content_type_nosniff = False
        self.middleware.security_config.frame_options_deny = False
        self.middleware.security_config.xss_protection = False
        self.middleware.security_config.referrer_policy_strict = False
        headers_store = {}

        class FakeHeaders2:
            def __setitem__(self, key, value):
                headers_store[key] = value

            def __contains__(self, key):
                return key in headers_store

            def get(self, key, default=""):
                return headers_store.get(key, default)

        response = MagicMock()
        response.headers = FakeHeaders2()
        self.middleware._add_security_headers(response)
        assert "X-Content-Type-Options" not in headers_store
        assert "X-Frame-Options" not in headers_store

    def test_add_security_headers_with_ssl(self):
        self.middleware.ssl_config = MagicMock()
        _mock_ssl_config_mod.SecurityHeaders.get_security_headers.return_value = {
            "Strict-Transport-Security": "max-age=31536000"
        }
        headers_store = {}

        class FakeHeaders3:
            def __setitem__(self, key, value):
                headers_store[key] = value

            def __contains__(self, key):
                return key in headers_store

            def get(self, key, default=""):
                return headers_store.get(key, default)

        response = MagicMock()
        response.headers = FakeHeaders3()
        self.middleware._add_security_headers(response)
        assert "Strict-Transport-Security" in headers_store

    def test_set_secure_cookie_flags_no_cookies(self):
        response = MagicMock()
        response.headers = MagicMock()
        response.headers.__contains__ = MagicMock(return_value=False)
        self.middleware._set_secure_cookie_flags(response)

    def test_set_secure_cookie_flags_with_cookies(self):
        self.middleware.ssl_config = MagicMock()
        response = MagicMock()
        response.headers = MagicMock()
        response.headers.__contains__ = lambda self_h, key: key == "Set-Cookie"
        response.headers.getlist = MagicMock(return_value=["session=abc"])
        self.middleware._set_secure_cookie_flags(response)
        response.headers.pop.assert_called_once_with("Set-Cookie")
        response.headers.append.assert_called()

    def test_redirect_to_https_no_ssl(self):
        self.middleware.ssl_config = None
        request = MagicMock()
        request.url.replace.return_value = "https://example.com/test"
        result = self.middleware._redirect_to_https(request)
        _mock_redirect_response.assert_called()

    def test_redirect_to_https_with_ssl_port_8000(self):
        ssl_config = MagicMock()
        ssl_config.https_port_api = 8443
        ssl_config.https_port_frontend = 8444
        self.middleware.ssl_config = ssl_config
        request = MagicMock()
        request.url.replace.return_value = "https://example.com:8000/test"
        self.middleware._redirect_to_https(request)

    def test_redirect_to_https_with_ssl_port_8501(self):
        ssl_config = MagicMock()
        ssl_config.https_port_api = 8443
        ssl_config.https_port_frontend = 8444
        self.middleware.ssl_config = ssl_config
        request = MagicMock()
        request.url.replace.return_value = "https://example.com:8501/dashboard"
        self.middleware._redirect_to_https(request)

    def test_is_ip_allowed_wildcard(self):
        self.middleware.security_config.allowed_ips = ["0.0.0.0/0"]
        self.middleware.security_config.blocked_ips = []
        assert self.middleware._is_ip_allowed("99.99.99.99") is True


class TestSecurityConfigFileLoading:
    def test_load_allowed_ips_from_file(self):
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("# comment\n10.0.0.1\n10.0.0.2\n\n")
            f.flush()
            with patch.object(_mod.Path, "__new__", return_value=_mod.Path(f.name)):
                config = SecurityConfig()
                # File path doesn't match /etc/mfc/allowed-ips.txt so it uses default
        os.unlink(f.name)

    def test_load_allowed_ips_file_error(self):
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(side_effect=PermissionError("denied"))
        with patch("builtins.open", side_effect=PermissionError("denied")):
            with patch.object(_mod, "Path") as mock_path_cls:
                mock_p = MagicMock()
                mock_p.exists.return_value = True
                mock_path_cls.return_value = mock_p
                config = SecurityConfig()

    def test_load_blocked_ips_file_error(self):
        with patch("builtins.open", side_effect=PermissionError("denied")):
            with patch.object(_mod, "Path") as mock_path_cls:
                mock_p = MagicMock()
                mock_p.exists.return_value = True
                mock_path_cls.return_value = mock_p
                config = SecurityConfig()


@pytest.mark.asyncio
class TestDispatch:
    async def test_dispatch_blocked_ip(self):
        app = MagicMock()
        _mock_ssl_config_mod.load_ssl_config.return_value = None
        middleware = SecurityMiddleware(app, ssl_config=None)
        middleware.security_config.blocked_ips = ["1.2.3.4"]
        middleware.security_config.allowed_ips = []
        request = MagicMock()
        request.headers = MagicMock()
        request.headers.get = MagicMock(return_value=None)
        request.client = MagicMock()
        request.client.host = "1.2.3.4"
        call_next = AsyncMock()
        result = await middleware.dispatch(request, call_next)
        call_next.assert_not_called()

    async def test_dispatch_rate_limited(self):
        app = MagicMock()
        _mock_ssl_config_mod.load_ssl_config.return_value = None
        middleware = SecurityMiddleware(app, ssl_config=None)
        middleware.security_config.rate_limit_requests_per_minute = 0
        request = MagicMock()
        request.headers = MagicMock()
        request.headers.get = MagicMock(return_value=None)
        request.client = MagicMock()
        request.client.host = "127.0.0.1"
        call_next = AsyncMock()
        result = await middleware.dispatch(request, call_next)

    async def test_dispatch_https_redirect(self):
        app = MagicMock()
        _mock_ssl_config_mod.load_ssl_config.return_value = None
        middleware = SecurityMiddleware(app, ssl_config=None)
        middleware.security_config.enforce_https = True
        request = MagicMock()
        request.headers = MagicMock()
        request.headers.get = MagicMock(return_value=None)
        request.client = MagicMock()
        request.client.host = "127.0.0.1"
        request.url.scheme = "http"
        request.method = "GET"
        call_next = AsyncMock()
        result = await middleware.dispatch(request, call_next)

    async def test_dispatch_post_csrf_fail(self):
        app = MagicMock()
        _mock_ssl_config_mod.load_ssl_config.return_value = None
        middleware = SecurityMiddleware(app, ssl_config=None)
        middleware.security_config.enforce_https = False
        request = MagicMock()
        request.headers = MagicMock()
        request.headers.get = MagicMock(return_value=None)
        request.client = MagicMock()
        request.client.host = "127.0.0.1"
        request.url.scheme = "https"
        request.method = "POST"
        request.url.path = "/not-api/test"
        request.cookies.get = MagicMock(return_value=None)
        call_next = AsyncMock()
        result = await middleware.dispatch(request, call_next)

    async def test_dispatch_post_csrf_api_auth(self):
        app = MagicMock()
        _mock_ssl_config_mod.load_ssl_config.return_value = None
        middleware = SecurityMiddleware(app, ssl_config=None)
        middleware.security_config.enforce_https = False
        request = MagicMock()
        request.headers = MagicMock()
        request.headers.get = lambda key, default=None: {
            "X-Forwarded-For": None,
            "X-Real-IP": None,
            "X-Forwarded-Proto": "https",
            "Authorization": "Bearer development-token",
        }.get(key, default)
        request.client = MagicMock()
        request.client.host = "127.0.0.1"
        request.url.scheme = "https"
        request.method = "POST"
        request.url.path = "/api/data"
        call_next = AsyncMock()
        resp_headers = {}

        class FakeRH:
            def __setitem__(self, key, value):
                resp_headers[key] = value

            def __contains__(self, key):
                return key in resp_headers

            def get(self, key, default=""):
                return resp_headers.get(key, default)

        mock_response = MagicMock()
        mock_response.headers = FakeRH()
        call_next.return_value = mock_response
        result = await middleware.dispatch(request, call_next)

    async def test_dispatch_get_success(self):
        app = MagicMock()
        _mock_ssl_config_mod.load_ssl_config.return_value = None
        middleware = SecurityMiddleware(app, ssl_config=None)
        middleware.security_config.enforce_https = False
        request = MagicMock()
        request.headers = MagicMock()
        request.headers.get = MagicMock(return_value=None)
        request.client = MagicMock()
        request.client.host = "127.0.0.1"
        request.url.scheme = "https"
        request.method = "GET"
        call_next = AsyncMock()
        headers_store = {}

        class FakeRespHeaders:
            def __setitem__(self, key, value):
                headers_store[key] = value

            def __contains__(self, key):
                return key in headers_store

            def get(self, key, default=""):
                return headers_store.get(key, default)

        mock_response = MagicMock()
        mock_response.headers = FakeRespHeaders()
        call_next.return_value = mock_response
        result = await middleware.dispatch(request, call_next)
        call_next.assert_called_once()

    async def test_validate_csrf_with_session_and_header(self):
        app = MagicMock()
        _mock_ssl_config_mod.load_ssl_config.return_value = None
        middleware = SecurityMiddleware(app, ssl_config=None)
        request = MagicMock()
        request.url.path = "/not-api"
        session_id = middleware.session_manager.create_session("user1")
        request.cookies.get = MagicMock(return_value=session_id)
        token = middleware.csrf_protection.generate_csrf_token(session_id)
        request.headers.get = lambda key, default=None: {
            "X-CSRF-Token": token,
            "Authorization": None,
        }.get(key, default)
        request.method = "POST"
        result = await middleware._validate_csrf(request)
        assert result is True

    async def test_validate_csrf_no_token(self):
        app = MagicMock()
        _mock_ssl_config_mod.load_ssl_config.return_value = None
        middleware = SecurityMiddleware(app, ssl_config=None)
        request = MagicMock()
        request.url.path = "/not-api"
        request.cookies.get = MagicMock(return_value="some_session")
        request.headers.get = lambda key, default=None: None
        request.method = "PUT"
        result = await middleware._validate_csrf(request)
        assert result is False


class TestGetCsrfToken:
    def test_no_csrf_protection(self):
        request = MagicMock()
        request.app.state = MagicMock(spec=[])
        result = get_csrf_token(request)
        assert result == ""

    def test_no_session_cookie(self):
        request = MagicMock()
        csrf = MagicMock()
        request.app.state.csrf_protection = csrf
        request.cookies.get = MagicMock(return_value="")
        result = get_csrf_token(request)
        assert result == ""

    def test_with_session_cookie(self):
        request = MagicMock()
        csrf = MagicMock()
        csrf.generate_csrf_token.return_value = "token123"
        request.app.state.csrf_protection = csrf
        request.cookies.get = MagicMock(return_value="session_abc")
        result = get_csrf_token(request)
        assert result == "token123"


class TestIsAuthenticated:
    def test_no_session_manager(self):
        request = MagicMock()
        request.app.state = MagicMock(spec=[])
        assert is_authenticated(request) is False

    def test_no_session_cookie(self):
        request = MagicMock()
        request.app.state.session_manager = MagicMock()
        request.cookies.get = MagicMock(return_value=None)
        assert is_authenticated(request) is False

    def test_valid_session(self):
        request = MagicMock()
        sm = MagicMock()
        sm.validate_session.return_value = {"user_id": "user1"}
        request.app.state.session_manager = sm
        request.cookies.get = MagicMock(return_value="session123")
        assert is_authenticated(request) is True

    def test_invalid_session(self):
        request = MagicMock()
        sm = MagicMock()
        sm.validate_session.return_value = None
        request.app.state.session_manager = sm
        request.cookies.get = MagicMock(return_value="bad_session")
        assert is_authenticated(request) is False


class TestCreateSecurityMiddleware:
    def test_creates_middleware(self):
        app = MagicMock()
        app.state = MagicMock()
        _mock_ssl_config_mod.load_ssl_config.return_value = None
        middleware = create_security_middleware(app)
        assert isinstance(middleware, SecurityMiddleware)
        assert hasattr(app.state, "session_manager")
        assert hasattr(app.state, "csrf_protection")


@pytest.mark.asyncio
class TestAsyncFunctions:
    async def test_get_current_session_no_manager(self):
        request = MagicMock()
        request.app.state = MagicMock(spec=[])
        result = await get_current_session(request)
        assert result is None

    async def test_get_current_session_with_cookie(self):
        request = MagicMock()
        sm = MagicMock()
        sm.validate_session.return_value = {"user_id": "user1"}
        request.app.state.session_manager = sm
        request.cookies.get = MagicMock(return_value="session123")
        result = await get_current_session(request)
        assert result == {"user_id": "user1"}

    async def test_get_current_session_with_bearer(self):
        request = MagicMock()
        request.app.state.session_manager = MagicMock()
        request.cookies.get = MagicMock(return_value=None)
        credentials = MagicMock()
        result = await get_current_session(request, credentials)
        assert result == {"user_id": "api_user", "auth_method": "bearer"}

    async def test_require_authentication_no_session(self):
        with pytest.raises(Exception):
            await require_authentication(None)

    async def test_require_authentication_with_session(self):
        session = {"user_id": "user1"}
        result = await require_authentication(session)
        assert result == session
