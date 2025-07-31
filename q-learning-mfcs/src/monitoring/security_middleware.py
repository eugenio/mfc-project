#!/usr/bin/env python3
"""
Security Middleware for MFC Monitoring System
Provides comprehensive security features including headers, CSRF protection, and session management.
"""

import os
import sys
import hmac
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
import json
import base64

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ssl_config import SSLConfig, SecurityHeaders, load_ssl_config  # noqa: E402
from fastapi import Request, Response, HTTPException, Depends  # noqa: E402
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials  # noqa: E402
from starlette.middleware.base import BaseHTTPMiddleware  # noqa: E402
from starlette.responses import RedirectResponse  # noqa: E402

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityConfig:
    """Security configuration class"""

    def __init__(self):
        # Session configuration
        self.session_timeout_minutes = 60
        self.session_secret_key = os.getenv("MFC_SESSION_SECRET", self._generate_secret_key())
        self.session_cookie_name = "mfc_session"

        # CSRF configuration
        self.csrf_token_name = "csrf_token"
        self.csrf_header_name = "X-CSRF-Token"
        self.csrf_secret_key = os.getenv("MFC_CSRF_SECRET", self._generate_secret_key())

        # JWT configuration
        self.jwt_secret_key = os.getenv("MFC_JWT_SECRET", self._generate_secret_key())
        self.jwt_algorithm = "HS256"
        self.jwt_expiry_hours = 24

        # Rate limiting
        self.rate_limit_requests_per_minute = 60
        self.rate_limit_burst = 10

        # IP filtering
        self.allowed_ips: List[str] = self._load_allowed_ips()
        self.blocked_ips: List[str] = self._load_blocked_ips()

        # Content Security Policy
        self.csp_policy = self._get_default_csp()

        # Feature flags
        self.enforce_https = True
        self.strict_transport_security = True
        self.content_type_nosniff = True
        self.frame_options_deny = True
        self.xss_protection = True
        self.referrer_policy_strict = True

    def _generate_secret_key(self) -> str:
        """Generate a secure random key"""
        return secrets.token_urlsafe(32)

    def _load_allowed_ips(self) -> List[str]:
        """Load allowed IP addresses"""
        allowed_file = Path("/etc/mfc/allowed-ips.txt")
        if allowed_file.exists():
            try:
                with open(allowed_file, 'r') as f:
                    return [line.strip() for line in f if line.strip() and not line.startswith('#')]
            except Exception as e:
                logger.warning(f"Failed to load allowed IPs: {e}")

        # Default allowed IPs (localhost and private networks)
        return ["127.0.0.1", "::1", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"]

    def _load_blocked_ips(self) -> List[str]:
        """Load blocked IP addresses"""
        blocked_file = Path("/etc/mfc/blocked-ips.txt")
        if blocked_file.exists():
            try:
                with open(blocked_file, 'r') as f:
                    return [line.strip() for line in f if line.strip() and not line.startswith('#')]
            except Exception as e:
                logger.warning(f"Failed to load blocked IPs: {e}")

        return []

    def _get_default_csp(self) -> str:
        """Get default Content Security Policy"""
        return (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net https://unpkg.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: https: blob:; "
            "connect-src 'self' ws: wss: https:; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )

class SessionManager:
    """Secure session management"""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.active_sessions: Dict[str, Dict] = {}

    def create_session(self, user_id: str, user_data: Optional[Dict[str, Any]] = None) -> str:
        """Create a new session"""
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(minutes=self.config.session_timeout_minutes)

        session_data = {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "expires_at": expires_at.isoformat(),
            "last_activity": datetime.now().isoformat(),
            "user_data": user_data or {}
        }

        self.active_sessions[session_id] = session_data
        return session_id

    def validate_session(self, session_id: str) -> Optional[Dict]:
        """Validate and return session data"""
        if not session_id or session_id not in self.active_sessions:
            return None

        session = self.active_sessions[session_id]
        expires_at = datetime.fromisoformat(session["expires_at"])

        if datetime.now() > expires_at:
            # Session expired
            del self.active_sessions[session_id]
            return None

        # Update last activity
        session["last_activity"] = datetime.now().isoformat()
        return session

    def destroy_session(self, session_id: str) -> bool:
        """Destroy a session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            return True
        return False

    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        now = datetime.now()
        expired_sessions = []

        for session_id, session in self.active_sessions.items():
            expires_at = datetime.fromisoformat(session["expires_at"])
            if now > expires_at:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            del self.active_sessions[session_id]

        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

class CSRFProtection:
    """CSRF token management"""

    def __init__(self, config: SecurityConfig):
        self.config = config

    def generate_csrf_token(self, session_id: str) -> str:
        """Generate CSRF token for session"""
        timestamp = str(int(datetime.now().timestamp()))
        data = f"{session_id}:{timestamp}"

        signature = hmac.new(
            self.config.csrf_secret_key.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()

        token = base64.b64encode(f"{data}:{signature}".encode()).decode()
        return token

    def validate_csrf_token(self, token: str, session_id: str) -> bool:
        """Validate CSRF token"""
        try:
            decoded = base64.b64decode(token.encode()).decode()
            parts = decoded.split(':')

            if len(parts) != 3:
                return False

            token_session_id, timestamp, signature = parts

            # Check session ID matches
            if token_session_id != session_id:
                return False

            # Check timestamp (token valid for 1 hour)
            token_time = datetime.fromtimestamp(int(timestamp))
            if datetime.now() - token_time > timedelta(hours=1):
                return False

            # Verify signature
            data = f"{token_session_id}:{timestamp}"
            expected_signature = hmac.new(
                self.config.csrf_secret_key.encode(),
                data.encode(),
                hashlib.sha256
            ).hexdigest()

            return hmac.compare_digest(signature, expected_signature)

        except Exception as e:
            logger.warning(f"CSRF token validation error: {e}")
            return False

class RateLimiter:
    """Rate limiting implementation"""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.request_counts: Dict[str, List[datetime]] = {}

    def is_rate_limited(self, client_ip: str) -> bool:
        """Check if client is rate limited"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)

        # Clean old requests
        if client_ip in self.request_counts:
            self.request_counts[client_ip] = [
                req_time for req_time in self.request_counts[client_ip]
                if req_time > minute_ago
            ]
        else:
            self.request_counts[client_ip] = []

        # Check rate limit
        recent_requests = len(self.request_counts[client_ip])

        if recent_requests >= self.config.rate_limit_requests_per_minute:
            logger.warning(f"Rate limit exceeded for IP {client_ip}")
            return True

        # Record this request
        self.request_counts[client_ip].append(now)
        return False

class SecurityMiddleware(BaseHTTPMiddleware):
    """Comprehensive security middleware"""

    def __init__(self, app, ssl_config: Optional[SSLConfig] = None):
        super().__init__(app)
        self.ssl_config = ssl_config or load_ssl_config()
        self.security_config = SecurityConfig()
        self.session_manager = SessionManager(self.security_config)
        self.csrf_protection = CSRFProtection(self.security_config)
        self.rate_limiter = RateLimiter(self.security_config)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Main security middleware dispatch"""

        # Get client IP
        client_ip = self._get_client_ip(request)

        # IP filtering
        if not self._is_ip_allowed(client_ip):
            logger.warning(f"Blocked request from IP: {client_ip}")
            return self._create_error_response("Access denied", 403)

        # Rate limiting
        if self.rate_limiter.is_rate_limited(client_ip):
            return self._create_error_response("Rate limit exceeded", 429)

        # HTTPS enforcement
        if self.security_config.enforce_https and not self._is_secure_request(request):
            return self._redirect_to_https(request)

        # CSRF protection for state-changing methods
        if request.method in ["POST", "PUT", "DELETE", "PATCH"]:
            if not await self._validate_csrf(request):
                return self._create_error_response("CSRF validation failed", 403)

        # Process request
        response = await call_next(request)

        # Add security headers
        self._add_security_headers(response)

        # Set secure cookies
        self._set_secure_cookie_flags(response)

        return response

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        # Check for forwarded headers (common in reverse proxy setups)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

    def _is_ip_allowed(self, client_ip: str) -> bool:
        """Check if IP address is allowed"""
        # Check blocked IPs first
        if client_ip in self.security_config.blocked_ips:
            return False

        # If no allowed IPs configured, allow all (except blocked)
        if not self.security_config.allowed_ips:
            return True

        # Check allowed IPs (support CIDR notation in production)
        for allowed in self.security_config.allowed_ips:
            if client_ip == allowed or allowed == "0.0.0.0/0":
                return True
            # In production, add proper CIDR checking here

        return False

    def _is_secure_request(self, request: Request) -> bool:
        """Check if request is secure (HTTPS)"""
        return (
            request.url.scheme == "https" or
            request.headers.get("X-Forwarded-Proto") == "https" or
            request.headers.get("X-Forwarded-SSL") == "on"
        )

    def _redirect_to_https(self, request: Request) -> RedirectResponse:
        """Redirect HTTP to HTTPS"""
        https_url = request.url.replace(scheme="https")
        if self.ssl_config:
            # Replace port if needed
            if ":8000" in str(https_url):
                https_url = str(https_url).replace(":8000", f":{self.ssl_config.https_port_api}")
            elif ":8501" in str(https_url):
                https_url = str(https_url).replace(":8501", f":{self.ssl_config.https_port_frontend}")

        return RedirectResponse(url=str(https_url), status_code=301)

    async def _validate_csrf(self, request: Request) -> bool:
        """Validate CSRF token"""
        # Skip CSRF for API endpoints with proper authentication
        if request.url.path.startswith("/api/") and self._has_valid_api_auth(request):
            return True

        # Get session ID from cookie
        session_cookie = request.cookies.get(self.security_config.session_cookie_name)
        if not session_cookie:
            return False

        # Get CSRF token from header or form
        csrf_token = (
            request.headers.get(self.security_config.csrf_header_name) or
            (await request.form()).get(self.security_config.csrf_token_name) if request.method == "POST" else None
        )

        if not csrf_token:
            return False

        return self.csrf_protection.validate_csrf_token(csrf_token, session_cookie)

    def _has_valid_api_auth(self, request: Request) -> bool:
        """Check if request has valid API authentication"""
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return False

        token = auth_header.split(" ", 1)[1]
        expected_token = os.getenv("MFC_API_TOKEN", "development-token")

        return hmac.compare_digest(token, expected_token)

    def _add_security_headers(self, response: Response):
        """Add comprehensive security headers"""
        if self.ssl_config:
            security_headers = SecurityHeaders.get_security_headers(self.ssl_config)
            for header, value in security_headers.items():
                response.headers[header] = value

        # Additional security headers
        if self.security_config.content_type_nosniff:
            response.headers["X-Content-Type-Options"] = "nosniff"

        if self.security_config.frame_options_deny:
            response.headers["X-Frame-Options"] = "DENY"

        if self.security_config.xss_protection:
            response.headers["X-XSS-Protection"] = "1; mode=block"

        if self.security_config.referrer_policy_strict:
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Content Security Policy
        response.headers["Content-Security-Policy"] = self.security_config.csp_policy

        # Feature Policy / Permissions Policy
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=(), "
            "payment=(), usb=(), magnetometer=(), gyroscope=()"
        )

        # Cache control for sensitive pages
        if "login" in str(response.headers.get("location", "")).lower():
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"

    def _set_secure_cookie_flags(self, response: Response):
        """Set secure flags on cookies"""
        if "Set-Cookie" in response.headers:
            cookies = response.headers.getlist("Set-Cookie")
            secure_cookies = []

            for cookie in cookies:
                # Add security flags
                if "Secure" not in cookie and self.ssl_config:
                    cookie += "; Secure"

                if "HttpOnly" not in cookie:
                    cookie += "; HttpOnly"

                if "SameSite" not in cookie:
                    cookie += "; SameSite=Strict"

                secure_cookies.append(cookie)

            # Replace cookies with secure versions
            response.headers.pop("Set-Cookie")
            for cookie in secure_cookies:
                response.headers.append("Set-Cookie", cookie)

    def _create_error_response(self, message: str, status_code: int) -> Response:
        """Create error response with security headers"""
        response = Response(
            content=json.dumps({"error": message, "status_code": status_code}),
            status_code=status_code,
            media_type="application/json"
        )
        self._add_security_headers(response)
        return response

# Authentication dependencies for FastAPI
security_bearer = HTTPBearer(auto_error=False)

async def get_current_session(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security_bearer)
) -> Optional[Dict]:
    """Get current session from request"""
    # Get session manager from middleware
    session_manager = getattr(request.app.state, "session_manager", None)
    if not session_manager:
        return None

    # Try session cookie first
    session_cookie = request.cookies.get("mfc_session")
    if session_cookie:
        return session_manager.validate_session(session_cookie)

    # Try bearer token
    if credentials:
        # In production, validate JWT token here
        return {"user_id": "api_user", "auth_method": "bearer"}

    return None

async def require_authentication(session: Dict = Depends(get_current_session)) -> Dict:
    """Require authentication for endpoint"""
    if not session:
        raise HTTPException(status_code=401, detail="Authentication required")
    return session

def create_security_middleware(app, ssl_config: Optional[SSLConfig] = None) -> SecurityMiddleware:
    """Factory function to create security middleware"""
    middleware = SecurityMiddleware(app, ssl_config)

    # Store session manager in app state for dependency injection
    app.state.session_manager = middleware.session_manager
    app.state.csrf_protection = middleware.csrf_protection

    return middleware

# Utility functions for templates
def get_csrf_token(request: Request) -> str:
    """Get CSRF token for use in templates"""
    csrf_protection = getattr(request.app.state, "csrf_protection", None)
    if not csrf_protection:
        return ""

    session_cookie = request.cookies.get("mfc_session", "")
    if not session_cookie:
        return ""

    return csrf_protection.generate_csrf_token(session_cookie)

def is_authenticated(request: Request) -> bool:
    """Check if user is authenticated"""
    session_manager = getattr(request.app.state, "session_manager", None)
    if not session_manager:
        return False

    session_cookie = request.cookies.get("mfc_session")
    if not session_cookie:
        return False

    session = session_manager.validate_session(session_cookie)
    return session is not None

if __name__ == "__main__":
    # Test security configuration
    config = SecurityConfig()
    print("Security Configuration:")
    print(f"  Session timeout: {config.session_timeout_minutes} minutes")
    print(f"  Rate limit: {config.rate_limit_requests_per_minute} requests/minute")
    print(f"  HTTPS enforcement: {config.enforce_https}")
    print(f"  Allowed IPs: {len(config.allowed_ips)}")
    print(f"  CSP policy length: {len(config.csp_policy)} characters")
