#!/usr/bin/env python3
"""
SSL/TLS Configuration Module for MFC Monitoring System
Provides certificate management, Let's Encrypt integration, and security configuration.
"""

import json
import logging
import os
import socket
import ssl
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SSLConfig:
    """SSL Configuration dataclass with production-ready defaults"""

    # Certificate paths
    cert_file: str = "/etc/ssl/certs/mfc-monitoring.crt"
    key_file: str = "/etc/ssl/private/mfc-monitoring.key"
    ca_file: Optional[str] = None

    # Let's Encrypt configuration
    use_letsencrypt: bool = True
    domain: str = "localhost"
    email: str = "admin@example.com"
    staging: bool = False  # Set to True for testing

    # SSL/TLS settings
    ssl_version: str = "TLSv1_2"  # Minimum TLS 1.2
    ciphers: str = "ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS"
    verify_mode: str = "CERT_NONE"  # For self-signed certificates

    # Security headers
    enable_hsts: bool = True
    hsts_max_age: int = 31536000  # 1 year
    enable_csp: bool = True

    # Ports
    https_port_api: int = 8443
    https_port_frontend: int = 8444
    wss_port_streaming: int = 8445

    # Auto-renewal settings
    auto_renew: bool = True
    renewal_days_before: int = 30

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'SSLConfig':
        """Create from dictionary"""
        return cls(**data)

class CertificateManager:
    """Manages SSL certificates including Let's Encrypt integration"""

    def __init__(self, config: SSLConfig):
        self.config = config
        self.cert_dir = Path("/etc/ssl/certs")
        self.key_dir = Path("/etc/ssl/private")
        self.letsencrypt_dir = Path("/etc/letsencrypt")

    def check_certificate_exists(self) -> bool:
        """Check if SSL certificate files exist"""
        cert_path = Path(self.config.cert_file)
        key_path = Path(self.config.key_file)

        return cert_path.exists() and key_path.exists()

    def check_certificate_validity(self) -> Tuple[bool, Optional[datetime]]:
        """
        Check if certificate is valid and return expiration date
        Returns: (is_valid, expiration_date)
        """
        try:
            # Check if certificate file exists and is readable
            with open(self.config.cert_file, 'rb'):
                pass  # File exists and is readable

            # Use openssl command to check certificate
            result = subprocess.run([
                'openssl', 'x509', '-in', self.config.cert_file,
                '-noout', '-dates'
            ], capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                # Parse expiration date from openssl output
                for line in result.stdout.split('\n'):
                    if line.startswith('notAfter='):
                        date_str = line.split('=', 1)[1]
                        # Parse openssl date format
                        try:
                            exp_date = datetime.strptime(date_str, '%b %d %H:%M:%S %Y %Z')
                            days_until_expiry = (exp_date - datetime.now()).days
                            is_valid = days_until_expiry > self.config.renewal_days_before
                            return is_valid, exp_date
                        except ValueError:
                            logger.warning(f"Could not parse certificate date: {date_str}")
                            return False, None

            return False, None

        except Exception as e:
            logger.error(f"Error checking certificate validity: {e}")
            return False, None

    def generate_self_signed_certificate(self) -> bool:
        """Generate self-signed certificate for development/testing"""
        try:
            # Ensure directories exist
            self.cert_dir.mkdir(parents=True, exist_ok=True)
            self.key_dir.mkdir(parents=True, exist_ok=True)

            # Generate private key
            key_cmd = [
                'openssl', 'genrsa', '-out', self.config.key_file, '2048'
            ]

            result = subprocess.run(key_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                logger.error(f"Failed to generate private key: {result.stderr}")
                return False

            # Generate certificate
            cert_cmd = [
                'openssl', 'req', '-new', '-x509', '-key', self.config.key_file,
                '-out', self.config.cert_file, '-days', '365',
                '-subj', f'/C=US/ST=State/L=City/O=MFC-Research/CN={self.config.domain}'
            ]

            result = subprocess.run(cert_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                logger.error(f"Failed to generate certificate: {result.stderr}")
                return False

            # Set appropriate permissions
            os.chmod(self.config.key_file, 0o600)
            os.chmod(self.config.cert_file, 0o644)

            logger.info(f"Generated self-signed certificate for {self.config.domain}")
            return True

        except Exception as e:
            logger.error(f"Error generating self-signed certificate: {e}")
            return False

    def request_letsencrypt_certificate(self) -> bool:
        """Request certificate from Let's Encrypt using certbot"""
        if not self.config.use_letsencrypt:
            return False

        try:
            # Check if certbot is available
            check_result = subprocess.run(['which', 'certbot'], capture_output=True)
            if check_result.returncode != 0:
                logger.error("Certbot not found. Please install certbot first.")
                logger.info("Install with: sudo apt-get install certbot")
                return False

            # Build certbot command
            certbot_cmd = [
                'certbot', 'certonly', '--standalone',
                '--email', self.config.email,
                '--agree-tos',
                '--no-eff-email',
                '-d', self.config.domain
            ]

            if self.config.staging:
                certbot_cmd.append('--staging')

            # Run certbot
            logger.info(f"Requesting Let's Encrypt certificate for {self.config.domain}")
            result = subprocess.run(certbot_cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                # Update config to point to Let's Encrypt files
                self.config.cert_file = f"/etc/letsencrypt/live/{self.config.domain}/fullchain.pem"
                self.config.key_file = f"/etc/letsencrypt/live/{self.config.domain}/privkey.pem"

                logger.info("Successfully obtained Let's Encrypt certificate")
                return True
            else:
                logger.error(f"Certbot failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error requesting Let's Encrypt certificate: {e}")
            return False

    def setup_auto_renewal(self) -> bool:
        """Setup automatic certificate renewal with cron"""
        try:
            # Create renewal script
            renewal_script = """#!/bin/bash
# Auto-renewal script for MFC monitoring certificates
certbot renew --quiet --post-hook "systemctl reload nginx"
# Restart MFC monitoring services
pkill -f "uvicorn.*dashboard_api"
pkill -f "streamlit.*dashboard_frontend"
pkill -f "websocket.*realtime_streamer"
sleep 5
# Services will be restarted by supervisor/systemd
"""

            renewal_script_path = Path("/usr/local/bin/mfc-cert-renewal.sh")
            renewal_script_path.write_text(renewal_script)
            renewal_script_path.chmod(0o755)

            # Add to crontab (check daily at 2 AM)
            cron_entry = "0 2 * * * /usr/local/bin/mfc-cert-renewal.sh"

            # Add cron job via subprocess
            result = subprocess.run([
                'bash', '-c', f'(crontab -l 2>/dev/null; echo "{cron_entry}") | crontab -'
            ], capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("Auto-renewal cron job installed successfully")
                return True
            else:
                logger.warning(f"Failed to install cron job: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error setting up auto-renewal: {e}")
            return False

class SecurityHeaders:
    """Manage security headers for HTTPS deployment"""

    @staticmethod
    def get_security_headers(config: SSLConfig) -> Dict[str, str]:
        """Get dictionary of security headers"""
        headers = {}

        if config.enable_hsts:
            headers['Strict-Transport-Security'] = f'max-age={config.hsts_max_age}; includeSubDomains'

        if config.enable_csp:
            # Content Security Policy for Streamlit and FastAPI
            csp_policy = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                "font-src 'self' https://fonts.gstatic.com; "
                "img-src 'self' data: https:; "
                "connect-src 'self' ws: wss:; "
                "frame-ancestors 'none'"
            )
            headers['Content-Security-Policy'] = csp_policy

        # Additional security headers
        headers.update({
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
        })

        return headers

class SSLContextManager:
    """Create and manage SSL contexts for different services"""

    def __init__(self, config: SSLConfig):
        self.config = config

    def create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context with security best practices"""
        # Create context with minimum TLS 1.2
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

        # Set minimum version
        context.minimum_version = ssl.TLSVersion.TLSv1_2

        # Load certificate and key
        try:
            context.load_cert_chain(self.config.cert_file, self.config.key_file)
        except Exception as e:
            logger.error(f"Failed to load SSL certificate: {e}")
            raise

        # Set cipher suites (prefer modern, secure ciphers)
        context.set_ciphers(self.config.ciphers)

        # Additional security options
        context.options |= ssl.OP_NO_SSLv2
        context.options |= ssl.OP_NO_SSLv3
        context.options |= ssl.OP_NO_TLSv1
        context.options |= ssl.OP_NO_TLSv1_1
        context.options |= ssl.OP_SINGLE_DH_USE
        context.options |= ssl.OP_SINGLE_ECDH_USE

        return context

    def get_uvicorn_ssl_config(self) -> Dict[str, Any]:
        """Get SSL configuration for uvicorn (FastAPI)"""
        return {
            'ssl_keyfile': self.config.key_file,
            'ssl_certfile': self.config.cert_file,
            'ssl_version': ssl.PROTOCOL_TLS_SERVER,
            'ssl_cert_reqs': ssl.CERT_NONE,
            'ssl_ca_certs': self.config.ca_file,
            'ssl_ciphers': self.config.ciphers
        }

def load_ssl_config(config_file: Optional[str] = None) -> SSLConfig:
    """Load SSL configuration from file or environment variables"""

    # Check for development config first
    dev_config_file = Path(__file__).parent / "ssl_config_dev.json"
    if dev_config_file.exists() and config_file is None:
        config_file = str(dev_config_file)
        logger.info("Using development SSL configuration")

    # Default config file location
    if config_file is None:
        config_file = os.getenv('MFC_SSL_CONFIG', '/etc/mfc/ssl-config.json')

    config = SSLConfig()

    # Try to load from file
    if os.path.exists(config_file):
        try:
            with open(config_file) as f:
                data = json.load(f)
            config = SSLConfig.from_dict(data)
            logger.info(f"Loaded SSL config from {config_file}")
        except Exception as e:
            logger.warning(f"Failed to load config from {config_file}: {e}")

    # Override with environment variables
    env_mapping = {
        'MFC_SSL_DOMAIN': 'domain',
        'MFC_SSL_EMAIL': 'email',
        'MFC_SSL_CERT_FILE': 'cert_file',
        'MFC_SSL_KEY_FILE': 'key_file',
        'MFC_SSL_USE_LETSENCRYPT': 'use_letsencrypt',
        'MFC_SSL_STAGING': 'staging',
        'MFC_HTTPS_API_PORT': 'https_port_api',
        'MFC_HTTPS_FRONTEND_PORT': 'https_port_frontend',
        'MFC_WSS_STREAMING_PORT': 'wss_port_streaming'
    }

    for env_var, attr_name in env_mapping.items():
        env_value = os.getenv(env_var)
        if env_value is not None:
            # Handle boolean conversion
            if attr_name in ['use_letsencrypt', 'staging', 'auto_renew', 'enable_hsts', 'enable_csp']:
                converted_value: Union[bool, int, str] = env_value.lower() in ('true', '1', 'yes', 'on')
            # Handle integer conversion
            elif attr_name in ['https_port_api', 'https_port_frontend', 'wss_port_streaming', 'hsts_max_age', 'renewal_days_before']:
                converted_value = int(env_value)
            else:
                converted_value = env_value

            setattr(config, attr_name, converted_value)

    return config

def save_ssl_config(config: SSLConfig, config_file: Optional[str] = None) -> bool:
    """Save SSL configuration to file"""
    if config_file is None:
        config_file = os.getenv('MFC_SSL_CONFIG', '/etc/mfc/ssl-config.json')

    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(config_file), exist_ok=True)

        with open(config_file, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

        logger.info(f"Saved SSL config to {config_file}")
        return True

    except Exception as e:
        logger.error(f"Failed to save config to {config_file}: {e}")
        return False

def initialize_ssl_infrastructure(config: Optional[SSLConfig] = None) -> Tuple[bool, SSLConfig]:
    """
    Initialize SSL infrastructure - certificates, configuration, etc.
    Returns: (success, config)
    """
    if config is None:
        config = load_ssl_config()

    cert_manager = CertificateManager(config)

    # Check if certificates exist and are valid
    cert_exists = cert_manager.check_certificate_exists()
    cert_valid = False

    if cert_exists:
        cert_valid, expiry_date = cert_manager.check_certificate_validity()
        if expiry_date:
            logger.info(f"Certificate expires on: {expiry_date}")

    # Handle certificate setup
    if not cert_exists or not cert_valid:
        logger.info("Setting up SSL certificates...")

        if config.use_letsencrypt and config.domain != "localhost":
            # Try Let's Encrypt first
            success = cert_manager.request_letsencrypt_certificate()
            if not success:
                logger.warning("Let's Encrypt failed, falling back to self-signed certificate")
                success = cert_manager.generate_self_signed_certificate()
        else:
            # Use self-signed certificate
            success = cert_manager.generate_self_signed_certificate()

        if not success:
            logger.error("Failed to setup SSL certificates")
            return False, config

    # Setup auto-renewal if using Let's Encrypt
    if config.use_letsencrypt and config.auto_renew:
        cert_manager.setup_auto_renewal()

    # Save configuration
    save_ssl_config(config)

    logger.info("SSL infrastructure initialized successfully")
    return True, config

def test_ssl_connection(host: str, port: int, timeout: int = 10) -> bool:
    """Test SSL connection to verify certificate is working"""
    try:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE  # For self-signed certificates

        with socket.create_connection((host, port), timeout) as sock:
            with context.wrap_socket(sock, server_hostname=host) as ssock:
                logger.info(f"SSL connection to {host}:{port} successful")
                logger.info(f"SSL version: {ssock.version()}")
                return True

    except Exception as e:
        logger.error(f"SSL connection test failed for {host}:{port}: {e}")
        return False

if __name__ == "__main__":
    # Command-line interface for SSL management
    import argparse

    parser = argparse.ArgumentParser(description="MFC SSL Configuration Manager")
    parser.add_argument('--init', action='store_true', help='Initialize SSL infrastructure')
    parser.add_argument('--test', action='store_true', help='Test SSL connections')
    parser.add_argument('--renew', action='store_true', help='Renew certificates')
    parser.add_argument('--domain', default='localhost', help='Domain name')
    parser.add_argument('--email', default='admin@example.com', help='Email for Let\'s Encrypt')
    parser.add_argument('--staging', action='store_true', help='Use Let\'s Encrypt staging')

    args = parser.parse_args()

    if args.init:
        config = SSLConfig(domain=args.domain, email=args.email, staging=args.staging)
        success, final_config = initialize_ssl_infrastructure(config)
        if success:
            print("✅ SSL infrastructure initialized successfully")
        else:
            print("❌ SSL initialization failed")
            exit(1)

    elif args.test:
        config = load_ssl_config()
        ports = [config.https_port_api, config.https_port_frontend, config.wss_port_streaming]
        for port in ports:
            test_ssl_connection(config.domain, port)

    elif args.renew:
        config = load_ssl_config()
        cert_manager = CertificateManager(config)
        if config.use_letsencrypt:
            # Run certbot renew
            result = subprocess.run(['certbot', 'renew'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Certificates renewed successfully")
            else:
                print(f"❌ Certificate renewal failed: {result.stderr}")
        else:
            print("Auto-renewal only available for Let's Encrypt certificates")

    else:
        parser.print_help()
