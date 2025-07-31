#!/usr/bin/env python3
"""
Development SSL Configuration for MFC Monitoring System
Sets up SSL certificates in local directories for development use.
"""

import os
import logging
from pathlib import Path
from ssl_config import SSLConfig, save_ssl_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dev_ssl_config() -> SSLConfig:
    """Create SSL configuration suitable for development"""
    
    # Use local directories
    project_root = Path(__file__).parent.parent.parent
    ssl_dir = project_root / "ssl_certificates"
    ssl_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (ssl_dir / "certs").mkdir(exist_ok=True)
    (ssl_dir / "private").mkdir(exist_ok=True)
    
    dev_config = SSLConfig(
        # Local certificate paths
        cert_file=str(ssl_dir / "certs" / "mfc-monitoring.crt"),
        key_file=str(ssl_dir / "private" / "mfc-monitoring.key"),
        ca_file=None,
        
        # Development settings
        use_letsencrypt=False,  # Use self-signed for development
        domain="localhost",
        email="dev@mfc-project.local",
        staging=True,
        
        # Relaxed security for development
        verify_mode="CERT_NONE",
        
        # Development ports (avoiding privileged ports)
        https_port_api=8443,
        https_port_frontend=8444,
        wss_port_streaming=8445,
        
        # Security headers (still enabled for testing)
        enable_hsts=True,
        hsts_max_age=86400,  # 1 day for development
        enable_csp=True
    )
    
    return dev_config

def setup_development_ssl() -> bool:
    """Set up SSL infrastructure for development"""
    logger.info("Setting up SSL infrastructure for development...")
    
    try:
        # Create development configuration
        config = create_dev_ssl_config()
        
        # Generate self-signed certificates for development
        success = generate_self_signed_certificates(config)
        if not success:
            logger.error("Failed to generate self-signed certificates")
            return False
        
        # Save the configuration
        config_file = Path(__file__).parent / "ssl_config_dev.json"
        save_success = save_ssl_config(config, str(config_file))
        if not save_success:
            logger.error("Failed to save SSL configuration")
            return False
        
        logger.info("✅ Development SSL infrastructure setup complete")
        logger.info(f"Certificate files created in: {Path(config.cert_file).parent}")
        logger.info(f"Configuration saved to: {config_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error setting up development SSL: {e}")
        return False

def generate_self_signed_certificates(config: SSLConfig) -> bool:
    """Generate self-signed certificates for development"""
    import subprocess
    
    try:
        # Ensure directories exist
        cert_dir = Path(config.cert_file).parent
        key_dir = Path(config.key_file).parent
        cert_dir.mkdir(parents=True, exist_ok=True)
        key_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate private key
        logger.info("Generating private key...")
        key_cmd = [
            'openssl', 'genrsa', '-out', config.key_file, '2048'
        ]
        key_result = subprocess.run(key_cmd, capture_output=True, text=True)
        if key_result.returncode != 0:
            logger.error(f"Failed to generate private key: {key_result.stderr}")
            return False
        
        # Generate self-signed certificate
        logger.info("Generating self-signed certificate...")
        cert_cmd = [
            'openssl', 'req', '-new', '-x509',
            '-key', config.key_file,
            '-out', config.cert_file,
            '-days', '365',
            '-subj', f'/C=US/ST=Development/L=Local/O=MFC Project/CN={config.domain}'
        ]
        cert_result = subprocess.run(cert_cmd, capture_output=True, text=True)
        if cert_result.returncode != 0:
            logger.error(f"Failed to generate certificate: {cert_result.stderr}")
            return False
        
        # Set appropriate permissions
        os.chmod(config.key_file, 0o600)  # Private key readable only by owner
        os.chmod(config.cert_file, 0o644)  # Certificate readable by all
        
        logger.info("✅ Self-signed certificates generated successfully")
        logger.info(f"Certificate: {config.cert_file}")
        logger.info(f"Private key: {config.key_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating certificates: {e}")
        return False

def main():
    """Main entry point for development SSL setup"""
    print("🔐 MFC Development SSL Setup")
    print("=" * 40)
    
    # Check if OpenSSL is available
    try:
        import subprocess
        result = subprocess.run(['openssl', 'version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ OpenSSL not found. Please install OpenSSL first.")
            print("   Ubuntu/Debian: sudo apt-get install openssl")
            print("   macOS: brew install openssl")
            return False
        print(f"✅ OpenSSL version: {result.stdout.strip()}")
    except FileNotFoundError:
        print("❌ OpenSSL not found. Please install OpenSSL first.")
        return False
    
    # Setup development SSL
    success = setup_development_ssl()
    if success:
        print("\n🎉 Development SSL setup complete!")
        print("\nNext steps:")
        print("1. Test SSL configuration: pixi run ssl-test")
        print("2. Start monitoring with HTTPS: pixi run start-monitoring")
        print("3. Start monitoring without HTTPS: pixi run start-monitoring-http")
    else:
        print("\n❌ Development SSL setup failed!")
        return False
    
    return True

if __name__ == "__main__":
    main()