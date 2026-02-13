# MFC Monitoring System - HTTPS Deployment Guide

**Version**: 1.2.0  
**Date**: 2025-07-31  
**Status**: Production Ready  

## Overview

This guide provides comprehensive instructions for deploying the MFC Monitoring System with HTTPS security in production environments. The system includes secure implementations of:

- **FastAPI Dashboard API** (Port 8443 HTTPS)
- **Streamlit Frontend** (Port 8444 HTTPS)
- **WebSocket Streaming** (Port 8445 WSS)

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [SSL Certificate Setup](#ssl-certificate-setup)
4. [Service Configuration](#service-configuration)
5. [Security Configuration](#security-configuration)
6. [Deployment Methods](#deployment-methods)
7. [Monitoring and Maintenance](#monitoring-and-maintenance)
8. [Troubleshooting](#troubleshooting)
9. [Security Best Practices](#security-best-practices)

## Prerequisites

### System Requirements

- **Operating System**: Ubuntu 20.04+ / CentOS 8+ / Debian 11+
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 10GB free space minimum
- **Network**: Internet access for Let's Encrypt certificates

### Required Packages

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3 python3-pip certbot nginx openssl curl

# CentOS/RHEL
sudo dnf install -y python3 python3-pip certbot nginx openssl curl

# Or use the project's pixi environment
cd /path/to/mfc-project
pixi install
```

### Network Configuration

Ensure the following ports are accessible:

- **8443**: HTTPS API Server
- **8444**: HTTPS Frontend Server  
- **8445**: WSS WebSocket Server
- **80**: HTTP (for Let's Encrypt validation)
- **443**: HTTPS (optional, for reverse proxy)

## Quick Start

### 1. Initialize SSL Infrastructure

```bash
cd q-learning-mfcs/src/monitoring

# Interactive SSL setup
python ssl_config.py --init --domain your-domain.com --email admin@your-domain.com

# Or use the startup script
python start_monitoring.py --init-ssl
```

### 2. Start All Services

```bash
# Start with HTTPS enabled
python start_monitoring.py

# Or start individual services
python dashboard_api.py --host 0.0.0.0 --port 8443
python dashboard_frontend.py run_https 8444
python realtime_streamer.py --host 0.0.0.0 --port 8445
```

### 3. Verify Deployment

```bash
# Test SSL connections
python start_monitoring.py --test-ssl

# Check service status
python start_monitoring.py --status
```

## SSL Certificate Setup

### Option 1: Let's Encrypt (Recommended for Production)

```bash
# Configure domain and email
python cert_manager.py --init --domain your-domain.com --email admin@your-domain.com

# Setup automatic renewal
python cert_manager.py --setup-cron

# Monitor certificate status
python cert_manager.py --monitor
```

**Requirements for Let's Encrypt:**
- Valid domain name pointing to your server
- Port 80 accessible for validation
- No conflicting web servers

### Option 2: Self-Signed Certificates (Development/Testing)

```bash
# Generate self-signed certificates
python ssl_config.py --init --domain localhost

# Certificates will be created in /etc/ssl/
```

**Note**: Self-signed certificates will show security warnings in browsers.

### Option 3: Custom Certificates

Place your certificates in the configured locations:

```bash
# Default locations
/etc/ssl/certs/mfc-monitoring.crt    # Certificate file
/etc/ssl/private/mfc-monitoring.key  # Private key file

# Ensure proper permissions
sudo chown root:ssl-cert /etc/ssl/private/mfc-monitoring.key
sudo chmod 640 /etc/ssl/private/mfc-monitoring.key
```

## Service Configuration

### SSL Configuration File

Create `/etc/mfc/ssl-config.json`:

```json
{
  "domain": "your-domain.com",
  "email": "admin@your-domain.com",
  "use_letsencrypt": true,
  "staging": false,
  "cert_file": "/etc/letsencrypt/live/your-domain.com/fullchain.pem",
  "key_file": "/etc/letsencrypt/live/your-domain.com/privkey.pem",
  "https_port_api": 8443,
  "https_port_frontend": 8444,
  "wss_port_streaming": 8445,
  "enable_hsts": true,
  "enable_csp": true,
  "auto_renew": true
}
```

### Environment Variables

```bash
# SSL Configuration
export MFC_SSL_DOMAIN="your-domain.com"
export MFC_SSL_EMAIL="admin@your-domain.com"
export MFC_SSL_USE_LETSENCRYPT="true"

# Security Keys (generate secure values)
export MFC_SESSION_SECRET="your-secure-session-key"
export MFC_CSRF_SECRET="your-secure-csrf-key"
export MFC_JWT_SECRET="your-secure-jwt-key"
export MFC_API_TOKEN="your-secure-api-token"

# Ports
export MFC_HTTPS_API_PORT="8443"
export MFC_HTTPS_FRONTEND_PORT="8444"
export MFC_WSS_STREAMING_PORT="8445"
```

## Security Configuration

### Firewall Setup

```bash
# Ubuntu/Debian (ufw)
sudo ufw allow 8443/tcp comment 'MFC API HTTPS'
sudo ufw allow 8444/tcp comment 'MFC Frontend HTTPS'
sudo ufw allow 8445/tcp comment 'MFC WebSocket WSS'
sudo ufw allow 80/tcp comment 'HTTP for Let\'s Encrypt'
sudo ufw enable

# CentOS/RHEL (firewalld)
sudo firewall-cmd --permanent --add-port=8443/tcp --add-port=8444/tcp --add-port=8445/tcp
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --reload
```

### Nginx Reverse Proxy (Optional)

Create `/etc/nginx/sites-available/mfc-monitoring`:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    # SSL Configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;

    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";

    # API Backend
    location /api/ {
        proxy_pass https://127.0.0.1:8443/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Frontend
    location / {
        proxy_pass https://127.0.0.1:8444/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket
    location /ws/ {
        proxy_pass https://127.0.0.1:8445/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable the configuration:

```bash
sudo ln -s /etc/nginx/sites-available/mfc-monitoring /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## Deployment Methods

### Method 1: Direct Deployment

```bash
# Clone repository
git clone <repository-url>
cd mfc-project/q-learning-mfcs/src/monitoring

# Install dependencies
pip install -r requirements.txt

# Initialize SSL
python start_monitoring.py --init-ssl

# Start services
python start_monitoring.py
```

### Method 2: Systemd Services

Create service files:

**`/etc/systemd/system/mfc-monitoring.service`**:

```ini
[Unit]
Description=MFC Monitoring System
After=network.target

[Service]
Type=exec
User=mfc
Group=mfc
WorkingDirectory=/opt/mfc-project/q-learning-mfcs/src/monitoring
Environment=MFC_SSL_DOMAIN=your-domain.com
Environment=MFC_SSL_EMAIL=admin@your-domain.com
ExecStart=/usr/bin/python3 start_monitoring.py --daemon
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable mfc-monitoring
sudo systemctl start mfc-monitoring
```

### Method 3: Docker Deployment

**`Dockerfile`**:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    certbot \
    openssl \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application
COPY . .

# Install Python dependencies
RUN pip install -r requirements.txt

# Create SSL directories
RUN mkdir -p /etc/ssl/certs /etc/ssl/private /etc/mfc

# Expose ports
EXPOSE 8443 8444 8445

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -k https://localhost:8443/health || exit 1

# Start services
CMD ["python", "start_monitoring.py", "--daemon"]
```

**`docker-compose.yml`**:

```yaml
version: '3.8'

services:
  mfc-monitoring:
    build: .
    ports:
      - "8443:8443"
      - "8444:8444"
      - "8445:8445"
      - "80:80"  # For Let's Encrypt
    environment:
      - MFC_SSL_DOMAIN=${DOMAIN}
      - MFC_SSL_EMAIL=${EMAIL}
      - MFC_SSL_USE_LETSENCRYPT=true
    volumes:
      - ./ssl-config:/etc/mfc
      - ./certs:/etc/letsencrypt
      - ./data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-k", "https://localhost:8443/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

Deploy with Docker:

```bash
# Set environment variables
export DOMAIN=your-domain.com
export EMAIL=admin@your-domain.com

# Start services
docker-compose up -d

# Check logs
docker-compose logs -f
```

## Monitoring and Maintenance

### Certificate Monitoring

```bash
# Check certificate status
python cert_manager.py --monitor

# Setup email notifications
python cert_manager.py --setup-notifications

# Test notifications
python cert_manager.py --test-notifications
```

### Service Health Monitoring

```bash
# Check service status
python start_monitoring.py --status

# View logs
tail -f /tmp/mfc-monitoring.log

# Test SSL connections
python start_monitoring.py --test-ssl
```

### Automatic Certificate Renewal

The system automatically sets up certificate renewal via cron:

```bash
# View cron jobs
crontab -l

# Manual renewal test
python cert_manager.py --renew-if-needed

# Force renewal
python cert_manager.py --renew
```

### Backup and Recovery

```bash
# Backup SSL certificates
sudo tar -czf mfc-ssl-backup-$(date +%Y%m%d).tar.gz \
    /etc/letsencrypt \
    /etc/mfc \
    /etc/ssl/certs/mfc-monitoring.crt \
    /etc/ssl/private/mfc-monitoring.key

# Backup configuration
cp /etc/mfc/ssl-config.json ssl-config-backup.json

# Recovery
sudo tar -xzf mfc-ssl-backup-YYYYMMDD.tar.gz -C /
```

## Troubleshooting

### Common Issues

#### 1. Certificate Generation Failed

```bash
# Check DNS resolution
nslookup your-domain.com

# Check port 80 availability
sudo netstat -tlnp | grep :80

# Try staging environment first
python cert_manager.py --init --domain your-domain.com --staging

# Check logs
sudo tail -f /var/log/letsencrypt/letsencrypt.log
```

#### 2. Services Won't Start

```bash
# Check port availability
sudo netstat -tlnp | grep -E ':(8443|8444|8445)'

# Check file permissions
ls -la /etc/ssl/private/mfc-monitoring.key

# Start services individually
python dashboard_api.py --debug
python dashboard_frontend.py run_https 8444
python realtime_streamer.py --host 0.0.0.0 --port 8445
```

#### 3. SSL Connection Errors

```bash
# Test SSL configuration
openssl s_client -connect your-domain.com:8443 -servername your-domain.com

# Check certificate validity
openssl x509 -in /etc/ssl/certs/mfc-monitoring.crt -text -noout

# Verify certificate chain
openssl verify -CAfile /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/mfc-monitoring.crt
```

#### 4. Browser Security Warnings

- **Self-signed certificates**: Expected behavior, add security exception
- **Wrong domain**: Regenerate certificate with correct domain
- **Expired certificate**: Renew certificate
- **Mixed content**: Ensure all resources load over HTTPS

### Debug Mode

Enable debug logging:

```bash
export MFC_DEBUG=true
python start_monitoring.py --debug
```

### Performance Issues

```bash
# Check system resources
htop
df -h
netstat -i

# Monitor SSL handshake performance
curl -w "@curl-format.txt" -o /dev/null -s https://your-domain.com:8443/health
```

## Security Best Practices

### 1. Certificate Security

- Use strong private keys (2048-bit RSA minimum)
- Protect private key files (600 permissions)
- Regular certificate rotation
- Monitor certificate expiration

### 2. Network Security

- Use firewall to restrict access
- Implement IP whitelisting where appropriate
- Regular security updates
- Monitor access logs

### 3. Application Security

- Change default API tokens
- Use strong session secrets
- Enable all security headers
- Regular dependency updates

### 4. Operational Security

- Regular backups
- Log monitoring
- Incident response plan
- Security audits

## Configuration Examples

### Development Environment

```json
{
  "domain": "localhost",
  "use_letsencrypt": false,
  "staging": false,
  "cert_file": "/etc/ssl/certs/mfc-monitoring.crt",
  "key_file": "/etc/ssl/private/mfc-monitoring.key",
  "https_port_api": 8443,
  "https_port_frontend": 8444,
  "wss_port_streaming": 8445,
  "enable_hsts": false,
  "enable_csp": true,
  "auto_renew": false
}
```

### Production Environment

```json
{
  "domain": "mfc-monitoring.company.com",
  "email": "admin@company.com",
  "use_letsencrypt": true,
  "staging": false,
  "cert_file": "/etc/letsencrypt/live/mfc-monitoring.company.com/fullchain.pem",
  "key_file": "/etc/letsencrypt/live/mfc-monitoring.company.com/privkey.pem",
  "https_port_api": 8443,
  "https_port_frontend": 8444,
  "wss_port_streaming": 8445,
  "enable_hsts": true,
  "enable_csp": true,
  "auto_renew": true,
  "renewal_days_before": 30
}
```

## Support and Maintenance

### Regular Maintenance Tasks

- **Daily**: Monitor service health
- **Weekly**: Check certificate expiration
- **Monthly**: Review security logs
- **Quarterly**: Update dependencies
- **Annually**: Security audit

### Getting Help

1. Check logs: `/tmp/mfc-monitoring.log`
2. Run diagnostics: `python start_monitoring.py --test-ssl`
3. Review this documentation
4. Check project issues on GitLab

### Updates and Upgrades

```bash
# Update application
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart services
python start_monitoring.py --stop
python start_monitoring.py
```

---

**Document Version**: 1.2.0  
**Last Updated**: 2025-07-31  
**Next Review**: 2025-10-31

For additional support or questions about HTTPS deployment, please refer to the project documentation or create an issue in the GitLab repository.