# TTS Service Deployment System

**Agent Zeta - Deployment and Process Management**

This directory contains a comprehensive deployment and process management system for the TTS (Text-to-Speech) service, designed to ensure reliable deployment, service lifecycle management, and integration with the main MFC project.

## 🎯 Overview

The deployment system provides:

- **Robust Service Management**: Complete lifecycle management with startup, shutdown, restart, and health monitoring
- **Cross-Platform Process Management**: Support for systemd, supervisor, and manual process management
- **Service Discovery**: Automatic service registration and discovery mechanism
- **Deployment Automation**: Multiple deployment strategies with environment detection
- **Log Management**: Comprehensive logging with rotation, compression, and monitoring
- **Service Orchestration**: Dependency management and coordinated startup/shutdown
- **Comprehensive Testing**: Full test suite for all deployment components

## 📁 File Structure

```
deployment/
├── README.md                      # This documentation
├── deploy.sh                      # Master deployment script
├── tts_service_manager.py         # Core service management
├── start_tts_service.sh          # Service startup script
├── stop_tts_service.sh           # Service shutdown script
├── process_manager.py            # Multi-platform process management
├── service_discovery.py          # Service discovery and registration
├── deploy_tts_service.py         # Automated deployment with strategies
├── log_management.py             # Advanced log management
├── service_orchestrator.py       # Service orchestration and dependencies
├── systemd/
│   └── tts-service.service       # Systemd service configuration
└── supervisor/
    └── tts-service.conf          # Supervisor service configuration
```

## 🚀 Quick Start

### 1. Basic Deployment

```bash
# Development deployment (default)
./deploy.sh

# Production deployment
./deploy.sh --environment production

# Blue-green deployment
./deploy.sh --strategy blue_green --environment staging
```

### 2. Manual Service Management

```bash
# Start TTS service
./start_tts_service.sh

# Stop TTS service
./stop_tts_service.sh

# Check service status
python3 tts_service_manager.py --status
```

### 3. Process Management

```bash
# Install and start with systemd
python3 process_manager.py --manager systemd --install --start

# Use supervisor
python3 process_manager.py --manager supervisor --install --start

# Manual management
python3 process_manager.py --manager manual --start
```

## 🔧 Components

### Service Manager (`tts_service_manager.py`)

The core service management component that handles:

- Service configuration loading
- Process lifecycle management (start, stop, restart)
- Health monitoring and automatic recovery
- Service metrics collection
- Multiple service instance management

**Key Features:**
- Configurable timeout settings
- Automatic restart on failure
- Process supervision
- Resource monitoring
- Graceful shutdown handling

### Startup/Shutdown Scripts

**`start_tts_service.sh`:**
- Environment validation
- Dependency checking
- Service initialization
- Health verification
- Service registration

**`stop_tts_service.sh`:**
- Graceful shutdown
- Force termination fallback
- Resource cleanup
- Service deregistration
- Status verification

### Process Manager (`process_manager.py`)

Multi-platform process management supporting:

- **Systemd**: Linux systems with systemd
- **Supervisor**: Cross-platform supervision
- **Manual**: Direct process management

**Capabilities:**
- Automatic platform detection
- Service installation/uninstallation
- Enable/disable automatic startup
- Status monitoring
- Cross-platform compatibility

### Service Discovery (`service_discovery.py`)

Service registration and discovery system featuring:

- Service registry with persistence
- Automatic heartbeat monitoring
- Service health tracking
- Stale service cleanup
- Event-driven notifications

**Components:**
- `ServiceRegistry`: Central service registry
- `TTSServiceDiscovery`: TTS-specific discovery client
- `ServiceHealthMonitor`: Health monitoring daemon

### Deployment Automation (`deploy_tts_service.py`)

Automated deployment with multiple strategies:

**Deployment Strategies:**
- **Direct**: Immediate service replacement
- **Blue-Green**: Zero-downtime deployment
- **Rolling**: Gradual service updates
- **Canary**: Risk-mitigated deployments

**Features:**
- Pre-deployment validation
- Environment detection
- Backup and rollback
- Post-deployment verification
- Deployment metrics

### Log Management (`log_management.py`)

Advanced logging system with:

- Custom TTS log formatting
- Automatic log rotation
- Compression and archival
- Real-time monitoring
- Log statistics and export
- Alert thresholds

### Service Orchestrator (`service_orchestrator.py`)

Coordinates complex service interactions:

- Dependency management
- Phased startup/shutdown
- Health monitoring
- Integration coordination
- Resource orchestration

## 🛠️ Configuration

### Service Configuration (`tts_services.json`)

```json
{
  "services": [
    {
      "name": "tts-primary",
      "engine_type": "hybrid",
      "port": 9000,
      "host": "localhost",
      "log_level": "INFO",
      "max_restarts": 3,
      "restart_delay": 5,
      "health_check_interval": 30,
      "enable_coqui": true,
      "coqui_model": "tts_models/en/ljspeech/tacotron2-DDC",
      "log_file": "/tmp/tts-service-logs/tts-primary.log",
      "pid_file": "/tmp/tts-service-pids/tts-primary.pid"
    }
  ]
}
```

### Orchestration Configuration (`orchestration_config.json`)

```json
{
  "dependencies": [
    {
      "service_name": "monitoring-api",
      "dependency_type": "optional",
      "timeout": 15,
      "health_check_url": "http://localhost:8000/health"
    }
  ]
}
```

## 📊 Monitoring and Logging

### Service Status

```bash
# Detailed service status
python3 tts_service_manager.py --status

# Process manager status
python3 process_manager.py --status

# Orchestration status
python3 service_orchestrator.py --status
```

### Log Management

```bash
# View log statistics
python3 log_management.py --stats

# Export logs
python3 log_management.py --export logs.json --start-time "2024-01-01 00:00:00"

# Create log archive
python3 log_management.py --archive

# Start log monitoring
python3 log_management.py --monitor
```

### Service Discovery

```bash
# List discovered services
python3 service_discovery.py --list

# Clean up stale services
python3 service_discovery.py --cleanup

# Start health monitoring
python3 service_discovery.py --monitor
```

## 🧪 Testing

### Running Deployment Tests

```bash
# Full test suite
python3 -m pytest tests/test_deployment_lifecycle.py -v

# Specific test categories
python3 -m pytest tests/test_deployment_lifecycle.py::TestTTSServiceManager -v
python3 -m pytest tests/test_deployment_lifecycle.py::TestServiceDiscovery -v
python3 -m pytest tests/test_deployment_lifecycle.py::TestProcessManager -v
```

### Integration Testing

The test suite includes:

- **Unit Tests**: Individual component testing
- **Integration Tests**: Cross-component interactions
- **Error Handling**: Edge cases and failure scenarios
- **Resource Management**: Memory and file handle testing
- **Performance Testing**: Load and stress testing

## 🔍 Troubleshooting

### Common Issues

**Service Won't Start:**
```bash
# Check system requirements
./deploy.sh --environment development --skip-tests

# Validate environment
python3 deploy_tts_service.py --environment test

# Check logs
tail -f /tmp/tts-service-logs/*.log
```

**Permission Issues:**
```bash
# Fix permissions
sudo chown -R $USER:$USER /tmp/tts-service-*
chmod +x start_tts_service.sh stop_tts_service.sh
```

**Port Conflicts:**
```bash
# Check port usage
netstat -tuln | grep :9000

# Use different port
# Edit tts_services.json and change port number
```

**Dependency Issues:**
```bash
# Check Python imports
python3 -c "
import sys
sys.path.append('../')
from notifications.tts_handler import TTSNotificationHandler
print('Imports OK')
"

# Install missing dependencies
pixi install -e tts-dev  # For Coqui TTS support
```

### Debug Mode

```bash
# Enable debug logging
export TTS_LOG_LEVEL=DEBUG

# Run with verbose output
./start_tts_service.sh --skip-checks

# Monitor in real-time
python3 service_orchestrator.py --config orchestration_config.json --start
```

### Manual Cleanup

```bash
# Stop all services
./stop_tts_service.sh --force

# Clean up files
rm -rf /tmp/tts-service-* /tmp/test_registry.json

# Reset process managers
sudo systemctl stop tts-service 2>/dev/null || true
supervisorctl stop tts-service:* 2>/dev/null || true
```

## 🔒 Security Considerations

### File Permissions
- Service files: 755 (executable)
- Configuration files: 644 (read-write owner)
- Log files: 644 (read-write owner)
- PID files: 644 (read-write owner)

### Process Security
- Services run as non-root user when possible
- Systemd security features enabled
- Resource limits enforced
- Network access restricted

### Data Protection
- No sensitive data in configuration files
- Logs rotated and compressed
- Temporary files cleaned up
- Service registration uses non-sensitive identifiers

## 🚀 Production Deployment

### Recommended Production Setup

1. **Environment Preparation:**
   ```bash
   # Create service user
   sudo useradd -r -s /bin/false tts-service
   
   # Set up directories
   sudo mkdir -p /var/log/tts-service /var/run/tts-service
   sudo chown tts-service:tts-service /var/log/tts-service /var/run/tts-service
   ```

2. **Production Deployment:**
   ```bash
   ./deploy.sh --environment production --strategy blue_green --process-manager systemd
   ```

3. **Enable Auto-Start:**
   ```bash
   python3 process_manager.py --manager systemd --enable
   ```

4. **Monitor Health:**
   ```bash
   python3 service_orchestrator.py --daemon &
   ```

### Performance Tuning

- **Resource Limits**: Configure appropriate CPU and memory limits
- **Log Rotation**: Set up aggressive log rotation for high-volume environments
- **Health Checks**: Tune health check intervals based on load
- **Dependency Timeouts**: Adjust timeouts based on network conditions

## 📚 API Reference

### TTSServiceManager
```python
manager = TTSServiceManager(config_file="services.json")
manager.start_service("service-name")
manager.stop_service("service-name", force=False)
manager.restart_service("service-name")
status = manager.get_status_all()
```

### ServiceRegistry
```python
registry = ServiceRegistry("registry.json")
registry.register_service(service_info)
services = registry.find_services(service_type="text-to-speech")
registry.cleanup_stale_services()
```

### ProcessManager
```python
pm = ProcessManager(ProcessManagerType.AUTO)
pm.install_service("service-name")
pm.start_service("service-name")
status = pm.get_status("service-name")
```

## 🤝 Contributing

When contributing to the deployment system:

1. **Follow TDD**: Write tests before implementation
2. **Error Handling**: Implement comprehensive error handling
3. **Logging**: Add appropriate logging at all levels
4. **Documentation**: Update documentation for new features
5. **Testing**: Ensure all deployment scenarios are tested

### Development Workflow

```bash
# Set up development environment
pixi install -e dev

# Run tests during development
python3 -m pytest tests/test_deployment_lifecycle.py -v --tb=short

# Test specific components
python3 tts_service_manager.py --config test_config.json --status
```

## 📝 License

This deployment system is part of the MFC TTS Service project and follows the project's licensing terms.

---

**Agent Zeta - Deployment and Process Management**  
*Ensuring reliable TTS service deployment and lifecycle management*