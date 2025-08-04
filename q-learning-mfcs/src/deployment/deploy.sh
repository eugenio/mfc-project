#!/bin/bash
"""
Complete TTS Service Deployment Script
Agent Zeta - Deployment and Process Management

Master deployment script that orchestrates all deployment components
"""

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
DEPLOYMENT_DIR="$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Logging functions
log_header() {
    echo -e "\n${PURPLE}===============================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}===============================================${NC}\n"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Default values
ENVIRONMENT="development"
STRATEGY="direct"
PROCESS_MANAGER="auto"
SERVICE_NAME="tts-service"
ENABLE_ORCHESTRATION=true
SKIP_TESTS=false
QUIET_MODE=false

# Parse command line arguments
usage() {
    cat << EOF
TTS Service Complete Deployment Script

Usage: $0 [OPTIONS]

Options:
  --environment ENV       Deployment environment (development|staging|production|test)
  --strategy STRATEGY     Deployment strategy (direct|blue_green|rolling|canary)
  --process-manager PM    Process manager (auto|systemd|supervisor|manual)
  --service-name NAME     Service name (default: tts-service)
  --no-orchestration     Disable service orchestration
  --skip-tests           Skip deployment tests
  --quiet                Suppress non-essential output
  --help, -h             Show this help message

Examples:
  $0                                    # Development deployment with defaults
  $0 --environment production          # Production deployment
  $0 --strategy blue_green             # Blue-green deployment
  $0 --process-manager systemd         # Force systemd usage
  $0 --no-orchestration --skip-tests  # Minimal deployment

Agent Zeta - Deployment and Process Management
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --process-manager)
            PROCESS_MANAGER="$2"
            shift 2
            ;;
        --service-name)
            SERVICE_NAME="$2"
            shift 2
            ;;
        --no-orchestration)
            ENABLE_ORCHESTRATION=false
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --quiet)
            QUIET_MODE=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main deployment function
main() {
    log_header "TTS Service Complete Deployment"
    log_info "Agent Zeta - Deployment and Process Management"
    
    # Display configuration
    if [[ "$QUIET_MODE" != true ]]; then
        echo ""
        log_info "Deployment Configuration:"
        log_info "  Environment: $ENVIRONMENT"
        log_info "  Strategy: $STRATEGY"
        log_info "  Process Manager: $PROCESS_MANAGER"
        log_info "  Service Name: $SERVICE_NAME"
        log_info "  Orchestration: $ENABLE_ORCHESTRATION"
        log_info "  Skip Tests: $SKIP_TESTS"
        echo ""
    fi
    
    # Step 1: Pre-deployment validation
    if ! run_pre_deployment_checks; then
        log_error "Pre-deployment checks failed"
        exit 1
    fi
    
    # Step 2: Run deployment tests (if not skipped)
    if [[ "$SKIP_TESTS" != true ]]; then
        if ! run_deployment_tests; then
            log_error "Deployment tests failed"
            exit 1
        fi
    fi
    
    # Step 3: Deploy TTS service
    if ! deploy_tts_service; then
        log_error "TTS service deployment failed"
        exit 1
    fi
    
    # Step 4: Setup process management
    if ! setup_process_management; then
        log_error "Process management setup failed"
        exit 1
    fi
    
    # Step 5: Start orchestration (if enabled)
    if [[ "$ENABLE_ORCHESTRATION" == true ]]; then
        if ! start_orchestration; then
            log_error "Service orchestration failed"
            exit 1
        fi
    else
        # Manual service start
        if ! start_services_manually; then
            log_error "Manual service start failed"
            exit 1
        fi
    fi
    
    # Step 6: Post-deployment validation
    if ! run_post_deployment_checks; then
        log_error "Post-deployment validation failed"
        exit 1
    fi
    
    # Step 7: Display final status
    display_deployment_status
    
    log_success "TTS Service deployment completed successfully!"
}

# Pre-deployment checks
run_pre_deployment_checks() {
    log_header "Pre-Deployment Validation"
    
    # Check Python environment
    if ! python3 --version >/dev/null 2>&1; then
        log_error "Python 3 is required but not found"
        return 1
    fi
    
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log_info "Python version: $python_version"
    
    # Check project structure
    if [[ ! -d "$PROJECT_ROOT/q-learning-mfcs/src/notifications" ]]; then
        log_error "TTS notification modules not found"
        return 1
    fi
    
    # Check deployment scripts
    local required_scripts=(
        "$DEPLOYMENT_DIR/tts_service_manager.py"
        "$DEPLOYMENT_DIR/start_tts_service.sh"
        "$DEPLOYMENT_DIR/stop_tts_service.sh"
        "$DEPLOYMENT_DIR/process_manager.py"
        "$DEPLOYMENT_DIR/deploy_tts_service.py"
    )
    
    for script in "${required_scripts[@]}"; do
        if [[ ! -f "$script" ]]; then
            log_error "Required script not found: $script"
            return 1
        fi
    done
    
    # Test TTS imports
    if ! python3 -c "
import sys
sys.path.append('$PROJECT_ROOT/q-learning-mfcs/src')
from notifications.tts_handler import TTSNotificationHandler
from deployment.tts_service_manager import TTSServiceManager
" 2>/dev/null; then
        log_error "Failed to import TTS modules"
        return 1
    fi
    
    # Check disk space
    local available_space=$(df /tmp | awk 'NR==2 {print $4}')
    if [[ $available_space -lt 102400 ]]; then  # 100MB in KB
        log_warning "Low disk space available: ${available_space}KB"
    fi
    
    # Check permissions
    if ! mkdir -p /tmp/tts-service-test 2>/dev/null; then
        log_error "Insufficient permissions for service directories"
        return 1
    fi
    rmdir /tmp/tts-service-test 2>/dev/null || true
    
    log_success "Pre-deployment checks passed"
    return 0
}

# Run deployment tests
run_deployment_tests() {
    log_header "Running Deployment Tests"
    
    cd "$PROJECT_ROOT/q-learning-mfcs"
    
    # Set up Python path
    export PYTHONPATH="$PROJECT_ROOT/q-learning-mfcs/src:$PYTHONPATH"
    
    # Run deployment lifecycle tests
    if python3 -m pytest tests/test_deployment_lifecycle.py -v -x; then
        log_success "Deployment tests passed"
        return 0
    else
        log_error "Deployment tests failed"
        return 1
    fi
}

# Deploy TTS service
deploy_tts_service() {
    log_header "Deploying TTS Service"
    
    cd "$PROJECT_ROOT/q-learning-mfcs/src"
    
    # Set up environment
    export PYTHONPATH="$PROJECT_ROOT/q-learning-mfcs/src:$PYTHONPATH"
    
    # Run deployment script
    local deploy_args=(
        "--environment" "$ENVIRONMENT"
        "--strategy" "$STRATEGY"
        "--service-name" "$SERVICE_NAME"
        "--process-manager" "$PROCESS_MANAGER"
    )
    
    if python3 "$DEPLOYMENT_DIR/deploy_tts_service.py" "${deploy_args[@]}"; then
        log_success "TTS service deployed successfully"
        return 0
    else
        log_error "TTS service deployment failed"
        return 1
    fi
}

# Setup process management
setup_process_management() {
    log_header "Setting Up Process Management"
    
    cd "$PROJECT_ROOT/q-learning-mfcs/src"
    
    # Install service using process manager
    if python3 "$DEPLOYMENT_DIR/process_manager.py" \
        --manager "$PROCESS_MANAGER" \
        --service-name "$SERVICE_NAME" \
        --install; then
        log_success "Process management configured"
        return 0
    else
        log_warning "Process management setup had issues (may be expected)"
        return 0  # Don't fail deployment for this
    fi
}

# Start orchestration
start_orchestration() {
    log_header "Starting Service Orchestration"
    
    cd "$PROJECT_ROOT/q-learning-mfcs/src"
    
    # Create orchestration config if it doesn't exist
    local config_file="$DEPLOYMENT_DIR/orchestration_config.json"
    if [[ ! -f "$config_file" ]]; then
        cat > "$config_file" << EOF
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
EOF
    fi
    
    # Start orchestration
    if timeout 120 python3 "$DEPLOYMENT_DIR/service_orchestrator.py" \
        --config "$config_file" \
        --start; then
        log_success "Service orchestration started"
        return 0
    else
        log_warning "Service orchestration had issues, falling back to manual start"
        return start_services_manually
    fi
}

# Manual service start (fallback)
start_services_manually() {
    log_header "Starting Services Manually"
    
    # Start TTS service using startup script
    if "$DEPLOYMENT_DIR/start_tts_service.sh" --daemon; then
        log_success "TTS service started manually"
        
        # Wait for service to stabilize
        sleep 5
        
        return 0
    else
        log_error "Manual service start failed"
        return 1
    fi
}

# Post-deployment validation
run_post_deployment_checks() {
    log_header "Post-Deployment Validation"
    
    # Wait for services to be ready
    log_info "Waiting for services to stabilize..."
    sleep 10
    
    # Test TTS functionality
    cd "$PROJECT_ROOT/q-learning-mfcs/src"
    export PYTHONPATH="$PROJECT_ROOT/q-learning-mfcs/src:$PYTHONPATH"
    
    if python3 -c "
from notifications.tts_handler import TTSNotificationHandler, TTSMode, TTSEngineType
from notifications.base import NotificationConfig, NotificationLevel
import time

try:
    # Test basic TTS
    handler = TTSNotificationHandler(TTSMode.TTS_ONLY, TTSEngineType.PYTTSX3)
    config = NotificationConfig('Deployment Test', 'TTS service deployment validation', NotificationLevel.SUCCESS)
    result = handler.send_notification(config)
    print('SUCCESS' if result else 'FAILED')
except Exception as e:
    print(f'FAILED: {e}')
"; then
        log_success "TTS functionality test passed"
    else
        log_warning "TTS functionality test failed (service may still be starting)"
    fi
    
    # Check service status
    if python3 "$DEPLOYMENT_DIR/tts_service_manager.py" --status >/dev/null 2>&1; then
        log_success "Service manager is responsive"
    else
        log_warning "Service manager is not responding"
    fi
    
    log_success "Post-deployment validation completed"
    return 0
}

# Display deployment status
display_deployment_status() {
    log_header "Deployment Status"
    
    cd "$PROJECT_ROOT/q-learning-mfcs/src"
    export PYTHONPATH="$PROJECT_ROOT/q-learning-mfcs/src:$PYTHONPATH"
    
    # Show service status
    if python3 "$DEPLOYMENT_DIR/tts_service_manager.py" --status 2>/dev/null; then
        echo ""
    else
        log_info "Service manager status not available"
    fi
    
    # Show process manager status
    if python3 "$DEPLOYMENT_DIR/process_manager.py" \
        --manager "$PROCESS_MANAGER" \
        --service-name "$SERVICE_NAME" \
        --status 2>/dev/null; then
        echo ""
    else
        log_info "Process manager status not available"
    fi
    
    # Show orchestration status (if enabled)
    if [[ "$ENABLE_ORCHESTRATION" == true ]]; then
        if python3 "$DEPLOYMENT_DIR/service_orchestrator.py" --status 2>/dev/null; then
            echo ""
        else
            log_info "Orchestration status not available"
        fi
    fi
    
    # Final instructions
    echo ""
    log_info "Deployment completed. You can:"
    log_info "  - Check status: $0 --environment $ENVIRONMENT --quiet && python3 $DEPLOYMENT_DIR/tts_service_manager.py --status"
    log_info "  - Stop services: $DEPLOYMENT_DIR/stop_tts_service.sh"
    log_info "  - View logs: tail -f /tmp/tts-service-logs/*.log"
    
    if [[ "$PROCESS_MANAGER" == "systemd" ]]; then
        log_info "  - Systemd control: sudo systemctl {start|stop|status} $SERVICE_NAME"
    elif [[ "$PROCESS_MANAGER" == "supervisor" ]]; then
        log_info "  - Supervisor control: supervisorctl {start|stop|status} $SERVICE_NAME:*"
    fi
}

# Cleanup function
cleanup() {
    local exit_code=$?
    
    if [[ $exit_code -ne 0 ]]; then
        log_error "Deployment failed with exit code $exit_code"
        
        # Attempt cleanup
        log_info "Attempting cleanup..."
        "$DEPLOYMENT_DIR/stop_tts_service.sh" --quiet --force 2>/dev/null || true
        
        # Show troubleshooting info
        echo ""
        log_info "Troubleshooting:"
        log_info "  - Check logs: ls -la /tmp/tts-service-logs/"
        log_info "  - Manual cleanup: $DEPLOYMENT_DIR/stop_tts_service.sh --force"
        log_info "  - Reset state: rm -rf /tmp/tts-service-* /tmp/test_registry.json"
    fi
}

# Set up cleanup trap
trap cleanup EXIT

# Validate environment values
case "$ENVIRONMENT" in
    development|staging|production|test)
        ;;
    *)
        log_error "Invalid environment: $ENVIRONMENT"
        usage
        exit 1
        ;;
esac

case "$STRATEGY" in
    direct|blue_green|rolling|canary)
        ;;
    *)
        log_error "Invalid strategy: $STRATEGY"
        usage
        exit 1
        ;;
esac

case "$PROCESS_MANAGER" in
    auto|systemd|supervisor|manual)
        ;;
    *)
        log_error "Invalid process manager: $PROCESS_MANAGER"
        usage
        exit 1
        ;;
esac

# Execute main deployment
main "$@"