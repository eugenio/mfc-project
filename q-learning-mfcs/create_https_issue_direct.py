"""
Direct GitLab API call to create HTTPS/SSL enhancement issue
"""
import sys
import os

# Add tests directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests'))

from gitlab_issue_manager import GitLabIssueManager, IssueData, IssueType, IssueSeverity, IssueUrgency

def create_mypy_issue():
    """Create the mypy type checking issue"""
    
    # Create issue data
    issue_data = IssueData(
        title="🔧 TYPE CHECKING: Fix mypy type annotation errors in stability analysis system",
        description="""## Type Checking Issue Summary

MyPy analysis of the stability analysis system revealed **175 type errors** across 6 files that need to be addressed for production-quality code.

### Error Categories:

#### 1. Missing Dataclass Definitions (Major Impact)
**Files Affected**: All stability modules
**Problem**: Custom dataclasses not properly defined, causing mypy to treat them as builtin types
- StabilityMetrics - 17 field errors
- ReliabilityPrediction - 8 field errors  
- TimeSeriesMetrics - 10 field errors
- ComponentReliability - Multiple attribute errors
- OptimizationResult - 4 field errors

#### 2. Missing Type Annotations (Moderate Impact)  
**Count**: ~15 variables need explicit type annotations

#### 3. Type Compatibility Issues (Low-Medium Impact)
**Count**: ~10 return type mismatches

#### 4. Collection Type Issues (Low Impact)
**Count**: ~8 attribute access errors

### Files Requiring Fixes:

1. **src/stability/stability_framework.py** - 21 errors
2. **src/stability/reliability_analyzer.py** - 30 errors  
3. **src/stability/degradation_detector.py** - 45 errors
4. **src/stability/maintenance_scheduler.py** - 25 errors
5. **src/stability/data_manager.py** - 20 errors
6. **src/stability/stability_visualizer.py** - 34 errors

### Impact Assessment:
- **Severity**: Medium (code works but lacks type safety)
- **Priority**: Medium (improvement, not breaking) 
- **Effort**: 2-3 days for complete resolution
- **Risk**: Low (isolated to type system)

### Success Criteria:
- MyPy passes with 0 errors on src/stability/
- All dataclasses properly defined with type hints
- No regression in functionality

### MyPy Command Used:
```bash
mypy src/stability/ tests/test_stability_system.py --ignore-missing-imports --show-error-codes
```

### Code Quality Impact:  
- **Before**: 175 mypy errors
- **Target**: 0 mypy errors
- **Benefit**: Better IDE support, early bug detection, improved maintainability""",
        severity=IssueSeverity.MEDIUM,
        urgency=IssueUrgency.MEDIUM,
        issue_type=IssueType.ENHANCEMENT,
        labels=["type-checking", "code-quality", "mypy", "enhancement", "stability", "maintenance"],
        component="stability"
    )
    
    try:
        # Create issue manager
        issue_manager = GitLabIssueManager()
        
        # Create the issue
        print("🚀 Creating GitLab issue for mypy type checking errors...")
        created_issue = issue_manager.create_issue(issue_data)
        
        print(f"\n✅ Successfully created GitLab issue:")
        print(f"   📝 Title: {created_issue['title']}")
        print(f"   🔗 URL: {created_issue['web_url']}")
        print(f"   🆔 Issue ID: #{created_issue['iid']}")
        print(f"   🏷️  Labels: {', '.join(created_issue['labels'])}")
        print(f"   📊 State: {created_issue['state']}")
        
        print(f"\n📋 MyPy type checking issue now tracked in GitLab")
        print(f"🎯 175 type errors identified across 6 stability analysis files")
        
        return created_issue
        
    except Exception as e:
        print(f"❌ Error creating GitLab issue: {e}")
        return None

def create_https_ssl_issue():
    """Create the HTTPS/SSL enhancement issue"""
    
    # Create issue data
    issue_data = IssueData(
        title="HTTPS/SSL Support for Real-time Monitoring System",
        description="""## Enhancement Request: HTTPS/SSL Support for Monitoring Dashboard

### Overview
Implement HTTPS support with Let's Encrypt certificates for the real-time monitoring dashboard system to enable secure production deployments.

### Current Status
The monitoring system currently runs on HTTP (ports 8000, 8501, 8001) which is not suitable for production environments due to security concerns.

### Proposed Solution
Implement SSL/TLS encryption using Let's Encrypt certificates for:

#### 1. FastAPI Dashboard API (Port 8000 → 8443)
- Configure uvicorn with SSL certificates
- Set up automatic certificate renewal
- Update CORS and security headers
- Implement secure session management

#### 2. Streamlit Frontend (Port 8501 → 8444)
- Configure Streamlit for HTTPS
- Update API client to use HTTPS endpoints
- Handle SSL certificate validation

#### 3. WebSocket Streaming (Port 8001 → 8445)
- Upgrade WebSocket connections to WSS (WebSocket Secure)
- Update client connection strings
- Implement secure authentication

#### 4. Certificate Management
- Automated Let's Encrypt certificate provisioning
- Certificate renewal automation (cron job/systemd timer)
- Certificate validation and monitoring
- Backup and recovery procedures

### Implementation Tasks
- [ ] Install and configure certbot for Let's Encrypt
- [ ] Create SSL certificate provisioning script
- [ ] Update FastAPI server configuration for HTTPS
- [ ] Modify Streamlit configuration for SSL
- [ ] Update WebSocket server for WSS support
- [ ] Update all client-side connections to use HTTPS/WSS
- [ ] Implement certificate auto-renewal
- [ ] Add SSL certificate monitoring to dashboard
- [ ] Update deployment documentation
- [ ] Create production deployment guide
- [ ] Add SSL/TLS tests to test suite
- [ ] Update startup scripts for HTTPS mode

### Security Considerations
- Use strong SSL/TLS configurations (TLS 1.2+)
- Implement proper certificate validation
- Add security headers (HSTS, CSP, etc.)
- Set up secure cookie configurations
- Implement rate limiting for HTTPS endpoints

### Dependencies
- `certbot` (Let's Encrypt client)
- `cryptography` (Python SSL library)
- `uvicorn[standard]` (with SSL support)
- Potentially `nginx` or `caddy` for reverse proxy

### Expected Benefits
- Production-ready secure deployment
- Data encryption in transit
- Authentication security improvements
- Compliance with security best practices
- Trust indicators for end users

### Priority: High
This enhancement is essential for production deployment of the monitoring system in secure environments.

### Estimated Effort
- Development: 2-3 days
- Testing: 1 day
- Documentation: 1 day
- Total: 4-5 days

### Related Components
- `q-learning-mfcs/src/monitoring/dashboard_api.py`
- `q-learning-mfcs/src/monitoring/dashboard_frontend.py`
- `q-learning-mfcs/src/monitoring/realtime_streamer.py`
- `q-learning-mfcs/src/monitoring/start_monitoring.py`""",
        severity=IssueSeverity.HIGH,
        urgency=IssueUrgency.HIGH,
        issue_type=IssueType.ENHANCEMENT,
        labels=["enhancement", "monitoring", "security", "production", "https", "ssl", "deployment"],
        component="monitoring"
    )
    
    try:
        # Create issue manager
        issue_manager = GitLabIssueManager()
        
        # Create the issue
        print("🚀 Creating GitLab enhancement issue for HTTPS/SSL monitoring system...")
        created_issue = issue_manager.create_issue(issue_data)
        
        print(f"\n✅ Successfully created GitLab issue:")
        print(f"   📝 Title: {created_issue['title']}")
        print(f"   🔗 URL: {created_issue['web_url']}")
        print(f"   🆔 Issue ID: #{created_issue['iid']}")
        print(f"   🏷️  Labels: {', '.join(created_issue['labels'])}")
        print(f"   📊 State: {created_issue['state']}")
        
        print(f"\n📋 Todo item now tracked in GitLab issue #{created_issue['iid']}")
        print(f"🎯 The HTTPS/SSL enhancement can be implemented when ready for production deployment")
        
        return created_issue
        
    except Exception as e:
        print(f"❌ Error creating GitLab issue: {e}")
        
        # Check if it's a configuration issue
        if "GitLab token" in str(e) or "project ID" in str(e):
            print("\n💡 GitLab configuration may be missing:")
            print("   - Make sure GITLAB_TOKEN environment variable is set")
            print("   - Make sure GITLAB_PROJECT_ID environment variable is set")
        
        return None

if __name__ == "__main__":
    result = create_https_ssl_issue()
    
    if result:
        print(f"\n🎯 Next Steps:")
        print("1. GitLab issue created and ready for implementation")
        print("2. Issue will be referenced in commits and pull requests")
        print("3. Progress can be tracked through GitLab")
        print("4. Todo item is now properly documented")
    else:
        print(f"\n⚠️  Issue creation failed, but the requirement is documented")
        print("1. The HTTPS/SSL enhancement is still needed for production")
        print("2. Implementation details have been prepared")
        print("3. Manual issue creation may be required")