"""
Create GitLab issues for test failures found in comprehensive test run.
"""
import sys
sys.path.append('/home/uge/mfc-project/q-learning-mfcs/tests')

from gitlab_issue_manager import GitLabIssueManager, IssueData, IssueType, IssueSeverity, IssueUrgency
def create_issues():
    """Create issues for all major test failures."""
    
    manager = GitLabIssueManager()
    
    issues = [
        # Issue 1: Missing get_state_hash method - CRITICAL/URGENT
        IssueData(
            title="CRITICAL: Missing get_state_hash method in AdvancedQLearningFlowController",
            description="""## Problem
Multiple integration tests are failing due to missing `get_state_hash` method in `AdvancedQLearningFlowController` class.

## Error Details
```
AttributeError: 'AdvancedQLearningFlowController' object has no attribute 'get_state_hash'
```

## Affected Files
- `src/integrated_mfc_model.py:299`
- Multiple test files in integration and performance suites

## Failing Tests
- `test_integrated_model.py`: All integration tests (8 failures)
- `test_performance_stress.py`: Performance scaling tests (4 failures)  
- `test_biofilm_model.py`: Several biofilm dynamics tests (3 failures)

## Impact
- **22+ test failures**
- Blocks integration testing completely
- Prevents performance benchmarking
- Core simulation functionality broken

## Suggested Fix
Implement missing `get_state_hash` method in the flow controller or update the calling code to use the correct method name.

## Priority Justification
- **Severity: CRITICAL** - Core functionality broken, blocks development
- **Urgency: URGENT** - Affects multiple test suites, immediate fix needed""",
            severity=IssueSeverity.CRITICAL,
            urgency=IssueUrgency.URGENT,
            issue_type=IssueType.BUG,
            labels=['bug', 'critical', 'urgent', 'test-failure', 'integration'],
            component='flow_controller',
            test_case='test_integrated_model.py, test_performance_stress.py',
            error_message="AttributeError: 'AdvancedQLearningFlowController' object has no attribute 'get_state_hash'"
        ),
        
        # Issue 2: Config serialization failures - HIGH/HIGH
        IssueData(
            title="HIGH: Config serialization/deserialization failures",
            description="""## Problem
Configuration I/O system failing for dataclass serialization with computed fields.

## Error Details
```
TypeError: QLearningConfig.__init__() got an unexpected keyword argument 'total_anode_area'
yaml.constructor.ConstructorError: could not determine a constructor for the tag 'tag:yaml.org,2002:python/tuple'
```

## Affected Files
- `src/config/config_io.py`
- `tests/config/test_config_io.py`

## Failing Tests
- 9 out of 15 config I/O tests failing
- YAML and JSON serialization broken
- Config merging functionality broken

## Impact
- Configuration management broken
- Cannot save/load simulation parameters
- Blocks parameter sweeps and optimization

## Root Cause
Computed fields (`total_anode_area`, etc.) being serialized but not accepted in constructor.

## Priority Justification
- **Severity: HIGH** - Core configuration system broken
- **Urgency: HIGH** - Blocks parameter management and reproducibility""",
            severity=IssueSeverity.HIGH,
            urgency=IssueUrgency.HIGH,
            issue_type=IssueType.BUG,
            labels=['bug', 'high', 'config', 'serialization'],
            component='config_system',
            test_case='tests/config/test_config_io.py',
            error_message="TypeError: QLearningConfig.__init__() got an unexpected keyword argument 'total_anode_area'"
        ),
        
        # Issue 3: Monitoring system import errors - MEDIUM/HIGH
        IssueData(
            title="MEDIUM: Monitoring system import path and class definition errors",
            description="""## Problem
Monitoring system tests failing due to import path issues and missing class definitions.

## Error Details
```
ImportError: cannot import name 'RealTimeStreamer' from 'src.monitoring.realtime_streamer'
ImportError: cannot import name 'StreamEventType' from 'src.monitoring.realtime_streamer'
```

## Affected Files
- `tests/test_monitoring_system.py`
- `src/monitoring/realtime_streamer.py`

## Failing Tests
- 19 monitoring system test errors
- All real-time streaming tests affected
- Safety monitoring integration broken

## Impact
- Real-time monitoring system broken
- Safety alerts not functional
- Dashboard integration compromised

## Priority Justification
- **Severity: MEDIUM** - Feature-specific, doesn't break core simulation
- **Urgency: HIGH** - Safety monitoring is critical for operation""",
            severity=IssueSeverity.MEDIUM,
            urgency=IssueUrgency.HIGH,
            issue_type=IssueType.BUG,
            labels=['bug', 'medium', 'monitoring', 'import-error'],
            component='monitoring_system',
            test_case='tests/test_monitoring_system.py',
            error_message="ImportError: cannot import name 'RealTimeStreamer'"
        ),
        
        # Issue 4: GPU acceleration compatibility issues - MEDIUM/MEDIUM
        IssueData(
            title="MEDIUM: GPU acceleration compatibility issues with JAX/ROCm",
            description="""## Problem
GPU acceleration tests failing due to JAX version incompatibility and ROCm configuration issues.

## Error Details
```
jaxlib version 0.6.2 is newer than and incompatible with jax version 0.4.31
partially initialized module 'jax' has no attribute 'version'
```

## Affected Tests
- GPU acceleration performance tests
- ROCm backend detection working but JAX integration broken
- Performance benchmarks showing poor GPU speedup (0.09x instead of expected >0.5x)

## Impact
- GPU acceleration non-functional
- Performance benefits not realized
- AMD ROCm users affected

## Environment Details
- Device: Radeon RX 7900 XTX detected
- ROCm backend available
- JAX version conflict preventing GPU usage

## Priority Justification
- **Severity: MEDIUM** - Performance feature, doesn't break core functionality
- **Urgency: MEDIUM** - Important for computational efficiency""",
            severity=IssueSeverity.MEDIUM,
            urgency=IssueUrgency.MEDIUM,
            issue_type=IssueType.PERFORMANCE,
            labels=['performance', 'gpu', 'jax', 'rocm'],
            component='gpu_acceleration',
            test_case='tests/test_performance_stress.py::test_gpu_acceleration_performance',
            error_message="jaxlib version 0.6.2 is newer than and incompatible with jax version 0.4.31"
        ),
        
        # Issue 5: Biofilm model integration failures - MEDIUM/MEDIUM  
        IssueData(
            title="MEDIUM: Biofilm kinetics model integration and initialization failures",
            description="""## Problem
Several biofilm kinetics tests failing due to model initialization and integration issues.

## Failing Tests
- `test_biofilm_dynamics_step`
- `test_environmental_condition_updates` 
- `test_model_initialization`
- `test_nernst_monod_growth_rate`
- `test_theoretical_maximum_current`

## Common Issues
- Model initialization problems
- GPU acceleration availability checks failing
- Environmental condition update mechanisms broken
- Growth rate calculations incorrect

## Impact
- Biofilm modeling accuracy compromised
- Environmental adaptation not working
- Theoretical predictions unreliable

## Priority Justification
- **Severity: MEDIUM** - Affects model accuracy but simulation still runs
- **Urgency: MEDIUM** - Important for scientific validity""",
            severity=IssueSeverity.MEDIUM,
            urgency=IssueUrgency.MEDIUM,
            issue_type=IssueType.BUG,
            labels=['bug', 'medium', 'biofilm', 'modeling'],
            component='biofilm_kinetics',
            test_case='tests/biofilm_kinetics/test_biofilm_model.py',
            error_message="Multiple biofilm model test failures"
        )
    ]
    
    created_issues = []
    for issue in issues:
        try:
            result = manager.create_issue(issue)
            created_issues.append(result)
            print(f"✅ Created issue: {issue.title}")
        except Exception as e:
            print(f"❌ Failed to create issue '{issue.title}': {e}")
    
    return created_issues

if __name__ == "__main__":
    created = create_issues()
    print(f"\n📋 Summary: Created {len(created)} GitLab issues for test failures")