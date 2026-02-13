# CI/CD Test Failures Summary

**Date:** 2026-01-25
**Pipeline:** #156
**Results:** 493 passed, 34 failed, 93 skipped, 7 errors

---

## Summary by Category

| Category | Count | Root Cause |
|----------|-------|------------|
| Hook/Guardian Tests | 22 | Missing `utils.enhanced_security_guardian` and `utils.git_guardian` modules |
| Integration Tests | 2 | `SessionManager` API changed (missing `get_session` method) |
| Edge Case Tests | 2 | Test assertions/data issues |
| Config Tests | 1 | Missing `biofilm_max_deviation` attribute |
| Output Tests | 1 | Empty figures directory (no PNG files) |
| GPU Tests | 6 | No GPU hardware available (expected skips treated as failures) |

---

## Detailed Failures

### 1. Hook/Guardian Tests (22 failures)

**Root Cause:** Missing modules - `utils.enhanced_security_guardian`, `utils.git_guardian`, `utils.constants`

#### test_enhanced_security_guardian.py (11 failures)
| Test | Error |
|------|-------|
| `TestSecurityIntegrationFunctions::test_secure_chunked_edit_failure` | `AttributeError: module 'utils' has no attribute 'enhanced_security_guardian'` |
| `TestSecurityIntegrationFunctions::test_secure_chunked_edit_success` | Same as above |
| `TestSecurityIntegrationFunctions::test_secure_chunked_file_creation_security_failure` | Same as above |
| `TestSecurityIntegrationFunctions::test_secure_chunked_file_creation_success` | Same as above |
| `TestMaliciousPatternDetection::test_data_exfiltration_detection` | `TypeError: 'NoneType' object is not callable` |
| `TestMaliciousPatternDetection::test_file_system_access_detection` | Same as above |
| `TestMaliciousPatternDetection::test_fragmentation_attack_detection` | Same as above |
| `TestMaliciousPatternDetection::test_network_activity_detection` | Same as above |
| `TestMaliciousPatternDetection::test_obfuscated_code_detection` | Same as above |
| `TestRollbackCapabilities::test_rollback_failure_handling` | Same as above |
| `TestRollbackCapabilities::test_rollback_fragment_series` | Same as above |

#### test_git_guardian_integration.py (11 failures)
| Test | Error |
|------|-------|
| `TestGitGuardianIntegrationFunctions::test_fallback_to_direct_commit_failure` | `TypeError: 'NoneType' object is not callable` |
| `TestGitGuardianIntegrationFunctions::test_fallback_to_direct_commit_success` | Same as above |
| `TestGitGuardianIntegrationFunctions::test_request_guardian_commit_success` | `AttributeError: module 'utils' has no attribute 'git_guardian'` |
| `TestGitGuardianIntegrationFunctions::test_request_guardian_commit_with_security_issues` | Same as above |
| `TestPreToolUseIntegration::test_chunked_edit_with_guardian_failure_and_fallback` | `ModuleNotFoundError: No module named 'utils.constants'` |
| `TestPreToolUseIntegration::test_chunked_edit_with_guardian_success` | Same as above |
| `TestEnhancedFileChunkingIntegration::test_create_file_with_chunks_guardian_success` | Same as above |
| `TestGitGuardianLoggging::test_log_guardian_request` | Same as above |
| `TestIntegrationEndToEnd::test_guardian_already_active` | Same as above |
| `TestIntegrationEndToEnd::test_start_git_guardian_if_needed` | Same as above |

#### test_hook_guardian_behavior.py (7 failures)
| Test | Error |
|------|-------|
| `TestEditThresholdWithGuardian::test_large_edit_triggers_chunked_with_guardian` | Module import issues |
| `TestSecurityBlocks::test_dangerous_rm_detection` | `AttributeError: 'NoneType' object has no attribute 'is_dangerous_rm_command'` |
| `TestSecurityBlocks::test_env_file_access_blocked` | `AttributeError: 'NoneType' object has no attribute 'is_env_file_access'` |
| `TestCodeAnalysis::test_analyze_javascript_code` | `AttributeError: 'NoneType' object has no attribute 'analyze_code_content'` |
| `TestCodeAnalysis::test_analyze_python_code` | Same as above |
| `TestCodeAnalysis::test_meaningful_commit_message_generation` | `AttributeError: 'NoneType' object has no attribute 'generate_meaningful_commit_message'` |
| `TestMockHookExecution::test_pre_tool_hook_execution` | `ModuleNotFoundError: No module named 'utils.constants'` |

**Fix:** Create or properly expose the missing modules:
- `utils/enhanced_security_guardian.py`
- `utils/git_guardian.py`
- `utils/constants.py`

---

### 2. Integration Tests (2 failures)

**File:** `test_security_integration.py`

| Test | Error |
|------|-------|
| `TestSecurityFeatures::test_authentication_flow` | `AttributeError: 'SessionManager' object has no attribute 'get_session'` |
| `TestSecurityCompliance::test_session_security_features` | Same as above |

**Fix:** Update `SessionManager` class to include `get_session()` method, or update tests to use `create_session()`.

---

### 3. Edge Case Tests (2 failures)

**File:** `test_comprehensive_edge_cases.py`

| Test | Error |
|------|-------|
| `TestBoundaryConditions::test_very_large_values` | `AssertionError: np.False_ is not true` |
| `TestErrorRecovery::test_sensor_failure_handling` | `TypeError: QCMModel.__init__() got an unexpected keyword argument 'crystal_frequency'` |

**Fix:**
- Review boundary condition logic for large values
- Update `QCMModel` constructor signature or test parameters

---

### 4. Config Tests (1 failure)

**File:** `test_epsilon_fix.py`

| Test | Error |
|------|-------|
| `test_epsilon_decay` | `AttributeError: 'QLearningConfig' object has no attribute 'biofilm_max_deviation'` |

**Fix:** Add `biofilm_max_deviation` attribute to `QLearningConfig` class or update test.

---

### 5. Output Tests (1 failure)

**File:** `test_actual_executions.py`

| Test | Error |
|------|-------|
| `TestFileOutputPatterns::test_expected_output_directories_have_content` | `AssertionError: 0 not greater than 0 : Figures directory should contain PNG files` |

**Fix:** This test expects pre-existing output files. Either:
- Generate test fixtures before running tests
- Skip this test in CI (it's an integration test requiring prior execution)
- Mark as `@pytest.mark.skip` with reason

---

### 6. GPU Tests (6 failures - expected on non-GPU systems)

**File:** `test_gpu_capability.py`

| Test | Status |
|------|--------|
| `test_pytorch_cuda_availability` | No CUDA GPU |
| `test_pytorch_rocm_availability` | No ROCm GPU |
| `test_pytorch_gpu_vs_cpu_performance` | No GPU |

**Fix:** These should be skipped on systems without GPU. Add proper skip conditions:
```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA GPU available")
```

---

## Recommended Actions

### Priority 1: Quick Fixes
1. Add `@pytest.mark.skip` decorators to GPU tests when no GPU present
2. Skip `test_expected_output_directories_have_content` in CI

### Priority 2: Module Fixes
3. Create missing `utils/constants.py` module
4. Expose `enhanced_security_guardian` and `git_guardian` properly in `utils/__init__.py`

### Priority 3: API Updates
5. Add `get_session()` method to `SessionManager` or update tests
6. Add `biofilm_max_deviation` to `QLearningConfig`
7. Fix `QCMModel` constructor parameters

### Priority 4: Test Logic
8. Fix boundary condition test for large values
9. Review sensor failure handling test setup

---

## CI/CD Infrastructure Status

✅ **Fixed issues:**
- Runner DNS resolution (using 8.8.8.8, immutable resolv.conf)
- Runner tag configuration (`run_untagged=true`)
- Pixi installed on runner
- Simplified `.gitlab-ci.yml` (removed Auto-DevOps)
- Pipeline executes successfully

The CI/CD infrastructure is now fully operational. The 34 test failures are code/test issues, not infrastructure problems.
