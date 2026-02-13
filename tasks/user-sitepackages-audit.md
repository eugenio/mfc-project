# User Site-Packages Audit Report

Generated: 2026-01-25

## Summary

This report identifies packages that are loaded from user site-packages (`~/.local/lib/python*/site-packages`) during test execution when running **without** the `PYTHONNOUSERSITE=1` isolation flag.

### Key Findings

| Scenario | Packages from User Site-Packages |
|----------|----------------------------------|
| Without `PYTHONNOUSERSITE=1` | 18 unique packages |
| With `PYTHONNOUSERSITE=1` | 0 packages |

The pixi environment is properly configured - when `PYTHONNOUSERSITE=1` is set, all imports come from the pixi-managed environment.

## User Site-Packages Patterns Checked

- `/home/uge/.local/lib`
- `/home/uge/.local/share`

## Packages Found in User Site-Packages

When running without isolation (`PYTHONNOUSERSITE=1` not set), the following packages were loaded from user site-packages:

| Package | Version | Import Name |
|---------|---------|-------------|
| blinker | 1.9.0 | blinker |
| cachetools | 5.5.2 | cachetools |
| click | 8.1.8 | click |
| cycler | 0.12.1 | cycler |
| googleapis-common-protos | 1.70.0 | google.protobuf |
| iniconfig | 2.1.0 | iniconfig |
| kiwisolver | 1.4.8 | kiwisolver |
| matplotlib | 3.10.3 | matplotlib |
| mpl_toolkits | unknown | mpl_toolkits |
| multipart | unknown | multipart |
| overrides | 7.7.0 | overrides |
| pandas | 2.3.1 | pandas |
| pluggy | 1.6.0 | pluggy |
| pytest | 8.4.1 | pytest |
| python-multipart | 0.0.20 | python_multipart |
| torch | unknown | torch |
| typing_extensions | 4.14.1 | typing_extensions |
| websockets | 15.0.1 | websockets |

## Analysis

### Why These Packages Appear

Many of these packages appear in the user site-packages report because:

1. **pytest and related** (`pytest`, `pluggy`, `iniconfig`) - Testing framework dependencies
2. **Data science stack** (`pandas`, `matplotlib`, `torch`) - Core scientific computing packages
3. **Utility packages** (`click`, `typing_extensions`, `cachetools`) - Common utilities

### Recommendation

These packages are already available in the pixi environment. The key finding is that:

1. **With PYTHONNOUSERSITE=1**: Tests run in complete isolation with 0 user packages
2. **Without PYTHONNOUSERSITE=1**: User site-packages can shadow pixi packages

### Action Items

1. **Always use `PYTHONNOUSERSITE=1`** when running tests for CI/CD
2. **Update CI pipeline** to enforce isolated testing
3. **Ensure pixi.toml** contains all required dependencies (verified - no missing dependencies detected when running isolated)

## Test Verification

Verification command used:
```bash
PYTHONNOUSERSITE=1 pixi run -e dev python scripts/trace_user_sitepackages.py
```

Result: 0 packages from user site-packages - pixi environment is properly configured.

## How to Run This Audit

```bash
# Standard tracing (may include user site-packages)
pixi run -e dev python scripts/trace_user_sitepackages.py

# Isolated tracing (verify pixi environment completeness)
PYTHONNOUSERSITE=1 pixi run -e dev python scripts/trace_user_sitepackages.py

# Using pytest collection method
pixi run -e dev python scripts/trace_user_sitepackages.py --method pytest
```

## Conclusion

The pixi environment is properly configured with all necessary dependencies. When `PYTHONNOUSERSITE=1` is set, no imports come from user site-packages. This confirms that:

1. All required test dependencies are present in pixi.toml
2. The environment isolation works correctly
3. CI/CD should always use `PYTHONNOUSERSITE=1` to ensure consistent behavior
