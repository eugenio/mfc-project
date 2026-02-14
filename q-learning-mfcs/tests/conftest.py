"""Root test configuration — sys.modules leak guard.

Many coverage test files mock heavy dependencies (torch, numpy, pandas, etc.)
at module level by directly assigning into sys.modules. These mocks persist
across the entire pytest session and accumulate, causing multi-GB memory usage.

This conftest snapshots sys.modules at session start and provides a
per-module cleanup fixture that removes leaked MagicMock entries after each
test module finishes.
"""

from __future__ import annotations

import gc
import sys
from unittest.mock import MagicMock

import pytest


def _is_mock(obj: object) -> bool:
    """Check if an object is a MagicMock or its subclass."""
    return isinstance(obj, MagicMock)


@pytest.fixture(scope="session", autouse=True)
def _guard_sys_modules():
    """Snapshot sys.modules at session start; restore after session ends.

    This prevents module-level MagicMock injections from leaking across
    test subdirectories and accumulating memory.
    """
    snapshot = dict(sys.modules)
    yield
    # Restore originals, remove mocks that weren't there before
    to_delete = []
    to_restore = {}
    for key in list(sys.modules):
        current = sys.modules[key]
        if key not in snapshot:
            if _is_mock(current):
                to_delete.append(key)
        elif _is_mock(current) and not _is_mock(snapshot[key]):
            to_restore[key] = snapshot[key]

    for key in to_delete:
        del sys.modules[key]
    for key, val in to_restore.items():
        sys.modules[key] = val

    gc.collect()


@pytest.fixture(autouse=True)
def _per_test_mock_cleanup():
    """Per-test fixture that tracks sys.modules changes within each test.

    Removes any MagicMock entries added during a single test function,
    preventing accumulation across hundreds of tests.
    """
    pre_keys = set(sys.modules)
    pre_mocks = {k for k, v in sys.modules.items() if _is_mock(v)}
    yield
    # Remove mocks that were added during this test
    new_mock_keys = []
    for k in list(sys.modules):
        if k not in pre_keys and _is_mock(sys.modules[k]):
            new_mock_keys.append(k)
        elif k in pre_keys and k not in pre_mocks and _is_mock(sys.modules[k]):
            new_mock_keys.append(k)

    for k in new_mock_keys:
        if k in pre_keys:
            # Was a real module before, but now mocked — skip restore here
            # (session guard handles this)
            pass
        else:
            del sys.modules[k]
