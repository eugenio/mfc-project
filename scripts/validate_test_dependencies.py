"""Validate test dependencies against pixi.toml.

This script checks that all imports used in test files are available
in the pixi environment. It's designed to be run as a pre-commit hook
or CI check to prevent dependency drift.

Usage:
    python scripts/validate_test_dependencies.py
    # or via pixi task:
    pixi run validate-deps

Exit codes:
    0 - All dependencies satisfied
    1 - Missing dependencies found
    2 - Script error
"""

from __future__ import annotations

import ast
import sys
from collections import defaultdict
from pathlib import Path

import tomllib

# Standard library modules (Python 3.12)
STDLIB_MODULES = {
    "abc", "argparse", "array", "ast", "asyncio", "atexit", "base64",
    "bisect", "builtins", "bz2", "calendar", "cmath", "cmd", "code",
    "codecs", "collections", "colorsys", "concurrent", "configparser",
    "contextlib", "contextvars", "copy", "copyreg", "csv", "ctypes",
    "dataclasses", "datetime", "decimal", "difflib", "dis", "email",
    "encodings", "enum", "errno", "faulthandler", "filecmp", "fileinput",
    "fnmatch", "fractions", "functools", "gc", "getopt", "getpass", "glob",
    "graphlib", "gzip", "hashlib", "heapq", "hmac", "html", "http",
    "importlib", "inspect", "io", "ipaddress", "itertools", "json",
    "keyword", "linecache", "locale", "logging", "lzma", "mailbox",
    "marshal", "math", "mimetypes", "mmap", "multiprocessing", "netrc",
    "numbers", "operator", "os", "pathlib", "pickle", "pkgutil", "platform",
    "plistlib", "pprint", "profile", "pstats", "pty", "pwd", "py_compile",
    "queue", "quopri", "random", "re", "readline", "reprlib", "resource",
    "rlcompleter", "runpy", "sched", "secrets", "select", "selectors",
    "shelve", "shlex", "shutil", "signal", "site", "smtplib", "socket",
    "socketserver", "sqlite3", "ssl", "stat", "statistics", "string",
    "struct", "subprocess", "sunau", "symtable", "sys", "sysconfig",
    "tabnanny", "tarfile", "telnetlib", "tempfile", "termios", "textwrap",
    "threading", "time", "timeit", "token", "tokenize", "tomllib", "trace",
    "traceback", "tracemalloc", "tty", "turtle", "types", "typing",
    "unicodedata", "unittest", "urllib", "uu", "uuid", "venv", "warnings",
    "wave", "weakref", "webbrowser", "wsgiref", "xdrlib", "xml", "xmlrpc",
    "zipapp", "zipfile", "zipimport", "zlib", "_thread", "typing_extensions",
}

# Import name to package name mapping for common cases
IMPORT_TO_PACKAGE = {
    "cv2": "opencv-python",
    "PIL": "pillow",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    "google": "google-cloud",
    "scipy": "scipy",
    "np": "numpy",
    "pd": "pandas",
    "plt": "matplotlib",
}


def get_imports_from_file(filepath: Path) -> set[str]:
    """Extract import names from a Python file."""
    imports: set[str] = set()
    try:
        with filepath.open(encoding="utf-8") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_module = alias.name.split(".")[0]
                    imports.add(top_module)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_module = node.module.split(".")[0]
                imports.add(top_module)
    except (SyntaxError, UnicodeDecodeError):
        pass  # Skip files with syntax errors
    return imports


def get_pixi_packages(project_root: Path) -> set[str]:
    """Get all packages from pixi.toml."""
    pixi_toml = project_root / "pixi.toml"
    if not pixi_toml.exists():
        return set()

    packages: set[str] = set()
    with pixi_toml.open("rb") as f:
        config = tomllib.load(f)

    # Extract from various sections
    sections = [
        "dependencies",
        "pypi-dependencies",
    ]
    for section in sections:
        if section in config:
            packages.update(config[section].keys())

    # Extract from features
    for feature_config in config.get("feature", {}).values():
        for section in ["dependencies", "pypi-dependencies"]:
            if section in feature_config:
                packages.update(feature_config[section].keys())

    return packages


def is_local_module(module_name: str, project_root: Path) -> bool:  # noqa: PLR0911
    """Check if a module is local to the project."""
    src_dir = project_root / "q-learning-mfcs" / "src"
    tests_dir = project_root / "q-learning-mfcs" / "tests"
    hooks_dir = project_root / ".claude" / "hooks"

    # Check src directory
    if (src_dir / module_name).is_dir():
        return True
    if (src_dir / f"{module_name}.py").is_file():
        return True

    # Check tests directory
    if (tests_dir / module_name).is_dir():
        return True
    if (tests_dir / f"{module_name}.py").is_file():
        return True

    # Check hooks directory
    if (hooks_dir / module_name).is_dir():
        return True
    if (hooks_dir / f"{module_name}.py").is_file():
        return True
    if (hooks_dir / "utils" / f"{module_name}.py").is_file():
        return True

    # Common project module patterns
    if module_name in {"src", "q_learning_mfcs", "mfc", "tests"}:
        return True

    # Test file imports (test_* from __init__.py)
    if module_name.startswith("test_"):
        return True

    # Streamlit test utilities
    return module_name in {"streamlit_test_server"}


def validate_dependencies() -> int:  # noqa: C901, PLR0912
    """Validate test dependencies against pixi.toml.

    Returns:
        0 if all dependencies are satisfied
        1 if missing dependencies found

    """
    project_root = Path(__file__).parent.parent
    tests_dir = project_root / "q-learning-mfcs" / "tests"

    if not tests_dir.exists():
        print(f"Error: Tests directory not found: {tests_dir}")  # noqa: T201
        return 2

    # Collect all imports from test files
    all_imports: dict[str, set[str]] = defaultdict(set)
    test_files = list(tests_dir.rglob("*.py"))

    for filepath in test_files:
        if "__pycache__" in str(filepath):
            continue
        imports = get_imports_from_file(filepath)
        for imp in imports:
            all_imports[imp].add(str(filepath.relative_to(project_root)))

    # Get pixi packages
    pixi_packages = get_pixi_packages(project_root)
    # Normalize package names (replace - with _)
    normalized_packages = {pkg.replace("-", "_").lower() for pkg in pixi_packages}
    normalized_packages.update({pkg.lower() for pkg in pixi_packages})

    # Find missing packages
    missing = []
    for module_name in sorted(all_imports.keys()):
        # Skip standard library
        if module_name in STDLIB_MODULES:
            continue
        # Skip local modules
        if is_local_module(module_name, project_root):
            continue
        # Check if package or normalized name is in pixi
        normalized_module = module_name.replace("-", "_").lower()
        if normalized_module in normalized_packages:
            continue
        if module_name.lower() in normalized_packages:
            continue
        # Check package name mapping
        mapped_package = IMPORT_TO_PACKAGE.get(module_name)
        if mapped_package and mapped_package.lower() in normalized_packages:
            continue
        # Known test utilities that are part of pytest
        if module_name in {"pytest", "_pytest", "conftest"}:
            continue
        # Check for type stubs
        if f"types-{module_name}".lower() in normalized_packages:
            continue

        # Optional GPU packages (have pytest.skip in tests)
        if module_name in {"cupy", "tensorflow", "torch", "jax", "optuna"}:
            continue

        missing.append((module_name, all_imports[module_name]))

    # CLI output for validation results
    max_locations_shown = 3
    if missing:
        print("Missing dependencies found:")  # noqa: T201
        print("-" * 50)  # noqa: T201
        for module_name, locations in missing:
            print(f"\n{module_name}:")  # noqa: T201
            for loc in sorted(locations)[:max_locations_shown]:
                print(f"  - {loc}")  # noqa: T201
            if len(locations) > max_locations_shown:
                print(f"  ... and {len(locations) - max_locations_shown} more files")  # noqa: T201
        print("\n" + "-" * 50)  # noqa: T201
        print(f"Total: {len(missing)} missing package(s)")  # noqa: T201
        return 1

    print("All test dependencies are satisfied!")  # noqa: T201
    print(f"Checked {len(test_files)} test files")  # noqa: T201
    print(f"Found {len(all_imports)} unique imports")  # noqa: T201
    print(f"Verified against {len(pixi_packages)} pixi packages")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(validate_dependencies())
