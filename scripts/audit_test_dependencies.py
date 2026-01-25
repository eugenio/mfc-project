"""Audit test file dependencies against pixi.toml.

This script scans all test files for imports and compares them against
the packages declared in pixi.toml to identify missing dependencies.

Usage:
    python scripts/audit_test_dependencies.py

Output:
    - List of all unique imports found in test files
    - Packages declared in pixi.toml
    - Missing packages with file locations
"""

from __future__ import annotations

import ast
import sys
from collections import defaultdict
from pathlib import Path

import tomllib

# Constants for output formatting
MAX_LOCATIONS_SHOWN = 5
LINE_WIDTH = 70

# Standard library modules that should be excluded from the audit
# This is a comprehensive list of Python 3.12 standard library modules
STDLIB_MODULES = {
    "abc",
    "aifc",
    "argparse",
    "array",
    "ast",
    "asyncio",
    "atexit",
    "base64",
    "bdb",
    "binascii",
    "bisect",
    "builtins",
    "bz2",
    "calendar",
    "cgi",
    "cgitb",
    "chunk",
    "cmath",
    "cmd",
    "code",
    "codecs",
    "codeop",
    "collections",
    "colorsys",
    "compileall",
    "concurrent",
    "configparser",
    "contextlib",
    "contextvars",
    "copy",
    "copyreg",
    "cProfile",
    "crypt",
    "csv",
    "ctypes",
    "curses",
    "dataclasses",
    "datetime",
    "dbm",
    "decimal",
    "difflib",
    "dis",
    "doctest",
    "email",
    "encodings",
    "enum",
    "errno",
    "faulthandler",
    "fcntl",
    "filecmp",
    "fileinput",
    "fnmatch",
    "fractions",
    "ftplib",
    "functools",
    "gc",
    "getopt",
    "getpass",
    "gettext",
    "glob",
    "graphlib",
    "grp",
    "gzip",
    "hashlib",
    "heapq",
    "hmac",
    "html",
    "http",
    "idlelib",
    "imaplib",
    "imghdr",
    "imp",
    "importlib",
    "inspect",
    "io",
    "ipaddress",
    "itertools",
    "json",
    "keyword",
    "lib2to3",
    "linecache",
    "locale",
    "logging",
    "lzma",
    "mailbox",
    "mailcap",
    "marshal",
    "math",
    "mimetypes",
    "mmap",
    "modulefinder",
    "multiprocessing",
    "netrc",
    "nis",
    "nntplib",
    "numbers",
    "operator",
    "optparse",
    "os",
    "ossaudiodev",
    "pathlib",
    "pdb",
    "pickle",
    "pickletools",
    "pipes",
    "pkgutil",
    "platform",
    "plistlib",
    "poplib",
    "posix",
    "posixpath",
    "pprint",
    "profile",
    "pstats",
    "pty",
    "pwd",
    "py_compile",
    "pyclbr",
    "pydoc",
    "queue",
    "quopri",
    "random",
    "re",
    "readline",
    "reprlib",
    "resource",
    "rlcompleter",
    "runpy",
    "sched",
    "secrets",
    "select",
    "selectors",
    "shelve",
    "shlex",
    "shutil",
    "signal",
    "site",
    "smtpd",
    "smtplib",
    "sndhdr",
    "socket",
    "socketserver",
    "spwd",
    "sqlite3",
    "ssl",
    "stat",
    "statistics",
    "string",
    "stringprep",
    "struct",
    "subprocess",
    "sunau",
    "symtable",
    "sys",
    "sysconfig",
    "syslog",
    "tabnanny",
    "tarfile",
    "telnetlib",
    "tempfile",
    "termios",
    "test",
    "textwrap",
    "threading",
    "time",
    "timeit",
    "tkinter",
    "token",
    "tokenize",
    "tomllib",
    "trace",
    "traceback",
    "tracemalloc",
    "tty",
    "turtle",
    "turtledemo",
    "types",
    "typing",
    "typing_extensions",
    "unicodedata",
    "unittest",
    "urllib",
    "uu",
    "uuid",
    "venv",
    "warnings",
    "wave",
    "weakref",
    "webbrowser",
    "winreg",
    "winsound",
    "wsgiref",
    "xdrlib",
    "xml",
    "xmlrpc",
    "zipapp",
    "zipfile",
    "zipimport",
    "zlib",
    "zoneinfo",
    "_thread",
    "__future__",
}

# Mapping of import names to package names when they differ
IMPORT_TO_PACKAGE = {
    "PIL": "pillow",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    "mpl_toolkits": "matplotlib",
    "pkg_resources": "setuptools",
    "google": "google-api-python-client",
    "Bio": "biopython",
    "OpenSSL": "pyopenssl",
    "jwt": "pyjwt",
    "Crypto": "pycryptodome",
    "magic": "python-magic",
    "dotenv": "python-dotenv",
    "gitlab": "python-gitlab",
    "psutil": "psutil",
    "starlette": "starlette",
    "uvicorn": "uvicorn",
    "fastapi": "fastapi",
    "pydantic": "pydantic",
    "pydantic_core": "pydantic",
    "pydantic_settings": "pydantic-settings",
    "streamlit_autorefresh": "streamlit-autorefresh",
    "webdriver_manager": "webdriver-manager",
    "detect_secrets": "detect-secrets",
    "cchooks": "cchooks",
    # GPU-related packages with platform-specific names
    "cupy": "cupy-cuda12x",
    "numba": "numba",
    "tensorflow": "tensorflow",
    "pyarrow": "pyarrow",
}


def get_top_level_module(module_name: str) -> str:
    """Extract the top-level module name from an import."""
    return module_name.split(".")[0]


def extract_imports_from_file(file_path: Path) -> set[str]:
    """Extract all import statements from a Python file using AST.

    Args:
        file_path: Path to the Python file

    Returns:
        Set of top-level module names imported in the file

    """
    imports: set[str] = set()
    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(file_path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(get_top_level_module(alias.name))
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(get_top_level_module(node.module))
    except SyntaxError as e:
        print(f"Warning: Could not parse {file_path}: {e}")  # noqa: T201
    except OSError as e:
        print(f"Warning: Error reading {file_path}: {e}")  # noqa: T201

    return imports


def find_test_files(base_path: Path) -> list[Path]:
    """Find all Python test files in the project.

    Args:
        base_path: Root path of the project

    Returns:
        List of paths to test Python files

    """
    test_files = []
    tests_dir = base_path / "q-learning-mfcs" / "tests"

    if tests_dir.exists():
        for py_file in tests_dir.rglob("*.py"):
            # Skip __pycache__ directories
            if "__pycache__" in str(py_file):
                continue
            test_files.append(py_file)

    return test_files


def _parse_deps_section(
    config: dict,
    section_key: str,
    packages: dict[str, set[str]],
    pkg_type: str,
) -> None:
    """Parse a dependencies section from pixi config.

    Args:
        config: The config dictionary to parse
        section_key: Key to look for (dependencies or pypi-dependencies)
        packages: Dictionary to add packages to
        pkg_type: Type key in packages dict (conda or pypi)

    """
    if section_key in config:
        for pkg in config[section_key]:
            normalized = pkg.lower()
            if pkg_type == "pypi":
                normalized = normalized.replace("_", "-")
            packages[pkg_type].add(normalized)


def parse_pixi_toml(pixi_path: Path) -> dict[str, set[str]]:
    """Parse pixi.toml to extract all declared dependencies.

    Args:
        pixi_path: Path to pixi.toml

    Returns:
        Dictionary with keys 'conda' and 'pypi' containing package names

    """
    packages: dict[str, set[str]] = {"conda": set(), "pypi": set()}

    with pixi_path.open("rb") as f:
        config = tomllib.load(f)

    # Parse top-level dependencies
    _parse_deps_section(config, "dependencies", packages, "conda")
    _parse_deps_section(config, "pypi-dependencies", packages, "pypi")

    # Parse feature dependencies
    if "feature" in config:
        for feature_config in config["feature"].values():
            _parse_deps_section(feature_config, "dependencies", packages, "conda")
            _parse_deps_section(feature_config, "pypi-dependencies", packages, "pypi")

    return packages


def normalize_package_name(name: str) -> str:
    """Normalize package name for comparison.

    Package names are case-insensitive and underscores/hyphens are equivalent.
    """
    return name.lower().replace("_", "-")


def is_local_import(module_name: str, base_path: Path) -> bool:
    """Check if the import is a local project import.

    Args:
        module_name: The module name to check
        base_path: Root path of the project

    Returns:
        True if this is a local project import

    """
    # Check common project module patterns
    local_modules = {
        "src",
        "q_learning_mfcs",
        "qlearning",
        "config",
        "gui",
        "monitoring",
        "hooks",
        "tests",
        "helpers",
        "utils",
    }
    if module_name.lower() in local_modules:
        return True

    # Check various project locations for the module (as directory or .py file)
    search_paths = [
        base_path / "q-learning-mfcs" / "src",
        base_path / "q-learning-mfcs" / "tests",
        base_path / ".claude" / "hooks",
    ]

    for search_path in search_paths:
        # Check as directory
        if (search_path / module_name).exists():
            return True
        # Check as .py file
        if (search_path / f"{module_name}.py").exists():
            return True
        # Check recursively in subdirectories for .py files
        for py_file in search_path.rglob(f"{module_name}.py"):
            if py_file.exists():
                return True

    return False


def print_missing_packages(missing_packages: dict[str, list[str]]) -> None:
    """Print the missing packages report section.

    Args:
        missing_packages: Dictionary mapping package names to file locations

    """
    print("-" * LINE_WIDTH)  # noqa: T201
    print("MISSING PACKAGES")  # noqa: T201
    print("-" * LINE_WIDTH)  # noqa: T201
    for pkg, locations in sorted(missing_packages.items()):
        mapped = IMPORT_TO_PACKAGE.get(pkg, pkg)
        if pkg != mapped:
            print(f"\n{pkg} (package: {mapped})")  # noqa: T201
        else:
            print(f"\n{pkg}")  # noqa: T201
        print("  Used in:")  # noqa: T201
        for loc in sorted(locations)[:MAX_LOCATIONS_SHOWN]:
            print(f"    - {loc}")  # noqa: T201
        if len(locations) > MAX_LOCATIONS_SHOWN:
            remaining = len(locations) - MAX_LOCATIONS_SHOWN
            print(f"    ... and {remaining} more files")  # noqa: T201
    print()  # noqa: T201
    print(f"Total missing packages: {len(missing_packages)}")  # noqa: T201


def collect_imports(
    test_files: list[Path],
    base_path: Path,
) -> tuple[set[str], dict[str, list[str]]]:
    """Collect imports from test files.

    Args:
        test_files: List of test file paths
        base_path: Base path of the project

    Returns:
        Tuple of (external_imports set, import_locations dict)

    """
    all_imports: set[str] = set()
    import_locations: dict[str, list[str]] = defaultdict(list)

    for test_file in test_files:
        file_imports = extract_imports_from_file(test_file)
        all_imports.update(file_imports)

        for imp in file_imports:
            rel_path = test_file.relative_to(base_path)
            import_locations[imp].append(str(rel_path))

    # Filter out stdlib and local imports
    external_imports = {
        imp
        for imp in all_imports
        if imp not in STDLIB_MODULES and not is_local_import(imp, base_path)
    }

    return external_imports, import_locations


def find_missing_packages(
    external_imports: set[str],
    all_pixi_packages: set[str],
    import_locations: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Find packages imported but not in pixi.toml.

    Args:
        external_imports: Set of external import names
        all_pixi_packages: Set of package names from pixi.toml
        import_locations: Dict mapping imports to file locations

    Returns:
        Dictionary of missing packages with their file locations

    """
    missing_packages: dict[str, list[str]] = {}

    for imp in sorted(external_imports):
        package_name = IMPORT_TO_PACKAGE.get(imp, imp)
        normalized = normalize_package_name(package_name)
        normalized_imp = normalize_package_name(imp)

        found = any(pkg in {normalized, normalized_imp} for pkg in all_pixi_packages)

        if not found:
            missing_packages[imp] = import_locations[imp]

    return missing_packages


def print_report(
    external_imports: set[str],
    pixi_packages: dict[str, set[str]],
    missing_packages: dict[str, list[str]],
) -> None:
    """Print the full audit report.

    Args:
        external_imports: Set of external imports found
        pixi_packages: Dict with conda and pypi package sets
        missing_packages: Dict of missing packages with locations

    """
    print("-" * LINE_WIDTH)  # noqa: T201
    print("EXTERNAL IMPORTS FOUND IN TESTS")  # noqa: T201
    print("-" * LINE_WIDTH)  # noqa: T201
    for imp in sorted(external_imports):
        pkg_name = IMPORT_TO_PACKAGE.get(imp, imp)
        if imp != pkg_name:
            print(f"  {imp} -> {pkg_name}")  # noqa: T201
        else:
            print(f"  {imp}")  # noqa: T201
    print()  # noqa: T201

    print("-" * LINE_WIDTH)  # noqa: T201
    print("PACKAGES IN PIXI.TOML")  # noqa: T201
    print("-" * LINE_WIDTH)  # noqa: T201
    print("Conda packages:")  # noqa: T201
    for pkg in sorted(pixi_packages["conda"]):
        print(f"  {pkg}")  # noqa: T201
    print()  # noqa: T201
    print("PyPI packages:")  # noqa: T201
    for pkg in sorted(pixi_packages["pypi"]):
        print(f"  {pkg}")  # noqa: T201
    print()  # noqa: T201

    if missing_packages:
        print_missing_packages(missing_packages)


def main() -> int:
    """Run the audit and return exit code."""
    base_path = Path(__file__).parent.parent
    pixi_path = base_path / "pixi.toml"

    print("=" * LINE_WIDTH)  # noqa: T201
    print("Test Dependency Audit Report")  # noqa: T201
    print("=" * LINE_WIDTH)  # noqa: T201
    print()  # noqa: T201

    # Find all test files
    test_files = find_test_files(base_path)
    print(f"Found {len(test_files)} test files")  # noqa: T201
    print()  # noqa: T201

    # Collect imports
    external_imports, import_locations = collect_imports(test_files, base_path)
    print(f"Found {len(external_imports)} external package imports")  # noqa: T201
    print()  # noqa: T201

    # Parse pixi.toml dependencies
    pixi_packages = parse_pixi_toml(pixi_path)
    all_pixi_packages = pixi_packages["conda"] | pixi_packages["pypi"]

    print(f"Found {len(all_pixi_packages)} packages in pixi.toml")  # noqa: T201
    print(f"  - Conda packages: {len(pixi_packages['conda'])}")  # noqa: T201
    print(f"  - PyPI packages: {len(pixi_packages['pypi'])}")  # noqa: T201
    print()  # noqa: T201

    # Find missing packages
    missing_packages = find_missing_packages(
        external_imports,
        all_pixi_packages,
        import_locations,
    )

    # Print report
    print_report(external_imports, pixi_packages, missing_packages)

    if missing_packages:
        return 1

    print("-" * LINE_WIDTH)  # noqa: T201
    print("SUCCESS: All test dependencies are declared in pixi.toml!")  # noqa: T201
    print("-" * LINE_WIDTH)  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
