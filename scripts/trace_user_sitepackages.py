"""Trace imports from user site-packages during test execution.

This script uses Python's import hooks to capture all imports that resolve
to user site-packages (~/.local/lib/python*/site-packages).

Usage:
    python scripts/trace_user_sitepackages.py
    # Or with specific test path:
    python scripts/trace_user_sitepackages.py q-learning-mfcs/tests/config
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import importlib.metadata
import importlib.util
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import ModuleType


@dataclass
class ImportRecord:
    """Record of an import from user site-packages."""

    module_name: str
    file_path: str
    package_name: str = ""
    version: str = ""
    imported_from: list[str] = field(default_factory=list)


class UserSitePackagesTracer:
    """Trace imports from user site-packages."""

    def __init__(self) -> None:
        """Initialize the tracer with user site-packages patterns."""
        self.user_site_patterns = self._get_user_site_patterns()
        self.traced_imports: dict[str, ImportRecord] = {}
        self.import_locations: dict[str, set[str]] = defaultdict(set)
        # Using getattr to access __builtins__.__import__ for monkey-patching
        builtins_import = getattr(__builtins__, "__import__")  # noqa: B009
        self._original_import: Callable[
            [
                str,
                Mapping[str, object] | None,
                Mapping[str, object] | None,
                Sequence[str],
                int,
            ],
            ModuleType,
        ] = builtins_import

    def get_user_site_patterns(self) -> list[str]:
        """Return the user site-packages patterns for external access."""
        return self.user_site_patterns

    def _get_user_site_patterns(self) -> list[str]:
        """Get patterns that identify user site-packages."""
        patterns = []
        home = Path.home()
        patterns.append(str(home / ".local" / "lib"))
        patterns.append(str(home / ".local" / "share"))
        if hasattr(sys, "real_prefix"):
            patterns.append(sys.real_prefix)
        return patterns

    def _is_user_site_path(self, path: str | None) -> bool:
        """Check if a path is in user site-packages."""
        if not path:
            return False
        path_str = str(path)
        return any(pattern in path_str for pattern in self.user_site_patterns)

    def _get_module_path(self, module_name: str) -> str | None:
        """Get the file path for a module."""
        exceptions = (ImportError, ModuleNotFoundError, ValueError, AttributeError)
        with contextlib.suppress(*exceptions):
            spec = importlib.util.find_spec(module_name)
            if spec and spec.origin:
                return spec.origin
            if spec and spec.submodule_search_locations:
                locs = list(spec.submodule_search_locations)
                if locs:
                    return locs[0]
        return None

    def _get_package_info(self, module_name: str) -> tuple[str, str]:
        """Get package name and version for a module."""
        top_level = module_name.split(".")[0]
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            dist = importlib.metadata.distribution(top_level)
            return dist.metadata["Name"], dist.version

        for dist in importlib.metadata.distributions():
            with contextlib.suppress(FileNotFoundError):
                top_levels = dist.read_text("top_level.txt")
                if top_levels and top_level in top_levels.split():
                    return dist.metadata["Name"], dist.version

        return top_level, "unknown"

    def trace_import(
        self,
        name: str,
        globals_dict: dict[str, Any] | None = None,
        locals_dict: dict[str, Any] | None = None,
        fromlist: Sequence[str] | None = None,
        level: int = 0,
    ) -> ModuleType:
        """Trace an import and record if from user site-packages."""
        fromlist_resolved = fromlist if fromlist is not None else ()
        module = self._original_import(
            name, globals_dict, locals_dict, fromlist_resolved, level,
        )
        module_path = self._get_module_path(name)

        if self._is_user_site_path(module_path):
            if name not in self.traced_imports:
                pkg_name, version = self._get_package_info(name)
                self.traced_imports[name] = ImportRecord(
                    module_name=name,
                    file_path=module_path or "",
                    package_name=pkg_name,
                    version=version,
                )

            caller_file = ""
            if globals_dict and "__file__" in globals_dict:
                caller_file = globals_dict["__file__"]
            if caller_file:
                self.traced_imports[name].imported_from.append(caller_file)
                self.import_locations[name].add(caller_file)

        return module

    def start_tracing(self) -> None:
        """Start tracing imports."""
        # Using setattr to modify __builtins__.__import__ for monkey-patching
        setattr(__builtins__, "__import__", self.trace_import)  # noqa: B010

    def stop_tracing(self) -> None:
        """Stop tracing imports."""
        # Using setattr to restore __builtins__.__import__ after monkey-patching
        setattr(__builtins__, "__import__", self._original_import)  # noqa: B010

    def get_report(self) -> str:
        """Generate a markdown report of traced imports."""
        now = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
        lines = [
            "# User Site-Packages Audit Report",
            "",
            f"Generated: {now}",
            "",
            "## Summary",
            "",
            f"- **Total packages from user site-packages**: {len(self.traced_imports)}",
            "- **User site-packages patterns checked**: ",
        ]
        lines.extend(f"  - `{p}`" for p in self.user_site_patterns)

        lines.extend(["", "## Packages Found", ""])

        if not self.traced_imports:
            lines.append(
                "_No imports from user site-packages detected during test run._",
            )
        else:
            lines.append("| Package | Version | Module | Path |")
            lines.append("|---------|---------|--------|------|")

            seen_packages: set[str] = set()
            for record in sorted(
                self.traced_imports.values(),
                key=lambda r: r.package_name,
            ):
                if record.package_name not in seen_packages:
                    seen_packages.add(record.package_name)
                    short_path = record.file_path
                    if "site-packages" in short_path:
                        short_path = "..." + short_path.split("site-packages")[-1]
                    lines.append(
                        f"| {record.package_name} | {record.version} "
                        f"| {record.module_name} | `{short_path}` |",
                    )

        lines.extend(["", "## Import Locations", ""])

        if self.import_locations:
            lines.append(
                "Files that imported packages from user site-packages:",
            )
            lines.append("")

            for module_name, locations in sorted(self.import_locations.items()):
                if locations:
                    lines.append(f"### {module_name}")
                    lines.append("")
                    lines.extend(f"- `{loc}`" for loc in sorted(locations))
                    lines.append("")

        lines.extend(
            [
                "## Recommendations",
                "",
                "Packages listed above should be added to `pixi.toml` to ensure tests "
                "run correctly in an isolated environment.",
                "",
                "### Adding to pixi.toml",
                "",
                "For conda-forge packages (preferred):",
                "```toml",
                "[feature.dev.dependencies]",
                'package-name = "*"',
                "```",
                "",
                "For PyPI-only packages:",
                "```toml",
                "[feature.dev.pypi-dependencies]",
                'package-name = "*"',
                "```",
            ],
        )

        return "\n".join(lines)


def run_tests_with_tracing(test_path: str | None = None) -> UserSitePackagesTracer:
    """Run pytest with import tracing enabled."""
    tracer = UserSitePackagesTracer()
    tracer.start_tracing()

    try:
        import pytest  # noqa: PLC0415

        args = ["-v", "--collect-only", "-q"]
        if test_path:
            args.append(test_path)
        else:
            args.append("q-learning-mfcs/tests")
        pytest.main(args)
    except ImportError as e:
        sys.stderr.write(f"Error: pytest not available: {e}\n")
    finally:
        tracer.stop_tracing()

    return tracer


def trace_all_test_imports() -> UserSitePackagesTracer:
    """Trace imports by actually importing test modules."""
    tracer = UserSitePackagesTracer()
    project_root = Path(__file__).parent.parent
    test_dir = project_root / "q-learning-mfcs" / "tests"

    sys.path.insert(0, str(project_root / "q-learning-mfcs" / "src"))
    sys.path.insert(0, str(project_root))

    tracer.start_tracing()

    try:
        for test_file in test_dir.rglob("test_*.py"):
            module_path = test_file.relative_to(project_root)
            module_name = str(module_path).replace("/", ".").replace(".py", "")
            with contextlib.suppress(Exception):
                importlib.import_module(module_name)

        for test_file in test_dir.rglob("*_test.py"):
            module_path = test_file.relative_to(project_root)
            module_name = str(module_path).replace("/", ".").replace(".py", "")
            with contextlib.suppress(Exception):
                importlib.import_module(module_name)
    finally:
        tracer.stop_tracing()

    return tracer


def main() -> None:
    """Run the user site-packages tracing script."""
    parser = argparse.ArgumentParser(
        description="Trace imports from user site-packages during tests",
    )
    parser.add_argument(
        "test_path",
        nargs="?",
        default=None,
        help="Optional test path to trace",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="tasks/user-sitepackages-audit.md",
        help="Output file for the report",
    )
    parser.add_argument(
        "--method",
        choices=["pytest", "import"],
        default="import",
        help="Tracing method: pytest (collection) or import (direct)",
    )
    args = parser.parse_args()

    tracer_instance = UserSitePackagesTracer()
    sys.stderr.write("Starting user site-packages tracing...\n")
    sys.stderr.write(f"Patterns: {tracer_instance.get_user_site_patterns()}\n")

    if args.method == "pytest":
        tracer = run_tests_with_tracing(args.test_path)
    else:
        tracer = trace_all_test_imports()

    report = tracer.get_report()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)

    sys.stderr.write(f"\nReport saved to: {output_path}\n")
    sys.stderr.write(
        f"Found {len(tracer.traced_imports)} packages from user site-packages\n",
    )

    if tracer.traced_imports:
        sys.stderr.write("\nPackages found:\n")
        seen: set[str] = set()
        for record in tracer.traced_imports.values():
            if record.package_name not in seen:
                seen.add(record.package_name)
                sys.stderr.write(f"  - {record.package_name} ({record.version})\n")


if __name__ == "__main__":
    main()
