"""
Coverage Utilities Module

Provides comprehensive test coverage analysis and reporting utilities
for the Q-Learning MFC project.
"""
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def run_quick_coverage_check(source_dir: str, test_dir: str) -> bool:
    """Run a quick coverage check and return success status."""
    analyzer = CoverageAnalyzer()

    # Run coverage analysis
    result = analyzer.run_coverage_analysis(source_dir, test_dir)

    if not result['success']:
        print(f"Coverage analysis failed: {result['stderr']}")
        return False

    # Parse results
    try:
        coverage_results = analyzer.parse_coverage_json()
        summary = analyzer.get_coverage_summary(coverage_results)

        overall_coverage = summary['overall_line_coverage']
        return overall_coverage >= 95.0
    except Exception as e:
        print(f"Error parsing coverage results: {e}")
        return False
@dataclass
class CoverageThresholds:
    """Configuration class for coverage thresholds."""

    # Overall coverage thresholds
    total_line_threshold: float = 95.0
    total_branch_threshold: float = 90.0

    # Per-module thresholds
    module_line_threshold: float = 90.0
    module_branch_threshold: float = 85.0

    # Per-file thresholds
    file_line_threshold: float = 85.0
    file_branch_threshold: float = 80.0

    # Warning thresholds (below these trigger warnings)
    warning_line_threshold: float = 70.0
    warning_branch_threshold: float = 65.0

    # Critical thresholds (below these trigger failures)
    critical_line_threshold: float = 50.0
    critical_branch_threshold: float = 45.0

@dataclass
class CoverageResult:
    """Results from coverage analysis."""

    file_path: str
    line_coverage: float
    branch_coverage: float = 0.0
    lines_covered: int = 0
    lines_total: int = 0
    branches_covered: int = 0
    branches_total: int = 0
    missing_lines: list[int] = field(default_factory=list)
    partial_lines: list[int] = field(default_factory=list)

    @property
    def passes_thresholds(self) -> bool:
        """Check if result passes minimum thresholds."""
        thresholds = CoverageThresholds()
        return (self.line_coverage >= thresholds.file_line_threshold and
                self.branch_coverage >= thresholds.file_branch_threshold)

    @property
    def is_warning_level(self) -> bool:
        """Check if result is at warning level."""
        thresholds = CoverageThresholds()
        return (self.line_coverage < thresholds.warning_line_threshold or
                self.branch_coverage < thresholds.warning_branch_threshold)

    @property
    def is_critical_level(self) -> bool:
        """Check if result is at critical level."""
        thresholds = CoverageThresholds()
        return (self.line_coverage < thresholds.critical_line_threshold or
                self.branch_coverage < thresholds.critical_branch_threshold)

class CoverageAnalyzer:
    """Analyzer for test coverage data."""

    def __init__(self, project_root: str | None = None):
        """Initialize coverage analyzer."""
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.coverage_data_file = self.project_root / ".coverage"
        self.thresholds = CoverageThresholds()

    def run_coverage_analysis(
        self,
        source_dir: str = "src",
        test_dir: str = "tests",
        output_format: str = "json",
        include_branches: bool = True
    ) -> dict[str, Any]:
        """Run pytest with coverage and return analysis results."""

        # Build coverage command
        cmd = [
            sys.executable, "-m", "pytest",
            "--cov=" + source_dir,
            "--cov-report=term-missing",
            test_dir
        ]

        if include_branches:
            cmd.append("--cov-branch")

        if output_format == "json":
            cmd.append("--cov-report=json:coverage.json")
        elif output_format == "xml":
            cmd.append("--cov-report=xml:coverage.xml")
        elif output_format == "html":
            cmd.append("--cov-report=html:htmlcov")

        # Run coverage
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=False
            )

            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'stdout': '',
                'stderr': str(e),
                'return_code': -1
            }

    def parse_coverage_json(self, json_file: str = "coverage.json") -> dict[str, CoverageResult]:
        """Parse coverage JSON output into structured results."""
        json_path = self.project_root / json_file

        if not json_path.exists():
            raise FileNotFoundError(f"Coverage JSON file not found: {json_path}")

        with open(json_path) as f:
            data = json.load(f)

        results = {}

        # Parse file-level coverage data
        files = data.get('files', {})
        for file_path, file_data in files.items():
            summary = file_data.get('summary', {})

            result = CoverageResult(
                file_path=file_path,
                line_coverage=summary.get('percent_covered', 0.0),
                lines_covered=summary.get('covered_lines', 0),
                lines_total=summary.get('num_statements', 0),
                missing_lines=file_data.get('missing_lines', []),
                partial_lines=file_data.get('excluded_lines', [])
            )

            # Add branch coverage if available
            if 'percent_covered_display' in summary:
                # Try to extract branch coverage from display string
                display = summary['percent_covered_display']
                if '/' in display:
                    try:
                        branch_part = display.split('/')[-1].replace('%', '')
                        result.branch_coverage = float(branch_part)
                    except (ValueError, IndexError):
                        pass

            results[file_path] = result

        return results

    def parse_coverage_xml(self, xml_file: str = "coverage.xml") -> dict[str, CoverageResult]:
        """Parse coverage XML output into structured results."""
        xml_path = self.project_root / xml_file

        if not xml_path.exists():
            raise FileNotFoundError(f"Coverage XML file not found: {xml_path}")

        tree = ET.parse(xml_path)
        root = tree.getroot()

        results = {}

        # Parse package and class elements
        for package in root.findall('.//package'):
            for class_elem in package.findall('classes/class'):
                filename = class_elem.get('filename', '')

                # Calculate line coverage
                lines = class_elem.findall('lines/line')
                total_lines = len(lines)
                covered_lines = len([line for line in lines if line.get('hits', '0') != '0'])

                line_coverage = (covered_lines / total_lines * 100) if total_lines > 0 else 0.0

                # Calculate branch coverage
                branches_total = 0
                branches_covered = 0

                for line in lines:
                    branch = line.get('branch')
                    if branch == 'true':
                        condition_coverage = line.get('condition-coverage', '')
                        if condition_coverage:
                            # Parse condition-coverage format: "50% (1/2)"
                            try:
                                parts = condition_coverage.split('(')[1].split(')')[0].split('/')
                                covered = int(parts[0])
                                total = int(parts[1])
                                branches_covered += covered
                                branches_total += total
                            except (IndexError, ValueError):
                                pass

                branch_coverage = (branches_covered / branches_total * 100) if branches_total > 0 else 0.0

                # Find missing lines
                missing_lines = [
                    int(line.get('number', 0))
                    for line in lines
                    if line.get('hits', '0') == '0'
                ]

                result = CoverageResult(
                    file_path=filename,
                    line_coverage=line_coverage,
                    branch_coverage=branch_coverage,
                    lines_covered=covered_lines,
                    lines_total=total_lines,
                    branches_covered=branches_covered,
                    branches_total=branches_total,
                    missing_lines=missing_lines
                )

                results[filename] = result

        return results

    def analyze_coverage_trends(
        self,
        current_results: dict[str, CoverageResult],
        previous_results: dict[str, CoverageResult] | None = None
    ) -> dict[str, Any]:
        """Analyze coverage trends compared to previous results."""

        if previous_results is None:
            return {'trend': 'no_baseline', 'changes': {}}

        trends = {
            'improved': [],
            'declined': [],
            'new_files': [],
            'removed_files': [],
            'unchanged': []
        }

        # Files in current but not in previous
        current_files = set(current_results.keys())
        previous_files = set(previous_results.keys())

        trends['new_files'] = list(current_files - previous_files)
        trends['removed_files'] = list(previous_files - current_files)

        # Compare common files
        common_files = current_files & previous_files

        for file_path in common_files:
            current = current_results[file_path]
            previous = previous_results[file_path]

            current_total = current.line_coverage + current.branch_coverage
            previous_total = previous.line_coverage + previous.branch_coverage

            change = current_total - previous_total

            if abs(change) < 0.1:  # Essentially unchanged
                trends['unchanged'].append(file_path)
            elif change > 0:
                trends['improved'].append({
                    'file': file_path,
                    'change': change,
                    'current': current_total,
                    'previous': previous_total
                })
            else:
                trends['declined'].append({
                    'file': file_path,
                    'change': change,
                    'current': current_total,
                    'previous': previous_total
                })

        return trends

    def get_coverage_summary(self, results: dict[str, CoverageResult]) -> dict[str, Any]:
        """Get overall coverage summary."""
        if not results:
            return {
                'total_files': 0,
                'overall_line_coverage': 0.0,
                'overall_branch_coverage': 0.0,
                'files_passing_threshold': 0,
                'files_at_warning_level': 0,
                'files_at_critical_level': 0
            }

        total_lines = sum(r.lines_total for r in results.values())
        total_covered_lines = sum(r.lines_covered for r in results.values())
        total_branches = sum(r.branches_total for r in results.values())
        total_covered_branches = sum(r.branches_covered for r in results.values())

        overall_line_coverage = (total_covered_lines / total_lines * 100) if total_lines > 0 else 0.0
        overall_branch_coverage = (total_covered_branches / total_branches * 100) if total_branches > 0 else 0.0

        files_passing = sum(1 for r in results.values() if r.passes_thresholds)
        files_warning = sum(1 for r in results.values() if r.is_warning_level)
        files_critical = sum(1 for r in results.values() if r.is_critical_level)

        return {
            'total_files': len(results),
            'overall_line_coverage': overall_line_coverage,
            'overall_branch_coverage': overall_branch_coverage,
            'total_lines': total_lines,
            'total_covered_lines': total_covered_lines,
            'total_branches': total_branches,
            'total_covered_branches': total_covered_branches,
            'files_passing_threshold': files_passing,
            'files_at_warning_level': files_warning,
            'files_at_critical_level': files_critical,
            'threshold_pass_rate': (files_passing / len(results) * 100) if results else 0.0
        }

class CoverageReporter:
    """Generate coverage reports in various formats."""

    def __init__(self, analyzer: CoverageAnalyzer):
        """Initialize coverage reporter."""
        self.analyzer = analyzer

    def generate_text_report(
        self,
        results: dict[str, CoverageResult],
        summary: dict[str, Any],
        show_details: bool = True
    ) -> str:
        """Generate a text coverage report."""

        lines = []
        lines.append("=" * 80)
        lines.append("COVERAGE ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Summary section
        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Total Files: {summary['total_files']}")
        lines.append(f"Overall Line Coverage: {summary['overall_line_coverage']:.2f}%")
        lines.append(f"Overall Branch Coverage: {summary['overall_branch_coverage']:.2f}%")
        lines.append(f"Files Passing Threshold: {summary['files_passing_threshold']}/{summary['total_files']}")
        lines.append(f"Threshold Pass Rate: {summary['threshold_pass_rate']:.2f}%")
        lines.append("")

        # Threshold status
        if summary['files_at_critical_level'] > 0:
            lines.append(f"⚠️  CRITICAL: {summary['files_at_critical_level']} files below critical threshold")
        if summary['files_at_warning_level'] > 0:
            lines.append(f"⚠️  WARNING: {summary['files_at_warning_level']} files below warning threshold")
        lines.append("")

        if show_details:
            # Detailed file results
            lines.append("DETAILED RESULTS")
            lines.append("-" * 40)
            lines.append(f"{'File':<50} {'Line%':<8} {'Branch%':<8} {'Status':<10}")
            lines.append("-" * 80)

            for file_path, result in sorted(results.items()):
                status = "PASS"
                if result.is_critical_level:
                    status = "CRITICAL"
                elif result.is_warning_level:
                    status = "WARNING"
                elif not result.passes_thresholds:
                    status = "FAIL"

                # Truncate long file paths
                display_path = file_path
                if len(display_path) > 48:
                    display_path = "..." + display_path[-45:]

                lines.append(
                    f"{display_path:<50} "
                    f"{result.line_coverage:>6.1f}% "
                    f"{result.branch_coverage:>6.1f}% "
                    f"{status:<10}"
                )

            lines.append("-" * 80)

        return "\n".join(lines)

    def generate_json_report(
        self,
        results: dict[str, CoverageResult],
        summary: dict[str, Any],
        trends: dict[str, Any] | None = None
    ) -> str:
        """Generate a JSON coverage report."""

        # Convert CoverageResult objects to dictionaries
        results_dict = {}
        for file_path, result in results.items():
            results_dict[file_path] = {
                'file_path': result.file_path,
                'line_coverage': result.line_coverage,
                'branch_coverage': result.branch_coverage,
                'lines_covered': result.lines_covered,
                'lines_total': result.lines_total,
                'branches_covered': result.branches_covered,
                'branches_total': result.branches_total,
                'missing_lines': result.missing_lines,
                'partial_lines': result.partial_lines,
                'passes_thresholds': result.passes_thresholds,
                'is_warning_level': result.is_warning_level,
                'is_critical_level': result.is_critical_level
            }

        report_data = {
            'summary': summary,
            'thresholds': {
                'total_line_threshold': self.analyzer.thresholds.total_line_threshold,
                'total_branch_threshold': self.analyzer.thresholds.total_branch_threshold,
                'file_line_threshold': self.analyzer.thresholds.file_line_threshold,
                'file_branch_threshold': self.analyzer.thresholds.file_branch_threshold
            },
            'results': results_dict
        }

        if trends:
            report_data['trends'] = trends

        return json.dumps(report_data, indent=2)

    def generate_html_report(
        self,
        results: dict[str, CoverageResult],
        summary: dict[str, Any]
    ) -> str:
        """Generate an HTML coverage report."""

        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Coverage Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .critical {{ color: #d32f2f; }}
                .warning {{ color: #f57c00; }}
                .pass {{ color: #388e3c; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .progress-bar {{
                    width: 100px;
                    height: 20px;
                    background-color: #f0f0f0;
                    border-radius: 10px;
                    overflow: hidden;
                }}
                .progress-fill {{
                    height: 100%;
                    background-color: #4caf50;
                    transition: width 0.3s ease;
                }}
            </style>
        </head>
        <body>
            <h1>Coverage Analysis Report</h1>

            <div class="summary">
                <h2>Summary</h2>
                <p>Total Files: {total_files}</p>
                <p>Overall Line Coverage: {overall_line_coverage:.2f}%</p>
                <p>Overall Branch Coverage: {overall_branch_coverage:.2f}%</p>
                <p>Files Passing Threshold: {files_passing_threshold}/{total_files}</p>
                <p>Threshold Pass Rate: {threshold_pass_rate:.2f}%</p>
            </div>

            <h2>File Details</h2>
            <table>
                <thead>
                    <tr>
                        <th>File</th>
                        <th>Line Coverage</th>
                        <th>Branch Coverage</th>
                        <th>Status</th>
                        <th>Missing Lines</th>
                    </tr>
                </thead>
                <tbody>
                    {file_rows}
                </tbody>
            </table>
        </body>
        </html>
        """

        # Generate file rows
        file_rows = []
        for file_path, result in sorted(results.items()):
            status_class = "pass"
            status_text = "PASS"

            if result.is_critical_level:
                status_class = "critical"
                status_text = "CRITICAL"
            elif result.is_warning_level:
                status_class = "warning"
                status_text = "WARNING"
            elif not result.passes_thresholds:
                status_class = "warning"
                status_text = "FAIL"

            missing_lines_str = ", ".join(map(str, result.missing_lines[:10]))
            if len(result.missing_lines) > 10:
                missing_lines_str += f" ... and {len(result.missing_lines) - 10} more"

            file_rows.append(f"""
                <tr>
                    <td>{file_path}</td>
                    <td>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {result.line_coverage}%"></div>
                        </div>
                        {result.line_coverage:.1f}%
                    </td>
                    <td>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {result.branch_coverage}%"></div>
                        </div>
                        {result.branch_coverage:.1f}%
                    </td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{missing_lines_str}</td>
                </tr>
            """)

        return html_template.format(
            file_rows="".join(file_rows),
            **summary
        )

    def save_report(
        self,
        content: str,
        filename: str,
        output_dir: str = "coverage_reports"
    ) -> str:
        """Save report content to file."""
        output_path = self.analyzer.project_root / output_dir
        output_path.mkdir(exist_ok=True)

        file_path = output_path / filename
        with open(file_path, 'w') as f:
            f.write(content)

        return str(file_path)


# Utility functions for coverage management
def quick_coverage_check(
    source_dir: str = "src",
    test_dir: str = "tests",
    threshold: float = 95.0
) -> str:
    """Quick coverage check with default parameters."""
    pass


def generate_coverage_badge(coverage_percentage: float) -> str:
    """Generate a simple text badge for coverage percentage."""
    if coverage_percentage >= 95:
        return f"![Coverage](https://img.shields.io/badge/coverage-{coverage_percentage:.1f}%25-brightgreen)"
    elif coverage_percentage >= 80:
        return f"![Coverage](https://img.shields.io/badge/coverage-{coverage_percentage:.1f}%25-yellow)"
    else:
        return f"![Coverage](https://img.shields.io/badge/coverage-{coverage_percentage:.1f}%25-red)"
