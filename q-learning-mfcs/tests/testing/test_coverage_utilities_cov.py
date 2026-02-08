import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from testing.coverage_utilities import (
    CoverageThresholds,
    CoverageResult,
    CoverageAnalyzer,
    CoverageReporter,
    run_quick_coverage_check,
    generate_coverage_badge,
    quick_coverage_check,
)


class TestCoverageThresholds:
    def test_defaults(self):
        t = CoverageThresholds()
        assert t.total_line_threshold == 95.0
        assert t.total_branch_threshold == 90.0
        assert t.module_line_threshold == 90.0
        assert t.file_line_threshold == 85.0
        assert t.warning_line_threshold == 70.0
        assert t.critical_line_threshold == 50.0

    def test_custom(self):
        t = CoverageThresholds(total_line_threshold=80.0)
        assert t.total_line_threshold == 80.0


class TestCoverageResult:
    def test_defaults(self):
        r = CoverageResult(file_path="test.py", line_coverage=90.0)
        assert r.file_path == "test.py"
        assert r.branch_coverage == 0.0
        assert r.lines_covered == 0
        assert r.missing_lines == []

    def test_passes_thresholds_true(self):
        r = CoverageResult(file_path="t.py", line_coverage=90.0, branch_coverage=85.0)
        assert r.passes_thresholds is True

    def test_passes_thresholds_false(self):
        r = CoverageResult(file_path="t.py", line_coverage=50.0, branch_coverage=40.0)
        assert r.passes_thresholds is False

    def test_is_warning_level_true(self):
        r = CoverageResult(file_path="t.py", line_coverage=60.0, branch_coverage=50.0)
        assert r.is_warning_level is True

    def test_is_warning_level_false(self):
        r = CoverageResult(file_path="t.py", line_coverage=90.0, branch_coverage=85.0)
        assert r.is_warning_level is False

    def test_is_critical_level_true(self):
        r = CoverageResult(file_path="t.py", line_coverage=40.0, branch_coverage=30.0)
        assert r.is_critical_level is True

    def test_is_critical_level_false(self):
        r = CoverageResult(file_path="t.py", line_coverage=90.0, branch_coverage=85.0)
        assert r.is_critical_level is False


class TestCoverageAnalyzer:
    def test_init_default(self):
        a = CoverageAnalyzer()
        assert a.project_root == Path.cwd()

    def test_init_custom(self, tmp_path):
        a = CoverageAnalyzer(str(tmp_path))
        assert a.project_root == tmp_path

    def test_run_coverage_analysis(self, tmp_path):
        a = CoverageAnalyzer(str(tmp_path))
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
            result = a.run_coverage_analysis()
            assert result["success"] is True

    def test_run_coverage_analysis_fail(self, tmp_path):
        a = CoverageAnalyzer(str(tmp_path))
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="fail")
            result = a.run_coverage_analysis()
            assert result["success"] is False

    def test_run_coverage_analysis_exception(self, tmp_path):
        a = CoverageAnalyzer(str(tmp_path))
        with patch("subprocess.run", side_effect=Exception("fail")):
            result = a.run_coverage_analysis()
            assert result["success"] is False
            assert result["return_code"] == -1

    def test_run_coverage_xml_format(self, tmp_path):
        a = CoverageAnalyzer(str(tmp_path))
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
            result = a.run_coverage_analysis(output_format="xml")
            assert result["success"] is True

    def test_run_coverage_html_format(self, tmp_path):
        a = CoverageAnalyzer(str(tmp_path))
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
            result = a.run_coverage_analysis(output_format="html")
            assert result["success"] is True

    def test_run_coverage_no_branches(self, tmp_path):
        a = CoverageAnalyzer(str(tmp_path))
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
            result = a.run_coverage_analysis(include_branches=False)
            assert result["success"] is True

    def test_parse_coverage_json(self, tmp_path):
        a = CoverageAnalyzer(str(tmp_path))
        data = {
            "files": {
                "src/mod.py": {
                    "summary": {
                        "percent_covered": 85.0,
                        "covered_lines": 85,
                        "num_statements": 100,
                        "percent_covered_display": "85%/80%",
                    },
                    "missing_lines": [10, 20],
                    "excluded_lines": [5],
                }
            }
        }
        (tmp_path / "coverage.json").write_text(json.dumps(data))
        results = a.parse_coverage_json()
        assert "src/mod.py" in results
        assert results["src/mod.py"].line_coverage == 85.0

    def test_parse_coverage_json_no_display_slash(self, tmp_path):
        a = CoverageAnalyzer(str(tmp_path))
        data = {
            "files": {
                "src/mod.py": {
                    "summary": {
                        "percent_covered": 85.0,
                        "covered_lines": 85,
                        "num_statements": 100,
                        "percent_covered_display": "85%",
                    },
                    "missing_lines": [],
                    "excluded_lines": [],
                }
            }
        }
        (tmp_path / "coverage.json").write_text(json.dumps(data))
        results = a.parse_coverage_json()
        assert results["src/mod.py"].branch_coverage == 0.0

    def test_parse_coverage_json_bad_display(self, tmp_path):
        a = CoverageAnalyzer(str(tmp_path))
        data = {
            "files": {
                "src/mod.py": {
                    "summary": {
                        "percent_covered": 85.0,
                        "covered_lines": 85,
                        "num_statements": 100,
                        "percent_covered_display": "bad/data",
                    },
                    "missing_lines": [],
                    "excluded_lines": [],
                }
            }
        }
        (tmp_path / "coverage.json").write_text(json.dumps(data))
        results = a.parse_coverage_json()
        assert "src/mod.py" in results

    def test_parse_coverage_json_not_found(self, tmp_path):
        a = CoverageAnalyzer(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            a.parse_coverage_json()

    def test_parse_coverage_xml(self, tmp_path):
        a = CoverageAnalyzer(str(tmp_path))
        xml_content = """<?xml version="1.0" ?>
<coverage><packages>
<package name="src"><classes>
<class filename="src/mod.py"><lines>
<line number="1" hits="1"/>
<line number="2" hits="0"/>
<line number="3" hits="1" branch="true" condition-coverage="50% (1/2)"/>
</lines></class>
</classes></package>
</packages></coverage>"""
        (tmp_path / "coverage.xml").write_text(xml_content)
        results = a.parse_coverage_xml()
        assert "src/mod.py" in results
        r = results["src/mod.py"]
        assert r.lines_total == 3
        assert r.lines_covered == 2

    def test_parse_coverage_xml_not_found(self, tmp_path):
        a = CoverageAnalyzer(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            a.parse_coverage_xml()

    def test_parse_coverage_xml_bad_condition(self, tmp_path):
        a = CoverageAnalyzer(str(tmp_path))
        xml_content = """<?xml version="1.0" ?>
<coverage><packages>
<package name="src"><classes>
<class filename="src/mod.py"><lines>
<line number="1" hits="1" branch="true" condition-coverage="bad"/>
</lines></class>
</classes></package>
</packages></coverage>"""
        (tmp_path / "coverage.xml").write_text(xml_content)
        results = a.parse_coverage_xml()
        assert results["src/mod.py"].branches_total == 0

    def test_analyze_trends_no_baseline(self):
        a = CoverageAnalyzer()
        current = {"a.py": CoverageResult("a.py", 90.0)}
        trends = a.analyze_coverage_trends(current, None)
        assert trends["trend"] == "no_baseline"

    def test_analyze_trends_improved(self):
        a = CoverageAnalyzer()
        current = {"a.py": CoverageResult("a.py", 90.0, 80.0)}
        previous = {"a.py": CoverageResult("a.py", 70.0, 60.0)}
        trends = a.analyze_coverage_trends(current, previous)
        assert len(trends["improved"]) == 1

    def test_analyze_trends_declined(self):
        a = CoverageAnalyzer()
        current = {"a.py": CoverageResult("a.py", 70.0, 60.0)}
        previous = {"a.py": CoverageResult("a.py", 90.0, 80.0)}
        trends = a.analyze_coverage_trends(current, previous)
        assert len(trends["declined"]) == 1

    def test_analyze_trends_unchanged(self):
        a = CoverageAnalyzer()
        current = {"a.py": CoverageResult("a.py", 90.0, 80.0)}
        previous = {"a.py": CoverageResult("a.py", 90.0, 80.0)}
        trends = a.analyze_coverage_trends(current, previous)
        assert "a.py" in trends["unchanged"]

    def test_analyze_trends_new_removed(self):
        a = CoverageAnalyzer()
        current = {"b.py": CoverageResult("b.py", 90.0)}
        previous = {"a.py": CoverageResult("a.py", 90.0)}
        trends = a.analyze_coverage_trends(current, previous)
        assert "b.py" in trends["new_files"]
        assert "a.py" in trends["removed_files"]

    def test_get_coverage_summary_empty(self):
        a = CoverageAnalyzer()
        s = a.get_coverage_summary({})
        assert s["total_files"] == 0
        assert s["overall_line_coverage"] == 0.0

    def test_get_coverage_summary(self):
        a = CoverageAnalyzer()
        results = {
            "a.py": CoverageResult("a.py", 90.0, 85.0, 90, 100, 17, 20),
            "b.py": CoverageResult("b.py", 40.0, 30.0, 40, 100, 6, 20),
        }
        s = a.get_coverage_summary(results)
        assert s["total_files"] == 2
        assert s["total_lines"] == 200
        assert s["files_passing_threshold"] >= 0
        assert s["files_at_warning_level"] >= 0
        assert s["files_at_critical_level"] >= 0

    def test_get_coverage_summary_zero_branches(self):
        a = CoverageAnalyzer()
        results = {"a.py": CoverageResult("a.py", 90.0, 0.0, 90, 100, 0, 0)}
        s = a.get_coverage_summary(results)
        assert s["overall_branch_coverage"] == 0.0


class TestCoverageReporter:
    def setup_method(self):
        self.analyzer = CoverageAnalyzer()
        self.reporter = CoverageReporter(self.analyzer)
        self.results = {
            "src/good.py": CoverageResult("src/good.py", 95.0, 90.0, 95, 100, 18, 20),
            "src/bad.py": CoverageResult("src/bad.py", 40.0, 30.0, 40, 100, 6, 20, [5, 10, 15]),
            "src/warn.py": CoverageResult("src/warn.py", 65.0, 60.0, 65, 100, 12, 20),
            "src/fail.py": CoverageResult("src/fail.py", 80.0, 70.0, 80, 100, 14, 20),
        }
        self.summary = self.analyzer.get_coverage_summary(self.results)

    def test_text_report(self):
        text = self.reporter.generate_text_report(self.results, self.summary)
        assert "COVERAGE ANALYSIS REPORT" in text
        assert "SUMMARY" in text
        assert "DETAILED RESULTS" in text

    def test_text_report_no_details(self):
        text = self.reporter.generate_text_report(self.results, self.summary, show_details=False)
        assert "SUMMARY" in text
        assert "DETAILED RESULTS" not in text

    def test_text_report_long_path(self):
        long_results = {"a" * 60 + ".py": CoverageResult("a" * 60 + ".py", 90.0, 85.0)}
        summary = self.analyzer.get_coverage_summary(long_results)
        text = self.reporter.generate_text_report(long_results, summary)
        assert "..." in text

    def test_json_report(self):
        j = self.reporter.generate_json_report(self.results, self.summary)
        data = json.loads(j)
        assert "summary" in data
        assert "results" in data
        assert "thresholds" in data

    def test_json_report_with_trends(self):
        trends = {"improved": [], "declined": []}
        j = self.reporter.generate_json_report(self.results, self.summary, trends)
        data = json.loads(j)
        assert "trends" in data

    def test_html_report(self):
        html = self.reporter.generate_html_report(self.results, self.summary)
        assert "Coverage Analysis Report" in html
        assert "src/good.py" in html
        assert "CRITICAL" in html

    def test_html_report_many_missing(self):
        results = {"x.py": CoverageResult("x.py", 50.0, 40.0, 50, 100, 8, 20, list(range(50)))}
        summary = self.analyzer.get_coverage_summary(results)
        html = self.reporter.generate_html_report(results, summary)
        assert "more" in html

    def test_save_report(self, tmp_path):
        analyzer = CoverageAnalyzer(str(tmp_path))
        reporter = CoverageReporter(analyzer)
        path = reporter.save_report("test content", "test.txt")
        assert os.path.exists(path)
        with open(path) as f:
            assert f.read() == "test content"


class TestRunQuickCoverageCheck:
    def test_success(self, tmp_path):
        with patch.object(CoverageAnalyzer, "run_coverage_analysis", return_value={"success": True}):
            mock_results = {"a.py": CoverageResult("a.py", 96.0, 90.0, 96, 100, 18, 20)}
            with patch.object(CoverageAnalyzer, "parse_coverage_json", return_value=mock_results):
                with patch.object(CoverageAnalyzer, "get_coverage_summary", return_value={"overall_line_coverage": 96.0}):
                    result = run_quick_coverage_check("src", "tests")
                    assert result is True

    def test_low_coverage(self):
        with patch.object(CoverageAnalyzer, "run_coverage_analysis", return_value={"success": True}):
            with patch.object(CoverageAnalyzer, "parse_coverage_json", return_value={
                "a.py": CoverageResult("a.py", 50.0, 40.0, 50, 100, 8, 20)
            }):
                result = run_quick_coverage_check("src", "tests")
                assert result is False

    def test_analysis_failure(self):
        with patch.object(CoverageAnalyzer, "run_coverage_analysis", return_value={"success": False, "stderr": "fail"}):
            result = run_quick_coverage_check("src", "tests")
            assert result is False

    def test_parse_exception(self):
        with patch.object(CoverageAnalyzer, "run_coverage_analysis", return_value={"success": True}):
            with patch.object(CoverageAnalyzer, "parse_coverage_json", side_effect=Exception("parse fail")):
                result = run_quick_coverage_check("src", "tests")
                assert result is False


class TestGenerateCoverageBadge:
    def test_high_coverage(self):
        badge = generate_coverage_badge(96.0)
        assert "brightgreen" in badge

    def test_medium_coverage(self):
        badge = generate_coverage_badge(85.0)
        assert "yellow" in badge

    def test_low_coverage(self):
        badge = generate_coverage_badge(50.0)
        assert "red" in badge

    def test_boundary_95(self):
        badge = generate_coverage_badge(95.0)
        assert "brightgreen" in badge

    def test_boundary_80(self):
        badge = generate_coverage_badge(80.0)
        assert "yellow" in badge


class TestQuickCoverageCheck:
    def test_returns_none(self):
        result = quick_coverage_check()
        assert result is None
