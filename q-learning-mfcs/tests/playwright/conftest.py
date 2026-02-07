"""Playwright E2E test configuration."""

import datetime
import json
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

# Add this directory to sys.path for imports
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import pytest  # noqa: E402
from playwright.sync_api import Page  # noqa: E402

STREAMLIT_PORT = 8502
STREAMLIT_URL = f"http://localhost:{STREAMLIT_PORT}"
_THIS_PATH = Path(__file__).resolve().parent
APP_FILE = str(
    _THIS_PATH / ".." / ".." / "src" / "gui" / "enhanced_main_app.py",
)
SCREENSHOT_DIR = str(_THIS_PATH / "screenshots")

PAGE_LABELS = [
    "Dashboard",
    "Electrode System",
    "Cell Configuration",
    "Physics Simulation",
    "ML Optimization",
    "GSM Integration",
    "Literature Validation",
    "Performance Monitor",
    "Configuration",
]

# Map plain labels to emoji-prefixed radio labels
_LABEL_TO_RADIO = {
    "Dashboard": "\U0001f3e0 Dashboard",
    "Electrode System": "\U0001f50b Electrode System",
    "Cell Configuration": "\U0001f3d7\ufe0f Cell Configuration",
    "Physics Simulation": "\u2697\ufe0f Physics Simulation",
    "ML Optimization": "\U0001f9e0 ML Optimization",
    "GSM Integration": "\U0001f9ec GSM Integration",
    "Literature Validation": "\U0001f4da Literature Validation",
    "Performance Monitor": "\U0001f4ca Performance Monitor",
    "Configuration": "\u2699\ufe0f Configuration",
}


def _port_in_use(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("localhost", port)) == 0


def _wait_for_server(port: int, t: float = 30.0) -> bool:
    start = time.time()
    while time.time() - start < t:
        if _port_in_use(port):
            return True
        time.sleep(0.5)
    return False


@pytest.fixture(scope="session")
def streamlit_server() -> Generator[str, None, None]:
    """Start/stop Streamlit server for E2E tests."""
    if _port_in_use(STREAMLIT_PORT):
        yield STREAMLIT_URL
        return

    proc = subprocess.Popen(  # noqa: S603
        [
            sys.executable, "-m", "streamlit", "run",
            APP_FILE,
            "--server.port", str(STREAMLIT_PORT),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(Path(APP_FILE).parent),
    )

    if not _wait_for_server(STREAMLIT_PORT):
        proc.kill()
        msg = f"Server failed on port {STREAMLIT_PORT}"
        raise RuntimeError(msg)

    yield STREAMLIT_URL

    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture
def streamlit_page(
    streamlit_server: str, page: Page,
) -> Page:
    """Navigate to app and wait for load."""
    page.goto(streamlit_server)
    page.wait_for_selector(
        '[data-testid="stAppViewContainer"]',
        timeout=15000,
    )
    # Wait for sidebar radio widget to render
    page.wait_for_selector(
        '[data-testid="stRadio"]',
        timeout=10000,
    )
    return page


def click_radio_page(page: Page, label: str) -> None:
    """Click a page label in the sidebar radio widget.

    Uses exact emoji-prefixed labels and stRadio testid
    to avoid strict mode violations.
    """
    radio_label = _LABEL_TO_RADIO.get(label, label)
    radio = page.locator('[data-testid="stRadio"]')
    radio.get_by_text(radio_label, exact=True).click()
    page.wait_for_timeout(1500)


# --- Report Generation ---

_REPORT_DIR = Path(__file__).resolve().parent / ".." / ".." / ".." / "tmp"


_pw_results: list[dict[str, Any]] = []


def pytest_runtest_logreport(report: Any) -> None:  # noqa: ANN401
    """Collect test results for report."""
    if report.when != "call":
        return
    _pw_results.append({
        "nodeid": report.nodeid,
        "outcome": report.outcome,
        "duration": round(report.duration, 2),
        "longrepr": (
            str(report.longrepr)
            if report.failed
            else None
        ),
    })


def pytest_sessionfinish(
    session: Any,  # noqa: ANN401, ARG001
    exitstatus: int,  # noqa: ARG001
) -> None:
    """Generate JSON and Markdown E2E reports."""
    results = _pw_results
    if not results:
        return

    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now(datetime.timezone.utc)
    ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    passed = sum(
        1 for r in results if r["outcome"] == "passed"
    )
    failed = sum(
        1 for r in results if r["outcome"] == "failed"
    )
    xfailed = sum(
        1 for r in results if r["outcome"] == "skipped"
    )
    total = len(results)
    duration = sum(r["duration"] for r in results)

    # JSON report
    report_data = {
        "timestamp": ts,
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "xfailed": xfailed,
            "duration_s": round(duration, 2),
        },
        "results": results,
    }
    json_path = _REPORT_DIR / "ui-test-report.json"
    json_path.write_text(json.dumps(report_data, indent=2))

    # Markdown report
    md_path = _REPORT_DIR / "ui-test-report.md"
    status = "PASS" if failed == 0 else "FAIL"
    lines = [
        f"# E2E UI Test Report - {status}",
        "",
        f"**Date:** {ts}",
        f"**Duration:** {duration:.1f}s",
        "",
        "## Summary",
        "",
        "| Metric | Count |",
        "|--------|-------|",
        f"| Total  | {total} |",
        f"| Passed | {passed} |",
        f"| Failed | {failed} |",
        f"| XFail  | {xfailed} |",
        "",
    ]
    if failed > 0:
        lines.extend([
            "## Failures",
            "",
        ])
        for r in results:
            if r["outcome"] == "failed":
                lines.append(f"### {r['nodeid']}")
                lines.append("```")
                lines.append(
                    r["longrepr"][:500]
                    if r["longrepr"]
                    else "No details",
                )
                lines.append("```")
                lines.append("")

    lines.extend([
        "## All Results",
        "",
        "| Test | Status | Duration |",
        "|------|--------|----------|",
    ])
    for r in results:
        icon = {
            "passed": "PASS",
            "failed": "FAIL",
            "skipped": "XFAIL",
        }.get(r["outcome"], r["outcome"])
        lines.append(
            f"| {r['nodeid'].split('::')[-1]} "
            f"| {icon} | {r['duration']}s |",
        )

    md_path.write_text("\n".join(lines) + "\n")
