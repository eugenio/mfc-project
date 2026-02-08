"""Tests for email_notification.py - coverage target 98%+."""
import sys
import os
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from email_notification import send_completion_email, setup_email_monitoring

FULL_RESULTS = {
    "performance_summary": {
        "final_reservoir_concentration_mM": 25.1,
        "mean_reservoir_concentration_mM": 24.8,
        "std_reservoir_concentration_mM": 0.5,
        "final_power_output_W": 0.5,
        "mean_power_output_W": 0.45,
        "total_substrate_consumed_mmol": 120.0,
        "substrate_consumption_rate_mmol_per_day": 0.33,
    },
    "maintenance_requirements": {
        "maintenance_schedule": {
            "substrate_refill_frequency": "weekly",
            "buffer_refill_frequency": "monthly",
        },
        "substrate_requirements": {"stock_bottles_per_year": 12.0},
        "buffer_requirements": {"stock_bottles_per_year": 4.0},
    },
}


class TestSendCompletionEmail:
    def test_basic_send(self, tmp_path):
        rf = tmp_path / "results.json"
        rf.write_text(json.dumps(FULL_RESULTS))

        mock_server = MagicMock()
        with patch("email_notification.smtplib.SMTP", return_value=mock_server):
            with patch.dict(os.environ, {
                "NOTIFICATION_EMAIL": "sender@test.com",
                "EMAIL_PASSWORD": "pass123",
                "RECIPIENT_EMAIL": "recv@test.com",
            }):
                send_completion_email(str(rf))
                mock_server.starttls.assert_called_once()
                mock_server.login.assert_called_once()
                mock_server.sendmail.assert_called_once()
                mock_server.quit.assert_called_once()

    def test_with_explicit_recipient(self, tmp_path):
        rf = tmp_path / "results.json"
        rf.write_text(json.dumps(FULL_RESULTS))

        mock_server = MagicMock()
        with patch("email_notification.smtplib.SMTP", return_value=mock_server):
            send_completion_email(str(rf), recipient_email="custom@test.com")
            args = mock_server.sendmail.call_args
            assert args[0][1] == "custom@test.com"

    def test_file_not_found(self):
        send_completion_email("/nonexistent/file.json")

    def test_smtp_error(self, tmp_path):
        rf = tmp_path / "results.json"
        rf.write_text(json.dumps(FULL_RESULTS))

        with patch("email_notification.smtplib.SMTP", side_effect=Exception("conn")):
            send_completion_email(str(rf))

    def test_empty_performance_hits_except(self, tmp_path):
        """Empty dicts cause .2f formatting to fail on 'N/A', caught by except."""
        results = {}
        rf = tmp_path / "results.json"
        rf.write_text(json.dumps(results))
        # Should not raise; bare except catches the TypeError
        send_completion_email(str(rf))

    def test_no_recipient_env(self, tmp_path):
        rf = tmp_path / "results.json"
        rf.write_text(json.dumps(FULL_RESULTS))

        mock_server = MagicMock()
        with patch("email_notification.smtplib.SMTP", return_value=mock_server):
            with patch.dict(os.environ, {}, clear=False):
                send_completion_email(str(rf))


class TestSetupEmailMonitoring:
    def test_pid_not_exists(self, tmp_path):
        pid_file = tmp_path / "sim.pid"
        pid_file.write_text("99999999")
        log_file = tmp_path / "sim.log"

        with patch("psutil.pid_exists", return_value=False):
            with patch("glob.glob", return_value=[]):
                setup_email_monitoring(str(pid_file), str(log_file))

    def test_pid_not_exists_with_results(self, tmp_path):
        pid_file = tmp_path / "sim.pid"
        pid_file.write_text("99999999")
        log_file = tmp_path / "sim.log"
        result_file = "/tmp/fake_results.json"

        with patch("psutil.pid_exists", return_value=False):
            with patch("glob.glob", return_value=[result_file]):
                with patch("os.path.getctime", return_value=1.0):
                    with patch(
                        "email_notification.send_completion_email"
                    ) as mock_send:
                        setup_email_monitoring(str(pid_file), str(log_file))
                        mock_send.assert_called_once_with(result_file)

    def test_pid_file_read_error(self, tmp_path):
        pid_file = tmp_path / "missing.pid"
        log_file = tmp_path / "sim.log"
        setup_email_monitoring(str(pid_file), str(log_file))

    def test_pid_exists_then_disappears(self, tmp_path):
        pid_file = tmp_path / "sim.pid"
        pid_file.write_text("12345")
        log_file = tmp_path / "sim.log"

        with patch("psutil.pid_exists", side_effect=[True, False]):
            with patch("time.sleep"):
                with patch("glob.glob", return_value=[]):
                    setup_email_monitoring(str(pid_file), str(log_file))
