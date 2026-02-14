"""Coverage boost tests for email_notification.py."""
import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from email_notification import send_completion_email, setup_email_monitoring


@pytest.mark.coverage_extra
class TestSendCompletionEmail:
    def test_send_email_success(self, tmp_path):
        results = {
            "performance_summary": {
                "final_reservoir_concentration_mM": 25.0,
                "mean_reservoir_concentration_mM": 24.5,
                "std_reservoir_concentration_mM": 0.5,
                "final_power_output_W": 1.5,
                "mean_power_output_W": 1.3,
                "total_substrate_consumed_mmol": 1000.0,
                "substrate_consumption_rate_mmol_per_day": 2.74,
            },
            "maintenance_requirements": {
                "maintenance_schedule": {
                    "substrate_refill_frequency": "weekly",
                    "buffer_refill_frequency": "monthly",
                },
                "substrate_requirements": {"stock_bottles_per_year": 12},
                "buffer_requirements": {"stock_bottles_per_year": 4},
            },
        }
        f = tmp_path / "results.json"
        f.write_text(json.dumps(results))
        with patch("email_notification.smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server
            send_completion_email(str(f), "test@test.com")

    def test_send_email_missing_file(self, tmp_path):
        send_completion_email(str(tmp_path / "nope.json"))

    def test_send_email_missing_keys(self, tmp_path):
        f = tmp_path / "results.json"
        f.write_text(json.dumps({"empty": True}))
        with patch("email_notification.smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server
            send_completion_email(str(f), "test@test.com")


@pytest.mark.coverage_extra
class TestSetupEmailMonitoring:
    def test_monitoring_pid_not_exists(self, tmp_path):
        pid_file = tmp_path / "test.pid"
        pid_file.write_text("99999999")
        log_file = tmp_path / "test.log"
        with patch("psutil.pid_exists", return_value=False), \
             patch("glob.glob", return_value=[]):
            setup_email_monitoring(str(pid_file), str(log_file))

    def test_monitoring_pid_with_results(self, tmp_path):
        pid_file = tmp_path / "test.pid"
        pid_file.write_text("99999999")
        results_file = tmp_path / "results.json"
        results_file.write_text(json.dumps({"test": True}))
        log_file = tmp_path / "test.log"
        with patch("psutil.pid_exists", return_value=False), \
             patch("glob.glob", return_value=[str(results_file)]), \
             patch("email_notification.send_completion_email") as mock_send:
            setup_email_monitoring(str(pid_file), str(log_file))
            mock_send.assert_called_once()

    def test_monitoring_bad_pid(self, tmp_path):
        pid_file = tmp_path / "test.pid"
        pid_file.write_text("not_a_number")
        log_file = tmp_path / "test.log"
        setup_email_monitoring(str(pid_file), str(log_file))
