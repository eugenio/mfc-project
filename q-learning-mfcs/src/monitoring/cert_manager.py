#!/usr/bin/env python3
"""Certificate Management Utility for MFC Monitoring System
Handles SSL certificate lifecycle including generation, renewal, and monitoring.
"""

from __future__ import annotations

import argparse
import json
import logging
import smtplib
import subprocess
import sys
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# ruff: noqa: E402
from monitoring.ssl_config import (
    CertificateManager,
    SSLConfig,
    initialize_ssl_infrastructure,
    load_ssl_config,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class EnhancedCertificateManager(CertificateManager):
    """Enhanced certificate manager with additional features."""

    def __init__(self, config: SSLConfig) -> None:
        super().__init__(config)
        self.notification_config = self._load_notification_config()

    def _load_notification_config(self) -> dict:
        """Load email notification configuration."""
        config_file = Path("/etc/mfc/notification-config.json")
        default_config = {
            "enabled": False,
            "smtp_server": "localhost",
            "smtp_port": 587,
            "smtp_username": "",
            "smtp_password": "",
            "from_email": "mfc-system@localhost",
            "to_emails": [],
            "use_tls": True,
        }

        if config_file.exists():
            try:
                with open(config_file) as f:
                    return {**default_config, **json.load(f)}
            except Exception as e:
                logger.warning(f"Failed to load notification config: {e}")

        return default_config

    def check_certificate_expiry(self) -> tuple[bool, datetime | None, int]:
        """Check certificate expiry and return detailed information
        Returns: (needs_renewal, expiry_date, days_until_expiry).
        """
        is_valid, expiry_date = self.check_certificate_validity()

        if not expiry_date:
            return True, None, 0

        days_until_expiry = (expiry_date - datetime.now()).days
        needs_renewal = days_until_expiry <= self.config.renewal_days_before

        return needs_renewal, expiry_date, days_until_expiry

    def send_notification(
        self,
        subject: str,
        message: str,
        is_critical: bool = False,
    ) -> None:
        """Send email notification about certificate status."""
        if not self.notification_config.get("enabled", False):
            logger.info("Email notifications disabled")
            return

        if not self.notification_config.get("to_emails"):
            logger.warning("No notification email addresses configured")
            return

        try:
            # Create message
            msg = MIMEMultipart()
            msg["From"] = self.notification_config["from_email"]
            msg["To"] = ", ".join(self.notification_config["to_emails"])
            msg["Subject"] = f"[MFC-SSL{' CRITICAL' if is_critical else ''}] {subject}"

            # Add body
            body = f"""
MFC Monitoring System - SSL Certificate Notification

{message}

System Information:
- Domain: {self.config.domain}
- Certificate File: {self.config.cert_file}
- Timestamp: {datetime.now().isoformat()}

This is an automated message from the MFC Monitoring System.
"""

            msg.attach(MIMEText(body, "plain"))

            # Send email
            server = smtplib.SMTP(
                self.notification_config["smtp_server"],
                self.notification_config["smtp_port"],
            )

            if self.notification_config.get("use_tls", True):
                server.starttls()

            if self.notification_config.get("smtp_username"):
                server.login(
                    self.notification_config["smtp_username"],
                    self.notification_config["smtp_password"],
                )

            server.send_message(msg)
            server.quit()

            logger.info(f"Notification sent: {subject}")

        except Exception as e:
            logger.exception(f"Failed to send notification: {e}")

    def monitor_certificate(self) -> dict[str, Any]:
        """Monitor certificate status and return comprehensive report."""
        report: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "domain": self.config.domain,
            "certificate_exists": False,
            "certificate_valid": False,
            "expiry_date": None,
            "days_until_expiry": None,
            "needs_renewal": False,
            "auto_renewal_configured": False,
            "recommendations": [],
        }

        # Check if certificate exists
        report["certificate_exists"] = self.check_certificate_exists()

        if not report["certificate_exists"]:
            report["recommendations"].append(
                "Certificate files not found - run certificate generation",
            )
            return report

        # Check certificate validity
        needs_renewal, expiry_date, days_until_expiry = self.check_certificate_expiry()

        report["certificate_valid"] = not needs_renewal or days_until_expiry > 0
        report["expiry_date"] = expiry_date.isoformat() if expiry_date else None
        report["days_until_expiry"] = days_until_expiry
        report["needs_renewal"] = needs_renewal

        # Check auto-renewal configuration
        report["auto_renewal_configured"] = self._check_cron_job_exists()

        # Generate recommendations
        if needs_renewal:
            if days_until_expiry <= 7:
                report["recommendations"].append(
                    "URGENT: Certificate expires in less than 7 days - renew immediately",
                )
            elif days_until_expiry <= self.config.renewal_days_before:
                report["recommendations"].append("Certificate should be renewed soon")

        if not report["auto_renewal_configured"]:
            report["recommendations"].append(
                "Auto-renewal not configured - setup cron job",
            )

        if report["certificate_valid"] and days_until_expiry > 30:
            report["recommendations"].append("Certificate is healthy")

        return report

    def _check_cron_job_exists(self) -> bool:
        """Check if auto-renewal cron job exists."""
        try:
            # Use python-crontab if available, otherwise fallback to command
            try:
                from crontab import CronTab

                cron = CronTab(user=True)

                for job in cron:
                    if job.command and "mfc-cert-renewal" in job.command:
                        return True

                return False

            except ImportError:
                # Fallback to command line
                result = subprocess.run(
                    ["crontab", "-l"],
                    check=False,
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0:
                    return "mfc-cert-renewal" in result.stdout

                return False

        except Exception as e:
            logger.warning(f"Could not check cron job: {e}")
            return False

    def renew_certificate_if_needed(self) -> tuple[bool, str]:
        """Renew certificate if needed."""
        needs_renewal, expiry_date, days_until_expiry = self.check_certificate_expiry()

        if not needs_renewal:
            return True, f"Certificate valid for {days_until_expiry} more days"

        logger.info(f"Certificate needs renewal (expires in {days_until_expiry} days)")

        if self.config.use_letsencrypt:
            # Use certbot to renew
            try:
                result = subprocess.run(
                    ["certbot", "renew", "--quiet"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )

                if result.returncode == 0:
                    message = (
                        f"Certificate renewed successfully for {self.config.domain}"
                    )
                    self.send_notification("Certificate Renewed", message)
                    return True, message
                error_msg = f"Certificate renewal failed: {result.stderr}"
                self.send_notification(
                    "Certificate Renewal Failed",
                    error_msg,
                    is_critical=True,
                )
                return False, error_msg

            except subprocess.TimeoutExpired:
                error_msg = "Certificate renewal timed out"
                self.send_notification(
                    "Certificate Renewal Failed",
                    error_msg,
                    is_critical=True,
                )
                return False, error_msg
            except Exception as e:
                error_msg = f"Certificate renewal error: {e!s}"
                self.send_notification(
                    "Certificate Renewal Failed",
                    error_msg,
                    is_critical=True,
                )
                return False, error_msg
        # Generate new self-signed certificate
        elif self.generate_self_signed_certificate():
            message = f"Self-signed certificate regenerated for {self.config.domain}"
            self.send_notification("Certificate Regenerated", message)
            return True, message
        else:
            error_msg = "Failed to regenerate self-signed certificate"
            self.send_notification(
                "Certificate Generation Failed",
                error_msg,
                is_critical=True,
            )
            return False, error_msg

    def setup_monitoring_cron(self) -> bool:
        """Setup cron job for certificate monitoring and renewal."""
        try:
            script_path = Path(__file__).resolve()

            # Create cron job that runs daily at 2 AM
            cron_command = (
                f"0 2 * * * {sys.executable} {script_path} --monitor --renew-if-needed"
            )

            # Add cron job
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    f'(crontab -l 2>/dev/null | grep -v "mfc-cert-renewal"; echo "{cron_command} # mfc-cert-renewal") | crontab -',
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                logger.info("Certificate monitoring cron job installed")
                return True
            logger.error(f"Failed to install cron job: {result.stderr}")
            return False

        except Exception as e:
            logger.exception(f"Error setting up monitoring cron: {e}")
            return False

    def remove_monitoring_cron(self) -> bool:
        """Remove certificate monitoring cron job."""
        try:
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'crontab -l 2>/dev/null | grep -v "mfc-cert-renewal" | crontab -',
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                logger.info("Certificate monitoring cron job removed")
                return True
            logger.error(f"Failed to remove cron job: {result.stderr}")
            return False

        except Exception as e:
            logger.exception(f"Error removing monitoring cron: {e}")
            return False


def print_certificate_report(report: dict) -> None:
    """Print formatted certificate report."""
    # Status indicators
    "✅" if report["certificate_exists"] else "❌"
    "✅" if report["certificate_valid"] else "❌"
    "✅" if report["auto_renewal_configured"] else "❌"

    if report["expiry_date"]:
        if report["needs_renewal"]:
            pass
        else:
            pass

    # Recommendations
    if report["recommendations"]:
        for _i, _rec in enumerate(report["recommendations"], 1):
            pass


def setup_notification_config() -> bool | None:
    """Setup email notification configuration interactively."""
    config = {
        "enabled": False,
        "smtp_server": "localhost",
        "smtp_port": 587,
        "smtp_username": "",
        "smtp_password": "",
        "from_email": "mfc-system@localhost",
        "to_emails": [],
        "use_tls": True,
    }

    enable = input("Enable email notifications? (y/N): ").strip().lower()
    if enable not in ["y", "yes"]:
        config["enabled"] = False
    else:
        config["enabled"] = True

        config["smtp_server"] = (
            input("SMTP server (default: localhost): ").strip() or "localhost"
        )

        port_input = input("SMTP port (default: 587): ").strip()
        config["smtp_port"] = int(port_input) if port_input else 587

        config["smtp_username"] = input("SMTP username (optional): ").strip()
        if config["smtp_username"]:
            import getpass

            config["smtp_password"] = getpass.getpass("SMTP password: ")

        config["from_email"] = (
            input("From email (default: mfc-system@localhost): ").strip()
            or "mfc-system@localhost"
        )

        to_emails = input("To emails (comma-separated): ").strip()
        config["to_emails"] = [
            email.strip() for email in to_emails.split(",") if email.strip()
        ]

        use_tls = input("Use TLS? (Y/n): ").strip().lower()
        config["use_tls"] = use_tls not in ["n", "no"]

    # Save configuration
    config_file = Path("/etc/mfc/notification-config.json")
    config_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        return True
    except Exception:
        return False


def main() -> None:
    """Main entry point for certificate management utility."""
    parser = argparse.ArgumentParser(description="MFC Certificate Management Utility")

    # Main actions
    parser.add_argument(
        "--init",
        action="store_true",
        help="Initialize SSL certificates",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Monitor certificate status",
    )
    parser.add_argument(
        "--renew",
        action="store_true",
        help="Force certificate renewal",
    )
    parser.add_argument(
        "--renew-if-needed",
        action="store_true",
        help="Renew certificate if needed",
    )

    # Cron management
    parser.add_argument(
        "--setup-cron",
        action="store_true",
        help="Setup monitoring cron job",
    )
    parser.add_argument(
        "--remove-cron",
        action="store_true",
        help="Remove monitoring cron job",
    )

    # Configuration
    parser.add_argument(
        "--setup-notifications",
        action="store_true",
        help="Setup email notifications",
    )
    parser.add_argument(
        "--test-notifications",
        action="store_true",
        help="Test email notifications",
    )

    # Options
    parser.add_argument("--domain", help="Domain name")
    parser.add_argument("--email", help="Email for Let's Encrypt")
    parser.add_argument(
        "--staging",
        action="store_true",
        help="Use Let's Encrypt staging",
    )
    parser.add_argument("--quiet", action="store_true", help="Quiet mode (less output)")

    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Load SSL configuration
    ssl_config = load_ssl_config()

    # Override with command line arguments
    if args.domain:
        ssl_config.domain = args.domain
    if args.email:
        ssl_config.email = args.email
    if args.staging:
        ssl_config.staging = True

    # Create enhanced certificate manager
    cert_manager = EnhancedCertificateManager(ssl_config)

    if args.setup_notifications:
        setup_notification_config()
        return

    if args.test_notifications:
        cert_manager.send_notification(
            "Test Notification",
            "This is a test notification from the MFC Certificate Manager.",
        )
        return

    if args.init:
        logger.info("Initializing SSL certificates...")
        success, updated_config = initialize_ssl_infrastructure(ssl_config)
        if success:
            pass
        else:
            sys.exit(1)
        return

    if args.setup_cron:
        if cert_manager.setup_monitoring_cron():
            pass
        else:
            sys.exit(1)
        return

    if args.remove_cron:
        if cert_manager.remove_monitoring_cron():
            pass
        else:
            sys.exit(1)
        return

    if args.monitor:
        report = cert_manager.monitor_certificate()

        if not args.quiet:
            print_certificate_report(report)

        # Exit with error code if certificate needs attention
        if not report["certificate_exists"] or report["needs_renewal"]:
            if (
                report["days_until_expiry"] is not None
                and report["days_until_expiry"] <= 7
            ):
                sys.exit(2)  # Critical - expires soon
            else:
                sys.exit(1)  # Warning - needs attention

        return

    if args.renew or args.renew_if_needed:
        if args.renew_if_needed:
            # Only renew if needed
            needs_renewal, _, days_until_expiry = (
                cert_manager.check_certificate_expiry()
            )
            if not needs_renewal:
                if not args.quiet:
                    pass
                return

        success, message = cert_manager.renew_certificate_if_needed()

        if success:
            pass
        else:
            sys.exit(1)

        return

    # Default action - show status
    report = cert_manager.monitor_certificate()
    print_certificate_report(report)


if __name__ == "__main__":
    main()
