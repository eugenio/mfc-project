#!/usr/bin/env python3
"""
Email notification system for MFC simulation completion.

Created: 2025-07-26
"""

import smtplib
import os
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
import sys


def send_completion_email(results_file, recipient_email=None):
    """
    Send email notification when simulation completes.
    
    Args:
        results_file: Path to simulation results JSON file
        recipient_email: Email address to send notification to
    """

    # Email configuration - you'll need to set these
    SMTP_SERVER = "smtp.gmail.com"  # Change for your email provider
    SMTP_PORT = 587
    SENDER_EMAIL = os.getenv("NOTIFICATION_EMAIL", "your-email@gmail.com")
    SENDER_PASSWORD = os.getenv("EMAIL_PASSWORD", "your-app-password")

    if not recipient_email:
        recipient_email = os.getenv("RECIPIENT_EMAIL", "your-email@gmail.com")

    try:
        # Load simulation results
        with open(results_file, 'r') as f:
            results = json.load(f)

        # Create email content
        subject = "🎉 MFC 1-Year Simulation Complete!"

        # Extract key metrics
        perf = results.get("performance_summary", {})
        maint = results.get("maintenance_requirements", {})

        body = f"""
MFC 1-Year Continuous Operation Simulation Complete!

SIMULATION SUMMARY:
==================
Duration: 8,784 hours (365 days)
Completion Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Target Concentration: 25.0 mM

PERFORMANCE RESULTS:
===================
Final Reservoir Concentration: {perf.get('final_reservoir_concentration_mM', 'N/A'):.2f} mM
Mean Concentration: {perf.get('mean_reservoir_concentration_mM', 'N/A'):.2f} ± {perf.get('std_reservoir_concentration_mM', 'N/A'):.2f} mM
Final Power Output: {perf.get('final_power_output_W', 'N/A'):.2f} W
Mean Power Output: {perf.get('mean_power_output_W', 'N/A'):.2f} W

SUBSTRATE CONSUMPTION:
=====================
Total Consumed: {perf.get('total_substrate_consumed_mmol', 'N/A'):.2f} mmol
Daily Rate: {perf.get('substrate_consumption_rate_mmol_per_day', 'N/A'):.2f} mmol/day

MAINTENANCE REQUIREMENTS:
========================
Substrate Refill: {maint.get('maintenance_schedule', {}).get('substrate_refill_frequency', 'N/A')}
Buffer Refill: {maint.get('maintenance_schedule', {}).get('buffer_refill_frequency', 'N/A')}
Stock Bottles/Year: {maint.get('substrate_requirements', {}).get('stock_bottles_per_year', 'N/A'):.0f} substrate, {maint.get('buffer_requirements', {}).get('stock_bottles_per_year', 'N/A'):.0f} buffer

Results saved to: {results_file}

Best regards,
MFC Simulation System
        """

        # Create email message
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = recipient_email
        msg['Subject'] = subject

        # Add body to email
        msg.attach(MIMEText(body, 'plain'))

        # Attach results file
        with open(results_file, "rb") as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())

        encoders.encode_base64(part)
        part.add_header(
            'Content-Disposition',
            f'attachment; filename= {os.path.basename(results_file)}'
        )
        msg.attach(part)

        # Send email
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        text = msg.as_string()
        server.sendmail(SENDER_EMAIL, recipient_email, text)
        server.quit()

        print(f"✅ Email notification sent successfully to {recipient_email}")

    except Exception as e:
        print(f"❌ Failed to send email notification: {e}")
        print("💡 Make sure to set EMAIL environment variables:")
        print("   export NOTIFICATION_EMAIL='your-email@gmail.com'")
        print("   export EMAIL_PASSWORD='your-app-password'")
        print("   export RECIPIENT_EMAIL='recipient@gmail.com'")


def setup_email_monitoring(pid_file, log_file):
    """
    Monitor simulation process and send email when complete.
    
    Args:
        pid_file: Path to PID file
        log_file: Path to log file
    """
    import time
    import psutil

    print("📧 Email monitoring setup...")
    print("💡 To configure email notifications:")
    print("   export NOTIFICATION_EMAIL='your-sender-email@gmail.com'")
    print("   export EMAIL_PASSWORD='your-app-password'")
    print("   export RECIPIENT_EMAIL='your-recipient-email@gmail.com'")
    print()

    try:
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())

        print(f"🔍 Monitoring process {pid}...")

        # Monitor process
        while True:
            if not psutil.pid_exists(pid):
                print("🎉 Simulation process completed!")

                # Look for results file
                import glob
                results_pattern = "/home/uge/mfc-project/q-learning-mfcs/data/simulation_data/1year_optimized_*/1year_simulation_results_*.json"
                results_files = glob.glob(results_pattern)

                if results_files:
                    latest_results = max(results_files, key=os.path.getctime)
                    print(f"📊 Found results file: {latest_results}")
                    send_completion_email(latest_results)
                else:
                    print("⚠️  No results file found. Sending basic notification.")
                    # Could send basic completion email here

                break

            time.sleep(300)  # Check every 5 minutes

    except Exception as e:
        print(f"❌ Email monitoring failed: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Send email for specific results file
        send_completion_email(sys.argv[1])
    else:
        # Monitor simulation
        pid_file = "/home/uge/mfc-project/q-learning-mfcs/data/logs/1year_simulation.pid"
        log_file = "/home/uge/mfc-project/q-learning-mfcs/data/logs/1year_simulation_20250726_065416.log"
        setup_email_monitoring(pid_file, log_file)
