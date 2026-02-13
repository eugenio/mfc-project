#!/bin/bash
# Setup email notifications for MFC simulation completion
#
# Created: 2025-07-26

echo "📧 MFC Simulation Email Notification Setup"
echo "=========================================="
echo ""

# Check if psutil is installed
if ! python -c "import psutil" 2>/dev/null; then
    echo "📦 Installing required psutil package..."
    pip install psutil --user
fi

echo "To enable email notifications when the simulation completes,"
echo "you need to set up your email credentials as environment variables:"
echo ""

# Get user input
read -p "📧 Enter your sender email (Gmail/etc): " SENDER_EMAIL
echo ""

echo "🔐 For Gmail, you'll need an 'App Password' (not your regular password):"
echo "   1. Go to https://myaccount.google.com/security"
echo "   2. Enable 2-Step Verification if not already enabled"
echo "   3. Go to 'App passwords' and generate a new password"
echo "   4. Use this 16-character app password below"
echo ""

read -s -p "🔐 Enter your app password: " APP_PASSWORD
echo ""
echo ""

read -p "📨 Enter recipient email address: " RECIPIENT_EMAIL
echo ""

# Add to .bashrc
echo "# MFC Simulation Email Notifications" >> ~/.bashrc
echo "export NOTIFICATION_EMAIL='$SENDER_EMAIL'" >> ~/.bashrc
echo "export EMAIL_PASSWORD='$APP_PASSWORD'" >> ~/.bashrc
echo "export RECIPIENT_EMAIL='$RECIPIENT_EMAIL'" >> ~/.bashrc

# Export for current session
export NOTIFICATION_EMAIL="$SENDER_EMAIL"
export EMAIL_PASSWORD="$APP_PASSWORD"
export RECIPIENT_EMAIL="$RECIPIENT_EMAIL"

echo "✅ Email configuration saved to ~/.bashrc"
echo ""
echo "📧 Test email notification with:"
echo "   python email_notification.py"
echo ""
echo "🔄 Reload your shell or run:"
echo "   source ~/.bashrc"
echo ""
echo "🎉 Email notifications are now configured!"
echo "   The running simulation will automatically send you an email when complete."