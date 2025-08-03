#!/usr/bin/env python3
"""
Alert Configuration UI Component
================================

Streamlit-based UI for configuring and managing alerts in the MFC monitoring system.
Provides an intuitive interface for setting thresholds, viewing alert history,
and managing notification preferences.

Created: 2025-08-01
Author: MFC Research Team
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.append(str(Path(__file__).parent.parent))

from monitoring.alert_management import (
    Alert,
    AlertManager,
    AlertThreshold,
    EscalationRule,
)


class AlertConfigurationUI:
    """Streamlit UI component for alert configuration."""

    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager

        # MFC parameters with metadata
        self.parameter_info = {
            "power_density": {
                "name": "Power Density",
                "unit": "W/m²",
                "typical_range": (0.1, 2.0),
                "critical_range": (0.05, 3.0),
                "description": "Electrical power output per unit area"
            },
            "substrate_concentration": {
                "name": "Substrate Concentration",
                "unit": "mM",
                "typical_range": (5.0, 50.0),
                "critical_range": (2.0, 70.0),
                "description": "Fuel concentration in the anode chamber"
            },
            "pH": {
                "name": "pH",
                "unit": "",
                "typical_range": (6.5, 7.5),
                "critical_range": (5.5, 8.5),
                "description": "Acidity/alkalinity of the solution"
            },
            "temperature": {
                "name": "Temperature",
                "unit": "°C",
                "typical_range": (25.0, 35.0),
                "critical_range": (20.0, 40.0),
                "description": "Operating temperature"
            },
            "biofilm_thickness": {
                "name": "Biofilm Thickness",
                "unit": "μm",
                "typical_range": (50.0, 200.0),
                "critical_range": (20.0, 300.0),
                "description": "Thickness of the biofilm on the anode"
            },
            "conductivity": {
                "name": "Conductivity",
                "unit": "S/m",
                "typical_range": (1.0, 10.0),
                "critical_range": (0.5, 15.0),
                "description": "Electrical conductivity of the solution"
            },
            "dissolved_oxygen": {
                "name": "Dissolved Oxygen",
                "unit": "mg/L",
                "typical_range": (0.0, 2.0),
                "critical_range": (0.0, 5.0),
                "description": "Oxygen concentration in the anode (should be low)"
            },
            "coulombic_efficiency": {
                "name": "Coulombic Efficiency",
                "unit": "%",
                "typical_range": (40.0, 80.0),
                "critical_range": (20.0, 95.0),
                "description": "Efficiency of electron recovery"
            }
        }

    def render(self):
        """Render the alert configuration interface."""
        st.title("🚨 Alert Configuration & Management")

        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "⚙️ Threshold Settings",
            "📊 Alert Dashboard",
            "📜 Alert History",
            "📧 Notifications",
            "⚡ Escalation Rules"
        ])

        with tab1:
            self._render_threshold_settings()

        with tab2:
            self._render_alert_dashboard()

        with tab3:
            self._render_alert_history()

        with tab4:
            self._render_notification_settings()

        with tab5:
            self._render_escalation_rules()

    def _render_threshold_settings(self):
        """Render threshold configuration interface."""
        st.header("Parameter Threshold Configuration")

        # Parameter selection
        selected_param = st.selectbox(
            "Select Parameter to Configure",
            options=list(self.parameter_info.keys()),
            format_func=lambda x: self.parameter_info[x]["name"]
        )

        param_info = self.parameter_info[selected_param]
        st.info(f"**{param_info['name']}**: {param_info['description']}")

        # Get current threshold
        current_threshold = self.alert_manager.thresholds.get(
            selected_param,
            AlertThreshold(parameter=selected_param, unit=param_info["unit"])
        )

        # Threshold configuration
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("⚠️ Warning Thresholds")

            # Warning range
            typical_min, typical_max = param_info["typical_range"]

            min_val = st.number_input(
                f"Minimum Value ({param_info['unit']})",
                value=float(current_threshold.min_value) if current_threshold.min_value else typical_min,
                key=f"{selected_param}_min"
            )

            max_val = st.number_input(
                f"Maximum Value ({param_info['unit']})",
                value=float(current_threshold.max_value) if current_threshold.max_value else typical_max,
                key=f"{selected_param}_max"
            )

        with col2:
            st.subheader("🚨 Critical Thresholds")

            # Critical range
            critical_min, critical_max = param_info["critical_range"]

            critical_min_val = st.number_input(
                f"Critical Minimum ({param_info['unit']})",
                value=float(current_threshold.critical_min) if current_threshold.critical_min else critical_min,
                key=f"{selected_param}_critical_min"
            )

            critical_max_val = st.number_input(
                f"Critical Maximum ({param_info['unit']})",
                value=float(current_threshold.critical_max) if current_threshold.critical_max else critical_max,
                key=f"{selected_param}_critical_max"
            )

        # Enable/disable toggle
        enabled = st.checkbox(
            "Enable alerts for this parameter",
            value=current_threshold.enabled,
            key=f"{selected_param}_enabled"
        )

        # Visualization of thresholds
        self._visualize_thresholds(
            selected_param, param_info,
            min_val, max_val, critical_min_val, critical_max_val
        )

        # Save button
        if st.button("💾 Save Threshold Settings", key=f"{selected_param}_save"):
            self.alert_manager.set_threshold(
                selected_param,
                min_value=min_val,
                max_value=max_val,
                critical_min=critical_min_val,
                critical_max=critical_max_val,
                unit=param_info["unit"],
                enabled=enabled
            )
            st.success(f"Threshold settings saved for {param_info['name']}")

        # Bulk operations
        st.divider()
        st.subheader("Bulk Operations")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔧 Apply Typical Values to All"):
                for param, info in self.parameter_info.items():
                    typical_min, typical_max = info["typical_range"]
                    critical_min, critical_max = info["critical_range"]
                    self.alert_manager.set_threshold(
                        param,
                        min_value=typical_min,
                        max_value=typical_max,
                        critical_min=critical_min,
                        critical_max=critical_max,
                        unit=info["unit"],
                        enabled=True
                    )
                st.success("Typical values applied to all parameters")

        with col2:
            if st.button("📥 Export Configuration"):
                config = self.alert_manager.export_alert_config()
                st.download_button(
                    "Download Config JSON",
                    data=json.dumps(config, indent=2),
                    file_name=f"alert_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

        with col3:
            uploaded_file = st.file_uploader(
                "📤 Import Configuration",
                type=['json'],
                key="config_upload"
            )
            if uploaded_file:
                config = json.load(uploaded_file)
                self.alert_manager.import_alert_config(config)
                st.success("Configuration imported successfully")

    def _visualize_thresholds(self, parameter: str, param_info: dict,
                             min_val: float, max_val: float,
                             critical_min: float, critical_max: float):
        """Visualize threshold ranges."""
        fig = go.Figure()

        # Get value range for visualization
        vis_min = min(critical_min, min_val) * 0.8
        vis_max = max(critical_max, max_val) * 1.2

        # Add threshold regions
        # Critical low
        fig.add_shape(
            type="rect",
            x0=vis_min, x1=critical_min,
            y0=0, y1=1,
            fillcolor="red",
            opacity=0.3,
            layer="below",
            line_width=0
        )

        # Warning low
        fig.add_shape(
            type="rect",
            x0=critical_min, x1=min_val,
            y0=0, y1=1,
            fillcolor="orange",
            opacity=0.3,
            layer="below",
            line_width=0
        )

        # Normal range
        fig.add_shape(
            type="rect",
            x0=min_val, x1=max_val,
            y0=0, y1=1,
            fillcolor="green",
            opacity=0.3,
            layer="below",
            line_width=0
        )

        # Warning high
        fig.add_shape(
            type="rect",
            x0=max_val, x1=critical_max,
            y0=0, y1=1,
            fillcolor="orange",
            opacity=0.3,
            layer="below",
            line_width=0
        )

        # Critical high
        fig.add_shape(
            type="rect",
            x0=critical_max, x1=vis_max,
            y0=0, y1=1,
            fillcolor="red",
            opacity=0.3,
            layer="below",
            line_width=0
        )

        # Add threshold lines and labels
        for val, label, color in [
            (critical_min, "Critical Min", "red"),
            (min_val, "Warning Min", "orange"),
            (max_val, "Warning Max", "orange"),
            (critical_max, "Critical Max", "red")
        ]:
            fig.add_vline(x=val, line_color=color, line_width=2, line_dash="dash")
            fig.add_annotation(
                x=val, y=0.5,
                text=f"{label}<br>{val:.2f}",
                showarrow=False,
                yshift=20
            )

        fig.update_layout(
            title=f"Threshold Visualization: {param_info['name']}",
            xaxis_title=f"Value ({param_info['unit']})",
            yaxis_visible=False,
            height=200,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_alert_dashboard(self):
        """Render active alerts dashboard."""
        st.header("Active Alerts Dashboard")

        # Get active alerts
        active_alerts = self.alert_manager.get_active_alerts()

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Active Alerts",
                len(active_alerts),
                delta=None
            )

        with col2:
            critical_count = sum(1 for a in active_alerts if a.severity == "critical")
            st.metric(
                "Critical Alerts",
                critical_count,
                delta=None
            )

        with col3:
            warning_count = sum(1 for a in active_alerts if a.severity == "warning")
            st.metric(
                "Warning Alerts",
                warning_count,
                delta=None
            )

        with col4:
            escalated_count = sum(1 for a in active_alerts if a.escalated)
            st.metric(
                "Escalated Alerts",
                escalated_count,
                delta=None
            )

        # Active alerts table
        if active_alerts:
            st.subheader("Active Alerts")

            # Convert to dataframe
            alert_data = []
            for alert in active_alerts:
                param_info = self.parameter_info.get(alert.parameter, {})
                alert_data.append({
                    "ID": alert.id,
                    "Time": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "Parameter": param_info.get("name", alert.parameter),
                    "Value": f"{alert.value:.3f} {param_info.get('unit', '')}",
                    "Severity": alert.severity.upper(),
                    "Message": alert.message,
                    "Escalated": "Yes" if alert.escalated else "No"
                })

            df = pd.DataFrame(alert_data)

            # Interactive table with acknowledge buttons
            for _idx, row in df.iterrows():
                col1, col2, col3 = st.columns([3, 4, 1])

                with col1:
                    severity_color = "🔴" if row["Severity"] == "CRITICAL" else "🟡"
                    st.write(f"{severity_color} **{row['Parameter']}** - {row['Value']}")

                with col2:
                    st.write(f"{row['Time']} - {row['Message']}")

                with col3:
                    if st.button("✅ Acknowledge", key=f"ack_{row['ID']}"):
                        self.alert_manager.acknowledge_alert(row['ID'], "operator")
                        st.rerun()
        else:
            st.success("✅ No active alerts")

        # Alert timeline visualization
        if active_alerts:
            st.subheader("Alert Timeline")
            self._render_alert_timeline(active_alerts)

    def _render_alert_timeline(self, alerts: list[Alert]):
        """Render alert timeline visualization."""
        fig = go.Figure()

        # Group alerts by parameter
        param_alerts = {}
        for alert in alerts:
            if alert.parameter not in param_alerts:
                param_alerts[alert.parameter] = []
            param_alerts[alert.parameter].append(alert)

        # Create timeline for each parameter
        y_pos = 0
        for param, param_alert_list in param_alerts.items():
            param_info = self.parameter_info.get(param, {"name": param})

            for alert in param_alert_list:
                color = "red" if alert.severity == "critical" else "orange"

                fig.add_trace(go.Scatter(
                    x=[alert.timestamp],
                    y=[y_pos],
                    mode='markers',
                    marker={
                        "size": 15,
                        "color": color,
                        "symbol": 'circle' if not alert.escalated else 'star'
                    },
                    text=f"{param_info['name']}: {alert.value:.3f}<br>{alert.message}",
                    hoverinfo='text',
                    showlegend=False
                ))

            y_pos += 1

        # Update layout
        fig.update_layout(
            title="Alert Timeline",
            xaxis_title="Time",
            yaxis={
                "tickmode": 'array',
                "tickvals": list(range(len(param_alerts))),
                "ticktext": [self.parameter_info.get(p, {"name": p})["name"]
                         for p in param_alerts.keys()]
            },
            height=400,
            hovermode='closest'
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_alert_history(self):
        """Render alert history interface."""
        st.header("Alert History")

        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            hours = st.selectbox(
                "Time Range",
                options=[1, 6, 12, 24, 48, 168],
                format_func=lambda x: f"Last {x} hours" if x < 168 else "Last week",
                index=3
            )

        with col2:
            param_filter = st.selectbox(
                "Filter by Parameter",
                options=["All"] + list(self.parameter_info.keys()),
                format_func=lambda x: x if x == "All" else self.parameter_info[x]["name"]
            )

        with col3:
            severity_filter = st.selectbox(
                "Filter by Severity",
                options=["All", "critical", "warning"]
            )

        # Get filtered alerts
        alerts = self.alert_manager.get_alert_history(
            hours=hours,
            parameter=None if param_filter == "All" else param_filter
        )

        # Apply severity filter
        if severity_filter != "All":
            alerts = [a for a in alerts if a.severity == severity_filter]

        # Display statistics
        st.subheader("Alert Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Alerts", len(alerts))

        with col2:
            acknowledged = sum(1 for a in alerts if a.acknowledged)
            st.metric("Acknowledged", acknowledged)

        with col3:
            critical = sum(1 for a in alerts if a.severity == "critical")
            st.metric("Critical", critical)

        with col4:
            escalated = sum(1 for a in alerts if a.escalated)
            st.metric("Escalated", escalated)

        # Alert frequency chart
        if alerts:
            st.subheader("Alert Frequency")
            self._render_alert_frequency_chart(alerts)

        # Detailed history table
        if alerts:
            st.subheader("Detailed History")

            history_data = []
            for alert in alerts:
                param_info = self.parameter_info.get(alert.parameter, {})
                history_data.append({
                    "Time": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "Parameter": param_info.get("name", alert.parameter),
                    "Value": f"{alert.value:.3f} {param_info.get('unit', '')}",
                    "Severity": alert.severity.upper(),
                    "Threshold": alert.threshold_violated,
                    "Acknowledged": "Yes" if alert.acknowledged else "No",
                    "By": alert.acknowledged_by or "-",
                    "Escalated": "Yes" if alert.escalated else "No"
                })

            df = pd.DataFrame(history_data)
            st.dataframe(df, use_container_width=True)

            # Export option
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download History CSV",
                csv,
                f"alert_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )

    def _render_alert_frequency_chart(self, alerts: list[Alert]):
        """Render alert frequency chart."""
        # Create hourly bins
        df = pd.DataFrame([{
            'timestamp': a.timestamp,
            'parameter': a.parameter,
            'severity': a.severity
        } for a in alerts])

        df['hour'] = df['timestamp'].dt.floor('H')

        # Count by parameter and severity
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Alerts by Parameter", "Alerts by Severity"),
            shared_xaxes=True
        )

        # By parameter
        for param in df['parameter'].unique():
            param_df = df[df['parameter'] == param]
            hourly_counts = param_df.groupby('hour').size()

            param_info = self.parameter_info.get(param, {"name": param})
            fig.add_trace(
                go.Scatter(
                    x=hourly_counts.index,
                    y=hourly_counts.values,
                    name=param_info["name"],
                    mode='lines+markers'
                ),
                row=1, col=1
            )

        # By severity
        for severity in ['warning', 'critical']:
            sev_df = df[df['severity'] == severity]
            if not sev_df.empty:
                hourly_counts = sev_df.groupby('hour').size()

                fig.add_trace(
                    go.Bar(
                        x=hourly_counts.index,
                        y=hourly_counts.values,
                        name=severity.upper(),
                        marker_color='orange' if severity == 'warning' else 'red'
                    ),
                    row=2, col=1
                )

        fig.update_layout(height=600, showlegend=True)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Alert Count", row=1, col=1)
        fig.update_yaxes(title_text="Alert Count", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

    def _render_notification_settings(self):
        """Render notification settings interface."""
        st.header("Notification Settings")

        # Email configuration
        st.subheader("📧 Email Configuration")

        # Note about email setup
        st.info(
            "Email notifications require SMTP server configuration. "
            "Contact your system administrator for setup."
        )

        # Admin emails
        admin_emails = st.text_area(
            "Administrator Email Addresses",
            value="\n".join(self.alert_manager.admin_emails),
            help="One email address per line. These receive critical alerts.",
            height=100
        )

        # User emails
        user_emails = st.text_area(
            "User Email Addresses",
            value="\n".join(self.alert_manager.user_emails),
            help="One email address per line. These receive escalated alerts.",
            height=100
        )

        # Save email settings
        if st.button("💾 Save Email Settings"):
            self.alert_manager.admin_emails = [
                email.strip() for email in admin_emails.split('\n')
                if email.strip()
            ]
            self.alert_manager.user_emails = [
                email.strip() for email in user_emails.split('\n')
                if email.strip()
            ]
            st.success("Email settings saved")

        # Notification preferences
        st.subheader("🔔 Notification Preferences")

        col1, col2 = st.columns(2)

        with col1:
            st.checkbox("Enable browser notifications", value=True, key="browser_notif")
            st.checkbox("Enable dashboard popups", value=True, key="dashboard_popup")
            st.checkbox("Enable sound alerts", value=False, key="sound_alerts")

        with col2:
            st.checkbox("Email on critical alerts", value=True, key="email_critical")
            st.checkbox("Email on escalations", value=True, key="email_escalation")
            st.checkbox("Daily digest email", value=False, key="daily_digest")

        # Test notifications
        st.subheader("🧪 Test Notifications")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Test Dashboard Alert"):
                test_alert = Alert(
                    parameter="test",
                    value=0.0,
                    severity="warning",
                    message="This is a test alert",
                    threshold_violated="Test threshold"
                )
                st.warning(f"🚨 Test Alert: {test_alert.message}")

        with col2:
            if st.button("Test Email Alert"):
                if self.alert_manager.email_service and self.alert_manager.admin_emails:
                    test_alert = Alert(
                        parameter="test",
                        value=0.0,
                        severity="critical",
                        message="This is a test email alert",
                        threshold_violated="Test threshold"
                    )
                    self.alert_manager.email_service.send_alert_email(
                        self.alert_manager.admin_emails[:1],
                        test_alert
                    )
                    st.success("Test email sent to first admin address")
                else:
                    st.error("Email service not configured or no admin emails set")

        with col3:
            if st.button("Test Browser Notification"):
                st.info("Browser notifications require user permission and HTTPS")
                st.code("""
// JavaScript for browser notifications
if (Notification.permission === "granted") {
    new Notification("MFC Alert", {
        body: "Test notification from MFC monitoring system",
        icon: "🚨"
    });
}
""", language="javascript")

    def _render_escalation_rules(self):
        """Render escalation rules configuration."""
        st.header("Escalation Rules")

        st.info(
            "Escalation rules automatically increase alert priority when "
            "multiple alerts occur within a time window."
        )

        # Current rules
        st.subheader("Current Escalation Rules")

        for idx, rule in enumerate(self.alert_manager.escalation_rules):
            with st.expander(f"Rule {idx + 1}: {rule.severity.upper()} - {rule.escalation_action}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Severity**: {rule.severity.upper()}")
                    st.write(f"**Time Window**: {rule.time_window_minutes} minutes")
                    st.write(f"**Threshold Count**: {rule.threshold_count} alerts")

                with col2:
                    st.write(f"**Action**: {rule.escalation_action}")
                    st.write(f"**Cooldown**: {rule.cooldown_minutes} minutes")

                if st.button("🗑️ Remove Rule", key=f"remove_rule_{idx}"):
                    self.alert_manager.escalation_rules.pop(idx)
                    st.rerun()

        # Add new rule
        st.subheader("Add New Escalation Rule")

        col1, col2 = st.columns(2)

        with col1:
            new_severity = st.selectbox(
                "Alert Severity",
                options=["warning", "critical"],
                key="new_rule_severity"
            )

            new_window = st.number_input(
                "Time Window (minutes)",
                min_value=1,
                max_value=1440,
                value=30,
                key="new_rule_window"
            )

            new_count = st.number_input(
                "Alert Count Threshold",
                min_value=1,
                max_value=100,
                value=5,
                key="new_rule_count"
            )

        with col2:
            new_action = st.selectbox(
                "Escalation Action",
                options=["email_admin", "email_all", "dashboard_popup"],
                format_func=lambda x: {
                    "email_admin": "Email Administrators",
                    "email_all": "Email All Users",
                    "dashboard_popup": "Dashboard Popup"
                }[x],
                key="new_rule_action"
            )

            new_cooldown = st.number_input(
                "Cooldown Period (minutes)",
                min_value=1,
                max_value=1440,
                value=60,
                key="new_rule_cooldown"
            )

        if st.button("➕ Add Escalation Rule"):
            new_rule = EscalationRule(
                severity=new_severity,
                time_window_minutes=new_window,
                threshold_count=new_count,
                escalation_action=new_action,
                cooldown_minutes=new_cooldown
            )
            self.alert_manager.escalation_rules.append(new_rule)
            st.success("Escalation rule added")
            st.rerun()

        # Example scenarios
        st.subheader("Example Escalation Scenarios")

        scenarios = [
            {
                "title": "Rapid pH Deterioration",
                "description": "5 pH warnings in 30 minutes → Email administrators",
                "rule": EscalationRule("warning", 30, 5, "email_admin", 60)
            },
            {
                "title": "Critical System Failure",
                "description": "3 critical alerts in 5 minutes → Email all users",
                "rule": EscalationRule("critical", 5, 3, "email_all", 120)
            },
            {
                "title": "Persistent Warnings",
                "description": "10 warnings in 2 hours → Dashboard popup",
                "rule": EscalationRule("warning", 120, 10, "dashboard_popup", 180)
            }
        ]

        for scenario in scenarios:
            with st.expander(scenario["title"]):
                st.write(scenario["description"])
                if st.button("Apply This Rule", key=f"apply_{scenario['title']}"):
                    self.alert_manager.escalation_rules.append(scenario["rule"])
                    st.success(f"Rule '{scenario['title']}' added")
                    st.rerun()


# Standalone utility functions
def create_alert_rule(parameter: str, threshold: float, condition: str = "greater_than"):
    """Create a new alert rule."""
    return {
        'parameter': parameter,
        'threshold': threshold,
        'condition': condition,
        'created_at': datetime.now().isoformat()
    }

def check_alerts(current_values: dict, alert_rules: list):
    """Check current values against alert rules."""
    triggered_alerts = []
    for rule in alert_rules:
        param = rule['parameter']
        if param in current_values:
            value = current_values[param]
            threshold = rule['threshold']
            condition = rule['condition']
            
            triggered = False
            if condition == "greater_than" and value > threshold:
                triggered = True
            elif condition == "less_than" and value < threshold:
                triggered = True
            elif condition == "equals" and abs(value - threshold) < 0.001:
                triggered = True
                
            if triggered:
                triggered_alerts.append({
                    'rule': rule,
                    'current_value': value,
                    'timestamp': datetime.now().isoformat()
                })
    
    return triggered_alerts

# Integration function for the main GUI
def render_alert_configuration(alert_manager: AlertManager = None):
    """Render the alert configuration UI."""
    if alert_manager is None:
        # Create a default manager for testing
        alert_manager = AlertManager()
    
    ui = AlertConfigurationUI(alert_manager)
    ui.render()


if __name__ == "__main__":
    # Standalone testing
    st.set_page_config(
        page_title="MFC Alert Configuration",
        page_icon="🚨",
        layout="wide"
    )

    # Create alert manager for testing
    manager = AlertManager()

    # Render UI
    render_alert_configuration(manager)
