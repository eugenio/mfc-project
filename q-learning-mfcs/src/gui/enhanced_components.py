#!/usr/bin/env python3
"""
Enhanced UI Components for Scientific Community Engagement

This module provides advanced Streamlit components designed specifically for 
scientific researchers and practitioners working with MFC systems.

Features:
- Interactive parameter validation with literature references
- Real-time scientific visualization with publication-ready export
- Collaborative research tools and data sharing capabilities
- Advanced statistical analysis and comparison tools

Created: 2025-07-31
Literature References:
1. Nielsen, J. (1993). "Usability Engineering"
2. Shneiderman, B. (2016). "Designing the User Interface"
3. Few, S. (2009). "Now You See It: Simple Visualization Techniques"
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime
import io

# Import existing configurations
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.visualization_config import get_publication_visualization_config

class ComponentTheme(Enum):
    """UI theme options for different contexts."""
    LIGHT = "light"
    DARK = "dark"
    SCIENTIFIC = "scientific"
    HIGH_CONTRAST = "high_contrast"

@dataclass
class UIThemeConfig:
    """UI theme configuration for enhanced components."""
    primary_color: str = "#2E86AB"
    secondary_color: str = "#A23B72"
    success_color: str = "#27AE60"
    warning_color: str = "#F39C12"
    error_color: str = "#E74C3C"
    background_color: str = "#FFFFFF"
    text_color: str = "#2C3E50"
    border_color: str = "#BDC3C7"
    accent_color: str = "#9B59B6"

class ScientificParameterInput:
    """Enhanced parameter input with scientific validation and literature references."""

    def __init__(self, theme: UIThemeConfig = UIThemeConfig()):
        """Initialize scientific parameter input component.
        
        Args:
            theme: UI theme configuration
        """
        self.theme = theme
        self._initialize_custom_css()

    def _initialize_custom_css(self):
        """Initialize custom CSS styles for scientific components."""
        st.markdown(f"""
        <style>
        .scientific-container {{
            background-color: {self.theme.background_color};
            border: 2px solid {self.theme.border_color};
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .parameter-header {{
            color: {self.theme.primary_color};
            font-weight: bold;
            font-size: 18px;
            margin-bottom: 10px;
            border-bottom: 2px solid {self.theme.accent_color};
            padding-bottom: 5px;
        }}
        
        .literature-reference {{
            background-color: #F8F9FA;
            border-left: 4px solid {self.theme.secondary_color};
            padding: 10px;
            margin: 10px 0;
            font-style: italic;
            font-size: 12px;
        }}
        
        .validation-success {{
            color: {self.theme.success_color};
            font-weight: bold;
        }}
        
        .validation-warning {{
            color: {self.theme.warning_color};
            font-weight: bold;
        }}
        
        .validation-error {{
            color: {self.theme.error_color};
            font-weight: bold;
        }}
        
        .scientific-unit {{
            color: {self.theme.secondary_color};
            font-weight: bold;
            font-style: italic;
        }}
        
        .parameter-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        </style>
        """, unsafe_allow_html=True)

    def render_parameter_section(
        self,
        title: str,
        parameters: Dict[str, Dict[str, Any]],
        key_prefix: str = ""
    ) -> Dict[str, Any]:
        """Render a section of scientific parameters with validation.
        
        Args:
            title: Section title
            parameters: Dictionary of parameter configurations
            key_prefix: Unique key prefix for Streamlit components
            
        Returns:
            Dictionary of parameter values
        """
        st.markdown('<div class="scientific-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="parameter-header">{title}</div>', unsafe_allow_html=True)

        values = {}
        cols = st.columns(2)

        for i, (param_name, config) in enumerate(parameters.items()):
            col = cols[i % 2]

            with col:
                values[param_name] = self._render_single_parameter(
                    param_name, config, f"{key_prefix}_{param_name}"
                )

        st.markdown('</div>', unsafe_allow_html=True)
        return values

    def _render_single_parameter(
        self,
        param_name: str,
        config: Dict[str, Any],
        key: str
    ) -> Union[float, int, bool, str]:
        """Render a single parameter input with validation.
        
        Args:
            param_name: Parameter name
            config: Parameter configuration
            key: Unique key for Streamlit component
            
        Returns:
            Parameter value
        """
        param_type = config.get('type', 'float')
        default_value = config.get('default', 0.0)
        min_val = config.get('min', None)
        max_val = config.get('max', None)
        unit = config.get('unit', '')
        description = config.get('description', '')
        literature_ref = config.get('literature_reference', '')

        # Display parameter name and description
        st.markdown(f"**{param_name.replace('_', ' ').title()}** "
                   f"<span class='scientific-unit'>({unit})</span>",
                   unsafe_allow_html=True)

        if description:
            st.markdown(f"*{description}*")

        # Input widget based on parameter type
        value: Union[float, int, bool, str]
        if param_type == 'float':
            value = st.number_input(
                label="",
                min_value=min_val,
                max_value=max_val,
                value=float(default_value),
                step=0.01,
                key=key,
                label_visibility="collapsed"
            )
        elif param_type == 'int':
            value = st.number_input(
                label="",
                min_value=int(min_val) if min_val else None,
                max_value=int(max_val) if max_val else None,
                value=int(default_value),
                step=1,
                key=key,
                label_visibility="collapsed"
            )
        elif param_type == 'bool':
            value = st.checkbox(
                label="",
                value=bool(default_value),
                key=key,
                label_visibility="collapsed"
            )
        elif param_type == 'select':
            options = config.get('options', [])
            value = st.selectbox(
                label="",
                options=options,
                index=options.index(default_value) if default_value in options else 0,
                key=key,
                label_visibility="collapsed"
            )
        else:
            text_value = st.text_input(
                label="",
                value=str(default_value),
                key=key,
                label_visibility="collapsed"
            )
            # Convert based on expected parameter type
            param_type = config.get('type', 'float')
            if param_type == 'float':
                try:
                    value = float(text_value)
                except ValueError:
                    value = float(default_value)
            elif param_type == 'int':
                try:
                    value = int(text_value)
                except ValueError:
                    value = int(default_value)
            else:
                value = text_value

        # Validation and literature reference (only for numeric values)
        if min_val is not None and max_val is not None and isinstance(value, (float, int)):
            self._display_validation_status(value, min_val, max_val, param_name)

        if literature_ref:
            self._display_literature_reference(literature_ref)

        st.markdown("---")
        return value

    def _display_validation_status(
        self,
        value: float,
        min_val: float,
        max_val: float,
        param_name: str
    ):
        """Display parameter validation status with scientific context."""
        if min_val <= value <= max_val:
            st.markdown(
                f'<span class="validation-success">✓ Valid range: {min_val} - {max_val}</span>',
                unsafe_allow_html=True
            )
        elif value < min_val:
            st.markdown(
                f'<span class="validation-error">⚠ Below minimum ({min_val}): '
                f'May affect biofilm viability</span>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<span class="validation-error">⚠ Above maximum ({max_val}): '
                f'May exceed system limits</span>',
                unsafe_allow_html=True
            )

    def _display_literature_reference(self, reference: str):
        """Display literature reference for parameter."""
        st.markdown(
            f'<div class="literature-reference">📚 Literature: {reference}</div>',
            unsafe_allow_html=True
        )

class InteractiveVisualization:
    """Enhanced interactive visualization component for scientific data."""

    def __init__(self, theme: UIThemeConfig = UIThemeConfig()):
        """Initialize interactive visualization component.
        
        Args:
            theme: UI theme configuration
        """
        self.theme = theme
        self.viz_config = get_publication_visualization_config()

    def render_multi_panel_dashboard(
        self,
        data: Dict[str, pd.DataFrame],
        layout: str = "2x2",
        title: str = "MFC Performance Dashboard"
    ) -> go.Figure:
        """Render multi-panel interactive dashboard.
        
        Args:
            data: Dictionary of DataFrames for each panel
            layout: Layout configuration (e.g., "2x2", "1x4")
            title: Dashboard title
            
        Returns:
            Plotly figure object
        """
        # Parse layout
        rows, cols = map(int, layout.split('x'))

        # Create subplot structure
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=list(data.keys()),
            vertical_spacing=0.1,
            horizontal_spacing=0.1,
            specs=[[{"secondary_y": True} for _ in range(cols)] for _ in range(rows)]
        )

        # Add traces for each panel
        for i, (panel_name, df) in enumerate(data.items()):
            row = (i // cols) + 1
            col = (i % cols) + 1

            self._add_panel_traces(fig, df, row, col, panel_name)

        # Update layout
        fig.update_layout(
            title=title,
            title_font_size=20,
            showlegend=True,
            template="plotly_white",
            height=800,
            hovermode='x unified'
        )

        return fig

    def _add_panel_traces(
        self,
        fig: go.Figure,
        df: pd.DataFrame,
        row: int,
        col: int,
        panel_name: str
    ):
        """Add traces to a specific panel."""
        # Determine trace type based on panel name and data structure
        if 'time' in df.columns:
            # Time series data
            for column in df.columns:
                if column != 'time':
                    fig.add_trace(
                        go.Scatter(
                            x=df['time'],
                            y=df[column],
                            name=f"{panel_name}: {column}",
                            mode='lines',
                            line=dict(width=2)
                        ),
                        row=row, col=col
                    )
        else:
            # Non-time series data
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                fig.add_trace(
                    go.Scatter(
                        x=df[numeric_cols[0]],
                        y=df[numeric_cols[1]],
                        name=panel_name,
                        mode='markers',
                        marker=dict(size=6)
                    ),
                    row=row, col=col
                )

    def render_real_time_monitor(
        self,
        data_stream: Callable[[], Dict[str, float]],
        refresh_interval: int = 5,
        max_points: int = 100
    ):
        """Render real-time monitoring component.
        
        Args:
            data_stream: Function that returns current data point
            refresh_interval: Refresh interval in seconds
            max_points: Maximum points to display
        """
        # Initialize session state for real-time data
        if 'realtime_data' not in st.session_state:
            st.session_state.realtime_data = {
                'timestamps': [],
                'values': {}
            }

        # Control panel
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            is_monitoring = st.checkbox("🔴 Live Monitoring", value=False)

        with col2:
            if st.button("🔄 Refresh Data"):
                self._update_realtime_data(data_stream)

        with col3:
            st.markdown(f"**Refresh Rate**: {refresh_interval}s | **Buffer Size**: {max_points} points")

        # Real-time plot
        if is_monitoring:
            # Note: Real-time streaming requires proper implementation
            # For now, show manual refresh functionality
            st.info("🔴 Live monitoring active - Use refresh button to update data")

            # Update data on monitoring enable
            self._update_realtime_data(data_stream, max_points)

            # Create real-time figure
            fig = self._create_realtime_figure()
            st.plotly_chart(fig, use_container_width=True)

            # Auto-refresh using st.rerun() approach (avoids infinite loop)
            placeholder = st.empty()
            with placeholder.container():
                if st.button("🔄 Auto-refresh (5s)", key="auto_refresh"):
                    import time
                    with st.spinner("Refreshing data..."):
                        time.sleep(1)  # Brief pause for UX
                        self._update_realtime_data(data_stream, max_points)
                        st.rerun()
        else:
            # Static display
            if st.session_state.realtime_data['timestamps']:
                fig = self._create_realtime_figure()
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Enable live monitoring to see real-time data")

    def _update_realtime_data(
        self,
        data_stream: Callable[[], Dict[str, float]],
        max_points: int = 100
    ):
        """Update real-time data buffer."""
        try:
            # Get new data point
            new_data = data_stream()
            current_time = datetime.now()

            # Update timestamps
            st.session_state.realtime_data['timestamps'].append(current_time)

            # Update values
            for key, value in new_data.items():
                if key not in st.session_state.realtime_data['values']:
                    st.session_state.realtime_data['values'][key] = []
                st.session_state.realtime_data['values'][key].append(value)

            # Maintain buffer size
            if len(st.session_state.realtime_data['timestamps']) > max_points:
                st.session_state.realtime_data['timestamps'] = \
                    st.session_state.realtime_data['timestamps'][-max_points:]

                for key in st.session_state.realtime_data['values']:
                    st.session_state.realtime_data['values'][key] = \
                        st.session_state.realtime_data['values'][key][-max_points:]

        except Exception as e:
            st.error(f"Error updating real-time data: {e}")

    def _create_realtime_figure(self) -> go.Figure:
        """Create real-time monitoring figure."""
        fig = go.Figure()

        timestamps = st.session_state.realtime_data['timestamps']
        values = st.session_state.realtime_data['values']

        for metric_name, metric_values in values.items():
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=metric_values,
                    name=metric_name,
                    mode='lines+markers',
                    line=dict(width=2),
                    marker=dict(size=4)
                )
            )

        fig.update_layout(
            title="Real-Time MFC Monitoring",
            title_font_size=18,
            xaxis_title="Time",
            yaxis_title="Value",
            template="plotly_white",
            height=400,
            showlegend=True,
            hovermode='x unified'
        )

        return fig

class ExportManager:
    """Advanced export functionality for research data and visualizations."""

    def __init__(self):
        """Initialize export manager."""
        self.supported_formats = {
            'data': ['csv', 'json', 'xlsx', 'hdf5'],
            'figures': ['png', 'pdf', 'svg', 'html'],
            'reports': ['pdf', 'html', 'docx']
        }

    def render_export_panel(
        self,
        data: Optional[Dict[str, pd.DataFrame]] = None,
        figures: Optional[Dict[str, go.Figure]] = None
    ):
        """Render comprehensive export panel.
        
        Args:
            data: Dictionary of DataFrames to export
            figures: Dictionary of figures to export
        """
        st.markdown("### 📤 Export Center")

        # Export tabs
        tab1, tab2, tab3 = st.tabs(["📊 Data Export", "📈 Figure Export", "📄 Report Export"])

        with tab1:
            self._render_data_export(data)

        with tab2:
            self._render_figure_export(figures)

        with tab3:
            self._render_report_export(data, figures)

    def _render_data_export(self, data: Optional[Dict[str, pd.DataFrame]]):
        """Render data export options."""
        if not data:
            st.info("No data available for export")
            return

        st.markdown("#### Available Datasets")

        # Dataset selection
        selected_datasets = []
        for dataset_name, df in data.items():
            if st.checkbox(f"{dataset_name} ({len(df)} rows)", key=f"export_data_{dataset_name}"):
                selected_datasets.append(dataset_name)

        if selected_datasets:
            # Format selection
            col1, col2 = st.columns(2)

            with col1:
                export_format = st.selectbox(
                    "Export Format",
                    options=self.supported_formats['data'],
                    key="data_export_format"
                )

            with col2:
                include_metadata = st.checkbox(
                    "Include Metadata",
                    value=True,
                    key="include_metadata"
                )

            # Export button
            if st.button("📥 Download Selected Data", key="download_data"):
                self._export_data(
                    {name: data[name] for name in selected_datasets},
                    export_format,
                    include_metadata
                )

    def _render_figure_export(self, figures: Optional[Dict[str, go.Figure]]):
        """Render figure export options."""
        if not figures:
            st.info("No figures available for export")
            return

        st.markdown("#### Available Figures")

        # Figure selection
        selected_figures = []
        for fig_name, fig in figures.items():
            if st.checkbox(f"{fig_name}", key=f"export_fig_{fig_name}"):
                selected_figures.append(fig_name)

        if selected_figures:
            # Export options
            col1, col2, col3 = st.columns(3)

            with col1:
                export_format = st.selectbox(
                    "Format",
                    options=self.supported_formats['figures'],
                    key="figure_export_format"
                )

            with col2:
                resolution = st.selectbox(
                    "Resolution (DPI)",
                    options=[300, 600, 1200],
                    index=1,
                    key="figure_resolution"
                )

            with col3:
                include_data = st.checkbox(
                    "Include Data",
                    value=True,
                    key="figure_include_data"
                )

            # Export button
            if st.button("📥 Download Selected Figures", key="download_figures"):
                self._export_figures(
                    {name: figures[name] for name in selected_figures},
                    export_format,
                    resolution,
                    include_data
                )

    def _render_report_export(
        self,
        data: Optional[Dict[str, pd.DataFrame]],
        figures: Optional[Dict[str, go.Figure]]
    ):
        """Render comprehensive report export."""
        st.markdown("#### Generate Comprehensive Report")

        # Report configuration
        col1, col2 = st.columns(2)

        with col1:
            report_title = st.text_input(
                "Report Title",
                value="MFC Analysis Report",
                key="report_title"
            )

            report_format = st.selectbox(
                "Report Format",
                options=self.supported_formats['reports'],
                key="report_format"
            )

        with col2:
            include_methods = st.checkbox(
                "Include Methods Section",
                value=True,
                key="include_methods"
            )

            include_references = st.checkbox(
                "Include Literature References",
                value=True,
                key="include_references"
            )

        # Report sections
        st.markdown("**Report Sections**")
        sections = {
            "Executive Summary": st.checkbox("Executive Summary", value=True, key="section_summary"),
            "Methodology": st.checkbox("Methodology", value=include_methods, key="section_methods"),
            "Results": st.checkbox("Results", value=True, key="section_results"),
            "Discussion": st.checkbox("Discussion", value=True, key="section_discussion"),
            "Conclusions": st.checkbox("Conclusions", value=True, key="section_conclusions"),
            "References": st.checkbox("References", value=include_references, key="section_references")
        }

        # Generate report
        if st.button("📄 Generate Report", key="generate_report"):
            self._generate_comprehensive_report(
                report_title,
                report_format,
                sections,
                data,
                figures
            )

    def _export_data(
        self,
        datasets: Dict[str, pd.DataFrame],
        format: str,
        include_metadata: bool
    ):
        """Export selected datasets."""
        try:
            if format == 'csv':
                # Create ZIP file for multiple CSVs
                import zipfile
                zip_buffer = io.BytesIO()

                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    for name, df in datasets.items():
                        csv_buffer = io.StringIO()

                        if include_metadata:
                            # Add metadata header
                            csv_buffer.write(f"# Dataset: {name}\n")
                            csv_buffer.write(f"# Generated: {datetime.now().isoformat()}\n")
                            csv_buffer.write(f"# Rows: {len(df)}, Columns: {len(df.columns)}\n")
                            csv_buffer.write(f"# Columns: {', '.join(df.columns)}\n")
                            csv_buffer.write("#\n")

                        df.to_csv(csv_buffer, index=False)
                        zip_file.writestr(f"{name}.csv", csv_buffer.getvalue())

                # Download button
                st.download_button(
                    label="📥 Download CSV Files",
                    data=zip_buffer.getvalue(),
                    file_name=f"mfc_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip"
                )

            elif format == 'json':
                # Export as JSON
                export_data = {}
                for name, df in datasets.items():
                    export_data[name] = {
                        'metadata': {
                            'rows': len(df),
                            'columns': len(df.columns),
                            'generated': datetime.now().isoformat()
                        } if include_metadata else {},
                        'data': df.to_dict('records')
                    }

                json_str = json.dumps(export_data, indent=2, default=str)

                st.download_button(
                    label="📥 Download JSON File",
                    data=json_str,
                    file_name=f"mfc_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

            st.success(f"Data exported successfully in {format.upper()} format!")

        except Exception as e:
            st.error(f"Export failed: {e}")

    def _export_figures(
        self,
        figures: Dict[str, go.Figure],
        format: str,
        resolution: int,
        include_data: bool
    ):
        """Export selected figures."""
        try:
            import zipfile
            zip_buffer = io.BytesIO()

            with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                for name, fig in figures.items():
                    if format == 'png':
                        img_bytes = fig.to_image(format="png", scale=resolution/300)
                        zip_file.writestr(f"{name}.png", img_bytes)

                    elif format == 'pdf':
                        pdf_bytes = fig.to_image(format="pdf")
                        zip_file.writestr(f"{name}.pdf", pdf_bytes)

                    elif format == 'svg':
                        svg_str = fig.to_image(format="svg").decode()
                        zip_file.writestr(f"{name}.svg", svg_str)

                    elif format == 'html':
                        html_str = fig.to_html(include_plotlyjs=True)
                        zip_file.writestr(f"{name}.html", html_str)

            # Download button
            st.download_button(
                label="📥 Download Figures",
                data=zip_buffer.getvalue(),
                file_name=f"mfc_figures_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip"
            )

            st.success(f"Figures exported successfully in {format.upper()} format!")

        except Exception as e:
            st.error(f"Figure export failed: {e}")

    def _generate_comprehensive_report(
        self,
        title: str,
        format: str,
        sections: Dict[str, bool],
        data: Optional[Dict[str, pd.DataFrame]],
        figures: Optional[Dict[str, go.Figure]]
    ):
        """Generate comprehensive research report."""
        try:
            if format == 'html':
                html_content = self._generate_html_report(title, sections, data, figures)

                st.download_button(
                    label="📄 Download HTML Report",
                    data=html_content,
                    file_name=f"mfc_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html"
                )

            st.success("Report generated successfully!")

        except Exception as e:
            st.error(f"Report generation failed: {e}")

    def _generate_html_report(
        self,
        title: str,
        sections: Dict[str, bool],
        data: Optional[Dict[str, pd.DataFrame]],
        figures: Optional[Dict[str, go.Figure]]
    ) -> str:
        """Generate HTML report content."""
        html_parts = [
            f"<html><head><title>{title}</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 40px; }",
            "h1 { color: #2E86AB; border-bottom: 2px solid #2E86AB; }",
            "h2 { color: #A23B72; }",
            "table { border-collapse: collapse; width: 100%; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #f2f2f2; }",
            "</style></head><body>",
            f"<h1>{title}</h1>",
            f"<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
        ]

        # Add selected sections
        if sections.get("Executive Summary", False):
            html_parts.extend([
                "<h2>Executive Summary</h2>",
                "<p>This report presents comprehensive analysis of MFC performance data, ",
                "including real-time monitoring results, parameter optimization outcomes, ",
                "and comparative studies against literature benchmarks.</p>"
            ])

        if sections.get("Results", False) and data:
            html_parts.append("<h2>Results</h2>")
            for name, df in data.items():
                html_parts.extend([
                    f"<h3>{name}</h3>",
                    df.head(10).to_html(),
                    f"<p><em>Showing first 10 rows of {len(df)} total records.</em></p>"
                ])

        html_parts.append("</body></html>")
        return "\n".join(html_parts)

# Utility functions for component integration

def initialize_enhanced_ui(theme: ComponentTheme = ComponentTheme.SCIENTIFIC) -> Tuple[UIThemeConfig, Dict[str, Any]]:
    """Initialize enhanced UI components with theme.
    
    Args:
        theme: UI theme selection
        
    Returns:
        Tuple of theme configuration and component instances
    """
    # Select theme configuration
    if theme == ComponentTheme.SCIENTIFIC:
        theme_config = UIThemeConfig(
            primary_color="#2E86AB",
            secondary_color="#A23B72",
            background_color="#FDFDFD"
        )
    elif theme == ComponentTheme.DARK:
        theme_config = UIThemeConfig(
            primary_color="#64B5F6",
            secondary_color="#BA68C8",
            background_color="#1E1E1E",
            text_color="#FFFFFF"
        )
    else:
        theme_config = UIThemeConfig()

    # Initialize components
    components = {
        'parameter_input': ScientificParameterInput(theme_config),
        'visualization': InteractiveVisualization(theme_config),
        'export_manager': ExportManager()
    }

    return theme_config, components

def render_enhanced_sidebar() -> Dict[str, Any]:
    """Render enhanced sidebar with scientific tools.
    
    Returns:
        Dictionary of sidebar selections and configurations
    """
    st.sidebar.markdown("## 🔬 Scientific Tools")

    # Theme selection
    theme = st.sidebar.selectbox(
        "UI Theme",
        options=[t.value for t in ComponentTheme],
        index=2  # Default to scientific theme
    )

    # Visualization options
    st.sidebar.markdown("### 📊 Visualization")
    viz_options = {
        'publication_ready': st.sidebar.checkbox("Publication Ready", value=True),
        'interactive_plots': st.sidebar.checkbox("Interactive Plots", value=True),
        'real_time_monitoring': st.sidebar.checkbox("Real-time Monitoring", value=False),
        'export_enabled': st.sidebar.checkbox("Enable Export", value=True)
    }

    # Advanced options
    with st.sidebar.expander("🔧 Advanced Options"):
        advanced_options = {
            'parameter_validation': st.checkbox("Parameter Validation", value=True),
            'literature_references': st.checkbox("Literature References", value=True),
            'statistical_analysis': st.checkbox("Statistical Analysis", value=True),
            'collaboration_tools': st.checkbox("Collaboration Tools", value=False)
        }

    return {
        'theme': ComponentTheme(theme),
        'visualization': viz_options,
        'advanced': advanced_options
    }
