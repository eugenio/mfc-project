"""
Interactive Q-Table Visualization Component

This module provides interactive visualizations for Q-learning tables,
including heatmaps, convergence tracking, and policy analysis.

User Story 1.2.1: Interactive Q-Table Analysis
Created: 2025-07-31
Last Modified: 2025-07-31
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Import Q-table analyzer
from analysis.qtable_analyzer import QTABLE_ANALYZER, ConvergenceStatus, QTableMetrics
from plotly.subplots import make_subplots


class QTableVisualization:
    """Interactive Q-table visualization component."""

    def __init__(self):
        """Initialize Q-table visualization component."""
        self.analyzer = QTABLE_ANALYZER

        # Initialize session state
        if 'selected_qtables' not in st.session_state:
            st.session_state.selected_qtables = []
        if 'qtable_analysis_cache' not in st.session_state:
            st.session_state.qtable_analysis_cache = {}
        if 'comparison_results' not in st.session_state:
            st.session_state.comparison_results = {}

    def render_qtable_analysis_interface(self) -> dict[str, Any]:
        """
        Render the main Q-table analysis interface.

        Returns:
            Analysis results and metadata
        """
        st.header("🧠 Interactive Q-Table Analysis")
        st.markdown("""
        Analyze Q-learning algorithm behavior with convergence indicators, policy quality metrics,
        and interactive visualizations. Upload or select existing Q-table files for analysis.
        """)

        # File selection interface
        self._render_file_selection_section()

        # Analysis results
        if st.session_state.selected_qtables:
            self._render_analysis_results_section()
            self._render_visualization_section()
            self._render_comparison_section()
            self._render_export_section()

        return {
            'selected_qtables': st.session_state.selected_qtables,
            'analysis_cache': st.session_state.qtable_analysis_cache,
            'comparison_results': st.session_state.comparison_results
        }

    def _render_file_selection_section(self):
        """Render Q-table file selection interface."""
        st.subheader("📁 Q-Table File Selection")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Get available Q-table files
            available_files = self.analyzer.get_available_qtables()

            if available_files:
                file_options = [str(f) for f in available_files]
                file_names = [f.name for f in available_files]

                selected_indices = st.multiselect(
                    "Select Q-table files for analysis",
                    options=range(len(file_options)),
                    format_func=lambda x: f"{file_names[x]} ({self._get_file_size(available_files[x])})",
                    help="Select one or more Q-table files to analyze"
                )

                st.session_state.selected_qtables = [file_options[i] for i in selected_indices]
            else:
                st.warning("No Q-table files found in the models directory.")
                st.info("Expected location: `q_learning_models/*.pkl`")

        with col2:
            if st.button("🔄 Refresh File List"):
                # Clear cache to refresh file list
                st.rerun()

            if st.button("📊 Quick Analysis"):
                if st.session_state.selected_qtables:
                    self._perform_quick_analysis()
                else:
                    st.warning("Please select Q-table files first.")

    def _render_analysis_results_section(self):
        """Render analysis results for selected Q-tables."""
        st.subheader("📈 Analysis Results")

        # Analyze selected Q-tables
        analysis_results = {}
        for file_path in st.session_state.selected_qtables:
            if file_path not in st.session_state.qtable_analysis_cache:
                with st.spinner(f"Analyzing {Path(file_path).name}..."):
                    metrics = self.analyzer.analyze_qtable(file_path)
                    if metrics:
                        st.session_state.qtable_analysis_cache[file_path] = metrics

            if file_path in st.session_state.qtable_analysis_cache:
                analysis_results[file_path] = st.session_state.qtable_analysis_cache[file_path]

        if analysis_results:
            self._display_analysis_summary(analysis_results)
            self._display_detailed_metrics(analysis_results)

    def _display_analysis_summary(self, results: dict[str, QTableMetrics]):
        """Display summary of analysis results."""
        st.markdown("### 📊 Analysis Summary")

        # Create summary metrics
        total_files = len(results)
        converged_files = sum(1 for m in results.values() if m.convergence_status == ConvergenceStatus.CONVERGED)
        avg_convergence = np.mean([m.convergence_score for m in results.values()])
        avg_exploration = np.mean([m.exploration_coverage for m in results.values()])

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Q-Tables", total_files)

        with col2:
            st.metric(
                "Converged Tables",
                converged_files,
                delta=f"{converged_files/total_files:.1%} success rate"
            )

        with col3:
            st.metric(
                "Avg Convergence Score",
                f"{avg_convergence:.3f}",
                delta="0.900 target"
            )

        with col4:
            st.metric(
                "Avg Exploration Coverage",
                f"{avg_exploration:.1%}",
                delta="Higher is better"
            )

    def _display_detailed_metrics(self, results: dict[str, QTableMetrics]):
        """Display detailed metrics table."""
        st.markdown("### 📋 Detailed Metrics")

        # Create detailed metrics table
        rows = []
        for file_path, metrics in results.items():
            rows.append({
                'File': Path(file_path).name,
                'States': metrics.total_states,
                'Actions': metrics.total_actions,
                'Convergence Score': f"{metrics.convergence_score:.3f}",
                'Status': metrics.convergence_status.value.title(),
                'Policy Entropy': f"{metrics.policy_entropy:.3f}",
                'Exploration': f"{metrics.exploration_coverage:.1%}",
                'Sparsity': f"{metrics.sparsity:.1%}",
                'Q-Value Range': f"{metrics.q_value_range:.3f}"
            })

        df = pd.DataFrame(rows)

        # Color code the status column
        def color_status(val):
            colors = {
                'Converged': 'background-color: #d4edda',
                'Converging': 'background-color: #fff3cd',
                'Unstable': 'background-color: #f8d7da',
                'Diverging': 'background-color: #f5c6cb',
                'Unknown': 'background-color: #e2e3e5'
            }
            return colors.get(val, '')

        styled_df = df.style.apply(lambda x: [color_status(val) if x.name == 'Status' else '' for val in x], axis=0)
        st.dataframe(styled_df, use_container_width=True)

    def _render_visualization_section(self):
        """Render interactive visualizations."""
        st.subheader("📊 Interactive Visualizations")

        # Visualization type selector
        viz_tabs = st.tabs([
            "🔥 Q-Table Heatmap",
            "📈 Convergence Trends",
            "🎯 Policy Analysis",
            "🗺️ State-Action Exploration"
        ])

        with viz_tabs[0]:
            self._render_qtable_heatmap()

        with viz_tabs[1]:
            self._render_convergence_trends()

        with viz_tabs[2]:
            self._render_policy_analysis()

        with viz_tabs[3]:
            self._render_exploration_visualization()

    def _render_qtable_heatmap(self):
        """Render interactive Q-table heatmap."""
        st.markdown("#### 🔥 Q-Table Value Heatmap")

        if not st.session_state.selected_qtables:
            st.info("Select a Q-table file to display heatmap")
            return

        # File selector for heatmap
        file_options = st.session_state.selected_qtables
        file_names = [Path(f).name for f in file_options]

        selected_file_idx = st.selectbox(
            "Select Q-table for heatmap",
            range(len(file_options)),
            format_func=lambda x: file_names[x]
        )

        selected_file = file_options[selected_file_idx]

        # Load and display Q-table
        qtable = self.analyzer.load_qtable(selected_file)
        if qtable is not None:
            col1, col2 = st.columns([3, 1])

            with col1:
                # Create heatmap
                fig = self._create_qtable_heatmap(qtable, Path(selected_file).name)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Heatmap controls
                st.markdown("**Display Options**")

                colorscale = st.selectbox(
                    "Color Scale",
                    ["Viridis", "Plasma", "RdYlBu", "Blues", "Reds"],
                    key="heatmap_colorscale"
                )

                show_values = st.checkbox("Show Values", value=False, key="show_heatmap_values")

                # Update heatmap with new settings
                if st.button("Update Heatmap"):
                    fig = self._create_qtable_heatmap(
                        qtable,
                        Path(selected_file).name,
                        colorscale=colorscale.lower(),
                        show_values=show_values
                    )
                    st.plotly_chart(fig, use_container_width=True)

    def _create_qtable_heatmap(
        self,
        qtable: np.ndarray,
        title: str,
        colorscale: str = "viridis",
        show_values: bool = False
    ) -> go.Figure:
        """Create interactive Q-table heatmap."""

        fig = go.Figure(data=go.Heatmap(
            z=qtable,
            x=[f"Action {i}" for i in range(qtable.shape[1])],
            y=[f"State {i}" for i in range(qtable.shape[0])],
            colorscale=colorscale,
            text=qtable if show_values else None,
            texttemplate="%{text:.3f}" if show_values else None,
            hoverongaps=False,
            hovertemplate="State: %{y}<br>Action: %{x}<br>Q-Value: %{z:.4f}<extra></extra>"
        ))

        fig.update_layout(
            title=f"Q-Table Heatmap: {title}",
            xaxis_title="Actions",
            yaxis_title="States",
            height=max(400, min(800, qtable.shape[0] * 20)),
            font={'size': 12}
        )

        return fig

    def _render_convergence_trends(self):
        """Render convergence trend analysis."""
        st.markdown("#### 📈 Convergence Analysis")

        if len(st.session_state.selected_qtables) < 2:
            st.info("Select multiple Q-table files to show convergence trends")
            return

        # Extract timestamps and metrics for trend analysis
        trend_data = []
        for file_path in st.session_state.selected_qtables:
            if file_path in st.session_state.qtable_analysis_cache:
                metrics = st.session_state.qtable_analysis_cache[file_path]

                # Extract timestamp from filename or use file modification time
                file_name = Path(file_path).name
                timestamp = self._extract_timestamp_from_filename(file_name)

                trend_data.append({
                    'file': file_name,
                    'timestamp': timestamp,
                    'convergence_score': metrics.convergence_score,
                    'stability_measure': metrics.stability_measure,
                    'exploration_coverage': metrics.exploration_coverage,
                    'policy_entropy': metrics.policy_entropy
                })

        if trend_data:
            df = pd.DataFrame(trend_data)
            df = df.sort_values('timestamp')

            # Create convergence trend plot
            fig = self._create_convergence_trend_plot(df)
            st.plotly_chart(fig, use_container_width=True)

            # Show trend statistics
            self._display_trend_statistics(df)

    def _create_convergence_trend_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create convergence trend plot."""

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Convergence Score', 'Stability Measure', 'Exploration Coverage', 'Policy Entropy'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Convergence Score
        fig.add_trace(
            go.Scatter(x=df.index, y=df['convergence_score'], name='Convergence Score',
                      line={'color': 'blue', 'width': 2}, mode='lines+markers'),
            row=1, col=1
        )

        # Stability Measure
        fig.add_trace(
            go.Scatter(x=df.index, y=df['stability_measure'], name='Stability',
                      line={'color': 'green', 'width': 2}, mode='lines+markers'),
            row=1, col=2
        )

        # Exploration Coverage
        fig.add_trace(
            go.Scatter(x=df.index, y=df['exploration_coverage'], name='Exploration',
                      line={'color': 'orange', 'width': 2}, mode='lines+markers'),
            row=2, col=1
        )

        # Policy Entropy
        fig.add_trace(
            go.Scatter(x=df.index, y=df['policy_entropy'], name='Policy Entropy',
                      line={'color': 'red', 'width': 2}, mode='lines+markers'),
            row=2, col=2
        )

        fig.update_layout(
            title="Q-Learning Convergence Trends Over Time",
            height=600,
            showlegend=False
        )

        return fig

    def _render_policy_analysis(self):
        """Render policy quality analysis."""
        st.markdown("#### 🎯 Policy Quality Analysis")

        if not st.session_state.qtable_analysis_cache:
            st.info("No analysis results available")
            return

        # Policy quality metrics comparison
        policy_data = []
        for file_path, metrics in st.session_state.qtable_analysis_cache.items():
            policy_data.append({
                'File': Path(file_path).name,
                'Policy Entropy': metrics.policy_entropy,
                'Action Diversity': metrics.action_diversity,
                'State Value Variance': metrics.state_value_variance,
                'Convergence Score': metrics.convergence_score
            })

        df = pd.DataFrame(policy_data)

        # Create policy quality plot
        fig = px.scatter(
            df,
            x='Policy Entropy',
            y='Action Diversity',
            size='Convergence Score',
            color='State Value Variance',
            hover_name='File',
            title="Policy Quality Analysis",
            labels={
                'Policy Entropy': 'Policy Entropy (Higher = More Diverse)',
                'Action Diversity': 'Action Diversity Score',
                'State Value Variance': 'State Value Variance'
            }
        )

        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Policy insights
        self._display_policy_insights(df)

    def _render_exploration_visualization(self):
        """Render state-action exploration visualization."""
        st.markdown("#### 🗺️ State-Action Exploration")

        if not st.session_state.selected_qtables:
            st.info("Select Q-table files to analyze exploration patterns")
            return

        # File selector
        file_options = st.session_state.selected_qtables
        file_names = [Path(f).name for f in file_options]

        selected_file_idx = st.selectbox(
            "Select Q-table for exploration analysis",
            range(len(file_options)),
            format_func=lambda x: file_names[x],
            key="exploration_file_selector"
        )

        selected_file = file_options[selected_file_idx]
        qtable = self.analyzer.load_qtable(selected_file)

        if qtable is not None:
            # Create exploration heatmap (visited vs unvisited states)
            exploration_map = (qtable != 0).astype(int)

            fig = go.Figure(data=go.Heatmap(
                z=exploration_map,
                x=[f"Action {i}" for i in range(qtable.shape[1])],
                y=[f"State {i}" for i in range(qtable.shape[0])],
                colorscale=[[0, 'lightgray'], [1, 'darkgreen']],
                text=None,
                hoverongaps=False,
                hovertemplate="State: %{y}<br>Action: %{x}<br>Explored: %{z}<extra></extra>",
                colorbar={
                    'title': "Exploration Status",
                    'tickvals': [0, 1],
                    'ticktext': ['Unexplored', 'Explored']
                }
            ))

            fig.update_layout(
                title=f"State-Action Exploration Map: {Path(selected_file).name}",
                xaxis_title="Actions",
                yaxis_title="States",
                height=max(400, min(800, qtable.shape[0] * 15))
            )

            st.plotly_chart(fig, use_container_width=True)

            # Exploration statistics
            if selected_file in st.session_state.qtable_analysis_cache:
                metrics = st.session_state.qtable_analysis_cache[selected_file]

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Exploration Coverage", f"{metrics.exploration_coverage:.1%}")
                with col2:
                    st.metric("Visited States", metrics.visited_states)
                with col3:
                    st.metric("Unvisited States", metrics.unvisited_states)

    def _render_comparison_section(self):
        """Render Q-table comparison interface."""
        if len(st.session_state.selected_qtables) < 2:
            return

        st.subheader("🔄 Q-Table Comparison")

        col1, col2, col3 = st.columns([1, 1, 1])

        file_options = st.session_state.selected_qtables
        file_names = [Path(f).name for f in file_options]

        with col1:
            table1_idx = st.selectbox("First Q-Table", range(len(file_options)), format_func=lambda x: file_names[x])

        with col2:
            table2_idx = st.selectbox("Second Q-Table", range(len(file_options)), format_func=lambda x: file_names[x])

        with col3:
            if st.button("Compare Tables"):
                self._perform_qtable_comparison(file_options[table1_idx], file_options[table2_idx])

        # Display comparison results
        self._display_comparison_results()

    def _render_export_section(self):
        """Render export functionality."""
        st.subheader("📤 Export Analysis Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Export Metrics CSV"):
                self._export_metrics_csv()

        with col2:
            if st.button("Export Visualizations"):
                self._export_visualizations()

        with col3:
            if st.button("Generate Analysis Report"):
                self._generate_analysis_report()

    def _get_file_size(self, file_path: Path) -> str:
        """Get human-readable file size."""
        try:
            size = float(file_path.stat().st_size)
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024:
                    return f"{size:.1f} {unit}"
                size = size / 1024
            return f"{size:.1f} TB"
        except OSError:
            return "Unknown"

    def _perform_quick_analysis(self):
        """Perform quick analysis of selected Q-tables."""
        with st.spinner("Performing quick analysis..."):
            for file_path in st.session_state.selected_qtables:
                if file_path not in st.session_state.qtable_analysis_cache:
                    metrics = self.analyzer.analyze_qtable(file_path)
                    if metrics:
                        st.session_state.qtable_analysis_cache[file_path] = metrics

        st.success("Quick analysis completed!")
        st.rerun()

    def _extract_timestamp_from_filename(self, filename: str) -> str:
        """Extract timestamp from filename or return filename."""
        # Try to extract timestamp patterns from filename
        import re

        # Pattern for YYYYMMDD_HHMMSS
        pattern = r'(\d{8}_\d{6})'
        match = re.search(pattern, filename)

        if match:
            return match.group(1)

        # Return filename as fallback
        return filename

    def _display_trend_statistics(self, df: pd.DataFrame):
        """Display trend analysis statistics."""
        st.markdown("**Trend Analysis**")

        col1, col2 = st.columns(2)

        with col1:
            if len(df) >= 2:
                convergence_trend = df['convergence_score'].iloc[-1] - df['convergence_score'].iloc[0]
                stability_trend = df['stability_measure'].iloc[-1] - df['stability_measure'].iloc[0]

                st.metric("Convergence Improvement", f"{convergence_trend:+.3f}")
                st.metric("Stability Change", f"{stability_trend:+.3f}")

        with col2:
            max_convergence = df['convergence_score'].max()
            avg_exploration = df['exploration_coverage'].mean()

            st.metric("Peak Convergence", f"{max_convergence:.3f}")
            st.metric("Avg Exploration", f"{avg_exploration:.1%}")

    def _display_policy_insights(self, df: pd.DataFrame):
        """Display policy analysis insights."""
        st.markdown("**Policy Insights**")

        # Find best performing policies
        best_entropy = df.loc[df['Policy Entropy'].idxmax()]
        best_diversity = df.loc[df['Action Diversity'].idxmax()]
        best_convergence = df.loc[df['Convergence Score'].idxmax()]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Most Diverse Policy**")
            st.markdown(f"📁 {best_entropy['File']}")
            st.markdown(f"🎯 Entropy: {best_entropy['Policy Entropy']:.3f}")

        with col2:
            st.markdown("**Best Action Coverage**")
            st.markdown(f"📁 {best_diversity['File']}")
            st.markdown(f"🎯 Diversity: {best_diversity['Action Diversity']:.3f}")

        with col3:
            st.markdown("**Most Converged**")
            st.markdown(f"📁 {best_convergence['File']}")
            st.markdown(f"🎯 Score: {best_convergence['Convergence Score']:.3f}")

    def _perform_qtable_comparison(self, file1: str, file2: str):
        """Perform Q-table comparison."""
        comparison_key = f"{file1}|{file2}"

        if comparison_key not in st.session_state.comparison_results:
            with st.spinner("Comparing Q-tables..."):
                comparison = self.analyzer.compare_qtables(file1, file2)
                if comparison:
                    st.session_state.comparison_results[comparison_key] = comparison
                    st.success("Comparison completed!")
                else:
                    st.error("Comparison failed - tables may have incompatible shapes")

    def _display_comparison_results(self):
        """Display Q-table comparison results."""
        if not st.session_state.comparison_results:
            return

        for comparison_key, comparison in st.session_state.comparison_results.items():
            file1, file2 = comparison_key.split('|')

            st.markdown(f"**Comparison: {Path(file1).name} vs {Path(file2).name}**")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Policy Agreement", f"{comparison.policy_agreement:.1%}")

            with col2:
                st.metric("Convergence Change", f"{comparison.convergence_improvement:+.3f}")

            with col3:
                st.metric("Learning Progress", f"{comparison.learning_progress:+.3f}")

            with col4:
                st.metric("Stability Change", f"{comparison.stability_change:+.3f}")

    def _export_metrics_csv(self):
        """Export analysis metrics to CSV."""
        if st.session_state.qtable_analysis_cache:
            results = st.session_state.qtable_analysis_cache
            output_file = f"qtable_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

            self.analyzer.export_analysis_results(results, output_file)
            st.success(f"Metrics exported to {output_file}")
        else:
            st.warning("No analysis results to export")

    def _export_visualizations(self):
        """Export visualizations."""
        st.info("Visualization export functionality coming soon!")

    def _generate_analysis_report(self):
        """Generate comprehensive analysis report."""
        st.info("Analysis report generation coming soon!")


def render_qtable_analysis_interface():
    """Main function to render Q-table analysis interface."""
    component = QTableVisualization()
    return component.render_qtable_analysis_interface()


if __name__ == "__main__":
    # For testing the component
    st.title("Q-Table Analysis Component Test")
    result = render_qtable_analysis_interface()
    st.json(result)
