"""
Policy Evolution Visualization Component

This module provides interactive visualizations for tracking Q-learning
policy development over training episodes, including policy stability,
action frequency analysis, and learning curve visualization.

User Story 1.2.2: Policy Evolution Tracking
Created: 2025-07-31
Last Modified: 2025-07-31
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import policy evolution tracker
from analysis.policy_evolution_tracker import (
    POLICY_EVOLUTION_TRACKER,
    PolicyEvolutionMetrics,
    PolicyStability
)


class PolicyEvolutionVisualization:
    """Interactive policy evolution visualization component."""

    def __init__(self):
        """Initialize policy evolution visualization component."""
        self.tracker = POLICY_EVOLUTION_TRACKER

        # Initialize session state
        if 'policy_snapshots_loaded' not in st.session_state:
            st.session_state.policy_snapshots_loaded = False
        if 'policy_evolution_metrics' not in st.session_state:
            st.session_state.policy_evolution_metrics = None
        if 'selected_episodes' not in st.session_state:
            st.session_state.selected_episodes = []

    def render_policy_evolution_interface(self) -> Dict[str, Any]:
        """
        Render the main policy evolution tracking interface.
        
        Returns:
            Analysis results and metadata
        """
        st.header("📈 Policy Evolution Tracking")
        st.markdown("""
        Track Q-learning policy development over training episodes with convergence detection,
        action frequency analysis, and learning curve visualization for research insights.
        """)

        # Data loading section
        self._render_data_loading_section()

        # Analysis sections (only show if data is loaded)
        if st.session_state.policy_snapshots_loaded and self.tracker.policy_snapshots:
            metrics = self._render_analysis_overview()
            self._render_policy_visualization_tabs(metrics)
            self._render_export_section(metrics)

        return {
            'snapshots_loaded': len(self.tracker.policy_snapshots),
            'evolution_metrics': st.session_state.policy_evolution_metrics,
            'selected_episodes': st.session_state.selected_episodes
        }

    def _render_data_loading_section(self):
        """Render data loading and configuration interface."""
        st.subheader("📁 Policy Data Loading")

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            # File pattern configuration
            file_pattern = st.text_input(
                "Q-table File Pattern",
                value="*qtable*.pkl",
                help="Pattern to match Q-table files for policy analysis"
            )

            max_snapshots = st.number_input(
                "Maximum Snapshots",
                min_value=5,
                max_value=1000,
                value=50,
                help="Maximum number of policy snapshots to analyze"
            )

        with col2:
            if st.button("🔄 Load Policy Data", type="primary"):
                with st.spinner("Loading policy snapshots..."):
                    count = self.tracker.load_policy_snapshots_from_files(
                        file_pattern=file_pattern,
                        max_snapshots=max_snapshots
                    )

                    if count > 0:
                        st.session_state.policy_snapshots_loaded = True
                        st.success(f"✅ Loaded {count} policy snapshots!")
                        st.rerun()
                    else:
                        st.error("❌ No valid policy snapshots found")

        with col3:
            if st.button("📊 Quick Analysis"):
                if self.tracker.policy_snapshots:
                    with st.spinner("Analyzing policy evolution..."):
                        metrics = self.tracker.analyze_policy_evolution()
                        if metrics:
                            st.session_state.policy_evolution_metrics = metrics
                            st.success("✅ Analysis completed!")
                            st.rerun()
                        else:
                            st.error("❌ Analysis failed")
                else:
                    st.warning("⚠️ Load policy data first")

        # Show current status
        if self.tracker.policy_snapshots:
            st.info(f"📈 Currently loaded: {len(self.tracker.policy_snapshots)} policy snapshots")

    def _render_analysis_overview(self) -> Optional[PolicyEvolutionMetrics]:
        """Render policy evolution analysis overview."""
        if not st.session_state.policy_evolution_metrics:
            # Run quick analysis if not done yet
            metrics = self.tracker.analyze_policy_evolution()
            if metrics:
                st.session_state.policy_evolution_metrics = metrics
            else:
                st.error("❌ Unable to analyze policy evolution")
                return None

        metrics = st.session_state.policy_evolution_metrics

        st.subheader("📊 Policy Evolution Overview")

        # Key metrics display
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Training Episodes",
                metrics.total_episodes,
                delta=f"{metrics.snapshots_count} snapshots"
            )

        with col2:
            stability_delta = "🟢 Stable" if metrics.stability_status == PolicyStability.STABLE else "🟡 Learning"
            st.metric(
                "Policy Stability",
                f"{metrics.stability_score:.1%}",
                delta=stability_delta
            )

        with col3:
            convergence_text = f"Episode {metrics.convergence_episode}" if metrics.convergence_episode else "Not detected"
            st.metric(
                "Convergence Point",
                convergence_text,
                delta="Early convergence" if metrics.convergence_episode and metrics.convergence_episode < metrics.total_episodes * 0.5 else None
            )

        with col4:
            st.metric(
                "Action Changes",
                metrics.action_preference_changes,
                delta=f"{len(metrics.dominant_actions)} unique actions"
            )

        # Policy evolution status
        status_color = {
            PolicyStability.STABLE: "success",
            PolicyStability.CONVERGING: "info",
            PolicyStability.UNSTABLE: "warning",
            PolicyStability.OSCILLATING: "error",
            PolicyStability.UNKNOWN: "secondary"
        }

        status_message = {
            PolicyStability.STABLE: "✅ Policy has converged to a stable strategy",
            PolicyStability.CONVERGING: "🔄 Policy is converging towards stability",
            PolicyStability.UNSTABLE: "⚠️ Policy shows unstable behavior",
            PolicyStability.OSCILLATING: "🔄 Policy is oscillating between strategies",
            PolicyStability.UNKNOWN: "❓ Policy stability status is unclear"
        }

        getattr(st, status_color[metrics.stability_status])(
            status_message[metrics.stability_status]
        )

        return metrics

    def _render_policy_visualization_tabs(self, metrics: PolicyEvolutionMetrics):
        """Render policy visualization tabs."""
        viz_tabs = st.tabs([
            "📈 Policy Evolution Timeline",
            "🎯 Action Frequency Analysis",
            "📊 Learning Curves",
            "🔄 Policy Stability Analysis",
            "⚖️ Episode Comparison"
        ])

        with viz_tabs[0]:
            self._render_policy_evolution_timeline(metrics)

        with viz_tabs[1]:
            self._render_action_frequency_analysis(metrics)

        with viz_tabs[2]:
            self._render_learning_curves(metrics)

        with viz_tabs[3]:
            self._render_policy_stability_analysis(metrics)

        with viz_tabs[4]:
            self._render_episode_comparison()

    def _render_policy_evolution_timeline(self, metrics: PolicyEvolutionMetrics):
        """Render policy evolution timeline visualization."""
        st.markdown("#### 📈 Policy Evolution Over Time")

        if not self.tracker.policy_snapshots:
            st.info("No policy snapshots available")
            return

        # Prepare timeline data
        episodes = [s.episode for s in self.tracker.policy_snapshots]
        policy_entropies = [s.policy_entropy for s in self.tracker.policy_snapshots]
        state_coverages = [s.state_coverage for s in self.tracker.policy_snapshots]

        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Policy Entropy Evolution', 'State Coverage Evolution', 'Policy Changes Between Episodes'),
            vertical_spacing=0.12
        )

        # Policy entropy timeline
        fig.add_trace(
            go.Scatter(
                x=episodes,
                y=policy_entropies,
                mode='lines+markers',
                name='Policy Entropy',
                line=dict(color='blue', width=2),
                hovertemplate='Episode: %{x}<br>Entropy: %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )

        # State coverage timeline
        fig.add_trace(
            go.Scatter(
                x=episodes,
                y=state_coverages,
                mode='lines+markers',
                name='State Coverage',
                line=dict(color='green', width=2),
                hovertemplate='Episode: %{x}<br>Coverage: %{y:.1%}<extra></extra>'
            ),
            row=2, col=1
        )

        # Policy changes timeline
        if metrics.policy_changes:
            change_episodes = episodes[1:]  # Changes start from episode 1
            fig.add_trace(
                go.Scatter(
                    x=change_episodes,
                    y=metrics.policy_changes,
                    mode='lines+markers',
                    name='Policy Changes',
                    line=dict(color='red', width=2),
                    hovertemplate='Episode: %{x}<br>Changes: %{y}<extra></extra>'
                ),
                row=3, col=1
            )

        # Add convergence point marker if detected
        if metrics.convergence_episode:
            for row in range(1, 4):
                fig.add_vline(
                    x=metrics.convergence_episode,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text="Convergence",
                    row=row, col=1
                )

        fig.update_layout(
            title="Policy Evolution Timeline Analysis",
            height=800,
            showlegend=True
        )

        fig.update_xaxes(title_text="Training Episode", row=3, col=1)
        fig.update_yaxes(title_text="Entropy", row=1, col=1)
        fig.update_yaxes(title_text="Coverage %", row=2, col=1)
        fig.update_yaxes(title_text="Changes", row=3, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # Timeline insights
        self._render_timeline_insights(metrics)

    def _render_action_frequency_analysis(self, metrics: PolicyEvolutionMetrics):
        """Render action frequency analysis visualization."""
        st.markdown("#### 🎯 Action Usage Analysis")

        # Get action frequency matrix
        freq_matrix = self.tracker.get_action_frequency_matrix()
        if freq_matrix is None:
            st.info("No action frequency data available")
            return

        col1, col2 = st.columns([2, 1])

        with col1:
            # Action frequency heatmap over episodes
            action_cols = [col for col in freq_matrix.columns if col.startswith('Action_')]

            if action_cols:
                heatmap_data = freq_matrix[action_cols].values
                episodes = freq_matrix['Episode'].values

                fig = go.Figure(data=go.Heatmap(
                    z=heatmap_data.T,
                    x=episodes,
                    y=action_cols,
                    colorscale='Viridis',
                    hoverongaps=False,
                    hovertemplate='Episode: %{x}<br>Action: %{y}<br>Frequency: %{z}<extra></extra>'
                ))

                fig.update_layout(
                    title="Action Frequency Heatmap Over Episodes",
                    xaxis_title="Training Episode",
                    yaxis_title="Actions",
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Dominant action distribution
            st.markdown("**Overall Action Distribution**")

            actions = list(metrics.dominant_actions.keys())
            percentages = list(metrics.dominant_actions.values())

            fig_pie = go.Figure(data=[go.Pie(
                labels=[f'Action {a}' for a in actions],
                values=percentages,
                hole=0.3,
                hovertemplate='%{label}<br>Usage: %{value:.1%}<extra></extra>'
            )])

            fig_pie.update_layout(
                title="Action Usage Distribution",
                height=400
            )

            st.plotly_chart(fig_pie, use_container_width=True)

        # Action preference changes timeline
        st.markdown("**Action Preference Evolution**")

        if len(self.tracker.policy_snapshots) > 1:
            # Calculate dominant action for each episode
            dominant_actions_timeline = []
            for snapshot in self.tracker.policy_snapshots:
                if snapshot.action_frequencies:
                    dominant_action = max(snapshot.action_frequencies.items(), key=lambda x: x[1])[0]
                    dominant_actions_timeline.append(dominant_action)
                else:
                    dominant_actions_timeline.append(0)

            episodes = [s.episode for s in self.tracker.policy_snapshots]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=episodes,
                y=dominant_actions_timeline,
                mode='lines+markers',
                name='Dominant Action',
                line=dict(width=3),
                hovertemplate='Episode: %{x}<br>Dominant Action: %{y}<extra></extra>'
            ))

            fig.update_layout(
                title="Dominant Action Changes Over Time",
                xaxis_title="Training Episode",
                yaxis_title="Dominant Action ID",
                height=300
            )

            st.plotly_chart(fig, use_container_width=True)

    def _render_learning_curves(self, metrics: PolicyEvolutionMetrics):
        """Render learning curve visualization."""
        st.markdown("#### 📊 Learning Progress Analysis")

        if not metrics.performance_trend:
            st.info("No performance data available for learning curves. Performance data can be extracted from Q-table files with reward metadata.")

            # Show learning velocity instead
            if metrics.learning_velocity:
                episodes = list(range(1, len(metrics.learning_velocity) + 1))

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=episodes,
                    y=metrics.learning_velocity,
                    mode='lines+markers',
                    name='Learning Velocity',
                    line=dict(color='purple', width=2),
                    hovertemplate='Episode: %{x}<br>Velocity: %{y:.3f}<extra></extra>'
                ))

                fig.update_layout(
                    title="Learning Velocity (Rate of Policy Change)",
                    xaxis_title="Training Episode",
                    yaxis_title="Policy Change Rate",
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

                st.info("💡 **Learning Velocity**: Positive values indicate increasing policy changes, negative values indicate stabilization.")
        else:
            # Show performance trend
            performance_episodes = list(range(len(metrics.performance_trend)))

            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Episode Performance/Reward', 'Learning Velocity'),
                vertical_spacing=0.15
            )

            # Performance curve
            fig.add_trace(
                go.Scatter(
                    x=performance_episodes,
                    y=metrics.performance_trend,
                    mode='lines+markers',
                    name='Episode Reward',
                    line=dict(color='green', width=2)
                ),
                row=1, col=1
            )

            # Learning velocity
            if metrics.learning_velocity:
                velocity_episodes = list(range(len(metrics.learning_velocity)))
                fig.add_trace(
                    go.Scatter(
                        x=velocity_episodes,
                        y=metrics.learning_velocity,
                        mode='lines+markers',
                        name='Learning Velocity',
                        line=dict(color='purple', width=2)
                    ),
                    row=2, col=1
                )

            fig.update_layout(
                title="Learning Curves Analysis",
                height=600,
                showlegend=True
            )

            fig.update_xaxes(title_text="Training Episode", row=2, col=1)
            fig.update_yaxes(title_text="Reward", row=1, col=1)
            fig.update_yaxes(title_text="Velocity", row=2, col=1)

            st.plotly_chart(fig, use_container_width=True)

        # Exploration decay analysis
        if metrics.exploration_decay:
            st.markdown("**Exploration Decay Analysis**")

            decay_episodes = list(range(1, len(metrics.exploration_decay) + 1))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=decay_episodes,
                y=metrics.exploration_decay,
                mode='lines+markers',
                name='Exploration Decay Rate',
                line=dict(color='orange', width=2),
                hovertemplate='Episode: %{x}<br>Decay Rate: %{y:.3f}<extra></extra>'
            ))

            fig.update_layout(
                title="Exploration Decay Over Time",
                xaxis_title="Training Episode",
                yaxis_title="Decay Rate",
                height=350
            )

            st.plotly_chart(fig, use_container_width=True)

    def _render_policy_stability_analysis(self, metrics: PolicyEvolutionMetrics):
        """Render policy stability analysis."""
        st.markdown("#### 🔄 Policy Stability Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Stability score gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = metrics.stability_score * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Policy Stability Score (%)"},
                delta = {'reference': 95, 'increasing': {'color': "green"}},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 95], 'color': "orange"},
                        {'range': [95, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 95
                    }
                }
            ))

            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Policy changes distribution
            if metrics.policy_changes:
                fig = go.Figure(data=[go.Histogram(
                    x=metrics.policy_changes,
                    nbinsx=20,
                    name='Policy Changes',
                    hovertemplate='Changes: %{x}<br>Frequency: %{y}<extra></extra>'
                )])

                fig.update_layout(
                    title="Distribution of Policy Changes",
                    xaxis_title="Number of State Changes",
                    yaxis_title="Frequency",
                    height=300
                )

                st.plotly_chart(fig, use_container_width=True)

        # Stability insights
        self._render_stability_insights(metrics)

    def _render_episode_comparison(self):
        """Render episode-to-episode policy comparison."""
        st.markdown("#### ⚖️ Episode Comparison Analysis")

        if len(self.tracker.policy_snapshots) < 2:
            st.info("Need at least 2 episodes for comparison")
            return

        episodes = [s.episode for s in self.tracker.policy_snapshots]

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            reference_episode = st.selectbox(
                "Reference Episode",
                options=episodes,
                index=0,
                help="Episode to compare others against"
            )

        with col2:
            compare_episodes = st.multiselect(
                "Episodes to Compare",
                options=episodes,
                default=episodes[-3:] if len(episodes) >= 3 else episodes[1:],
                help="Select episodes to compare with reference"
            )

        with col3:
            if st.button("📊 Generate Comparison"):
                if compare_episodes:
                    self._generate_episode_comparison(reference_episode, compare_episodes)
                else:
                    st.warning("Select episodes to compare")

    def _generate_episode_comparison(self, reference_episode: int, compare_episodes: List[int]):
        """Generate detailed episode comparison."""
        # Get policy comparison matrix
        similarity_matrix = self.tracker.get_policy_comparison_matrix(reference_episode)

        if similarity_matrix is None:
            st.error("Unable to generate comparison matrix")
            return

        # Find snapshots for comparison
        ref_snapshot = None
        compare_snapshots = []

        for snapshot in self.tracker.policy_snapshots:
            if snapshot.episode == reference_episode:
                ref_snapshot = snapshot
            if snapshot.episode in compare_episodes:
                compare_snapshots.append(snapshot)

        if not ref_snapshot or not compare_snapshots:
            st.error("Unable to find comparison snapshots")
            return

        # Calculate similarity scores
        similarities = []
        for snapshot in compare_snapshots:
            episode_idx = next(i for i, s in enumerate(self.tracker.policy_snapshots) if s.episode == snapshot.episode)
            similarity = np.mean(similarity_matrix[episode_idx])
            similarities.append((snapshot.episode, similarity))

        # Display comparison results
        st.markdown(f"**Comparison with Reference Episode {reference_episode}**")

        comparison_data = []
        for episode, similarity in similarities:
            comparison_data.append({
                'Episode': episode,
                'Policy Similarity': f"{similarity:.1%}",
                'Policy Entropy': f"{next(s.policy_entropy for s in compare_snapshots if s.episode == episode):.3f}",
                'State Coverage': f"{next(s.state_coverage for s in compare_snapshots if s.episode == episode):.1%}"
            })

        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)

        # Similarity visualization
        episodes_list = [ep for ep, _ in similarities]
        similarity_scores = [sim for _, sim in similarities]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"Episode {ep}" for ep in episodes_list],
            y=similarity_scores,
            name='Policy Similarity',
            hovertemplate='Episode: %{x}<br>Similarity: %{y:.1%}<extra></extra>'
        ))

        fig.update_layout(
            title=f"Policy Similarity to Reference Episode {reference_episode}",
            xaxis_title="Comparison Episodes",
            yaxis_title="Similarity Score",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_timeline_insights(self, metrics: PolicyEvolutionMetrics):
        """Render insights from timeline analysis."""
        st.markdown("**🔍 Timeline Insights**")

        insights = []

        # Convergence insight
        if metrics.convergence_episode:
            convergence_pct = (metrics.convergence_episode / metrics.total_episodes) * 100
            if convergence_pct < 25:
                insights.append(f"🚀 **Early Convergence**: Policy converged at episode {metrics.convergence_episode} ({convergence_pct:.1f}% of training)")
            elif convergence_pct < 50:
                insights.append(f"✅ **Good Convergence**: Policy converged at episode {metrics.convergence_episode} ({convergence_pct:.1f}% of training)")
            else:
                insights.append(f"🐌 **Late Convergence**: Policy converged at episode {metrics.convergence_episode} ({convergence_pct:.1f}% of training)")
        else:
            insights.append("⚠️ **No Convergence Detected**: Policy may need more training episodes")

        # Stability insight
        if metrics.stability_score >= 0.95:
            insights.append("🎯 **Highly Stable**: Policy shows excellent stability")
        elif metrics.stability_score >= 0.8:
            insights.append("✅ **Stable**: Policy demonstrates good stability")
        else:
            insights.append("⚠️ **Unstable**: Policy shows instability, consider adjusting learning parameters")

        # Action diversity insight
        num_actions = len(metrics.dominant_actions)
        if num_actions == 1:
            insights.append("🎯 **Single Strategy**: Policy converged to one dominant action")
        elif num_actions <= 3:
            insights.append(f"🎯 **Focused Strategy**: Policy uses {num_actions} main actions")
        else:
            insights.append(f"🌟 **Diverse Strategy**: Policy utilizes {num_actions} different actions")

        for insight in insights:
            st.markdown(insight)

    def _render_stability_insights(self, metrics: PolicyEvolutionMetrics):
        """Render insights from stability analysis."""
        st.markdown("**🔍 Stability Insights**")

        insights = []

        # Stability status insights
        status_insights = {
            PolicyStability.STABLE: "🎯 **Optimal**: Policy has reached a stable, consistent strategy",
            PolicyStability.CONVERGING: "📈 **Progressing**: Policy is moving towards stability",
            PolicyStability.UNSTABLE: "⚠️ **Needs Attention**: Consider reducing learning rate or increasing episodes",
            PolicyStability.OSCILLATING: "🔄 **Oscillating**: Policy may be stuck between strategies",
            PolicyStability.UNKNOWN: "❓ **Unclear**: More data needed for stability assessment"
        }

        insights.append(status_insights[metrics.stability_status])

        # Policy changes insight
        if metrics.policy_changes:
            avg_changes = np.mean(metrics.policy_changes)

            if avg_changes < 5:
                insights.append(f"✅ **Low Volatility**: Average {avg_changes:.1f} changes per episode")
            elif avg_changes < 15:
                insights.append(f"📊 **Moderate Changes**: Average {avg_changes:.1f} changes per episode")
            else:
                insights.append(f"⚠️ **High Volatility**: Average {avg_changes:.1f} changes per episode")

        # Action preference changes insight
        if metrics.action_preference_changes == 0:
            insights.append("🎯 **Consistent Strategy**: No dominant action changes")
        elif metrics.action_preference_changes <= 3:
            insights.append(f"📊 **Stable Preferences**: {metrics.action_preference_changes} strategy shifts")
        else:
            insights.append(f"🔄 **Dynamic Strategy**: {metrics.action_preference_changes} major strategy changes")

        for insight in insights:
            st.markdown(insight)

    def _render_export_section(self, metrics: PolicyEvolutionMetrics):
        """Render export functionality for policy analysis."""
        st.subheader("📤 Export Policy Analysis")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📊 Export Metrics CSV"):
                output_file = f"policy_evolution_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                self.tracker.export_evolution_analysis(metrics, output_file)
                st.success(f"✅ Metrics exported to {output_file}")

        with col2:
            if st.button("📈 Export Action Matrix"):
                freq_matrix = self.tracker.get_action_frequency_matrix()
                if freq_matrix is not None:
                    output_file = f"action_frequency_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    freq_matrix.to_csv(output_file, index=False)
                    st.success(f"✅ Action matrix exported to {output_file}")
                else:
                    st.error("❌ No action frequency data available")

        with col3:
            if st.button("📋 Generate Report"):
                self._generate_policy_evolution_report(metrics)

    def _generate_policy_evolution_report(self, metrics: PolicyEvolutionMetrics):
        """Generate comprehensive policy evolution report."""
        report = f"""
# Policy Evolution Analysis Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

- **Training Episodes**: {metrics.total_episodes}
- **Policy Stability**: {metrics.stability_score:.1%} ({metrics.stability_status.value})
- **Convergence**: {'Episode ' + str(metrics.convergence_episode) if metrics.convergence_episode else 'Not detected'}
- **Action Diversity**: {len(metrics.dominant_actions)} unique actions used

## Key Findings

### Policy Stability
The policy achieved a stability score of {metrics.stability_score:.1%}, indicating {metrics.stability_status.value} behavior.

### Action Usage
{'The policy shows focused behavior with ' + str(len(metrics.dominant_actions)) + ' main actions used.' if len(metrics.dominant_actions) <= 3 else 'The policy demonstrates diverse action selection across ' + str(len(metrics.dominant_actions)) + ' different actions.'}

### Learning Progress
{'Convergence was detected at episode ' + str(metrics.convergence_episode) + ', suggesting efficient learning.' if metrics.convergence_episode else 'No clear convergence point was detected, which may indicate the need for extended training.'}

## Recommendations

{"- ✅ Policy is well-trained and stable" if metrics.stability_score >= 0.9 else "- ⚠️ Consider additional training or parameter tuning"}
{"- 🎯 Good convergence timing" if metrics.convergence_episode and metrics.convergence_episode < metrics.total_episodes * 0.5 else "- 📈 Monitor for convergence in extended training"}

---
*Generated by MFC Policy Evolution Tracker*
        """

        st.markdown(report)

        # Download button for report
        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name=f"policy_evolution_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )


def render_policy_evolution_interface():
    """Main function to render the policy evolution interface."""
    component = PolicyEvolutionVisualization()
    return component.render_policy_evolution_interface()


if __name__ == "__main__":
    # For testing the component
    st.title("Policy Evolution Tracking Component Test")
    result = render_policy_evolution_interface()
    st.json(result)
