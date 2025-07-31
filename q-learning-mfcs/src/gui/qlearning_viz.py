#!/usr/bin/env python3
"""
Advanced Q-Learning Visualization Components

This module provides specialized visualization components for Q-learning 
algorithm analysis in MFC systems, designed for researchers and practitioners
studying reinforcement learning applications.

Features:
- Interactive Q-table visualization with heatmaps
- Policy evolution tracking and comparison
- Action selection analysis and patterns
- Learning convergence monitoring
- Reward landscape visualization
- Performance metrics dashboard

Created: 2025-07-31
Literature References:
1. Sutton, R.S. & Barto, A.G. (2018). "Reinforcement Learning: An Introduction"
2. Watkins, C.J.C.H. (1989). "Learning from delayed rewards"
3. Mnih, V. et al. (2015). "Human-level control through deep reinforcement learning"
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Optional, Tuple
import json
import pickle
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Import existing configurations
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class QLearningVisualizationType(Enum):
    """Types of Q-learning visualizations available."""
    Q_TABLE_HEATMAP = "q_table_heatmap"
    POLICY_EVOLUTION = "policy_evolution"
    ACTION_DISTRIBUTION = "action_distribution"
    LEARNING_CURVES = "learning_curves"
    REWARD_LANDSCAPE = "reward_landscape"
    STATE_VISITATION = "state_visitation"
    EXPLORATION_VS_EXPLOITATION = "exploration_exploitation"

@dataclass
class QLearningVisualizationConfig:
    """Configuration for Q-learning visualizations."""
    colormap: str = "RdYlBu_r"
    show_values: bool = True
    interpolation: str = "nearest"
    figure_size: Tuple[int, int] = (12, 8)
    font_size: int = 10
    show_policy_arrows: bool = True
    animation_speed: float = 0.5
    confidence_intervals: bool = True

class QLearningVisualizer:
    """Advanced Q-learning visualization component."""
    
    def __init__(self, config: QLearningVisualizationConfig = QLearningVisualizationConfig()):
        """Initialize Q-learning visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config
        self._initialize_styling()
    
    def _initialize_styling(self):
        """Initialize custom styling for Q-learning components."""
        st.markdown("""
        <style>
        .qlearning-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            padding: 20px;
            margin: 15px 0;
            color: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .qlearning-header {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 15px;
            text-align: center;
            border-bottom: 2px solid rgba(255,255,255,0.3);
            padding-bottom: 10px;
        }
        
        .metric-highlight {
            background: rgba(255,255,255,0.2);
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            text-align: center;
        }
        
        .convergence-indicator {
            font-size: 18px;
            font-weight: bold;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            margin: 10px 0;
        }
        
        .convergence-good {
            background-color: #27AE60;
            color: white;
        }
        
        .convergence-poor {
            background-color: #E74C3C;
            color: white;
        }
        
        .convergence-moderate {
            background-color: #F39C12;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_qlearning_dashboard(
        self,
        q_table: Optional[np.ndarray] = None,
        training_history: Optional[Dict[str, List[float]]] = None,
        current_policy: Optional[np.ndarray] = None,
        title: str = "Q-Learning Analysis Dashboard"
    ) -> Dict[str, go.Figure]:
        """Render comprehensive Q-learning analysis dashboard.
        
        Args:
            q_table: Current Q-table values
            training_history: Dictionary with training metrics history
            current_policy: Current policy derived from Q-table
            title: Dashboard title
            
        Returns:
            Dictionary of generated figures
        """
        st.markdown('<div class="qlearning-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="qlearning-header">{title}</div>', unsafe_allow_html=True)
        
        figures = {}
        
        # Dashboard tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔥 Q-Table Analysis",
            "📈 Learning Progress", 
            "🎯 Policy Visualization",
            "📊 Performance Metrics"
        ])
        
        with tab1:
            if q_table is not None:
                figures['q_table'] = self._render_qtable_analysis(q_table)
            else:
                st.info("Q-table data not available. Load a trained model to see Q-table analysis.")
        
        with tab2:
            if training_history is not None:
                figures['learning_curves'] = self._render_learning_curves(training_history)
            else:
                st.info("Training history not available. Run training with logging enabled.")
        
        with tab3:
            if current_policy is not None:
                figures['policy'] = self._render_policy_visualization(current_policy, q_table)
            else:
                st.info("Policy data not available. Requires Q-table to derive policy.")
        
        with tab4:
            figures['metrics'] = self._render_performance_metrics(q_table, training_history)
        
        st.markdown('</div>', unsafe_allow_html=True)
        return figures
    
    def _render_qtable_analysis(self, q_table: np.ndarray) -> go.Figure:
        """Render Q-table heatmap and analysis.
        
        Args:
            q_table: Q-table array with shape (n_states, n_actions)
            
        Returns:
            Plotly figure for Q-table visualization
        """
        st.markdown("### 🔥 Q-Table Heatmap Analysis")
        
        # Q-table statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("States", q_table.shape[0])
        with col2:
            st.metric("Actions", q_table.shape[1])
        with col3:
            st.metric("Max Q-Value", f"{np.max(q_table):.3f}")
        with col4:
            st.metric("Min Q-Value", f"{np.min(q_table):.3f}")
        
        # Interactive Q-table heatmap
        fig = go.Figure(data=go.Heatmap(
            z=q_table,
            x=[f"Action {i}" for i in range(q_table.shape[1])],
            y=[f"State {i}" for i in range(q_table.shape[0])],
            colorscale='RdYlBu_r',
            showscale=True,
            text=q_table,
            texttemplate="%{text:.3f}",
            textfont={"size": 8},
            hoverongaps=False,
            hovertemplate="State: %{y}<br>Action: %{x}<br>Q-Value: %{z:.4f}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Q-Table Heatmap: State-Action Values",
            title_font_size=16,
            xaxis_title="Actions",
            yaxis_title="States",
            height=600,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Q-table analysis insights
        self._display_qtable_insights(q_table)
        
        return fig
    
    def _display_qtable_insights(self, q_table: np.ndarray):
        """Display insights from Q-table analysis."""
        st.markdown("#### 🔍 Q-Table Insights")
        
        # Calculate insights
        best_actions = np.argmax(q_table, axis=1)
        action_counts = np.bincount(best_actions, minlength=q_table.shape[1])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Policy Summary:**")
            for action_id, count in enumerate(action_counts):
                percentage = (count / len(best_actions)) * 100
                st.write(f"• Action {action_id}: {count} states ({percentage:.1f}%)")
        
        with col2:
            st.markdown("**Q-Value Distribution:**")
            st.write(f"• Mean Q-value: {np.mean(q_table):.4f}")
            st.write(f"• Std deviation: {np.std(q_table):.4f}")
            st.write(f"• Value range: {np.max(q_table) - np.min(q_table):.4f}")
            
            # Convergence indicator
            convergence_score = self._calculate_convergence_score(q_table)
            if convergence_score > 0.8:
                st.markdown(
                    f'<div class="convergence-indicator convergence-good">'
                    f'✅ Well Converged (Score: {convergence_score:.2f})</div>',
                    unsafe_allow_html=True
                )
            elif convergence_score > 0.5:
                st.markdown(
                    f'<div class="convergence-indicator convergence-moderate">'
                    f'⚠️ Partially Converged (Score: {convergence_score:.2f})</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="convergence-indicator convergence-poor">'
                    f'❌ Poor Convergence (Score: {convergence_score:.2f})</div>',
                    unsafe_allow_html=True
                )
    
    def _calculate_convergence_score(self, q_table: np.ndarray) -> float:
        """Calculate convergence score for Q-table.
        
        Args:
            q_table: Q-table array
            
        Returns:
            Convergence score between 0 and 1
        """
        # Calculate relative difference between best and second-best actions
        sorted_q = np.sort(q_table, axis=1)
        if q_table.shape[1] < 2:
            return 1.0
        
        best_values = sorted_q[:, -1]
        second_best = sorted_q[:, -2]
        
        # Avoid division by zero
        differences = np.abs(best_values - second_best)
        max_values = np.maximum(np.abs(best_values), np.abs(second_best))
        
        # Calculate relative differences (0 = no difference, 1 = completely different)
        relative_diffs = np.divide(
            differences, 
            max_values + 1e-10,  # Small epsilon to avoid division by zero
            out=np.zeros_like(differences),
            where=(max_values != 0)
        )
        
        # Convergence score: higher when actions are clearly differentiated
        return float(np.mean(relative_diffs))
    
    def _render_learning_curves(self, training_history: Dict[str, List[float]]) -> go.Figure:
        """Render learning curves and training progress.
        
        Args:
            training_history: Dictionary with training metrics
            
        Returns:
            Plotly figure for learning curves
        """
        st.markdown("### 📈 Learning Progress Analysis")
        
        # Create subplots for different metrics
        metrics = list(training_history.keys())
        n_metrics = len(metrics)
        
        if n_metrics == 0:
            st.warning("No training metrics available")
            return go.Figure()
        
        fig = make_subplots(
            rows=(n_metrics + 1) // 2,
            cols=2,
            subplot_titles=metrics,
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, (metric_name, values) in enumerate(training_history.items()):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            # Convert to numpy array for easier manipulation
            values_array = np.array(values)
            episodes = np.arange(len(values_array))
            
            # Add main trace
            fig.add_trace(
                go.Scatter(
                    x=episodes,
                    y=values_array,
                    name=metric_name,
                    line=dict(color=colors[i % len(colors)], width=2),
                    mode='lines',
                    hovertemplate=f"{metric_name}: %{{y:.4f}}<br>Episode: %{{x}}<extra></extra>"
                ),
                row=row, col=col
            )
            
            # Add smoothed trend line if enough data points
            if len(values_array) > 10:
                window_size = max(len(values_array) // 20, 5)
                smoothed = pd.Series(values_array).rolling(window=window_size, center=True).mean()
                
                fig.add_trace(
                    go.Scatter(
                        x=episodes,
                        y=smoothed,
                        name=f"{metric_name} (trend)",
                        line=dict(color=colors[i % len(colors)], width=3, dash='dash'),
                        mode='lines',
                        opacity=0.7,
                        hovertemplate=f"{metric_name} Trend: %{{y:.4f}}<br>Episode: %{{x}}<extra></extra>"
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title="Learning Curves: Training Progress Over Time",
            title_font_size=16,
            height=400 * ((n_metrics + 1) // 2),
            showlegend=False,
            template="plotly_white"
        )
        
        # Update x-axis labels
        fig.update_xaxes(title_text="Episode/Iteration")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Learning statistics
        self._display_learning_statistics(training_history)
        
        return fig
    
    def _display_learning_statistics(self, training_history: Dict[str, List[float]]):
        """Display learning statistics and insights."""
        st.markdown("#### 📊 Learning Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Training Summary:**")
            for metric_name, values in training_history.items():
                if values:
                    final_value = values[-1]
                    initial_value = values[0]
                    improvement = ((final_value - initial_value) / abs(initial_value + 1e-10)) * 100
                    
                    st.write(f"• **{metric_name}**:")
                    st.write(f"  - Initial: {initial_value:.4f}")
                    st.write(f"  - Final: {final_value:.4f}")
                    st.write(f"  - Change: {improvement:+.1f}%")
        
        with col2:
            st.markdown("**Convergence Analysis:**")
            
            # Analyze convergence for each metric
            for metric_name, values in training_history.items():
                if len(values) > 20:
                    # Calculate coefficient of variation for last 20% of training
                    last_portion = values[int(len(values) * 0.8):]
                    cv = np.std(last_portion) / (abs(np.mean(last_portion)) + 1e-10)
                    
                    if cv < 0.1:
                        status = "✅ Stable"
                    elif cv < 0.3:
                        status = "⚠️ Moderate"
                    else:
                        status = "❌ Unstable"
                    
                    st.markdown(f"• **{metric_name}**: {status} (CV: {cv:.3f})")
    
    def _render_policy_visualization(
        self, 
        policy: np.ndarray, 
        q_table: Optional[np.ndarray] = None
    ) -> go.Figure:
        """Render policy visualization with action preferences.
        
        Args:
            policy: Policy array with preferred actions per state
            q_table: Optional Q-table for additional analysis
            
        Returns:
            Plotly figure for policy visualization
        """
        st.markdown("### 🎯 Policy Visualization")
        
        # Policy statistics
        action_distribution = np.bincount(policy)
        n_actions = len(action_distribution)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Action distribution pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=[f"Action {i}" for i in range(n_actions)],
                values=action_distribution,
                hole=0.3,
                textinfo='label+percent',
                textposition='auto',
                hovertemplate="Action %{label}<br>States: %{value}<br>Percentage: %{percent}<extra></extra>"
            )])
            
            fig_pie.update_layout(
                title="Policy Action Distribution",
                title_font_size=14,
                height=400
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Policy heatmap (if Q-table available)
            if q_table is not None:
                policy_confidence = self._calculate_policy_confidence(q_table)
                
                fig_conf = go.Figure(data=go.Heatmap(
                    z=policy_confidence.reshape(-1, 1),
                    y=[f"State {i}" for i in range(len(policy_confidence))],
                    x=["Confidence"],
                    colorscale='Viridis',
                    showscale=True,
                    hovertemplate="State: %{y}<br>Confidence: %{z:.3f}<extra></extra>"
                ))
                
                fig_conf.update_layout(
                    title="Policy Confidence by State",
                    title_font_size=14,
                    height=400,
                    yaxis=dict(autorange="reversed")
                )
                
                st.plotly_chart(fig_conf, use_container_width=True)
            else:
                st.info("Q-table required for confidence analysis")
        
        # Policy analysis
        self._display_policy_analysis(policy, q_table)
        
        return fig_pie
    
    def _calculate_policy_confidence(self, q_table: np.ndarray) -> np.ndarray:
        """Calculate confidence scores for policy decisions.
        
        Args:
            q_table: Q-table array
            
        Returns:
            Array of confidence scores per state
        """
        # Calculate confidence as difference between best and second-best actions
        sorted_q = np.sort(q_table, axis=1)
        
        if q_table.shape[1] < 2:
            return np.ones(q_table.shape[0])
        
        best_values = sorted_q[:, -1]
        second_best = sorted_q[:, -2]
        
        # Normalize confidence by the absolute values
        max_abs = np.maximum(np.abs(best_values), np.abs(second_best))
        confidence = np.divide(
            np.abs(best_values - second_best),
            max_abs + 1e-10,
            out=np.zeros_like(best_values),
            where=(max_abs != 0)
        )
        
        return confidence
    
    def _display_policy_analysis(self, policy: np.ndarray, q_table: Optional[np.ndarray]):
        """Display policy analysis and insights."""
        st.markdown("#### 🔍 Policy Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Policy Statistics:**")
            action_counts = np.bincount(policy)
            total_states = len(policy)
            
            for action_id, count in enumerate(action_counts):
                percentage = (count / total_states) * 100
                st.write(f"• Action {action_id}: {count}/{total_states} states ({percentage:.1f}%)")
            
            # Policy diversity (entropy)
            probabilities = action_counts / total_states
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
            max_entropy = np.log(len(action_counts))
            diversity = entropy / max_entropy if max_entropy > 0 else 0
            
            st.write(f"• **Policy Diversity**: {diversity:.3f} (0=deterministic, 1=uniform)")
        
        with col2:
            if q_table is not None:
                st.markdown("**Confidence Analysis:**")
                confidence_scores = self._calculate_policy_confidence(q_table)
                
                st.write(f"• Mean Confidence: {np.mean(confidence_scores):.3f}")
                st.write(f"• Min Confidence: {np.min(confidence_scores):.3f}")
                st.write(f"• Max Confidence: {np.max(confidence_scores):.3f}")
                
                # Identify low-confidence states
                low_confidence_threshold = 0.3
                low_conf_states = np.where(confidence_scores < low_confidence_threshold)[0]
                
                if len(low_conf_states) > 0:
                    st.warning(f"📊 {len(low_conf_states)} states have low confidence (<{low_confidence_threshold})")
                    st.write("Low confidence states:", low_conf_states[:10].tolist())
                else:
                    st.success("✅ All states have high policy confidence")
    
    def _render_performance_metrics(
        self, 
        q_table: Optional[np.ndarray], 
        training_history: Optional[Dict[str, List[float]]]
    ) -> go.Figure:
        """Render comprehensive performance metrics dashboard.
        
        Args:
            q_table: Q-table array
            training_history: Training history dictionary
            
        Returns:
            Plotly figure for performance metrics
        """
        st.markdown("### 📊 Performance Metrics Dashboard")
        
        # Performance summary cards
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate metrics
        if q_table is not None:
            convergence_score = self._calculate_convergence_score(q_table)
            policy_diversity = self._calculate_policy_diversity(q_table)
            value_stability = self._calculate_value_stability(q_table)
            
            with col1:
                st.markdown(
                    f'<div class="metric-highlight">'
                    f'<h3>{convergence_score:.3f}</h3>'
                    f'<p>Convergence Score</p></div>',
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(
                    f'<div class="metric-highlight">'
                    f'<h3>{policy_diversity:.3f}</h3>'
                    f'<p>Policy Diversity</p></div>',
                    unsafe_allow_html=True
                )
            
            with col3:
                st.markdown(
                    f'<div class="metric-highlight">'
                    f'<h3>{value_stability:.3f}</h3>'
                    f'<p>Value Stability</p></div>',
                    unsafe_allow_html=True
                )
            
            with col4:
                exploration_rate = self._estimate_exploration_rate(q_table)
                st.markdown(
                    f'<div class="metric-highlight">'
                    f'<h3>{exploration_rate:.3f}</h3>'
                    f'<p>Est. Exploration</p></div>',
                    unsafe_allow_html=True
                )
        
        # Performance trends
        if training_history:
            self._render_performance_trends(training_history)
        
        # Performance recommendations
        self._render_performance_recommendations(q_table, training_history)
        
        return go.Figure()  # Placeholder figure
    
    def _calculate_policy_diversity(self, q_table: np.ndarray) -> float:
        """Calculate policy diversity score."""
        policy = np.argmax(q_table, axis=1)
        action_counts = np.bincount(policy, minlength=q_table.shape[1])
        probabilities = action_counts / len(policy)
        
        # Calculate entropy (diversity measure)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        max_entropy = np.log(q_table.shape[1])
        
        return entropy / max_entropy if max_entropy > 0 else 0
    
    def _calculate_value_stability(self, q_table: np.ndarray) -> float:
        """Calculate Q-value stability across states."""
        # Coefficient of variation across states for each action
        action_cvs = []
        for action in range(q_table.shape[1]):
            action_values = q_table[:, action]
            cv = np.std(action_values) / (abs(np.mean(action_values)) + 1e-10)
            action_cvs.append(cv)
        
        # Return inverse of mean CV (higher = more stable)
        mean_cv = np.mean(action_cvs)
        return 1 / (1 + mean_cv)
    
    def _estimate_exploration_rate(self, q_table: np.ndarray) -> float:
        """Estimate current exploration rate based on Q-table characteristics."""
        # Calculate how "spread out" the Q-values are
        q_ranges = np.max(q_table, axis=1) - np.min(q_table, axis=1)
        mean_range = np.mean(q_ranges)
        
        # Normalize by typical Q-value scale
        mean_abs_q = np.mean(np.abs(q_table))
        normalized_range = mean_range / (mean_abs_q + 1e-10)
        
        # Convert to exploration estimate (0 = no exploration, 1 = full exploration)
        return min(normalized_range, 1.0)
    
    def _render_performance_trends(self, training_history: Dict[str, List[float]]):
        """Render performance trends over training."""
        st.markdown("#### 📈 Performance Trends")
        
        # Calculate performance trend indicators
        trend_indicators = {}
        
        for metric_name, values in training_history.items():
            if len(values) > 10:
                # Calculate trend using linear regression slope
                x = np.arange(len(values))
                y = np.array(values)
                
                # Simple linear regression
                slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
                trend_indicators[metric_name] = slope
        
        # Display trends
        if trend_indicators:
            cols = st.columns(len(trend_indicators))
            
            for i, (metric, slope) in enumerate(trend_indicators.items()):
                with cols[i]:
                    if slope > 0.01:
                        trend_emoji = "📈"
                        trend_text = "Improving"
                        color = "green"
                    elif slope < -0.01:
                        trend_emoji = "📉"
                        trend_text = "Declining"
                        color = "red"
                    else:
                        trend_emoji = "➡️"
                        trend_text = "Stable"
                        color = "blue"
                    
                    st.markdown(
                        f'<div style="text-align: center; color: {color};">'
                        f'<h4>{trend_emoji}</h4>'
                        f'<p><strong>{metric}</strong><br>{trend_text}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
    
    def _render_performance_recommendations(
        self,
        q_table: Optional[np.ndarray],
        training_history: Optional[Dict[str, List[float]]]
    ):
        """Render performance improvement recommendations."""
        st.markdown("#### 💡 Performance Recommendations")
        
        recommendations = []
        
        if q_table is not None:
            convergence_score = self._calculate_convergence_score(q_table)
            policy_diversity = self._calculate_policy_diversity(q_table)
            
            if convergence_score < 0.5:
                recommendations.append(
                    "🔄 **Low Convergence**: Consider increasing training episodes or reducing learning rate"
                )
            
            if policy_diversity < 0.2:
                recommendations.append(
                    "🎯 **Low Diversity**: Policy may be too deterministic - consider increasing exploration"
                )
            elif policy_diversity > 0.8:
                recommendations.append(
                    "🎲 **High Diversity**: Policy may be too random - consider decreasing exploration rate"
                )
        
        if training_history:
            # Check for learning stagnation
            if 'reward' in training_history:
                recent_rewards = training_history['reward'][-20:] if len(training_history['reward']) > 20 else training_history['reward']
                if len(recent_rewards) > 5 and np.std(recent_rewards) < 0.01:
                    recommendations.append(
                        "📊 **Learning Stagnation**: Reward variance is very low - consider adjusting hyperparameters"
                    )
        
        if not recommendations:
            recommendations.append("✅ **Good Performance**: Current configuration appears to be working well!")
        
        for rec in recommendations:
            st.markdown(rec)

# Utility functions for Q-learning visualization

def load_qtable_from_file(file_path: str) -> Optional[np.ndarray]:
    """Load Q-table from file (pickle or numpy format).
    
    Args:
        file_path: Path to Q-table file
        
    Returns:
        Q-table array or None if loading fails
    """
    try:
        path = Path(file_path)
        
        if path.suffix == '.pkl':
            with open(path, 'rb') as f:
                q_table = pickle.load(f)
        elif path.suffix in ['.npy', '.npz']:
            q_table = np.load(path)
        else:
            # Try JSON format
            with open(path, 'r') as f:
                data = json.load(f)
                q_table = np.array(data['q_table'])
        
        return q_table
    
    except Exception as e:
        st.error(f"Failed to load Q-table from {file_path}: {e}")
        return None

def save_visualization_config(config: QLearningVisualizationConfig, file_path: str):
    """Save visualization configuration to file.
    
    Args:
        config: Visualization configuration
        file_path: Output file path
    """
    try:
        config_dict = {
            'colormap': config.colormap,
            'show_values': config.show_values,
            'interpolation': config.interpolation,
            'figure_size': config.figure_size,
            'font_size': config.font_size,
            'show_policy_arrows': config.show_policy_arrows,
            'animation_speed': config.animation_speed,
            'confidence_intervals': config.confidence_intervals
        }
        
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        st.success(f"Configuration saved to {file_path}")
    
    except Exception as e:
        st.error(f"Failed to save configuration: {e}")

def create_demo_qlearning_data() -> Tuple[np.ndarray, Dict[str, List[float]], np.ndarray]:
    """Create demo Q-learning data for testing visualizations.
    
    Returns:
        Tuple of (q_table, training_history, policy)
    """
    # Create demo Q-table
    n_states, n_actions = 20, 4
    q_table = np.random.randn(n_states, n_actions) * 0.5
    
    # Add some structure to make it more realistic
    for i in range(n_states):
        best_action = i % n_actions
        q_table[i, best_action] += np.random.uniform(0.5, 1.5)
    
    # Create demo training history
    n_episodes = 1000
    training_data = {
        'reward': np.cumsum(np.random.randn(n_episodes) * 0.1) + np.linspace(0, 10, n_episodes),
        'epsilon': np.maximum(0.01, 1.0 - np.linspace(0, 0.99, n_episodes)),
        'loss': np.maximum(0, 5.0 * np.exp(-np.linspace(0, 5, n_episodes)) + np.random.randn(n_episodes) * 0.1)
    }
    
    # Convert to lists with explicit float conversion
    training_history: Dict[str, List[float]] = {k: [float(x) for x in v.tolist()] for k, v in training_data.items()}
    
    # Create policy with explicit type
    policy = np.argmax(q_table, axis=1).astype(np.int64)
    
    return q_table, training_history, policy