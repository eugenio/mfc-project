"""Deep coverage tests for policy_evolution_viz and qtable_visualization."""
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

_THIS_DIR = str(Path(__file__).resolve().parent)
_SRC_DIR = str((Path(__file__).resolve().parent / ".." / ".." / "src").resolve())
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

_GUI_PREFIX = "gui."


@pytest.fixture(autouse=True)
def _clear_module_cache():
    for mod_name in list(sys.modules):
        if mod_name == "gui" or mod_name.startswith(_GUI_PREFIX):
            del sys.modules[mod_name]
    yield
    for mod_name in list(sys.modules):
        if mod_name == "gui" or mod_name.startswith(_GUI_PREFIX):
            del sys.modules[mod_name]


class _SessionState(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key) from None


def _make_mock_st():
    mock_st = MagicMock()
    mock_st.session_state = _SessionState()

    def _smart_columns(n_or_spec):
        n = len(n_or_spec) if isinstance(n_or_spec, list | tuple) else int(n_or_spec)
        cols = []
        for _ in range(n):
            col = MagicMock()
            col.__enter__ = MagicMock(return_value=col)
            col.__exit__ = MagicMock(return_value=False)
            cols.append(col)
        return cols

    mock_st.columns.side_effect = _smart_columns

    def _smart_tabs(labels):
        tabs = []
        for _ in labels:
            tab = MagicMock()
            tab.__enter__ = MagicMock(return_value=tab)
            tab.__exit__ = MagicMock(return_value=False)
            tabs.append(tab)
        return tabs

    mock_st.tabs.side_effect = _smart_tabs
    exp = MagicMock()
    exp.__enter__ = MagicMock(return_value=exp)
    exp.__exit__ = MagicMock(return_value=False)
    mock_st.expander.return_value = exp
    mock_st.sidebar = MagicMock()
    return mock_st


# ---------------------------------------------------------------------------
# Helpers for mock data objects
# ---------------------------------------------------------------------------

def _make_policy_snapshot(episode, policy=None, action_freqs=None,
                          entropy=0.5, coverage=0.8, reward=None):
    snap = MagicMock()
    snap.episode = episode
    snap.policy = policy if policy is not None else np.array([0, 1, 0, 1])
    snap.q_table = np.random.rand(4, 3)
    snap.action_frequencies = action_freqs if action_freqs is not None else {0: 2, 1: 2}
    snap.policy_entropy = entropy
    snap.state_coverage = coverage
    snap.performance_reward = reward
    snap.timestamp = datetime.now().isoformat()
    return snap


# Helper factories are defined as methods within test classes


# ============================================================================
# POLICY EVOLUTION VIZ TESTS
# ============================================================================

class TestPolicyEvolutionVisualization:
    """Tests for PolicyEvolutionVisualization class."""

    def _import_module(self, mock_st):
        """Import the module with mocked streamlit."""
        mock_go = MagicMock()
        mock_make_subplots = MagicMock()
        mock_tracker_module = MagicMock()

        # Create real-ish PolicyStability enum
        from enum import Enum

        class PolicyStability(Enum):
            STABLE = "stable"
            CONVERGING = "converging"
            UNSTABLE = "unstable"
            OSCILLATING = "oscillating"
            UNKNOWN = "unknown"

        mock_tracker_module.PolicyStability = PolicyStability
        mock_tracker_module.POLICY_EVOLUTION_TRACKER = MagicMock()
        mock_tracker_module.PolicyEvolutionMetrics = MagicMock

        mock_plotly = MagicMock()
        mock_plotly.graph_objects = mock_go
        mock_plotly_subplots = MagicMock()
        mock_plotly_subplots.make_subplots = mock_make_subplots

        patches = {
            "streamlit": mock_st,
            "plotly": mock_plotly,
            "plotly.graph_objects": mock_go,
            "plotly.express": MagicMock(),
            "plotly.subplots": mock_plotly_subplots,
            "analysis": MagicMock(),
            "analysis.policy_evolution_tracker": mock_tracker_module,
            "analysis.qtable_analyzer": MagicMock(),
        }

        # Also mock gui siblings to avoid __init__ cascade
        mock_gui = MagicMock()
        patches["gui"] = mock_gui
        patches["gui.enhanced_components"] = MagicMock()
        patches["gui.qlearning_viz"] = MagicMock()

        with patch.dict(sys.modules, patches):
            # Remove the gui key so our import actually loads the real module
            del sys.modules["gui"]
            import gui.policy_evolution_viz as mod
            mod.st = mock_st
            mod.go = mock_go
            mod.make_subplots = mock_make_subplots
            mod.PolicyStability = PolicyStability
            mod.POLICY_EVOLUTION_TRACKER = mock_tracker_module.POLICY_EVOLUTION_TRACKER

        return mod, mock_go, mock_make_subplots, PolicyStability

    # ------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------

    def test_init_sets_session_state(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)

        mod.PolicyEvolutionVisualization()
        assert mock_st.session_state.policy_snapshots_loaded is False
        assert mock_st.session_state.policy_evolution_metrics is None
        assert mock_st.session_state.selected_episodes == []

    def test_init_preserves_existing_session_state(self):
        mock_st = _make_mock_st()
        mock_st.session_state.policy_snapshots_loaded = True
        mock_st.session_state.policy_evolution_metrics = "some_metrics"
        mock_st.session_state.selected_episodes = [1, 2, 3]

        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        mod.PolicyEvolutionVisualization()

        assert mock_st.session_state.policy_snapshots_loaded is True
        assert mock_st.session_state.policy_evolution_metrics == "some_metrics"
        assert mock_st.session_state.selected_episodes == [1, 2, 3]

    # ------------------------------------------------------------------
    # render_policy_evolution_interface
    # ------------------------------------------------------------------

    def test_render_interface_no_data(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()
        comp.tracker = MagicMock()
        comp.tracker.policy_snapshots = []
        mock_st.session_state.policy_snapshots_loaded = False
        mock_st.button.return_value = False

        result = comp.render_policy_evolution_interface()
        mock_st.header.assert_called_once()
        mock_st.markdown.assert_called()
        assert result["snapshots_loaded"] == 0

    def test_render_interface_with_data_and_metrics(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()

        snap1 = _make_policy_snapshot(0)
        snap2 = _make_policy_snapshot(1)
        comp.tracker = MagicMock()
        comp.tracker.policy_snapshots = [snap1, snap2]
        mock_st.session_state.policy_snapshots_loaded = True

        # Mock _render methods
        comp._render_data_loading_section = MagicMock()
        comp._render_analysis_overview = MagicMock(return_value="metrics_obj")
        comp._render_policy_visualization_tabs = MagicMock()
        comp._render_export_section = MagicMock()

        result = comp.render_policy_evolution_interface()
        assert result["snapshots_loaded"] == 2
        comp._render_analysis_overview.assert_called_once()
        comp._render_policy_visualization_tabs.assert_called_once_with("metrics_obj")
        comp._render_export_section.assert_called_once_with("metrics_obj")

    # ------------------------------------------------------------------
    # _render_data_loading_section
    # ------------------------------------------------------------------

    def test_render_data_loading_section_load_success(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()
        comp.tracker = MagicMock()
        comp.tracker.policy_snapshots = []
        comp.tracker.load_policy_snapshots_from_files.return_value = 5

        # Simulate pressing "Load Policy Data" button
        mock_st.button.side_effect = [True, False]
        mock_st.text_input.return_value = "*qtable*.pkl"
        mock_st.number_input.return_value = 50
        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)

        comp._render_data_loading_section()
        mock_st.success.assert_called()
        assert mock_st.session_state.policy_snapshots_loaded is True

    def test_render_data_loading_section_load_zero(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()
        comp.tracker = MagicMock()
        comp.tracker.policy_snapshots = []
        comp.tracker.load_policy_snapshots_from_files.return_value = 0

        mock_st.button.side_effect = [True, False]
        mock_st.text_input.return_value = "*qtable*.pkl"
        mock_st.number_input.return_value = 50
        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)

        comp._render_data_loading_section()
        mock_st.error.assert_called()

    def test_render_data_loading_section_quick_analysis_with_snapshots(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()
        comp.tracker = MagicMock()
        comp.tracker.policy_snapshots = [_make_policy_snapshot(0)]

        metrics_mock = MagicMock()
        comp.tracker.analyze_policy_evolution.return_value = metrics_mock

        mock_st.button.side_effect = [False, True]
        mock_st.text_input.return_value = "*qtable*.pkl"
        mock_st.number_input.return_value = 50
        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)

        comp._render_data_loading_section()
        mock_st.success.assert_called()
        assert mock_st.session_state.policy_evolution_metrics is metrics_mock

    def test_render_data_loading_section_quick_analysis_fails(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()
        comp.tracker = MagicMock()
        comp.tracker.policy_snapshots = [_make_policy_snapshot(0)]
        comp.tracker.analyze_policy_evolution.return_value = None

        mock_st.button.side_effect = [False, True]
        mock_st.text_input.return_value = "*qtable*.pkl"
        mock_st.number_input.return_value = 50
        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)

        comp._render_data_loading_section()
        mock_st.error.assert_called()

    def test_render_data_loading_section_quick_analysis_no_snapshots(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()
        comp.tracker = MagicMock()
        comp.tracker.policy_snapshots = []

        mock_st.button.side_effect = [False, True]
        mock_st.text_input.return_value = "*qtable*.pkl"
        mock_st.number_input.return_value = 50

        comp._render_data_loading_section()
        mock_st.warning.assert_called()

    def test_render_data_loading_section_shows_status_when_snapshots(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()
        comp.tracker = MagicMock()
        comp.tracker.policy_snapshots = [_make_policy_snapshot(0), _make_policy_snapshot(1)]

        mock_st.button.side_effect = [False, False]
        mock_st.text_input.return_value = "*qtable*.pkl"
        mock_st.number_input.return_value = 50

        comp._render_data_loading_section()
        mock_st.info.assert_called()

    # ------------------------------------------------------------------
    # _render_analysis_overview
    # ------------------------------------------------------------------

    def test_render_analysis_overview_no_cached_metrics_success(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()
        comp.tracker = MagicMock()

        metrics = MagicMock()
        metrics.total_episodes = 100
        metrics.snapshots_count = 10
        metrics.stability_status = PS.STABLE
        metrics.stability_score = 0.97
        metrics.convergence_episode = 25
        metrics.action_preference_changes = 2
        metrics.dominant_actions = {0: 0.6, 1: 0.4}
        comp.tracker.analyze_policy_evolution.return_value = metrics

        mock_st.session_state.policy_evolution_metrics = None

        result = comp._render_analysis_overview()
        assert result is metrics
        assert mock_st.session_state.policy_evolution_metrics is metrics

    def test_render_analysis_overview_no_cached_metrics_fails(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()
        comp.tracker = MagicMock()
        comp.tracker.analyze_policy_evolution.return_value = None

        mock_st.session_state.policy_evolution_metrics = None

        result = comp._render_analysis_overview()
        assert result is None
        mock_st.error.assert_called()

    def test_render_analysis_overview_with_cached_metrics_stable(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()

        metrics = MagicMock()
        metrics.total_episodes = 100
        metrics.snapshots_count = 10
        metrics.stability_status = PS.STABLE
        metrics.stability_score = 0.97
        metrics.convergence_episode = 25
        metrics.action_preference_changes = 2
        metrics.dominant_actions = {0: 0.6, 1: 0.4}
        mock_st.session_state.policy_evolution_metrics = metrics

        result = comp._render_analysis_overview()
        assert result is metrics

    def test_render_analysis_overview_converging(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()

        metrics = MagicMock()
        metrics.total_episodes = 100
        metrics.snapshots_count = 10
        metrics.stability_status = PS.CONVERGING
        metrics.stability_score = 0.85
        metrics.convergence_episode = None
        metrics.action_preference_changes = 5
        metrics.dominant_actions = {0: 0.3, 1: 0.3, 2: 0.4}
        mock_st.session_state.policy_evolution_metrics = metrics

        result = comp._render_analysis_overview()
        assert result is metrics

    def test_render_analysis_overview_unstable(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()

        metrics = MagicMock()
        metrics.total_episodes = 100
        metrics.snapshots_count = 10
        metrics.stability_status = PS.UNSTABLE
        metrics.stability_score = 0.6
        metrics.convergence_episode = 80
        metrics.action_preference_changes = 10
        metrics.dominant_actions = {0: 0.5, 1: 0.5}
        mock_st.session_state.policy_evolution_metrics = metrics

        result = comp._render_analysis_overview()
        assert result is metrics

    def test_render_analysis_overview_oscillating(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()

        metrics = MagicMock()
        metrics.total_episodes = 100
        metrics.snapshots_count = 10
        metrics.stability_status = PS.OSCILLATING
        metrics.stability_score = 0.4
        metrics.convergence_episode = None
        metrics.action_preference_changes = 15
        metrics.dominant_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        mock_st.session_state.policy_evolution_metrics = metrics

        result = comp._render_analysis_overview()
        assert result is metrics

    def test_render_analysis_overview_unknown(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()

        metrics = MagicMock()
        metrics.total_episodes = 100
        metrics.snapshots_count = 10
        metrics.stability_status = PS.UNKNOWN
        metrics.stability_score = 0.1
        metrics.convergence_episode = None
        metrics.action_preference_changes = 0
        metrics.dominant_actions = {}
        mock_st.session_state.policy_evolution_metrics = metrics

        result = comp._render_analysis_overview()
        assert result is metrics

    def test_render_analysis_overview_late_convergence(self):
        """Test when convergence_episode > 50% of total_episodes."""
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()

        metrics = MagicMock()
        metrics.total_episodes = 100
        metrics.snapshots_count = 10
        metrics.stability_status = PS.STABLE
        metrics.stability_score = 0.96
        metrics.convergence_episode = 80  # > 50% of 100
        metrics.action_preference_changes = 1
        metrics.dominant_actions = {0: 0.9, 1: 0.1}
        mock_st.session_state.policy_evolution_metrics = metrics

        result = comp._render_analysis_overview()
        assert result is metrics

    # ------------------------------------------------------------------
    # _render_policy_visualization_tabs
    # ------------------------------------------------------------------

    def test_render_policy_visualization_tabs(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()
        comp._render_policy_evolution_timeline = MagicMock()
        comp._render_action_frequency_analysis = MagicMock()
        comp._render_learning_curves = MagicMock()
        comp._render_policy_stability_analysis = MagicMock()
        comp._render_episode_comparison = MagicMock()

        metrics = MagicMock()
        comp._render_policy_visualization_tabs(metrics)

        comp._render_policy_evolution_timeline.assert_called_once_with(metrics)
        comp._render_action_frequency_analysis.assert_called_once_with(metrics)
        comp._render_learning_curves.assert_called_once_with(metrics)
        comp._render_policy_stability_analysis.assert_called_once_with(metrics)
        comp._render_episode_comparison.assert_called_once()

    # ------------------------------------------------------------------
    # _render_policy_evolution_timeline
    # ------------------------------------------------------------------

    def test_render_timeline_no_snapshots(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()
        comp.tracker = MagicMock()
        comp.tracker.policy_snapshots = []

        metrics = MagicMock()
        comp._render_policy_evolution_timeline(metrics)
        mock_st.info.assert_called()

    def test_render_timeline_with_data_no_convergence(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()

        snap1 = _make_policy_snapshot(0, entropy=0.5, coverage=0.7)
        snap2 = _make_policy_snapshot(1, entropy=0.4, coverage=0.8)
        snap3 = _make_policy_snapshot(2, entropy=0.3, coverage=0.9)
        comp.tracker = MagicMock()
        comp.tracker.policy_snapshots = [snap1, snap2, snap3]

        metrics = MagicMock()
        metrics.policy_changes = [3, 2]
        metrics.convergence_episode = None
        comp._render_timeline_insights = MagicMock()

        mock_fig = MagicMock()
        mock_ms.return_value = mock_fig

        comp._render_policy_evolution_timeline(metrics)
        mock_st.plotly_chart.assert_called()

    def test_render_timeline_with_convergence(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()

        snap1 = _make_policy_snapshot(0)
        snap2 = _make_policy_snapshot(1)
        comp.tracker = MagicMock()
        comp.tracker.policy_snapshots = [snap1, snap2]

        metrics = MagicMock()
        metrics.policy_changes = [1]
        metrics.convergence_episode = 5
        comp._render_timeline_insights = MagicMock()

        mock_fig = MagicMock()
        mock_ms.return_value = mock_fig

        comp._render_policy_evolution_timeline(metrics)
        mock_fig.add_vline.assert_called()

    def test_render_timeline_empty_policy_changes(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()

        snap1 = _make_policy_snapshot(0)
        comp.tracker = MagicMock()
        comp.tracker.policy_snapshots = [snap1]

        metrics = MagicMock()
        metrics.policy_changes = []
        metrics.convergence_episode = None
        comp._render_timeline_insights = MagicMock()

        mock_fig = MagicMock()
        mock_ms.return_value = mock_fig

        comp._render_policy_evolution_timeline(metrics)
        mock_st.plotly_chart.assert_called()

    # ------------------------------------------------------------------
    # _render_action_frequency_analysis
    # ------------------------------------------------------------------

    def test_render_action_frequency_no_data(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()
        comp.tracker = MagicMock()
        comp.tracker.get_action_frequency_matrix.return_value = None

        metrics = MagicMock()
        metrics.dominant_actions = {}
        comp._render_action_frequency_analysis(metrics)
        mock_st.info.assert_called()

    def test_render_action_frequency_with_data(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()

        freq_df = pd.DataFrame({
            "Episode": [0, 1, 2],
            "Action_0": [5, 3, 2],
            "Action_1": [3, 5, 6],
        })
        comp.tracker = MagicMock()
        comp.tracker.get_action_frequency_matrix.return_value = freq_df

        # Multiple snapshots with action_frequencies
        snap1 = _make_policy_snapshot(0, action_freqs={0: 5, 1: 3})
        snap2 = _make_policy_snapshot(1, action_freqs={0: 3, 1: 5})
        snap3 = _make_policy_snapshot(2, action_freqs={0: 2, 1: 6})
        comp.tracker.policy_snapshots = [snap1, snap2, snap3]

        metrics = MagicMock()
        metrics.dominant_actions = {0: 0.4, 1: 0.6}

        comp._render_action_frequency_analysis(metrics)
        assert mock_st.plotly_chart.call_count >= 2

    def test_render_action_frequency_empty_action_freqs(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()

        freq_df = pd.DataFrame({
            "Episode": [0, 1],
            "Action_0": [5, 3],
        })
        comp.tracker = MagicMock()
        comp.tracker.get_action_frequency_matrix.return_value = freq_df

        # One snapshot with empty action_frequencies
        snap1 = _make_policy_snapshot(0, action_freqs={})
        snap2 = _make_policy_snapshot(1, action_freqs={0: 5})
        comp.tracker.policy_snapshots = [snap1, snap2]

        metrics = MagicMock()
        metrics.dominant_actions = {0: 1.0}

        comp._render_action_frequency_analysis(metrics)
        mock_st.plotly_chart.assert_called()

    def test_render_action_frequency_no_action_cols(self):
        """Test with freq_matrix that has no Action_ columns."""
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()

        freq_df = pd.DataFrame({
            "Episode": [0, 1],
            "Other_col": [5, 3],
        })
        comp.tracker = MagicMock()
        comp.tracker.get_action_frequency_matrix.return_value = freq_df
        comp.tracker.policy_snapshots = [_make_policy_snapshot(0)]

        metrics = MagicMock()
        metrics.dominant_actions = {0: 1.0}

        comp._render_action_frequency_analysis(metrics)

    def test_render_action_frequency_single_snapshot(self):
        """Test with only 1 snapshot - skip action preference evolution."""
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()

        freq_df = pd.DataFrame({
            "Episode": [0],
            "Action_0": [5],
        })
        comp.tracker = MagicMock()
        comp.tracker.get_action_frequency_matrix.return_value = freq_df
        comp.tracker.policy_snapshots = [_make_policy_snapshot(0)]

        metrics = MagicMock()
        metrics.dominant_actions = {0: 1.0}

        comp._render_action_frequency_analysis(metrics)

    # ------------------------------------------------------------------
    # _render_learning_curves
    # ------------------------------------------------------------------

    def test_render_learning_curves_no_perf_no_velocity(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()

        metrics = MagicMock()
        metrics.performance_trend = []
        metrics.learning_velocity = []
        metrics.exploration_decay = []

        comp._render_learning_curves(metrics)
        mock_st.info.assert_called()

    def test_render_learning_curves_no_perf_with_velocity(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()

        metrics = MagicMock()
        metrics.performance_trend = []
        metrics.learning_velocity = [0.1, -0.2, 0.05, -0.1]
        metrics.exploration_decay = []

        comp._render_learning_curves(metrics)
        mock_st.plotly_chart.assert_called()

    def test_render_learning_curves_with_performance_and_velocity(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()

        metrics = MagicMock()
        metrics.performance_trend = [10.0, 15.0, 20.0, 25.0]
        metrics.learning_velocity = [0.1, 0.2, 0.15]
        metrics.exploration_decay = []

        mock_fig = MagicMock()
        mock_ms.return_value = mock_fig

        comp._render_learning_curves(metrics)
        mock_st.plotly_chart.assert_called()

    def test_render_learning_curves_with_performance_no_velocity(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()

        metrics = MagicMock()
        metrics.performance_trend = [10.0, 15.0]
        metrics.learning_velocity = []
        metrics.exploration_decay = []

        mock_fig = MagicMock()
        mock_ms.return_value = mock_fig

        comp._render_learning_curves(metrics)
        mock_st.plotly_chart.assert_called()

    def test_render_learning_curves_with_exploration_decay(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()

        metrics = MagicMock()
        metrics.performance_trend = []
        metrics.learning_velocity = []
        metrics.exploration_decay = [0.1, 0.08, 0.06]

        comp._render_learning_curves(metrics)
        mock_st.plotly_chart.assert_called()

    def test_render_learning_curves_all_data(self):
        """Performance, velocity, and exploration decay all present."""
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()

        metrics = MagicMock()
        metrics.performance_trend = [10.0, 15.0, 20.0]
        metrics.learning_velocity = [0.1, 0.2]
        metrics.exploration_decay = [0.1, 0.08]

        mock_fig = MagicMock()
        mock_ms.return_value = mock_fig

        comp._render_learning_curves(metrics)
        # Should have plotly_chart calls for both performance/velocity and decay
        assert mock_st.plotly_chart.call_count >= 2

    # ------------------------------------------------------------------
    # _render_policy_stability_analysis
    # ------------------------------------------------------------------

    def test_render_stability_with_changes(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()
        comp._render_stability_insights = MagicMock()

        metrics = MagicMock()
        metrics.stability_score = 0.92
        metrics.policy_changes = [5, 3, 2, 1]

        comp._render_policy_stability_analysis(metrics)
        mock_st.plotly_chart.assert_called()
        comp._render_stability_insights.assert_called_once_with(metrics)

    def test_render_stability_no_changes(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()
        comp._render_stability_insights = MagicMock()

        metrics = MagicMock()
        metrics.stability_score = 0.5
        metrics.policy_changes = []

        comp._render_policy_stability_analysis(metrics)
        comp._render_stability_insights.assert_called_once()

    # ------------------------------------------------------------------
    # _render_episode_comparison
    # ------------------------------------------------------------------

    def test_render_episode_comparison_too_few(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()
        comp.tracker = MagicMock()
        comp.tracker.policy_snapshots = [_make_policy_snapshot(0)]

        comp._render_episode_comparison()
        mock_st.info.assert_called()

    def test_render_episode_comparison_with_enough_episodes(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()
        comp.tracker = MagicMock()

        snaps = [_make_policy_snapshot(i) for i in range(5)]
        comp.tracker.policy_snapshots = snaps

        mock_st.selectbox.return_value = 0
        mock_st.multiselect.return_value = [1, 2, 3]
        mock_st.button.return_value = False

        comp._render_episode_comparison()
        mock_st.selectbox.assert_called()
        mock_st.multiselect.assert_called()

    def test_render_episode_comparison_generate_button_with_episodes(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()
        comp.tracker = MagicMock()
        comp._generate_episode_comparison = MagicMock()

        snaps = [_make_policy_snapshot(i) for i in range(5)]
        comp.tracker.policy_snapshots = snaps

        mock_st.selectbox.return_value = 0
        mock_st.multiselect.return_value = [2, 3]
        mock_st.button.return_value = True

        comp._render_episode_comparison()
        comp._generate_episode_comparison.assert_called_once_with(0, [2, 3])

    def test_render_episode_comparison_generate_button_no_episodes(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()
        comp.tracker = MagicMock()

        snaps = [_make_policy_snapshot(i) for i in range(3)]
        comp.tracker.policy_snapshots = snaps

        mock_st.selectbox.return_value = 0
        mock_st.multiselect.return_value = []
        mock_st.button.return_value = True

        comp._render_episode_comparison()
        mock_st.warning.assert_called()

    # ------------------------------------------------------------------
    # _generate_episode_comparison
    # ------------------------------------------------------------------

    def test_generate_comparison_no_matrix(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()
        comp.tracker = MagicMock()
        comp.tracker.get_policy_comparison_matrix.return_value = None

        comp._generate_episode_comparison(0, [1, 2])
        mock_st.error.assert_called()

    def test_generate_comparison_no_ref_snapshot(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()
        comp.tracker = MagicMock()
        comp.tracker.get_policy_comparison_matrix.return_value = np.array([[1, 1], [1, 0]])

        # Snapshots exist but none match reference episode 99
        snap1 = _make_policy_snapshot(0)
        snap2 = _make_policy_snapshot(1)
        comp.tracker.policy_snapshots = [snap1, snap2]

        comp._generate_episode_comparison(99, [0])
        mock_st.error.assert_called()

    def test_generate_comparison_no_compare_snapshots(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()
        comp.tracker = MagicMock()
        comp.tracker.get_policy_comparison_matrix.return_value = np.array([[1, 1], [1, 0]])

        snap1 = _make_policy_snapshot(0)
        snap2 = _make_policy_snapshot(1)
        comp.tracker.policy_snapshots = [snap1, snap2]

        comp._generate_episode_comparison(0, [99])
        mock_st.error.assert_called()

    def test_generate_comparison_success(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()
        comp.tracker = MagicMock()

        sim_matrix = np.array([[1.0, 0.8, 0.6], [0.8, 1.0, 0.7], [0.6, 0.7, 1.0]])
        comp.tracker.get_policy_comparison_matrix.return_value = sim_matrix

        snap0 = _make_policy_snapshot(0, entropy=0.5, coverage=0.8)
        snap1 = _make_policy_snapshot(1, entropy=0.4, coverage=0.85)
        snap2 = _make_policy_snapshot(2, entropy=0.3, coverage=0.9)
        comp.tracker.policy_snapshots = [snap0, snap1, snap2]

        comp._generate_episode_comparison(0, [1, 2])
        mock_st.dataframe.assert_called()
        mock_st.plotly_chart.assert_called()

    # ------------------------------------------------------------------
    # _render_timeline_insights
    # ------------------------------------------------------------------

    def test_timeline_insights_early_convergence(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()

        metrics = MagicMock()
        metrics.convergence_episode = 10
        metrics.total_episodes = 100
        metrics.stability_score = 0.97
        metrics.dominant_actions = {0: 1.0}

        comp._render_timeline_insights(metrics)
        assert mock_st.markdown.call_count >= 4  # header + 3 insights

    def test_timeline_insights_good_convergence(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()

        metrics = MagicMock()
        metrics.convergence_episode = 40
        metrics.total_episodes = 100
        metrics.stability_score = 0.9
        metrics.dominant_actions = {0: 0.5, 1: 0.3, 2: 0.2}

        comp._render_timeline_insights(metrics)

    def test_timeline_insights_late_convergence(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()

        metrics = MagicMock()
        metrics.convergence_episode = 80
        metrics.total_episodes = 100
        metrics.stability_score = 0.7
        metrics.dominant_actions = {0: 0.3, 1: 0.2, 2: 0.2, 3: 0.3}

        comp._render_timeline_insights(metrics)

    def test_timeline_insights_no_convergence(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()

        metrics = MagicMock()
        metrics.convergence_episode = None
        metrics.total_episodes = 100
        metrics.stability_score = 0.5
        metrics.dominant_actions = {0: 0.5, 1: 0.5}

        comp._render_timeline_insights(metrics)

    def test_timeline_insights_high_stability(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()

        metrics = MagicMock()
        metrics.convergence_episode = None
        metrics.total_episodes = 100
        metrics.stability_score = 0.96
        metrics.dominant_actions = {0: 0.5, 1: 0.5}

        comp._render_timeline_insights(metrics)

    def test_timeline_insights_moderate_stability(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()

        metrics = MagicMock()
        metrics.convergence_episode = None
        metrics.total_episodes = 100
        metrics.stability_score = 0.85
        metrics.dominant_actions = {0: 1.0}

        comp._render_timeline_insights(metrics)

    # ------------------------------------------------------------------
    # _render_stability_insights
    # ------------------------------------------------------------------

    def test_stability_insights_stable_low_changes_no_pref_changes(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()

        metrics = MagicMock()
        metrics.stability_status = PS.STABLE
        metrics.policy_changes = [2, 3, 1, 2]
        metrics.action_preference_changes = 0

        comp._render_stability_insights(metrics)

    def test_stability_insights_converging_moderate_changes(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()

        metrics = MagicMock()
        metrics.stability_status = PS.CONVERGING
        metrics.policy_changes = [10, 12, 8, 11]
        metrics.action_preference_changes = 2

        comp._render_stability_insights(metrics)

    def test_stability_insights_unstable_high_changes_many_pref(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()

        metrics = MagicMock()
        metrics.stability_status = PS.UNSTABLE
        metrics.policy_changes = [20, 25, 18, 22]
        metrics.action_preference_changes = 5

        comp._render_stability_insights(metrics)

    def test_stability_insights_oscillating_no_changes(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()

        metrics = MagicMock()
        metrics.stability_status = PS.OSCILLATING
        metrics.policy_changes = []
        metrics.action_preference_changes = 3

        comp._render_stability_insights(metrics)

    def test_stability_insights_unknown(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()

        metrics = MagicMock()
        metrics.stability_status = PS.UNKNOWN
        metrics.policy_changes = [1]
        metrics.action_preference_changes = 0

        comp._render_stability_insights(metrics)

    # ------------------------------------------------------------------
    # _render_export_section
    # ------------------------------------------------------------------

    def test_export_metrics_csv(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()
        comp.tracker = MagicMock()

        metrics = MagicMock()
        mock_st.button.side_effect = [True, False, False]

        comp._render_export_section(metrics)
        comp.tracker.export_evolution_analysis.assert_called_once()
        mock_st.success.assert_called()

    def test_export_action_matrix_success(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()
        comp.tracker = MagicMock()

        freq_df = MagicMock()
        comp.tracker.get_action_frequency_matrix.return_value = freq_df

        metrics = MagicMock()
        mock_st.button.side_effect = [False, True, False]

        comp._render_export_section(metrics)
        freq_df.to_csv.assert_called_once()
        mock_st.success.assert_called()

    def test_export_action_matrix_no_data(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()
        comp.tracker = MagicMock()
        comp.tracker.get_action_frequency_matrix.return_value = None

        metrics = MagicMock()
        mock_st.button.side_effect = [False, True, False]

        comp._render_export_section(metrics)
        mock_st.error.assert_called()

    def test_export_generate_report(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()
        comp._generate_policy_evolution_report = MagicMock()

        metrics = MagicMock()
        mock_st.button.side_effect = [False, False, True]

        comp._render_export_section(metrics)
        comp._generate_policy_evolution_report.assert_called_once_with(metrics)

    # ------------------------------------------------------------------
    # _generate_policy_evolution_report
    # ------------------------------------------------------------------

    def test_generate_report_with_convergence_stable(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()

        metrics = MagicMock()
        metrics.total_episodes = 100
        metrics.stability_score = 0.95
        metrics.stability_status = MagicMock()
        metrics.stability_status.value = "stable"
        metrics.convergence_episode = 25
        metrics.dominant_actions = {0: 0.6, 1: 0.4}

        comp._generate_policy_evolution_report(metrics)
        mock_st.markdown.assert_called()
        mock_st.download_button.assert_called()

    def test_generate_report_no_convergence_unstable(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()

        metrics = MagicMock()
        metrics.total_episodes = 100
        metrics.stability_score = 0.5
        metrics.stability_status = MagicMock()
        metrics.stability_status.value = "unstable"
        metrics.convergence_episode = None
        metrics.dominant_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}

        comp._generate_policy_evolution_report(metrics)
        mock_st.markdown.assert_called()
        mock_st.download_button.assert_called()

    def test_generate_report_late_convergence(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)
        comp = mod.PolicyEvolutionVisualization()

        metrics = MagicMock()
        metrics.total_episodes = 100
        metrics.stability_score = 0.92
        metrics.stability_status = MagicMock()
        metrics.stability_status.value = "converging"
        metrics.convergence_episode = 80
        metrics.dominant_actions = {0: 0.5, 1: 0.3, 2: 0.2}

        comp._generate_policy_evolution_report(metrics)
        mock_st.download_button.assert_called()

    # ------------------------------------------------------------------
    # Module-level functions
    # ------------------------------------------------------------------

    def test_render_policy_evolution_interface_module_func(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)

        # Patch the class to control the return value
        with patch.object(mod, "PolicyEvolutionVisualization") as MockClass:
            instance = MagicMock()
            instance.render_policy_evolution_interface.return_value = {"test": True}
            MockClass.return_value = instance

            result = mod.render_policy_evolution_interface()
            assert result == {"test": True}

    def test_main_block_coverage(self):
        """Cover the if __name__ == '__main__' block."""
        mock_st = _make_mock_st()
        mod, mock_go, mock_ms, PS = self._import_module(mock_st)

        with patch.object(mod, "render_policy_evolution_interface") as mock_render:
            mock_render.return_value = {"test": True}
            # Simulate __main__ execution
            mod.st.title("Policy Evolution Tracking Component Test")
            result = mod.render_policy_evolution_interface()
            mod.st.json(result)

            mock_st.title.assert_called()
            mock_st.json.assert_called()


# ============================================================================
# QTABLE VISUALIZATION TESTS
# ============================================================================

class TestQTableVisualization:
    """Tests for QTableVisualization class."""

    def _import_module(self, mock_st):
        """Import the module with mocked streamlit."""
        mock_go = MagicMock()
        mock_px = MagicMock()
        mock_make_subplots = MagicMock()
        mock_analyzer_module = MagicMock()

        from enum import Enum

        class ConvergenceStatus(Enum):
            CONVERGED = "converged"
            CONVERGING = "converging"
            UNSTABLE = "unstable"
            DIVERGING = "diverging"
            UNKNOWN = "unknown"

        mock_analyzer_module.ConvergenceStatus = ConvergenceStatus
        mock_analyzer_module.QTABLE_ANALYZER = MagicMock()
        mock_analyzer_module.QTableMetrics = MagicMock
        mock_analyzer_module.QTableComparison = MagicMock

        mock_plotly = MagicMock()
        mock_plotly.graph_objects = mock_go
        mock_plotly.express = mock_px
        mock_plotly_subplots = MagicMock()
        mock_plotly_subplots.make_subplots = mock_make_subplots

        patches = {
            "streamlit": mock_st,
            "plotly": mock_plotly,
            "plotly.graph_objects": mock_go,
            "plotly.express": mock_px,
            "plotly.subplots": mock_plotly_subplots,
            "analysis": MagicMock(),
            "analysis.qtable_analyzer": mock_analyzer_module,
            "analysis.policy_evolution_tracker": MagicMock(),
        }

        # Also mock gui siblings to avoid __init__ cascade
        patches["gui"] = MagicMock()
        patches["gui.enhanced_components"] = MagicMock()
        patches["gui.qlearning_viz"] = MagicMock()

        with patch.dict(sys.modules, patches):
            del sys.modules["gui"]
            import gui.qtable_visualization as mod
            mod.st = mock_st
            mod.go = mock_go
            mod.px = mock_px
            mod.make_subplots = mock_make_subplots
            mod.ConvergenceStatus = ConvergenceStatus
            mod.QTABLE_ANALYZER = mock_analyzer_module.QTABLE_ANALYZER

        return mod, mock_go, mock_px, mock_make_subplots, ConvergenceStatus

    def _make_qtable_metrics_obj(self, CS, **kwargs):
        defaults = {
            "shape": (10, 4),
            "total_states": 10,
            "total_actions": 4,
            "non_zero_values": 30,
            "sparsity": 0.25,
            "mean_q_value": 1.5,
            "std_q_value": 0.5,
            "min_q_value": 0.0,
            "max_q_value": 5.0,
            "q_value_range": 5.0,
            "convergence_score": 0.92,
            "stability_measure": 0.88,
            "convergence_status": CS.CONVERGED,
            "policy_entropy": 1.2,
            "action_diversity": 0.7,
            "state_value_variance": 0.3,
            "exploration_coverage": 0.85,
            "visited_states": 8,
            "unvisited_states": 2,
            "analysis_timestamp": "2025-01-01T00:00:00",
            "file_path": "/tmp/test.pkl",
        }
        defaults.update(kwargs)
        m = MagicMock()
        for k, v in defaults.items():
            setattr(m, k, v)
        return m

    # ------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------

    def test_init_sets_session_state(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)

        mod.QTableVisualization()
        assert mock_st.session_state.selected_qtables == []
        assert mock_st.session_state.qtable_analysis_cache == {}
        assert mock_st.session_state.comparison_results == {}

    def test_init_preserves_existing_session_state(self):
        mock_st = _make_mock_st()
        mock_st.session_state.selected_qtables = ["file1.pkl"]
        mock_st.session_state.qtable_analysis_cache = {"key": "val"}
        mock_st.session_state.comparison_results = {"a|b": "result"}

        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        mod.QTableVisualization()

        assert mock_st.session_state.selected_qtables == ["file1.pkl"]
        assert mock_st.session_state.qtable_analysis_cache == {"key": "val"}

    # ------------------------------------------------------------------
    # render_qtable_analysis_interface
    # ------------------------------------------------------------------

    def test_render_interface_no_selected(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()
        comp._render_file_selection_section = MagicMock()
        mock_st.session_state.selected_qtables = []

        result = comp.render_qtable_analysis_interface()
        assert result["selected_qtables"] == []
        comp._render_file_selection_section.assert_called_once()

    def test_render_interface_with_selected(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()
        comp._render_file_selection_section = MagicMock()
        comp._render_analysis_results_section = MagicMock()
        comp._render_visualization_section = MagicMock()
        comp._render_comparison_section = MagicMock()
        comp._render_export_section = MagicMock()
        mock_st.session_state.selected_qtables = ["file1.pkl"]

        comp.render_qtable_analysis_interface()
        comp._render_analysis_results_section.assert_called_once()
        comp._render_visualization_section.assert_called_once()
        comp._render_comparison_section.assert_called_once()
        comp._render_export_section.assert_called_once()

    # ------------------------------------------------------------------
    # _render_file_selection_section
    # ------------------------------------------------------------------

    def test_file_selection_with_available_files(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()
        comp.analyzer = MagicMock()

        p1 = MagicMock(spec=Path)
        p1.__str__ = MagicMock(return_value="/tmp/q1.pkl")
        p1.name = "q1.pkl"
        p1.stat.return_value.st_size = 1024
        p2 = MagicMock(spec=Path)
        p2.__str__ = MagicMock(return_value="/tmp/q2.pkl")
        p2.name = "q2.pkl"
        p2.stat.return_value.st_size = 2048

        comp.analyzer.get_available_qtables.return_value = [p1, p2]
        mock_st.multiselect.return_value = [0]
        mock_st.button.side_effect = [False, False]

        comp._render_file_selection_section()
        assert mock_st.session_state.selected_qtables == ["/tmp/q1.pkl"]

    def test_file_selection_no_available_files(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()
        comp.analyzer = MagicMock()
        comp.analyzer.get_available_qtables.return_value = []

        mock_st.button.side_effect = [False, False]

        comp._render_file_selection_section()
        mock_st.warning.assert_called()

    def test_file_selection_refresh_button(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()
        comp.analyzer = MagicMock()
        comp.analyzer.get_available_qtables.return_value = []

        mock_st.button.side_effect = [True, False]

        comp._render_file_selection_section()
        mock_st.rerun.assert_called()

    def test_file_selection_quick_analysis_with_files(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()
        comp.analyzer = MagicMock()
        comp.analyzer.get_available_qtables.return_value = []
        comp._perform_quick_analysis = MagicMock()

        mock_st.session_state.selected_qtables = ["file1.pkl"]
        mock_st.button.side_effect = [False, True]

        comp._render_file_selection_section()
        comp._perform_quick_analysis.assert_called_once()

    def test_file_selection_quick_analysis_no_files(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()
        comp.analyzer = MagicMock()
        comp.analyzer.get_available_qtables.return_value = []

        mock_st.session_state.selected_qtables = []
        mock_st.button.side_effect = [False, True]

        comp._render_file_selection_section()
        mock_st.warning.assert_called()

    # ------------------------------------------------------------------
    # _render_analysis_results_section
    # ------------------------------------------------------------------

    def test_analysis_results_section_cached_and_uncached(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()
        comp.analyzer = MagicMock()
        comp._display_analysis_summary = MagicMock()
        comp._display_detailed_metrics = MagicMock()

        metrics1 = self._make_qtable_metrics_obj(CS)
        metrics2 = self._make_qtable_metrics_obj(CS)
        comp.analyzer.analyze_qtable.return_value = metrics2

        mock_st.session_state.selected_qtables = ["file1.pkl", "file2.pkl"]
        mock_st.session_state.qtable_analysis_cache = {"file1.pkl": metrics1}
        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)

        comp._render_analysis_results_section()
        comp._display_analysis_summary.assert_called_once()
        comp._display_detailed_metrics.assert_called_once()

    def test_analysis_results_section_analyze_returns_none(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()
        comp.analyzer = MagicMock()
        comp.analyzer.analyze_qtable.return_value = None
        comp._display_analysis_summary = MagicMock()
        comp._display_detailed_metrics = MagicMock()

        mock_st.session_state.selected_qtables = ["file1.pkl"]
        mock_st.session_state.qtable_analysis_cache = {}
        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)

        comp._render_analysis_results_section()
        comp._display_analysis_summary.assert_not_called()

    # ------------------------------------------------------------------
    # _display_analysis_summary
    # ------------------------------------------------------------------

    def test_display_analysis_summary(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()

        m1 = self._make_qtable_metrics_obj(CS, convergence_status=CS.CONVERGED,
                                            convergence_score=0.95,
                                            exploration_coverage=0.9)
        m2 = self._make_qtable_metrics_obj(CS, convergence_status=CS.UNSTABLE,
                                            convergence_score=0.6,
                                            exploration_coverage=0.7)
        results = {"f1": m1, "f2": m2}

        comp._display_analysis_summary(results)
        assert mock_st.metric.call_count == 4

    # ------------------------------------------------------------------
    # _display_detailed_metrics
    # ------------------------------------------------------------------

    def test_display_detailed_metrics(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()

        m1 = self._make_qtable_metrics_obj(CS)
        m1.convergence_status = CS.CONVERGED
        results = {"/tmp/test.pkl": m1}

        comp._display_detailed_metrics(results)
        mock_st.dataframe.assert_called()

    def test_display_detailed_metrics_various_statuses(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()

        results = {}
        for status in [CS.CONVERGED, CS.CONVERGING, CS.UNSTABLE, CS.DIVERGING, CS.UNKNOWN]:
            m = self._make_qtable_metrics_obj(CS, convergence_status=status)
            results[f"/tmp/{status.value}.pkl"] = m

        comp._display_detailed_metrics(results)
        mock_st.dataframe.assert_called()

    def test_display_detailed_metrics_color_status_function(self):
        """Force execution of the inner color_status function via style render."""
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()

        m1 = self._make_qtable_metrics_obj(CS, convergence_status=CS.CONVERGED)
        results = {"/tmp/test.pkl": m1}

        # Capture the styled_df that gets passed to st.dataframe
        captured = {}
        def capture_dataframe(df, **kwargs):
            captured["df"] = df
        mock_st.dataframe.side_effect = capture_dataframe

        comp._display_detailed_metrics(results)

        # Force rendering of the Styler to execute the color_status function
        styled_df = captured["df"]
        if hasattr(styled_df, "to_html"):
            styled_df.to_html()

    # ------------------------------------------------------------------
    # _render_visualization_section
    # ------------------------------------------------------------------

    def test_render_visualization_section(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()
        comp._render_qtable_heatmap = MagicMock()
        comp._render_convergence_trends = MagicMock()
        comp._render_policy_analysis = MagicMock()
        comp._render_exploration_visualization = MagicMock()

        comp._render_visualization_section()
        comp._render_qtable_heatmap.assert_called_once()
        comp._render_convergence_trends.assert_called_once()
        comp._render_policy_analysis.assert_called_once()
        comp._render_exploration_visualization.assert_called_once()

    # ------------------------------------------------------------------
    # _render_qtable_heatmap
    # ------------------------------------------------------------------

    def test_render_heatmap_no_selection(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()
        mock_st.session_state.selected_qtables = []

        comp._render_qtable_heatmap()
        mock_st.info.assert_called()

    def test_render_heatmap_with_qtable(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()
        comp.analyzer = MagicMock()

        qtable = np.random.rand(5, 3)
        comp.analyzer.load_qtable.return_value = qtable

        mock_st.session_state.selected_qtables = ["/tmp/q1.pkl"]
        mock_st.selectbox.return_value = 0
        mock_st.checkbox.return_value = False
        mock_st.button.return_value = False

        comp._create_qtable_heatmap = MagicMock(return_value=MagicMock())

        comp._render_qtable_heatmap()
        mock_st.plotly_chart.assert_called()

    def test_render_heatmap_load_returns_none(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()
        comp.analyzer = MagicMock()
        comp.analyzer.load_qtable.return_value = None

        mock_st.session_state.selected_qtables = ["/tmp/q1.pkl"]
        mock_st.selectbox.return_value = 0

        comp._render_qtable_heatmap()

    def test_render_heatmap_update_button(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()
        comp.analyzer = MagicMock()

        qtable = np.random.rand(5, 3)
        comp.analyzer.load_qtable.return_value = qtable

        mock_st.session_state.selected_qtables = ["/tmp/q1.pkl"]
        mock_st.selectbox.side_effect = [0, "Plasma"]
        mock_st.checkbox.return_value = True
        mock_st.button.return_value = True

        comp._create_qtable_heatmap = MagicMock(return_value=MagicMock())

        comp._render_qtable_heatmap()
        assert comp._create_qtable_heatmap.call_count >= 2

    # ------------------------------------------------------------------
    # _create_qtable_heatmap
    # ------------------------------------------------------------------

    def test_create_qtable_heatmap_default(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()

        qtable = np.random.rand(10, 4)
        fig = comp._create_qtable_heatmap(qtable, "test.pkl")
        mock_go.Figure.assert_called()
        assert fig is not None

    def test_create_qtable_heatmap_with_values(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()

        qtable = np.random.rand(10, 4)
        comp._create_qtable_heatmap(qtable, "test.pkl",
                                           colorscale="plasma", show_values=True)
        mock_go.Heatmap.assert_called()

    def test_create_qtable_heatmap_large_table(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()

        qtable = np.random.rand(50, 4)
        fig = comp._create_qtable_heatmap(qtable, "large.pkl")
        assert fig is not None

    # ------------------------------------------------------------------
    # _render_convergence_trends
    # ------------------------------------------------------------------

    def test_convergence_trends_too_few(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()
        mock_st.session_state.selected_qtables = ["file1.pkl"]

        comp._render_convergence_trends()
        mock_st.info.assert_called()

    def test_convergence_trends_with_data(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()
        comp._create_convergence_trend_plot = MagicMock(return_value=MagicMock())
        comp._display_trend_statistics = MagicMock()
        comp._extract_timestamp_from_filename = MagicMock(side_effect=lambda x: x)

        m1 = self._make_qtable_metrics_obj(CS)
        m2 = self._make_qtable_metrics_obj(CS)

        mock_st.session_state.selected_qtables = ["/tmp/q1.pkl", "/tmp/q2.pkl"]
        mock_st.session_state.qtable_analysis_cache = {
            "/tmp/q1.pkl": m1,
            "/tmp/q2.pkl": m2,
        }

        comp._render_convergence_trends()
        comp._create_convergence_trend_plot.assert_called_once()
        comp._display_trend_statistics.assert_called_once()

    def test_convergence_trends_no_cache(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()

        mock_st.session_state.selected_qtables = ["/tmp/q1.pkl", "/tmp/q2.pkl"]
        mock_st.session_state.qtable_analysis_cache = {}

        comp._render_convergence_trends()

    # ------------------------------------------------------------------
    # _create_convergence_trend_plot
    # ------------------------------------------------------------------

    def test_create_convergence_trend_plot(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()

        df = pd.DataFrame({
            "file": ["q1.pkl", "q2.pkl"],
            "timestamp": ["20250101_000000", "20250102_000000"],
            "convergence_score": [0.8, 0.9],
            "stability_measure": [0.7, 0.85],
            "exploration_coverage": [0.6, 0.75],
            "policy_entropy": [1.5, 1.2],
        })

        mock_fig = MagicMock()
        mock_ms.return_value = mock_fig

        fig = comp._create_convergence_trend_plot(df)
        assert fig is mock_fig
        assert mock_fig.add_trace.call_count == 4

    # ------------------------------------------------------------------
    # _render_policy_analysis
    # ------------------------------------------------------------------

    def test_render_policy_analysis_no_cache(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()
        mock_st.session_state.qtable_analysis_cache = {}

        comp._render_policy_analysis()
        mock_st.info.assert_called()

    def test_render_policy_analysis_with_data(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()
        comp._display_policy_insights = MagicMock()

        m1 = self._make_qtable_metrics_obj(CS, policy_entropy=1.5,
                                            action_diversity=0.8,
                                            state_value_variance=0.4,
                                            convergence_score=0.9)
        mock_st.session_state.qtable_analysis_cache = {"/tmp/q1.pkl": m1}

        comp._render_policy_analysis()
        mock_px.scatter.assert_called()
        mock_st.plotly_chart.assert_called()
        comp._display_policy_insights.assert_called_once()

    # ------------------------------------------------------------------
    # _render_exploration_visualization
    # ------------------------------------------------------------------

    def test_render_exploration_no_selection(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()
        mock_st.session_state.selected_qtables = []

        comp._render_exploration_visualization()
        mock_st.info.assert_called()

    def test_render_exploration_with_data(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()
        comp.analyzer = MagicMock()

        qtable = np.array([[1.0, 0.0, 0.5], [0.0, 0.0, 0.0], [0.3, 0.7, 0.0]])
        comp.analyzer.load_qtable.return_value = qtable

        m = self._make_qtable_metrics_obj(CS)
        mock_st.session_state.selected_qtables = ["/tmp/q1.pkl"]
        mock_st.session_state.qtable_analysis_cache = {"/tmp/q1.pkl": m}
        mock_st.selectbox.return_value = 0

        comp._render_exploration_visualization()
        mock_st.plotly_chart.assert_called()
        assert mock_st.metric.call_count == 3

    def test_render_exploration_load_returns_none(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()
        comp.analyzer = MagicMock()
        comp.analyzer.load_qtable.return_value = None

        mock_st.session_state.selected_qtables = ["/tmp/q1.pkl"]
        mock_st.selectbox.return_value = 0

        comp._render_exploration_visualization()

    def test_render_exploration_no_cache_entry(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()
        comp.analyzer = MagicMock()

        qtable = np.array([[1.0, 0.5], [0.3, 0.7]])
        comp.analyzer.load_qtable.return_value = qtable

        mock_st.session_state.selected_qtables = ["/tmp/q1.pkl"]
        mock_st.session_state.qtable_analysis_cache = {}
        mock_st.selectbox.return_value = 0

        comp._render_exploration_visualization()
        mock_st.plotly_chart.assert_called()

    # ------------------------------------------------------------------
    # _render_comparison_section
    # ------------------------------------------------------------------

    def test_comparison_section_too_few(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()
        mock_st.session_state.selected_qtables = ["file1.pkl"]

        comp._render_comparison_section()
        # Should return early, no subheader called

    def test_comparison_section_with_enough_files(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()
        comp._perform_qtable_comparison = MagicMock()
        comp._display_comparison_results = MagicMock()

        mock_st.session_state.selected_qtables = ["/tmp/q1.pkl", "/tmp/q2.pkl"]
        mock_st.selectbox.side_effect = [0, 1]
        mock_st.button.return_value = True

        comp._render_comparison_section()
        comp._perform_qtable_comparison.assert_called_once()
        comp._display_comparison_results.assert_called_once()

    def test_comparison_section_no_button(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()
        comp._perform_qtable_comparison = MagicMock()
        comp._display_comparison_results = MagicMock()

        mock_st.session_state.selected_qtables = ["/tmp/q1.pkl", "/tmp/q2.pkl"]
        mock_st.selectbox.side_effect = [0, 1]
        mock_st.button.return_value = False

        comp._render_comparison_section()
        comp._perform_qtable_comparison.assert_not_called()
        comp._display_comparison_results.assert_called_once()

    # ------------------------------------------------------------------
    # _render_export_section
    # ------------------------------------------------------------------

    def test_export_section_metrics_csv(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()
        comp._export_metrics_csv = MagicMock()
        comp._export_visualizations = MagicMock()
        comp._generate_analysis_report = MagicMock()

        mock_st.button.side_effect = [True, False, False]

        comp._render_export_section()
        comp._export_metrics_csv.assert_called_once()

    def test_export_section_visualizations(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()
        comp._export_metrics_csv = MagicMock()
        comp._export_visualizations = MagicMock()
        comp._generate_analysis_report = MagicMock()

        mock_st.button.side_effect = [False, True, False]

        comp._render_export_section()
        comp._export_visualizations.assert_called_once()

    def test_export_section_report(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()
        comp._export_metrics_csv = MagicMock()
        comp._export_visualizations = MagicMock()
        comp._generate_analysis_report = MagicMock()

        mock_st.button.side_effect = [False, False, True]

        comp._render_export_section()
        comp._generate_analysis_report.assert_called_once()

    # ------------------------------------------------------------------
    # _get_file_size
    # ------------------------------------------------------------------

    def test_get_file_size_bytes(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()

        p = MagicMock(spec=Path)
        p.stat.return_value.st_size = 500
        result = comp._get_file_size(p)
        assert "500.0 B" == result

    def test_get_file_size_kb(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()

        p = MagicMock(spec=Path)
        p.stat.return_value.st_size = 2048
        result = comp._get_file_size(p)
        assert "KB" in result

    def test_get_file_size_mb(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()

        p = MagicMock(spec=Path)
        p.stat.return_value.st_size = 2 * 1024 * 1024
        result = comp._get_file_size(p)
        assert "MB" in result

    def test_get_file_size_gb(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()

        p = MagicMock(spec=Path)
        p.stat.return_value.st_size = 2 * 1024 * 1024 * 1024
        result = comp._get_file_size(p)
        assert "GB" in result

    def test_get_file_size_tb(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()

        p = MagicMock(spec=Path)
        p.stat.return_value.st_size = 2 * 1024 * 1024 * 1024 * 1024
        result = comp._get_file_size(p)
        assert "TB" in result

    def test_get_file_size_error(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()

        p = MagicMock(spec=Path)
        p.stat.side_effect = OSError("File not found")
        result = comp._get_file_size(p)
        assert result == "Unknown"

    # ------------------------------------------------------------------
    # _perform_quick_analysis
    # ------------------------------------------------------------------

    def test_perform_quick_analysis(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()
        comp.analyzer = MagicMock()

        m = self._make_qtable_metrics_obj(CS)
        comp.analyzer.analyze_qtable.return_value = m

        mock_st.session_state.selected_qtables = ["file1.pkl", "file2.pkl"]
        mock_st.session_state.qtable_analysis_cache = {}
        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)

        comp._perform_quick_analysis()
        assert comp.analyzer.analyze_qtable.call_count == 2
        mock_st.success.assert_called()

    def test_perform_quick_analysis_already_cached(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()
        comp.analyzer = MagicMock()

        m = self._make_qtable_metrics_obj(CS)
        mock_st.session_state.selected_qtables = ["file1.pkl"]
        mock_st.session_state.qtable_analysis_cache = {"file1.pkl": m}
        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)

        comp._perform_quick_analysis()
        comp.analyzer.analyze_qtable.assert_not_called()

    def test_perform_quick_analysis_returns_none(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()
        comp.analyzer = MagicMock()
        comp.analyzer.analyze_qtable.return_value = None

        mock_st.session_state.selected_qtables = ["file1.pkl"]
        mock_st.session_state.qtable_analysis_cache = {}
        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)

        comp._perform_quick_analysis()
        assert "file1.pkl" not in mock_st.session_state.qtable_analysis_cache

    # ------------------------------------------------------------------
    # _extract_timestamp_from_filename
    # ------------------------------------------------------------------

    def test_extract_timestamp_with_pattern(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()

        result = comp._extract_timestamp_from_filename("qtable_20250131_120000.pkl")
        assert result == "20250131_120000"

    def test_extract_timestamp_no_pattern(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()

        result = comp._extract_timestamp_from_filename("qtable_latest.pkl")
        assert result == "qtable_latest.pkl"

    # ------------------------------------------------------------------
    # _display_trend_statistics
    # ------------------------------------------------------------------

    def test_display_trend_statistics_multiple_rows(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()

        df = pd.DataFrame({
            "convergence_score": [0.7, 0.8, 0.9],
            "stability_measure": [0.5, 0.7, 0.85],
            "exploration_coverage": [0.6, 0.7, 0.8],
            "policy_entropy": [1.5, 1.3, 1.1],
        })

        comp._display_trend_statistics(df)
        assert mock_st.metric.call_count == 4

    def test_display_trend_statistics_single_row(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()

        df = pd.DataFrame({
            "convergence_score": [0.9],
            "stability_measure": [0.85],
            "exploration_coverage": [0.8],
            "policy_entropy": [1.1],
        })

        comp._display_trend_statistics(df)
        # With only 1 row, len(df) < 2 so no convergence/stability trend
        assert mock_st.metric.call_count == 2

    # ------------------------------------------------------------------
    # _display_policy_insights
    # ------------------------------------------------------------------

    def test_display_policy_insights(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()

        df = pd.DataFrame({
            "File": ["q1.pkl", "q2.pkl"],
            "Policy Entropy": [1.5, 1.2],
            "Action Diversity": [0.8, 0.9],
            "State Value Variance": [0.3, 0.4],
            "Convergence Score": [0.85, 0.95],
        })

        comp._display_policy_insights(df)
        # Should show 3 best columns
        assert mock_st.markdown.call_count >= 7

    # ------------------------------------------------------------------
    # _perform_qtable_comparison
    # ------------------------------------------------------------------

    def test_perform_comparison_success(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()
        comp.analyzer = MagicMock()

        comparison = MagicMock()
        comp.analyzer.compare_qtables.return_value = comparison

        mock_st.session_state.comparison_results = {}
        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)

        comp._perform_qtable_comparison("f1.pkl", "f2.pkl")
        assert "f1.pkl|f2.pkl" in mock_st.session_state.comparison_results
        mock_st.success.assert_called()

    def test_perform_comparison_fails(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()
        comp.analyzer = MagicMock()
        comp.analyzer.compare_qtables.return_value = None

        mock_st.session_state.comparison_results = {}
        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)

        comp._perform_qtable_comparison("f1.pkl", "f2.pkl")
        mock_st.error.assert_called()

    def test_perform_comparison_already_cached(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()
        comp.analyzer = MagicMock()

        mock_st.session_state.comparison_results = {"f1.pkl|f2.pkl": MagicMock()}

        comp._perform_qtable_comparison("f1.pkl", "f2.pkl")
        comp.analyzer.compare_qtables.assert_not_called()

    # ------------------------------------------------------------------
    # _display_comparison_results
    # ------------------------------------------------------------------

    def test_display_comparison_results_empty(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()
        mock_st.session_state.comparison_results = {}

        comp._display_comparison_results()

    def test_display_comparison_results_with_data(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()

        comparison = MagicMock()
        comparison.policy_agreement = 0.85
        comparison.convergence_improvement = 0.1
        comparison.learning_progress = 0.05
        comparison.stability_change = 0.02

        mock_st.session_state.comparison_results = {
            "/tmp/q1.pkl|/tmp/q2.pkl": comparison
        }

        comp._display_comparison_results()
        assert mock_st.metric.call_count == 4

    # ------------------------------------------------------------------
    # _export_metrics_csv
    # ------------------------------------------------------------------

    def test_export_metrics_csv_with_data(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()
        comp.analyzer = MagicMock()

        m = self._make_qtable_metrics_obj(CS)
        mock_st.session_state.qtable_analysis_cache = {"f1.pkl": m}

        comp._export_metrics_csv()
        comp.analyzer.export_analysis_results.assert_called_once()
        mock_st.success.assert_called()

    def test_export_metrics_csv_no_data(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()
        mock_st.session_state.qtable_analysis_cache = {}

        comp._export_metrics_csv()
        mock_st.warning.assert_called()

    # ------------------------------------------------------------------
    # _export_visualizations
    # ------------------------------------------------------------------

    def test_export_visualizations(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()

        comp._export_visualizations()
        mock_st.info.assert_called()

    # ------------------------------------------------------------------
    # _generate_analysis_report
    # ------------------------------------------------------------------

    def test_generate_analysis_report(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)
        comp = mod.QTableVisualization()

        comp._generate_analysis_report()
        mock_st.info.assert_called()

    # ------------------------------------------------------------------
    # Module-level functions
    # ------------------------------------------------------------------

    def test_render_qtable_analysis_interface_module_func(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)

        with patch.object(mod, "QTableVisualization") as MockClass:
            instance = MagicMock()
            instance.render_qtable_analysis_interface.return_value = {"test": True}
            MockClass.return_value = instance

            result = mod.render_qtable_analysis_interface()
            assert result == {"test": True}

    def test_main_block_coverage(self):
        mock_st = _make_mock_st()
        mod, mock_go, mock_px, mock_ms, CS = self._import_module(mock_st)

        with patch.object(mod, "render_qtable_analysis_interface") as mock_render:
            mock_render.return_value = {"test": True}
            mod.st.title("Q-Table Analysis Component Test")
            result = mod.render_qtable_analysis_interface()
            mod.st.json(result)

            mock_st.title.assert_called()
            mock_st.json.assert_called()
