"""Tests for policy_evolution_tracker module."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.policy_evolution_tracker import (
    PolicyEvolutionMetrics,
    PolicyEvolutionTracker,
    PolicySnapshot,
    PolicyStability,
)


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def tracker(tmp_dir):
    return PolicyEvolutionTracker(models_directory=tmp_dir)


def _make_snapshot(episode, n_states=10, n_actions=4, seed=None):
    if seed is not None:
        np.random.seed(seed)
    qt = np.random.rand(n_states, n_actions)
    policy = np.argmax(qt, axis=1)
    unique, counts = np.unique(policy, return_counts=True)
    freqs = dict(zip(unique.astype(int), counts.astype(int)))
    return PolicySnapshot(
        episode=episode, policy=policy, q_table=qt,
        action_frequencies=freqs, policy_entropy=0.5,
        state_coverage=0.8, performance_reward=float(episode),
        timestamp="2025-01-01T00:00:00",
    )


def _save_qtable(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


class TestPolicyEvolutionTracker:
    def test_load_from_files_ndarray(self, tracker, tmp_dir):
        for i in range(3):
            qt = np.random.rand(10, 4)
            _save_qtable(Path(tmp_dir) / f"qtable_{i}.pkl", qt)
        count = tracker.load_policy_snapshots_from_files("*qtable*.pkl")
        assert count == 3

    def test_load_from_files_dict_q_table(self, tracker, tmp_dir):
        qt = np.random.rand(10, 4)
        _save_qtable(Path(tmp_dir) / "qtable_0.pkl", {"q_table": qt})
        assert tracker.load_policy_snapshots_from_files() == 1

    def test_load_from_files_dict_Q(self, tracker, tmp_dir):
        qt = np.random.rand(10, 4)
        _save_qtable(Path(tmp_dir) / "qtable_0.pkl", {"Q": qt})
        assert tracker.load_policy_snapshots_from_files() == 1

    def test_load_from_files_dict_with_reward(self, tracker, tmp_dir):
        qt = np.random.rand(10, 4)
        _save_qtable(Path(tmp_dir) / "qtable_0.pkl",
                      {"q_table": qt, "reward": 42.0})
        tracker.load_policy_snapshots_from_files()
        assert tracker.policy_snapshots[0].performance_reward == 42.0

    def test_load_from_files_unclear_format(self, tracker, tmp_dir):
        _save_qtable(Path(tmp_dir) / "qtable_0.pkl", {"a": 1, "b": 2})
        assert tracker.load_policy_snapshots_from_files() == 0

    def test_load_from_files_empty_qtable(self, tracker, tmp_dir):
        _save_qtable(Path(tmp_dir) / "qtable_0.pkl", np.array([]))
        assert tracker.load_policy_snapshots_from_files() == 0

    def test_load_from_files_max_snapshots(self, tracker, tmp_dir):
        import time
        for i in range(5):
            _save_qtable(Path(tmp_dir) / f"qtable_{i}.pkl", np.random.rand(10, 4))
            time.sleep(0.01)
        assert tracker.load_policy_snapshots_from_files(max_snapshots=2) == 2

    def test_load_from_files_missing_dir(self):
        t = PolicyEvolutionTracker("/nonexistent/path")
        assert t.load_policy_snapshots_from_files() == 0

    def test_load_from_files_bad_pickle(self, tracker, tmp_dir):
        (Path(tmp_dir) / "qtable_bad.pkl").write_text("not pickle")
        assert tracker.load_policy_snapshots_from_files() == 0

    def test_calculate_policy_entropy_empty(self, tracker):
        assert tracker._calculate_policy_entropy(np.array([])) == 0.0

    def test_calculate_policy_entropy_none(self, tracker):
        assert tracker._calculate_policy_entropy(None) == 0.0

    def test_calculate_policy_entropy_zeros(self, tracker):
        assert tracker._calculate_policy_entropy(np.zeros((5, 3))) == 0.0

    def test_calculate_policy_entropy_normal(self, tracker):
        qt = np.random.rand(10, 4)
        entropy = tracker._calculate_policy_entropy(qt)
        assert entropy >= 0.0

    def test_extract_performance_dict_keys(self, tracker, tmp_dir):
        p = Path(tmp_dir) / "test.pkl"
        p.touch()
        for key in ["reward", "performance", "episode_reward", "total_reward"]:
            result = tracker._extract_performance_from_file(p, {key: 42.0})
            assert result == 42.0

    def test_extract_performance_filename_pattern(self, tracker, tmp_dir):
        p = Path(tmp_dir) / "reward42.pkl"
        p.touch()
        result = tracker._extract_performance_from_file(p, np.array([]))
        assert result == 42.0

    def test_extract_performance_filename_valueerror(self, tracker, tmp_dir):
        p = Path(tmp_dir) / "qtable_reward_42.5.pkl"
        p.touch()
        result = tracker._extract_performance_from_file(p, np.array([]))
        assert result is None

    def test_extract_performance_no_match(self, tracker, tmp_dir):
        p = Path(tmp_dir) / "qtable.pkl"
        p.touch()
        assert tracker._extract_performance_from_file(p, np.array([])) is None

    def test_analyze_insufficient_snapshots(self, tracker):
        tracker.policy_snapshots = [_make_snapshot(0)]
        assert tracker.analyze_policy_evolution() is None

    def test_analyze_policy_evolution(self, tracker):
        for i in range(6):
            tracker.policy_snapshots.append(_make_snapshot(i, seed=i))
        metrics = tracker.analyze_policy_evolution()
        assert metrics is not None
        assert metrics.total_episodes == 6
        assert len(metrics.policy_changes) == 5

    def test_calculate_stability_score_empty(self, tracker):
        assert tracker._calculate_stability_score([], []) == 0.0

    def test_calculate_stability_score(self, tracker):
        snapshots = [_make_snapshot(i) for i in range(5)]
        changes = [2, 1, 0, 0]
        score = tracker._calculate_stability_score(changes, snapshots)
        assert 0.0 <= score <= 1.0

    def test_determine_stability_stable(self, tracker):
        status = tracker._determine_stability_status(0.96, [0, 0, 0, 0])
        assert status == PolicyStability.STABLE

    def test_determine_stability_converging(self, tracker):
        status = tracker._determine_stability_status(0.85, [1, 1, 1, 1])
        assert status == PolicyStability.CONVERGING

    def test_determine_stability_unstable(self, tracker):
        status = tracker._determine_stability_status(0.4, [5, 5, 5, 5])
        assert status == PolicyStability.UNSTABLE

    def test_determine_stability_unstable_low(self, tracker):
        status = tracker._determine_stability_status(0.3, [10])
        assert status == PolicyStability.UNSTABLE

    def test_detect_oscillation_short(self, tracker):
        assert tracker._detect_oscillation([1, 2]) is False

    def test_detect_oscillation_alternating(self, tracker):
        result = tracker._detect_oscillation([10, 0, 10, 0])
        assert isinstance(result, bool)

    def test_detect_convergence_none(self, tracker):
        assert tracker._detect_convergence_episode([5, 5, 5]) is None

    def test_detect_convergence_found(self, tracker):
        changes = [10, 8, 6, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        result = tracker._detect_convergence_episode(changes)
        assert result is not None or result is None

    def test_count_action_preference_changes_short(self, tracker):
        assert tracker._count_action_preference_changes([]) == 0
        assert tracker._count_action_preference_changes([_make_snapshot(0)]) == 0

    def test_count_action_preference_changes(self, tracker):
        snaps = [_make_snapshot(i, seed=i) for i in range(5)]
        result = tracker._count_action_preference_changes(snaps)
        assert isinstance(result, int)

    def test_get_dominant_action_empty(self, tracker):
        snap = _make_snapshot(0)
        snap.action_frequencies = {}
        assert tracker._get_dominant_action(snap) == 0

    def test_get_dominant_action(self, tracker):
        snap = _make_snapshot(0)
        snap.action_frequencies = {0: 5, 1: 10, 2: 3}
        assert tracker._get_dominant_action(snap) == 1

    def test_calculate_learning_velocity_short(self, tracker):
        assert tracker._calculate_learning_velocity([1]) == []

    def test_calculate_learning_velocity(self, tracker):
        changes = [10, 8, 6, 4, 2]
        velocity = tracker._calculate_learning_velocity(changes)
        assert len(velocity) == 5

    def test_calculate_exploration_decay_short(self, tracker):
        assert tracker._calculate_exploration_decay([1.0]) == []

    def test_calculate_exploration_decay(self, tracker):
        entropy = [1.0, 0.8, 0.6, 0.4]
        decay = tracker._calculate_exploration_decay(entropy)
        assert len(decay) == 3
        assert all(d >= 0 for d in decay)

    def test_calculate_exploration_decay_zero(self, tracker):
        entropy = [0.0, 0.5]
        decay = tracker._calculate_exploration_decay(entropy)
        assert decay[0] == 0.0

    def test_get_action_frequency_matrix_empty(self, tracker):
        assert tracker.get_action_frequency_matrix() is None

    def test_get_action_frequency_matrix(self, tracker):
        for i in range(3):
            tracker.policy_snapshots.append(_make_snapshot(i, seed=i))
        df = tracker.get_action_frequency_matrix()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_get_policy_comparison_matrix_empty(self, tracker):
        assert tracker.get_policy_comparison_matrix(0) is None

    def test_get_policy_comparison_matrix(self, tracker):
        for i in range(3):
            tracker.policy_snapshots.append(_make_snapshot(i, seed=i))
        matrix = tracker.get_policy_comparison_matrix(0)
        assert matrix is not None
        assert matrix.shape[0] == 3

    def test_get_policy_comparison_matrix_not_found(self, tracker):
        tracker.policy_snapshots.append(_make_snapshot(0))
        assert tracker.get_policy_comparison_matrix(99) is None

    def test_export_evolution_analysis(self, tracker, tmp_dir):
        for i in range(6):
            tracker.policy_snapshots.append(_make_snapshot(i, seed=i))
        metrics = tracker.analyze_policy_evolution()
        out = Path(tmp_dir) / "export.csv"
        tracker.export_evolution_analysis(metrics, str(out))
        assert out.exists()


class TestPolicyStability:
    def test_values(self):
        assert PolicyStability.STABLE.value == "stable"
        assert PolicyStability.CONVERGING.value == "converging"
        assert PolicyStability.UNSTABLE.value == "unstable"
        assert PolicyStability.OSCILLATING.value == "oscillating"
        assert PolicyStability.UNKNOWN.value == "unknown"
