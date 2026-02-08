"""Tests for qtable_analyzer module."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from analysis.qtable_analyzer import (
    ConvergenceStatus,
    QTableAnalyzer,
    QTableComparison,
    QTableMetrics,
)


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def analyzer(tmp_dir):
    return QTableAnalyzer(models_directory=tmp_dir)


def _save_qtable(path, qtable):
    with open(path, "wb") as f:
        pickle.dump(qtable, f)


def _save_dict_qtable(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


class TestQTableAnalyzer:
    def test_load_qtable_ndarray(self, analyzer, tmp_dir):
        qt = np.random.rand(10, 4)
        p = Path(tmp_dir) / "qt.pkl"
        _save_qtable(p, qt)
        loaded = analyzer.load_qtable(p)
        np.testing.assert_array_equal(loaded, qt)

    def test_load_qtable_dict_q_table(self, analyzer, tmp_dir):
        qt = np.random.rand(10, 4)
        p = Path(tmp_dir) / "qt.pkl"
        _save_dict_qtable(p, {"q_table": qt})
        loaded = analyzer.load_qtable(p)
        np.testing.assert_array_equal(loaded, qt)

    def test_load_qtable_dict_Q(self, analyzer, tmp_dir):
        qt = np.random.rand(10, 4)
        p = Path(tmp_dir) / "qt.pkl"
        _save_dict_qtable(p, {"Q": qt})
        loaded = analyzer.load_qtable(p)
        np.testing.assert_array_equal(loaded, qt)

    def test_load_qtable_dict_single_key(self, analyzer, tmp_dir):
        qt = np.random.rand(10, 4)
        p = Path(tmp_dir) / "qt.pkl"
        _save_dict_qtable(p, {"data": qt})
        loaded = analyzer.load_qtable(p)
        np.testing.assert_array_equal(loaded, qt)

    def test_load_qtable_dict_unclear(self, analyzer, tmp_dir):
        p = Path(tmp_dir) / "qt.pkl"
        _save_dict_qtable(p, {"a": 1, "b": 2})
        assert analyzer.load_qtable(p) is None

    def test_load_qtable_unknown_type(self, analyzer, tmp_dir):
        p = Path(tmp_dir) / "qt.pkl"
        with open(p, "wb") as f:
            pickle.dump("string", f)
        assert analyzer.load_qtable(p) is None

    def test_load_qtable_error(self, analyzer, tmp_dir):
        p = Path(tmp_dir) / "qt.pkl"
        p.write_text("not pickle")
        assert analyzer.load_qtable(p) is None

    def test_convergence_score_empty(self, analyzer):
        assert analyzer.calculate_convergence_score(np.array([])) == 0.0

    def test_convergence_score_none(self, analyzer):
        assert analyzer.calculate_convergence_score(None) == 0.0

    def test_convergence_score_all_zero(self, analyzer):
        assert analyzer.calculate_convergence_score(np.zeros((5, 3))) == 0.0

    def test_convergence_score_single_nonzero(self, analyzer):
        qt = np.zeros((5, 3))
        qt[0, 0] = 1.0
        assert analyzer.calculate_convergence_score(qt) == 0.0

    def test_convergence_score_normal(self, analyzer):
        qt = np.ones((10, 4)) * 5.0
        qt += np.random.rand(10, 4) * 0.1
        score = analyzer.calculate_convergence_score(qt)
        assert 0.0 <= score <= 1.0

    def test_policy_entropy_empty(self, analyzer):
        assert analyzer.calculate_policy_entropy(np.array([])) == 0.0

    def test_policy_entropy_none(self, analyzer):
        assert analyzer.calculate_policy_entropy(None) == 0.0

    def test_policy_entropy_all_zero(self, analyzer):
        assert analyzer.calculate_policy_entropy(np.zeros((5, 3))) == 0.0

    def test_policy_entropy_nonzero(self, analyzer):
        qt = np.random.rand(10, 4)
        entropy = analyzer.calculate_policy_entropy(qt)
        assert entropy >= 0.0

    def test_exploration_coverage_empty(self, analyzer):
        cov, vis, unvis = analyzer.calculate_exploration_coverage(np.array([]))
        assert cov == 0.0

    def test_exploration_coverage_none(self, analyzer):
        cov, vis, unvis = analyzer.calculate_exploration_coverage(None)
        assert cov == 0.0

    def test_exploration_coverage_partial(self, analyzer):
        qt = np.zeros((10, 4))
        qt[0:5] = np.random.rand(5, 4)
        cov, vis, unvis = analyzer.calculate_exploration_coverage(qt)
        assert vis == 5
        assert unvis == 5
        assert cov == pytest.approx(0.5)

    def test_determine_convergence_converged(self, analyzer):
        m = QTableMetrics(
            shape=(10, 4), total_states=10, total_actions=4,
            non_zero_values=40, sparsity=0.0,
            mean_q_value=1.0, std_q_value=0.01, min_q_value=0.9, max_q_value=1.1,
            q_value_range=0.2, convergence_score=0.95, stability_measure=0.98,
            convergence_status=ConvergenceStatus.UNKNOWN,
            policy_entropy=0.5, action_diversity=0.5, state_value_variance=0.01,
            exploration_coverage=1.0, visited_states=10, unvisited_states=0,
            analysis_timestamp="now"
        )
        assert analyzer.determine_convergence_status(m) == ConvergenceStatus.CONVERGED

    def test_determine_convergence_converging(self, analyzer):
        m = QTableMetrics(
            shape=(10, 4), total_states=10, total_actions=4,
            non_zero_values=40, sparsity=0.0,
            mean_q_value=1.0, std_q_value=0.1, min_q_value=0.5, max_q_value=1.5,
            q_value_range=1.0, convergence_score=0.75, stability_measure=0.85,
            convergence_status=ConvergenceStatus.UNKNOWN,
            policy_entropy=0.5, action_diversity=0.5, state_value_variance=0.1,
            exploration_coverage=1.0, visited_states=10, unvisited_states=0,
            analysis_timestamp="now"
        )
        assert analyzer.determine_convergence_status(m) == ConvergenceStatus.CONVERGING

    def test_determine_convergence_unstable(self, analyzer):
        m = QTableMetrics(
            shape=(10, 4), total_states=10, total_actions=4,
            non_zero_values=40, sparsity=0.0,
            mean_q_value=1.0, std_q_value=0.5, min_q_value=0.0, max_q_value=2.0,
            q_value_range=2.0, convergence_score=0.5, stability_measure=0.5,
            convergence_status=ConvergenceStatus.UNKNOWN,
            policy_entropy=0.5, action_diversity=0.5, state_value_variance=0.5,
            exploration_coverage=1.0, visited_states=10, unvisited_states=0,
            analysis_timestamp="now"
        )
        assert analyzer.determine_convergence_status(m) == ConvergenceStatus.UNSTABLE

    def test_determine_convergence_diverging(self, analyzer):
        m = QTableMetrics(
            shape=(10, 4), total_states=10, total_actions=4,
            non_zero_values=40, sparsity=0.0,
            mean_q_value=1.0, std_q_value=1.0, min_q_value=-1.0, max_q_value=3.0,
            q_value_range=4.0, convergence_score=0.1, stability_measure=0.1,
            convergence_status=ConvergenceStatus.UNKNOWN,
            policy_entropy=1.0, action_diversity=1.0, state_value_variance=1.0,
            exploration_coverage=0.5, visited_states=5, unvisited_states=5,
            analysis_timestamp="now"
        )
        assert analyzer.determine_convergence_status(m) == ConvergenceStatus.DIVERGING

    def test_analyze_qtable(self, analyzer, tmp_dir):
        qt = np.random.rand(20, 4) * 10
        p = Path(tmp_dir) / "qt.pkl"
        _save_qtable(p, qt)
        metrics = analyzer.analyze_qtable(p)
        assert metrics is not None
        assert metrics.total_states == 20
        assert metrics.total_actions == 4

    def test_analyze_qtable_cached(self, analyzer, tmp_dir):
        qt = np.random.rand(20, 4)
        p = Path(tmp_dir) / "qt.pkl"
        _save_qtable(p, qt)
        m1 = analyzer.analyze_qtable(p)
        m2 = analyzer.analyze_qtable(p)
        assert m1 is m2

    def test_analyze_qtable_none(self, analyzer, tmp_dir):
        p = Path(tmp_dir) / "qt.pkl"
        p.write_text("bad")
        assert analyzer.analyze_qtable(p) is None

    def test_compare_qtables(self, analyzer, tmp_dir):
        qt1 = np.random.rand(10, 4)
        qt2 = qt1 + np.random.rand(10, 4) * 0.1
        p1 = Path(tmp_dir) / "qt1.pkl"
        p2 = Path(tmp_dir) / "qt2.pkl"
        _save_qtable(p1, qt1)
        _save_qtable(p2, qt2)
        comp = analyzer.compare_qtables(p1, p2)
        assert comp is not None
        assert 0 <= comp.policy_agreement <= 1

    def test_compare_qtables_diff_shape(self, analyzer, tmp_dir):
        p1 = Path(tmp_dir) / "qt1.pkl"
        p2 = Path(tmp_dir) / "qt2.pkl"
        _save_qtable(p1, np.random.rand(10, 4))
        _save_qtable(p2, np.random.rand(5, 3))
        # Need to clear cache since shapes differ
        analyzer.analysis_cache.clear()
        assert analyzer.compare_qtables(p1, p2) is None

    def test_compare_qtables_bad_file(self, analyzer, tmp_dir):
        p1 = Path(tmp_dir) / "qt1.pkl"
        p2 = Path(tmp_dir) / "qt2.pkl"
        p1.write_text("bad")
        p2.write_text("bad")
        assert analyzer.compare_qtables(p1, p2) is None

    def test_get_available_qtables(self, analyzer, tmp_dir):
        _save_qtable(Path(tmp_dir) / "a.pkl", np.zeros((5, 3)))
        _save_qtable(Path(tmp_dir) / "b.pkl", np.zeros((5, 3)))
        files = analyzer.get_available_qtables()
        assert len(files) == 2

    def test_get_available_qtables_missing_dir(self):
        a = QTableAnalyzer("/nonexistent/path")
        assert a.get_available_qtables() == []

    def test_batch_analyze(self, analyzer, tmp_dir):
        for i in range(3):
            _save_qtable(Path(tmp_dir) / f"qt{i}.pkl", np.random.rand(10, 4))
        results = analyzer.batch_analyze_qtables()
        assert len(results) == 3

    def test_batch_analyze_explicit(self, analyzer, tmp_dir):
        p = Path(tmp_dir) / "qt.pkl"
        _save_qtable(p, np.random.rand(10, 4))
        results = analyzer.batch_analyze_qtables(file_paths=[p])
        assert len(results) == 1

    def test_export_analysis_results(self, analyzer, tmp_dir):
        qt = np.random.rand(10, 4)
        p = Path(tmp_dir) / "qt.pkl"
        _save_qtable(p, qt)
        metrics = analyzer.analyze_qtable(p)
        out = Path(tmp_dir) / "results.csv"
        analyzer.export_analysis_results({str(p): metrics}, str(out))
        assert out.exists()

    def test_export_empty_results(self, analyzer, tmp_dir):
        out = Path(tmp_dir) / "results.csv"
        analyzer.export_analysis_results({}, str(out))
        assert not out.exists()


class TestConvergenceStatus:
    def test_values(self):
        assert ConvergenceStatus.CONVERGED.value == "converged"
        assert ConvergenceStatus.CONVERGING.value == "converging"
        assert ConvergenceStatus.UNSTABLE.value == "unstable"
        assert ConvergenceStatus.DIVERGING.value == "diverging"
        assert ConvergenceStatus.UNKNOWN.value == "unknown"
