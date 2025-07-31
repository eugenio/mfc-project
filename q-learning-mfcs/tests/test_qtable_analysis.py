#!/usr/bin/env python3
"""
Comprehensive test suite for Q-Table Analysis System

Tests for User Story 1.2.1: Interactive Q-Table Analysis
Created: 2025-07-31
"""

import pytest
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import tempfile
import os

# Import the modules to test
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.qtable_analyzer import (
    QTableAnalyzer,
    QTableMetrics,
    QTableComparison,
    ConvergenceStatus
)


class TestQTableAnalyzer:
    """Test Q-table analysis functionality."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a QTableAnalyzer instance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield QTableAnalyzer(models_directory=temp_dir)
    
    @pytest.fixture
    def sample_q_table(self):
        """Create a sample Q-table for testing."""
        np.random.seed(42)  # For reproducible tests
        n_states, n_actions = 20, 4
        q_table = np.random.randn(n_states, n_actions)
        
        # Add some structure to make analysis meaningful
        for i in range(n_states):
            best_action = i % n_actions
            q_table[i, best_action] += 2.0  # Make this action clearly better
        
        return q_table
    
    @pytest.fixture
    def converged_q_table(self):
        """Create a converged Q-table for testing."""
        n_states, n_actions = 20, 4
        q_table = np.zeros((n_states, n_actions))
        
        # Create a deterministic policy (converged)
        for i in range(n_states):
            best_action = i % n_actions
            q_table[i, best_action] = 10.0  # Dominant action
            # Add small noise to other actions
            for j in range(n_actions):
                if j != best_action:
                    q_table[i, j] = np.random.normal(0, 0.1)
        
        return q_table
    
    def test_analyzer_initialization(self, analyzer):
        """Test QTableAnalyzer initialization."""
        assert analyzer is not None
        assert hasattr(analyzer, 'analyze_qtable')
        assert hasattr(analyzer, 'compare_qtables')
        assert hasattr(analyzer, 'load_qtable')
        assert hasattr(analyzer, 'calculate_convergence_score')
        assert hasattr(analyzer, 'calculate_policy_entropy')
    
    def test_load_qtable_from_pickle(self, analyzer, sample_q_table):
        """Test loading Q-table from pickle file."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            pickle.dump(sample_q_table, temp_file)
            temp_file.flush()
            
            loaded_qtable = analyzer.load_qtable(temp_file.name)
            
            assert loaded_qtable is not None
            assert np.array_equal(loaded_qtable, sample_q_table)
            
            os.unlink(temp_file.name)
    
    def test_calculate_convergence_score(self, analyzer, converged_q_table, sample_q_table):
        """Test convergence score calculation."""
        # Converged table should have higher score
        conv_score = analyzer.calculate_convergence_score(converged_q_table)
        random_score = analyzer.calculate_convergence_score(sample_q_table)
        
        assert 0 <= conv_score <= 1
        assert 0 <= random_score <= 1
        assert conv_score >= random_score  # Converged should score higher
    
    def test_calculate_policy_entropy(self, analyzer, converged_q_table, sample_q_table):
        """Test policy entropy calculation."""
        # Converged table should have lower entropy (more deterministic)
        conv_entropy = analyzer.calculate_policy_entropy(converged_q_table)
        random_entropy = analyzer.calculate_policy_entropy(sample_q_table)
        
        assert conv_entropy >= 0
        assert random_entropy >= 0
        assert conv_entropy <= random_entropy  # Converged should be more deterministic
    
    def test_calculate_exploration_coverage(self, analyzer, sample_q_table):
        """Test exploration coverage calculation."""
        coverage, visited, unvisited = analyzer.calculate_exploration_coverage(sample_q_table)
        
        assert 0 <= coverage <= 1
        assert visited >= 0
        assert unvisited >= 0
        assert visited + unvisited == sample_q_table.shape[0]  # Total states
    
    def test_analyze_qtable_from_file(self, analyzer, sample_q_table):
        """Test complete Q-table analysis from file."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            pickle.dump(sample_q_table, temp_file)
            temp_file.flush()
            
            metrics = analyzer.analyze_qtable(temp_file.name)
            
            assert metrics is not None
            assert isinstance(metrics, QTableMetrics)
            assert metrics.shape == sample_q_table.shape
            assert metrics.total_states == sample_q_table.shape[0]
            assert metrics.total_actions == sample_q_table.shape[1]
            assert 0 <= metrics.convergence_score <= 1
            assert 0 <= metrics.sparsity <= 1
            assert metrics.convergence_status in ConvergenceStatus
            
            os.unlink(temp_file.name)
    
    def test_compare_qtables(self, analyzer, sample_q_table, converged_q_table):
        """Test Q-table comparison functionality."""
        # Create temporary files for both Q-tables
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file1:
            pickle.dump(sample_q_table, temp_file1)
            temp_file1.flush()
            
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file2:
                pickle.dump(converged_q_table, temp_file2)
                temp_file2.flush()
                
                comparison = analyzer.compare_qtables(temp_file1.name, temp_file2.name)
                
                assert comparison is not None
                assert isinstance(comparison, QTableComparison)
                assert hasattr(comparison, 'table1_metrics')
                assert hasattr(comparison, 'table2_metrics')
                assert hasattr(comparison, 'policy_agreement')
                assert hasattr(comparison, 'convergence_improvement')
                assert 0 <= comparison.policy_agreement <= 1
                
                os.unlink(temp_file1.name)
                os.unlink(temp_file2.name)
    
    def test_batch_analyze_qtables(self, analyzer, sample_q_table):
        """Test batch analysis of multiple Q-tables."""
        # Create multiple Q-table files
        temp_files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False, dir=analyzer.models_dir) as temp_file:
                # Create slightly different Q-tables
                q_table = sample_q_table + np.random.randn(*sample_q_table.shape) * 0.1
                pickle.dump(q_table, temp_file)
                temp_files.append(Path(temp_file.name))
        
        try:
            # Batch analyze
            results = analyzer.batch_analyze_qtables(temp_files)
            
            assert isinstance(results, dict)
            assert len(results) == 3
            
            for file_path, metrics in results.items():
                assert isinstance(metrics, QTableMetrics)
                assert 0 <= metrics.convergence_score <= 1
                
        finally:
            # Clean up
            for temp_file in temp_files:
                if temp_file.exists():
                    temp_file.unlink()
    
    def test_export_analysis_results(self, analyzer, sample_q_table):
        """Test exporting analysis results to CSV."""
        # Create temporary Q-table file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            pickle.dump(sample_q_table, temp_file)
            temp_file.flush()
            
            # Analyze the Q-table
            metrics = analyzer.analyze_qtable(temp_file.name)
            results = {temp_file.name: metrics}
            
            # Export results
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as output_file:
                analyzer.export_analysis_results(results, output_file.name)
                
                # Verify file was created and has content
                assert os.path.exists(output_file.name)
                assert os.path.getsize(output_file.name) > 0
                
                # Read and verify CSV structure
                df = pd.read_csv(output_file.name)
                assert 'file_path' in df.columns
                assert 'convergence_score' in df.columns
                assert 'convergence_status' in df.columns
                assert len(df) == 1  # One Q-table analyzed
                
                os.unlink(output_file.name)
            
            os.unlink(temp_file.name)


class TestQTableMetrics:
    """Test QTableMetrics dataclass functionality."""
    
    def test_qtable_metrics_creation(self):
        """Test creating QTableMetrics with all fields."""
        metrics = QTableMetrics(
            shape=(20, 4),
            total_states=20,
            total_actions=4,
            non_zero_values=60,
            sparsity=0.25,
            mean_q_value=1.5,
            std_q_value=2.0,
            min_q_value=-3.0,
            max_q_value=5.0,
            q_value_range=8.0,
            convergence_score=0.85,
            stability_measure=0.90,
            convergence_status=ConvergenceStatus.CONVERGING,
            policy_entropy=1.2,
            action_diversity=0.8,
            state_value_variance=0.5,
            exploration_coverage=0.75,
            visited_states=15,
            unvisited_states=5,
            analysis_timestamp="2025-07-31T12:00:00"
        )
        
        assert metrics.shape == (20, 4)
        assert metrics.total_states == 20
        assert metrics.convergence_score == 0.85
        assert metrics.convergence_status == ConvergenceStatus.CONVERGING
        assert metrics.visited_states + metrics.unvisited_states == metrics.total_states


class TestConvergenceStatus:
    """Test ConvergenceStatus enum."""
    
    def test_convergence_status_values(self):
        """Test that ConvergenceStatus has expected values."""
        expected_statuses = ['CONVERGED', 'CONVERGING', 'UNSTABLE', 'DIVERGING', 'UNKNOWN']
        
        for status_name in expected_statuses:
            assert hasattr(ConvergenceStatus, status_name)
            status_value = getattr(ConvergenceStatus, status_name)
            assert isinstance(status_value, ConvergenceStatus)


class TestIntegrationScenarios:
    """Test integration scenarios for Q-table analysis."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield QTableAnalyzer(models_directory=temp_dir)
    
    def test_full_qtable_analysis_workflow(self, analyzer):
        """Test complete Q-table analysis workflow."""
        # Create a progression of Q-tables showing convergence
        q_tables = []
        
        for i in range(5):  # 5 training snapshots
            n_states, n_actions = 15, 3
            q_table = np.random.randn(n_states, n_actions)
            
            # Add convergence pattern (less randomness over time)
            convergence_factor = i / 4.0  # 0 to 1
            for state in range(n_states):
                best_action = state % n_actions
                q_table[state, best_action] += 3.0 * (convergence_factor + 0.5)
            
            q_tables.append(q_table)
        
        # Create temporary files and analyze
        temp_files = []
        metrics_list = []
        
        try:
            for i, q_table in enumerate(q_tables):
                with tempfile.NamedTemporaryFile(suffix=f'_{i:03d}.pkl', delete=False, dir=analyzer.models_dir) as temp_file:
                    pickle.dump(q_table, temp_file)
                    temp_file.flush()
                    temp_files.append(temp_file.name)
                
                # Analyze each Q-table (after file is closed)
                metrics = analyzer.analyze_qtable(temp_files[-1])
                assert metrics is not None
                metrics_list.append(metrics)
            
            # Verify progression: later Q-tables should be more converged
            convergence_scores = [m.convergence_score for m in metrics_list]
            assert len(convergence_scores) == 5
            
            # Generally expect improvement (allowing some noise)
            assert convergence_scores[-1] >= convergence_scores[0] - 0.2
            
            # Test batch analysis
            file_paths = [Path(f) for f in temp_files]
            batch_results = analyzer.batch_analyze_qtables(file_paths)
            assert len(batch_results) == 5
            
        finally:
            # Clean up
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
    
    def test_performance_with_large_qtables(self, analyzer):
        """Test performance with moderately large Q-tables."""
        import time
        
        # Create a moderately large Q-table
        n_states, n_actions = 500, 6
        large_q_table = np.random.randn(n_states, n_actions)
        
        # Add structure to make analysis meaningful
        for state in range(n_states):
            best_action = state % n_actions
            large_q_table[state, best_action] += 1.5
        
        # Time the analysis
        start_time = time.time()
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            pickle.dump(large_q_table, temp_file)
            temp_file.flush()
            
            metrics = analyzer.analyze_qtable(temp_file.name)
            
            end_time = time.time()
            analysis_time = end_time - start_time
            
            # Analysis should complete in reasonable time (< 2 seconds)
            assert analysis_time < 2.0, f"Analysis took {analysis_time:.2f}s, expected <2s"
            
            # Results should still be valid
            assert metrics is not None
            assert 0 <= metrics.convergence_score <= 1
            assert metrics.total_states == n_states
            assert metrics.total_actions == n_actions
            
            os.unlink(temp_file.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])