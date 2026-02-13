"""
Q-Table Analysis System

This module provides comprehensive analysis capabilities for Q-learning tables,
including convergence metrics, policy quality analysis, and visualization support.

User Story 1.2.1: Interactive Q-Table Analysis
Created: 2025-07-31
Last Modified: 2025-07-31
"""

import logging
import pickle
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConvergenceStatus(Enum):
    """Q-table convergence status levels."""
    CONVERGED = "converged"
    CONVERGING = "converging"
    UNSTABLE = "unstable"
    DIVERGING = "diverging"
    UNKNOWN = "unknown"


@dataclass
class QTableMetrics:
    """Comprehensive Q-table metrics."""

    # Basic structure metrics
    shape: Tuple[int, ...]
    total_states: int
    total_actions: int
    non_zero_values: int
    sparsity: float

    # Value statistics
    mean_q_value: float
    std_q_value: float
    min_q_value: float
    max_q_value: float
    q_value_range: float

    # Convergence metrics
    convergence_score: float
    stability_measure: float
    convergence_status: ConvergenceStatus

    # Policy metrics
    policy_entropy: float
    action_diversity: float
    state_value_variance: float

    # Exploration metrics
    exploration_coverage: float
    visited_states: int
    unvisited_states: int

    # Metadata
    analysis_timestamp: str
    file_path: Optional[str] = None


@dataclass
class QTableComparison:
    """Comparison results between two Q-tables."""

    table1_metrics: QTableMetrics
    table2_metrics: QTableMetrics

    # Difference metrics
    value_difference_mean: float
    value_difference_std: float
    policy_agreement: float
    convergence_improvement: float

    # Evolution metrics
    learning_progress: float
    stability_change: float
    exploration_change: float

    comparison_timestamp: str


class QTableAnalyzer:
    """Comprehensive Q-table analysis system."""

    def __init__(self, models_directory: str = "q_learning_models"):
        """
        Initialize Q-table analyzer.
        
        Args:
            models_directory: Directory containing Q-table pickle files
        """
        self.models_dir = Path(models_directory)
        self.analysis_cache: Dict[str, QTableMetrics] = {}

        # Convergence thresholds
        self.convergence_thresholds = {
            'stability_threshold': 0.95,
            'convergence_threshold': 0.9,
            'divergence_threshold': 0.3
        }

    def load_qtable(self, file_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Load Q-table from pickle file.
        
        Args:
            file_path: Path to pickle file
            
        Returns:
            Q-table as numpy array
        """
        file_path = Path(file_path)

        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            # Handle different pickle file formats
            if isinstance(data, np.ndarray):
                return data
            elif isinstance(data, dict):
                # Look for Q-table in dictionary
                if 'q_table' in data:
                    return data['q_table']
                elif 'Q' in data:
                    return data['Q']
                elif len(data) == 1:
                    # Single key dictionary
                    return list(data.values())[0]
                else:
                    logger.warning(f"Unclear Q-table format in {file_path}")
                    return None
            else:
                logger.warning(f"Unknown pickle format in {file_path}: {type(data)}")
                return None

        except Exception as e:
            logger.error(f"Error loading Q-table from {file_path}: {e}")
            return None

    def calculate_convergence_score(self, qtable: np.ndarray, window_size: int = 10) -> float:
        """
        Calculate convergence score based on Q-value stability.
        
        Args:
            qtable: Q-table array
            window_size: Window size for stability calculation
            
        Returns:
            Convergence score between 0 and 1
        """
        if qtable is None or qtable.size == 0:
            return 0.0

        # Calculate coefficient of variation for each state-action pair
        non_zero_mask = qtable != 0
        if not np.any(non_zero_mask):
            return 0.0

        # Use the range of Q-values as a proxy for convergence
        # Well-converged tables have consistent value differences
        q_values = qtable[non_zero_mask]

        if len(q_values) < 2:
            return 0.0

        # Calculate stability based on value distribution
        q_std = np.std(q_values)
        q_mean = np.abs(np.mean(q_values))

        if q_mean == 0:
            return 0.0

        # Coefficient of variation (lower is more converged)
        cv = q_std / q_mean

        # Convert to convergence score (0-1, higher is better)
        convergence_score = np.exp(-cv / 2)  # Exponential decay

        return min(max(convergence_score, 0.0), 1.0)

    def calculate_policy_entropy(self, qtable: np.ndarray) -> float:
        """
        Calculate policy entropy (measure of action selection diversity).
        
        Args:
            qtable: Q-table array
            
        Returns:
            Policy entropy
        """
        if qtable is None or qtable.size == 0:
            return 0.0

        # For each state, find the action distribution
        total_entropy = 0.0
        valid_states = 0

        for state_idx in range(qtable.shape[0]):
            state_q_values = qtable[state_idx]

            # Skip states with all zero Q-values
            if np.all(state_q_values == 0):
                continue

            # Convert Q-values to action probabilities using softmax
            exp_q = np.exp(state_q_values - np.max(state_q_values))
            probabilities = exp_q / np.sum(exp_q)

            # Calculate entropy for this state
            state_entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
            total_entropy += state_entropy
            valid_states += 1

        return total_entropy / max(valid_states, 1)

    def calculate_exploration_coverage(self, qtable: np.ndarray) -> Tuple[float, int, int]:
        """
        Calculate exploration coverage metrics.
        
        Args:
            qtable: Q-table array
            
        Returns:
            Tuple of (coverage_ratio, visited_states, unvisited_states)
        """
        if qtable is None or qtable.size == 0:
            return 0.0, 0, 0

        # Count states that have been visited (non-zero Q-values)
        visited_states = 0
        total_states = qtable.shape[0]

        for state_idx in range(total_states):
            if np.any(qtable[state_idx] != 0):
                visited_states += 1

        unvisited_states = total_states - visited_states
        coverage_ratio = visited_states / total_states if total_states > 0 else 0.0

        return coverage_ratio, visited_states, unvisited_states

    def determine_convergence_status(self, metrics: QTableMetrics) -> ConvergenceStatus:
        """
        Determine convergence status based on metrics.
        
        Args:
            metrics: Q-table metrics
            
        Returns:
            Convergence status
        """
        convergence_score = metrics.convergence_score
        stability = metrics.stability_measure

        if convergence_score >= self.convergence_thresholds['convergence_threshold'] and \
           stability >= self.convergence_thresholds['stability_threshold']:
            return ConvergenceStatus.CONVERGED
        elif convergence_score >= 0.7 and stability >= 0.8:
            return ConvergenceStatus.CONVERGING
        elif convergence_score >= self.convergence_thresholds['divergence_threshold']:
            return ConvergenceStatus.UNSTABLE
        elif convergence_score < self.convergence_thresholds['divergence_threshold']:
            return ConvergenceStatus.DIVERGING
        else:
            return ConvergenceStatus.UNKNOWN

    def analyze_qtable(self, file_path: Union[str, Path]) -> Optional[QTableMetrics]:
        """
        Perform comprehensive analysis of a Q-table.
        
        Args:
            file_path: Path to Q-table pickle file
            
        Returns:
            Q-table metrics or None if analysis failed
        """
        file_path = Path(file_path)

        # Check cache first
        cache_key = str(file_path)
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]

        # Load Q-table
        qtable = self.load_qtable(file_path)
        if qtable is None:
            return None

        try:
            # Basic structure metrics
            shape = qtable.shape
            total_states = shape[0] if len(shape) >= 1 else 0
            total_actions = shape[1] if len(shape) >= 2 else 1
            non_zero_values = np.count_nonzero(qtable)
            sparsity = 1.0 - (non_zero_values / qtable.size) if qtable.size > 0 else 1.0

            # Value statistics
            mean_q_value = np.mean(qtable)
            std_q_value = np.std(qtable)
            min_q_value = np.min(qtable)
            max_q_value = np.max(qtable)
            q_value_range = max_q_value - min_q_value

            # Convergence metrics
            convergence_score = self.calculate_convergence_score(qtable)
            stability_measure = 1.0 - (std_q_value / (abs(mean_q_value) + 1e-10))
            stability_measure = max(0.0, min(1.0, stability_measure))

            # Policy metrics
            policy_entropy = self.calculate_policy_entropy(qtable)
            action_diversity = policy_entropy / np.log(total_actions) if total_actions > 1 else 0.0
            state_value_variance = np.var(np.max(qtable, axis=1)) if len(shape) >= 2 else 0.0

            # Exploration metrics
            exploration_coverage, visited_states, unvisited_states = self.calculate_exploration_coverage(qtable)

            # Create metrics object
            metrics = QTableMetrics(
                shape=shape,
                total_states=total_states,
                total_actions=total_actions,
                non_zero_values=int(non_zero_values),
                sparsity=float(sparsity),
                mean_q_value=mean_q_value,
                std_q_value=std_q_value,
                min_q_value=min_q_value,
                max_q_value=max_q_value,
                q_value_range=q_value_range,
                convergence_score=convergence_score,
                stability_measure=stability_measure,
                convergence_status=ConvergenceStatus.UNKNOWN,  # Will be set later
                policy_entropy=policy_entropy,
                action_diversity=action_diversity,
                state_value_variance=state_value_variance,
                exploration_coverage=exploration_coverage,
                visited_states=visited_states,
                unvisited_states=unvisited_states,
                analysis_timestamp=datetime.now().isoformat(),
                file_path=str(file_path)
            )

            # Determine convergence status after metrics are created
            metrics.convergence_status = self.determine_convergence_status(metrics)

            # Cache result
            self.analysis_cache[cache_key] = metrics

            return metrics

        except Exception as e:
            logger.error(f"Error analyzing Q-table {file_path}: {e}")
            return None

    def compare_qtables(self, file_path1: Union[str, Path], file_path2: Union[str, Path]) -> Optional[QTableComparison]:
        """
        Compare two Q-tables and analyze evolution.
        
        Args:
            file_path1: Path to first Q-table
            file_path2: Path to second Q-table
            
        Returns:
            Comparison results or None if comparison failed
        """
        # Analyze both tables
        metrics1 = self.analyze_qtable(file_path1)
        metrics2 = self.analyze_qtable(file_path2)

        if metrics1 is None or metrics2 is None:
            return None

        # Load Q-tables for direct comparison
        qtable1 = self.load_qtable(file_path1)
        qtable2 = self.load_qtable(file_path2)

        if qtable1 is None or qtable2 is None:
            return None

        try:
            # Ensure same shape for comparison
            if qtable1.shape != qtable2.shape:
                logger.warning(f"Q-tables have different shapes: {qtable1.shape} vs {qtable2.shape}")
                return None

            # Calculate value differences
            value_diff = qtable2 - qtable1
            value_difference_mean = np.mean(np.abs(value_diff))
            value_difference_std = np.std(value_diff)

            # Calculate policy agreement (how often both tables choose same action)
            policy_agreement = 0.0
            if qtable1.shape[0] > 0:
                best_actions1 = np.argmax(qtable1, axis=1)
                best_actions2 = np.argmax(qtable2, axis=1)
                policy_agreement = np.mean(best_actions1 == best_actions2)

            # Calculate convergence improvement
            convergence_improvement = metrics2.convergence_score - metrics1.convergence_score

            # Calculate learning progress (improvement in Q-values)
            learning_progress = np.mean(qtable2) - np.mean(qtable1)

            # Calculate stability change
            stability_change = metrics2.stability_measure - metrics1.stability_measure

            # Calculate exploration change
            exploration_change = metrics2.exploration_coverage - metrics1.exploration_coverage

            return QTableComparison(
                table1_metrics=metrics1,
                table2_metrics=metrics2,
                value_difference_mean=value_difference_mean,
                value_difference_std=value_difference_std,
                policy_agreement=policy_agreement,
                convergence_improvement=convergence_improvement,
                learning_progress=learning_progress,
                stability_change=stability_change,
                exploration_change=exploration_change,
                comparison_timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            logger.error(f"Error comparing Q-tables: {e}")
            return None

    def get_available_qtables(self, pattern: str = "*.pkl") -> List[Path]:
        """
        Get list of available Q-table files.
        
        Args:
            pattern: File pattern to match
            
        Returns:
            List of Q-table file paths
        """
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            return []

        qtable_files = list(self.models_dir.glob(pattern))
        qtable_files.sort(key=lambda x: x.stat().st_mtime)  # Sort by modification time

        return qtable_files

    def batch_analyze_qtables(self, file_paths: Optional[List[Path]] = None) -> Dict[str, QTableMetrics]:
        """
        Analyze multiple Q-tables in batch.
        
        Args:
            file_paths: List of file paths to analyze (if None, analyze all available)
            
        Returns:
            Dictionary mapping file paths to metrics
        """
        if file_paths is None:
            file_paths = self.get_available_qtables()

        results = {}

        for file_path in file_paths:
            logger.info(f"Analyzing {file_path.name}")
            metrics = self.analyze_qtable(file_path)
            if metrics is not None:
                results[str(file_path)] = metrics

        return results

    def export_analysis_results(self, results: Dict[str, QTableMetrics], output_file: str):
        """
        Export analysis results to CSV file.
        
        Args:
            results: Analysis results dictionary
            output_file: Output CSV file path
        """
        if not results:
            logger.warning("No results to export")
            return

        # Convert metrics to DataFrame
        rows = []
        for file_path, metrics in results.items():
            row = {
                'file_path': file_path,
                'file_name': Path(file_path).name,
                'total_states': metrics.total_states,
                'total_actions': metrics.total_actions,
                'sparsity': metrics.sparsity,
                'mean_q_value': metrics.mean_q_value,
                'convergence_score': metrics.convergence_score,
                'stability_measure': metrics.stability_measure,
                'convergence_status': metrics.convergence_status.value,
                'policy_entropy': metrics.policy_entropy,
                'action_diversity': metrics.action_diversity,
                'exploration_coverage': metrics.exploration_coverage,
                'visited_states': metrics.visited_states,
                'analysis_timestamp': metrics.analysis_timestamp
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        logger.info(f"Analysis results exported to {output_file}")


# Global analyzer instance
QTABLE_ANALYZER = QTableAnalyzer()
