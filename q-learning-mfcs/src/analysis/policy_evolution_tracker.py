"""
Policy Evolution Tracking System

This module provides comprehensive analysis capabilities for tracking Q-learning
policy development over training episodes, including action frequency analysis,
stability metrics, and convergence detection.

User Story 1.2.2: Policy Evolution Tracking
Created: 2025-07-31
Last Modified: 2025-07-31
"""

import logging
import pickle
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolicyStability(Enum):
    """Policy stability levels."""
    STABLE = "stable"
    CONVERGING = "converging"
    UNSTABLE = "unstable"
    OSCILLATING = "oscillating"
    UNKNOWN = "unknown"


@dataclass
class PolicySnapshot:
    """Single policy snapshot at a specific training episode."""

    episode: int
    policy: np.ndarray  # Best action for each state
    q_table: np.ndarray | None
    action_frequencies: dict[int, int]  # Action -> frequency count
    policy_entropy: float
    state_coverage: float  # Percentage of states with valid actions
    performance_reward: float | None  # Episode reward if available
    timestamp: str


@dataclass
class PolicyEvolutionMetrics:
    """Comprehensive metrics for policy evolution analysis."""

    # Evolution tracking
    total_episodes: int
    snapshots_count: int
    episode_range: tuple[int, int]

    # Policy stability metrics
    policy_changes: list[int]  # Number of policy changes per episode
    stability_score: float  # 0-1, higher = more stable
    stability_status: PolicyStability
    convergence_episode: int | None  # Episode where policy converged

    # Action analysis
    action_diversity_evolution: list[float]  # Action diversity over time
    dominant_actions: dict[int, float]  # Action -> percentage usage
    action_preference_changes: int  # Total action preference changes

    # Learning progress
    learning_velocity: list[float]  # Rate of policy change over episodes
    performance_trend: list[float]  # Performance/reward trend if available
    exploration_decay: list[float]  # Exploration rate decay over time

    # Metadata
    analysis_timestamp: str


class PolicyEvolutionTracker:
    """Comprehensive policy evolution tracking system."""

    def __init__(self, models_directory: str = "q_learning_models"):
        """
        Initialize policy evolution tracker.

        Args:
            models_directory: Directory containing Q-table snapshots
        """
        self.models_dir = Path(models_directory)
        self.policy_snapshots: list[PolicySnapshot] = []
        self.evolution_cache: dict[str, PolicyEvolutionMetrics] = {}

        # Analysis parameters
        self.stability_threshold = 0.95  # Threshold for stable policy
        self.convergence_window = 10     # Episodes to check for convergence
        self.min_snapshots = 5           # Minimum snapshots for analysis

    def load_policy_snapshots_from_files(
        self,
        file_pattern: str = "*qtable*.pkl",
        max_snapshots: int | None = None
    ) -> int:
        """
        Load policy snapshots from Q-table files in chronological order.

        Args:
            file_pattern: File pattern to match Q-table files
            max_snapshots: Maximum number of snapshots to load

        Returns:
            Number of snapshots loaded
        """
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            return 0

        # Find Q-table files
        qtable_files = list(self.models_dir.glob(file_pattern))
        qtable_files.sort(key=lambda x: x.stat().st_mtime)  # Sort by modification time

        if max_snapshots:
            qtable_files = qtable_files[:max_snapshots]

        logger.info(f"Loading {len(qtable_files)} Q-table snapshots...")

        loaded_count = 0
        for i, file_path in enumerate(qtable_files):
            snapshot = self._create_policy_snapshot_from_file(file_path, episode=i)
            if snapshot:
                self.policy_snapshots.append(snapshot)
                loaded_count += 1

        logger.info(f"Successfully loaded {loaded_count} policy snapshots")
        return loaded_count

    def _create_policy_snapshot_from_file(
        self,
        file_path: Path,
        episode: int
    ) -> PolicySnapshot | None:
        """Create policy snapshot from Q-table file."""
        try:
            # Load Q-table
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            # Extract Q-table from different file formats
            if isinstance(data, np.ndarray):
                q_table = data
            elif isinstance(data, dict) and 'q_table' in data:
                q_table = data['q_table']
            elif isinstance(data, dict) and 'Q' in data:
                q_table = data['Q']
            else:
                logger.warning(f"Unclear Q-table format in {file_path}")
                return None

            if q_table is None or q_table.size == 0:
                return None

            # Extract policy (best action for each state)
            policy = np.argmax(q_table, axis=1)

            # Calculate action frequencies
            unique_actions, counts = np.unique(policy, return_counts=True)
            action_frequencies = dict(zip(unique_actions.astype(int), counts.astype(int), strict=False))

            # Calculate policy entropy
            policy_entropy = self._calculate_policy_entropy(q_table)

            # Calculate state coverage
            state_coverage = np.mean(np.any(q_table != 0, axis=1))

            # Try to extract performance reward from filename or metadata
            performance_reward = self._extract_performance_from_file(file_path, data)

            return PolicySnapshot(
                episode=episode,
                policy=policy,
                q_table=q_table,
                action_frequencies=action_frequencies,
                policy_entropy=policy_entropy,
                state_coverage=state_coverage,
                performance_reward=performance_reward,
                timestamp=datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            )

        except Exception as e:
            logger.error(f"Error creating policy snapshot from {file_path}: {e}")
            return None

    def _calculate_policy_entropy(self, q_table: np.ndarray) -> float:
        """Calculate policy entropy (measure of action selection diversity)."""
        if q_table is None or q_table.size == 0:
            return 0.0

        total_entropy = 0.0
        valid_states = 0

        for state_idx in range(q_table.shape[0]):
            state_q_values = q_table[state_idx]

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

    def _extract_performance_from_file(
        self,
        file_path: Path,
        data: Any
    ) -> float | None:
        """Extract performance/reward information from file or metadata."""
        # Try to extract from dictionary data
        if isinstance(data, dict):
            for key in ['reward', 'performance', 'episode_reward', 'total_reward']:
                if key in data:
                    return float(data[key])

        # Try to extract from filename patterns
        filename = file_path.name

        # Look for patterns like 'reward_123.45' or 'perf_0.67'
        import re
        patterns = [
            r'reward[_-]?([\d.]+)',
            r'perf[_-]?([\d.]+)',
            r'score[_-]?([\d.]+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        return None

    def analyze_policy_evolution(self) -> PolicyEvolutionMetrics | None:
        """
        Analyze policy evolution across all loaded snapshots.

        Returns:
            Comprehensive policy evolution metrics
        """
        if len(self.policy_snapshots) < self.min_snapshots:
            logger.warning(f"Need at least {self.min_snapshots} snapshots for analysis, got {len(self.policy_snapshots)}")
            return None

        # Sort snapshots by episode
        snapshots = sorted(self.policy_snapshots, key=lambda x: x.episode)

        # Calculate policy changes between consecutive episodes
        policy_changes = []
        for i in range(1, len(snapshots)):
            prev_policy = snapshots[i-1].policy
            curr_policy = snapshots[i].policy

            # Count number of states where policy changed
            changes = np.sum(prev_policy != curr_policy)
            policy_changes.append(changes)

        # Calculate stability metrics
        stability_score = self._calculate_stability_score(policy_changes, snapshots)
        stability_status = self._determine_stability_status(stability_score, policy_changes)
        convergence_episode = self._detect_convergence_episode(policy_changes)

        # Analyze action diversity evolution
        action_diversity_evolution = [s.policy_entropy for s in snapshots]

        # Calculate dominant actions across all episodes
        all_actions = []
        for snapshot in snapshots:
            all_actions.extend(snapshot.policy)

        unique_actions, counts = np.unique(all_actions, return_counts=True)
        total_actions = len(all_actions)
        dominant_actions = {
            int(action): float(count / total_actions)
            for action, count in zip(unique_actions, counts, strict=False)
        }

        # Count action preference changes
        action_preference_changes = self._count_action_preference_changes(snapshots)

        # Calculate learning velocity (rate of policy change)
        learning_velocity = self._calculate_learning_velocity(policy_changes)

        # Extract performance trend if available
        performance_trend = [
            s.performance_reward for s in snapshots
            if s.performance_reward is not None
        ]

        # Calculate exploration decay (decreasing policy entropy over time)
        exploration_decay = self._calculate_exploration_decay(action_diversity_evolution)

        return PolicyEvolutionMetrics(
            total_episodes=len(snapshots),
            snapshots_count=len(snapshots),
            episode_range=(snapshots[0].episode, snapshots[-1].episode),
            policy_changes=policy_changes,
            stability_score=stability_score,
            stability_status=stability_status,
            convergence_episode=convergence_episode,
            action_diversity_evolution=action_diversity_evolution,
            dominant_actions=dominant_actions,
            action_preference_changes=action_preference_changes,
            learning_velocity=learning_velocity,
            performance_trend=performance_trend,
            exploration_decay=exploration_decay,
            analysis_timestamp=datetime.now().isoformat()
        )

    def _calculate_stability_score(
        self,
        policy_changes: list[int],
        snapshots: list[PolicySnapshot]
    ) -> float:
        """Calculate policy stability score (0-1, higher = more stable)."""
        if not policy_changes or not snapshots:
            return 0.0

        # Get total number of states from first snapshot
        total_states = len(snapshots[0].policy)

        # Calculate average percentage of states that remain unchanged
        avg_unchanged = 1.0 - (np.mean(policy_changes) / total_states)

        # Apply sigmoid to smooth the score
        stability_score = 2 / (1 + np.exp(-5 * avg_unchanged)) - 1

        return max(0.0, min(1.0, stability_score))

    def _determine_stability_status(
        self,
        stability_score: float,
        policy_changes: list[int]
    ) -> PolicyStability:
        """Determine policy stability status based on metrics."""
        if stability_score >= self.stability_threshold:
            return PolicyStability.STABLE
        elif stability_score >= 0.8:
            return PolicyStability.CONVERGING
        elif stability_score >= 0.5:
            # Check for oscillations
            if len(policy_changes) >= 4:
                recent_changes = policy_changes[-4:]
                if self._detect_oscillation(recent_changes):
                    return PolicyStability.OSCILLATING
            return PolicyStability.UNSTABLE
        else:
            return PolicyStability.UNSTABLE

    def _detect_oscillation(self, changes: list[int]) -> bool:
        """Detect if policy changes show oscillating pattern."""
        if len(changes) < 4:
            return False

        # Simple oscillation detection: alternating high-low pattern
        high_threshold = np.mean(changes) + np.std(changes)
        low_threshold = np.mean(changes) - np.std(changes)

        pattern = []
        for change in changes:
            if change >= high_threshold:
                pattern.append('H')
            elif change <= low_threshold:
                pattern.append('L')
            else:
                pattern.append('M')

        # Check for alternating patterns
        pattern_str = ''.join(pattern)
        oscillation_patterns = ['HLHL', 'LHLH', 'HMHM', 'MHMH']

        return any(osc_pattern in pattern_str for osc_pattern in oscillation_patterns)

    def _detect_convergence_episode(self, policy_changes: list[int]) -> int | None:
        """Detect episode where policy converged (stable for convergence_window episodes)."""
        if len(policy_changes) < self.convergence_window:
            return None

        # Look for window of episodes with minimal changes
        convergence_threshold = max(1, np.mean(policy_changes) * 0.1)  # 10% of average changes

        for i in range(len(policy_changes) - self.convergence_window + 1):
            window = policy_changes[i:i + self.convergence_window]
            if all(change <= convergence_threshold for change in window):
                return i + 1  # Return episode number (1-indexed)

        return None

    def _count_action_preference_changes(self, snapshots: list[PolicySnapshot]) -> int:
        """Count significant changes in action preferences over time."""
        if len(snapshots) < 2:
            return 0

        preference_changes = 0
        prev_dominant = self._get_dominant_action(snapshots[0])

        for i in range(1, len(snapshots)):
            curr_dominant = self._get_dominant_action(snapshots[i])
            if curr_dominant != prev_dominant:
                preference_changes += 1
            prev_dominant = curr_dominant

        return preference_changes

    def _get_dominant_action(self, snapshot: PolicySnapshot) -> int:
        """Get the most frequently used action in a policy snapshot."""
        if not snapshot.action_frequencies:
            return 0

        return max(snapshot.action_frequencies.items(), key=lambda x: x[1])[0]

    def _calculate_learning_velocity(self, policy_changes: list[int]) -> list[float]:
        """Calculate learning velocity (rate of policy change) over time."""
        if len(policy_changes) < 2:
            return []

        # Calculate smoothed derivative of policy changes
        velocity = []
        window_size = min(3, len(policy_changes))

        for i in range(len(policy_changes)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(policy_changes), i + window_size // 2 + 1)

            window_changes = policy_changes[start_idx:end_idx]
            # Velocity as rate of change (negative means decreasing changes)
            if len(window_changes) > 1:
                velocity_val = (window_changes[-1] - window_changes[0]) / len(window_changes)
            else:
                velocity_val = 0.0

            velocity.append(velocity_val)

        return velocity

    def _calculate_exploration_decay(self, entropy_evolution: list[float]) -> list[float]:
        """Calculate exploration decay rate over time."""
        if len(entropy_evolution) < 2:
            return []

        # Calculate rate of entropy decrease (positive values indicate decreasing exploration)
        decay_rates = []
        for i in range(1, len(entropy_evolution)):
            # Decay rate: (previous - current) / previous
            prev_entropy = entropy_evolution[i-1]
            curr_entropy = entropy_evolution[i]

            if prev_entropy > 0:
                decay_rate = (prev_entropy - curr_entropy) / prev_entropy
            else:
                decay_rate = 0.0

            decay_rates.append(decay_rate)

        return decay_rates

    def get_action_frequency_matrix(self) -> pd.DataFrame | None:
        """
        Get action frequency matrix showing action usage across episodes.

        Returns:
            DataFrame with episodes as rows and actions as columns
        """
        if not self.policy_snapshots:
            return None

        # Collect all unique actions
        all_actions = set()
        for snapshot in self.policy_snapshots:
            all_actions.update(snapshot.action_frequencies.keys())

        all_actions = sorted(list(all_actions))

        # Create frequency matrix
        frequency_data = []
        for snapshot in self.policy_snapshots:
            row = {f'Action_{action}': snapshot.action_frequencies.get(action, 0)
                   for action in all_actions}
            row['Episode'] = snapshot.episode
            row['Timestamp'] = snapshot.timestamp
            frequency_data.append(row)

        return pd.DataFrame(frequency_data)

    def get_policy_comparison_matrix(self, reference_episode: int) -> np.ndarray | None:
        """
        Get policy comparison matrix showing similarity to reference episode.

        Args:
            reference_episode: Episode to use as reference

        Returns:
            Similarity matrix (episodes x states)
        """
        if not self.policy_snapshots:
            return None

        # Find reference snapshot
        reference_snapshot = None
        for snapshot in self.policy_snapshots:
            if snapshot.episode == reference_episode:
                reference_snapshot = snapshot
                break

        if reference_snapshot is None:
            logger.warning(f"Reference episode {reference_episode} not found")
            return None

        reference_policy = reference_snapshot.policy

        # Calculate similarity for each episode
        similarity_matrix = []
        for snapshot in self.policy_snapshots:
            # State-wise policy agreement (1 = same action, 0 = different action)
            agreement = (snapshot.policy == reference_policy).astype(float)
            similarity_matrix.append(agreement)

        return np.array(similarity_matrix)

    def export_evolution_analysis(
        self,
        metrics: PolicyEvolutionMetrics,
        output_file: str
    ) -> None:
        """
        Export policy evolution analysis to CSV file.

        Args:
            metrics: Policy evolution metrics
            output_file: Output file path
        """
        # Prepare data for export
        analysis_data = {
            'metric': [],
            'value': [],
            'description': []
        }

        # Basic metrics
        basic_metrics = [
            ('total_episodes', metrics.total_episodes, 'Total number of training episodes'),
            ('snapshots_count', metrics.snapshots_count, 'Number of policy snapshots analyzed'),
            ('stability_score', metrics.stability_score, 'Policy stability score (0-1)'),
            ('stability_status', metrics.stability_status.value, 'Policy stability status'),
            ('convergence_episode', metrics.convergence_episode or 'None', 'Episode where policy converged'),
            ('action_preference_changes', metrics.action_preference_changes, 'Number of dominant action changes')
        ]

        for metric, value, description in basic_metrics:
            analysis_data['metric'].append(metric)
            analysis_data['value'].append(str(value))
            analysis_data['description'].append(description)

        # Dominant actions
        for action, percentage in metrics.dominant_actions.items():
            analysis_data['metric'].append(f'dominant_action_{action}')
            analysis_data['value'].append(f'{percentage:.3f}')
            analysis_data['description'].append(f'Usage percentage for action {action}')

        # Create DataFrame and export
        df = pd.DataFrame(analysis_data)
        df.to_csv(output_file, index=False)
        logger.info(f"Policy evolution analysis exported to {output_file}")


# Global instance for easy access
POLICY_EVOLUTION_TRACKER = PolicyEvolutionTracker()
