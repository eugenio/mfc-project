"""High-performance Q-learning MFC controller demo using Mojo accelerated backend.

This demo shows how to use the Mojo-accelerated Q-learning algorithm
for microbial fuel cell control optimization.
"""

import matplotlib as mpl
import numpy as np

mpl.use("Agg")  # Use non-interactive backend
import time

import matplotlib.pyplot as plt
from odes import MFCModel


def run_mojo_qlearning_demo() -> None:
    """Run the Mojo-accelerated Q-learning demo."""
    # First, let's demonstrate the basic MFC model
    mfc_model = MFCModel()

    # Test basic MFC simulation
    initial_state = [1.0, 0.05, 1e-4, 0.1, 0.25, 1e-7, 0.05, 0.01, -0.01]
    current_density = 1.0

    try:
        mfc_model.mfc_odes(0, initial_state, current_density)
    except Exception:
        return

    # Since we can't directly import the compiled Mojo Q-learning module in this demo,
    # let's create a Python-based demonstration that shows the equivalent functionality
    # and structure that would be accelerated by Mojo

    class MFCQLearningDemo:
        """Python demonstration of the Mojo Q-learning controller structure."""

        def __init__(self) -> None:
            self.n_state_bins = 10
            self.n_action_bins = 10
            self.n_states = self.n_state_bins**6  # 6 state variables
            self.n_actions = self.n_action_bins**2  # 2 action variables

            # Initialize Q-table (this would be Tensor[DType] in Mojo)
            self.q_table = np.random.uniform(
                -0.01,
                0.01,
                (self.n_states, self.n_actions),
            )

            # State and action ranges
            self.state_ranges = np.array(
                [
                    [0.0, 3.0],  # C_AC
                    [0.0, 1.0],  # X
                    [0.0, 0.5],  # C_O2
                    [-0.5, 0.5],  # eta_a
                    [-0.5, 0.5],  # eta_c
                    [0.0, 10.0],  # power
                ],
            )

            self.action_ranges = np.array(
                [[0.1, 5.0], [0.5, 2.0]],  # current_density  # flow_rate_ratio
            )

            # Training parameters
            self.learning_rate = 0.1
            self.discount_factor = 0.9
            self.epsilon = 0.3
            self.epsilon_decay = 0.995
            self.epsilon_min = 0.01

        def discretize_state(self, state_values):
            """Discretize continuous state (vectorized in Mojo)."""
            state_indices = []

            for i, val in enumerate(state_values):
                min_val, max_val = self.state_ranges[i]
                clipped_val = np.clip(val, min_val, max_val)
                normalized = (clipped_val - min_val) / (max_val - min_val)
                bin_idx = int(normalized * (self.n_state_bins - 1))
                bin_idx = np.clip(bin_idx, 0, self.n_state_bins - 1)
                state_indices.append(bin_idx)

            # Convert to single index
            state_index = 0
            multiplier = 1
            for idx in reversed(state_indices):
                state_index += idx * multiplier
                multiplier *= self.n_state_bins

            return state_index

        def choose_action(self, state_index):
            """Epsilon-greedy action selection."""
            if np.random.random() < self.epsilon:
                return np.random.randint(0, self.n_actions)
            return np.argmax(self.q_table[state_index])

        def action_from_index(self, action_index):
            """Convert action index to continuous values."""
            flow_bin = action_index % self.n_action_bins
            current_bin = action_index // self.n_action_bins

            current_min, current_max = self.action_ranges[0]
            flow_min, flow_max = self.action_ranges[1]

            current_density = current_min + (
                current_max - current_min
            ) * current_bin / (self.n_action_bins - 1)
            flow_rate_ratio = flow_min + (flow_max - flow_min) * flow_bin / (
                self.n_action_bins - 1
            )

            return current_density, flow_rate_ratio

        def simulate_step(self, state, action):
            """Simulate one MFC step (would use Mojo MFC model)."""
            current_density, flow_rate_ratio = action

            # Simplified MFC dynamics for demo
            # In real implementation, this would call the Mojo MFC model
            next_state = state.copy()

            # Simple dynamics approximation
            dt = 1.0
            next_state[0] = max(
                0.0,
                state[0] - 0.01 * current_density * dt,
            )  # C_AC consumption
            next_state[3] = min(
                1.0,
                state[3] + 0.005 * current_density * dt,
            )  # X growth
            next_state[7] = state[7] + 0.001 * current_density * dt  # eta_a
            next_state[8] = state[8] - 0.001 * current_density * dt  # eta_c

            # Calculate power density
            power_density = current_density * (next_state[7] - next_state[8])

            return next_state, power_density

        def calculate_reward(self, state, power_density, action):
            """Calculate reward (vectorized in Mojo)."""
            current_density, flow_rate_ratio = action

            # Multi-objective reward
            power_reward = power_density / 10.0

            stability_penalty = 0.0
            if abs(state[7]) > 0.3 or abs(state[8]) > 0.3:
                stability_penalty = -0.5

            cod_removal_reward = 0.2 if state[0] < 1.0 else 0.0

            action_penalty = (
                -0.1 if current_density > 4.0 or flow_rate_ratio > 1.8 else 0.0
            )

            return (
                power_reward + stability_penalty + cod_removal_reward + action_penalty
            )

        def train_episode(self):
            """Train one episode."""
            # Initial state
            state = np.array([1.0, 0.05, 1e-4, 0.1, 0.25, 1e-7, 0.05, 0.01, -0.01])

            episode_reward = 0.0
            episode_power = 0.0
            steps = 0

            for _step in range(50):  # Max steps per episode
                # Extract state features for discretization
                state_features = [
                    state[0],
                    state[3],
                    state[4],
                    state[7],
                    state[8],
                    0.1,
                ]  # Last is power estimate
                state_index = self.discretize_state(state_features)

                # Choose and execute action
                action_index = self.choose_action(state_index)
                action = self.action_from_index(action_index)

                # Simulate step
                next_state, power_density = self.simulate_step(state, action)

                # Calculate reward
                reward = self.calculate_reward(next_state, power_density, action)

                # Update Q-table
                next_state_features = [
                    next_state[0],
                    next_state[3],
                    next_state[4],
                    next_state[7],
                    next_state[8],
                    power_density,
                ]
                next_state_index = self.discretize_state(next_state_features)

                old_value = self.q_table[state_index, action_index]
                next_max = np.max(self.q_table[next_state_index])
                new_value = old_value + self.learning_rate * (
                    reward + self.discount_factor * next_max - old_value
                )
                self.q_table[state_index, action_index] = new_value

                # Update for next step
                state = next_state
                episode_reward += reward
                episode_power += power_density
                steps += 1

                if power_density < -0.5:  # Termination condition
                    break

            return episode_reward, episode_power / steps if steps > 0 else 0.0

        def train(self, n_episodes=200):
            """Train the Q-learning controller."""
            rewards = []
            powers = []

            start_time = time.time()

            for episode in range(n_episodes):
                episode_reward, episode_power = self.train_episode()

                rewards.append(episode_reward)
                powers.append(episode_power)

                # Decay epsilon
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

                if episode % 50 == 0:
                    np.mean(rewards[-50:])
                    np.mean(powers[-50:])

            time.time() - start_time

            return rewards, powers

    # Run the demo
    demo_controller = MFCQLearningDemo()

    # Train the controller
    rewards, powers = demo_controller.train(n_episodes=200)

    # Plot results

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot rewards
    ax1.plot(rewards, alpha=0.7, label="Episode Reward")
    window_size = 20
    if len(rewards) > window_size:
        moving_avg = np.convolve(
            rewards,
            np.ones(window_size) / window_size,
            mode="valid",
        )
        ax1.plot(
            range(window_size - 1, len(rewards)),
            moving_avg,
            "r-",
            linewidth=2,
            label="Moving Average",
        )
    ax1.set_title("Q-Learning Training Progress - Rewards")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Cumulative Reward")
    ax1.legend()
    ax1.grid(True)

    # Plot power density
    ax2.plot(powers, alpha=0.7, label="Episode Power")
    if len(powers) > window_size:
        moving_avg = np.convolve(
            powers,
            np.ones(window_size) / window_size,
            mode="valid",
        )
        ax2.plot(
            range(window_size - 1, len(powers)),
            moving_avg,
            "r-",
            linewidth=2,
            label="Moving Average",
        )
    ax2.set_title("Q-Learning Training Progress - Power Density")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Average Power Density (W/m²)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("mfc_qlearning_training.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    try:
        run_mojo_qlearning_demo()
    except KeyboardInterrupt:
        pass
    except Exception:
        import traceback

        traceback.print_exc()
