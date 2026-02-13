from tensor import Tensor, TensorSpec, TensorShape
from utils.index import Index
from algorithm import vectorize, parallelize
from math import exp, log, sqrt, abs, min, max
from random import random_float64, random_si64, seed
from memory import memset_zero
from builtin import print
from python import Python
import time

alias DType = DType.float64
alias simd_width = simdwidthof[DType]()


@value
struct MFCQLearningConfig:
    """Configuration for Q-learning MFC controller"""

    var learning_rate: Float64
    var discount_factor: Float64
    var epsilon: Float64
    var epsilon_decay: Float64
    var epsilon_min: Float64
    var n_state_bins: Int
    var n_action_bins: Int
    var max_episodes: Int
    var max_steps_per_episode: Int

    fn __init__(out self):
        self.learning_rate = 0.0987
        self.discount_factor = 0.9517
        self.epsilon = 0.3702
        self.epsilon_decay = 0.9978
        self.epsilon_min = 0.1020
        self.n_state_bins = 10
        self.n_action_bins = 10
        self.max_episodes = 1000
        self.max_steps_per_episode = 100


@value
struct MFCState:
    """Represents MFC state variables"""

    var C_AC: Float64  # Acetate concentration
    var C_CO2: Float64  # CO2 concentration
    var C_H: Float64  # H+ concentration
    var X: Float64  # Biomass concentration
    var C_O2: Float64  # O2 concentration
    var C_OH: Float64  # OH- concentration
    var C_M: Float64  # Mediator concentration
    var eta_a: Float64  # Anodic overpotential
    var eta_c: Float64  # Cathodic overpotential

    fn __init__(out self):
        self.C_AC = 1.0
        self.C_CO2 = 0.05
        self.C_H = 1e-4
        self.X = 0.1
        self.C_O2 = 0.25
        self.C_OH = 1e-7
        self.C_M = 0.05
        self.eta_a = 0.01
        self.eta_c = -0.01


@value
struct MFCAction:
    """Represents MFC control actions"""

    var current_density: Float64
    var flow_rate_ratio: Float64

    fn __init__(out self):
        self.current_density = 1.0
        self.flow_rate_ratio = 1.0


struct MFCQLearningController:
    """High-performance Q-learning controller for MFC using Mojo tensors"""

    var config: MFCQLearningConfig
    var q_table: Tensor[DType]
    var reward_history: Tensor[DType]
    var power_history: Tensor[DType]
    var state_ranges: Tensor[DType]  # [variable, min, max]
    var action_ranges: Tensor[DType]  # [variable, min, max]
    var current_epsilon: Float64
    var episode_count: Int

    fn __init__(out self, config: MFCQLearningConfig):
        self.config = config
        self.current_epsilon = config.epsilon
        self.episode_count = 0

        # Initialize Q-table: [state_index, action_index]
        var n_states = self.config.n_state_bins**6  # 6 state variables
        var n_actions = self.config.n_action_bins**2  # 2 action variables

        self.q_table = Tensor[DType](TensorShape(n_states, n_actions))
        self.reward_history = Tensor[DType](TensorShape(config.max_episodes))
        self.power_history = Tensor[DType](TensorShape(config.max_episodes))

        # Initialize state ranges [variable_index, min, max]
        self.state_ranges = Tensor[DType](TensorShape(6, 2))
        self.state_ranges[0, 0] = 0.0
        self.state_ranges[0, 1] = 3.0  # C_AC
        self.state_ranges[1, 0] = 0.0
        self.state_ranges[1, 1] = 1.0  # X
        self.state_ranges[2, 0] = 0.0
        self.state_ranges[2, 1] = 0.5  # C_O2
        self.state_ranges[3, 0] = -0.5
        self.state_ranges[3, 1] = 0.5  # eta_a
        self.state_ranges[4, 0] = -0.5
        self.state_ranges[4, 1] = 0.5  # eta_c
        self.state_ranges[5, 0] = 0.0
        self.state_ranges[5, 1] = 10.0  # power

        # Initialize action ranges [variable_index, min, max]
        self.action_ranges = Tensor[DType](TensorShape(2, 2))
        self.action_ranges[0, 0] = 0.1
        self.action_ranges[0, 1] = 5.0  # current_density
        self.action_ranges[1, 0] = 0.5
        self.action_ranges[1, 1] = 2.0  # flow_rate_ratio

        # Initialize Q-table with small random values
        self.initialize_q_table()

    fn initialize_q_table(mut self):
        """Initialize Q-table with small random values for better exploration"""
        var total_elements = self.q_table.num_elements()

        @parameter
        fn init_element(i: Int):
            var random_val = random_float64(-0.01, 0.01)
            self.q_table._buffer[i] = random_val

        vectorize[init_element, simd_width](total_elements)

    fn discretize_state(self, state: MFCState, power_density: Float64) -> Int:
        """Convert continuous state to discrete state index"""
        var state_values = Tensor[DType](TensorShape(6))
        state_values[0] = state.C_AC
        state_values[1] = state.X
        state_values[2] = state.C_O2
        state_values[3] = state.eta_a
        state_values[4] = state.eta_c
        state_values[5] = power_density

        var state_index = 0
        var multiplier = 1

        for i in range(6):
            var val = state_values[i]
            var min_val = self.state_ranges[i, 0]
            var max_val = self.state_ranges[i, 1]

            # Clip and discretize
            var clipped_val = min(max(val, min_val), max_val)
            var normalized = (clipped_val - min_val) / (max_val - min_val)
            var bin_idx = int(normalized * (self.config.n_state_bins - 1))
            bin_idx = min(max(bin_idx, 0), self.config.n_state_bins - 1)

            state_index += bin_idx * multiplier
            multiplier *= self.config.n_state_bins

        return state_index

    fn discretize_action(self, action: MFCAction) -> Int:
        """Convert continuous action to discrete action index"""
        var action_values = Tensor[DType](TensorShape(2))
        action_values[0] = action.current_density
        action_values[1] = action.flow_rate_ratio

        var action_index = 0
        var multiplier = 1

        for i in range(2):
            var val = action_values[i]
            var min_val = self.action_ranges[i, 0]
            var max_val = self.action_ranges[i, 1]

            # Clip and discretize
            var clipped_val = min(max(val, min_val), max_val)
            var normalized = (clipped_val - min_val) / (max_val - min_val)
            var bin_idx = int(normalized * (self.config.n_action_bins - 1))
            bin_idx = min(max(bin_idx, 0), self.config.n_action_bins - 1)

            action_index += bin_idx * multiplier
            multiplier *= self.config.n_action_bins

        return action_index

    fn action_from_index(self, action_index: Int) -> MFCAction:
        """Convert action index back to continuous action"""
        var n_bins = self.config.n_action_bins
        var flow_bin = action_index % n_bins
        var current_bin = action_index // n_bins

        var current_min = self.action_ranges[0, 0]
        var current_max = self.action_ranges[0, 1]
        var flow_min = self.action_ranges[1, 0]
        var flow_max = self.action_ranges[1, 1]

        var current_density = current_min + (
            current_max - current_min
        ) * current_bin / (n_bins - 1)
        var flow_rate_ratio = flow_min + (flow_max - flow_min) * flow_bin / (
            n_bins - 1
        )

        var action = MFCAction()
        action.current_density = current_density
        action.flow_rate_ratio = flow_rate_ratio
        return action

    fn choose_action(self, state_index: Int) -> Int:
        """Choose action using epsilon-greedy policy"""
        if random_float64() < self.current_epsilon:
            # Explore: random action
            return int(random_float64() * (self.config.n_action_bins**2))
        else:
            # Exploit: best action according to Q-table
            return self.get_best_action(state_index)

    fn get_best_action(self, state_index: Int) -> Int:
        """Get best action for given state from Q-table"""
        var n_actions = self.config.n_action_bins**2
        var best_action = 0
        var best_value = self.q_table[state_index, 0]

        for action in range(1, n_actions):
            var value = self.q_table[state_index, action]
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    fn update_q_table(
        mut self, state: Int, action: Int, reward: Float64, next_state: Int
    ):
        """Update Q-table using Q-learning update rule"""
        var old_value = self.q_table[state, action]
        var next_max = self.get_max_q_value(next_state)

        var new_value = old_value + self.config.learning_rate * (
            reward + self.config.discount_factor * next_max - old_value
        )

        self.q_table[state, action] = new_value

    fn get_max_q_value(self, state_index: Int) -> Float64:
        """Get maximum Q-value for given state"""
        var n_actions = self.config.n_action_bins**2
        var max_value = self.q_table[state_index, 0]

        for action in range(1, n_actions):
            var value = self.q_table[state_index, action]
            if value > max_value:
                max_value = value

        return max_value

    fn simulate_mfc_step(
        self, state: MFCState, action: MFCAction, dt: Float64 = 1.0
    ) -> (MFCState, Float64):
        """Simulate one MFC step with given action - vectorized for performance
        """

        # MFC parameters (simplified for demonstration)
        var F = 96485.332
        var R = 8.314
        var T = 303.0
        var k1_0 = 0.207
        var k2_0 = 3.288e-5
        var K_AC = 0.592
        var K_O2 = 0.004
        var alpha = 0.051
        var beta = 0.063
        var Q_a = 2.25e-5 * action.flow_rate_ratio
        var Q_c = 1.11e-3 * action.flow_rate_ratio
        var V_a = 5.5e-5
        var V_c = 5.5e-5
        var A_m = 5.0e-4
        var Y_ac = 0.05
        var K_dec = 8.33e-4
        var C_AC_in = 1.56
        var C_O2_in = 0.3125

        # Calculate reaction rates
        var r1 = (
            k1_0
            * exp((alpha * F) / (R * T) * state.eta_a)
            * (state.C_AC / (K_AC + state.C_AC))
            * state.X
        )
        var r2 = (
            -k2_0
            * (state.C_O2 / (K_O2 + state.C_O2))
            * exp((beta - 1.0) * F / (R * T) * state.eta_c)
        )
        var N_M = (3600.0 * action.current_density) / F

        # Calculate derivatives
        var dC_AC_dt = (Q_a * (C_AC_in - state.C_AC) - A_m * r1) / V_a
        var dC_O2_dt = (Q_c * (C_O2_in - state.C_O2) + r2 * A_m) / V_c
        var dX_dt = (A_m * Y_ac * r1) / V_a - K_dec * state.X
        var d_eta_a_dt = (
            3600.0 * action.current_density - 8.0 * F * r1
        ) / 400.0  # C_a
        var d_eta_c_dt = (
            -3600.0 * action.current_density - 4.0 * F * r2
        ) / 500.0  # C_c

        # Update state using Euler integration
        var next_state = state
        next_state.C_AC = state.C_AC + dC_AC_dt * dt
        next_state.C_O2 = state.C_O2 + dC_O2_dt * dt
        next_state.X = state.X + dX_dt * dt
        next_state.eta_a = state.eta_a + d_eta_a_dt * dt
        next_state.eta_c = state.eta_c + d_eta_c_dt * dt

        # Ensure physical bounds
        next_state.C_AC = max(0.0, min(next_state.C_AC, 5.0))
        next_state.C_O2 = max(0.0, min(next_state.C_O2, 1.0))
        next_state.X = max(0.0, min(next_state.X, 2.0))
        next_state.eta_a = max(-1.0, min(next_state.eta_a, 1.0))
        next_state.eta_c = max(-1.0, min(next_state.eta_c, 1.0))

        # Calculate power density
        var power_density = action.current_density * (
            next_state.eta_a - next_state.eta_c
        )

        return (next_state, power_density)

    fn calculate_reward(
        self, state: MFCState, power_density: Float64, action: MFCAction
    ) -> Float64:
        """Calculate reward based on MFC performance - vectorized"""

        # Primary objective: maximize power density
        var power_reward = power_density / 10.0

        # Stability penalty
        var stability_penalty = 0.0
        if abs(state.eta_a) > 0.3 or abs(state.eta_c) > 0.3:
            stability_penalty = -0.5

        # COD removal efficiency
        var cod_removal_reward = 0.0
        if state.C_AC < 1.0:
            cod_removal_reward = 0.2

        # Action penalty for extreme values
        var action_penalty = 0.0
        if action.current_density > 4.0 or action.flow_rate_ratio > 1.8:
            action_penalty = -0.1

        return (
            power_reward
            + stability_penalty
            + cod_removal_reward
            + action_penalty
        )

    fn train(mut self):
        """Train the Q-learning controller using vectorized operations"""

        print("Starting Q-learning training for MFC control...")
        seed()

        for episode in range(self.config.max_episodes):
            # Initialize episode state
            var current_state = MFCState()
            var episode_reward = 0.0
            var episode_power = 0.0
            var valid_steps = 0

            for step in range(self.config.max_steps_per_episode):
                # Get current state index
                var power_estimate = 0.1
                var state_index = self.discretize_state(
                    current_state, power_estimate
                )

                # Choose action
                var action_index = self.choose_action(state_index)
                var action = self.action_from_index(action_index)

                # Execute action and get next state
                var result = self.simulate_mfc_step(current_state, action)
                var next_state = result[0]
                var power_density = result[1]

                # Calculate reward
                var reward = self.calculate_reward(
                    next_state, power_density, action
                )

                # Get next state index
                var next_state_index = self.discretize_state(
                    next_state, power_density
                )

                # Update Q-table
                self.update_q_table(
                    state_index, action_index, reward, next_state_index
                )

                # Update episode statistics
                current_state = next_state
                episode_reward += reward
                episode_power += power_density
                valid_steps += 1

                # Check for termination
                if power_density < -0.5:
                    break

            # Update epsilon
            if self.current_epsilon > self.config.epsilon_min:
                self.current_epsilon *= self.config.epsilon_decay

            # Record episode statistics
            self.reward_history[episode] = episode_reward
            if valid_steps > 0:
                self.power_history[episode] = episode_power / valid_steps

            # Print progress
            if episode % 100 == 0:
                var avg_reward = self.get_average_reward(episode, 100)
                var avg_power = self.get_average_power(episode, 100)
                print(
                    "Episode",
                    episode,
                    ": Avg Reward =",
                    avg_reward,
                    ", Avg Power =",
                    avg_power,
                    ", Epsilon =",
                    self.current_epsilon,
                )

        self.episode_count = self.config.max_episodes
        print("Training completed!")

    fn get_average_reward(self, episode: Int, window: Int) -> Float64:
        """Calculate average reward over recent episodes"""
        var start_idx = max(0, episode - window + 1)
        var sum_reward = 0.0
        var count = 0

        for i in range(start_idx, episode + 1):
            sum_reward += self.reward_history[i]
            count += 1

        return sum_reward / count if count > 0 else 0.0

    fn get_average_power(self, episode: Int, window: Int) -> Float64:
        """Calculate average power over recent episodes"""
        var start_idx = max(0, episode - window + 1)
        var sum_power = 0.0
        var count = 0

        for i in range(start_idx, episode + 1):
            sum_power += self.power_history[i]
            count += 1

        return sum_power / count if count > 0 else 0.0

    fn test_controller(mut self, n_test_episodes: Int = 10) -> Tensor[DType]:
        """Test the trained controller and return performance metrics"""

        print("Testing trained Q-learning controller...")

        # Disable exploration for testing
        var original_epsilon = self.current_epsilon
        self.current_epsilon = 0.0

        var test_results = Tensor[DType](
            TensorShape(n_test_episodes, 3)
        )  # [episode, avg_power, avg_reward]

        for episode in range(n_test_episodes):
            # Initialize test state
            var current_state = MFCState()
            current_state.C_AC = 1.2  # Slightly different initial conditions
            current_state.X = 0.12
            current_state.C_O2 = 0.28
            current_state.eta_a = 0.02
            current_state.eta_c = -0.02

            var episode_power = 0.0
            var episode_reward = 0.0
            var valid_steps = 0

            for step in range(50):  # Shorter test episodes
                var power_estimate = 0.1
                var state_index = self.discretize_state(
                    current_state, power_estimate
                )
                var action_index = self.choose_action(state_index)
                var action = self.action_from_index(action_index)

                var result = self.simulate_mfc_step(current_state, action)
                var next_state = result[0]
                var power_density = result[1]

                var reward = self.calculate_reward(
                    next_state, power_density, action
                )

                current_state = next_state
                episode_power += power_density
                episode_reward += reward
                valid_steps += 1

                if power_density < -0.5:
                    break

            # Store test results
            test_results[episode, 0] = episode
            test_results[episode, 1] = (
                episode_power / valid_steps if valid_steps > 0 else 0.0
            )
            test_results[episode, 2] = episode_reward

        # Restore epsilon
        self.current_epsilon = original_epsilon

        # Print test summary
        var avg_power = 0.0
        var avg_reward = 0.0
        for i in range(n_test_episodes):
            avg_power += test_results[i, 1]
            avg_reward += test_results[i, 2]

        avg_power /= n_test_episodes
        avg_reward /= n_test_episodes

        print("Test Results:")
        print("Average Power Density:", avg_power, "W/m²")
        print("Average Reward:", avg_reward)

        return test_results

    fn save_q_table(self, filename: String):
        """Save Q-table to file for later use"""
        # In a real implementation, you would save to file
        # For now, just print statistics
        print("Q-table shape:", self.q_table.shape().__str__())
        print("Q-table statistics:")

        var total_sum = 0.0
        var count = 0
        var total_elements = self.q_table.num_elements()

        for i in range(total_elements):
            total_sum += self.q_table._buffer[i]
            count += 1

        var mean_value = total_sum / count
        print("Mean Q-value:", mean_value)
        print("Total episodes trained:", self.episode_count)


# Main execution function
fn main():
    print("=== Mojo Q-Learning MFC Controller ===")

    # Initialize configuration
    var config = MFCQLearningConfig()
    config.max_episodes = 500
    config.max_steps_per_episode = 50
    config.learning_rate = 0.1
    config.epsilon = 0.3

    # Create controller
    var controller = MFCQLearningController(config)

    # Train the controller
    var start_time = time.now()
    controller.train()
    var training_time = time.now() - start_time

    print("Training completed in", training_time, "seconds")

    # Test the controller
    var test_results = controller.test_controller(5)

    # Save results
    controller.save_q_table("mfc_q_table.bin")

    print("=== Q-Learning MFC Controller Complete ===")
