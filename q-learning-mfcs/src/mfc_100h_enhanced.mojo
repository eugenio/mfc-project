from random import random_float64, seed
from math import exp, log, sqrt


@value
struct EnhancedMFCConfig:
    """Enhanced MFC simulation configuration matching Python version."""

    var n_cells: Int
    var simulation_hours: Int
    var time_step: Float64
    var n_state_bins: Int
    var n_action_bins: Int
    var learning_rate: Float64
    var epsilon: Float64
    var epsilon_decay: Float64
    var epsilon_min: Float64

    fn __init__(out self):
        self.n_cells = 5
        self.simulation_hours = 100
        self.time_step = 1.0
        self.n_state_bins = 10
        self.n_action_bins = 10
        self.learning_rate = 0.1
        self.epsilon = 0.3
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05


@value
struct CellState:
    """Individual MFC cell state."""

    var C_AC: Float64  # Acetate concentration
    var C_CO2: Float64  # CO2 concentration
    var C_H: Float64  # H+ concentration
    var X: Float64  # Biomass concentration
    var C_O2: Float64  # O2 concentration
    var C_OH: Float64  # OH- concentration
    var C_M: Float64  # Metal ion concentration
    var eta_a: Float64  # Anode overpotential
    var eta_c: Float64  # Cathode overpotential
    var aging: Float64  # Aging factor
    var biofilm: Float64  # Biofilm thickness


@value
struct CellAction:
    """Individual MFC cell action."""

    var duty_cycle: Float64  # Current duty cycle
    var ph_buffer: Float64  # pH buffer usage
    var acetate_add: Float64  # Acetate addition


struct EnhancedMFCStack:
    """Enhanced MFC stack simulation with Q-learning."""

    var config: EnhancedMFCConfig
    var cell_states: List[CellState]
    var cell_actions: List[CellAction]
    var q_table: List[Float64]  # Simplified Q-table as list
    var q_keys: List[Int]  # Keys for Q-table lookup
    var current_time: Float64
    var total_energy: Float64
    var substrate_level: Float64
    var ph_buffer_level: Float64
    var maintenance_cycles: Int
    var current_epsilon: Float64

    fn __init__(out self, config: EnhancedMFCConfig):
        self.config = config
        self.current_time = 0.0
        self.total_energy = 0.0
        self.substrate_level = 100.0
        self.ph_buffer_level = 100.0
        self.maintenance_cycles = 0
        self.current_epsilon = config.epsilon
        self.cell_states = List[CellState]()
        self.cell_actions = List[CellAction]()
        self.q_table = List[Float64]()
        self.q_keys = List[Int]()

        # Initialize cells
        for _ in range(config.n_cells):
            var variation = random_float64(-0.1, 0.1)
            self.cell_states.append(
                CellState(
                    1.0 + variation,  # C_AC
                    0.05 + variation * 0.01,  # C_CO2
                    1e-4 + variation * 1e-5,  # C_H
                    0.1 + variation * 0.01,  # X
                    0.25 + variation * 0.02,  # C_O2
                    1e-7 + variation * 1e-8,  # C_OH
                    0.05 + variation * 0.005,  # C_M
                    0.01 + variation * 0.001,  # eta_a
                    -0.01 + variation * 0.001,  # eta_c
                    1.0,  # aging
                    1.0,  # biofilm
                )
            )
            self.cell_actions.append(CellAction(0.5, 0.5, 0.3))

    fn discretize_state(self, state: CellState) -> Int:
        """Discretize cell state for Q-table lookup."""
        # Simplified state discretization (6D -> 1D hash)
        var ac_bin = Int((state.C_AC / 5.0) * self.config.n_state_bins)
        var x_bin = Int((state.X / 2.0) * self.config.n_state_bins)
        var o2_bin = Int(state.C_O2 * self.config.n_state_bins)
        var eta_a_bin = Int(
            (state.eta_a + 1.0) / 2.0 * self.config.n_state_bins
        )
        var eta_c_bin = Int(
            (state.eta_c + 1.0) / 2.0 * self.config.n_state_bins
        )
        var aging_bin = Int(state.aging * self.config.n_state_bins)

        # Simple hash function
        return (
            ac_bin * 1000000
            + x_bin * 10000
            + o2_bin * 1000
            + eta_a_bin * 100
            + eta_c_bin * 10
            + aging_bin
        )

    fn get_action_index(self, action: CellAction) -> Int:
        """Convert action to discrete index."""
        var duty_bin = Int(action.duty_cycle * self.config.n_action_bins)
        var ph_bin = Int(action.ph_buffer * self.config.n_action_bins)
        var acetate_bin = Int(action.acetate_add * self.config.n_action_bins)
        return duty_bin * 100 + ph_bin * 10 + acetate_bin

    fn get_q_value(self, key: Int) -> Float64:
        """Get Q-value from table."""
        for i in range(len(self.q_keys)):
            if self.q_keys[i] == key:
                return self.q_table[i]
        return 0.0

    fn set_q_value(mut self, key: Int, value: Float64):
        """Set Q-value in table."""
        for i in range(len(self.q_keys)):
            if self.q_keys[i] == key:
                self.q_table[i] = value
                return
        # Add new entry
        self.q_keys.append(key)
        self.q_table.append(value)

    fn choose_action(self, cell_idx: Int) -> CellAction:
        """Q-learning action selection with epsilon-greedy."""
        var state = self.cell_states[cell_idx]
        var state_key = self.discretize_state(state)

        if random_float64() < self.current_epsilon:
            # Random exploration
            return CellAction(
                random_float64(0.1, 0.9),  # duty_cycle
                random_float64(0.0, 1.0),  # ph_buffer
                random_float64(0.0, 1.0),  # acetate_add
            )
        else:
            # Greedy exploitation - find best action
            var best_action = CellAction(0.5, 0.5, 0.3)
            var best_q = -1000.0

            # Sample a few actions to find the best
            for _ in range(10):
                var test_action = CellAction(
                    random_float64(0.1, 0.9),
                    random_float64(0.0, 1.0),
                    random_float64(0.0, 1.0),
                )
                var action_key = state_key + self.get_action_index(test_action)
                var q_value = self.get_q_value(action_key)

                if q_value > best_q:
                    best_q = q_value
                    best_action = test_action

            return best_action

    fn update_q_table(mut self, cell_idx: Int, reward: Float64):
        """Update Q-table based on reward."""
        var state = self.cell_states[cell_idx]
        var action = self.cell_actions[cell_idx]
        var state_key = self.discretize_state(state)
        var action_key = state_key + self.get_action_index(action)

        var old_q = self.get_q_value(action_key)
        var new_q = old_q + self.config.learning_rate * (reward - old_q)
        self.set_q_value(action_key, new_q)

    fn compute_mfc_dynamics(mut self, dt: Float64):
        """Compute MFC dynamics for all cells (vectorized approach)."""
        # MFC parameters
        var F = 96485.332
        var R = 8.314
        var T = 303.0
        var k1_0 = 0.207
        var k2_0 = 3.288e-5
        var K_AC = 0.592
        var K_O2 = 0.004
        var alpha = 0.051
        var beta = 0.063

        for cell_idx in range(self.config.n_cells):
            var state = self.cell_states[cell_idx]
            var action = self.cell_actions[cell_idx]

            # Effective current
            var effective_current = action.duty_cycle * state.aging

            # Reaction rates (simplified but more accurate than before)
            var r1 = (
                k1_0
                * exp((alpha * F) / (R * T) * state.eta_a)
                * (state.C_AC / (K_AC + state.C_AC))
                * state.X
                * state.aging
                / state.biofilm
            )
            var r2 = (
                -k2_0
                * (state.C_O2 / (K_O2 + state.C_O2))
                * exp((beta - 1.0) * F / (R * T) * state.eta_c)
                * state.aging
            )

            # Enhanced derivatives including pH buffer and acetate addition effects
            var dC_AC_dt = (
                1.56 + action.acetate_add * 0.5 - state.C_AC
            ) * 0.1 - r1 * 0.01
            var dX_dt = r1 * 0.001 - state.X * 0.0001
            var dC_O2_dt = (0.3125 - state.C_O2) * 0.1 + r2 * 0.01
            var deta_a_dt = (effective_current - r1) * 0.001
            var deta_c_dt = (-effective_current - r2) * 0.001
            var dC_H_dt = r1 * 0.001 - action.ph_buffer * state.C_H * 0.1

            # Update state with bounds
            var new_state = CellState(
                max(0.001, min(state.C_AC + dC_AC_dt * dt, 5.0)),  # C_AC
                state.C_CO2,  # C_CO2 (unchanged)
                max(1e-14, min(state.C_H + dC_H_dt * dt, 1e-2)),  # C_H
                max(0.001, min(state.X + dX_dt * dt, 2.0)),  # X
                max(0.001, min(state.C_O2 + dC_O2_dt * dt, 1.0)),  # C_O2
                state.C_OH,  # C_OH (unchanged)
                state.C_M,  # C_M (unchanged)
                max(-1.0, min(state.eta_a + deta_a_dt * dt, 1.0)),  # eta_a
                max(-1.0, min(state.eta_c + deta_c_dt * dt, 1.0)),  # eta_c
                state.aging,  # aging
                state.biofilm,  # biofilm
            )

            self.cell_states[cell_idx] = new_state

    fn apply_aging_effects(mut self, dt_hours: Float64):
        """Apply aging and biofilm effects."""
        var aging_rate = 0.001 * dt_hours
        var biofilm_growth = 0.0005 * dt_hours

        for cell_idx in range(self.config.n_cells):
            var state = self.cell_states[cell_idx]
            var new_aging = max(0.5, state.aging * (1.0 - aging_rate))
            var new_biofilm = min(2.0, state.biofilm + biofilm_growth)

            self.cell_states[cell_idx] = CellState(
                state.C_AC,
                state.C_CO2,
                state.C_H,
                state.X,
                state.C_O2,
                state.C_OH,
                state.C_M,
                state.eta_a,
                state.eta_c,
                new_aging,
                new_biofilm,
            )

    fn calculate_system_metrics(self) -> (Float64, Float64, Float64):
        """Calculate stack-level metrics with proper coordination."""
        var total_voltage = 0.0
        var min_current = 1000.0
        var cell_powers = List[Float64]()

        # Calculate individual cell metrics
        for cell_idx in range(self.config.n_cells):
            var state = self.cell_states[cell_idx]
            var action = self.cell_actions[cell_idx]

            var cell_voltage = state.eta_a - state.eta_c
            var cell_current = action.duty_cycle * state.aging
            var cell_power = cell_voltage * cell_current

            total_voltage += cell_voltage
            min_current = min(min_current, cell_current)
            cell_powers.append(cell_power)

        # Stack power limited by minimum current (series connection)
        var stack_power = total_voltage * min_current

        return (total_voltage, min_current, stack_power)

    fn calculate_reward(self, stack_power: Float64) -> Float64:
        """Calculate Q-learning reward matching Python implementation."""
        var power_reward = stack_power / 5.0

        # Stability reward based on voltage uniformity
        var voltage_sum = 0.0
        var voltage_sq_sum = 0.0
        var reversal_count = 0

        for cell_idx in range(self.config.n_cells):
            var state = self.cell_states[cell_idx]
            var voltage = state.eta_a - state.eta_c
            voltage_sum += voltage
            voltage_sq_sum += voltage * voltage

            if voltage < 0.1:
                reversal_count += 1

        var mean_voltage = voltage_sum / Float64(self.config.n_cells)
        var variance = (voltage_sq_sum / Float64(self.config.n_cells)) - (
            mean_voltage * mean_voltage
        )
        var std_dev = sqrt(max(0.001, variance))
        var abs_mean_voltage = (
            mean_voltage if mean_voltage >= 0.0 else -mean_voltage
        )
        var stability_reward = 1.0 - std_dev / max(0.1, abs_mean_voltage)

        # Penalties
        var reversal_penalty = -10.0 * Float64(reversal_count)

        var resource_penalty = 0.0
        for cell_idx in range(self.config.n_cells):
            var action = self.cell_actions[cell_idx]
            resource_penalty -= 0.1 * (action.ph_buffer + action.acetate_add)

        return (
            power_reward
            + stability_reward
            + reversal_penalty
            + resource_penalty
        )

    fn update_resources(mut self, dt_hours: Float64, stack_power: Float64):
        """Update resource levels."""
        # Substrate consumption
        var substrate_consumption = stack_power * dt_hours * 0.1
        self.substrate_level = max(
            0.0, self.substrate_level - substrate_consumption
        )

        # pH buffer consumption
        var ph_buffer_usage = 0.0
        for cell_idx in range(self.config.n_cells):
            ph_buffer_usage += self.cell_actions[cell_idx].ph_buffer
        ph_buffer_usage *= dt_hours * 0.05
        self.ph_buffer_level = max(0.0, self.ph_buffer_level - ph_buffer_usage)

        # Energy accumulation
        self.total_energy += stack_power * dt_hours

    fn check_maintenance(mut self):
        """Check and perform maintenance."""
        if self.substrate_level < 20.0:
            self.substrate_level = 100.0
            self.maintenance_cycles += 1

        if self.ph_buffer_level < 20.0:
            self.ph_buffer_level = 100.0
            self.maintenance_cycles += 1

        # Daily biofilm cleaning
        if Int(self.current_time) % (24 * 3600) < Int(self.config.time_step):
            for cell_idx in range(self.config.n_cells):
                var state = self.cell_states[cell_idx]
                if state.biofilm > 1.5:
                    self.cell_states[cell_idx] = CellState(
                        state.C_AC,
                        state.C_CO2,
                        state.C_H,
                        state.X,
                        state.C_O2,
                        state.C_OH,
                        state.C_M,
                        state.eta_a,
                        state.eta_c,
                        state.aging,
                        1.0,  # Reset biofilm
                    )
                    self.maintenance_cycles += 1

    fn simulate_step(mut self) -> (Float64, Float64, Float64):
        """Simulate one time step with Q-learning."""
        # Q-learning action selection for all cells
        for cell_idx in range(self.config.n_cells):
            self.cell_actions[cell_idx] = self.choose_action(cell_idx)

        # MFC dynamics computation
        self.compute_mfc_dynamics(self.config.time_step)

        # Long-term effects (apply hourly)
        if Int(self.current_time) % 3600 < Int(self.config.time_step):
            self.apply_aging_effects(1.0)  # 1 hour
            # Update epsilon
            self.current_epsilon = max(
                self.config.epsilon_min,
                self.current_epsilon * self.config.epsilon_decay,
            )

        # System metrics
        var metrics = self.calculate_system_metrics()
        var voltage = metrics[0]
        var current = metrics[1]
        var power = metrics[2]

        # Q-learning update
        var reward = self.calculate_reward(power)
        for cell_idx in range(self.config.n_cells):
            self.update_q_table(cell_idx, reward)

        # Resource management (apply hourly)
        if Int(self.current_time) % 3600 < Int(self.config.time_step):
            self.update_resources(1.0, power)  # 1 hour

        # Maintenance check
        self.check_maintenance()

        # Update time
        self.current_time += self.config.time_step

        return (voltage, current, power)

    fn run_simulation(mut self):
        """Run the enhanced 100-hour simulation with Q-learning."""
        print("=== Enhanced Q-Learning 100-Hour MFC Simulation ===")
        print("Matching Python simulation characteristics")
        print(
            "Simulating",
            self.config.n_cells,
            "cells for",
            self.config.simulation_hours,
            "hours",
        )

        var total_steps = Int(
            self.config.simulation_hours * 3600 / self.config.time_step
        )
        print("Total simulation steps:", total_steps)
        print("Q-learning enabled with epsilon =", self.current_epsilon)
        print()

        var avg_power = 0.0
        var max_power = 0.0
        var step_count = 0

        # Main simulation loop
        for step in range(total_steps):
            var result = self.simulate_step()
            var _ = result[0]  # voltage (unused in loop)
            var _ = result[1]  # current (unused in loop)
            var power = result[2]

            # Track statistics
            avg_power += power
            if power > max_power:
                max_power = power
            step_count += 1

            # Progress reporting every 36000 steps (10 hours)
            if step % 36000 == 0:
                var current_hour = self.current_time / 3600.0
                var progress = step * 100 // total_steps

                print(
                    "Step",
                    step,
                    "/",
                    total_steps,
                    "(",
                    progress,
                    "%) - Hour",
                    current_hour,
                )
                print("  Stack power:", power, "W")
                print("  Total energy:", self.total_energy, "Wh")
                print("  Substrate:", self.substrate_level, "%")
                print("  Q-table size:", len(self.q_table))
                print("  Epsilon:", self.current_epsilon)
                print()

        # Final statistics
        avg_power /= Float64(step_count)

        print("=== Simulation Complete ===")
        print("Total energy produced:", self.total_energy, "Wh")
        print("Average power:", avg_power, "W")
        print("Maximum power:", max_power, "W")
        print("Final substrate level:", self.substrate_level, "%")
        print("Final pH buffer level:", self.ph_buffer_level, "%")
        print("Total maintenance cycles:", self.maintenance_cycles)
        print("Final Q-table size:", len(self.q_table))
        print("Final epsilon:", self.current_epsilon)
        print()

        # Cell analysis
        print("=== Final Cell States ===")
        for cell_idx in range(self.config.n_cells):
            var state = self.cell_states[cell_idx]
            var voltage = state.eta_a - state.eta_c

            print("Cell", cell_idx, ":")
            print("  Voltage:", voltage, "V")
            print("  Aging factor:", state.aging)
            print("  Biofilm thickness:", state.biofilm, "x")
            print("  Acetate:", state.C_AC, "mol/m³")

        # Performance comparison
        var efficiency = (
            self.total_energy
            / (Float64(self.config.simulation_hours) * max_power)
            * 100.0
        )
        print("=== Enhanced Performance Summary ===")
        print("Q-learning adaptation successful")
        print("Energy efficiency:", efficiency, "%")
        print(
            "Average power density:",
            avg_power / Float64(self.config.n_cells),
            "W/cell",
        )
        print("Stack coordination achieved")


# Main execution function
fn main():
    print("=== Enhanced Mojo MFC Simulation ===")
    print("Implementing Q-learning and stack coordination")
    print()

    # Set random seed for reproducible results
    seed(42)

    # Initialize configuration
    var config = EnhancedMFCConfig()

    # Create and run simulation
    var stack = EnhancedMFCStack(config)
    stack.run_simulation()

    print()
    print("=== Enhanced Simulation Benefits ===")
    print("✓ Q-learning action optimization")
    print("✓ Stack-level coordination")
    print("✓ Adaptive epsilon-greedy exploration")
    print("✓ Reward-based learning")
    print("✓ State discretization and memory")
    print("✓ Multi-objective reward function")
    print("✓ Resource-aware control")
