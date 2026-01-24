"""100-Hour MFC Stack Simulation with Q-Learning Control.

This extended simulation demonstrates:
- Long-term system stability over 100 hours
- Adaptive control learning and optimization
- Substrate depletion and replenishment cycles
- pH drift and buffer management
- Cell aging and performance degradation
- System recovery and maintenance cycles
"""

import json
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from mfc_stack_simulation import MFCStack, MFCStackQLearningController
from path_config import get_figure_path, get_simulation_data_path


class LongTermMFCStack(MFCStack):
    """Extended MFC stack with long-term effects."""

    def __init__(self) -> None:
        super().__init__()

        # Long-term parameters
        self.substrate_tank_level = 100.0  # % full
        self.ph_buffer_tank_level = 100.0  # % full
        self.maintenance_cycles = 0
        self.total_energy_produced = 0.0  # Wh

        # Aging parameters
        self.cell_aging_factors = [1.0] * 5  # Performance degradation over time
        self.biofilm_thickness = [1.0] * 5  # Affects mass transfer

        # Environmental conditions
        self.ambient_temperature = 30.0  # °C
        self.seasonal_variation = 0.0

        # Extended logging for 100h simulation
        self.hourly_data = {
            "hour": [],
            "stack_power": [],
            "stack_voltage": [],
            "substrate_level": [],
            "ph_buffer_level": [],
            "total_energy": [],
            "cell_aging": [],
            "maintenance_events": [],
            "system_efficiency": [],
        }

    def apply_long_term_effects(self, dt_hours) -> None:
        """Apply long-term degradation and aging effects."""
        # Cell aging (0.1% per hour typical)
        aging_rate = 0.001 * dt_hours
        for i in range(5):
            self.cell_aging_factors[i] *= 1 - aging_rate
            self.cell_aging_factors[i] = max(
                0.5,
                self.cell_aging_factors[i],
            )  # Minimum 50% performance

        # Biofilm growth (affects mass transfer)
        biofilm_growth_rate = 0.0005 * dt_hours
        for i in range(5):
            self.biofilm_thickness[i] += biofilm_growth_rate
            self.biofilm_thickness[i] = min(
                2.0,
                self.biofilm_thickness[i],
            )  # Maximum 2x thickness

        # Substrate consumption
        substrate_consumption = self.stack_power * dt_hours * 0.1  # 0.1% per Wh
        self.substrate_tank_level -= substrate_consumption
        self.substrate_tank_level = max(0, self.substrate_tank_level)

        # pH buffer consumption
        ph_buffer_usage = (
            sum(cell.actuators["ph_buffer"].get_value() for cell in self.cells)
            * dt_hours
            * 0.05
        )
        self.ph_buffer_tank_level -= ph_buffer_usage
        self.ph_buffer_tank_level = max(0, self.ph_buffer_tank_level)

        # Environmental variations (seasonal temperature)
        hour_of_year = (self.time / 3600) % (24 * 365)
        self.seasonal_variation = 5 * np.sin(2 * np.pi * hour_of_year / (24 * 365))
        self.ambient_temperature = 30.0 + self.seasonal_variation

        # Apply aging effects to cells
        for i, cell in enumerate(self.cells):
            # Reduce performance based on aging
            cell.state[3] *= self.cell_aging_factors[i]  # Biomass affected by aging

            # Mass transfer limitations from biofilm
            if self.biofilm_thickness[i] > 1.2:
                cell.state[0] *= 0.95  # Reduced substrate access
                cell.state[4] *= 0.95  # Reduced oxygen access

        # Total energy calculation
        self.total_energy_produced += self.stack_power * dt_hours

    def check_maintenance_needs(self):
        """Check if maintenance is needed and perform if required."""
        maintenance_needed = False

        # Check for maintenance triggers
        if self.substrate_tank_level < 20:
            maintenance_needed = True
            self.substrate_tank_level = 100.0

        if self.ph_buffer_tank_level < 20:
            maintenance_needed = True
            self.ph_buffer_tank_level = 100.0

        # Cell cleaning (every 24 hours)
        if self.time % (24 * 3600) < 10:  # Once per day
            for i in range(5):
                if self.biofilm_thickness[i] > 1.5:
                    self.biofilm_thickness[i] = 1.0
                    maintenance_needed = True
            if maintenance_needed:
                pass

        if maintenance_needed:
            self.maintenance_cycles += 1

        return maintenance_needed

    def log_hourly_data(self) -> None:
        """Log data every hour."""
        if self.time % 3600 < 10:  # Every hour
            current_hour = self.time / 3600

            self.hourly_data["hour"].append(current_hour)
            self.hourly_data["stack_power"].append(self.stack_power)
            self.hourly_data["stack_voltage"].append(self.stack_voltage)
            self.hourly_data["substrate_level"].append(self.substrate_tank_level)
            self.hourly_data["ph_buffer_level"].append(self.ph_buffer_tank_level)
            self.hourly_data["total_energy"].append(self.total_energy_produced)
            self.hourly_data["cell_aging"].append(np.mean(self.cell_aging_factors))
            self.hourly_data["maintenance_events"].append(self.maintenance_cycles)

            # Calculate system efficiency
            theoretical_max_power = 5 * 1.0  # 5 cells × 1W each
            efficiency = (
                self.stack_power / theoretical_max_power
                if theoretical_max_power > 0
                else 0
            )
            self.hourly_data["system_efficiency"].append(efficiency)


class LongTermController(MFCStackQLearningController):
    """Extended controller with long-term learning capabilities."""

    def __init__(self, stack) -> None:
        super().__init__(stack)

        # Long-term learning parameters
        self.learning_phases = {
            "exploration": {
                "duration": 24 * 3600,
                "epsilon": 0.5,
            },  # First 24h: high exploration
            "optimization": {
                "duration": 48 * 3600,
                "epsilon": 0.1,
            },  # Next 48h: optimization
            "maintenance": {
                "duration": 28 * 3600,
                "epsilon": 0.05,
            },  # Last 28h: maintenance mode
        }

        self.current_phase = "exploration"
        self.phase_start_time = 0

        # Performance tracking
        self.performance_windows = {"hourly": [], "daily": [], "weekly": []}

        # Adaptive parameters
        self.substrate_management_learned = False
        self.ph_optimization_learned = False
        self.load_balancing_learned = False

    def update_learning_phase(self) -> None:
        """Update learning phase based on elapsed time."""
        elapsed_time = self.stack.time - self.phase_start_time

        if (
            self.current_phase == "exploration"
            and elapsed_time > self.learning_phases["exploration"]["duration"]
        ):
            self.current_phase = "optimization"
            self.phase_start_time = self.stack.time
            self.epsilon = self.learning_phases["optimization"]["epsilon"]

        elif (
            self.current_phase == "optimization"
            and elapsed_time > self.learning_phases["optimization"]["duration"]
        ):
            self.current_phase = "maintenance"
            self.phase_start_time = self.stack.time
            self.epsilon = self.learning_phases["maintenance"]["epsilon"]

    def calculate_long_term_reward(self, state, actions):
        """Enhanced reward function for long-term optimization."""
        base_reward = super().calculate_reward(state, actions)

        # Long-term bonuses
        sustainability_bonus = 0.0

        # Substrate efficiency bonus
        if self.stack.substrate_tank_level > 50:
            sustainability_bonus += 0.2

        # pH buffer efficiency bonus
        if self.stack.ph_buffer_tank_level > 50:
            sustainability_bonus += 0.1

        # System longevity bonus
        avg_aging = np.mean(self.stack.cell_aging_factors)
        if avg_aging > 0.8:
            sustainability_bonus += 0.3

        # Energy production bonus
        if self.stack.total_energy_produced > 10.0:  # >10 Wh
            sustainability_bonus += 0.5

        return base_reward + sustainability_bonus

    def train_step(self):
        """Enhanced training step with long-term considerations."""
        # Update learning phase
        self.update_learning_phase()

        # Get current state
        current_state = self.stack.get_stack_state()

        # Get action with phase-appropriate exploration
        actions = self.get_action(current_state)

        # Apply long-term action modifications
        if self.current_phase == "maintenance":
            # Conservative actions in maintenance phase
            for i in range(5):
                actions[i * 3] = min(actions[i * 3], 0.7)  # Limit duty cycle
                actions[i * 3 + 1] = min(actions[i * 3 + 1], 0.5)  # Limit pH buffer

        # Apply actions
        self.stack.apply_control_actions(actions)

        # Update stack with long-term effects
        dt_hours = 1.0 / 3600  # 1 second = 1/3600 hours
        self.stack.apply_long_term_effects(dt_hours)
        self.stack.update_stack()

        # Check maintenance needs
        maintenance_performed = self.stack.check_maintenance_needs()

        # Get next state
        next_state = self.stack.get_stack_state()

        # Calculate enhanced reward
        reward = self.calculate_long_term_reward(next_state, actions)

        # Update Q-table
        self.update_q_table(current_state, actions, reward, next_state)

        # Log hourly data
        self.stack.log_hourly_data()

        # Update performance tracking
        self.performance_windows["hourly"].append(self.stack.stack_power)
        if len(self.performance_windows["hourly"]) > 3600:  # Keep last hour
            self.performance_windows["hourly"].pop(0)

        return reward, self.stack.stack_power, maintenance_performed


def run_100h_simulation():
    """Run 100-hour MFC stack simulation."""
    # Initialize system
    stack = LongTermMFCStack()
    controller = LongTermController(stack)

    # Simulation parameters
    simulation_time = 100 * 3600  # 100 hours in seconds
    dt = 1.0  # 1 second time step
    total_steps = int(simulation_time / dt)

    # Progress tracking
    progress_intervals = [10, 25, 50, 75, 90, 95, 99]  # Hours to report progress
    next_progress_idx = 0

    # Performance metrics
    metrics = {
        "start_time": time.time(),
        "steps_completed": 0,
        "maintenance_events": 0,
        "total_energy": 0.0,
        "avg_power": 0.0,
        "max_power": 0.0,
        "min_power": float("inf"),
        "power_history": [],
    }

    # Main simulation loop
    try:
        for step in range(total_steps):
            # Training step
            reward, power, maintenance = controller.train_step()

            # Update metrics
            metrics["steps_completed"] = step + 1
            metrics["total_energy"] = stack.total_energy_produced
            metrics["power_history"].append(power)
            metrics["max_power"] = max(metrics["max_power"], power)
            metrics["min_power"] = min(metrics["min_power"], power)

            if maintenance:
                metrics["maintenance_events"] += 1

            # Progress reporting
            current_hour = stack.time / 3600
            if (
                next_progress_idx < len(progress_intervals)
                and current_hour >= progress_intervals[next_progress_idx]
            ):
                time.time() - metrics["start_time"]
                (
                    np.mean(metrics["power_history"][-3600:])
                    if len(metrics["power_history"]) >= 3600
                    else np.mean(metrics["power_history"])
                )

                next_progress_idx += 1

            # Memory management - keep only recent data
            if len(metrics["power_history"]) > 10000:
                metrics["power_history"] = metrics["power_history"][-5000:]

    except KeyboardInterrupt:
        pass

    # Final analysis
    time.time() - metrics["start_time"]
    metrics["avg_power"] = np.mean(metrics["power_history"])

    stack.check_system_health()

    # Individual cell analysis
    for i, cell in enumerate(stack.cells):
        cell.get_sensor_readings()
        power = cell.get_power()
        stack.cell_aging_factors[i]
        stack.biofilm_thickness[i]

    # Save results
    save_simulation_results(stack, controller, metrics)

    return stack, controller, metrics


def save_simulation_results(stack, controller, metrics) -> None:
    """Save simulation results to files."""
    # Save hourly data
    results = {
        "simulation_info": {
            "duration_hours": 100,
            "total_steps": metrics["steps_completed"],
            "real_time_seconds": time.time() - metrics["start_time"],
            "timestamp": datetime.now().isoformat(),
        },
        "performance_metrics": {
            "total_energy_wh": stack.total_energy_produced,
            "average_power_w": metrics["avg_power"],
            "max_power_w": metrics["max_power"],
            "min_power_w": metrics["min_power"],
            "maintenance_events": stack.maintenance_cycles,
            "final_q_table_size": len(controller.q_table),
        },
        "hourly_data": stack.hourly_data,
        "final_cell_states": [],
    }

    # Add final cell states
    for i, cell in enumerate(stack.cells):
        readings = cell.get_sensor_readings()
        results["final_cell_states"].append(
            {
                "cell_id": i,
                "power": cell.get_power(),
                "voltage": readings["voltage"],
                "pH": readings["pH"],
                "acetate": readings["acetate"],
                "aging_factor": stack.cell_aging_factors[i],
                "biofilm_thickness": stack.biofilm_thickness[i],
                "reversed": cell.is_reversed,
            },
        )

    # Save to JSON
    with open(get_simulation_data_path("mfc_100h_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Generate plots
    generate_100h_plots(stack, controller, metrics)


def generate_100h_plots(stack, controller, metrics) -> None:
    """Generate plots for 100-hour simulation."""
    import matplotlib as mpl

    mpl.use("Agg")

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    # 1. Power evolution over 100 hours
    ax1 = axes[0, 0]
    hours = stack.hourly_data["hour"]
    power = stack.hourly_data["stack_power"]
    ax1.plot(hours, power, "b-", linewidth=1)
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Stack Power (W)")
    ax1.set_title("Power Evolution Over 100 Hours")
    ax1.grid(True)

    # 2. Cumulative energy production
    ax2 = axes[0, 1]
    energy = stack.hourly_data["total_energy"]
    ax2.plot(hours, energy, "g-", linewidth=2)
    ax2.set_xlabel("Time (hours)")
    ax2.set_ylabel("Cumulative Energy (Wh)")
    ax2.set_title("Total Energy Production")
    ax2.grid(True)

    # 3. System degradation
    ax3 = axes[1, 0]
    aging = stack.hourly_data["cell_aging"]
    ax3.plot(hours, aging, "r-", linewidth=2)
    ax3.set_xlabel("Time (hours)")
    ax3.set_ylabel("Average Cell Performance")
    ax3.set_title("System Aging Over Time")
    ax3.grid(True)

    # 4. Resource levels
    ax4 = axes[1, 1]
    substrate = stack.hourly_data["substrate_level"]
    ph_buffer = stack.hourly_data["ph_buffer_level"]
    ax4.plot(hours, substrate, "b-", label="Substrate", linewidth=2)
    ax4.plot(hours, ph_buffer, "r-", label="pH Buffer", linewidth=2)
    ax4.set_xlabel("Time (hours)")
    ax4.set_ylabel("Tank Level (%)")
    ax4.set_title("Resource Consumption")
    ax4.legend()
    ax4.grid(True)

    # 5. System efficiency
    ax5 = axes[2, 0]
    efficiency = stack.hourly_data["system_efficiency"]
    ax5.plot(hours, efficiency, "purple", linewidth=2)
    ax5.set_xlabel("Time (hours)")
    ax5.set_ylabel("System Efficiency")
    ax5.set_title("Efficiency Over Time")
    ax5.grid(True)

    # 6. Maintenance events
    ax6 = axes[2, 1]
    maintenance = stack.hourly_data["maintenance_events"]
    ax6.step(hours, maintenance, "orange", linewidth=2, where="post")
    ax6.set_xlabel("Time (hours)")
    ax6.set_ylabel("Cumulative Maintenance Events")
    ax6.set_title("Maintenance Schedule")
    ax6.grid(True)

    plt.tight_layout()
    plt.savefig(get_figure_path("mfc_100h_analysis.png"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    try:
        stack, controller, metrics = run_100h_simulation()

    except KeyboardInterrupt:
        pass

    except Exception:
        import traceback

        traceback.print_exc()
