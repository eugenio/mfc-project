from random import random_float64
from math import exp, log


@value
struct SimpleMFCConfig:
    """Simple MFC simulation configuration."""

    var n_cells: Int
    var simulation_hours: Int
    var time_step: Float64

    fn __init__(out self):
        self.n_cells = 5
        self.simulation_hours = 100
        self.time_step = 1.0


struct SimpleMFCStack:
    """Simple MFC stack simulation without tensors."""

    var config: SimpleMFCConfig
    var current_time: Float64
    var total_energy: Float64
    var substrate_level: Float64
    var ph_buffer_level: Float64
    var maintenance_cycles: Int

    fn __init__(out self, config: SimpleMFCConfig):
        self.config = config
        self.current_time = 0.0
        self.total_energy = 0.0
        self.substrate_level = 100.0
        self.ph_buffer_level = 100.0
        self.maintenance_cycles = 0

    fn simulate_cell(self, cell_idx: Int, dt: Float64) -> (Float64, Float64):
        """Simulate a single MFC cell."""
        # Simple cell state (C_AC, X, C_O2, eta_a, eta_c, aging_factor)
        var C_AC = 1.0 + random_float64(-0.1, 0.1)
        var X = 0.1 + random_float64(-0.01, 0.01)
        var C_O2 = 0.25 + random_float64(-0.02, 0.02)
        var eta_a = 0.01 + random_float64(-0.001, 0.001)
        var eta_c = -0.01 + random_float64(-0.001, 0.001)
        var aging_factor = max(0.5, 1.0 - self.current_time / 100000.0)

        # MFC parameters (simplified)
        var F = 96485.332
        var R = 8.314
        var T = 303.0
        var k1_0 = 0.207
        var K_AC = 0.592
        var alpha = 0.051

        # Simple Q-learning action (duty cycle)
        var duty_cycle = 0.5 + random_float64(-0.2, 0.2)
        var effective_current = duty_cycle * aging_factor

        # Calculate reaction rate
        var r1 = (
            k1_0
            * exp((alpha * F) / (R * T) * eta_a)
            * (C_AC / (K_AC + C_AC))
            * X
        )

        # Update states (simplified Euler integration)
        var new_C_AC = max(0.0, C_AC - r1 * dt * 0.1)
        var new_eta_a = max(
            -1.0, min(eta_a + effective_current * dt * 0.001, 1.0)
        )
        var new_eta_c = max(
            -1.0, min(eta_c - effective_current * dt * 0.001, 1.0)
        )

        # Calculate cell voltage and power
        var cell_voltage = new_eta_a - new_eta_c
        var cell_power = cell_voltage * effective_current

        return (cell_voltage, cell_power)

    fn simulate_step(mut self) -> (Float64, Float64):
        """Simulate one time step for all cells."""
        var total_voltage = 0.0
        var total_power = 0.0

        # Simulate each cell
        for cell_idx in range(self.config.n_cells):
            var result = self.simulate_cell(cell_idx, self.config.time_step)
            var voltage = result[0]
            var power = result[1]

            total_voltage += voltage
            total_power += power

        # Stack metrics
        var stack_voltage = total_voltage
        var stack_power = total_power

        # Update resources
        var dt_hours = self.config.time_step / 3600.0
        var substrate_consumption = stack_power * dt_hours * 0.1
        self.substrate_level = max(
            0.0, self.substrate_level - substrate_consumption
        )

        var ph_buffer_usage = dt_hours * 0.05
        self.ph_buffer_level = max(0.0, self.ph_buffer_level - ph_buffer_usage)

        # Energy accumulation
        self.total_energy += stack_power * dt_hours

        # Maintenance check
        if self.substrate_level < 20.0:
            self.substrate_level = 100.0
            self.maintenance_cycles += 1

        if self.ph_buffer_level < 20.0:
            self.ph_buffer_level = 100.0
            self.maintenance_cycles += 1

        # Update time
        self.current_time += self.config.time_step

        return (stack_voltage, stack_power)

    fn run_simulation(mut self):
        """Run the complete simulation."""
        print("=== Simple 100-Hour MFC Simulation ===")
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
        print()

        var avg_power = 0.0
        var max_power = 0.0
        var step_count = 0

        # Main simulation loop
        for step in range(total_steps):
            var result = self.simulate_step()
            var voltage = result[0]
            var power = result[1]

            # Track statistics
            avg_power += power
            if power > max_power:
                max_power = power
            step_count += 1

            # Progress reporting every 10000 steps (approximately every 3 hours)
            if step % 10000 == 0:
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
                print("  Current power:", power, "W")
                print("  Total energy:", self.total_energy, "Wh")
                print("  Substrate:", self.substrate_level, "%")
                print("  Maintenance cycles:", self.maintenance_cycles)
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
        print()

        # System summary
        var efficiency = (
            self.total_energy
            / (Float64(self.config.simulation_hours) * max_power)
            * 100.0
        )
        print("=== Performance Summary ===")
        print("Simulation completed successfully")
        print("Energy efficiency:", efficiency, "%")
        print(
            "Average power density:",
            avg_power / Float64(self.config.n_cells),
            "W/cell",
        )


# Main execution function
fn main():
    print("=== Mojo Simple MFC Simulation ===")
    print("100-hour microbial fuel cell simulation")
    print()

    # Initialize configuration
    var config = SimpleMFCConfig()

    # Create and run simulation
    var stack = SimpleMFCStack(config)
    stack.run_simulation()

    print()
    print("=== Simulation Benefits ===")
    print("- High-performance Mojo implementation")
    print("- Simplified MFC dynamics")
    print("- Resource tracking and maintenance")
    print("- Real-time performance monitoring")
    print("- Scalable cell simulation")
