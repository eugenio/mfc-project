from path_config import get_figure_path, get_simulation_data_path

"""
GPU-Accelerated 100-Hour MFC Simulation Runner

This script demonstrates how to run the Mojo GPU-accelerated simulation
and provides a Python fallback for demonstration purposes.
"""

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np


def run_mojo_simulation(simulation_name, mojo_file, timeout=600):
    """Run a specific Mojo simulation with timing."""
    start_time = time.time()
    results = {
        "name": simulation_name,
        "file": mojo_file,
        "success": False,
        "runtime": 0,
        "output": "",
        "error": "",
        "energy_output": 0,
        "avg_power": 0,
        "max_power": 0,
    }

    try:
        # Run Mojo simulation
        result = subprocess.run(
            ["mojo", "run", f"q-learning-mfcs/{mojo_file}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        results["runtime"] = time.time() - start_time

        if result.returncode == 0:
            results["success"] = True
            results["output"] = result.stdout

            # Parse energy output from results
            output_lines = result.stdout.split("\n")
            for line in output_lines:
                if "Total energy produced:" in line:
                    try:
                        energy_str = line.split(":")[1].strip().split(" ")[0]
                        results["energy_output"] = float(energy_str)
                    except:
                        pass
                elif "Average power:" in line:
                    try:
                        power_str = line.split(":")[1].strip().split(" ")[0]
                        results["avg_power"] = float(power_str)
                    except:
                        pass
                elif "Maximum power:" in line:
                    try:
                        power_str = line.split(":")[1].strip().split(" ")[0]
                        results["max_power"] = float(power_str)
                    except:
                        pass

        else:
            results["error"] = result.stderr

    except subprocess.TimeoutExpired:
        results["runtime"] = timeout
        results["error"] = "Timeout"
    except FileNotFoundError:
        results["error"] = "Mojo compiler not found"
    except Exception as e:
        results["runtime"] = time.time() - start_time
        results["error"] = str(e)

    return results


def run_all_mojo_simulations():
    """Run all Mojo simulations in parallel and collect results."""
    simulations = [
        ("Simple 100h MFC", "mfc_100h_simple.mojo"),
        ("Q-Learning MFC", "mfc_100h_qlearn.mojo"),
        ("Enhanced Q-Learning MFC", "mfc_100h_enhanced.mojo"),
        ("Advanced Q-Learning MFC", "mfc_100h_advanced.mojo"),
    ]

    all_results = []

    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit all simulations (but limit concurrent execution to avoid resource conflicts)
        future_to_sim = {}

        for name, file in simulations:
            future = executor.submit(run_mojo_simulation, name, file, timeout=900)
            future_to_sim[future] = (name, file)

        # Collect results as they complete
        for future in future_to_sim:
            results = future.result()
            all_results.append(results)

    return all_results


def run_accelerated_python_simulation():
    """Run an accelerated Python simulation as fallback."""
    # Simulation parameters
    n_cells = 5
    simulation_hours = 1000
    time_step = 60  # 1 minute steps for speed
    steps_per_hour = int(3600 / time_step)
    total_steps = simulation_hours * steps_per_hour

    # Initialize arrays for vectorized computation
    cell_states = np.array(
        [
            [
                1.0,
                0.05,
                1e-4,
                0.1,
                0.25,
                1e-7,
                0.05,
                0.01,
                -0.01,
                1.0,
                1.0,
            ]  # 11 state variables
            for _ in range(n_cells)
        ],
    )

    # Add slight variations between cells
    cell_states += np.random.normal(0, 0.05, cell_states.shape)
    cell_states = np.clip(cell_states, 0.001, 10.0)  # Prevent negative values

    # System parameters
    substrate_level = 100.0
    ph_buffer_level = 100.0
    maintenance_cycles = 0
    total_energy = 0.0

    # Logging arrays
    log_interval = steps_per_hour  # Log every hour
    n_log_points = simulation_hours

    performance_log = np.zeros((n_log_points, 8))

    # Q-learning parameters
    epsilon = 0.3
    epsilon_decay = 0.9995
    epsilon_min = 0.01

    # Simple Q-table (state -> action mapping)
    q_states = {}

    def discretize_state(state):
        """Simple state discretization."""
        return tuple(int(val * 10) % 10 for val in state[:5])

    def get_action(state, epsilon):
        """Epsilon-greedy action selection."""
        state_key = discretize_state(state)

        if np.random.random() < epsilon or state_key not in q_states:
            # Random action
            return np.random.uniform([0.1, 0.0, 0.0], [0.9, 1.0, 1.0])
        # Greedy action
        return q_states[state_key]

    def update_q_table(state, action, reward) -> None:
        """Update Q-table with new experience."""
        state_key = discretize_state(state)
        if state_key not in q_states:
            q_states[state_key] = action.copy()
        else:
            # Simple update
            q_states[state_key] = 0.9 * q_states[state_key] + 0.1 * action

    def compute_mfc_dynamics(states, actions, dt):
        """Vectorized MFC dynamics computation."""
        # MFC parameters
        F = 96485.332
        R = 8.314
        T = 303.0
        k1_0 = 0.207
        k2_0 = 3.288e-5
        K_AC = 0.592
        K_O2 = 0.004
        alpha = 0.051
        beta = 0.063

        # Extract states
        C_AC = states[:, 0]
        states[:, 1]
        C_H = states[:, 2]
        X = states[:, 3]
        C_O2 = states[:, 4]
        states[:, 5]
        states[:, 6]
        eta_a = states[:, 7]
        eta_c = states[:, 8]
        aging = states[:, 9]
        biofilm = states[:, 10]

        # Extract actions
        duty_cycle = actions[:, 0]
        ph_buffer = actions[:, 1]
        acetate_add = actions[:, 2]

        # Effective current
        effective_current = duty_cycle * aging

        # Reaction rates (vectorized)
        r1 = (
            k1_0
            * np.exp((alpha * F) / (R * T) * eta_a)
            * (C_AC / (K_AC + C_AC))
            * X
            * aging
            / biofilm
        )
        r2 = (
            -k2_0
            * (C_O2 / (K_O2 + C_O2))
            * np.exp((beta - 1.0) * F / (R * T) * eta_c)
            * aging
        )

        # Derivatives (simplified)
        dC_AC_dt = (1.56 + acetate_add * 0.5 - C_AC) * 0.1 - r1 * 0.01
        dX_dt = r1 * 0.001 - X * 0.0001
        dC_O2_dt = (0.3125 - C_O2) * 0.1 + r2 * 0.01
        deta_a_dt = (effective_current - r1) * 0.001
        deta_c_dt = (-effective_current - r2) * 0.001
        dC_H_dt = r1 * 0.001 - ph_buffer * C_H * 0.1

        # Update states
        states[:, 0] = np.clip(C_AC + dC_AC_dt * dt, 0.001, 5.0)
        states[:, 3] = np.clip(X + dX_dt * dt, 0.001, 2.0)
        states[:, 4] = np.clip(C_O2 + dC_O2_dt * dt, 0.001, 1.0)
        states[:, 7] = np.clip(eta_a + deta_a_dt * dt, -1.0, 1.0)
        states[:, 8] = np.clip(eta_c + deta_c_dt * dt, -1.0, 1.0)
        states[:, 2] = np.clip(C_H + dC_H_dt * dt, 1e-14, 1e-2)

        return states

    def apply_aging(states, dt_hours):
        """Apply aging effects."""
        aging_rate = 0.001 * dt_hours
        biofilm_growth = 0.0005 * dt_hours

        states[:, 9] *= 1 - aging_rate  # Aging
        states[:, 9] = np.clip(states[:, 9], 0.5, 1.0)

        states[:, 10] += biofilm_growth  # Biofilm growth
        states[:, 10] = np.clip(states[:, 10], 1.0, 2.0)

        return states

    def calculate_system_metrics(states, actions):
        """Calculate system performance metrics."""
        voltages = states[:, 7] - states[:, 8]  # eta_a - eta_c
        currents = actions[:, 0] * states[:, 9]  # duty_cycle * aging
        powers = voltages * currents

        stack_voltage = np.sum(voltages)
        stack_current = np.min(currents)
        stack_power = stack_voltage * stack_current

        return stack_voltage, stack_current, stack_power, powers

    def calculate_reward(states, actions, stack_power):
        """Calculate Q-learning reward."""
        # Power reward
        power_reward = stack_power / 5.0

        # Stability reward
        voltages = states[:, 7] - states[:, 8]
        stability_reward = 1.0 - np.std(voltages) / max(0.1, np.mean(voltages))

        # Reversal penalty
        reversal_penalty = -10.0 * np.sum(voltages < 0.1)

        # Resource penalty
        resource_penalty = -0.1 * (np.sum(actions[:, 1]) + np.sum(actions[:, 2]))

        return power_reward + stability_reward + reversal_penalty + resource_penalty

    # Main simulation loop
    start_time = time.time()

    log_idx = 0

    for step in range(total_steps):
        # Q-learning action selection
        actions = np.array(
            [get_action(cell_state, epsilon) for cell_state in cell_states],
        )

        # MFC dynamics
        cell_states = compute_mfc_dynamics(cell_states, actions, time_step)

        # Aging effects
        if step % steps_per_hour == 0:  # Apply hourly
            dt_hours = 1.0
            cell_states = apply_aging(cell_states, dt_hours)

            # Update epsilon
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # System metrics
        stack_voltage, stack_current, stack_power, cell_powers = (
            calculate_system_metrics(cell_states, actions)
        )

        # Update resources
        if step % steps_per_hour == 0:
            substrate_level -= stack_power * 0.1
            ph_buffer_level -= np.sum(actions[:, 1]) * 0.05
            total_energy += stack_power * 1.0  # Wh (hourly logging = 1 hour intervals)

            # Maintenance
            if substrate_level < 20:
                substrate_level = 100.0
                maintenance_cycles += 1
            if ph_buffer_level < 20:
                ph_buffer_level = 100.0
                maintenance_cycles += 1

        # Q-learning update
        reward = calculate_reward(cell_states, actions, stack_power)
        for i in range(n_cells):
            update_q_table(cell_states[i], actions[i], reward)

        # Logging
        if step % log_interval == 0 and log_idx < n_log_points:
            current_hour = step * time_step / 3600
            performance_log[log_idx] = [
                current_hour,
                stack_voltage,
                stack_current,
                stack_power,
                total_energy,
                substrate_level,
                ph_buffer_level,
                maintenance_cycles,
            ]
            log_idx += 1

        # Progress reporting
        if step % (total_steps // 10) == 0:
            (step / total_steps) * 100
            current_hour = step * time_step / 3600

    time.time() - start_time

    # Final analysis

    # Individual cell analysis
    for i in range(n_cells):
        voltage = cell_states[i, 7] - cell_states[i, 8]
        voltage * actions[i, 0] * cell_states[i, 9]
        cell_states[i, 9]
        cell_states[i, 10]

    # Save data to JSON
    save_performance_data(performance_log, log_idx)

    # Generate plots
    generate_plots(performance_log, log_idx)

    return performance_log[:log_idx]


def analyze_and_compare_results(all_results):
    """Analyze and compare all simulation results."""
    # Filter successful results
    successful_results = [r for r in all_results if r["success"]]
    failed_results = [r for r in all_results if not r["success"]]

    if not successful_results:
        return None

    # Summary table

    for result in all_results:
        "SUCCESS" if result["success"] else "FAILED"
        f"{result['runtime']:.1f}s" if result["success"] else "N/A"
        f"{result['energy_output']:.2f}" if result["success"] else "N/A"
        f"{result['avg_power']:.3f}" if result["success"] else "N/A"
        f"{result['max_power']:.3f}" if result["success"] else "N/A"

    # Performance analysis

    if len(successful_results) >= 2:
        # Find baseline (simple simulation)
        simple_sim = next(
            (r for r in successful_results if "Simple" in r["name"]),
            successful_results[0],
        )
        baseline_energy = simple_sim["energy_output"]
        baseline_power = simple_sim["avg_power"]

        for result in successful_results:
            if result != simple_sim:
                (
                    result["energy_output"] / baseline_energy
                    if baseline_energy > 0
                    else 0
                )
                (result["avg_power"] / baseline_power if baseline_power > 0 else 0)

    # Technology comparison

    technologies = {
        "Simple": [r for r in successful_results if "Simple" in r["name"]],
        "Q-Learning": [
            r
            for r in successful_results
            if "Q-Learning" in r["name"]
            and "Enhanced" not in r["name"]
            and "Advanced" not in r["name"]
        ],
        "Enhanced Q-Learning": [
            r for r in successful_results if "Enhanced" in r["name"]
        ],
        "Advanced Q-Learning": [
            r for r in successful_results if "Advanced" in r["name"]
        ],
    }

    for results in technologies.values():
        if results:
            result = results[0]  # Take first (should be only one)

            # Calculate efficiency metrics
            if result["energy_output"] > 0:
                result["energy_output"] / result["runtime"]

    # Generate comparison plots
    generate_comparison_plots(successful_results)

    # Failure analysis
    if failed_results:
        for result in failed_results:
            pass

    return successful_results


def generate_comparison_plots(successful_results) -> None:
    """Generate comparison visualization plots."""
    # Set up matplotlib
    import matplotlib as mpl

    mpl.use("Agg")

    if len(successful_results) < 2:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Extract data for plotting
    names = [
        r["name"].replace(" MFC", "").replace("100h ", "") for r in successful_results
    ]
    energies = [r["energy_output"] for r in successful_results]
    avg_powers = [r["avg_power"] for r in successful_results]
    max_powers = [r["max_power"] for r in successful_results]
    runtimes = [r["runtime"] for r in successful_results]

    # Energy comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(names, energies, color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"])
    ax1.set_ylabel("Total Energy Output (Wh)")
    ax1.set_title("Energy Production Comparison")
    ax1.tick_params(axis="x", rotation=45)

    # Add value labels on bars
    for _i, (bar, energy) in enumerate(zip(bars1, energies, strict=False)):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(energies) * 0.01,
            f"{energy:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Power comparison
    ax2 = axes[0, 1]
    x_pos = np.arange(len(names))
    width = 0.35
    ax2.bar(
        x_pos - width / 2,
        avg_powers,
        width,
        label="Average Power",
        color="#FF6B6B",
        alpha=0.7,
    )
    ax2.bar(
        x_pos + width / 2,
        max_powers,
        width,
        label="Peak Power",
        color="#4ECDC4",
        alpha=0.7,
    )
    ax2.set_ylabel("Power (W)")
    ax2.set_title("Power Output Comparison")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, rotation=45)
    ax2.legend()

    # Runtime comparison
    ax3 = axes[1, 0]
    bars3 = ax3.bar(names, runtimes, color=["#FFE66D", "#FF6B6B", "#4ECDC4", "#45B7D1"])
    ax3.set_ylabel("Execution Time (seconds)")
    ax3.set_title("Runtime Performance")
    ax3.tick_params(axis="x", rotation=45)

    # Add value labels
    for bar, runtime in zip(bars3, runtimes, strict=False):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(runtimes) * 0.01,
            f"{runtime:.1f}s",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Energy efficiency (Energy/Runtime ratio)
    ax4 = axes[1, 1]
    efficiency = [
        e / r if r > 0 else 0 for e, r in zip(energies, runtimes, strict=False)
    ]
    bars4 = ax4.bar(
        names,
        efficiency,
        color=["#96CEB4", "#FF6B6B", "#4ECDC4", "#45B7D1"],
    )
    ax4.set_ylabel("Energy/Time Efficiency (Wh/s)")
    ax4.set_title("Computational Efficiency")
    ax4.tick_params(axis="x", rotation=45)

    # Add value labels
    for bar, eff in zip(bars4, efficiency, strict=False):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(efficiency) * 0.01,
            f"{eff:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(
        get_figure_path("mfc_simulation_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )


def save_performance_data(performance_log, n_points):
    """Save performance data to JSON file."""
    import json

    # Prepare data for JSON serialization
    performance_data = {
        "metadata": {
            "simulation_type": "GPU-Accelerated MFC Performance Analysis",
            "total_hours": performance_log[n_points - 1, 0] if n_points > 0 else 0,
            "data_points": n_points,
            "final_energy": performance_log[n_points - 1, 4] if n_points > 0 else 0,
            "final_power": performance_log[n_points - 1, 3] if n_points > 0 else 0,
        },
        "time_series": {
            "hours": performance_log[:n_points, 0].tolist(),
            "stack_voltage": performance_log[:n_points, 1].tolist(),
            "stack_current": performance_log[:n_points, 2].tolist(),
            "stack_power": performance_log[:n_points, 3].tolist(),
            "cumulative_energy": performance_log[:n_points, 4].tolist(),
            "substrate_level": performance_log[:n_points, 5].tolist(),
            "ph_buffer_level": performance_log[:n_points, 6].tolist(),
            "maintenance_cycles": performance_log[:n_points, 7].tolist(),
        },
    }

    # Save to JSON file
    filename = get_simulation_data_path("mfc_performance_data.json")
    with open(filename, "w") as f:
        json.dump(performance_data, f, indent=2)

    return performance_data


def generate_plots(performance_log, n_points) -> None:
    """Generate visualization plots for Python simulation."""
    # Set up matplotlib
    import matplotlib as mpl

    mpl.use("Agg")

    if n_points < 2:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Subplot tags (alphabetic order: left to right, top to bottom)
    subplot_tags = ["(a)", "(b)", "(c)", "(d)"]

    hours = performance_log[:n_points, 0]

    # Power evolution
    ax1 = axes[0, 0]
    ax1.plot(hours, performance_log[:n_points, 3], "b-", linewidth=2)
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Stack Power (W)")
    ax1.set_title(f"{subplot_tags[0]} Power Evolution Over 1000 Hours")
    ax1.grid(True)

    # Cumulative energy
    ax2 = axes[0, 1]
    ax2.plot(hours, performance_log[:n_points, 4], "g-", linewidth=2)
    ax2.set_xlabel("Time (hours)")
    ax2.set_ylabel("Cumulative Energy (Wh)")
    ax2.set_title(f"{subplot_tags[1]} Total Energy Production")
    ax2.grid(True)

    # Resource levels
    ax3 = axes[1, 0]
    ax3.plot(hours, performance_log[:n_points, 5], "r-", label="Substrate", linewidth=2)
    ax3.plot(hours, performance_log[:n_points, 6], "b-", label="pH Buffer", linewidth=2)
    ax3.set_xlabel("Time (hours)")
    ax3.set_ylabel("Resource Level (%)")
    ax3.set_title(f"{subplot_tags[2]} Resource Consumption")
    ax3.legend()
    ax3.grid(True)

    # Maintenance events
    ax4 = axes[1, 1]
    ax4.step(hours, performance_log[:n_points, 7], "orange", linewidth=2, where="post")
    ax4.set_xlabel("Time (hours)")
    ax4.set_ylabel("Maintenance Events")
    ax4.set_title(f"{subplot_tags[3]} Maintenance Schedule")
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig(
        get_figure_path("mfc_100h_python_results.png"),
        dpi=300,
        bbox_inches="tight",
    )


def main() -> None:
    """Main execution function."""
    # Run all Mojo simulations in parallel
    all_results = run_all_mojo_simulations()

    # Analyze and compare results
    successful_results = analyze_and_compare_results(all_results)

    # If we have successful results, show the benefits
    if successful_results:
        # Show the best performer
        best_result = max(successful_results, key=lambda r: r["energy_output"])

        # Python target comparison
        python_target = 95.0  # Wh from original Python simulation
        achievement = best_result["energy_output"] / python_target * 100
        if achievement >= 100:
            pass
        else:
            pass

    else:
        # Run Python fallback simulation
        run_accelerated_python_simulation()


if __name__ == "__main__":
    main()
