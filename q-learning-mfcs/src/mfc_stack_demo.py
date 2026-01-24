"""Comprehensive MFC Stack Demonstration.

This demo shows:
1. Normal operation with Q-learning control
2. Cell reversal prevention
3. pH buffer management
4. Acetate addition for extended operation
5. Power stability maintenance
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from mfc_stack_simulation import MFCStack, MFCStackQLearningController


def run_comprehensive_demo():
    """Run comprehensive MFC stack demonstration."""
    # Initialize system
    stack = MFCStack()
    controller = MFCStackQLearningController(stack)

    # Demonstration phases
    phases = [
        {
            "name": "Initialization",
            "duration": 200,
            "description": "System startup and stabilization",
        },
        {
            "name": "Normal Operation",
            "duration": 300,
            "description": "Optimal power generation",
        },
        {
            "name": "Disturbance Recovery",
            "duration": 200,
            "description": "Recovery from substrate depletion",
        },
        {
            "name": "pH Management",
            "duration": 200,
            "description": "pH buffer control demonstration",
        },
        {
            "name": "Long-term Operation",
            "duration": 300,
            "description": "Extended operation with acetate addition",
        },
    ]

    current_step = 0
    phase_results = []

    for _phase_num, phase in enumerate(phases):
        phase_start_time = time.time()
        phase_data = {
            "name": phase["name"],
            "start_step": current_step,
            "end_step": current_step + phase["duration"],
            "powers": [],
            "voltages": [],
            "reversals": [],
            "rewards": [],
        }

        # Phase-specific initialization
        if phase["name"] == "Disturbance Recovery":
            # Simulate substrate depletion
            for cell in stack.cells:
                cell.state[0] *= 0.3  # Reduce acetate concentration

        elif phase["name"] == "pH Management":
            # Simulate pH disturbance
            for cell in stack.cells:
                cell.state[2] *= 10  # Increase H+ concentration (lower pH)

        elif phase["name"] == "Long-term Operation":
            # Enable acetate addition
            controller.enable_acetate_addition = True

        # Run phase
        for step in range(phase["duration"]):
            reward, power = controller.train_step()

            # Collect phase data
            phase_data["powers"].append(power)
            phase_data["voltages"].append(stack.stack_voltage)
            phase_data["reversals"].append(
                sum(1 for cell in stack.cells if cell.is_reversed),
            )
            phase_data["rewards"].append(reward)

            current_step += 1

            # Progress updates
            if step % 50 == 0:
                stack.check_system_health()

        time.time() - phase_start_time
        phase_results.append(phase_data)

        # Phase summary
        np.mean(phase_data["powers"])
        np.mean(phase_data["voltages"])
        max(phase_data["reversals"])
        np.mean(phase_data["rewards"][-10:])

    # Final analysis

    stack.check_system_health()

    # Individual cell status
    for _i, cell in enumerate(stack.cells):
        cell.get_sensor_readings()
        power = cell.get_power()
        cell.actuators["duty_cycle"].get_value()
        cell.actuators["ph_buffer"].get_value()
        cell.actuators["acetate_pump"].get_value()

    # Control system performance

    # Generate comprehensive plots
    plot_comprehensive_results(stack, controller, phase_results)

    return stack, controller, phase_results


def plot_comprehensive_results(stack, controller, phase_results) -> None:
    """Generate comprehensive result plots."""
    # Set up matplotlib
    import matplotlib as mpl

    mpl.use("Agg")

    fig, axes = plt.subplots(4, 2, figsize=(16, 20))

    # Phase boundaries for vertical lines
    phase_boundaries = [0]
    for phase in phase_results:
        phase_boundaries.append(phase["end_step"])

    # 1. Stack Power Evolution
    ax1 = axes[0, 0]
    ax1.plot(stack.data_log["time"], stack.data_log["stack_power"], "b-", linewidth=2)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Stack Power (W)")
    ax1.set_title("Stack Power Evolution Through Phases")
    ax1.grid(True)

    # Add phase boundaries
    for i, boundary in enumerate(phase_boundaries[1:-1], 1):
        ax1.axvline(x=boundary, color="r", linestyle="--", alpha=0.5)
        if i < len(phase_results):
            ax1.text(
                boundary + 10,
                ax1.get_ylim()[1] * 0.9,
                phase_results[i - 1]["name"],
                rotation=90,
                fontsize=8,
            )

    # 2. Individual Cell Voltages
    ax2 = axes[0, 1]
    colors = ["blue", "green", "red", "orange", "purple"]
    for i in range(5):
        ax2.plot(
            stack.data_log["time"],
            stack.data_log["cell_voltages"][i],
            color=colors[i],
            label=f"Cell {i}",
            alpha=0.8,
        )
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Cell Voltage (V)")
    ax2.set_title("Individual Cell Voltages")
    ax2.legend()
    ax2.grid(True)

    # 3. Cell Reversal Status
    ax3 = axes[1, 0]
    reversal_data = np.array([stack.data_log["cell_reversals"][i] for i in range(5)]).T
    ax3.imshow(reversal_data.T, aspect="auto", cmap="RdYlBu_r", interpolation="nearest")
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Cell Number")
    ax3.set_title("Cell Reversal Status (Red = Reversed)")
    ax3.set_yticks(range(5))
    ax3.set_yticklabels([f"Cell {i}" for i in range(5)])

    # 4. Duty Cycle Control
    ax4 = axes[1, 1]
    for i in range(5):
        ax4.plot(
            stack.data_log["time"],
            stack.data_log["duty_cycles"][i],
            color=colors[i],
            label=f"Cell {i}",
            alpha=0.8,
        )
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Duty Cycle")
    ax4.set_title("Duty Cycle Control")
    ax4.legend()
    ax4.grid(True)

    # 5. pH Buffer Usage
    ax5 = axes[2, 0]
    for i in range(5):
        ax5.plot(
            stack.data_log["time"],
            stack.data_log["ph_buffers"][i],
            color=colors[i],
            label=f"Cell {i}",
            alpha=0.8,
        )
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("pH Buffer Usage")
    ax5.set_title("pH Buffer Control")
    ax5.legend()
    ax5.grid(True)

    # 6. Acetate Addition
    ax6 = axes[2, 1]
    for i in range(5):
        ax6.plot(
            stack.data_log["time"],
            stack.data_log["acetate_additions"][i],
            color=colors[i],
            label=f"Cell {i}",
            alpha=0.8,
        )
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("Acetate Addition")
    ax6.set_title("Acetate Addition Control")
    ax6.legend()
    ax6.grid(True)

    # 7. Q-Learning Rewards
    ax7 = axes[3, 0]
    rewards = list(controller.reward_history)
    ax7.plot(rewards, alpha=0.3, color="blue", label="Raw Rewards")

    # Moving average
    if len(rewards) > 20:
        window_size = 20
        moving_avg = np.convolve(
            rewards,
            np.ones(window_size) / window_size,
            mode="valid",
        )
        ax7.plot(
            range(window_size - 1, len(rewards)),
            moving_avg,
            "r-",
            linewidth=2,
            label="Moving Average",
        )

    ax7.set_xlabel("Training Step")
    ax7.set_ylabel("Reward")
    ax7.set_title("Q-Learning Training Progress")
    ax7.legend()
    ax7.grid(True)

    # 8. System Health Metrics
    ax8 = axes[3, 1]

    # Calculate system health over time
    health_data = {"efficiency": [], "stability": [], "active_cells": []}

    for i in range(0, len(stack.data_log["time"]), 10):  # Sample every 10 steps
        # Calculate efficiency
        power = stack.data_log["stack_power"][i]
        voltage = stack.data_log["stack_voltage"][i]
        efficiency = power / max(0.1, voltage * 5.0)
        health_data["efficiency"].append(efficiency)

        # Calculate stability
        if i >= 10:
            recent_powers = stack.data_log["stack_power"][max(0, i - 10) : i]
            stability = 1.0 - np.std(recent_powers) / max(0.1, np.mean(recent_powers))
        else:
            stability = 1.0
        health_data["stability"].append(stability)

        # Calculate active cells
        active_cells = 5 - sum(stack.data_log["cell_reversals"][j][i] for j in range(5))
        health_data["active_cells"].append(active_cells)

    time_samples = stack.data_log["time"][::10]
    ax8.plot(
        time_samples,
        health_data["efficiency"],
        "b-",
        label="Efficiency",
        linewidth=2,
    )
    ax8.plot(
        time_samples,
        health_data["stability"],
        "g-",
        label="Stability",
        linewidth=2,
    )
    ax8.plot(
        time_samples,
        np.array(health_data["active_cells"]) / 5,
        "r-",
        label="Active Cells Ratio",
        linewidth=2,
    )

    ax8.set_xlabel("Time (s)")
    ax8.set_ylabel("Metric Value")
    ax8.set_title("System Health Metrics")
    ax8.legend()
    ax8.grid(True)

    plt.tight_layout()
    plt.savefig("mfc_stack_comprehensive.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    try:
        stack, controller, phase_results = run_comprehensive_demo()

    except KeyboardInterrupt:
        pass
    except Exception:
        import traceback

        traceback.print_exc()
