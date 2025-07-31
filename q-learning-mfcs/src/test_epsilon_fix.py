#!/usr/bin/env python3
"""
Test epsilon decay fix
"""

from mfc_unified_qlearning_optimized import OptimizedMFCSimulation

def test_epsilon_decay():
    """Test that epsilon decays properly"""
    print("Testing epsilon decay fix...")

    # Create simulation with shorter duration for testing
    sim = OptimizedMFCSimulation(use_gpu=False, target_outlet_conc=12.0)
    controller = sim.unified_controller

    # Override duration for quick test
    sim.total_time = 50 * 3600  # 50 hours
    sim.num_steps = int(sim.total_time / sim.dt)

    print(f"Initial epsilon: {controller.epsilon:.6f}")

    # Simulate some Q-learning updates
    state = (1, 2, 3, 4, 5, 6)  # Dummy state

    for i in range(500):  # Simulate 500 updates
        # Simulate a reward (poor performance to trigger slower decay)
        reward = -100.0
        controller.performance_history.append(reward)

        # Update Q-table (this also updates epsilon)
        controller.update_q_table(state, 0, reward, state)

        # Print epsilon every 100 updates
        if i % 100 == 99:
            print(f"After {i+1} updates: epsilon = {controller.epsilon:.6f}")

    print(f"Final epsilon: {controller.epsilon:.6f}")
    print(f"Epsilon min: {controller.epsilon_min:.6f}")

    # Verify epsilon is decreasing and above minimum
    if controller.epsilon >= controller.epsilon_min and controller.epsilon < 0.37:
        print("✅ Epsilon decay is working correctly!")
        return True
    else:
        print("❌ Epsilon decay is still broken!")
        return False

if __name__ == "__main__":
    success = test_epsilon_decay()
    exit(0 if success else 1)
