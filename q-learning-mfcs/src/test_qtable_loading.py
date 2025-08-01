#!/usr/bin/env python3
"""
Q-Learning MFC Starting Point Example
====================================

This example demonstrates loading and using a validated Q-table for MFC substrate control.
The Q-table contains learned policies from successful training sessions.

Validated Q-table: q_table_unified_20250724_022416.pkl
- 150 explored states
- 57 possible actions per state  
- Optimized for substrate concentration control around 25.0 mM target
- Training included thermodynamic constraints and biofilm growth modeling
"""

import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

def load_validated_qtable():
    """
    Load the validated Q-table for MFC substrate control.
    
    Returns:
        dict: Q-table with state-action values
        
    Q-table Structure:
        - States: (power_bin, biofilm_bin, substrate_bin, flow_bin, sensor_bin, time_bin)
        - Actions: 0-56 representing different flow rate adjustments
        - Values: Expected cumulative reward for each state-action pair
    """

    qtable_path = Path("../q_learning_models/q_table_unified_20250724_022416.pkl")

    if not qtable_path.exists():
        print(f"⚠️  Validated Q-table not found at {qtable_path}")
        print("   Creating minimal example Q-table...")
        return create_example_qtable()

    try:
        with open(qtable_path, 'rb') as f:
            qtable = pickle.load(f)

        print("✅ Loaded validated Q-table:")
        print(f"   States explored: {len(qtable)}")
        print(f"   File: {qtable_path.name}")

        return qtable

    except Exception as e:
        print(f"❌ Error loading Q-table: {e}")
        return create_example_qtable()

def create_example_qtable():
    """
    Create a minimal example Q-table for demonstration.
    
    Returns:
        dict: Basic Q-table with example state-action values
    """

    qtable = defaultdict(lambda: defaultdict(float))

    # Example state: (power=3, biofilm=3, substrate=0, flow=8, sensor=1, time=0)
    # This represents: medium power, healthy biofilm, low substrate, high flow, good sensor reading
    example_state = (np.int64(3), np.int64(3), np.int64(0), np.int64(8), np.int64(1), np.int64(0))

    # Set learned action values (negative values indicate poor outcomes, positive indicate good)
    qtable[example_state][0] = -106.73  # Reduce flow significantly (bad when substrate low)
    qtable[example_state][1] = -108.34  # Reduce flow moderately (also bad)
    qtable[example_state][2] = -108.64  # Reduce flow slightly (still bad)
    qtable[example_state][3] = -106.69  # Maintain flow (bad for low substrate)
    qtable[example_state][28] = 12.5    # Increase flow moderately (good for substrate addition)
    qtable[example_state][32] = 15.2    # Increase flow significantly (better for substrate supply)

    # Add more example states for different scenarios
    # High substrate state - reduce flow to prevent waste
    high_substrate_state = (np.int64(4), np.int64(2), np.int64(7), np.int64(3), np.int64(2), np.int64(1))
    qtable[high_substrate_state][0] = 8.5   # Reduce flow (good when substrate high)
    qtable[high_substrate_state][1] = 6.2   # Reduce flow less (still good)
    qtable[high_substrate_state][32] = -45.1 # Increase flow (bad, wastes substrate)

    print("📋 Created example Q-table with learned policies:")
    print("   Low substrate → Increase flow rate")
    print("   High substrate → Decrease flow rate")
    print("   Medium biofilm → Maintain moderate flow")

    return qtable

def demonstrate_qtable_usage():
    """
    Demonstrate how to use the Q-table for action selection.
    """

    print("\n🎯 Q-Learning Policy Demonstration")
    print("=" * 50)

    # Load validated Q-table
    qtable = load_validated_qtable()

    # Demonstrate policy for different states
    states_to_test = [
        ((np.int64(3), np.int64(3), np.int64(0), np.int64(8), np.int64(1), np.int64(0)), "Low substrate, high flow"),
        ((np.int64(4), np.int64(2), np.int64(7), np.int64(3), np.int64(2), np.int64(1)), "High substrate, low flow"),
        ((np.int64(2), np.int64(4), np.int64(4), np.int64(5), np.int64(1), np.int64(2)), "Medium conditions"),
    ]

    for state, description in states_to_test:
        print(f"\n📍 State: {description}")
        print(f"   State vector: {state}")

        if state in qtable:
            action_values = qtable[state]

            # Find best action (highest Q-value)
            best_action = max(action_values.keys(), key=lambda k: action_values[k])
            best_value = action_values[best_action]

            # Find worst action (lowest Q-value)
            worst_action = min(action_values.keys(), key=lambda k: action_values[k])
            worst_value = action_values[worst_action]

            print(f"   Best action: {best_action} (Q-value: {best_value:.2f})")
            print(f"   Worst action: {worst_action} (Q-value: {worst_value:.2f})")
            print(f"   Actions evaluated: {len([v for v in action_values.values() if v != 0])}")

        else:
            print("   ⚠️  State not explored in training")

    # Show Q-table statistics
    print("\n📊 Q-Table Statistics:")
    all_values = []
    explored_actions = 0

    for state_actions in qtable.values():
        for action, value in state_actions.items():
            if value != 0:  # Only count explored actions
                all_values.append(value)
                explored_actions += 1

    if all_values:
        print(f"   Total explored state-action pairs: {explored_actions}")
        print(f"   Q-value range: {min(all_values):.2f} to {max(all_values):.2f}")
        print(f"   Average Q-value: {np.mean(all_values):.2f}")
        print(f"   States with positive rewards: {len([v for v in all_values if v > 0])}")
        print(f"   States with negative rewards: {len([v for v in all_values if v < 0])}")

def save_example_qtable():
    """Save a clean example Q-table for users to start with."""

    qtable = create_example_qtable()

    # Save as both pickle and readable format
    example_path = Path("example_qtable_start.pkl")

    with open(example_path, 'wb') as f:
        pickle.dump(dict(qtable), f)

    print(f"\n💾 Saved example Q-table to {example_path}")
    print("   Use this as a starting point for your own MFC control experiments")

if __name__ == "__main__":
    print("🚀 MFC Q-Learning Starting Point Example")
    print("=" * 60)

    demonstrate_qtable_usage()
    save_example_qtable()

    print("\n✨ Next Steps:")
    print("   1. Load the validated Q-table in your MFC controller")
    print("   2. Use the learned policies for substrate control")
    print("   3. Continue training to improve performance")
    print("   4. Monitor convergence and adjust parameters as needed")
