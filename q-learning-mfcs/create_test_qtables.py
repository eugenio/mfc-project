#!/usr/bin/env python3
"""
Create test Q-table files for demonstrating the interactive Q-table analysis.

This script generates sample Q-tables with different characteristics:
- Converged Q-table (stable values)
- Training Q-table (in progress)
- Random Q-table (baseline)

Created: 2025-07-31
"""

import numpy as np
import pickle
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def create_converged_qtable(n_states=50, n_actions=4, seed=42):
    """Create a converged Q-table with stable, consistent values."""
    np.random.seed(seed)
    
    # Create a well-converged Q-table
    qtable = np.zeros((n_states, n_actions))
    
    for state in range(n_states):
        # Create distinct action values with clear preferences
        base_value = np.random.uniform(0.1, 0.8)
        
        # Best action gets highest value
        best_action = np.random.randint(0, n_actions)
        qtable[state, best_action] = base_value + np.random.uniform(0.2, 0.4)
        
        # Other actions get lower values with some variation
        for action in range(n_actions):
            if action != best_action:
                qtable[state, action] = base_value * np.random.uniform(0.3, 0.7)
    
    return qtable

def create_training_qtable(n_states=50, n_actions=4, seed=123):
    """Create a Q-table that appears to be in training (more variation)."""
    np.random.seed(seed)
    
    qtable = np.zeros((n_states, n_actions))
    
    for state in range(n_states):
        # More random values indicating ongoing learning
        if np.random.random() < 0.7:  # 70% of states have been visited
            for action in range(n_actions):
                qtable[state, action] = np.random.uniform(-0.1, 0.6)
        # Some states remain unvisited (all zeros)
    
    return qtable

def create_random_qtable(n_states=50, n_actions=4, seed=456):
    """Create a random Q-table (baseline/poor performance)."""
    np.random.seed(seed)
    
    # Completely random values
    qtable = np.random.uniform(-0.2, 0.3, (n_states, n_actions))
    
    return qtable

def create_mfc_realistic_qtable(n_states=25, n_actions=6, seed=789):
    """Create a realistic MFC Q-table with domain-specific characteristics."""
    np.random.seed(seed)
    
    qtable = np.zeros((n_states, n_actions))
    
    # MFC-specific Q-values (substrate concentration control)
    # States represent substrate concentration levels (0-50 mM in 2mM steps)
    # Actions represent control decisions (decrease large, decrease small, hold, increase small, increase medium, increase large)
    
    for state in range(n_states):
        # Substrate level mapping: state * 2  (0, 2, 4, ... 48 mM)
        # Optimal substrate range is 20-30 mM (states 10-15)
        if 10 <= state <= 15:
            # In optimal range - holding action gets highest reward
            qtable[state, 2] = np.random.uniform(0.6, 0.9)  # Hold action
            # Small adjustments get moderate rewards
            qtable[state, 1] = np.random.uniform(0.3, 0.5)  # Decrease small
            qtable[state, 3] = np.random.uniform(0.3, 0.5)  # Increase small
            # Large adjustments get negative rewards
            qtable[state, 0] = np.random.uniform(-0.2, 0.1)  # Decrease large
            qtable[state, 4] = np.random.uniform(-0.2, 0.1)  # Increase medium
            qtable[state, 5] = np.random.uniform(-0.3, -0.1)  # Increase large
        
        elif state < 10:
            # Below optimal - increase actions get higher rewards
            qtable[state, 3] = np.random.uniform(0.4, 0.7)  # Increase small
            qtable[state, 4] = np.random.uniform(0.5, 0.8)  # Increase medium
            qtable[state, 5] = np.random.uniform(0.6, 0.9)  # Increase large
            # Decrease actions get negative rewards
            qtable[state, 0] = np.random.uniform(-0.4, -0.1)  # Decrease large
            qtable[state, 1] = np.random.uniform(-0.3, 0.0)   # Decrease small
            qtable[state, 2] = np.random.uniform(0.0, 0.3)    # Hold
        
        else:
            # Above optimal - decrease actions get higher rewards
            qtable[state, 0] = np.random.uniform(0.6, 0.9)   # Decrease large
            qtable[state, 1] = np.random.uniform(0.4, 0.7)   # Decrease small
            qtable[state, 2] = np.random.uniform(0.0, 0.3)   # Hold
            # Increase actions get negative rewards
            qtable[state, 3] = np.random.uniform(-0.3, 0.0)  # Increase small
            qtable[state, 4] = np.random.uniform(-0.4, -0.1) # Increase medium
            qtable[state, 5] = np.random.uniform(-0.5, -0.2) # Increase large
    
    return qtable

def main():
    """Create test Q-table files."""
    models_dir = Path("q_learning_models")
    models_dir.mkdir(exist_ok=True)
    
    print("Creating test Q-table files...")
    
    # Create different types of Q-tables
    qtables = {
        "converged_qtable_20250131_120000.pkl": create_converged_qtable(),
        "training_qtable_20250131_115000.pkl": create_training_qtable(),
        "random_qtable_20250131_110000.pkl": create_random_qtable(),
        "mfc_realistic_qtable_20250131_130000.pkl": create_mfc_realistic_qtable(),
        "large_qtable_20250131_140000.pkl": create_converged_qtable(n_states=100, n_actions=8, seed=999)
    }
    
    for filename, qtable in qtables.items():
        filepath = models_dir / filename
        
        # Save as pickle file
        with open(filepath, 'wb') as f:
            pickle.dump(qtable, f)
        
        print(f"✅ Created {filename}: shape {qtable.shape}, non-zero values: {np.count_nonzero(qtable)}")
        
        # Show some statistics
        print(f"   - Mean Q-value: {np.mean(qtable):.3f}")
        print(f"   - Std Q-value: {np.std(qtable):.3f}")
        print(f"   - Range: [{np.min(qtable):.3f}, {np.max(qtable):.3f}]")
        print()
    
    print(f"🎉 Successfully created {len(qtables)} test Q-table files in {models_dir}/")
    print("\nYou can now use these files in the Interactive Q-Table Analysis interface!")

if __name__ == "__main__":
    main()