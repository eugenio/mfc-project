#!/usr/bin/env python3
"""
Compare Q-learning vs PID substrate control results
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Load both datasets
pid_data = pd.read_csv('data/simulation_data/sensor_mfc_simulation_data_20250725_153257.csv')
qlearning_data = pd.read_csv('data/simulation_data/mfc_recirculation_control_100h_validated_20250725_160622.csv')

# Create comparison plot
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Q-Learning vs PID Substrate Control Comparison', fontsize=16, fontweight='bold')

# Substrate concentration comparison
ax1 = axes[0, 0]
ax1.plot(pid_data['time_hours'], pid_data['substrate_concentrations'], 
         label='PID Controller', color='red', linewidth=2, alpha=0.8)
ax1.plot(qlearning_data['time_hours'], qlearning_data['reservoir_concentration'], 
         label='Q-Learning Controller', color='blue', linewidth=2, alpha=0.8)
ax1.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Target (20 mM)')
ax1.axhline(y=25, color='orange', linestyle='--', alpha=0.7, label='Max Threshold (25 mM)')
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Substrate Concentration (mM)')
ax1.set_title('Reservoir Substrate Concentration')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Power output comparison
ax2 = axes[0, 1]
ax2.plot(pid_data['time_hours'], pid_data['stack_power'] * 1000, 
         label='PID Controller', color='red', linewidth=2, alpha=0.8)
ax2.plot(qlearning_data['time_hours'], qlearning_data['total_power'] * 1000, 
         label='Q-Learning Controller', color='blue', linewidth=2, alpha=0.8)
ax2.set_xlabel('Time (hours)')
ax2.set_ylabel('Power Output (mW)')
ax2.set_title('Stack Power Output')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Substrate addition rates (estimate for PID from concentration change)
ax3 = axes[1, 0]
# Calculate PID addition rate from concentration changes
pid_additions = np.diff(pid_data['substrate_concentrations'], prepend=pid_data['substrate_concentrations'].iloc[0])
pid_additions = np.maximum(pid_additions, 0)  # Only positive additions

ax3.plot(pid_data['time_hours'], pid_additions, 
         label='PID Controller (estimated)', color='red', linewidth=2, alpha=0.8)
ax3.plot(qlearning_data['time_hours'], qlearning_data['substrate_addition_rate'], 
         label='Q-Learning Controller', color='blue', linewidth=2, alpha=0.8)
ax3.set_xlabel('Time (hours)')
ax3.set_ylabel('Substrate Addition Rate (mM/h)')
ax3.set_title('Substrate Addition Control')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Summary statistics
ax4 = axes[1, 1]
ax4.axis('off')

# Calculate summary stats
pid_max_conc = pid_data['substrate_concentrations'].max()
pid_final_conc = pid_data['substrate_concentrations'].iloc[-1]
pid_avg_power = pid_data['stack_power'].mean() * 1000

ql_max_conc = qlearning_data['reservoir_concentration'].max()
ql_final_conc = qlearning_data['reservoir_concentration'].iloc[-1]
ql_avg_power = qlearning_data['total_power'].mean() * 1000
ql_total_added = qlearning_data['substrate_addition_rate'].sum() * 0.00278  # Convert to total mmol

summary_text = f"""
Performance Summary (100 hours):

PID Controller:
• Max concentration: {pid_max_conc:.1f} mM
• Final concentration: {pid_final_conc:.1f} mM  
• Average power: {pid_avg_power:.2f} mW
• Control: Uncontrolled increase

Q-Learning Controller:
• Max concentration: {ql_max_conc:.1f} mM
• Final concentration: {ql_final_conc:.1f} mM
• Average power: {ql_avg_power:.2f} mW
• Total substrate added: {ql_total_added:.1f} mmol
• Control: Learned optimization

Improvement:
• Peak reduction: {((pid_max_conc - ql_max_conc)/pid_max_conc*100):.1f}%
• Better target control
• Eliminated runaway additions
• Reward-based learning
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

plt.tight_layout()
plt.savefig('data/figures/qlearning_vs_pid_comparison.png', dpi=300, bbox_inches='tight')
print("Comparison plot saved to: data/figures/qlearning_vs_pid_comparison.png")
# plt.show()  # Commented out for headless environment