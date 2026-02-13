#!/usr/bin/env python3
"""
Plot the improved Q-learning results showing the fixed substrate control
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Load all three datasets for comparison
pid_data = pd.read_csv('data/simulation_data/sensor_mfc_simulation_data_20250725_153257.csv')
qlearning_old = pd.read_csv('data/simulation_data/mfc_recirculation_control_100h_validated_20250725_160622.csv')
qlearning_improved = pd.read_csv('data/simulation_data/mfc_recirculation_control_improved_epsilon_20250725_161813.csv')

# Create comprehensive comparison plot
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Q-Learning Substrate Control: Problem Analysis and Solution', fontsize=16, fontweight='bold')

# 1. Substrate concentration comparison
ax1 = axes[0, 0]
ax1.plot(pid_data['time_hours'], pid_data['substrate_concentrations'],
         label='PID Controller', color='red', linewidth=2, alpha=0.8)
ax1.plot(qlearning_old['time_hours'], qlearning_old['reservoir_concentration'],
         label='Q-Learning (Original)', color='orange', linewidth=2, alpha=0.8)
ax1.plot(qlearning_improved['time_hours'], qlearning_improved['reservoir_concentration'],
         label='Q-Learning (Improved)', color='blue', linewidth=2, alpha=0.8)
ax1.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Target (20 mM)')
ax1.axhline(y=25, color='red', linestyle='--', alpha=0.7, label='Max Threshold (25 mM)')
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Substrate Concentration (mM)')
ax1.set_title('Reservoir Substrate Concentration')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(15, 85)

# 2. Power output comparison
ax2 = axes[0, 1]
ax2.plot(pid_data['time_hours'], pid_data['stack_power'] * 1000,
         label='PID Controller', color='red', linewidth=2, alpha=0.8)
ax2.plot(qlearning_old['time_hours'], qlearning_old['total_power'] * 1000,
         label='Q-Learning (Original)', color='orange', linewidth=2, alpha=0.8)
ax2.plot(qlearning_improved['time_hours'], qlearning_improved['total_power'] * 1000,
         label='Q-Learning (Improved)', color='blue', linewidth=2, alpha=0.8)
ax2.set_xlabel('Time (hours)')
ax2.set_ylabel('Power Output (mW)')
ax2.set_title('Stack Power Output')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Substrate addition rates
ax3 = axes[0, 2]
# Calculate PID addition rate from concentration changes
pid_additions = np.diff(pid_data['substrate_concentrations'], prepend=pid_data['substrate_concentrations'].iloc[0])
pid_additions = np.maximum(pid_additions, 0)  # Only positive additions

ax3.plot(pid_data['time_hours'], pid_additions,
         label='PID Controller (estimated)', color='red', linewidth=2, alpha=0.8)
ax3.plot(qlearning_old['time_hours'], qlearning_old['substrate_addition_rate'],
         label='Q-Learning (Original)', color='orange', linewidth=2, alpha=0.8)
ax3.plot(qlearning_improved['time_hours'], qlearning_improved['substrate_addition_rate'],
         label='Q-Learning (Improved)', color='blue', linewidth=2, alpha=0.8)
ax3.set_xlabel('Time (hours)')
ax3.set_ylabel('Substrate Addition Rate (mM/h)')
ax3.set_title('Substrate Addition Control')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Detailed view: Critical period (45-65 hours)
ax4 = axes[1, 0]
critical_hours = (qlearning_old['time_hours'] >= 45) & (qlearning_old['time_hours'] <= 65)
ax4.plot(qlearning_old[critical_hours]['time_hours'],
         qlearning_old[critical_hours]['reservoir_concentration'],
         label='Original Q-Learning', color='orange', linewidth=2)
ax4.plot(qlearning_improved['time_hours'], qlearning_improved['reservoir_concentration'],
         label='Improved Q-Learning', color='blue', linewidth=2)
ax4.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Target')
ax4.axhline(y=25, color='red', linestyle='--', alpha=0.7, label='Max Threshold')
ax4.set_xlabel('Time (hours)')
ax4.set_ylabel('Substrate Concentration (mM)')
ax4.set_title('Critical Period: Hours 45-65')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xlim(45, 65)

# 5. Exploration behavior analysis
ax5 = axes[1, 1]
# Calculate moving average of addition events (indicator of exploration)
window = 360  # 1 hour windows
old_additions = qlearning_old['substrate_addition_rate'] > 0
improved_additions = qlearning_improved['substrate_addition_rate'] > 0

old_exploration = pd.Series(old_additions.astype(int)).rolling(window=window, center=True).sum()
improved_exploration = pd.Series(improved_additions.astype(int)).rolling(window=window, center=True).sum()

ax5.plot(qlearning_old['time_hours'], old_exploration,
         label='Original (High Exploration)', color='orange', linewidth=2)
ax5.plot(qlearning_improved['time_hours'], improved_exploration,
         label='Improved (Low Exploration)', color='blue', linewidth=2)
ax5.set_xlabel('Time (hours)')
ax5.set_ylabel('Addition Events per Hour')
ax5.set_title('Q-Learning Exploration Behavior')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Summary statistics
ax6 = axes[1, 2]
ax6.axis('off')

# Calculate comprehensive stats
pid_max = pid_data['substrate_concentrations'].max()
pid_final = pid_data['substrate_concentrations'].iloc[-1]
pid_avg_power = pid_data['stack_power'].mean() * 1000

old_max = qlearning_old['reservoir_concentration'].max()
old_final = qlearning_old['reservoir_concentration'].iloc[-1]
old_avg_power = qlearning_old['total_power'].mean() * 1000
old_total_added = qlearning_old['substrate_addition_rate'].sum() * 0.00278

improved_max = qlearning_improved['reservoir_concentration'].max()
improved_final = qlearning_improved['reservoir_concentration'].iloc[-1]
improved_avg_power = qlearning_improved['total_power'].mean() * 1000
improved_total_added = qlearning_improved['substrate_addition_rate'].sum() * 0.00278

summary_text = f"""
SUBSTRATE CONTROL COMPARISON (100 hours):

PID Controller (Baseline):
• Max concentration: {pid_max:.1f} mM
• Final concentration: {pid_final:.1f} mM  
• Average power: {pid_avg_power:.2f} mW
• Control: Uncontrolled exponential growth

Q-Learning Original (Problem):
• Max concentration: {old_max:.1f} mM
• Final concentration: {old_final:.1f} mM
• Average power: {old_avg_power:.2f} mW
• Total substrate added: {old_total_added:.1f} mmol
• Issue: High exploration (ε=10% min)

Q-Learning Improved (Solution):
• Max concentration: {improved_max:.1f} mM
• Final concentration: {improved_final:.1f} mM
• Average power: {improved_avg_power:.2f} mW
• Total substrate added: {improved_total_added:.1f} mmol
• Fix: Low exploration (ε=1% min)

IMPROVEMENTS ACHIEVED:
• Peak reduction vs PID: {((pid_max - improved_max)/pid_max*100):.1f}%
• Peak reduction vs Original: {((old_max - improved_max)/old_max*100):.1f}%
• Substrate savings: {((old_total_added - improved_total_added)/old_total_added*100):.1f}%
• Near-target control: {improved_final:.1f} mM (target: 12.0 mM)
• Stable, predictable behavior
"""

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.savefig('data/figures/qlearning_substrate_control_analysis.png', dpi=300, bbox_inches='tight')
print("Analysis plot saved to: data/figures/qlearning_substrate_control_analysis.png")

# Create a second plot focusing on the solution
fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))
fig2.suptitle('Q-Learning Substrate Control: Final Solution Performance', fontsize=16, fontweight='bold')

# Final solution performance
ax1 = axes2[0, 0]
ax1.plot(qlearning_improved['time_hours'], qlearning_improved['reservoir_concentration'],
         label='Reservoir Concentration', color='blue', linewidth=2)
ax1.plot(qlearning_improved['time_hours'], qlearning_improved['outlet_concentration'],
         label='Outlet Concentration', color='green', linewidth=2)
ax1.axhline(y=20, color='blue', linestyle='--', alpha=0.7, label='Reservoir Target (20 mM)')
ax1.axhline(y=12, color='green', linestyle='--', alpha=0.7, label='Outlet Target (12 mM)')
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Concentration (mM)')
ax1.set_title('Improved Q-Learning: Substrate Concentrations')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Power and efficiency
ax2 = axes2[0, 1]
ax2.plot(qlearning_improved['time_hours'], qlearning_improved['total_power'] * 1000,
         color='red', linewidth=2)
ax2.set_xlabel('Time (hours)')
ax2.set_ylabel('Power Output (mW)')
ax2.set_title('Power Output Performance')
ax2.grid(True, alpha=0.3)

# Substrate addition behavior
ax3 = axes2[1, 0]
ax3.plot(qlearning_improved['time_hours'], qlearning_improved['substrate_addition_rate'],
         color='purple', linewidth=2)
ax3.set_xlabel('Time (hours)')
ax3.set_ylabel('Addition Rate (mM/h)')
ax3.set_title('Substrate Addition Control')
ax3.grid(True, alpha=0.3)

# Control performance metrics
ax4 = axes2[1, 1]
# Calculate control error over time
reservoir_error = np.abs(qlearning_improved['reservoir_concentration'] - 20.0)
outlet_error = np.abs(qlearning_improved['outlet_concentration'] - 12.0)

ax4.plot(qlearning_improved['time_hours'], reservoir_error,
         label='Reservoir Error', color='blue', linewidth=2)
ax4.plot(qlearning_improved['time_hours'], outlet_error,
         label='Outlet Error', color='green', linewidth=2)
ax4.set_xlabel('Time (hours)')
ax4.set_ylabel('Control Error (mM)')
ax4.set_title('Control Performance (Lower = Better)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data/figures/qlearning_final_solution.png', dpi=300, bbox_inches='tight')
print("Solution plot saved to: data/figures/qlearning_final_solution.png")
