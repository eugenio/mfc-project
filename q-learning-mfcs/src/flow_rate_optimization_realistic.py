#!/usr/bin/env python3
"""
Realistic Flow Rate Optimization for MFC Stack
Considers practical constraints and realistic power generation
"""

import matplotlib
import numpy as np

matplotlib.use('Agg')
from dataclasses import dataclass

import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from path_config import get_figure_path


@dataclass
class RealisticMFCParameters:
    """Realistic MFC system parameters"""
    # Physical parameters
    V_a: float = 5.5e-5      # Anodic chamber volume (m³)
    A_m: float = 5.0e-4      # Membrane area (m²)
    n_cells: float = 5       # Number of cells

    # Kinetic parameters from literature
    k1_0: float = 0.207      # Anodic rate constant
    K_AC: float = 0.592      # Acetate half-saturation (mol/m³)
    Y_ac: float = 0.05       # Biomass yield
    K_dec: float = 8.33e-4   # Biomass decay (s⁻¹)

    # Realistic power parameters
    V_cell_max: float = 0.8  # Maximum cell voltage (V)
    V_cell_typical: float = 0.5  # Typical operating voltage (V)
    i_max: float = 10.0      # Maximum current density (A/m²)

    # Operational parameters
    C_AC_in: float = 1.56    # Inlet acetate (mol/m³)
    T: float = 303           # Temperature (K)
    F: float = 96485         # Faraday constant
    R: float = 8.314         # Gas constant

    # Biofilm parameters
    tau_biofilm: float = 24 * 3600  # Biofilm doubling time (s)
    biofilm_max: float = 2.0        # Max thickness factor
    biofilm_k: float = 0.5          # Half-saturation for biofilm effect

class RealisticFlowOptimizer:
    def __init__(self, params: RealisticMFCParameters = RealisticMFCParameters()):
        self.params = params

    def residence_time(self, Q_a: float) -> float:
        """Hydraulic residence time (seconds)"""
        return self.params.V_a / Q_a

    def biofilm_factor(self, tau: float, operation_time: float = 100*3600) -> float:
        """Calculate biofilm thickness factor"""
        p = self.params
        # Logistic growth model
        growth_rate = np.log(2) / p.tau_biofilm
        biofilm = 1 + (p.biofilm_max - 1) * (1 - np.exp(-growth_rate * tau))
        return min(biofilm, p.biofilm_max)

    def steady_state_concentrations(self, Q_a: float, biofilm: float = 1.0):
        """Calculate steady-state concentrations and biomass"""
        p = self.params
        tau = self.residence_time(Q_a)

        # Simplified steady-state biomass (g/L)
        X_max = 2.0  # Maximum biomass concentration
        X = X_max * (1 - np.exp(-tau / 3600))  # Growth with residence time
        X = X / biofilm  # Reduced effectiveness with thick biofilm

        # Steady-state acetate from mass balance
        # Monod kinetics with biofilm limitation
        mu_max = 0.5  # Maximum specific growth rate (h⁻¹)
        r_acetate_max = mu_max * X / p.Y_ac / 3600  # mol/m³/s

        # Solve for steady-state acetate
        # Q(C_in - C) = V * r_max * C/(K + C) / biofilm
        a = Q_a
        b = Q_a * p.K_AC + p.V_a * r_acetate_max / biofilm - Q_a * p.C_AC_in
        c = -Q_a * p.K_AC * p.C_AC_in

        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            C_AC = p.C_AC_in * 0.1  # Fallback
        else:
            C_AC = (-b + np.sqrt(discriminant)) / (2*a)
            C_AC = max(0.01, min(C_AC, p.C_AC_in))

        return C_AC, X

    def calculate_power(self, Q_a: float, C_AC: float, X: float, biofilm: float) -> float:
        """Calculate realistic power output"""
        p = self.params

        # Current density from Butler-Volmer kinetics
        # Simplified: i = i0 * (C_AC/K_AC) * X * f(biofilm)
        i0 = 0.1  # Exchange current density (A/m²)
        biofilm_effect = p.biofilm_k / (p.biofilm_k + (biofilm - 1))

        i = i0 * (C_AC / (p.K_AC + C_AC)) * X * biofilm_effect
        i = min(i, p.i_max)  # Limit to maximum

        # Cell voltage considering losses
        # V = OCV - losses
        OCV = 0.8  # Open circuit voltage
        R_int = 0.01  # Internal resistance (Ω·m²)
        V_cell = OCV - i * R_int
        V_cell = max(0, min(V_cell, p.V_cell_max))

        # Total stack power
        power = p.n_cells * p.A_m * i * V_cell
        return power

    def substrate_efficiency(self, C_AC: float) -> float:
        """Calculate substrate conversion efficiency"""
        return (self.params.C_AC_in - C_AC) / self.params.C_AC_in

    def objective_function(self, Q_a: float) -> float:
        """Multi-objective function"""
        # Practical flow rate bounds
        if Q_a < 1e-6 or Q_a > 5e-5:  # 0.0036 to 180 mL/h
            return 1e10

        tau = self.residence_time(Q_a)
        biofilm = self.biofilm_factor(tau)
        C_AC, X = self.steady_state_concentrations(Q_a, biofilm)

        power = self.calculate_power(Q_a, C_AC, X, biofilm)
        efficiency = self.substrate_efficiency(C_AC)
        consumption = Q_a * (self.params.C_AC_in - C_AC)  # mol/s

        # Objectives with practical weights
        obj_power = power / 0.1  # Normalize to ~0.1W expected
        obj_efficiency = efficiency if efficiency > 0.8 else efficiency**2
        obj_consumption = consumption / 1e-6  # Normalize to μmol/s
        obj_biofilm = 1.0 / biofilm  # Minimize biofilm

        # Penalty for very short residence time
        tau_penalty = 0 if tau > 60 else (60 - tau)**2 / 3600

        # Combined objective
        objective = (
            5.0 * obj_power +
            10.0 * obj_efficiency +
            2.0 * obj_consumption +
            3.0 * obj_biofilm -
            tau_penalty
        )

        return -objective

    def optimize(self):
        """Find optimal flow rate"""
        # Search range
        Q_min = 1e-6  # 3.6 mL/h
        Q_max = 5e-5  # 180 mL/h

        result = minimize_scalar(
            self.objective_function,
            bounds=(Q_min, Q_max),
            method='bounded'
        )

        Q_opt = result.x
        tau = self.residence_time(Q_opt)
        biofilm = self.biofilm_factor(tau)
        C_AC, X = self.steady_state_concentrations(Q_opt, biofilm)
        power = self.calculate_power(Q_opt, C_AC, X, biofilm)
        efficiency = self.substrate_efficiency(C_AC)
        consumption = Q_opt * (self.params.C_AC_in - C_AC)

        return {
            'Q_opt': Q_opt,
            'Q_opt_mL_h': Q_opt * 3.6e6,
            'tau_min': tau / 60,
            'C_AC': C_AC,
            'X': X,
            'biofilm': biofilm,
            'power': power,
            'efficiency': efficiency * 100,
            'consumption_mol_h': consumption * 3600
        }

def analyze_and_plot():
    """Analyze flow rate optimization with realistic parameters"""
    optimizer = RealisticFlowOptimizer()

    # Get optimal conditions
    opt = optimizer.optimize()

    print("\nRealistic Flow Rate Optimization Results")
    print("=" * 50)
    print(f"Optimal flow rate: {opt['Q_opt_mL_h']:.1f} mL/h")
    print(f"Residence time: {opt['tau_min']:.1f} minutes")
    print(f"Outlet acetate: {opt['C_AC']:.3f} mol/m³")
    print(f"Biomass: {opt['X']:.2f} g/L")
    print(f"Biofilm factor: {opt['biofilm']:.2f}")
    print(f"Power output: {opt['power']:.3f} W")
    print(f"Substrate efficiency: {opt['efficiency']:.1f}%")
    print(f"Consumption rate: {opt['consumption_mol_h']:.4f} mol/h")

    # Analyze range
    Q_range = np.logspace(-6, -4.3, 100)  # 3.6 to 180 mL/h
    results = {
        'Q_mL_h': Q_range * 3.6e6,
        'tau_min': [],
        'power': [],
        'efficiency': [],
        'biofilm': [],
        'consumption': []
    }

    for Q in Q_range:
        tau = optimizer.residence_time(Q)
        biofilm = optimizer.biofilm_factor(tau)
        C_AC, X = optimizer.steady_state_concentrations(Q, biofilm)
        power = optimizer.calculate_power(Q, C_AC, X, biofilm)
        efficiency = optimizer.substrate_efficiency(C_AC)
        consumption = Q * (optimizer.params.C_AC_in - C_AC)

        results['tau_min'].append(tau / 60)
        results['power'].append(power)
        results['efficiency'].append(efficiency * 100)
        results['biofilm'].append(biofilm)
        results['consumption'].append(consumption * 3600)

    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Power vs flow rate
    ax1.plot(results['Q_mL_h'], results['power'], 'b-', linewidth=2)
    ax1.axvline(opt['Q_opt_mL_h'], color='r', linestyle='--', label='Optimal')
    ax1.axvline(81, color='g', linestyle=':', label='Current')
    ax1.set_xlabel('Flow Rate (mL/h)')
    ax1.set_ylabel('Power Output (W)')
    ax1.set_title('Power Generation vs Flow Rate')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Efficiency vs flow rate
    ax2.plot(results['Q_mL_h'], results['efficiency'], 'g-', linewidth=2)
    ax2.axvline(opt['Q_opt_mL_h'], color='r', linestyle='--')
    ax2.axvline(81, color='g', linestyle=':')
    ax2.set_xlabel('Flow Rate (mL/h)')
    ax2.set_ylabel('Substrate Efficiency (%)')
    ax2.set_title('Substrate Utilization Efficiency')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)

    # Biofilm vs flow rate
    ax3.plot(results['Q_mL_h'], results['biofilm'], 'm-', linewidth=2)
    ax3.axvline(opt['Q_opt_mL_h'], color='r', linestyle='--')
    ax3.axvline(81, color='g', linestyle=':')
    ax3.axhline(1.5, color='orange', linestyle=':', label='Maintenance needed')
    ax3.set_xlabel('Flow Rate (mL/h)')
    ax3.set_ylabel('Biofilm Factor')
    ax3.set_title('Biofilm Growth vs Flow Rate')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Combined performance metric
    ax4.plot(results['Q_mL_h'], results['tau_min'], 'c-', linewidth=2)
    ax4.axvline(opt['Q_opt_mL_h'], color='r', linestyle='--')
    ax4.axvline(81, color='g', linestyle=':')
    ax4.set_xlabel('Flow Rate (mL/h)')
    ax4.set_ylabel('Residence Time (min)')
    ax4.set_title('Hydraulic Residence Time')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Realistic MFC Flow Rate Optimization', fontsize=16)
    plt.tight_layout()
    plt.savefig(get_figure_path('mfc_flow_optimization_realistic.png'), dpi=300)
    plt.close()

    # Comparison with current operation
    current_Q = 81  # mL/h
    current_idx = np.argmin(np.abs(np.array(results['Q_mL_h']) - current_Q))

    print("\nComparison with Current Operation:")
    print(f"Current: {current_Q:.1f} mL/h → Optimal: {opt['Q_opt_mL_h']:.1f} mL/h")
    print(f"Power: {results['power'][current_idx]:.3f} W → {opt['power']:.3f} W "
          f"({(opt['power']/results['power'][current_idx]-1)*100:+.1f}%)")
    print(f"Efficiency: {results['efficiency'][current_idx]:.1f}% → {opt['efficiency']:.1f}% "
          f"({opt['efficiency']-results['efficiency'][current_idx]:+.1f}%)")
    print(f"Biofilm: {results['biofilm'][current_idx]:.2f} → {opt['biofilm']:.2f}")

    return opt, results

if __name__ == "__main__":
    opt, results = analyze_and_plot()
    print(f"\nOptimization complete! Check '{get_figure_path('mfc_flow_optimization_realistic.png')}'")
