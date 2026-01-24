#!/usr/bin/env python3
"""Flow Rate Optimization for MFC Stack

Optimizes anodic flow rate to maximize power output and substrate consumption
while minimizing biofilm growth.

Supports two modes:
- Basic mode (default): Original simplified model
- Realistic mode: Enhanced model with Butler-Volmer kinetics and internal resistance

Usage:
    # Basic mode (default)
    python flow_rate_optimization.py

    # Realistic mode
    python flow_rate_optimization.py --realistic
"""

import argparse

import matplotlib as mpl
import numpy as np

mpl.use("Agg")  # Use non-interactive backend
from dataclasses import dataclass

import matplotlib.pyplot as plt
from path_config import get_figure_path
from scipy.optimize import minimize_scalar


# =============================================================================
# Basic Mode Parameters and Optimizer
# =============================================================================


@dataclass
class MFCParameters:
    """MFC system parameters for basic mode."""

    # Physical parameters
    V_a: float = 5.5e-5  # Anodic chamber volume (m³)
    A_m: float = 5.0e-4  # Membrane area (m²)
    n_cells: float = 5  # Number of cells

    # Kinetic parameters
    k1_0: float = 0.207  # Anodic rate constant
    K_AC: float = 0.592  # Acetate half-saturation constant (mol/m³)
    Y_ac: float = 0.05  # Biomass yield (kg/mol acetate)
    K_dec: float = 8.33e-4  # Biomass decay constant (s⁻¹)

    # Operational parameters
    C_AC_in: float = 1.56  # Inlet acetate concentration (mol/m³)
    T: float = 303  # Temperature (K)
    F: float = 96485  # Faraday constant (C/mol)
    R: float = 8.314  # Gas constant (J/mol/K)

    # Biofilm parameters
    biofilm_growth_rate: float = 0.0005  # per hour
    biofilm_max: float = 2.0  # Maximum biofilm thickness factor
    biofilm_inhibition: float = 0.5  # Reduction in rate at max biofilm


class MFCFlowOptimizer:
    """Basic flow rate optimizer using simplified kinetics."""

    def __init__(self, params: MFCParameters | None = None) -> None:
        self.params = params or MFCParameters()

    def residence_time(self, Q_a: float) -> float:
        """Calculate hydraulic residence time."""
        return self.params.V_a / Q_a

    def steady_state_acetate(
        self,
        Q_a: float,
        X: float = 1.0,
        biofilm: float = 1.0,
    ) -> float:
        """Calculate steady-state acetate concentration in reactor."""
        p = self.params

        # Reaction rate at steady state (simplified)
        # Assuming average conditions for demonstration
        eta_a = 0.4  # Typical anode overpotential

        # Maximum reaction rate
        r1_max = p.k1_0 * np.exp(0.051 * p.F / (p.R * p.T) * eta_a) * X / biofilm

        # Solve quadratic equation for steady-state C_AC
        # Q_a(C_AC_in - C_AC) = A_m * r1_max * C_AC/(K_AC + C_AC)
        # This simplifies to a quadratic equation
        a = Q_a
        b = Q_a * p.K_AC + p.A_m * r1_max - Q_a * p.C_AC_in
        c = -Q_a * p.K_AC * p.C_AC_in

        # Positive root
        C_AC = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        return max(0, C_AC)

    def substrate_consumption_rate(self, Q_a: float, C_AC: float) -> float:
        """Calculate substrate consumption rate (mol/s)."""
        p = self.params
        return Q_a * (p.C_AC_in - C_AC)

    def power_output(self, Q_a: float, X: float = 1.0, biofilm: float = 1.0) -> float:
        """Estimate power output based on flow rate."""
        p = self.params

        # Get steady-state acetate concentration
        C_AC = self.steady_state_acetate(Q_a, X, biofilm)

        # Reaction rate
        eta_a = 0.4
        r1 = (
            p.k1_0
            * np.exp(0.051 * p.F / (p.R * p.T) * eta_a)
            * (C_AC / (p.K_AC + C_AC))
            * X
            / biofilm
        )

        # Current density (A/m²)
        i = 8 * p.F * r1  # 8 electrons per acetate

        # Power (simplified - assuming 0.7V per cell)
        return p.n_cells * p.A_m * i * 0.7

    def biofilm_growth_factor(self, residence_time: float) -> float:
        """Calculate biofilm growth based on residence time."""
        # Longer residence time promotes biofilm growth
        # This is a simplified model
        growth = 1.0 + self.params.biofilm_growth_rate * residence_time / 3600
        return min(growth, self.params.biofilm_max)

    def objective_function(self, Q_a: float, time_hours: float = 100) -> float:
        """Objective function to maximize (negative for minimization)
        Combines power output, substrate utilization, and biofilm penalty.
        """
        # Avoid extreme flow rates
        if Q_a < 1e-6 or Q_a > 1e-3:
            return 1e6

        # Calculate residence time
        tau = self.residence_time(Q_a)

        # Estimate biofilm growth
        biofilm = self.biofilm_growth_factor(tau * time_hours)

        # Calculate steady-state performance
        C_AC = self.steady_state_acetate(Q_a, X=1.0, biofilm=biofilm)

        # Substrate utilization efficiency
        substrate_efficiency = (self.params.C_AC_in - C_AC) / self.params.C_AC_in

        # Power output
        power = self.power_output(Q_a, X=1.0, biofilm=biofilm)

        # Substrate consumption rate
        consumption_rate = self.substrate_consumption_rate(Q_a, C_AC)

        # Normalize components for better balance
        power_norm = power / 1.0  # Expected ~1W range
        consumption_norm = consumption_rate / 0.001  # Expected ~0.001 mol/s

        # Strong penalty for very low substrate efficiency
        efficiency_penalty = (
            0 if substrate_efficiency > 0.5 else (0.5 - substrate_efficiency) ** 2 * 100
        )

        # Combined objective with better balance
        objective = (
            10.0 * power_norm  # Maximize power
            + 5.0 * consumption_norm  # Maximize substrate consumption
            + 20.0 * substrate_efficiency  # Prioritize efficiency > 50%
            - efficiency_penalty  # Penalty for low efficiency
            - 50.0 * (biofilm - 1.0) ** 2  # Strong penalty for biofilm growth
        )

        return -objective  # Negative for minimization

    def optimize_flow_rate(self) -> dict:
        """Find optimal flow rate."""
        # Initial guess based on typical values

        # Bounds for flow rate (m³/s)
        Q_min = 1e-6  # Very slow flow
        Q_max = 1e-4  # Fast flow

        # Optimize using bounded scalar minimization
        result = minimize_scalar(
            lambda Q: self.objective_function(Q),
            bounds=(Q_min, Q_max),
            method="bounded",
            options={"xatol": 1e-8},
        )

        Q_optimal = result.x

        # Calculate performance at optimal flow rate
        tau = self.residence_time(Q_optimal)
        biofilm = self.biofilm_growth_factor(tau * 100)  # 100 hour operation
        C_AC = self.steady_state_acetate(Q_optimal, X=1.0, biofilm=biofilm)
        power = self.power_output(Q_optimal, X=1.0, biofilm=biofilm)
        consumption = self.substrate_consumption_rate(Q_optimal, C_AC)
        efficiency = (self.params.C_AC_in - C_AC) / self.params.C_AC_in

        return {
            "Q_optimal": Q_optimal,
            "Q_optimal_mL_h": Q_optimal * 3.6e6,  # Convert to mL/h
            "residence_time_min": tau / 60,
            "C_AC_steady": C_AC,
            "substrate_efficiency": efficiency * 100,
            "power_W": power,
            "consumption_mol_h": consumption * 3600,
            "biofilm_factor": biofilm,
        }

    def analyze_flow_range(self, n_points: int = 100) -> dict:
        """Analyze performance across flow rate range."""
        Q_range = np.logspace(-6, -4, n_points)  # m³/s

        results = {
            "Q": Q_range,
            "Q_mL_h": Q_range * 3.6e6,
            "residence_time": [],
            "C_AC": [],
            "power": [],
            "efficiency": [],
            "consumption": [],
            "biofilm": [],
        }

        for Q in Q_range:
            tau = self.residence_time(Q)
            biofilm = self.biofilm_growth_factor(tau * 100)
            C_AC = self.steady_state_acetate(Q, X=1.0, biofilm=biofilm)

            results["residence_time"].append(tau / 60)  # minutes
            results["C_AC"].append(C_AC)
            results["power"].append(self.power_output(Q, X=1.0, biofilm=biofilm))
            results["efficiency"].append(
                (self.params.C_AC_in - C_AC) / self.params.C_AC_in * 100,
            )
            results["consumption"].append(
                self.substrate_consumption_rate(Q, C_AC) * 3600,
            )  # mol/h
            results["biofilm"].append(biofilm)

        return results


# =============================================================================
# Realistic Mode Parameters and Optimizer
# =============================================================================


@dataclass
class RealisticMFCParameters:
    """Realistic MFC system parameters with practical constraints."""

    # Physical parameters
    V_a: float = 5.5e-5  # Anodic chamber volume (m³)
    A_m: float = 5.0e-4  # Membrane area (m²)
    n_cells: float = 5  # Number of cells

    # Kinetic parameters from literature
    k1_0: float = 0.207  # Anodic rate constant
    K_AC: float = 0.592  # Acetate half-saturation (mol/m³)
    Y_ac: float = 0.05  # Biomass yield
    K_dec: float = 8.33e-4  # Biomass decay (s⁻¹)

    # Realistic power parameters
    V_cell_max: float = 0.8  # Maximum cell voltage (V)
    V_cell_typical: float = 0.5  # Typical operating voltage (V)
    i_max: float = 10.0  # Maximum current density (A/m²)

    # Operational parameters
    C_AC_in: float = 1.56  # Inlet acetate (mol/m³)
    T: float = 303  # Temperature (K)
    F: float = 96485  # Faraday constant
    R: float = 8.314  # Gas constant

    # Biofilm parameters
    tau_biofilm: float = 24 * 3600  # Biofilm doubling time (s)
    biofilm_max: float = 2.0  # Max thickness factor
    biofilm_k: float = 0.5  # Half-saturation for biofilm effect


class RealisticFlowOptimizer:
    """Realistic flow rate optimizer with Butler-Volmer kinetics and internal resistance."""

    def __init__(self, params: RealisticMFCParameters | None = None) -> None:
        self.params = params or RealisticMFCParameters()

    def residence_time(self, Q_a: float) -> float:
        """Hydraulic residence time (seconds)."""
        return self.params.V_a / Q_a

    def biofilm_factor(self, tau: float, operation_time: float = 100 * 3600) -> float:
        """Calculate biofilm thickness factor using logistic growth model."""
        p = self.params
        # Logistic growth model
        growth_rate = np.log(2) / p.tau_biofilm
        biofilm = 1 + (p.biofilm_max - 1) * (1 - np.exp(-growth_rate * tau))
        return min(biofilm, p.biofilm_max)

    def steady_state_concentrations(
        self, Q_a: float, biofilm: float = 1.0
    ) -> tuple[float, float]:
        """Calculate steady-state concentrations and biomass."""
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

        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            C_AC = p.C_AC_in * 0.1  # Fallback
        else:
            C_AC = (-b + np.sqrt(discriminant)) / (2 * a)
            C_AC = max(0.01, min(C_AC, p.C_AC_in))

        return C_AC, X

    def calculate_power(
        self,
        Q_a: float,
        C_AC: float,
        X: float,
        biofilm: float,
    ) -> float:
        """Calculate realistic power output with Butler-Volmer kinetics."""
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
        return p.n_cells * p.A_m * i * V_cell

    def substrate_efficiency(self, C_AC: float) -> float:
        """Calculate substrate conversion efficiency."""
        return (self.params.C_AC_in - C_AC) / self.params.C_AC_in

    def objective_function(self, Q_a: float) -> float:
        """Multi-objective function for realistic optimization."""
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
        tau_penalty = 0 if tau > 60 else (60 - tau) ** 2 / 3600

        # Combined objective
        objective = (
            5.0 * obj_power
            + 10.0 * obj_efficiency
            + 2.0 * obj_consumption
            + 3.0 * obj_biofilm
            - tau_penalty
        )

        return -objective

    def optimize(self) -> dict:
        """Find optimal flow rate."""
        # Search range
        Q_min = 1e-6  # 3.6 mL/h
        Q_max = 5e-5  # 180 mL/h

        result = minimize_scalar(
            self.objective_function,
            bounds=(Q_min, Q_max),
            method="bounded",
        )

        Q_opt = result.x
        tau = self.residence_time(Q_opt)
        biofilm = self.biofilm_factor(tau)
        C_AC, X = self.steady_state_concentrations(Q_opt, biofilm)
        power = self.calculate_power(Q_opt, C_AC, X, biofilm)
        efficiency = self.substrate_efficiency(C_AC)
        consumption = Q_opt * (self.params.C_AC_in - C_AC)

        return {
            "Q_optimal": Q_opt,
            "Q_optimal_mL_h": Q_opt * 3.6e6,
            "residence_time_min": tau / 60,
            "C_AC_steady": C_AC,
            "X": X,
            "biofilm_factor": biofilm,
            "power_W": power,
            "substrate_efficiency": efficiency * 100,
            "consumption_mol_h": consumption * 3600,
        }

    def analyze_flow_range(self, n_points: int = 100) -> dict:
        """Analyze performance across flow rate range."""
        Q_range = np.logspace(-6, -4.3, n_points)  # 3.6 to 180 mL/h

        results = {
            "Q": Q_range,
            "Q_mL_h": Q_range * 3.6e6,
            "residence_time": [],
            "C_AC": [],
            "power": [],
            "efficiency": [],
            "consumption": [],
            "biofilm": [],
        }

        for Q in Q_range:
            tau = self.residence_time(Q)
            biofilm = self.biofilm_factor(tau)
            C_AC, X = self.steady_state_concentrations(Q, biofilm)

            results["residence_time"].append(tau / 60)  # minutes
            results["C_AC"].append(C_AC)
            results["power"].append(self.calculate_power(Q, C_AC, X, biofilm))
            results["efficiency"].append(self.substrate_efficiency(C_AC) * 100)
            results["consumption"].append(
                Q * (self.params.C_AC_in - C_AC) * 3600
            )  # mol/h
            results["biofilm"].append(biofilm)

        return results


# =============================================================================
# Plotting Functions
# =============================================================================


def plot_optimization_results(
    results: dict, optimal: dict, realistic: bool = False
) -> None:
    """Create comprehensive optimization analysis plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    mode_str = "Realistic " if realistic else ""
    fig.suptitle(f"{mode_str}MFC Flow Rate Optimization Analysis", fontsize=16)

    # 1. Power vs Flow Rate
    ax = axes[0, 0]
    ax.plot(results["Q_mL_h"], results["power"], "b-", linewidth=2)
    ax.axvline(
        optimal["Q_optimal_mL_h"],
        color="r",
        linestyle="--",
        label=f"Optimal: {optimal['Q_optimal_mL_h']:.1f} mL/h",
    )
    ax.set_xlabel("Flow Rate (mL/h)")
    ax.set_ylabel("Power Output (W)")
    ax.set_title("Power vs Flow Rate")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. Substrate Efficiency vs Flow Rate
    ax = axes[0, 1]
    ax.plot(results["Q_mL_h"], results["efficiency"], "g-", linewidth=2)
    ax.axvline(optimal["Q_optimal_mL_h"], color="r", linestyle="--")
    ax.set_xlabel("Flow Rate (mL/h)")
    ax.set_ylabel("Substrate Efficiency (%)")
    ax.set_title("Substrate Utilization Efficiency")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    # 3. Biofilm Growth vs Flow Rate
    ax = axes[0, 2]
    ax.plot(results["Q_mL_h"], results["biofilm"], "m-", linewidth=2)
    ax.axvline(optimal["Q_optimal_mL_h"], color="r", linestyle="--")
    ax.axhline(1.5, color="orange", linestyle=":", label="Maintenance threshold")
    ax.set_xlabel("Flow Rate (mL/h)")
    ax.set_ylabel("Biofilm Factor")
    ax.set_title("Biofilm Growth Factor")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4. Residence Time vs Flow Rate
    ax = axes[1, 0]
    ax.plot(results["Q_mL_h"], results["residence_time"], "c-", linewidth=2)
    ax.axvline(optimal["Q_optimal_mL_h"], color="r", linestyle="--")
    ax.set_xlabel("Flow Rate (mL/h)")
    ax.set_ylabel("Residence Time (min)")
    ax.set_title("Hydraulic Residence Time")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # 5. Acetate Concentration vs Flow Rate
    ax = axes[1, 1]
    ax.plot(results["Q_mL_h"], results["C_AC"], "orange", linewidth=2)
    ax.axvline(optimal["Q_optimal_mL_h"], color="r", linestyle="--")
    ax.axhline(1.56, color="k", linestyle=":", label="Inlet concentration")
    ax.set_xlabel("Flow Rate (mL/h)")
    ax.set_ylabel("Acetate Concentration (mol/m³)")
    ax.set_title("Steady-State Acetate in Reactor")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 6. Combined Performance Metric
    ax = axes[1, 2]
    # Normalize metrics for comparison
    power_norm = np.array(results["power"]) / max(results["power"])
    efficiency_norm = np.array(results["efficiency"]) / 100
    biofilm_penalty = 2 - np.array(results["biofilm"])  # Inverted

    combined_metric = power_norm + efficiency_norm + 0.5 * biofilm_penalty

    ax.plot(results["Q_mL_h"], combined_metric, "k-", linewidth=2)
    ax.axvline(optimal["Q_optimal_mL_h"], color="r", linestyle="--")
    ax.set_xlabel("Flow Rate (mL/h)")
    ax.set_ylabel("Combined Performance Score")
    ax.set_title("Multi-Objective Performance")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    suffix = "_realistic" if realistic else ""
    plt.savefig(
        get_figure_path(f"mfc_flow_optimization_analysis{suffix}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Create summary table
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("tight")
    ax.axis("off")

    # Current vs Optimal comparison
    current_Q = 2.25e-5 * 3.6e6 if not realistic else 81  # Current flow rate in mL/h
    current_idx = np.argmin(np.abs(np.array(results["Q_mL_h"]) - current_Q))

    table_data = [
        ["Parameter", "Current Operation", "Optimal Operation", "Improvement"],
        [
            "Flow Rate (mL/h)",
            f"{current_Q:.1f}",
            f"{optimal['Q_optimal_mL_h']:.1f}",
            f"{(optimal['Q_optimal_mL_h'] / current_Q - 1) * 100:+.1f}%",
        ],
        [
            "Residence Time (min)",
            f"{results['residence_time'][current_idx]:.1f}",
            f"{optimal['residence_time_min']:.1f}",
            f"{(optimal['residence_time_min'] / results['residence_time'][current_idx] - 1) * 100:+.1f}%",
        ],
        [
            "Power Output (W)",
            f"{results['power'][current_idx]:.3f}",
            f"{optimal['power_W']:.3f}",
            f"{(optimal['power_W'] / results['power'][current_idx] - 1) * 100:+.1f}%",
        ],
        [
            "Substrate Efficiency (%)",
            f"{results['efficiency'][current_idx]:.1f}",
            f"{optimal['substrate_efficiency']:.1f}",
            f"{optimal['substrate_efficiency'] - results['efficiency'][current_idx]:+.1f}%",
        ],
        [
            "Biofilm Factor",
            f"{results['biofilm'][current_idx]:.2f}",
            f"{optimal['biofilm_factor']:.2f}",
            f"{(optimal['biofilm_factor'] / results['biofilm'][current_idx] - 1) * 100:+.1f}%",
        ],
        [
            "Acetate Consumption (mol/h)",
            f"{results['consumption'][current_idx]:.3f}",
            f"{optimal['consumption_mol_h']:.3f}",
            f"{(optimal['consumption_mol_h'] / results['consumption'][current_idx] - 1) * 100:+.1f}%",
        ],
    ]

    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style the header row
    for i in range(4):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Color code improvements
    for i in range(1, 7):
        cell = table[(i, 3)]
        text = cell.get_text().get_text()
        if "+" in text:
            cell.set_facecolor("#90EE90")
        elif "-" in text and i != 5:  # Biofilm reduction is good
            cell.set_facecolor("#FFB6C1")
        elif "-" in text and i == 5:  # Biofilm reduction is good
            cell.set_facecolor("#90EE90")

    ax.set_title(
        f"{mode_str}Flow Rate Optimization Summary\nCurrent vs Optimal Performance",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.savefig(
        get_figure_path(f"mfc_flow_optimization_summary{suffix}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# =============================================================================
# Main Function
# =============================================================================


def run_optimization(realistic: bool = False) -> tuple[dict, dict]:
    """Run flow rate optimization analysis.

    Args:
        realistic: If True, use realistic model with Butler-Volmer kinetics
                   and internal resistance. If False (default), use basic model.

    Returns:
        Tuple of (optimal_results, analysis_results)
    """
    if realistic:
        optimizer = RealisticFlowOptimizer()
        optimal = optimizer.optimize()
    else:
        optimizer = MFCFlowOptimizer()
        optimal = optimizer.optimize_flow_rate()

    # Analyze flow rate range
    results = optimizer.analyze_flow_range()

    # Create plots
    plot_optimization_results(results, optimal, realistic=realistic)

    return optimal, results


def main() -> None:
    """Run flow rate optimization analysis with CLI support."""
    parser = argparse.ArgumentParser(
        description="MFC Flow Rate Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python flow_rate_optimization.py           # Run basic mode
  python flow_rate_optimization.py --realistic  # Run realistic mode
        """,
    )
    parser.add_argument(
        "--realistic",
        action="store_true",
        help="Use realistic model with Butler-Volmer kinetics",
    )

    args = parser.parse_args()

    # Run optimization
    optimal, results = run_optimization(realistic=args.realistic)

    # Print summary
    mode = "Realistic" if args.realistic else "Basic"
    print(f"\n{mode} Mode Optimization Results:")
    print(f"  Optimal Flow Rate: {optimal['Q_optimal_mL_h']:.2f} mL/h")
    print(f"  Residence Time: {optimal['residence_time_min']:.2f} min")
    print(f"  Power Output: {optimal['power_W']:.4f} W")
    print(f"  Substrate Efficiency: {optimal['substrate_efficiency']:.1f}%")
    print(f"  Biofilm Factor: {optimal['biofilm_factor']:.3f}")


if __name__ == "__main__":
    main()
