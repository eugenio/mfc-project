#!/usr/bin/env python3
"""Phase 3: ML-Physics Integration for MFC Electrode Optimization.

This module connects the advanced physics models (Phase 2) with the machine learning
optimization framework (Phase 3) to enable intelligent electrode parameter optimization.

Created: 2025-08-01
Last Modified: 2025-08-01

Integration Architecture:
- Physics models provide optimization targets
- ML algorithms optimize electrode parameters
- Real-time feedback loop for parameter tuning
- Multi-objective optimization with physics constraints
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

# Import Configuration System
try:
    from ..config.electrode_config import (
        ElectrodeGeometry,
        ElectrodeMaterial,
        create_electrode_config,
    )
except ImportError:
    # Fall back to absolute import for testing
    from config.electrode_config import (
        ElectrodeGeometry,
        ElectrodeMaterial,
        create_electrode_config,
)

# Import Phase 2 Physics Models
try:
    from ..physics.advanced_electrode_model import (
        AdvancedElectrodeModel,
        BiofilmDynamics,
        CellGeometry,
        FluidDynamicsProperties,
        MassTransportProperties,
    )
except ImportError:
    from physics.advanced_electrode_model import (
        AdvancedElectrodeModel,
        BiofilmDynamics,
        CellGeometry,
        FluidDynamicsProperties,
        MassTransportProperties,
    )

# Import Phase 3 ML Optimization
try:
    from .electrode_optimization import (
        BayesianOptimizer,
        GaussianProcessSurrogate,
        MultiObjectiveOptimizer,
        NeuralNetworkSurrogate,
        OptimizationObjective,
        OptimizationParameter,
        OptimizationResult,
    )
except ImportError:
    from ml.electrode_optimization import (
        BayesianOptimizer,
        GaussianProcessSurrogate,
        MultiObjectiveOptimizer,
        NeuralNetworkSurrogate,
        OptimizationObjective,
        OptimizationParameter,
        OptimizationResult,
    )

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class PhysicsMLConfig:
    """Configuration for physics-ML integration."""

    # Physics simulation parameters
    simulation_timestep: float = 3600.0  # seconds (1 hour)
    max_simulation_time: float = 24 * 3600  # 24 hours max
    convergence_tolerance: float = 1e-6

    # ML optimization parameters
    n_initial_samples: int = 10
    n_optimization_iterations: int = 50
    surrogate_model_type: str = (
        "gaussian_process"  # 'gaussian_process' or 'neural_network'
    )
    acquisition_function: str = "expected_improvement"  # 'EI', 'UCB'

    # Multi-objective weights
    objective_weights: dict[str, float] = field(
        default_factory=lambda: {
            "current_density": 1.0,
            "substrate_utilization": 0.8,
            "pressure_drop": 0.6,
            "electrode_utilization": 0.7,
            "pore_blocking": 0.5,
            "biofilm_stability": 0.4,
        },
    )

    # Physics constraints
    min_current_density: float = 0.1  # A/m²
    max_pressure_drop: float = 500.0  # Pa
    min_substrate_utilization: float = 0.1  # fraction
    max_pore_blocking: float = 0.8  # fraction


class PhysicsMLIntegrator:
    """Main integration class connecting Phase 2 physics with Phase 3 ML optimization.

    This class orchestrates the optimization process by:
    1. Running physics simulations with different parameters
    2. Extracting optimization targets from physics results
    3. Using ML algorithms to suggest better parameters
    4. Iterating until convergence or maximum iterations
    """

    def __init__(
        self,
        base_cell_geometry: CellGeometry,
        config: PhysicsMLConfig | None = None,
    ) -> None:
        self.cell_geometry = base_cell_geometry
        self.config = config or PhysicsMLConfig()

        # Initialize optimization parameters for electrode design
        self.optimization_parameters = self._create_optimization_parameters()
        self.optimization_objectives = self._create_optimization_objectives()

        # Create model factory for physics evaluations
        self.model_factory = self._create_integrated_model_factory()

        # Initialize optimization algorithms
        self.bayesian_optimizer = None
        self.multi_objective_optimizer = None

        # Results tracking
        self.optimization_history = []
        self.best_parameters = {}
        self.best_performance = {}

    def _create_optimization_parameters(self) -> list[OptimizationParameter]:
        """Define optimization parameters for electrode design."""
        return [
            # Electrode geometry parameters
            OptimizationParameter(
                name="electrode_length",
                bounds=(0.02, 0.1),  # 2-10 cm
                parameter_type="continuous",
                units="m",
                description="Electrode length for surface area optimization",
            ),
            OptimizationParameter(
                name="electrode_width",
                bounds=(0.02, 0.1),  # 2-10 cm
                parameter_type="continuous",
                units="m",
                description="Electrode width for surface area optimization",
            ),
            OptimizationParameter(
                name="electrode_thickness",
                bounds=(0.001, 0.02),  # 1-20 mm
                parameter_type="continuous",
                units="m",
                description="Electrode thickness for volume optimization",
            ),
            # Fluid dynamics parameters
            OptimizationParameter(
                name="flow_rate",
                bounds=(0.5e-6, 5e-6),  # 0.5-5 mL/min
                parameter_type="continuous",
                units="m³/s",
                description="Flow rate for mass transport optimization",
            ),
            # Biofilm parameters
            OptimizationParameter(
                name="max_biofilm_density",
                bounds=(50.0, 120.0),  # kg/m³
                parameter_type="continuous",
                units="kg/m³",
                description="Maximum biofilm density for growth modeling",
            ),
            # Mass transport parameters
            OptimizationParameter(
                name="max_substrate_consumption",
                bounds=(0.05, 0.5),  # mol/m³/s
                parameter_type="continuous",
                units="mol/m³/s",
                description="Maximum substrate consumption rate",
            ),
        ]

    def _create_optimization_objectives(self) -> list[OptimizationObjective]:
        """Define optimization objectives based on physics targets."""
        return [
            OptimizationObjective(
                name="current_density",
                direction="maximize",
                weight=self.config.objective_weights["current_density"],
                constraint_type="soft",
                constraint_value=self.config.min_current_density,
            ),
            OptimizationObjective(
                name="substrate_utilization",
                direction="maximize",
                weight=self.config.objective_weights["substrate_utilization"],
                constraint_type="soft",
                constraint_value=self.config.min_substrate_utilization,
            ),
            OptimizationObjective(
                name="pressure_drop",
                direction="minimize",
                weight=self.config.objective_weights["pressure_drop"],
                constraint_type="hard",
                constraint_value=self.config.max_pressure_drop,
            ),
            OptimizationObjective(
                name="electrode_utilization",
                direction="maximize",
                weight=self.config.objective_weights["electrode_utilization"],
            ),
            OptimizationObjective(
                name="pore_blocking",
                direction="minimize",
                weight=self.config.objective_weights["pore_blocking"],
                constraint_type="soft",
                constraint_value=self.config.max_pore_blocking,
            ),
            OptimizationObjective(
                name="biofilm_stability",
                direction="maximize",
                weight=self.config.objective_weights["biofilm_stability"],
            ),
        ]

    def _create_integrated_model_factory(self) -> Callable:
        """Create factory function that integrates physics parameters."""

        def integrated_model_factory(
            parameters: dict[str, float],
        ) -> AdvancedElectrodeModel:
            """Create advanced electrode model with optimized parameters."""
            # Extract electrode geometry parameters
            length = parameters.get("electrode_length", 0.05)
            width = parameters.get("electrode_width", 0.05)
            thickness = parameters.get("electrode_thickness", 0.005)

            # Extract fluid dynamics parameters
            flow_rate = parameters.get("flow_rate", 1e-6)

            # Extract biofilm parameters
            max_biofilm_density = parameters.get("max_biofilm_density", 80.0)

            # Extract mass transport parameters
            max_substrate_consumption = parameters.get("max_substrate_consumption", 0.1)

            # Create electrode configuration with optimized parameters
            electrode_config = create_electrode_config(
                material=ElectrodeMaterial.CARBON_FELT,  # Could be optimized too
                geometry_type=ElectrodeGeometry.RECTANGULAR_PLATE,
                dimensions={"length": length, "width": width, "thickness": thickness},
            )

            # Create fluid properties with optimized flow rate
            fluid_properties = FluidDynamicsProperties(flow_rate=flow_rate)

            # Create mass transport properties with optimized consumption
            transport_properties = MassTransportProperties(
                max_substrate_consumption_rate=max_substrate_consumption,
            )

            # Create biofilm dynamics with optimized density
            biofilm_dynamics = BiofilmDynamics(max_biofilm_density=max_biofilm_density)

            # Create integrated advanced electrode model
            return AdvancedElectrodeModel(
                electrode_config=electrode_config,
                cell_geometry=self.cell_geometry,
                fluid_properties=fluid_properties,
                transport_properties=transport_properties,
                biofilm_dynamics=biofilm_dynamics,
            )

        return integrated_model_factory

    def evaluate_physics_performance(
        self,
        parameters: dict[str, float],
    ) -> dict[str, float]:
        """Evaluate physics performance for given parameters.

        This is the key integration function that runs physics simulation
        and extracts optimization targets.
        """
        try:
            # Create model with given parameters
            model = self.model_factory(parameters)

            # Run physics simulation
            dt = self.config.simulation_timestep
            max_time = self.config.max_simulation_time

            results_history = []
            current_time = 0.0

            # Multi-step simulation for better physics accuracy
            while current_time < max_time:
                results = model.step(dt)
                results_history.append(results)
                current_time += dt

                # Check for convergence (optional)
                if len(results_history) > 2:
                    recent_metrics = [
                        r["performance_metrics"] for r in results_history[-3:]
                    ]
                    substrate_change = abs(
                        recent_metrics[-1]["avg_substrate_mM"]
                        - recent_metrics[0]["avg_substrate_mM"],
                    )

                    if substrate_change < self.config.convergence_tolerance:
                        break

            # Extract final optimization targets
            targets = model.get_optimization_targets()

            # Map physics targets to optimization objectives
            performance = {
                "current_density": targets["maximize_current_density"],
                "substrate_utilization": targets["maximize_substrate_utilization"],
                "pressure_drop": targets["minimize_pressure_drop"],
                "electrode_utilization": targets["maximize_electrode_utilization"],
                "pore_blocking": 1
                - targets["minimize_pore_blocking"],  # Convert to minimize
                "biofilm_stability": targets["maximize_biofilm_stability"],
            }

            # Add constraint violations
            performance["constraint_violations"] = self._check_constraints(performance)
            performance["simulation_time"] = current_time
            performance["convergence_achieved"] = current_time < max_time

            return performance

        except Exception:
            # Return penalty values for failed evaluations
            return {
                "current_density": 0.0,
                "substrate_utilization": 0.0,
                "pressure_drop": 1000.0,  # High penalty
                "electrode_utilization": 0.0,
                "pore_blocking": 1.0,  # Maximum blocking
                "biofilm_stability": 0.0,
                "constraint_violations": 10.0,  # High penalty
                "simulation_time": 0.0,
                "convergence_achieved": False,
            }

    def _check_constraints(self, performance: dict[str, float]) -> float:
        """Check constraint violations and return penalty."""
        violations = 0.0

        # Hard constraints
        if performance["pressure_drop"] > self.config.max_pressure_drop:
            violations += (
                performance["pressure_drop"] - self.config.max_pressure_drop
            ) / self.config.max_pressure_drop

        # Soft constraints
        if performance["current_density"] < self.config.min_current_density:
            violations += (
                self.config.min_current_density - performance["current_density"]
            ) / self.config.min_current_density

        if performance["substrate_utilization"] < self.config.min_substrate_utilization:
            violations += (
                self.config.min_substrate_utilization
                - performance["substrate_utilization"]
            ) / self.config.min_substrate_utilization

        if performance["pore_blocking"] > self.config.max_pore_blocking:
            violations += (
                performance["pore_blocking"] - self.config.max_pore_blocking
            ) / self.config.max_pore_blocking

        return violations

    def run_bayesian_optimization(self) -> OptimizationResult:
        """Run Bayesian optimization using Gaussian Process surrogate model.

        This is the main ML optimization method that uses physics simulations
        as the objective function.
        """

        # Create custom objective function that uses physics evaluation
        def physics_objective_function(parameters: dict[str, float]) -> float:
            """Objective function that evaluates physics performance."""
            performance = self.evaluate_physics_performance(parameters)

            # Calculate weighted multi-objective value
            objective_value = 0.0
            constraint_penalty = (
                performance["constraint_violations"] * 100
            )  # Heavy penalty

            for obj in self.optimization_objectives:
                if obj.name in performance:
                    value = performance[obj.name]
                    if obj.direction == "maximize":
                        objective_value += obj.weight * value
                    else:  # minimize
                        objective_value -= obj.weight * value

            # Apply constraint penalty
            objective_value -= constraint_penalty

            # Store evaluation in history
            self.optimization_history.append(
                {
                    "parameters": parameters.copy(),
                    "performance": performance.copy(),
                    "objective_value": objective_value,
                    "constraint_violations": constraint_penalty,
                },
            )

            return objective_value

        # Create Bayesian optimizer with physics integration
        if self.config.surrogate_model_type == "gaussian_process":
            surrogate = GaussianProcessSurrogate()
        else:
            surrogate = NeuralNetworkSurrogate()

        self.bayesian_optimizer = BayesianOptimizer(
            parameters=self.optimization_parameters,
            objectives=self.optimization_objectives,
            electrode_model_factory=self.model_factory,  # Not used directly, we override objective
            surrogate_model=surrogate,
            acquisition_type=self.config.acquisition_function,
        )

        # Override the evaluation function to use our physics integration
        self.bayesian_optimizer._evaluate_objective = (
            lambda params: physics_objective_function(
                dict(
                    zip(
                        [p.name for p in self.optimization_parameters],
                        params,
                        strict=False,
                    ),
                ),
            )
        )

        # Run optimization
        results = self.bayesian_optimizer.optimize(
            n_iterations=self.config.n_optimization_iterations,
            n_initial=self.config.n_initial_samples,
        )

        # Store best results
        self.best_parameters = results.best_parameters
        self.best_performance = self.optimization_history[
            np.argmax([h["objective_value"] for h in self.optimization_history])
        ]["performance"]

        return results

    def run_multi_objective_optimization(self) -> dict[str, Any]:
        """Run multi-objective optimization to find Pareto optimal solutions."""
        # Create multi-objective optimizer
        self.multi_objective_optimizer = MultiObjectiveOptimizer(
            parameters=self.optimization_parameters,
            objectives=self.optimization_objectives,
            electrode_model_factory=self.model_factory,
            population_size=20,  # Smaller for physics integration
        )

        # Override evaluation to use physics integration

        def physics_population_evaluation(population: np.ndarray) -> np.ndarray:
            """Evaluate population using physics simulations."""
            objective_values = []

            for individual in population:
                param_dict = dict(
                    zip(
                        [p.name for p in self.optimization_parameters],
                        individual,
                        strict=False,
                    ),
                )
                performance = self.evaluate_physics_performance(param_dict)

                # Extract objective values for each individual
                individual_objectives = []
                for obj in self.optimization_objectives:
                    if obj.name in performance:
                        value = performance[obj.name]
                        # Convert to minimization problem for NSGA-II
                        if obj.direction == "maximize":
                            value = -value
                        individual_objectives.append(value)
                    else:
                        individual_objectives.append(
                            1e6,
                        )  # Penalty for missing objectives

                objective_values.append(individual_objectives)

            return np.array(objective_values)

        self.multi_objective_optimizer._evaluate_population = (
            physics_population_evaluation
        )

        # Run multi-objective optimization
        return self.multi_objective_optimizer.optimize(
            n_generations=10,
        )  # Fewer generations for physics

    def export_results(self, filepath: str) -> None:
        """Export optimization results to file."""
        results_data = {
            "config": {
                "simulation_timestep": self.config.simulation_timestep,
                "optimization_iterations": self.config.n_optimization_iterations,
                "surrogate_model": self.config.surrogate_model_type,
                "objective_weights": self.config.objective_weights,
            },
            "best_parameters": self.best_parameters,
            "best_performance": self.best_performance,
            "optimization_history": self.optimization_history,
            "total_evaluations": len(self.optimization_history),
        }

        with open(filepath, "w") as f:
            json.dump(results_data, f, indent=2, default=str)

    def create_summary_report(self) -> str:
        """Create summary report of optimization results."""
        if not self.best_parameters:
            return "No optimization results available."

        report = []
        report.append("# Phase 3: ML-Physics Integration Summary Report")
        report.append("=" * 50)
        report.append("")

        report.append("## Optimization Configuration")
        report.append(
            f"- Simulation Timestep: {self.config.simulation_timestep / 3600:.1f} hours",
        )
        report.append(f"- Total Iterations: {self.config.n_optimization_iterations}")
        report.append(f"- Surrogate Model: {self.config.surrogate_model_type}")
        report.append(f"- Total Physics Evaluations: {len(self.optimization_history)}")
        report.append("")

        report.append("## Best Parameters Found")
        for param_name, value in self.best_parameters.items():
            param_info = next(
                (p for p in self.optimization_parameters if p.name == param_name),
                None,
            )
            if param_info:
                report.append(
                    f"- {param_info.description}: {value:.4f} {param_info.units}",
                )
        report.append("")

        report.append("## Best Performance Achieved")
        report.append(
            f"- Current Density: {self.best_performance['current_density']:.3f} A/m²",
        )
        report.append(
            f"- Substrate Utilization: {self.best_performance['substrate_utilization']:.3f}",
        )
        report.append(
            f"- Pressure Drop: {self.best_performance['pressure_drop']:.1f} Pa",
        )
        report.append(
            f"- Electrode Utilization: {self.best_performance['electrode_utilization']:.3f}",
        )
        report.append(f"- Pore Blocking: {self.best_performance['pore_blocking']:.3f}")
        report.append(
            f"- Biofilm Stability: {self.best_performance['biofilm_stability']:.3f}",
        )
        report.append(
            f"- Constraint Violations: {self.best_performance['constraint_violations']:.3f}",
        )
        report.append("")

        report.append("## Integration Success Metrics")
        successful_evals = sum(
            1
            for h in self.optimization_history
            if h["performance"]["convergence_achieved"]
        )
        success_rate = successful_evals / len(self.optimization_history) * 100
        report.append(f"- Physics Simulation Success Rate: {success_rate:.1f}%")
        report.append(
            f"- Average Simulation Time: {np.mean([h['performance']['simulation_time'] for h in self.optimization_history]) / 3600:.1f} hours",
        )
        report.append("")

        return "\n".join(report)


def run_phase3_integration_example():
    """Example workflow demonstrating Phase 3 ML-Physics integration."""
    # Create base cell geometry
    cell_geometry = CellGeometry(
        length=0.1,
        width=0.1,
        height=0.05,  # 10x10x5 cm cell
        anode_chamber_volume=0.0002,  # 200 mL
        cathode_chamber_volume=0.0002,  # 200 mL
    )

    # Create integration configuration
    config = PhysicsMLConfig(
        simulation_timestep=1800.0,  # 30 minutes for faster execution
        n_initial_samples=5,  # Reduced for example
        n_optimization_iterations=10,  # Reduced for example
        surrogate_model_type="gaussian_process",
    )

    # Create ML-Physics integrator
    integrator = PhysicsMLIntegrator(base_cell_geometry=cell_geometry, config=config)

    # Test single physics evaluation
    test_params = {
        "electrode_length": 0.05,
        "electrode_width": 0.05,
        "electrode_thickness": 0.005,
        "flow_rate": 1e-6,
        "max_biofilm_density": 80.0,
        "max_substrate_consumption": 0.1,
    }

    integrator.evaluate_physics_performance(test_params)

    # Run Bayesian optimization
    optimization_results = integrator.run_bayesian_optimization()

    # Generate and save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"phase3_ml_physics_results_{timestamp}.json"
    integrator.export_results(results_file)

    # Print summary report

    return integrator, optimization_results


if __name__ == "__main__":
    # Run Phase 3 integration example
    integrator, results = run_phase3_integration_example()
