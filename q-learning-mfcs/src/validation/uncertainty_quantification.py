#!/usr/bin/env python3
"""Phase 5: Uncertainty Quantification Framework.

This module provides comprehensive uncertainty quantification for all model
parameters and predictions, including Monte Carlo analysis and sensitivity analysis.

Created: 2025-08-01
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class ParameterUncertainty:
    """Container for parameter uncertainty information."""

    parameter_name: str
    nominal_value: float
    units: str
    distribution_type: str = "normal"  # 'normal', 'lognormal', 'uniform', 'triangular'
    uncertainty_bounds: tuple[float, float] = (0.0, 0.0)
    confidence_level: float = 0.95
    source: str = ""
    correlation_matrix: dict[str, float] = field(default_factory=dict)


class MonteCarloAnalysis:
    """Monte Carlo uncertainty propagation analysis.

    Propagates parameter uncertainties through the MFC model to quantify
    prediction uncertainty bounds.
    """

    def __init__(
        self,
        parameter_uncertainties: dict[str, ParameterUncertainty],
    ) -> None:
        self.parameter_uncertainties = parameter_uncertainties
        self.sample_size = 1000
        self.results = {}

    def generate_parameter_samples(self, n_samples: int | None = None) -> pd.DataFrame:
        """Generate parameter samples based on uncertainty distributions."""
        if n_samples is None:
            n_samples = self.sample_size

        samples = {}
        np.random.seed(42)  # For reproducibility

        for param_name, uncertainty in self.parameter_uncertainties.items():
            nominal = uncertainty.nominal_value
            bounds = uncertainty.uncertainty_bounds
            dist_type = uncertainty.distribution_type

            if dist_type == "normal":
                # Use bounds as ±2σ (95% confidence interval)
                sigma = (bounds[1] - bounds[0]) / 4.0
                samples[param_name] = np.random.normal(nominal, sigma, n_samples)

            elif dist_type == "lognormal":
                # Log-normal distribution
                mu = np.log(nominal)
                sigma = (np.log(bounds[1]) - np.log(bounds[0])) / 4.0
                samples[param_name] = np.random.lognormal(mu, sigma, n_samples)

            elif dist_type == "uniform":
                # Uniform distribution within bounds
                samples[param_name] = np.random.uniform(bounds[0], bounds[1], n_samples)

            elif dist_type == "triangular":
                # Triangular distribution with mode at nominal value
                samples[param_name] = np.random.triangular(
                    bounds[0],
                    nominal,
                    bounds[1],
                    n_samples,
                )

            else:
                # Default to normal distribution
                sigma = (bounds[1] - bounds[0]) / 4.0
                samples[param_name] = np.random.normal(nominal, sigma, n_samples)

        return pd.DataFrame(samples)

    def propagate_uncertainty(
        self,
        model_function: Callable,
        output_names: list[str],
    ) -> dict[str, Any]:
        """Propagate parameter uncertainty through model function.

        Args:
            model_function: Function that takes parameter dict and returns output dict
            output_names: List of output variable names to analyze

        """
        # Generate parameter samples
        parameter_samples = self.generate_parameter_samples()

        # Initialize output arrays
        outputs = {name: [] for name in output_names}
        successful_runs = 0

        # Run model for each parameter sample
        for i, (_, params) in enumerate(parameter_samples.iterrows()):
            try:
                # Convert to dictionary
                param_dict = params.to_dict()

                # Run model
                result = model_function(param_dict)

                # Store outputs
                for name in output_names:
                    if name in result:
                        outputs[name].append(result[name])
                    else:
                        outputs[name].append(np.nan)

                successful_runs += 1

                if (i + 1) % 100 == 0:
                    pass

            except Exception:
                # Handle failed model runs
                for name in output_names:
                    outputs[name].append(np.nan)
                continue

        # Convert to numpy arrays and remove NaN values
        for name in output_names:
            outputs[name] = np.array(outputs[name])
            outputs[name] = outputs[name][~np.isnan(outputs[name])]

        # Calculate statistics
        uncertainty_results = {
            "parameter_samples": parameter_samples,
            "output_statistics": {},
            "successful_runs": successful_runs,
            "success_rate": successful_runs / len(parameter_samples),
        }

        for name in output_names:
            if len(outputs[name]) > 0:
                data = outputs[name]
                uncertainty_results["output_statistics"][name] = {
                    "mean": np.mean(data),
                    "median": np.median(data),
                    "std": np.std(data),
                    "min": np.min(data),
                    "max": np.max(data),
                    "percentiles": {
                        "5th": np.percentile(data, 5),
                        "25th": np.percentile(data, 25),
                        "75th": np.percentile(data, 75),
                        "95th": np.percentile(data, 95),
                    },
                    "confidence_interval_95": (
                        np.percentile(data, 2.5),
                        np.percentile(data, 97.5),
                    ),
                    "coefficient_of_variation": (
                        np.std(data) / np.mean(data) if np.mean(data) != 0 else np.inf
                    ),
                    "sample_count": len(data),
                }

        self.results = uncertainty_results
        return uncertainty_results

    def sensitivity_analysis(
        self,
        model_function: Callable,
        output_name: str,
    ) -> dict[str, float]:
        """Perform sensitivity analysis using correlation coefficients.

        Args:
            model_function: Model function
            output_name: Name of output variable to analyze

        """
        if not self.results:
            self.propagate_uncertainty(model_function, [output_name])

        parameter_samples = self.results["parameter_samples"]

        # Get output values
        outputs = []
        valid_indices = []

        for i, (_, params) in enumerate(parameter_samples.iterrows()):
            try:
                param_dict = params.to_dict()
                result = model_function(param_dict)

                if output_name in result and not np.isnan(result[output_name]):
                    outputs.append(result[output_name])
                    valid_indices.append(i)
            except (ValueError, TypeError, KeyError, AttributeError):
                continue

        outputs = np.array(outputs)
        valid_samples = parameter_samples.iloc[valid_indices]

        # Calculate sensitivity indices (correlation coefficients)
        sensitivity_indices = {}

        for param_name in self.parameter_uncertainties:
            param_values = valid_samples[param_name].values

            if len(param_values) > 1 and np.std(param_values) > 0:
                correlation = np.corrcoef(param_values, outputs)[0, 1]
                sensitivity_indices[param_name] = (
                    correlation**2
                )  # R-squared as sensitivity measure
            else:
                sensitivity_indices[param_name] = 0.0

        return sensitivity_indices


class UncertaintyQuantificationFramework:
    """Comprehensive uncertainty quantification framework for MFC models."""

    def __init__(self) -> None:
        self.parameter_uncertainties = {}
        self._build_uncertainty_database()

    def _build_uncertainty_database(self) -> None:
        """Build database of parameter uncertainties."""
        # Physics parameters
        self.parameter_uncertainties["fluid_density"] = ParameterUncertainty(
            parameter_name="Fluid density",
            nominal_value=1000.0,
            units="kg/m³",
            distribution_type="normal",
            uncertainty_bounds=(998.0, 1002.0),
            confidence_level=0.95,
            source="Literature + temperature variation",
        )

        self.parameter_uncertainties["dynamic_viscosity"] = ParameterUncertainty(
            parameter_name="Dynamic viscosity",
            nominal_value=0.00089,
            units="Pa·s",
            distribution_type="normal",
            uncertainty_bounds=(0.0008, 0.0011),
            confidence_level=0.90,
            source="Temperature and composition effects",
        )

        self.parameter_uncertainties["substrate_diffusivity"] = ParameterUncertainty(
            parameter_name="Substrate diffusivity",
            nominal_value=1e-9,
            units="m²/s",
            distribution_type="lognormal",
            uncertainty_bounds=(0.8e-9, 1.2e-9),
            confidence_level=0.85,
            source="Molecular size and medium effects",
        )

        self.parameter_uncertainties["electrode_permeability"] = ParameterUncertainty(
            parameter_name="Electrode permeability",
            nominal_value=1e-11,
            units="m²",
            distribution_type="lognormal",
            uncertainty_bounds=(5e-12, 2e-11),
            confidence_level=0.80,
            source="Material variability",
        )

        # Biofilm parameters
        self.parameter_uncertainties["max_biofilm_density"] = ParameterUncertainty(
            parameter_name="Maximum biofilm density",
            nominal_value=80.0,
            units="kg/m³",
            distribution_type="normal",
            uncertainty_bounds=(70.0, 95.0),
            confidence_level=0.90,
            source="Experimental measurements",
        )

        self.parameter_uncertainties["biofilm_yield"] = ParameterUncertainty(
            parameter_name="Biofilm yield coefficient",
            nominal_value=0.4,
            units="g biomass/g substrate",
            distribution_type="normal",
            uncertainty_bounds=(0.30, 0.45),
            confidence_level=0.85,
            source="Metabolic variability",
        )

        self.parameter_uncertainties["decay_rate"] = ParameterUncertainty(
            parameter_name="Biofilm decay rate",
            nominal_value=0.01,
            units="h⁻¹",
            distribution_type="lognormal",
            uncertainty_bounds=(0.005, 0.018),
            confidence_level=0.75,
            source="Operating condition dependence",
        )

        # Metabolic parameters
        self.parameter_uncertainties["max_growth_rate"] = ParameterUncertainty(
            parameter_name="Maximum growth rate",
            nominal_value=0.085,
            units="h⁻¹",
            distribution_type="normal",
            uncertainty_bounds=(0.075, 0.091),
            confidence_level=0.95,
            source="Constraint-based model validation",
        )

        self.parameter_uncertainties["max_lactate_uptake"] = ParameterUncertainty(
            parameter_name="Maximum lactate uptake rate",
            nominal_value=4.11,
            units="mmol/g AFDW/h",
            distribution_type="normal",
            uncertainty_bounds=(3.85, 4.20),
            confidence_level=0.90,
            source="Experimental flux measurements",
        )

        self.parameter_uncertainties["flavin_transfer_efficiency"] = (
            ParameterUncertainty(
                parameter_name="Flavin-mediated electron transfer efficiency",
                nominal_value=0.7,
                units="dimensionless",
                distribution_type="normal",
                uncertainty_bounds=(0.60, 0.75),
                confidence_level=0.85,
                source="Electrochemical measurements",
            )
        )

    def create_simplified_model_function(self):
        """Create simplified model function for uncertainty analysis."""

        def simplified_mfc_model(params: dict[str, float]) -> dict[str, float]:
            """Simplified MFC model for uncertainty propagation.

            This is a reduced-order model that captures the main relationships
            without full computational complexity.
            """
            # Extract parameters
            params.get("fluid_density", 1000.0)
            params.get("dynamic_viscosity", 0.00089)
            diffusivity = params.get("substrate_diffusivity", 1e-9)
            permeability = params.get("electrode_permeability", 1e-11)
            max_biofilm = params.get("max_biofilm_density", 80.0)
            yield_coeff = params.get("biofilm_yield", 0.4)
            growth_rate = params.get("max_growth_rate", 0.085)
            lactate_uptake = params.get("max_lactate_uptake", 4.11)
            flavin_efficiency = params.get("flavin_transfer_efficiency", 0.7)

            # Simplified calculations
            # Mass transport limitation
            transport_factor = min(1.0, diffusivity / 1e-9)

            # Biofilm activity
            biofilm_factor = (max_biofilm / 80.0) * yield_coeff * (growth_rate / 0.085)

            # Metabolic flux
            metabolic_flux = lactate_uptake * flavin_efficiency

            # Electrode performance
            electrode_factor = np.sqrt(permeability / 1e-11)

            # Combined effects
            current_density = (
                (metabolic_flux / 4.11)
                * biofilm_factor
                * transport_factor
                * electrode_factor
                * flavin_efficiency
            )

            # Substrate utilization
            substrate_utilization = min(
                1.0,
                (lactate_uptake / 4.11) * transport_factor * biofilm_factor,
            )

            # Power density (simplified)
            voltage = 0.6 * flavin_efficiency  # Simplified voltage model
            power_density = current_density * voltage

            # Biofilm thickness (time-dependent, simplified)
            biofilm_thickness = min(
                100e-6,
                growth_rate * 24 * 7 * 1e-6,
            )  # 1 week growth

            # System efficiency
            system_efficiency = (
                current_density * substrate_utilization * flavin_efficiency
            )

            return {
                "current_density": current_density,
                "power_density": power_density,
                "substrate_utilization": substrate_utilization,
                "biofilm_thickness": biofilm_thickness,
                "system_efficiency": system_efficiency,
                "voltage": voltage,
            }

        return simplified_mfc_model

    def run_comprehensive_uncertainty_analysis(self) -> dict[str, Any]:
        """Run comprehensive uncertainty quantification analysis."""
        # Create model function
        model_function = self.create_simplified_model_function()

        # Output variables to analyze
        output_variables = [
            "current_density",
            "power_density",
            "substrate_utilization",
            "biofilm_thickness",
            "system_efficiency",
        ]

        # Initialize Monte Carlo analysis
        mc_analysis = MonteCarloAnalysis(self.parameter_uncertainties)

        # Run uncertainty propagation
        uncertainty_results = mc_analysis.propagate_uncertainty(
            model_function,
            output_variables,
        )

        # Run sensitivity analysis for key outputs
        sensitivity_results = {}

        for output_var in ["current_density", "power_density", "system_efficiency"]:
            sensitivity = mc_analysis.sensitivity_analysis(model_function, output_var)
            sensitivity_results[output_var] = sensitivity

        # Compile comprehensive results
        return {
            "uncertainty_analysis": uncertainty_results,
            "sensitivity_analysis": sensitivity_results,
            "parameter_uncertainties": {
                name: {
                    "nominal_value": unc.nominal_value,
                    "units": unc.units,
                    "distribution": unc.distribution_type,
                    "bounds": unc.uncertainty_bounds,
                    "confidence_level": unc.confidence_level,
                    "source": unc.source,
                }
                for name, unc in self.parameter_uncertainties.items()
            },
            "analysis_metadata": {
                "sample_size": mc_analysis.sample_size,
                "output_variables": output_variables,
                "analysis_date": pd.Timestamp.now().isoformat(),
            },
        }

    def generate_uncertainty_summary(self, results: dict[str, Any]) -> dict[str, Any]:
        """Generate uncertainty analysis summary."""
        output_stats = results["uncertainty_analysis"]["output_statistics"]
        sensitivity = results["sensitivity_analysis"]

        summary = {
            "key_findings": [],
            "uncertainty_levels": {},
            "most_sensitive_parameters": {},
            "confidence_intervals": {},
        }

        # Analyze uncertainty levels
        for output_name, stats in output_stats.items():
            cv = stats["coefficient_of_variation"]
            ci_width = (
                stats["confidence_interval_95"][1] - stats["confidence_interval_95"][0]
            )

            summary["uncertainty_levels"][output_name] = {
                "coefficient_of_variation": cv,
                "confidence_interval_width": ci_width,
                "relative_uncertainty": (
                    ci_width / stats["mean"] if stats["mean"] != 0 else np.inf
                ),
            }

            summary["confidence_intervals"][output_name] = stats[
                "confidence_interval_95"
            ]

            # Uncertainty level classification
            if cv < 0.1:
                uncertainty_level = "Low"
            elif cv < 0.3:
                uncertainty_level = "Moderate"
            else:
                uncertainty_level = "High"

            summary["key_findings"].append(
                f"{output_name}: {uncertainty_level} uncertainty (CV = {cv:.2f})",
            )

        # Most sensitive parameters
        for output_name, sens_dict in sensitivity.items():
            sorted_sensitivity = sorted(
                sens_dict.items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            summary["most_sensitive_parameters"][output_name] = sorted_sensitivity[
                :3
            ]  # Top 3

        return summary


if __name__ == "__main__":
    # Example usage

    # Create uncertainty framework
    uq_framework = UncertaintyQuantificationFramework()

    # Run comprehensive analysis
    results = uq_framework.run_comprehensive_uncertainty_analysis()

    # Generate summary
    summary = uq_framework.generate_uncertainty_summary(results)

    # Print key findings
    for _finding in summary["key_findings"]:
        pass

    # Print most sensitive parameters
    for params in summary["most_sensitive_parameters"].values():
        for _i, (_param, _sensitivity) in enumerate(params):
            pass

    # Export results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"uncertainty_analysis_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    summary_file = f"uncertainty_summary_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
