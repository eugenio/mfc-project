"""
Parameter Sensitivity Analysis Framework

This module provides comprehensive tools for analyzing parameter sensitivity
in MFC systems, including local and global sensitivity analysis methods.

Classes:
- SensitivityAnalyzer: Main sensitivity analysis framework
- ParameterSpace: Parameter space definition and sampling
- SensitivityResult: Results container for sensitivity analysis
- SensitivityVisualizer: Visualization tools for sensitivity results

Methods:
- Local Sensitivity Analysis (One-at-a-time, gradient-based)
- Global Sensitivity Analysis (Sobol indices, Morris method)
- Monte Carlo sampling
- Latin Hypercube sampling
- Sensitivity ranking and importance measures

Literature References:
1. Saltelli, A., et al. (2008). "Global Sensitivity Analysis: The Primer"
2. Sobol, I. M. (2001). "Global sensitivity indices for nonlinear mathematical models"
3. Morris, M. D. (1991). "Factorial sampling plans for preliminary computational experiments"
4. Iooss, B., & Lemaître, P. (2015). "A review on global sensitivity analysis methods"
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import qmc

# Import configuration classes
from .visualization_config import VisualizationConfig


class SensitivityMethod(Enum):
    """Available sensitivity analysis methods."""
    ONE_AT_A_TIME = "one_at_a_time"  # Local OAT method
    GRADIENT_BASED = "gradient_based"  # Gradient-based local sensitivity
    MORRIS = "morris"  # Morris elementary effects method
    SOBOL = "sobol"  # Sobol global sensitivity indices
    FAST = "fast"  # Fourier Amplitude Sensitivity Test
    DELTA = "delta"  # Delta moment-independent measure


class SamplingMethod(Enum):
    """Available parameter sampling methods."""
    RANDOM = "random"  # Random sampling
    LATIN_HYPERCUBE = "latin_hypercube"  # Latin Hypercube sampling
    SOBOL_SEQUENCE = "sobol_sequence"  # Sobol sequence sampling
    HALTON = "halton"  # Halton sequence sampling
    GRID = "grid"  # Grid-based sampling


@dataclass
class ParameterBounds:
    """Parameter bounds definition."""
    min_value: float
    max_value: float
    distribution: str = "uniform"  # "uniform", "normal", "lognormal"
    nominal_value: float | None = None

    def __post_init__(self):
        """Validate parameter bounds."""
        if self.min_value >= self.max_value:
            raise ValueError("Minimum value must be less than maximum value")

        if self.nominal_value is not None:
            if not (self.min_value <= self.nominal_value <= self.max_value):
                raise ValueError("Nominal value must be within bounds")


@dataclass
class ParameterDefinition:
    """Complete parameter definition for sensitivity analysis."""
    name: str
    bounds: ParameterBounds
    config_path: list[str]  # Path to parameter in configuration (e.g., ["control", "flow_control", "max_flow_rate"])
    description: str = ""
    units: str = ""
    category: str = "general"  # Parameter category for grouping


@dataclass
class SensitivityResult:
    """Results container for sensitivity analysis."""

    # Method information
    method: SensitivityMethod
    parameter_names: list[str]
    output_names: list[str]

    # Sensitivity indices
    first_order_indices: dict[str, np.ndarray] | None = None  # S1 indices
    total_order_indices: dict[str, np.ndarray] | None = None  # ST indices
    second_order_indices: dict[str, np.ndarray] | None = None  # S2 indices

    # Morris method results
    morris_means: dict[str, np.ndarray] | None = None  # Morris μ values
    morris_stds: dict[str, np.ndarray] | None = None   # Morris σ values
    morris_means_star: dict[str, np.ndarray] | None = None  # Morris μ* values

    # OAT results
    local_sensitivities: dict[str, np.ndarray] | None = None

    # Raw data
    parameter_samples: np.ndarray | None = None
    output_samples: dict[str, np.ndarray] | None = None

    # Metadata
    n_samples: int = 0
    computation_time: float = 0.0
    confidence_intervals: dict[str, tuple[np.ndarray, np.ndarray]] | None = None

    # Analysis metadata
    created_at: str = field(default_factory=lambda: pd.Timestamp.now().isoformat())
    configuration_snapshot: dict[str, Any] | None = None


class ParameterSpace:
    """Parameter space definition and sampling utilities."""

    def __init__(self, parameters: list[ParameterDefinition]):
        """
        Initialize parameter space.

        Args:
            parameters: List of parameter definitions

        Raises:
            ValueError: If parameters list is empty
        """
        if not parameters:
            raise ValueError("Parameter space cannot be empty")

        self.parameters = parameters
        self.parameter_names = [p.name for p in parameters]
        self.n_parameters = len(parameters)

        # Create bounds matrix for sampling
        self.bounds_matrix = np.array([
            [p.bounds.min_value, p.bounds.max_value]
            for p in parameters
        ])

        # Nominal values
        self.nominal_values = np.array([
            p.bounds.nominal_value if p.bounds.nominal_value is not None
            else (p.bounds.min_value + p.bounds.max_value) / 2
            for p in parameters
        ])

        self.logger = logging.getLogger(__name__)

    def sample(self, n_samples: int, method: SamplingMethod = SamplingMethod.LATIN_HYPERCUBE,
               seed: int | None = None) -> np.ndarray:
        """
        Sample parameters from the parameter space.

        Args:
            n_samples: Number of samples to generate
            method: Sampling method
            seed: Random seed for reproducibility

        Returns:
            Array of parameter samples (n_samples x n_parameters)
        """
        if seed is not None:
            np.random.seed(seed)

        if method == SamplingMethod.RANDOM:
            samples = np.random.uniform(0, 1, (n_samples, self.n_parameters))

        elif method == SamplingMethod.LATIN_HYPERCUBE:
            sampler = qmc.LatinHypercube(d=self.n_parameters, seed=seed)
            samples = sampler.random(n_samples)

        elif method == SamplingMethod.SOBOL_SEQUENCE:
            sampler = qmc.Sobol(d=self.n_parameters, seed=seed)
            samples = sampler.random(n_samples)

        elif method == SamplingMethod.HALTON:
            sampler = qmc.Halton(d=self.n_parameters, seed=seed)
            samples = sampler.random(n_samples)

        elif method == SamplingMethod.GRID:
            # Create grid sampling
            n_per_dim = int(np.ceil(n_samples ** (1/self.n_parameters)))
            grid_1d = np.linspace(0, 1, n_per_dim)
            grid_nd = np.meshgrid(*[grid_1d] * self.n_parameters)
            samples = np.column_stack([g.ravel() for g in grid_nd])[:n_samples]

        else:
            raise ValueError(f"Unknown sampling method: {method}")

        # Scale samples to parameter bounds
        scaled_samples = self._scale_samples(samples)

        return scaled_samples

    def _scale_samples(self, unit_samples: np.ndarray) -> np.ndarray:
        """Scale unit samples to parameter bounds."""
        scaled_samples = np.zeros_like(unit_samples)

        for i, param in enumerate(self.parameters):
            bounds = param.bounds

            if bounds.distribution == "uniform":
                scaled_samples[:, i] = (bounds.min_value +
                                       unit_samples[:, i] * (bounds.max_value - bounds.min_value))

            elif bounds.distribution == "normal":
                # Use inverse CDF for normal distribution
                mean = (bounds.min_value + bounds.max_value) / 2
                std = (bounds.max_value - bounds.min_value) / 6  # 3-sigma rule
                from scipy.stats import norm
                scaled_samples[:, i] = norm.ppf(unit_samples[:, i], loc=mean, scale=std)

                # Clip to bounds
                scaled_samples[:, i] = np.clip(scaled_samples[:, i], bounds.min_value, bounds.max_value)

            elif bounds.distribution == "lognormal":
                # Use inverse CDF for lognormal distribution
                from scipy.stats import lognorm
                mu = np.log((bounds.min_value + bounds.max_value) / 2)
                sigma = (np.log(bounds.max_value) - np.log(bounds.min_value)) / 6
                scaled_samples[:, i] = lognorm.ppf(unit_samples[:, i], s=sigma, scale=np.exp(mu))

                # Clip to bounds
                scaled_samples[:, i] = np.clip(scaled_samples[:, i], bounds.min_value, bounds.max_value)

        return scaled_samples

    def get_parameter_by_name(self, name: str) -> ParameterDefinition:
        """Get parameter definition by name."""
        for param in self.parameters:
            if param.name == name:
                return param
        raise ValueError(f"Parameter not found: {name}")


class SensitivityAnalyzer:
    """Main sensitivity analysis framework."""

    def __init__(self, parameter_space: ParameterSpace | None,
                 model_function: Callable[[np.ndarray], dict[str, np.ndarray]] | None,
                 output_names: list[str]):
        """
        Initialize sensitivity analyzer.

        Args:
            parameter_space: Parameter space definition
            model_function: Function that takes parameter array and returns output dict
            output_names: Names of model outputs to analyze
        """
        self.parameter_space = parameter_space
        self.model_function = model_function
        self.output_names = output_names
        self.logger = logging.getLogger(__name__)

        # Cache for model evaluations
        self._model_cache: dict[tuple, dict[str, np.ndarray]] = {}
        self.cache_enabled = True

    def _evaluate_model(self, parameter_samples: np.ndarray) -> dict[str, np.ndarray]:
        """
        Evaluate model with caching support.

        Args:
            parameter_samples: Parameter samples array

        Returns:
            Dictionary of output arrays
        """
        # Check cache if enabled
        if self.cache_enabled:
            cache_key = tuple(parameter_samples.flatten())
            if cache_key in self._model_cache:
                return self._model_cache[cache_key]

        # Evaluate model
        outputs = self.model_function(parameter_samples)

        # Cache results
        if self.cache_enabled:
            cache_key = tuple(parameter_samples.flatten())
            self._model_cache[cache_key] = outputs

        return outputs

    def analyze_sensitivity(self, method: SensitivityMethod,
                          n_samples: int = 1000,
                          sampling_method: SamplingMethod = SamplingMethod.LATIN_HYPERCUBE,
                          **kwargs) -> SensitivityResult:
        """
        Perform sensitivity analysis using specified method.

        Args:
            method: Sensitivity analysis method
            n_samples: Number of samples for analysis
            sampling_method: Parameter sampling method
            **kwargs: Method-specific parameters

        Returns:
            Sensitivity analysis results
        """
        import time
        start_time = time.time()

        self.logger.info(f"Starting {method.value} sensitivity analysis with {n_samples} samples")

        if method == SensitivityMethod.ONE_AT_A_TIME:
            result = self._analyze_one_at_a_time(n_samples, **kwargs)
        elif method == SensitivityMethod.GRADIENT_BASED:
            result = self._analyze_gradient_based(**kwargs)
        elif method == SensitivityMethod.MORRIS:
            result = self._analyze_morris(n_samples, **kwargs)
        elif method == SensitivityMethod.SOBOL:
            result = self._analyze_sobol(n_samples, sampling_method, **kwargs)
        else:
            raise ValueError(f"Method not implemented: {method}")

        # Set metadata
        result.method = method
        result.parameter_names = self.parameter_space.parameter_names
        result.output_names = self.output_names
        result.n_samples = n_samples
        result.computation_time = time.time() - start_time

        self.logger.info(f"Completed sensitivity analysis in {result.computation_time:.2f} seconds")

        return result

    def _analyze_one_at_a_time(self, n_samples: int,
                              perturbation_factor: float = 0.1) -> SensitivityResult:
        """
        Perform One-At-a-Time sensitivity analysis.

        Args:
            n_samples: Number of samples per parameter
            perturbation_factor: Relative perturbation factor

        Returns:
            Sensitivity analysis results
        """
        result = SensitivityResult(
            method=SensitivityMethod.ONE_AT_A_TIME,
            parameter_names=self.parameter_space.parameter_names,
            output_names=self.output_names
        )

        # Baseline evaluation at nominal values
        baseline_outputs = self._evaluate_model(
            self.parameter_space.nominal_values.reshape(1, -1)
        )

        # Initialize sensitivity arrays
        local_sensitivities = {}
        for output_name in self.output_names:
            local_sensitivities[output_name] = np.zeros(self.parameter_space.n_parameters)

        # Analyze each parameter
        for i, param in enumerate(self.parameter_space.parameters):
            # Create perturbed parameter sets
            nominal = self.parameter_space.nominal_values.copy()
            perturbation = perturbation_factor * (param.bounds.max_value - param.bounds.min_value)

            # Positive perturbation
            nominal_plus = nominal.copy()
            nominal_plus[i] = min(param.bounds.max_value, nominal[i] + perturbation)

            # Negative perturbation
            nominal_minus = nominal.copy()
            nominal_minus[i] = max(param.bounds.min_value, nominal[i] - perturbation)

            # Evaluate model
            outputs_plus = self._evaluate_model(nominal_plus.reshape(1, -1))
            outputs_minus = self._evaluate_model(nominal_minus.reshape(1, -1))

            # Calculate local sensitivity
            for output_name in self.output_names:
                if outputs_plus[output_name].size > 0 and outputs_minus[output_name].size > 0:
                    dy = outputs_plus[output_name][0] - outputs_minus[output_name][0]
                    dx = nominal_plus[i] - nominal_minus[i]

                    if dx != 0:
                        # Normalized sensitivity: (dy/y) / (dx/x)
                        y_baseline = baseline_outputs[output_name][0]
                        x_baseline = nominal[i]

                        if y_baseline != 0 and x_baseline != 0:
                            local_sensitivities[output_name][i] = (dy / y_baseline) / (dx / x_baseline)
                        else:
                            local_sensitivities[output_name][i] = dy / dx

        result.local_sensitivities = local_sensitivities
        return result

    def _analyze_gradient_based(self, step_size: float = 1e-6) -> SensitivityResult:
        """
        Perform gradient-based sensitivity analysis.

        Args:
            step_size: Finite difference step size

        Returns:
            Sensitivity analysis results
        """
        result = SensitivityResult(
            method=SensitivityMethod.GRADIENT_BASED,
            parameter_names=self.parameter_space.parameter_names,
            output_names=self.output_names
        )

        # Calculate gradients using finite differences
        gradients = {}
        for output_name in self.output_names:
            gradients[output_name] = np.zeros(self.parameter_space.n_parameters)

        # Baseline evaluation
        baseline_outputs = self._evaluate_model(
            self.parameter_space.nominal_values.reshape(1, -1)
        )

        # Calculate partial derivatives
        for i in range(self.parameter_space.n_parameters):
            # Perturbed parameter set
            perturbed = self.parameter_space.nominal_values.copy()
            param = self.parameter_space.parameters[i]

            # Use relative step size
            step = step_size * (param.bounds.max_value - param.bounds.min_value)
            perturbed[i] = min(param.bounds.max_value, perturbed[i] + step)

            # Evaluate model
            perturbed_outputs = self._evaluate_model(perturbed.reshape(1, -1))

            # Calculate gradient
            for output_name in self.output_names:
                if perturbed_outputs[output_name].size > 0:
                    dy = perturbed_outputs[output_name][0] - baseline_outputs[output_name][0]
                    dx = perturbed[i] - self.parameter_space.nominal_values[i]

                    if dx != 0:
                        gradients[output_name][i] = dy / dx

        result.local_sensitivities = gradients
        return result

    def _analyze_morris(self, n_samples: int, n_levels: int = 4,
                       grid_jump: int = 2) -> SensitivityResult:
        """
        Perform Morris elementary effects method.

        Args:
            n_samples: Number of trajectories
            n_levels: Number of levels for parameter grid
            grid_jump: Grid jump size

        Returns:
            Sensitivity analysis results
        """
        result = SensitivityResult(
            method=SensitivityMethod.MORRIS,
            parameter_names=self.parameter_space.parameter_names,
            output_names=self.output_names
        )

        # Generate Morris trajectories
        trajectories = self._generate_morris_trajectories(n_samples, n_levels, grid_jump)

        # Evaluate model for all trajectories
        all_outputs: dict[str, list[float]] = {}
        for output_name in self.output_names:
            all_outputs[output_name] = []

        for trajectory in trajectories:
            for point in trajectory:
                outputs = self._evaluate_model(point.reshape(1, -1))
                for output_name in self.output_names:
                    all_outputs[output_name].append(outputs[output_name][0])

        # Calculate elementary effects
        morris_means = {}
        morris_stds = {}
        morris_means_star = {}

        for output_name in self.output_names:
            n_params = self.parameter_space.n_parameters
            elementary_effects = np.zeros((n_samples, n_params))

            output_idx = 0
            for traj_idx, trajectory in enumerate(trajectories):
                for i in range(n_params):
                    # Calculate elementary effect
                    dy = all_outputs[output_name][output_idx + i + 1] - all_outputs[output_name][output_idx + i]
                    dx = trajectory[i + 1, i] - trajectory[i, i]

                    if dx != 0:
                        elementary_effects[traj_idx, i] = dy / dx

                output_idx += len(trajectory)

            # Calculate Morris statistics
            morris_means[output_name] = np.mean(elementary_effects, axis=0)
            morris_stds[output_name] = np.std(elementary_effects, axis=0)
            morris_means_star[output_name] = np.mean(np.abs(elementary_effects), axis=0)

        result.morris_means = morris_means
        result.morris_stds = morris_stds
        result.morris_means_star = morris_means_star

        return result

    def _generate_morris_trajectories(self, n_trajectories: int, n_levels: int,
                                    grid_jump: int) -> list[np.ndarray]:
        """Generate Morris trajectories for elementary effects method."""
        trajectories = []
        n_params = self.parameter_space.n_parameters

        for _ in range(n_trajectories):
            # Start with random base point on grid
            base_point = np.random.randint(0, n_levels - grid_jump, n_params)

            # Create trajectory
            trajectory = np.zeros((n_params + 1, n_params))
            trajectory[0] = base_point

            # Randomly order parameters
            param_order = np.random.permutation(n_params)

            # Build trajectory
            for i, param_idx in enumerate(param_order):
                trajectory[i + 1] = trajectory[i].copy()
                trajectory[i + 1, param_idx] += grid_jump

            # Scale trajectory to parameter bounds
            scaled_trajectory = np.zeros_like(trajectory, dtype=float)
            for i, param in enumerate(self.parameter_space.parameters):
                scaled_trajectory[:, i] = (param.bounds.min_value +
                                         trajectory[:, i] * (param.bounds.max_value - param.bounds.min_value) / (n_levels - 1))

            trajectories.append(scaled_trajectory)

        return trajectories

    def _analyze_sobol(self, n_samples: int,
                      sampling_method: SamplingMethod = SamplingMethod.SOBOL_SEQUENCE,
                      calc_second_order: bool = False) -> SensitivityResult:
        """
        Perform Sobol global sensitivity analysis.

        Args:
            n_samples: Number of samples
            sampling_method: Sampling method
            calc_second_order: Whether to calculate second-order indices

        Returns:
            Sensitivity analysis results
        """
        result = SensitivityResult(
            method=SensitivityMethod.SOBOL,
            parameter_names=self.parameter_space.parameter_names,
            output_names=self.output_names
        )

        # Generate sample matrices for Sobol analysis
        # We need: A, B, and C_i matrices (Saltelli et al., 2008)
        n_params = self.parameter_space.n_parameters

        # Generate base samples
        if sampling_method == SamplingMethod.SOBOL_SEQUENCE:
            sampler = qmc.Sobol(d=n_params, scramble=True)
            unit_samples_A = sampler.random(n_samples)
            unit_samples_B = sampler.random(n_samples)
        else:
            unit_samples_A = np.random.uniform(0, 1, (n_samples, n_params))
            unit_samples_B = np.random.uniform(0, 1, (n_samples, n_params))

        # Scale samples
        samples_A = self.parameter_space._scale_samples(unit_samples_A)
        samples_B = self.parameter_space._scale_samples(unit_samples_B)

        # Generate C_i matrices (A with column i from B)
        samples_C = []
        for i in range(n_params):
            C_i = samples_A.copy()
            C_i[:, i] = samples_B[:, i]
            samples_C.append(C_i)

        # Evaluate model for all sample sets
        outputs_A = self._evaluate_model(samples_A)
        outputs_B = self._evaluate_model(samples_B)
        outputs_C = [self._evaluate_model(C_i) for C_i in samples_C]

        # Calculate Sobol indices
        first_order = {}
        total_order = {}

        for output_name in self.output_names:
            if output_name not in outputs_A:
                continue

            y_A = outputs_A[output_name]
            y_B = outputs_B[output_name]
            y_C = [outputs_C[i][output_name] for i in range(n_params)]

            # Handle multi-dimensional outputs (take mean if needed)
            if y_A.ndim > 1:
                y_A = np.mean(y_A, axis=tuple(range(1, y_A.ndim)))
                y_B = np.mean(y_B, axis=tuple(range(1, y_B.ndim)))
                y_C = [np.mean(y_c, axis=tuple(range(1, y_c.ndim))) for y_c in y_C]

            # Calculate variance
            y_all = np.concatenate([y_A, y_B] + y_C)
            var_y = np.var(y_all)

            if var_y == 0:
                # No variance in output
                first_order[output_name] = np.zeros(n_params)
                total_order[output_name] = np.zeros(n_params)
                continue

            # First-order indices: S_i = V_i / V
            s1 = np.zeros(n_params)
            for i in range(n_params):
                v_i = np.mean(y_B * (y_C[i] - y_A))
                s1[i] = v_i / var_y

            # Total-order indices: ST_i = 1 - V_{~i} / V
            st = np.zeros(n_params)
            for i in range(n_params):
                v_not_i = np.mean(y_A * (y_A - y_C[i]))
                st[i] = 1 - v_not_i / var_y

            first_order[output_name] = np.clip(s1, 0, 1)  # Clip to valid range
            total_order[output_name] = np.clip(st, 0, 1)

        result.first_order_indices = first_order
        result.total_order_indices = total_order
        result.parameter_samples = np.vstack([samples_A, samples_B] + samples_C)
        result.output_samples = {}

        for output_name in self.output_names:
            if output_name in outputs_A:
                result.output_samples[output_name] = np.concatenate([
                    outputs_A[output_name],
                    outputs_B[output_name]
                ] + [outputs_C[i][output_name] for i in range(n_params)])

        return result

    def rank_parameters(self, result: SensitivityResult,
                       output_name: str, metric: str = "total_order") -> list[tuple[str, float]]:
        """
        Rank parameters by sensitivity importance.

        Args:
            result: Sensitivity analysis results
            output_name: Name of output to analyze
            metric: Sensitivity metric to use for ranking

        Returns:
            List of (parameter_name, sensitivity_value) tuples, sorted by importance
        """
        if metric == "total_order" and result.total_order_indices:
            values = result.total_order_indices[output_name]
        elif metric == "first_order" and result.first_order_indices:
            values = result.first_order_indices[output_name]
        elif metric == "morris_mean_star" and result.morris_means_star:
            values = result.morris_means_star[output_name]
        elif metric == "local" and result.local_sensitivities:
            values = np.abs(result.local_sensitivities[output_name])
        else:
            raise ValueError(f"Metric '{metric}' not available in results")

        # Create ranking
        ranking = list(zip(result.parameter_names, values, strict=False))
        ranking.sort(key=lambda x: abs(x[1]), reverse=True)

        return ranking


class SensitivityVisualizer:
    """Visualization tools for sensitivity analysis results."""

    def __init__(self, visualization_config: VisualizationConfig | None = None):
        """
        Initialize sensitivity visualizer.

        Args:
            visualization_config: Visualization configuration
        """
        self.config = visualization_config or VisualizationConfig()

        # Apply style configuration
        from .visualization_config import apply_style_config, get_colors_for_scheme
        apply_style_config(self.config.plot_style)
        self.colors = get_colors_for_scheme(self.config.color_scheme_type, self.config.color_scheme)

    def plot_sensitivity_indices(self, result: SensitivityResult,
                                output_name: str, save_path: str | None = None) -> plt.Figure:
        """
        Plot sensitivity indices (first-order and total-order).

        Args:
            result: Sensitivity analysis results
            output_name: Output to visualize
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        param_names = result.parameter_names
        x_pos = np.arange(len(param_names))

        # Plot first-order indices
        if result.first_order_indices and output_name in result.first_order_indices:
            s1 = result.first_order_indices[output_name]
            ax.bar(x_pos - 0.2, s1, 0.4, label='First-order (S₁)',
                  color=self.colors[0], alpha=0.8)

        # Plot total-order indices
        if result.total_order_indices and output_name in result.total_order_indices:
            st = result.total_order_indices[output_name]
            ax.bar(x_pos + 0.2, st, 0.4, label='Total-order (Sₜ)',
                  color=self.colors[1], alpha=0.8)

        ax.set_xlabel('Parameters')
        ax.set_ylabel('Sensitivity Index')
        ax.set_title(f'Sobol Sensitivity Indices - {output_name}')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(param_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_morris_results(self, result: SensitivityResult,
                           output_name: str, save_path: str | None = None) -> plt.Figure:
        """
        Plot Morris method results (μ* vs σ).

        Args:
            result: Sensitivity analysis results
            output_name: Output to visualize
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if not (result.morris_means_star and result.morris_stds):
            raise ValueError("Morris results not available")

        fig, ax = plt.subplots(figsize=(10, 8))

        mu_star = result.morris_means_star[output_name]
        sigma = result.morris_stds[output_name]
        param_names = result.parameter_names

        # Scatter plot
        ax.scatter(mu_star, sigma, c=self.colors[0],
                  s=100, alpha=0.7, edgecolors='black')

        # Add parameter labels
        for i, name in enumerate(param_names):
            ax.annotate(name, (mu_star[i], sigma[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, alpha=0.8)

        ax.set_xlabel('μ* (Mean of Absolute Elementary Effects)')
        ax.set_ylabel('σ (Standard Deviation of Elementary Effects)')
        ax.set_title(f'Morris Method Results - {output_name}')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_parameter_ranking(self, result: SensitivityResult,
                              output_name: str, metric: str = "total_order",
                              top_n: int | None = None,
                              save_path: str | None = None) -> plt.Figure:
        """
        Plot parameter ranking by sensitivity.

        Args:
            result: Sensitivity analysis results
            output_name: Output to visualize
            metric: Sensitivity metric for ranking
            top_n: Number of top parameters to show
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        # Get ranking
        analyzer = SensitivityAnalyzer(None, None, [])  # Dummy instance for ranking
        ranking = analyzer.rank_parameters(result, output_name, metric)

        if top_n:
            ranking = ranking[:top_n]

        fig, ax = plt.subplots(figsize=(12, 6))

        param_names = [r[0] for r in ranking]
        values = [abs(r[1]) for r in ranking]

        bars = ax.barh(range(len(param_names)), values,
                      color=self.colors[0], alpha=0.8)

        ax.set_yticks(range(len(param_names)))
        ax.set_yticklabels(param_names)
        ax.set_xlabel(f'Sensitivity ({metric.replace("_", " ").title()})')
        ax.set_title(f'Parameter Ranking - {output_name}')
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values, strict=False)):
            ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}', va='center', fontsize=9)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig
