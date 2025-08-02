"""
Uncertainty Quantification Framework

This module provides comprehensive uncertainty quantification methods for MFC systems,
including Monte Carlo methods, polynomial chaos expansion, and Bayesian inference.

Classes:
- UncertaintyQuantifier: Main uncertainty quantification framework
- MonteCarloAnalyzer: Monte Carlo uncertainty propagation
- PolynomialChaosAnalyzer: Polynomial Chaos Expansion methods
- BayesianInference: Bayesian parameter estimation and model updating
- UncertaintyResult: Results container for uncertainty analysis

Features:
- Forward uncertainty propagation
- Inverse uncertainty quantification (parameter estimation)
- Sensitivity-based uncertainty reduction
- Confidence interval estimation
- Model validation under uncertainty

Literature References:
1. Xiu, D. (2010). "Numerical Methods for Stochastic Computations"
2. Smith, R. C. (2013). "Uncertainty Quantification: Theory, Implementation, and Applications"
3. Ghanem, R., et al. (2017). "Handbook of Uncertainty Quantification"
4. Kennedy, M. C., & O'Hagan, A. (2001). "Bayesian calibration of computer models"
"""

import logging
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

# Optional dependencies
try:
    from scipy.linalg import cholesky
    from scipy.optimize import minimize
    from scipy.stats import gaussian_kde, multivariate_normal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("SciPy not available. Some uncertainty quantification features will be limited.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    warnings.warn("Matplotlib/Seaborn not available. Plotting features will be limited.")

# Import configuration classes


class UncertaintyMethod(Enum):
    """Available uncertainty quantification methods."""
    MONTE_CARLO = "monte_carlo"
    LATIN_HYPERCUBE = "latin_hypercube"
    POLYNOMIAL_CHAOS = "polynomial_chaos"
    BAYESIAN_INFERENCE = "bayesian_inference"
    BOOTSTRAP = "bootstrap"
    QUASI_MONTE_CARLO = "quasi_monte_carlo"


class DistributionType(Enum):
    """Supported probability distributions."""
    NORMAL = "normal"
    UNIFORM = "uniform"
    LOGNORMAL = "lognormal"
    BETA = "beta"
    GAMMA = "gamma"
    EXPONENTIAL = "exponential"
    TRIANGULAR = "triangular"


@dataclass
class UncertainParameter:
    """Definition of uncertain parameter with probability distribution."""
    name: str
    distribution: DistributionType
    parameters: Dict[str, float]  # Distribution parameters (e.g., {'mean': 1.0, 'std': 0.1})
    bounds: Optional[Tuple[float, float]] = None
    description: str = ""

    def sample(self, n_samples: int, seed: Optional[int] = None) -> np.ndarray:
        """Sample from the parameter distribution."""
        if seed is not None:
            np.random.seed(seed)

        if self.distribution == DistributionType.NORMAL:
            samples = np.random.normal(
                self.parameters['mean'],
                self.parameters['std'],
                n_samples
            )
        elif self.distribution == DistributionType.UNIFORM:
            samples = np.random.uniform(
                self.parameters['low'],
                self.parameters['high'],
                n_samples
            )
        elif self.distribution == DistributionType.LOGNORMAL:
            samples = np.random.lognormal(
                self.parameters['mean'],
                self.parameters['sigma'],
                n_samples
            )
        elif self.distribution == DistributionType.BETA:
            samples = np.random.beta(
                self.parameters['alpha'],
                self.parameters['beta'],
                n_samples
            )
        elif self.distribution == DistributionType.GAMMA:
            samples = np.random.gamma(
                self.parameters['shape'],
                self.parameters['scale'],
                n_samples
            )
        elif self.distribution == DistributionType.EXPONENTIAL:
            samples = np.random.exponential(
                self.parameters['scale'],
                n_samples
            )
        elif self.distribution == DistributionType.TRIANGULAR:
            samples = np.random.triangular(
                self.parameters['left'],
                self.parameters['mode'],
                self.parameters['right'],
                n_samples
            )
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")

        # Apply bounds if specified
        if self.bounds is not None:
            samples = np.clip(samples, self.bounds[0], self.bounds[1])

        return samples

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Calculate probability density function."""
        if self.distribution == DistributionType.NORMAL:
            return stats.norm.pdf(x, self.parameters['mean'], self.parameters['std'])
        elif self.distribution == DistributionType.UNIFORM:
            return stats.uniform.pdf(x, self.parameters['low'],
                                   self.parameters['high'] - self.parameters['low'])
        elif self.distribution == DistributionType.LOGNORMAL:
            return stats.lognorm.pdf(x, self.parameters['sigma'],
                                   scale=np.exp(self.parameters['mean']))
        # Add other distributions as needed
        else:
            raise NotImplementedError(f"PDF not implemented for {self.distribution}")


@dataclass
class UncertaintyResult:
    """Results container for uncertainty quantification analysis."""

    # Method information
    method: UncertaintyMethod
    parameter_names: List[str]
    output_names: List[str]
    n_samples: int

    # Input samples and outputs
    parameter_samples: Optional[np.ndarray] = None
    output_samples: Optional[Dict[str, np.ndarray]] = None

    # Statistical measures
    output_mean: Optional[Dict[str, float]] = None
    output_std: Optional[Dict[str, float]] = None
    output_var: Optional[Dict[str, float]] = None
    output_skewness: Optional[Dict[str, float]] = None
    output_kurtosis: Optional[Dict[str, float]] = None

    # Confidence intervals
    confidence_intervals: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None

    # Percentiles
    percentiles: Optional[Dict[str, Dict[str, float]]] = None

    # Correlation analysis
    parameter_correlations: Optional[np.ndarray] = None
    output_correlations: Optional[np.ndarray] = None
    parameter_output_correlations: Optional[Dict[str, np.ndarray]] = None

    # Bayesian inference results
    posterior_samples: Optional[np.ndarray] = None
    evidence: Optional[float] = None
    bayes_factor: Optional[float] = None

    # Polynomial chaos expansion results
    pce_coefficients: Optional[Dict[str, np.ndarray]] = None
    pce_variance_contributions: Optional[Dict[str, np.ndarray]] = None

    # Computation metadata
    computation_time: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


class UncertaintyQuantifier(ABC):
    """Abstract base class for uncertainty quantification methods."""

    def __init__(self, uncertain_parameters: List[UncertainParameter],
                 random_seed: Optional[int] = None):
        """
        Initialize uncertainty quantifier.
        
        Args:
            uncertain_parameters: List of uncertain parameter definitions
            random_seed: Random seed for reproducibility
        """
        self.uncertain_parameters = uncertain_parameters
        self.parameter_names = [p.name for p in uncertain_parameters]
        self.n_parameters = len(uncertain_parameters)
        self.random_seed = random_seed
        self.logger = logging.getLogger(__name__)

        if random_seed is not None:
            np.random.seed(random_seed)

    @abstractmethod
    def propagate_uncertainty(self, model_function: Callable[[np.ndarray], Dict[str, float]],
                            n_samples: int,
                            **kwargs) -> UncertaintyResult:
        """
        Propagate parameter uncertainty through model.
        
        Args:
            model_function: Model function to evaluate
            n_samples: Number of samples for uncertainty propagation
            **kwargs: Method-specific parameters
            
        Returns:
            Uncertainty quantification results
        """
        pass

    def _sample_parameters(self, n_samples: int) -> np.ndarray:
        """Sample from all uncertain parameters."""
        samples = np.zeros((n_samples, self.n_parameters))

        for i, param in enumerate(self.uncertain_parameters):
            samples[:, i] = param.sample(n_samples, seed=self.random_seed)

        return samples

    def _evaluate_model_batch(self, model_function: Callable,
                            parameter_samples: np.ndarray,
                            parallel: bool = True) -> Dict[str, np.ndarray]:
        """Evaluate model for batch of parameter samples."""
        output_names = None
        all_outputs = {}

        if parallel and len(parameter_samples) > 10:
            # Parallel evaluation
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(model_function, params)
                          for params in parameter_samples]

                for i, future in enumerate(as_completed(futures)):
                    try:
                        result = future.result()
                        if output_names is None:
                            output_names = list(result.keys())
                            for name in output_names:
                                all_outputs[name] = np.zeros(len(parameter_samples))

                        # Find the index of this future
                        future_idx = futures.index(future)
                        for name in output_names:
                            all_outputs[name][future_idx] = result[name]

                    except Exception as e:
                        self.logger.warning(f"Model evaluation failed: {e}")
                        # Set to NaN for failed evaluations
                        if output_names is not None:
                            for name in output_names:
                                all_outputs[name][i] = np.nan
        else:
            # Sequential evaluation
            for i, params in enumerate(parameter_samples):
                try:
                    result = model_function(params)
                    if output_names is None:
                        output_names = list(result.keys())
                        for name in output_names:
                            all_outputs[name] = np.zeros(len(parameter_samples))

                    for name in output_names:
                        all_outputs[name][i] = result[name]

                except Exception as e:
                    self.logger.warning(f"Model evaluation failed at sample {i}: {e}")
                    if output_names is not None:
                        for name in output_names:
                            all_outputs[name][i] = np.nan

        return all_outputs

    def _calculate_statistics(self, output_samples: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Calculate statistical measures for outputs."""
        statistics = {}

        for output_name, samples in output_samples.items():
            # Remove NaN values
            valid_samples = samples[~np.isnan(samples)]

            if len(valid_samples) == 0:
                statistics[output_name] = {
                    'mean': np.nan, 'std': np.nan, 'var': np.nan,
                    'skewness': np.nan, 'kurtosis': np.nan
                }
            else:
                statistics[output_name] = {
                    'mean': np.mean(valid_samples),
                    'std': np.std(valid_samples),
                    'var': np.var(valid_samples),
                    'skewness': stats.skew(valid_samples),
                    'kurtosis': stats.kurtosis(valid_samples)
                }

        return statistics

    def _calculate_confidence_intervals(self, output_samples: Dict[str, np.ndarray],
                                      confidence_levels: List[float] = [0.95, 0.99]) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Calculate confidence intervals for outputs."""
        intervals = {}

        for output_name, samples in output_samples.items():
            valid_samples = samples[~np.isnan(samples)]
            intervals[output_name] = {}

            for level in confidence_levels:
                if len(valid_samples) > 0:
                    alpha = 1 - level
                    lower = np.percentile(valid_samples, 100 * alpha / 2)
                    upper = np.percentile(valid_samples, 100 * (1 - alpha / 2))
                    intervals[output_name][f'{level:.2f}'] = (lower, upper)
                else:
                    intervals[output_name][f'{level:.2f}'] = (np.nan, np.nan)

        return intervals

    def _calculate_percentiles(self, output_samples: Dict[str, np.ndarray],
                             percentile_values: List[float] = [5, 25, 50, 75, 95]) -> Dict[str, Dict[str, float]]:
        """Calculate percentiles for outputs."""
        percentiles = {}

        for output_name, samples in output_samples.items():
            valid_samples = samples[~np.isnan(samples)]
            percentiles[output_name] = {}

            for p in percentile_values:
                if len(valid_samples) > 0:
                    percentiles[output_name][f'p{p}'] = np.percentile(valid_samples, p)
                else:
                    percentiles[output_name][f'p{p}'] = np.nan

        return percentiles


class MonteCarloAnalyzer(UncertaintyQuantifier):
    """Monte Carlo uncertainty quantification."""

    def __init__(self, uncertain_parameters: List[UncertainParameter],
                 sampling_method: str = "random",
                 random_seed: Optional[int] = None):
        """
        Initialize Monte Carlo analyzer.
        
        Args:
            uncertain_parameters: List of uncertain parameter definitions
            sampling_method: Sampling method ("random", "latin_hypercube", "sobol")
            random_seed: Random seed for reproducibility
        """
        super().__init__(uncertain_parameters, random_seed)
        self.sampling_method = sampling_method

    def propagate_uncertainty(self, model_function: Callable[[np.ndarray], Dict[str, float]],
                            n_samples: int,
                            parallel: bool = True,
                            **kwargs) -> UncertaintyResult:
        """
        Perform Monte Carlo uncertainty propagation.
        
        Args:
            model_function: Model function to evaluate
            n_samples: Number of Monte Carlo samples
            parallel: Whether to use parallel evaluation
            **kwargs: Additional parameters
            
        Returns:
            Uncertainty quantification results
        """
        import time
        start_time = time.time()

        # Generate parameter samples
        if self.sampling_method == "latin_hypercube":
            parameter_samples = self._latin_hypercube_sampling(n_samples)
        elif self.sampling_method == "sobol":
            parameter_samples = self._sobol_sampling(n_samples)
        else:
            parameter_samples = self._sample_parameters(n_samples)

        # Evaluate model
        output_samples = self._evaluate_model_batch(model_function, parameter_samples, parallel)

        # Calculate statistics
        stats_dict = self._calculate_statistics(output_samples)

        # Create result object
        result = UncertaintyResult(
            method=UncertaintyMethod.MONTE_CARLO,
            parameter_names=self.parameter_names,
            output_names=list(output_samples.keys()),
            n_samples=n_samples,
            parameter_samples=parameter_samples,
            output_samples=output_samples,
            computation_time=time.time() - start_time
        )

        # Extract statistics
        result.output_mean = {name: stats_dict[name]['mean'] for name in stats_dict}
        result.output_std = {name: stats_dict[name]['std'] for name in stats_dict}
        result.output_var = {name: stats_dict[name]['var'] for name in stats_dict}
        result.output_skewness = {name: stats_dict[name]['skewness'] for name in stats_dict}
        result.output_kurtosis = {name: stats_dict[name]['kurtosis'] for name in stats_dict}

        # Calculate confidence intervals and percentiles
        result.confidence_intervals = self._calculate_confidence_intervals(output_samples)
        result.percentiles = self._calculate_percentiles(output_samples)

        # Calculate correlations
        result.parameter_correlations = np.corrcoef(parameter_samples.T)

        output_matrix = np.column_stack([output_samples[name] for name in result.output_names])
        result.output_correlations = np.corrcoef(output_matrix.T)

        # Parameter-output correlations
        result.parameter_output_correlations = {}
        for i, output_name in enumerate(result.output_names):
            result.parameter_output_correlations[output_name] = np.corrcoef(
                parameter_samples.T, output_samples[output_name]
            )[:self.n_parameters, -1]

        return result

    def _latin_hypercube_sampling(self, n_samples: int) -> np.ndarray:
        """Generate Latin Hypercube samples."""
        from scipy.stats import qmc

        sampler = qmc.LatinHypercube(d=self.n_parameters, seed=self.random_seed)
        unit_samples = sampler.random(n_samples)

        # Transform to parameter distributions
        samples = np.zeros_like(unit_samples)
        for i, param in enumerate(self.uncertain_parameters):
            if param.distribution == DistributionType.NORMAL:
                samples[:, i] = stats.norm.ppf(unit_samples[:, i],
                                             param.parameters['mean'],
                                             param.parameters['std'])
            elif param.distribution == DistributionType.UNIFORM:
                samples[:, i] = param.parameters['low'] + unit_samples[:, i] * \
                               (param.parameters['high'] - param.parameters['low'])
            # Add other distributions as needed

        return samples

    def _sobol_sampling(self, n_samples: int) -> np.ndarray:
        """Generate Sobol sequence samples."""
        from scipy.stats import qmc

        sampler = qmc.Sobol(d=self.n_parameters, scramble=True, seed=self.random_seed)
        unit_samples = sampler.random(n_samples)

        # Transform to parameter distributions (similar to Latin Hypercube)
        samples = np.zeros_like(unit_samples)
        for i, param in enumerate(self.uncertain_parameters):
            if param.distribution == DistributionType.NORMAL:
                samples[:, i] = stats.norm.ppf(unit_samples[:, i],
                                             param.parameters['mean'],
                                             param.parameters['std'])
            elif param.distribution == DistributionType.UNIFORM:
                samples[:, i] = param.parameters['low'] + unit_samples[:, i] * \
                               (param.parameters['high'] - param.parameters['low'])

        return samples


class BayesianInference(UncertaintyQuantifier):
    """Bayesian inference for parameter estimation and model updating."""

    def __init__(self, uncertain_parameters: List[UncertainParameter],
                 prior_distributions: Optional[List[UncertainParameter]] = None,
                 random_seed: Optional[int] = None):
        """
        Initialize Bayesian inference.
        
        Args:
            uncertain_parameters: List of uncertain parameter definitions
            prior_distributions: Prior distributions for parameters
            random_seed: Random seed for reproducibility
        """
        super().__init__(uncertain_parameters, random_seed)
        self.prior_distributions = prior_distributions or uncertain_parameters

    def calibrate_parameters(self, model_function: Callable[[np.ndarray], Dict[str, float]],
                           experimental_data: Dict[str, np.ndarray],
                           observation_noise: Dict[str, float],
                           n_samples: int = 10000,
                           n_chains: int = 4,
                           algorithm: str = "metropolis_hastings") -> UncertaintyResult:
        """
        Perform Bayesian parameter calibration.
        
        Args:
            model_function: Model function to calibrate
            experimental_data: Experimental observations
            observation_noise: Observation noise standard deviations
            n_samples: Number of MCMC samples
            n_chains: Number of MCMC chains
            algorithm: MCMC algorithm ("metropolis_hastings", "gibbs")
            
        Returns:
            Calibration results with posterior samples
        """
        import time
        start_time = time.time()

        # Define log-likelihood function
        def log_likelihood(parameters):
            try:
                model_output = model_function(parameters)
                ll = 0.0

                for output_name, exp_data in experimental_data.items():
                    if output_name in model_output:
                        model_value = model_output[output_name]
                        noise = observation_noise.get(output_name, 1.0)

                        # Gaussian likelihood
                        ll += -0.5 * np.sum((exp_data - model_value)**2 / noise**2)
                        ll += -0.5 * len(exp_data) * np.log(2 * np.pi * noise**2)

                return ll
            except Exception:
                return -np.inf

        # Define log-prior function
        def log_prior(parameters):
            lp = 0.0
            for i, (param, prior) in enumerate(zip(parameters, self.prior_distributions)):
                lp += prior.pdf(np.array([param]))[0]
            return np.log(lp) if lp > 0 else -np.inf

        # MCMC sampling
        if algorithm == "metropolis_hastings":
            posterior_samples = self._metropolis_hastings_sampling(
                log_likelihood, log_prior, n_samples, n_chains
            )
        else:
            raise NotImplementedError(f"Algorithm {algorithm} not implemented")

        # Calculate model evidence (marginal likelihood)
        evidence = self._estimate_evidence(posterior_samples, log_likelihood, log_prior)

        # Create result
        result = UncertaintyResult(
            method=UncertaintyMethod.BAYESIAN_INFERENCE,
            parameter_names=self.parameter_names,
            output_names=list(experimental_data.keys()),
            n_samples=n_samples,
            posterior_samples=posterior_samples,
            evidence=evidence,
            computation_time=time.time() - start_time
        )

        return result

    def _metropolis_hastings_sampling(self, log_likelihood: Callable, log_prior: Callable,
                                    n_samples: int, n_chains: int) -> np.ndarray:
        """Metropolis-Hastings MCMC sampling."""
        all_samples = []

        for chain in range(n_chains):
            samples = np.zeros((n_samples, self.n_parameters))

            # Initialize chain
            current_params = self._sample_parameters(1)[0]
            current_ll = log_likelihood(current_params)
            current_lp = log_prior(current_params)
            current_posterior = current_ll + current_lp

            n_accepted = 0

            for i in range(n_samples):
                # Propose new parameters
                proposal_cov = 0.1 * np.eye(self.n_parameters)  # Simple proposal
                proposed_params = np.random.multivariate_normal(current_params, proposal_cov)

                # Calculate posterior for proposed parameters
                proposed_ll = log_likelihood(proposed_params)
                proposed_lp = log_prior(proposed_params)
                proposed_posterior = proposed_ll + proposed_lp

                # Acceptance probability
                alpha = min(1.0, np.exp(proposed_posterior - current_posterior))

                # Accept or reject
                if np.random.random() < alpha:
                    current_params = proposed_params
                    current_posterior = proposed_posterior
                    n_accepted += 1

                samples[i] = current_params

            self.logger.info(f"Chain {chain}: Acceptance rate = {n_accepted/n_samples:.3f}")
            all_samples.append(samples)

        # Combine all chains
        return np.vstack(all_samples)

    def _estimate_evidence(self, posterior_samples: np.ndarray,
                         log_likelihood: Callable, log_prior: Callable) -> float:
        """Estimate model evidence using harmonic mean estimator."""
        n_samples = len(posterior_samples)
        log_likelihoods = np.array([log_likelihood(params) for params in posterior_samples])

        # Harmonic mean estimator (can be unstable)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            evidence = n_samples / np.sum(1.0 / np.exp(log_likelihoods))

        return np.log(evidence) if evidence > 0 else -np.inf


class PolynomialChaosAnalyzer(UncertaintyQuantifier):
    """Polynomial Chaos Expansion for uncertainty quantification."""

    def __init__(self, uncertain_parameters: List[UncertainParameter],
                 polynomial_order: int = 3,
                 random_seed: Optional[int] = None):
        """
        Initialize Polynomial Chaos analyzer.
        
        Args:
            uncertain_parameters: List of uncertain parameter definitions
            polynomial_order: Order of polynomial expansion
            random_seed: Random seed for reproducibility
        """
        super().__init__(uncertain_parameters, random_seed)
        self.polynomial_order = polynomial_order

    def propagate_uncertainty(self, model_function: Callable[[np.ndarray], Dict[str, float]],
                            n_samples: int = None,
                            **kwargs) -> UncertaintyResult:
        """
        Perform uncertainty propagation using Polynomial Chaos Expansion.
        
        Args:
            model_function: Model function to evaluate
            n_samples: Number of samples for PCE construction
            **kwargs: Additional parameters
            
        Returns:
            PCE-based uncertainty results
        """
        import time
        start_time = time.time()

        # Determine number of samples based on polynomial order
        if n_samples is None:
            n_samples = int(2 * self._calculate_n_terms())

        # Generate quadrature points or samples
        parameter_samples = self._generate_pce_samples(n_samples)

        # Evaluate model at sample points
        output_samples = self._evaluate_model_batch(model_function, parameter_samples)

        # Build PCE for each output
        pce_coefficients = {}
        variance_contributions = {}

        for output_name, outputs in output_samples.items():
            # Construct polynomial basis
            basis_matrix = self._construct_polynomial_basis(parameter_samples)

            # Solve for PCE coefficients using least squares
            coefficients = np.linalg.lstsq(basis_matrix, outputs, rcond=None)[0]
            pce_coefficients[output_name] = coefficients

            # Calculate variance contributions
            var_contrib = self._calculate_variance_contributions(coefficients)
            variance_contributions[output_name] = var_contrib

        # Generate statistics from PCE
        pce_statistics = self._calculate_pce_statistics(pce_coefficients, variance_contributions)

        # Create result
        result = UncertaintyResult(
            method=UncertaintyMethod.POLYNOMIAL_CHAOS,
            parameter_names=self.parameter_names,
            output_names=list(output_samples.keys()),
            n_samples=n_samples,
            parameter_samples=parameter_samples,
            output_samples=output_samples,
            pce_coefficients=pce_coefficients,
            pce_variance_contributions=variance_contributions,
            computation_time=time.time() - start_time
        )

        # Set statistics from PCE
        result.output_mean = pce_statistics['mean']
        result.output_var = pce_statistics['variance']
        result.output_std = {name: np.sqrt(var) for name, var in pce_statistics['variance'].items()}

        return result

    def _calculate_n_terms(self) -> int:
        """Calculate number of terms in PCE expansion."""
        from math import factorial
        n = self.n_parameters
        p = self.polynomial_order
        return factorial(n + p) // (factorial(n) * factorial(p))

    def _generate_pce_samples(self, n_samples: int) -> np.ndarray:
        """Generate samples for PCE construction."""
        # For simplicity, use random sampling
        # In practice, quadrature rules or sparse grids would be better
        return self._sample_parameters(n_samples)

    def _construct_polynomial_basis(self, parameter_samples: np.ndarray) -> np.ndarray:
        """Construct polynomial basis matrix."""
        n_samples, n_params = parameter_samples.shape
        n_terms = self._calculate_n_terms()

        basis_matrix = np.ones((n_samples, n_terms))

        # Generate multi-indices for polynomial terms
        multi_indices = self._generate_multi_indices()

        for i, multi_index in enumerate(multi_indices):
            if i == 0:  # Constant term
                continue

            basis_column = np.ones(n_samples)
            for j, order in enumerate(multi_index):
                if order > 0:
                    # Use Hermite polynomials for normal distributions
                    # For simplicity, use power basis here
                    basis_column *= parameter_samples[:, j] ** order

            basis_matrix[:, i] = basis_column

        return basis_matrix

    def _generate_multi_indices(self) -> List[List[int]]:
        """Generate multi-indices for polynomial expansion."""
        multi_indices = []

        # Generate all combinations up to polynomial order
        def generate_recursive(current_index, remaining_order, param_idx):
            if param_idx == self.n_parameters:
                if remaining_order >= 0:
                    multi_indices.append(current_index.copy())
                return

            for order in range(remaining_order + 1):
                current_index.append(order)
                generate_recursive(current_index, remaining_order - order, param_idx + 1)
                current_index.pop()

        generate_recursive([], self.polynomial_order, 0)
        return multi_indices

    def _calculate_variance_contributions(self, coefficients: np.ndarray) -> np.ndarray:
        """Calculate variance contributions from PCE coefficients."""
        # Simplified variance calculation
        # In practice, this would use orthogonality properties of polynomials
        return coefficients[1:]**2  # Exclude constant term

    def _calculate_pce_statistics(self, pce_coefficients: Dict[str, np.ndarray],
                                variance_contributions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Calculate statistics from PCE representation."""
        statistics = {'mean': {}, 'variance': {}}

        for output_name, coeffs in pce_coefficients.items():
            # Mean is the constant term
            statistics['mean'][output_name] = coeffs[0]

            # Variance is sum of squared non-constant coefficients
            statistics['variance'][output_name] = np.sum(variance_contributions[output_name])

        return statistics


# Utility functions for uncertainty quantification
def calculate_reliability(samples: np.ndarray, threshold: float,
                        failure_criterion: str = "greater") -> float:
    """
    Calculate reliability (probability of not exceeding threshold).
    
    Args:
        samples: Output samples
        threshold: Failure threshold
        failure_criterion: "greater" or "less" than threshold
        
    Returns:
        Reliability estimate
    """
    if failure_criterion == "greater":
        failures = np.sum(samples > threshold)
    else:
        failures = np.sum(samples < threshold)

    return 1.0 - failures / len(samples)


def calculate_sensitivity_indices_from_pce(pce_coefficients: np.ndarray,
                                         multi_indices: List[List[int]]) -> Dict[str, float]:
    """
    Calculate Sobol sensitivity indices from PCE coefficients.
    
    Args:
        pce_coefficients: PCE coefficients
        multi_indices: Multi-indices corresponding to coefficients
        
    Returns:
        Dictionary of sensitivity indices
    """
    total_variance = np.sum(pce_coefficients[1:]**2)  # Exclude constant term

    sensitivity_indices = {}
    n_params = len(multi_indices[0]) if multi_indices else 0

    for i in range(n_params):
        # First-order sensitivity index
        first_order_var = 0.0
        for j, (coeff, multi_index) in enumerate(zip(pce_coefficients[1:], multi_indices[1:])):
            if np.sum(multi_index) == multi_index[i] and multi_index[i] > 0:
                first_order_var += coeff**2

        sensitivity_indices[f'S1_{i}'] = first_order_var / total_variance if total_variance > 0 else 0.0

    return sensitivity_indices


def bootstrap_uncertainty(samples: np.ndarray, statistic_func: Callable,
                         n_bootstrap: int = 1000, confidence_level: float = 0.95) -> Tuple[float, Tuple[float, float]]:
    """
    Bootstrap estimation of uncertainty in a statistic.
    
    Args:
        samples: Original samples
        statistic_func: Function to calculate statistic
        n_bootstrap: Number of bootstrap resamples
        confidence_level: Confidence level for interval
        
    Returns:
        Tuple of (statistic_value, confidence_interval)
    """
    original_statistic = statistic_func(samples)
    bootstrap_statistics = []

    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(samples, size=len(samples), replace=True)
        bootstrap_statistics.append(statistic_func(bootstrap_sample))

    bootstrap_statistics = np.array(bootstrap_statistics)

    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_statistics, 100 * alpha / 2)
    upper = np.percentile(bootstrap_statistics, 100 * (1 - alpha / 2))

    return original_statistic, (lower, upper)
