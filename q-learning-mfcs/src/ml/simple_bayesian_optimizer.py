#!/usr/bin/env python3
"""
Simplified Bayesian Optimization for MFC Electrode Parameter Tuning

This module provides a lightweight Bayesian optimization implementation
specifically designed for electrode parameter optimization using the Phase 2
physics models.

Created: 2025-08-01
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


@dataclass
class OptimizationBounds:
    """Parameter bounds for optimization."""
    name: str
    lower: float
    upper: float
    units: str = ""
    description: str = ""

class SimpleGaussianProcess:
    """
    Simplified Gaussian Process implementation for surrogate modeling.

    Uses RBF kernel with noise for fast approximation.
    """

    def __init__(self, kernel_lengthscale: float = 1.0, noise_level: float = 0.1):
        self.lengthscale = kernel_lengthscale
        self.noise = noise_level
        self.X_train = None
        self.y_train = None
        self.fitted = False

    def rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Radial basis function kernel."""
        if X1.ndim == 1:
            X1 = X1.reshape(-1, 1)
        if X2.ndim == 1:
            X2 = X2.reshape(-1, 1)

        # Compute squared distances
        sq_dists = np.sum(X1**2, axis=1, keepdims=True) + \
                   np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)

        return np.exp(-sq_dists / (2 * self.lengthscale**2))

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the Gaussian Process to training data."""
        self.X_train = np.array(X)
        self.y_train = np.array(y)

        if self.X_train.ndim == 1:
            self.X_train = self.X_train.reshape(-1, 1)

        # Compute kernel matrix with noise
        K = self.rbf_kernel(self.X_train, self.X_train)
        self.K_inv = np.linalg.inv(K + self.noise * np.eye(len(self.X_train)))

        self.fitted = True

    def predict(self, X: np.ndarray, return_std: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and standard deviation at test points."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Compute kernel vectors
        K_s = self.rbf_kernel(self.X_train, X)
        K_ss = self.rbf_kernel(X, X)

        # Predict mean
        mean = K_s.T @ self.K_inv @ self.y_train.reshape(-1, 1)
        mean = mean.flatten()

        if return_std:
            # Predict variance
            var = np.diag(K_ss) - np.sum((K_s.T @ self.K_inv) * K_s.T, axis=1)
            var = np.maximum(var, 1e-8)  # Numerical stability
            std = np.sqrt(var)
            return mean, std
        else:
            return mean, None

class SimpleBayesianOptimizer:
    """
    Simplified Bayesian optimization for electrode parameter tuning.

    Uses Gaussian Process surrogate model with Expected Improvement acquisition.
    """

    def __init__(self,
                 parameter_bounds: list[OptimizationBounds],
                 objective_function: Callable[[dict[str, float]], float],
                 acquisition_type: str = 'EI'):

        self.parameter_bounds = parameter_bounds
        self.objective_function = objective_function
        self.acquisition_type = acquisition_type

        # Initialize Gaussian Process
        self.gp = SimpleGaussianProcess()

        # Training data
        self.X_obs = []
        self.y_obs = []
        self.param_history = []

        # Best observed values
        self.best_params = None
        self.best_value = -np.inf

    def _normalize_parameters(self, params: dict[str, float]) -> np.ndarray:
        """Normalize parameters to [0,1] range."""
        normalized = []
        for bound in self.parameter_bounds:
            value = params[bound.name]
            norm_value = (value - bound.lower) / (bound.upper - bound.lower)
            normalized.append(norm_value)
        return np.array(normalized)

    def _denormalize_parameters(self, normalized: np.ndarray) -> dict[str, float]:
        """Convert normalized parameters back to original scale."""
        params = {}
        for i, bound in enumerate(self.parameter_bounds):
            value = bound.lower + normalized[i] * (bound.upper - bound.lower)
            params[bound.name] = value
        return params

    def _sample_random_parameters(self, n_samples: int) -> list[dict[str, float]]:
        """Sample random parameters within bounds."""
        samples = []
        for _ in range(n_samples):
            params = {}
            for bound in self.parameter_bounds:
                value = np.random.uniform(bound.lower, bound.upper)
                params[bound.name] = value
            samples.append(params)
        return samples

    def _expected_improvement(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """Expected Improvement acquisition function."""
        if not self.gp.fitted:
            return np.ones(X.shape[0])

        mu, sigma = self.gp.predict(X, return_std=True)

        # Current best value
        f_best = np.max(self.y_obs)

        # Calculate EI
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            improvement = mu - f_best - xi
            Z = improvement / sigma
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei

    def _optimize_acquisition(self) -> dict[str, float]:
        """Optimize acquisition function to find next sampling point."""
        best_acquisition = -np.inf
        best_params = None

        # Try multiple random starts
        n_starts = 20
        for _ in range(n_starts):
            # Random starting point
            x0 = np.random.uniform(0, 1, len(self.parameter_bounds))

            # Optimize acquisition function
            def neg_acquisition(x):
                return -self._expected_improvement(x.reshape(1, -1))[0]

            bounds = [(0, 1) for _ in self.parameter_bounds]

            try:
                res = minimize(neg_acquisition, x0, bounds=bounds, method='L-BFGS-B')

                if res.success and -res.fun > best_acquisition:
                    best_acquisition = -res.fun
                    best_params = self._denormalize_parameters(res.x)
            except:
                continue

        # Fallback to random sampling if optimization fails
        if best_params is None:
            best_params = self._sample_random_parameters(1)[0]

        return best_params

    def optimize(self, n_initial: int = 5, n_iterations: int = 20) -> dict:
        """
        Run Bayesian optimization.

        Args:
            n_initial: Number of initial random samples
            n_iterations: Number of optimization iterations

        Returns:
            Optimization results dictionary
        """

        print("🚀 Starting Simplified Bayesian Optimization")
        print(f"   Initial samples: {n_initial}")
        print(f"   Optimization iterations: {n_iterations}")
        print("=" * 50)

        # Phase 1: Initial random sampling
        print("📊 Phase 1: Initial random sampling...")
        initial_params = self._sample_random_parameters(n_initial)

        for i, params in enumerate(initial_params):
            print(f"  Initial sample {i+1}/{n_initial}")

            # Evaluate objective function
            try:
                objective_value = self.objective_function(params)

                # Store results
                self.param_history.append(params)
                self.X_obs.append(self._normalize_parameters(params))
                self.y_obs.append(objective_value)

                # Update best
                if objective_value > self.best_value:
                    self.best_value = objective_value
                    self.best_params = params.copy()

                print(f"    Objective: {objective_value:.4f}")

            except Exception as e:
                print(f"    Failed: {e}")
                continue

        # Fit initial GP model
        if len(self.X_obs) >= 2:
            self.gp.fit(np.array(self.X_obs), np.array(self.y_obs))
            print(f"✅ Fitted GP model with {len(self.X_obs)} samples")
        else:
            print("❌ Insufficient samples for GP fitting")
            return self._create_results()

        # Phase 2: Bayesian optimization iterations
        print("\n🎯 Phase 2: Bayesian optimization iterations...")

        for iteration in range(n_iterations):
            print(f"  Iteration {iteration+1}/{n_iterations}")

            # Find next sampling point
            next_params = self._optimize_acquisition()

            # Evaluate objective
            try:
                objective_value = self.objective_function(next_params)

                # Store results
                self.param_history.append(next_params)
                self.X_obs.append(self._normalize_parameters(next_params))
                self.y_obs.append(objective_value)

                # Update best
                if objective_value > self.best_value:
                    self.best_value = objective_value
                    self.best_params = next_params.copy()
                    print(f"    🏆 New best! Objective: {objective_value:.4f}")
                else:
                    print(f"    Objective: {objective_value:.4f}")

                # Update GP model
                self.gp.fit(np.array(self.X_obs), np.array(self.y_obs))

            except Exception as e:
                print(f"    Failed: {e}")
                continue

        # Return results
        results = self._create_results()

        print("\n" + "=" * 50)
        print("🎉 Optimization completed!")
        print(f"Best objective value: {self.best_value:.4f}")
        print(f"Total evaluations: {len(self.y_obs)}")

        return results

    def _create_results(self) -> dict:
        """Create results dictionary."""
        return {
            'best_parameters': self.best_params,
            'best_objective_value': self.best_value,
            'all_parameters': self.param_history,
            'all_objective_values': self.y_obs,
            'n_evaluations': len(self.y_obs),
            'parameter_bounds': {b.name: (b.lower, b.upper) for b in self.parameter_bounds}
        }

def create_electrode_optimization_bounds() -> list[OptimizationBounds]:
    """Create standard electrode optimization parameter bounds."""
    return [
        OptimizationBounds(
            name='electrode_length',
            lower=0.02, upper=0.1,
            units='m',
            description='Electrode length'
        ),
        OptimizationBounds(
            name='electrode_width',
            lower=0.02, upper=0.1,
            units='m',
            description='Electrode width'
        ),
        OptimizationBounds(
            name='electrode_thickness',
            lower=0.001, upper=0.02,
            units='m',
            description='Electrode thickness'
        ),
        OptimizationBounds(
            name='flow_rate',
            lower=0.5e-6, upper=5e-6,
            units='m³/s',
            description='Flow rate'
        ),
        OptimizationBounds(
            name='max_biofilm_density',
            lower=50.0, upper=120.0,
            units='kg/m³',
            description='Maximum biofilm density'
        ),
        OptimizationBounds(
            name='max_substrate_consumption',
            lower=0.05, upper=0.5,
            units='mol/m³/s',
            description='Maximum substrate consumption rate'
        )
    ]

if __name__ == "__main__":
    # Example usage
    def dummy_objective(params):
        """Dummy objective function for testing."""
        # Simple quadratic function with noise
        x = params['electrode_length']
        y = params['flow_rate'] * 1e6  # Scale for visibility
        return -(x - 0.06)**2 - (y - 2)**2 + np.random.normal(0, 0.1)

    bounds = create_electrode_optimization_bounds()
    optimizer = SimpleBayesianOptimizer(bounds, dummy_objective)
    results = optimizer.optimize(n_initial=5, n_iterations=10)

    print(f"\nBest parameters: {results['best_parameters']}")
    print(f"Best value: {results['best_objective_value']:.4f}")
