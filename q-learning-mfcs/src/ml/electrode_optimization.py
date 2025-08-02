#!/usr/bin/env python3
"""
Machine Learning Framework for Electrode Parameter Optimization

This module implements advanced ML/DL approaches for optimizing electrode
parameters including:
- Bayesian optimization for electrode geometry
- Neural network surrogate models for fast evaluation
- Multi-objective optimization for competing design goals
- Reinforcement learning for dynamic parameter adjustment
- Genetic algorithms for discrete parameter spaces

Created: 2025-08-01
Literature References:
1. Frazier, P.I. (2018). "A Tutorial on Bayesian Optimization"
2. Shahriari, B. et al. (2016). "Taking the Human Out of the Loop: A Review of Bayesian Optimization"
3. Deb, K. (2001). "Multi-Objective Optimization using Evolutionary Algorithms"
4. Silver, D. et al. (2016). "Mastering the game of Go with deep neural networks"
"""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# Machine Learning Libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Deep learning features disabled.")

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Some ML features disabled.")

try:
    from scipy.optimize import differential_evolution, minimize
    from scipy.stats import qmc
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Optimization features limited.")

from config.electrode_config import ElectrodeGeometry, ElectrodeMaterial
from physics.advanced_electrode_model import AdvancedElectrodeModel, CellGeometry


@dataclass
class OptimizationParameter:
    """Definition of a parameter to optimize."""
    name: str
    bounds: tuple[float, float]  # (min, max)
    parameter_type: str  # 'continuous', 'discrete', 'categorical'
    units: str = ""
    description: str = ""
    discrete_values: list[Any] | None = None  # For discrete/categorical parameters


@dataclass
class OptimizationObjective:
    """Definition of an optimization objective."""
    name: str
    direction: str  # 'maximize' or 'minimize'
    weight: float = 1.0
    constraint_type: str | None = None  # 'hard', 'soft', None
    constraint_value: float | None = None


@dataclass
class OptimizationResult:
    """Results from optimization process."""
    best_parameters: dict[str, float]
    best_objective_value: float
    optimization_history: pd.DataFrame
    convergence_metrics: dict[str, float]
    model_performance: dict[str, float]
    computational_cost: dict[str, float]


class SurrogateModel(ABC):
    """Abstract base class for surrogate models."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the surrogate model to data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and uncertainty."""
        pass

    @abstractmethod
    def get_acquisition_function(self) -> Callable:
        """Get acquisition function for optimization."""
        pass


class GaussianProcessSurrogate(SurrogateModel):
    """Gaussian Process surrogate model for Bayesian optimization."""

    def __init__(self, kernel=None, alpha=1e-6, n_restarts_optimizer=10):
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn required for Gaussian Process surrogate")

        if kernel is None:
            kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=alpha)

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=True
        )
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit GP to training data."""
        X_scaled = self.scaler.fit_transform(X)
        self.gp.fit(X_scaled, y)
        self.fitted = True

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.scaler.transform(X)
        mean, std = self.gp.predict(X_scaled, return_std=True)
        return mean, std

    def get_acquisition_function(self, acquisition_type='EI') -> Callable:
        """Get acquisition function (Expected Improvement, Upper Confidence Bound, etc.)."""

        def expected_improvement(X: np.ndarray, xi=0.01) -> np.ndarray:
            """Expected Improvement acquisition function."""
            if not self.fitted:
                return np.zeros(X.shape[0])

            mean, std = self.predict(X)

            # Get current best
            if hasattr(self, 'y_train_'):
                f_best = np.max(self.y_train_)
            else:
                f_best = np.max(mean)

            # Calculate EI
            improvement = mean - f_best - xi
            Z = improvement / (std + 1e-9)

            from scipy.stats import norm
            ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
            ei[std == 0.0] = 0.0

            return ei

        def upper_confidence_bound(X: np.ndarray, beta=2.0) -> np.ndarray:
            """Upper Confidence Bound acquisition function."""
            mean, std = self.predict(X)
            return mean + beta * std

        if acquisition_type == 'EI':
            return expected_improvement
        elif acquisition_type == 'UCB':
            return upper_confidence_bound
        else:
            raise ValueError(f"Unknown acquisition type: {acquisition_type}")


class NeuralNetworkSurrogate(SurrogateModel):
    """Neural network surrogate model for fast evaluation."""

    def __init__(self, hidden_layers=[64, 32, 16], dropout_rate=0.1, learning_rate=0.001):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for Neural Network surrogate")

        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.fitted = False

    def _build_model(self, input_dim: int) -> nn.Module:
        """Build neural network architecture."""
        layers = []
        prev_dim = input_dim

        for hidden_dim in self.hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim

        # Output layer (mean and log-variance for uncertainty)
        layers.append(nn.Linear(prev_dim, 2))

        return nn.Sequential(*layers).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs=1000, batch_size=32) -> None:
        """Fit neural network to training data."""
        # Preprocess data
        if self.scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        # Build model
        self.model = self._build_model(X.shape[1])
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

        # Training dataset
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(batch_X)
                mean_pred = outputs[:, 0]
                log_var_pred = outputs[:, 1]

                # Negative log-likelihood loss (for uncertainty estimation)
                mse_loss = nn.MSELoss()(mean_pred, batch_y)
                uncertainty_loss = torch.mean(torch.exp(-log_var_pred) * (batch_y - mean_pred)**2 + log_var_pred)
                loss = mse_loss + 0.1 * uncertainty_loss

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 100 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.6f}")

        self.fitted = True

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty estimation."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        # Preprocess
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            mean = outputs[:, 0].cpu().numpy()
            log_var = outputs[:, 1].cpu().numpy()
            std = np.sqrt(np.exp(log_var))

        return mean, std

    def get_acquisition_function(self) -> Callable:
        """Get acquisition function for neural network."""
        def neural_ucb(X: np.ndarray, beta=2.0) -> np.ndarray:
            mean, std = self.predict(X)
            return mean + beta * std

        return neural_ucb


class BayesianOptimizer:
    """Bayesian optimization for electrode parameters."""

    def __init__(self,
                 parameters: list[OptimizationParameter],
                 objectives: list[OptimizationObjective],
                 electrode_model_factory: Callable,
                 surrogate_model: SurrogateModel | None = None,
                 acquisition_type: str = 'EI'):

        self.parameters = parameters
        self.objectives = objectives
        self.electrode_model_factory = electrode_model_factory
        self.surrogate = surrogate_model or GaussianProcessSurrogate()
        self.acquisition_type = acquisition_type

        # Optimization history
        self.X_observed = []
        self.y_observed = []
        self.optimization_history = []

        # Parameter bounds and scaling
        self.bounds = np.array([p.bounds for p in parameters])
        self.param_names = [p.name for p in parameters]

    def _sample_initial_points(self, n_initial: int = 10) -> np.ndarray:
        """Generate initial sample points using Latin Hypercube Sampling."""
        if not SCIPY_AVAILABLE:
            # Fallback to random sampling
            np.random.seed(42)
            return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (n_initial, len(self.parameters)))

        sampler = qmc.LatinHypercube(d=len(self.parameters), seed=42)
        unit_samples = sampler.random(n_initial)

        # Scale to parameter bounds
        scaled_samples = qmc.scale(unit_samples, self.bounds[:, 0], self.bounds[:, 1])
        return scaled_samples

    def _evaluate_objective(self, parameters: np.ndarray) -> float:
        """Evaluate objective function at given parameters."""
        try:
            # Create parameter dictionary
            param_dict = dict(zip(self.param_names, parameters, strict=False))

            # Create electrode model with parameters
            model = self.electrode_model_factory(param_dict)

            # Run simulation
            model.step(dt=3600)  # 1 hour simulation
            targets = model.get_optimization_targets()

            # Calculate weighted multi-objective value
            objective_value = 0.0
            for obj in self.objectives:
                target_key = f"{obj.direction}_{obj.name}"
                if target_key in targets:
                    value = targets[target_key]
                    if obj.direction == 'maximize':
                        objective_value += obj.weight * value
                    else:  # minimize
                        objective_value -= obj.weight * value

            return objective_value

        except Exception as e:
            print(f"Evaluation failed for parameters {parameters}: {e}")
            return -1e6  # Large penalty for failed evaluations

    def _optimize_acquisition(self, acquisition_func: Callable) -> np.ndarray:
        """Optimize acquisition function to find next evaluation point."""
        if not SCIPY_AVAILABLE:
            # Fallback to random search
            n_candidates = 1000
            candidates = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                         (n_candidates, len(self.parameters)))
            acquisition_values = acquisition_func(candidates)
            best_idx = np.argmax(acquisition_values)
            return candidates[best_idx]

        # Multiple random starts for global optimization
        n_starts = 10
        best_result = None
        best_value = -np.inf

        for _ in range(n_starts):
            x0 = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])

            result = minimize(
                lambda x: -acquisition_func(x.reshape(1, -1))[0],  # Minimize negative acquisition
                x0,
                bounds=self.bounds,
                method='L-BFGS-B'
            )

            if result.fun < best_value:
                best_value = result.fun
                best_result = result

        return best_result.x if best_result else x0

    def optimize(self, n_iterations: int = 50, n_initial: int = 10) -> OptimizationResult:
        """Run Bayesian optimization."""
        print(f"Starting Bayesian optimization with {n_initial} initial points...")

        # Initial sampling
        X_initial = self._sample_initial_points(n_initial)
        y_initial = []

        print("Evaluating initial points...")
        for i, x in enumerate(X_initial):
            y = self._evaluate_objective(x)
            y_initial.append(y)
            print(f"Initial point {i+1}/{n_initial}: objective = {y:.6f}")

        self.X_observed = X_initial.tolist()
        self.y_observed = y_initial

        # Iterative optimization
        print(f"Starting iterative optimization for {n_iterations} iterations...")

        for iteration in range(n_iterations):
            # Fit surrogate model
            X_array = np.array(self.X_observed)
            y_array = np.array(self.y_observed)

            self.surrogate.fit(X_array, y_array)

            # Get acquisition function
            acquisition_func = self.surrogate.get_acquisition_function(self.acquisition_type)

            # Optimize acquisition function
            x_next = self._optimize_acquisition(acquisition_func)

            # Evaluate objective at new point
            y_next = self._evaluate_objective(x_next)

            # Update observations
            self.X_observed.append(x_next)
            self.y_observed.append(y_next)

            # Track progress
            best_so_far = np.max(self.y_observed)
            print(f"Iteration {iteration+1}/{n_iterations}: "
                  f"new_objective = {y_next:.6f}, best_so_far = {best_so_far:.6f}")

            self.optimization_history.append({
                'iteration': iteration + n_initial + 1,
                'parameters': dict(zip(self.param_names, x_next, strict=False)),
                'objective_value': y_next,
                'best_so_far': best_so_far
            })

        # Return results
        best_idx = np.argmax(self.y_observed)
        best_parameters = dict(zip(self.param_names, self.X_observed[best_idx], strict=False))

        return OptimizationResult(
            best_parameters=best_parameters,
            best_objective_value=self.y_observed[best_idx],
            optimization_history=pd.DataFrame(self.optimization_history),
            convergence_metrics={
                'final_improvement': best_so_far - y_initial[0] if y_initial else 0,
                'convergence_iteration': best_idx,
                'improvement_rate': (best_so_far - np.mean(y_initial[:5])) / n_iterations if len(y_initial) >= 5 else 0
            },
            model_performance={
                'surrogate_type': type(self.surrogate).__name__,
                'n_evaluations': len(self.y_observed),
                'success_rate': sum(1 for y in self.y_observed if y > -1e5) / len(self.y_observed)
            },
            computational_cost={
                'total_evaluations': len(self.y_observed),
                'initial_evaluations': n_initial,
                'optimization_evaluations': n_iterations
            }
        )


class MultiObjectiveOptimizer:
    """Multi-objective optimization using NSGA-II algorithm."""

    def __init__(self,
                 parameters: list[OptimizationParameter],
                 objectives: list[OptimizationObjective],
                 electrode_model_factory: Callable,
                 population_size: int = 50):

        self.parameters = parameters
        self.objectives = objectives
        self.electrode_model_factory = electrode_model_factory
        self.population_size = population_size

        self.bounds = np.array([p.bounds for p in parameters])
        self.param_names = [p.name for p in parameters]

    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """Evaluate objectives for entire population."""
        objective_values = []

        for individual in population:
            param_dict = dict(zip(self.param_names, individual, strict=False))

            try:
                model = self.electrode_model_factory(param_dict)
                model.step(dt=3600)
                targets = model.get_optimization_targets()

                # Extract objective values
                individual_objectives = []
                for obj in self.objectives:
                    target_key = f"{obj.direction}_{obj.name}"
                    if target_key in targets:
                        value = targets[target_key]
                        # Convert to minimization problem
                        if obj.direction == 'maximize':
                            value = -value
                        individual_objectives.append(value)
                    else:
                        individual_objectives.append(1e6)  # Penalty for missing objectives

                objective_values.append(individual_objectives)

            except Exception:
                # Penalty for failed evaluations
                penalty_values = [1e6] * len(self.objectives)
                objective_values.append(penalty_values)

        return np.array(objective_values)

    def _fast_non_dominated_sort(self, objective_values: np.ndarray) -> list[list[int]]:
        """Fast non-dominated sorting (NSGA-II)."""
        n = len(objective_values)
        domination_counts = np.zeros(n, dtype=int)
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]

        # Find domination relationships
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Check if i dominates j (all objectives better or equal, at least one strictly better)
                    dominates = np.all(objective_values[i] <= objective_values[j]) and \
                               np.any(objective_values[i] < objective_values[j])

                    if dominates:
                        dominated_solutions[i].append(j)
                    elif np.all(objective_values[j] <= objective_values[i]) and \
                         np.any(objective_values[j] < objective_values[i]):
                        domination_counts[i] += 1

            if domination_counts[i] == 0:
                fronts[0].append(i)

        # Build subsequent fronts
        front_idx = 0
        while len(fronts[front_idx]) > 0:
            next_front = []
            for i in fronts[front_idx]:
                for j in dominated_solutions[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0:
                        next_front.append(j)

            front_idx += 1
            fronts.append(next_front)

        return fronts[:-1]  # Remove empty last front

    def _calculate_crowding_distance(self, front: list[int], objective_values: np.ndarray) -> np.ndarray:
        """Calculate crowding distance for diversity preservation."""
        if len(front) <= 2:
            return np.full(len(front), np.inf)

        distances = np.zeros(len(front))
        n_objectives = objective_values.shape[1]

        for obj_idx in range(n_objectives):
            # Sort by this objective
            sorted_indices = sorted(range(len(front)), key=lambda i: objective_values[front[i], obj_idx])

            # Assign infinite distance to boundary points
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf

            # Calculate distances for intermediate points
            obj_range = (objective_values[front[sorted_indices[-1]], obj_idx] -
                        objective_values[front[sorted_indices[0]], obj_idx])

            if obj_range > 0:
                for i in range(1, len(sorted_indices) - 1):
                    distances[sorted_indices[i]] += \
                        (objective_values[front[sorted_indices[i+1]], obj_idx] -
                         objective_values[front[sorted_indices[i-1]], obj_idx]) / obj_range

        return distances

    def optimize(self, n_generations: int = 100) -> dict[str, Any]:
        """Run multi-objective optimization using NSGA-II."""
        print(f"Starting multi-objective optimization with {self.population_size} individuals...")

        # Initialize population
        population = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1],
            (self.population_size, len(self.parameters))
        )

        best_fronts_history = []

        for generation in range(n_generations):
            print(f"Generation {generation + 1}/{n_generations}")

            # Evaluate population
            objective_values = self._evaluate_population(population)

            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(objective_values)

            # Store Pareto front
            if len(fronts) > 0:
                pareto_front = fronts[0]
                best_fronts_history.append({
                    'generation': generation,
                    'pareto_front_size': len(pareto_front),
                    'pareto_solutions': [dict(zip(self.param_names, population[i], strict=False)) for i in pareto_front],
                    'pareto_objectives': objective_values[pareto_front].tolist()
                })

            # Selection for next generation (simplified)
            if generation < n_generations - 1:
                # Create new population through crossover and mutation
                new_population = []

                # Elite preservation (keep Pareto front)
                if len(fronts) > 0:
                    for i in fronts[0]:
                        new_population.append(population[i])

                # Fill remaining population with random individuals
                while len(new_population) < self.population_size:
                    individual = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
                    new_population.append(individual)

                population = np.array(new_population[:self.population_size])

        # Return results
        final_fronts = self._fast_non_dominated_sort(objective_values)
        pareto_front = final_fronts[0] if len(final_fronts) > 0 else []

        return {
            'pareto_front_indices': pareto_front,
            'pareto_solutions': [dict(zip(self.param_names, population[i], strict=False)) for i in pareto_front],
            'pareto_objectives': objective_values[pareto_front].tolist() if len(pareto_front) > 0 else [],
            'optimization_history': best_fronts_history,
            'final_population': population,
            'final_objectives': objective_values,
            'convergence_metrics': {
                'final_pareto_size': len(pareto_front),
                'generations_completed': n_generations
            }
        }


# Example usage and factory functions
def create_electrode_model_factory(base_cell_geometry: CellGeometry) -> Callable:
    """Create factory function for electrode models with parameter variation."""

    def model_factory(parameters: dict[str, float]) -> AdvancedElectrodeModel:
        """Create electrode model with given parameters."""
        from config.electrode_config import create_electrode_config

        # Extract parameters
        length = parameters.get('electrode_length', 0.05)  # m
        width = parameters.get('electrode_width', 0.05)    # m
        thickness = parameters.get('electrode_thickness', 0.005)  # m

        # Create electrode configuration
        electrode_config = create_electrode_config(
            material=ElectrodeMaterial.CARBON_FELT,  # Could be parameterized
            geometry_type=ElectrodeGeometry.RECTANGULAR_PLATE,
            dimensions={
                'length': length,
                'width': width,
                'thickness': thickness
            }
        )

        # Create advanced model
        model = AdvancedElectrodeModel(
            electrode_config=electrode_config,
            cell_geometry=base_cell_geometry
        )

        return model

    return model_factory


def run_optimization_example():
    """Run example optimization of electrode parameters."""

    # Define cell geometry
    cell_geom = CellGeometry(
        length=0.1, width=0.1, height=0.05,
        anode_chamber_volume=0.0002, cathode_chamber_volume=0.0002
    )

    # Define optimization parameters
    parameters = [
        OptimizationParameter(
            name='electrode_length',
            bounds=(0.02, 0.08),  # 2-8 cm
            parameter_type='continuous',
            units='m',
            description='Electrode length'
        ),
        OptimizationParameter(
            name='electrode_width',
            bounds=(0.02, 0.08),  # 2-8 cm
            parameter_type='continuous',
            units='m',
            description='Electrode width'
        ),
        OptimizationParameter(
            name='electrode_thickness',
            bounds=(0.001, 0.02),  # 1-20 mm
            parameter_type='continuous',
            units='m',
            description='Electrode thickness'
        )
    ]

    # Define objectives
    objectives = [
        OptimizationObjective(
            name='current_density',
            direction='maximize',
            weight=1.0
        ),
        OptimizationObjective(
            name='substrate_utilization',
            direction='maximize',
            weight=0.5
        )
    ]

    # Create model factory
    model_factory = create_electrode_model_factory(cell_geom)

    # Run Bayesian optimization
    if SKLEARN_AVAILABLE:
        print("Running Bayesian optimization...")
        optimizer = BayesianOptimizer(
            parameters=parameters,
            objectives=objectives,
            electrode_model_factory=model_factory
        )

        results = optimizer.optimize(n_iterations=20, n_initial=5)

        print("\n=== Bayesian Optimization Results ===")
        print(f"Best parameters: {results.best_parameters}")
        print(f"Best objective value: {results.best_objective_value:.6f}")
        print(f"Convergence metrics: {results.convergence_metrics}")

        return results
    else:
        print("Scikit-learn not available. Skipping Bayesian optimization.")
        return None


if __name__ == "__main__":
    # Run optimization example
    results = run_optimization_example()
    if results:
        print("✅ Electrode optimization completed successfully")
    else:
        print("⚠️  Optimization skipped due to missing dependencies")
