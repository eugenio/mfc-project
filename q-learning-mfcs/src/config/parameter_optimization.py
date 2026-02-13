"""
Parameter Optimization Framework

This module provides comprehensive parameter optimization algorithms for MFC systems,
including Bayesian optimization, genetic algorithms, and gradient-based methods.

Classes:
- ParameterOptimizer: Main optimization framework
- BayesianOptimizer: Gaussian Process-based Bayesian optimization
- GeneticOptimizer: Genetic algorithm implementation
- GradientOptimizer: Gradient-based optimization methods
- OptimizationResult: Results container for optimization runs

Features:
- Multi-objective optimization with Pareto frontiers
- Constraint handling and bounds checking
- Parallel evaluation support
- Convergence analysis and early stopping
- Hyperparameter tuning for optimization algorithms

Literature References:
1. Shahriari, B., et al. (2016). "Taking the Human Out of the Loop: A Review of Bayesian Optimization"
2. Deb, K., et al. (2002). "A fast and elitist multiobjective genetic algorithm: NSGA-II"
3. Nocedal, J., & Wright, S. (2006). "Numerical Optimization"
4. Forrester, A., et al. (2008). "Engineering Design via Surrogate Modelling"
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod
import warnings
from datetime import datetime

# Optional dependencies for advanced optimization
try:
    from scipy.optimize import minimize, differential_evolution
    from scipy.stats import norm
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("SciPy not available. Some optimization features will be limited.")

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("Scikit-learn not available. Bayesian optimization will be limited.")

# Import configuration classes
from .sensitivity_analysis import ParameterSpace


class OptimizationMethod(Enum):
    """Available optimization methods."""
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    GRADIENT_BASED = "gradient_based"
    PARTICLE_SWARM = "particle_swarm"
    SIMULATED_ANNEALING = "simulated_annealing"
    NELDER_MEAD = "nelder_mead"


class ObjectiveType(Enum):
    """Optimization objective types."""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass
class OptimizationObjective:
    """Definition of optimization objective."""
    name: str
    type: ObjectiveType
    weight: float = 1.0
    tolerance: float = 1e-6
    description: str = ""


@dataclass
class OptimizationConstraint:
    """Definition of optimization constraint."""
    name: str
    constraint_function: Callable[[np.ndarray], float]
    constraint_type: str = "ineq"  # "eq" for equality, "ineq" for inequality
    tolerance: float = 1e-6
    description: str = ""


@dataclass
class OptimizationResult:
    """Results container for parameter optimization."""
    
    # Optimization metadata
    method: OptimizationMethod
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_evaluations: int = 0
    convergence_history: List[float] = field(default_factory=list)
    
    # Best solution
    best_parameters: Optional[np.ndarray] = None
    best_objective_values: Optional[Dict[str, float]] = None
    best_overall_score: Optional[float] = None
    
    # Pareto frontier (for multi-objective optimization)
    pareto_parameters: Optional[np.ndarray] = None
    pareto_objectives: Optional[np.ndarray] = None
    
    # All evaluated points
    all_parameters: List[np.ndarray] = field(default_factory=list)
    all_objectives: List[Dict[str, float]] = field(default_factory=list)
    
    # Convergence information
    converged: bool = False
    convergence_message: str = ""
    
    # Statistics
    success_rate: float = 0.0
    mean_evaluation_time: float = 0.0
    std_evaluation_time: float = 0.0
    
    def set_end_time(self):
        """Set the end time of optimization."""
        self.end_time = datetime.now()
    
    def get_optimization_time(self) -> float:
        """Get total optimization time in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


class ParameterOptimizer(ABC):
    """Abstract base class for parameter optimizers."""
    
    def __init__(self, parameter_space: ParameterSpace,
                 objectives: List[OptimizationObjective],
                 constraints: Optional[List[OptimizationConstraint]] = None,
                 random_seed: Optional[int] = None):
        """
        Initialize parameter optimizer.
        
        Args:
            parameter_space: Parameter space definition
            objectives: List of optimization objectives
            constraints: Optional list of constraints
            random_seed: Random seed for reproducibility
        """
        self.parameter_space = parameter_space
        self.objectives = objectives
        self.constraints = constraints or []
        self.random_seed = random_seed
        self.logger = logging.getLogger(__name__)
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    @abstractmethod
    def optimize(self, objective_function: Callable[[np.ndarray], Dict[str, float]],
                max_evaluations: int = 100,
                **kwargs) -> OptimizationResult:
        """
        Perform parameter optimization.
        
        Args:
            objective_function: Function to optimize
            max_evaluations: Maximum number of function evaluations
            **kwargs: Method-specific parameters
            
        Returns:
            Optimization results
        """
        pass
    
    def _evaluate_objectives(self, parameters: np.ndarray,
                           objective_function: Callable) -> Dict[str, float]:
        """
        Evaluate objectives for given parameters.
        
        Args:
            parameters: Parameter values
            objective_function: Objective function
            
        Returns:
            Dictionary of objective values
        """
        try:
            return objective_function(parameters)
        except Exception as e:
            self.logger.warning(f"Objective evaluation failed: {e}")
            # Return worst possible values
            return {obj.name: float('inf') if obj.type == ObjectiveType.MINIMIZE else -float('inf') 
                   for obj in self.objectives}
    
    def _check_constraints(self, parameters: np.ndarray) -> bool:
        """
        Check if parameters satisfy all constraints.
        
        Args:
            parameters: Parameter values
            
        Returns:
            True if all constraints are satisfied
        """
        for constraint in self.constraints:
            try:
                value = constraint.constraint_function(parameters)
                if constraint.constraint_type == "eq":
                    if abs(value) > constraint.tolerance:
                        return False
                elif constraint.constraint_type == "ineq":
                    if value < -constraint.tolerance:
                        return False
            except Exception:
                return False
        return True
    
    def _calculate_overall_score(self, objective_values: Dict[str, float]) -> float:
        """
        Calculate overall optimization score from multiple objectives.
        
        Args:
            objective_values: Dictionary of objective values
            
        Returns:
            Combined score
        """
        score = 0.0
        for obj in self.objectives:
            value = objective_values.get(obj.name, 0.0)
            if obj.type == ObjectiveType.MINIMIZE:
                score -= obj.weight * value
            else:
                score += obj.weight * value
        return score


class BayesianOptimizer(ParameterOptimizer):
    """Bayesian optimization using Gaussian Processes."""
    
    def __init__(self, parameter_space: ParameterSpace,
                 objectives: List[OptimizationObjective],
                 constraints: Optional[List[OptimizationConstraint]] = None,
                 acquisition_function: str = "expected_improvement",
                 kernel_type: str = "matern",
                 random_seed: Optional[int] = None):
        """
        Initialize Bayesian optimizer.
        
        Args:
            parameter_space: Parameter space definition
            objectives: List of optimization objectives
            constraints: Optional list of constraints
            acquisition_function: Acquisition function ("ei", "ucb", "poi")
            kernel_type: GP kernel type ("matern", "rbf", "periodic")
            random_seed: Random seed for reproducibility
        """
        super().__init__(parameter_space, objectives, constraints, random_seed)
        self.acquisition_function = acquisition_function
        self.kernel_type = kernel_type
        
        if not HAS_SKLEARN:
            raise ImportError("Scikit-learn required for Bayesian optimization")
    
    def optimize(self, objective_function: Callable[[np.ndarray], Dict[str, float]],
                max_evaluations: int = 100,
                n_initial_points: int = 10,
                acquisition_optimizer: str = "lbfgs",
                **kwargs) -> OptimizationResult:
        """
        Perform Bayesian optimization.
        
        Args:
            objective_function: Function to optimize
            max_evaluations: Maximum number of function evaluations
            n_initial_points: Number of initial random points
            acquisition_optimizer: Optimizer for acquisition function
            **kwargs: Additional parameters
            
        Returns:
            Optimization results
        """
        result = OptimizationResult(method=OptimizationMethod.BAYESIAN)
        
        # Initialize with random points
        X = self._generate_initial_points(n_initial_points)
        y = []
        
        for x in X:
            obj_values = self._evaluate_objectives(x, objective_function)
            overall_score = self._calculate_overall_score(obj_values)
            y.append(overall_score)
            
            result.all_parameters.append(x)
            result.all_objectives.append(obj_values)
            result.convergence_history.append(overall_score)
        
        y = np.array(y)
        
        # Initialize Gaussian Process
        gp = self._create_gaussian_process()
        
        # Bayesian optimization loop
        for i in range(n_initial_points, max_evaluations):
            # Fit GP to current data
            gp.fit(X, y)
            
            # Find next point using acquisition function
            next_x = self._optimize_acquisition(gp, X, y)
            
            # Evaluate objective at next point
            obj_values = self._evaluate_objectives(next_x, objective_function)
            overall_score = self._calculate_overall_score(obj_values)
            
            # Update data
            X = np.vstack([X, next_x])
            y = np.append(y, overall_score)
            
            result.all_parameters.append(next_x)
            result.all_objectives.append(obj_values)
            result.convergence_history.append(overall_score)
            
            # Check for convergence
            if self._check_convergence(result.convergence_history):
                result.converged = True
                result.convergence_message = "Converged based on improvement threshold"
                break
        
        # Find best solution
        best_idx = np.argmax(y) if self.objectives[0].type == ObjectiveType.MAXIMIZE else np.argmin(y)
        result.best_parameters = X[best_idx]
        result.best_objective_values = result.all_objectives[best_idx]
        result.best_overall_score = y[best_idx]
        result.total_evaluations = len(result.all_parameters)
        
        result.set_end_time()
        return result
    
    def _generate_initial_points(self, n_points: int) -> np.ndarray:
        """Generate initial random points for Bayesian optimization."""
        return self.parameter_space.sample(n_points, 
                                         method="latin_hypercube" if n_points > 5 else "random",
                                         seed=self.random_seed)
    
    def _create_gaussian_process(self):
        """Create Gaussian Process regressor."""
        if self.kernel_type == "matern":
            kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-5)
        elif self.kernel_type == "rbf":
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5)
        else:
            kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-5)
        
        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self.random_seed
        )
    
    def _optimize_acquisition(self, gp, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Optimize acquisition function to find next evaluation point."""
        bounds = [(param.bounds.min_value, param.bounds.max_value) 
                 for param in self.parameter_space.parameters]
        
        # Multiple random starts for acquisition optimization
        best_x = None
        best_acq = -float('inf')
        
        for _ in range(10):  # Multiple restarts
            x0 = self.parameter_space.sample(1, seed=None)[0]
            
            def acquisition_objective(x):
                return -self._acquisition_function(x.reshape(1, -1), gp, y)
            
            try:
                if HAS_SCIPY:
                    res = minimize(acquisition_objective, x0, bounds=bounds, 
                                 method='L-BFGS-B')
                    if res.success:
                        acq_val = -res.fun
                        if acq_val > best_acq:
                            best_acq = acq_val
                            best_x = res.x
            except Exception:
                continue
        
        return best_x if best_x is not None else x0
    
    def _acquisition_function(self, X: np.ndarray, gp, y_best: np.ndarray) -> np.ndarray:
        """Calculate acquisition function value."""
        mean, std = gp.predict(X, return_std=True)
        
        if self.acquisition_function == "expected_improvement":
            return self._expected_improvement(mean, std, np.max(y_best))
        elif self.acquisition_function == "upper_confidence_bound":
            return self._upper_confidence_bound(mean, std, beta=2.0)
        elif self.acquisition_function == "probability_of_improvement":
            return self._probability_of_improvement(mean, std, np.max(y_best))
        else:
            return self._expected_improvement(mean, std, np.max(y_best))
    
    def _expected_improvement(self, mean: np.ndarray, std: np.ndarray, 
                            f_best: float, xi: float = 0.01) -> np.ndarray:
        """Expected Improvement acquisition function."""
        improvement = mean - f_best - xi
        Z = improvement / (std + 1e-9)
        ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
        return ei
    
    def _upper_confidence_bound(self, mean: np.ndarray, std: np.ndarray, 
                              beta: float = 2.0) -> np.ndarray:
        """Upper Confidence Bound acquisition function."""
        return mean + beta * std
    
    def _probability_of_improvement(self, mean: np.ndarray, std: np.ndarray,
                                  f_best: float, xi: float = 0.01) -> np.ndarray:
        """Probability of Improvement acquisition function."""
        improvement = mean - f_best - xi
        Z = improvement / (std + 1e-9)
        return norm.cdf(Z)
    
    def _check_convergence(self, history: List[float], 
                          window: int = 10, threshold: float = 1e-6) -> bool:
        """Check convergence based on improvement history."""
        if len(history) < window:
            return False
        
        recent_best = max(history[-window:])
        older_best = max(history[-2*window:-window]) if len(history) >= 2*window else -float('inf')
        
        improvement = recent_best - older_best
        return improvement < threshold


class GeneticOptimizer(ParameterOptimizer):
    """Genetic Algorithm optimizer."""
    
    def __init__(self, parameter_space: ParameterSpace,
                 objectives: List[OptimizationObjective],
                 constraints: Optional[List[OptimizationConstraint]] = None,
                 population_size: int = 50,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 selection_method: str = "tournament",
                 random_seed: Optional[int] = None):
        """
        Initialize Genetic Algorithm optimizer.
        
        Args:
            parameter_space: Parameter space definition
            objectives: List of optimization objectives
            constraints: Optional list of constraints
            population_size: Size of population
            crossover_rate: Crossover probability
            mutation_rate: Mutation probability
            selection_method: Selection method ("tournament", "roulette", "rank")
            random_seed: Random seed for reproducibility
        """
        super().__init__(parameter_space, objectives, constraints, random_seed)
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.selection_method = selection_method
    
    def optimize(self, objective_function: Callable[[np.ndarray], Dict[str, float]],
                max_evaluations: int = 1000,
                max_generations: Optional[int] = None,
                elite_size: int = 2,
                **kwargs) -> OptimizationResult:
        """
        Perform genetic algorithm optimization.
        
        Args:
            objective_function: Function to optimize
            max_evaluations: Maximum number of function evaluations
            max_generations: Maximum number of generations (optional)
            elite_size: Number of elite individuals to preserve
            **kwargs: Additional parameters
            
        Returns:
            Optimization results
        """
        result = OptimizationResult(method=OptimizationMethod.GENETIC)
        
        if max_generations is None:
            max_generations = max_evaluations // self.population_size
        
        # Initialize population
        population = self.parameter_space.sample(self.population_size, seed=self.random_seed)
        
        # Evaluate initial population
        fitness_scores = []
        for individual in population:
            obj_values = self._evaluate_objectives(individual, objective_function)
            overall_score = self._calculate_overall_score(obj_values)
            fitness_scores.append(overall_score)
            
            result.all_parameters.append(individual)
            result.all_objectives.append(obj_values)
        
        fitness_scores = np.array(fitness_scores)
        
        # Evolution loop
        for generation in range(max_generations):
            if result.total_evaluations >= max_evaluations:
                break
            
            # Record best fitness in this generation
            best_fitness = np.max(fitness_scores)
            result.convergence_history.append(best_fitness)
            
            # Selection
            selected_indices = self._selection(fitness_scores, self.population_size - elite_size)
            selected_population = population[selected_indices]
            
            # Elitism - preserve best individuals
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            elite_population = population[elite_indices]
            
            # Crossover and mutation
            offspring = []
            for i in range(0, len(selected_population), 2):
                parent1 = selected_population[i]
                parent2 = selected_population[min(i + 1, len(selected_population) - 1)]
                
                if np.random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                child1 = self._mutation(child1)
                child2 = self._mutation(child2)
                
                offspring.extend([child1, child2])
            
            # Create new population
            offspring = np.array(offspring[:self.population_size - elite_size])
            population = np.vstack([elite_population, offspring])
            
            # Evaluate new individuals
            new_fitness = []
            for i, individual in enumerate(population):
                if i < elite_size:
                    # Elite individuals already evaluated
                    new_fitness.append(fitness_scores[elite_indices[i]])
                else:
                    obj_values = self._evaluate_objectives(individual, objective_function)
                    overall_score = self._calculate_overall_score(obj_values)
                    new_fitness.append(overall_score)
                    
                    result.all_parameters.append(individual)
                    result.all_objectives.append(obj_values)
                    result.total_evaluations += 1
            
            fitness_scores = np.array(new_fitness)
            
            # Check convergence
            if self._check_convergence(result.convergence_history):
                result.converged = True
                result.convergence_message = "Converged based on fitness improvement"
                break
        
        # Find best solution
        best_idx = np.argmax(fitness_scores)
        result.best_parameters = population[best_idx]
        
        # Find corresponding objective values
        for i, params in enumerate(result.all_parameters):
            if np.allclose(params, result.best_parameters):
                result.best_objective_values = result.all_objectives[i]
                break
        
        result.best_overall_score = fitness_scores[best_idx]
        result.set_end_time()
        
        return result
    
    def _selection(self, fitness_scores: np.ndarray, n_select: int) -> np.ndarray:
        """Select individuals for reproduction."""
        if self.selection_method == "tournament":
            return self._tournament_selection(fitness_scores, n_select)
        elif self.selection_method == "roulette":
            return self._roulette_selection(fitness_scores, n_select)
        elif self.selection_method == "rank":
            return self._rank_selection(fitness_scores, n_select)
        else:
            return self._tournament_selection(fitness_scores, n_select)
    
    def _tournament_selection(self, fitness_scores: np.ndarray, 
                            n_select: int, tournament_size: int = 3) -> np.ndarray:
        """Tournament selection."""
        selected = []
        for _ in range(n_select):
            tournament_indices = np.random.choice(len(fitness_scores), tournament_size, replace=False)
            tournament_fitness = fitness_scores[tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(winner_idx)
        return np.array(selected)
    
    def _roulette_selection(self, fitness_scores: np.ndarray, n_select: int) -> np.ndarray:
        """Roulette wheel selection."""
        # Handle negative fitness scores
        min_fitness = np.min(fitness_scores)
        if min_fitness < 0:
            adjusted_fitness = fitness_scores - min_fitness + 1e-6
        else:
            adjusted_fitness = fitness_scores + 1e-6
        
        probabilities = adjusted_fitness / np.sum(adjusted_fitness)
        selected = np.random.choice(len(fitness_scores), n_select, 
                                  replace=True, p=probabilities)
        return selected
    
    def _rank_selection(self, fitness_scores: np.ndarray, n_select: int) -> np.ndarray:
        """Rank-based selection."""
        ranks = np.argsort(np.argsort(fitness_scores)) + 1
        probabilities = ranks / np.sum(ranks)
        selected = np.random.choice(len(fitness_scores), n_select, 
                                  replace=True, p=probabilities)
        return selected
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform crossover operation."""
        mask = np.random.random(len(parent1)) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        
        # Ensure children are within bounds
        child1 = self._clip_to_bounds(child1)
        child2 = self._clip_to_bounds(child2)
        
        return child1, child2
    
    def _mutation(self, individual: np.ndarray) -> np.ndarray:
        """Gaussian mutation operation."""
        mutated = individual.copy()
        
        for i in range(len(individual)):
            if np.random.random() < self.mutation_rate:
                param = self.parameter_space.parameters[i]
                mutation_range = (param.bounds.max_value - param.bounds.min_value) * 0.1
                mutation = np.random.normal(0, mutation_range)
                mutated[i] += mutation
        
        return self._clip_to_bounds(mutated)
    
    def _clip_to_bounds(self, individual: np.ndarray) -> np.ndarray:
        """Clip individual to parameter bounds."""
        clipped = individual.copy()
        for i, param in enumerate(self.parameter_space.parameters):
            clipped[i] = np.clip(clipped[i], param.bounds.min_value, param.bounds.max_value)
        return clipped
    
    def _check_convergence(self, history: List[float], 
                          window: int = 20, threshold: float = 1e-8) -> bool:
        """Check convergence based on fitness improvement."""
        if len(history) < window:
            return False
        
        recent_improvement = max(history[-window:]) - max(history[-2*window:-window]) \
                            if len(history) >= 2*window else float('inf')
        
        return recent_improvement < threshold


class GradientOptimizer(ParameterOptimizer):
    """Gradient-based optimizer using scipy methods."""
    
    def __init__(self, parameter_space: ParameterSpace,
                 objectives: List[OptimizationObjective],
                 constraints: Optional[List[OptimizationConstraint]] = None,
                 method: str = "L-BFGS-B",
                 random_seed: Optional[int] = None):
        """
        Initialize gradient-based optimizer.
        
        Args:
            parameter_space: Parameter space definition
            objectives: List of optimization objectives
            constraints: Optional list of constraints
            method: Optimization method ("L-BFGS-B", "SLSQP", "trust-constr")
            random_seed: Random seed for reproducibility
        """
        super().__init__(parameter_space, objectives, constraints, random_seed)
        self.method = method
        
        if not HAS_SCIPY:
            raise ImportError("SciPy required for gradient-based optimization")
    
    def optimize(self, objective_function: Callable[[np.ndarray], Dict[str, float]],
                max_evaluations: int = 1000,
                n_restarts: int = 5,
                **kwargs) -> OptimizationResult:
        """
        Perform gradient-based optimization.
        
        Args:
            objective_function: Function to optimize
            max_evaluations: Maximum number of function evaluations
            n_restarts: Number of random restarts
            **kwargs: Additional parameters for scipy.optimize.minimize
            
        Returns:
            Optimization results
        """
        result = OptimizationResult(method=OptimizationMethod.GRADIENT_BASED)
        
        bounds = [(param.bounds.min_value, param.bounds.max_value) 
                 for param in self.parameter_space.parameters]
        
        # Prepare constraints for scipy
        scipy_constraints = []
        for constraint in self.constraints:
            scipy_constraints.append({
                'type': constraint.constraint_type,
                'fun': constraint.constraint_function,
                'tol': constraint.tolerance
            })
        
        best_result = None
        best_score = -float('inf')
        
        # Multiple restarts for global optimization
        for restart in range(n_restarts):
            # Random starting point
            x0 = self.parameter_space.sample(1, seed=None)[0]
            
            # Objective function wrapper
            def scipy_objective(x):
                obj_values = self._evaluate_objectives(x, objective_function)
                score = self._calculate_overall_score(obj_values)
                
                result.all_parameters.append(x.copy())
                result.all_objectives.append(obj_values)
                result.convergence_history.append(score)
                
                # Minimize negative score for maximization
                return -score if self.objectives[0].type == ObjectiveType.MAXIMIZE else score
            
            try:
                # Run optimization
                options = {'maxiter': max_evaluations // n_restarts}
                options.update(kwargs.get('options', {}))
                
                res = minimize(
                    scipy_objective,
                    x0,
                    method=self.method,
                    bounds=bounds,
                    constraints=scipy_constraints if scipy_constraints else None,
                    options=options
                )
                
                if res.success and res.fun < best_score:
                    best_result = res
                    best_score = res.fun
                    
            except Exception as e:
                self.logger.warning(f"Optimization restart {restart} failed: {e}")
                continue
        
        if best_result is not None:
            result.best_parameters = best_result.x
            result.best_overall_score = -best_result.fun if self.objectives[0].type == ObjectiveType.MAXIMIZE else best_result.fun
            result.converged = best_result.success
            result.convergence_message = best_result.message
            
            # Find corresponding objective values
            for i, params in enumerate(result.all_parameters):
                if np.allclose(params, result.best_parameters, rtol=1e-6):
                    result.best_objective_values = result.all_objectives[i]
                    break
        
        result.total_evaluations = len(result.all_parameters)
        result.set_end_time()
        
        return result


# Utility functions for multi-objective optimization
def calculate_pareto_frontier(objectives: np.ndarray) -> np.ndarray:
    """
    Calculate Pareto frontier from multi-objective results.
    
    Args:
        objectives: Array of objective values (n_points x n_objectives)
        
    Returns:
        Boolean array indicating Pareto optimal points
    """
    n_points, n_objectives = objectives.shape
    is_pareto = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        if is_pareto[i]:
            # Check if any other point dominates this one
            dominated = np.all(objectives[i+1:] >= objectives[i], axis=1) & \
                       np.any(objectives[i+1:] > objectives[i], axis=1)
            is_pareto[i+1:][dominated] = False
    
    return is_pareto


def hypervolume_indicator(pareto_front: np.ndarray, reference_point: np.ndarray) -> float:
    """
    Calculate hypervolume indicator for multi-objective optimization.
    
    Args:
        pareto_front: Pareto frontier points
        reference_point: Reference point for hypervolume calculation
        
    Returns:
        Hypervolume value
    """
    if len(pareto_front) == 0:
        return 0.0
    
    # Simple hypervolume calculation for 2D case
    if pareto_front.shape[1] == 2:
        sorted_front = pareto_front[np.argsort(pareto_front[:, 0])]
        volume = 0.0
        
        for i, point in enumerate(sorted_front):
            if i == 0:
                width = point[0] - reference_point[0]
            else:
                width = point[0] - sorted_front[i-1, 0]
            height = point[1] - reference_point[1]
            volume += width * height
        
        return volume
    
    # For higher dimensions, use approximation
    else:
        # Monte Carlo approximation
        n_samples = 10000
        count = 0
        
        # Generate random points in the reference space
        mins = np.minimum(np.min(pareto_front, axis=0), reference_point)
        maxs = np.maximum(np.max(pareto_front, axis=0), reference_point)
        
        for _ in range(n_samples):
            point = np.random.uniform(mins, maxs)
            
            # Check if point is dominated by any Pareto point
            dominated = np.any(np.all(pareto_front >= point, axis=1))
            if dominated and np.all(point >= reference_point):
                count += 1
        
        volume = count / n_samples * np.prod(maxs - mins)
        return volume