"""
Q-learning Hyperparameter Optimization for MFC Substrate Control

Uses Bayesian optimization with Ray Tune to find optimal Q-learning parameters
for stable substrate concentration control at 25 mM target.

Created: 2025-07-26
"""

import json
import os
import sys
import tempfile
from typing import Any, Dict, Tuple

import numpy as np
import optuna
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.qlearning_config import QLearningConfig, QLearningRewardWeights
from mfc_recirculation_control import run_mfc_simulation


class SubstrateControlObjective:
    """
    Objective function for optimizing Q-learning hyperparameters.
    
    Evaluates substrate control performance over shorter simulation periods
    to enable efficient optimization while maintaining relevance to 1000h runs.
    """

    def __init__(self,
                 duration_hours: int = 200,
                 target_concentration: float = 25.0,
                 tolerance: float = 2.0):
        """
        Initialize optimization objective.
        
        Args:
            duration_hours: Simulation duration for evaluation (shorter for speed)
            target_concentration: Target substrate concentration (mM)
            tolerance: Acceptable deviation from target (mM)
        """
        self.duration_hours = duration_hours
        self.target_concentration = target_concentration
        self.tolerance = tolerance

    def __call__(self, config: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate Q-learning configuration performance.
        
        Args:
            config: Ray Tune configuration dictionary
            
        Returns:
            Dictionary with optimization metrics
        """
        try:
            # Create temporary directory for this trial
            with tempfile.TemporaryDirectory() as temp_dir:
                # Build Q-learning configuration from Ray Tune parameters
                qlearning_config = self._build_qlearning_config(config)

                # Run simulation
                results = run_mfc_simulation(
                    duration_hours=self.duration_hours,
                    output_dir=temp_dir,
                    config=qlearning_config,
                    n_cells=5,
                    initial_substrate_concentration=self.target_concentration,
                    user_suffix="optimization_trial",
                    verbose=False
                )

                # Debug: Check what we got
                if not results:
                    print("ERROR: No results returned from simulation")
                    raise ValueError("Simulation returned empty results")

                print(f"DEBUG: Results keys: {list(results.keys())}")
                print(f"DEBUG: Reservoir concentration length: {len(results.get('reservoir_concentration', []))}")

                # Calculate performance metrics
                metrics = self._calculate_performance_metrics(results)

                # Ray Tune expects to minimize the objective
                return {
                    "loss": metrics["substrate_control_loss"],
                    "substrate_deviation": metrics["substrate_deviation"],
                    "stability_score": metrics["stability_score"],
                    "power_efficiency": metrics["power_efficiency"],
                    "final_concentration": metrics["final_concentration"]
                }

        except Exception as e:
            print(f"ERROR in optimization trial: {e}")
            import traceback
            traceback.print_exc()
            # Return high loss for failed trials
            return {
                "loss": 1000.0,
                "substrate_deviation": 100.0,
                "stability_score": 0.0,
                "power_efficiency": 0.0,
                "final_concentration": 0.0
            }

    def _build_qlearning_config(self, config: Dict[str, Any]) -> QLearningConfig:
        """Build QLearningConfig from optimization parameters."""

        # Create reward weights with optimized parameters
        reward_weights = QLearningRewardWeights(
            power_weight=config["power_weight"],
            substrate_reward_multiplier=config["substrate_reward_multiplier"],
            substrate_penalty_multiplier=config["substrate_penalty_multiplier"],
            substrate_excess_penalty=config["substrate_excess_penalty"],
            biofilm_weight=config["biofilm_weight"],
            efficiency_weight=config["efficiency_weight"]
        )

        # Create base config with optimized learning parameters
        qlearning_config = QLearningConfig()
        qlearning_config.learning_rate = config["learning_rate"]
        qlearning_config.discount_factor = config["discount_factor"]
        qlearning_config.epsilon = config["epsilon_initial"]
        qlearning_config.epsilon_decay = config["epsilon_decay"]
        qlearning_config.epsilon_min = config["epsilon_min"]
        qlearning_config.rewards = reward_weights

        # Set outlet penalty multiplier and other config-level parameters
        qlearning_config.outlet_penalty_multiplier = config["outlet_penalty_multiplier"]
        qlearning_config.substrate_penalty_base_multiplier = config["substrate_penalty_base_multiplier"]

        return qlearning_config

    def _calculate_performance_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate optimization metrics from simulation results."""

        # Extract time series data
        reservoir_conc = results.get("reservoir_concentration", [])
        outlet_conc = results.get("outlet_concentration", [])
        power_output = results.get("total_power", [])  # Fixed key name

        if not reservoir_conc or len(reservoir_conc) < 5:
            return {
                "substrate_control_loss": 1000.0,
                "substrate_deviation": 100.0,
                "stability_score": 0.0,
                "power_efficiency": 0.0,
                "final_concentration": 0.0
            }

        # Convert to numpy arrays
        reservoir_conc = np.array(reservoir_conc)
        outlet_conc = np.array(outlet_conc)
        power_output = np.array(power_output)

        # 1. Substrate control performance (primary objective)
        target_deviation = np.abs(reservoir_conc - self.target_concentration)
        mean_deviation = np.mean(target_deviation)
        max_deviation = np.max(target_deviation)

        # Penalize both mean deviation and maximum excursions
        substrate_control_loss = (
            mean_deviation +
            0.5 * max_deviation +
            10.0 * np.sum(target_deviation > (self.tolerance * 2))  # Heavy penalty for large excursions
        )

        # 2. Stability assessment (second half of simulation)
        second_half_idx = len(reservoir_conc) // 2
        second_half_conc = reservoir_conc[second_half_idx:]
        stability_variance = np.var(second_half_conc)
        stability_score = max(0.0, 1.0 - stability_variance / 100.0)  # Higher is better

        # 3. Power efficiency (avoid excessive power consumption)
        mean_power = np.mean(power_output)
        power_efficiency = min(1.0, 100.0 / max(mean_power, 1.0))  # Reward moderate power

        # 4. Final concentration check
        final_concentration = reservoir_conc[-1]

        return {
            "substrate_control_loss": substrate_control_loss,
            "substrate_deviation": mean_deviation,
            "stability_score": stability_score,
            "power_efficiency": power_efficiency,
            "final_concentration": final_concentration
        }


def setup_optimization_search_space() -> Dict[str, Any]:
    """
    Define the hyperparameter search space for Bayesian optimization.
    
    Based on analysis of current Q-learning performance issues, focuses on:
    - Learning rate and exploration parameters
    - Reward/penalty balance for substrate control
    - Multi-objective weight tuning
    """

    search_space = {
        # Core Q-learning parameters
        "learning_rate": tune.uniform(0.01, 0.3),
        "discount_factor": tune.uniform(0.90, 0.99),
        "epsilon_initial": tune.uniform(0.1, 0.5),
        "epsilon_decay": tune.uniform(0.995, 0.9999),
        "epsilon_min": tune.uniform(0.001, 0.05),

        # Reward system weights (critical for substrate control)
        "power_weight": tune.uniform(5.0, 20.0),
        "substrate_reward_multiplier": tune.uniform(10.0, 50.0),
        "substrate_penalty_multiplier": tune.uniform(30.0, 100.0),
        "substrate_excess_penalty": tune.uniform(-200.0, -50.0),
        "biofilm_weight": tune.uniform(20.0, 80.0),
        "efficiency_weight": tune.uniform(10.0, 40.0),

        # Outlet sensor penalty system
        "outlet_penalty_multiplier": tune.uniform(1.05, 1.5),
        "substrate_penalty_base_multiplier": tune.uniform(0.5, 2.0),
    }

    return search_space


def run_bayesian_optimization(
    num_samples: int = 50,
    max_concurrent_trials: int = 4,
    duration_hours: int = 200,
    target_concentration: float = 25.0
) -> Tuple[Dict[str, Any], str]:
    """
    Run Bayesian optimization to find optimal Q-learning hyperparameters.
    
    Args:
        num_samples: Number of trials to run
        max_concurrent_trials: Maximum parallel trials
        duration_hours: Simulation duration per trial
        target_concentration: Target substrate concentration
        
    Returns:
        Tuple of (best_config, results_path)
    """

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=max_concurrent_trials, ignore_reinit_error=True)

    # Create objective function
    objective = SubstrateControlObjective(
        duration_hours=duration_hours,
        target_concentration=target_concentration
    )

    # Set up Optuna search
    optuna_search = OptunaSearch(
        metric="loss",
        mode="min",
        points_to_evaluate=None,
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    # Set up ASHA scheduler for early stopping
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=duration_hours,
        grace_period=duration_hours // 4,
        reduction_factor=2
    )

    # Configure Ray Tune
    tuner = tune.Tuner(
        objective,
        param_space=setup_optimization_search_space(),
        tune_config=tune.TuneConfig(
            search_alg=optuna_search,
            scheduler=scheduler,
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent_trials,
        ),
        run_config=ray.tune.RunConfig(
            name="qlearning_substrate_optimization",
            storage_path=os.path.abspath("./optimization_results"),
            stop={"training_iteration": 1},  # Single evaluation per trial
        )
    )

    # Run optimization
    print(f"Starting Bayesian optimization with {num_samples} trials...")
    print(f"Target: {target_concentration} mM substrate concentration")
    print(f"Simulation duration per trial: {duration_hours} hours")

    results = tuner.fit()

    # Get best result
    best_result = results.get_best_result("loss", "min")
    best_config = best_result.config
    best_metrics = best_result.metrics

    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"Best loss: {best_metrics['loss']:.3f}")
    print(f"Substrate deviation: {best_metrics['substrate_deviation']:.3f} mM")
    print(f"Stability score: {best_metrics['stability_score']:.3f}")
    print(f"Power efficiency: {best_metrics['power_efficiency']:.3f}")
    print(f"Final concentration: {best_metrics['final_concentration']:.1f} mM")
    print("\nBest hyperparameters:")
    for key, value in best_config.items():
        print(f"  {key}: {value:.4f}")

    # Save results
    results_dir = "./optimization_results"
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, "best_qlearning_config.json")
    with open(results_file, 'w') as f:
        json.dump({
            "best_config": best_config,
            "best_metrics": best_metrics,
            "optimization_settings": {
                "num_samples": num_samples,
                "duration_hours": duration_hours,
                "target_concentration": target_concentration
            }
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Shutdown Ray
    ray.shutdown()

    return best_config, results_file


def apply_optimized_config(best_config: Dict[str, Any]) -> QLearningConfig:
    """
    Apply optimized hyperparameters to create production Q-learning configuration.
    
    Args:
        best_config: Optimized hyperparameters from Bayesian optimization
        
    Returns:
        QLearningConfig with optimized parameters
    """

    # Create optimized reward weights
    optimized_rewards = QLearningRewardWeights(
        power_weight=best_config["power_weight"],
        substrate_reward_multiplier=best_config["substrate_reward_multiplier"],
        substrate_penalty_multiplier=best_config["substrate_penalty_multiplier"],
        substrate_excess_penalty=best_config["substrate_excess_penalty"],
        biofilm_weight=best_config["biofilm_weight"],
        efficiency_weight=best_config["efficiency_weight"]
    )

    # Create optimized Q-learning configuration
    optimized_config = QLearningConfig()
    optimized_config.learning_rate = best_config["learning_rate"]
    optimized_config.discount_factor = best_config["discount_factor"]
    optimized_config.epsilon = best_config["epsilon_initial"]
    optimized_config.epsilon_decay = best_config["epsilon_decay"]
    optimized_config.epsilon_min = best_config["epsilon_min"]
    optimized_config.rewards = optimized_rewards

    # Set config-level parameters
    optimized_config.outlet_penalty_multiplier = best_config["outlet_penalty_multiplier"]
    optimized_config.substrate_penalty_base_multiplier = best_config["substrate_penalty_base_multiplier"]

    return optimized_config


if __name__ == "__main__":
    """
    Run Q-learning hyperparameter optimization.
    
    Usage:
        python hyperparameter_optimization.py [num_samples] [concurrent_trials]
    """

    import argparse

    parser = argparse.ArgumentParser(description="Q-learning Hyperparameter Optimization")
    parser.add_argument("--samples", type=int, default=50,
                       help="Number of optimization trials (default: 50)")
    parser.add_argument("--concurrent", type=int, default=4,
                       help="Maximum concurrent trials (default: 4)")
    parser.add_argument("--duration", type=int, default=200,
                       help="Simulation duration per trial in hours (default: 200)")
    parser.add_argument("--target", type=float, default=25.0,
                       help="Target substrate concentration in mM (default: 25.0)")

    args = parser.parse_args()

    # Run optimization
    best_config, results_path = run_bayesian_optimization(
        num_samples=args.samples,
        max_concurrent_trials=args.concurrent,
        duration_hours=args.duration,
        target_concentration=args.target
    )

    print("\nTo test optimized configuration, run:")
    print(f"python test_optimized_config.py {results_path}")
