#!/usr/bin/env python3
"""
MFC Unified Q-Learning Hyperparameter Optimization with Optuna
==============================================================

This module implements automated hyperparameter optimization for the MFC
unified Q-learning controller using Optuna framework. It systematically
explores the parameter space to find optimal reward function weights and
control parameters.

Key Features:
- Bayesian optimization with Tree-structured Parzen Estimator (TPE)
- Multi-objective optimization (energy, biofilm control, concentration precision)
- Parallel trial execution for faster optimization
- Comprehensive logging and result tracking
- Automatic best parameters extraction

Author: Claude & User
Date: 2025-07-23
"""

import json
import logging
import multiprocessing as mp
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna

# Import the MFC simulation class
from mfc_unified_qlearning_control import MFCUnifiedQLearningSimulation


class MFCOptunaOptimizer:
    """
    Automated hyperparameter optimization for MFC Q-learning controller using Optuna
    """

    def __init__(self,
                 n_trials: int = 100,
                 n_jobs: int = 4,
                 study_name: str = "mfc_optimization",
                 storage: str | None = None,
                 timeout: int = 3600):  # 1 hour timeout per trial
        """
        Initialize Optuna optimizer

        Args:
            n_trials: Number of optimization trials
            n_jobs: Number of parallel jobs
            study_name: Name for the optimization study
            storage: Database storage URL (None for in-memory)
            timeout: Timeout per trial in seconds
        """
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.study_name = study_name
        self.storage = storage
        self.timeout = timeout

        # Create results directory
        self.results_dir = Path("optuna_results")
        self.results_dir.mkdir(exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Optimization targets
        self.target_biofilm = 1.3
        self.target_outlet_conc = 12.0

    def setup_logging(self):
        """Setup comprehensive logging"""
        log_file = self.results_dir / f"optuna_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def define_search_space(self, trial) -> dict:
        """
        Define hyperparameter search space for Optuna

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of hyperparameters to optimize
        """
        params = {}

        # === REWARD FUNCTION PARAMETERS ===

        # Biofilm reward parameters
        params['biofilm_base_reward'] = trial.suggest_float('biofilm_base_reward', 20.0, 60.0)
        params['biofilm_steady_bonus'] = trial.suggest_float('biofilm_steady_bonus', 10.0, 40.0)
        params['biofilm_penalty_multiplier'] = trial.suggest_float('biofilm_penalty_multiplier', 50.0, 100.0)

        # Power reward parameters
        params['power_increase_multiplier'] = trial.suggest_float('power_increase_multiplier', 30.0, 80.0)
        params['power_decrease_multiplier'] = trial.suggest_float('power_decrease_multiplier', 60.0, 150.0)

        # Substrate reward parameters
        params['substrate_increase_multiplier'] = trial.suggest_float('substrate_increase_multiplier', 20.0, 50.0)
        params['substrate_decrease_multiplier'] = trial.suggest_float('substrate_decrease_multiplier', 40.0, 100.0)

        # Concentration control parameters
        params['conc_precise_reward'] = trial.suggest_float('conc_precise_reward', 15.0, 30.0)
        params['conc_acceptable_reward'] = trial.suggest_float('conc_acceptable_reward', 3.0, 8.0)
        params['conc_poor_penalty'] = trial.suggest_float('conc_poor_penalty', -15.0, -5.0)

        # Flow penalty parameters
        params['flow_penalty_threshold'] = trial.suggest_float('flow_penalty_threshold', 15.0, 25.0)
        params['flow_penalty_multiplier'] = trial.suggest_float('flow_penalty_multiplier', 15.0, 35.0)
        params['biofilm_threshold_ratio'] = trial.suggest_float('biofilm_threshold_ratio', 0.85, 0.95)

        # === ACTION SPACE PARAMETERS ===

        # Flow action bounds
        params['max_flow_decrease'] = trial.suggest_int('max_flow_decrease', -12, -6)
        params['max_flow_increase'] = trial.suggest_int('max_flow_increase', 3, 8)

        # Substrate action bounds
        params['max_substrate_decrease'] = trial.suggest_float('max_substrate_decrease', -3.0, -1.0)
        params['max_substrate_increase'] = trial.suggest_float('max_substrate_increase', 1.0, 2.5)
        params['substrate_increment_fineness'] = trial.suggest_categorical('substrate_increment_fineness',
                                                                         ['coarse', 'medium', 'fine'])

        # === Q-LEARNING PARAMETERS ===

        params['learning_rate'] = trial.suggest_float('learning_rate', 0.05, 0.2)
        params['discount_factor'] = trial.suggest_float('discount_factor', 0.90, 0.99)
        params['initial_epsilon'] = trial.suggest_float('initial_epsilon', 0.3, 0.5)
        params['epsilon_decay'] = trial.suggest_float('epsilon_decay', 0.990, 0.999)
        params['epsilon_min'] = trial.suggest_float('epsilon_min', 0.05, 0.15)

        return params

    def create_modified_simulation(self, params: dict) -> MFCUnifiedQLearningSimulation:
        """
        Create MFC simulation with optimized parameters

        Args:
            params: Optimized hyperparameters

        Returns:
            Modified MFC simulation instance
        """
        # Create simulation instance with correct parameters
        sim = MFCUnifiedQLearningSimulation(
            use_gpu=False,  # CPU for stability during optimization
            target_outlet_conc=self.target_outlet_conc
        )

        # Override simulation parameters for optimization speed
        sim.total_time = 120 * 3600  # 120 hours in seconds for parallel efficiency
        sim.num_steps = int(sim.total_time / sim.dt)

        # Reinitialize arrays with new dimensions
        array_func = np.zeros if not sim.use_gpu else (lambda shape: np.zeros(shape))

        # Cell state arrays [time_step, cell_index]
        sim.cell_voltages = array_func((sim.num_steps, sim.num_cells))
        sim.biofilm_thickness = array_func((sim.num_steps, sim.num_cells))
        sim.acetate_concentrations = array_func((sim.num_steps, sim.num_cells))
        sim.current_densities = array_func((sim.num_steps, sim.num_cells))
        sim.power_outputs = array_func((sim.num_steps, sim.num_cells))
        sim.substrate_consumptions = array_func((sim.num_steps, sim.num_cells))

        # Stack-level arrays [time_step]
        sim.stack_voltages = array_func(sim.num_steps)
        sim.stack_powers = array_func(sim.num_steps)
        sim.flow_rates = array_func(sim.num_steps)
        sim.objective_values = array_func(sim.num_steps)
        sim.substrate_utilizations = array_func(sim.num_steps)
        sim.q_rewards = array_func(sim.num_steps)
        sim.q_actions = array_func(sim.num_steps)

        # Unified control arrays
        sim.inlet_concentrations = array_func(sim.num_steps)
        sim.outlet_concentrations = array_func(sim.num_steps)
        sim.concentration_errors = array_func(sim.num_steps)

        # Reinitialize starting conditions
        sim.biofilm_thickness[0, :] = 1.0
        sim.acetate_concentrations[0, :] = 20.0
        sim.flow_rates[0] = 0.010  # 10 mL/h in L/h
        sim.inlet_concentrations[0] = 20.0

        # Override reward calculation method with optimized parameters
        # Store original method (currently unused but may be needed for restoration)
        # original_calculate_reward = sim.unified_controller.calculate_unified_reward

        def optimized_reward_function(power, biofilm_deviation, substrate_utilization,
                                    outlet_conc, prev_power, prev_biofilm_dev,
                                    prev_substrate_util, prev_outlet_conc,
                                    biofilm_thickness_history=None):
            """Optimized reward function with Optuna parameters"""

            # Calculate changes
            power_change = power - prev_power
            substrate_change = substrate_utilization - prev_substrate_util
            error_improvement = abs(prev_outlet_conc - self.target_outlet_conc) - abs(outlet_conc - self.target_outlet_conc)

            # 1. POWER COMPONENT (optimized)
            if power_change > 0:
                power_reward = power_change * params['power_increase_multiplier']
            else:
                power_reward = power_change * params['power_decrease_multiplier']

            power_base = 10.0 if power > 0.005 else -5.0

            # 2. SUBSTRATE COMPONENT (optimized)
            if substrate_change > 0:
                substrate_reward = substrate_change * params['substrate_increase_multiplier']
            else:
                substrate_reward = substrate_change * params['substrate_decrease_multiplier']

            substrate_base = 5.0 if substrate_utilization > 10.0 else -2.0

            # 3. BIOFILM COMPONENT (optimized)
            optimal_thickness = self.target_biofilm
            deviation_threshold = 0.05 * optimal_thickness

            if biofilm_deviation <= deviation_threshold:
                biofilm_reward = params['biofilm_base_reward'] - (biofilm_deviation / deviation_threshold) * 15.0

                # Steady state bonus (optimized)
                if biofilm_thickness_history is not None and len(biofilm_thickness_history) >= 3:
                    recent_thickness = biofilm_thickness_history[-3:]
                    if len(recent_thickness) >= 2:
                        growth_rate = abs(recent_thickness[-1] - recent_thickness[-2])
                        if growth_rate < 0.01:
                            biofilm_reward += params['biofilm_steady_bonus']
            else:
                excess_deviation = biofilm_deviation - deviation_threshold
                biofilm_reward = -params['biofilm_penalty_multiplier'] * (excess_deviation / deviation_threshold)

            # 4. CONCENTRATION CONTROL (optimized)
            outlet_error = abs(outlet_conc - self.target_outlet_conc)

            if outlet_error <= 0.5:
                concentration_reward = params['conc_precise_reward'] - (outlet_error * 10.0)
            elif outlet_error <= 2.0:
                concentration_reward = params['conc_acceptable_reward'] - (outlet_error * 2.5)
            else:
                concentration_reward = params['conc_poor_penalty'] - (outlet_error * 5.0)

            concentration_base = 3.0 if error_improvement > 0 else -1.0

            # 5. STABILITY BONUS
            stability_bonus = 0
            if (abs(power_change) < 0.001 and abs(substrate_change) < 0.5 and
                outlet_error < 1.0 and biofilm_deviation <= deviation_threshold):
                stability_bonus = 30.0

            # 6. FLOW PENALTY (optimized)
            flow_penalty = 0
            current_flow_rate = getattr(sim.unified_controller, 'current_flow_rate', 10.0)
            if biofilm_thickness_history is not None and len(biofilm_thickness_history) > 0:
                avg_biofilm = np.mean(biofilm_thickness_history[-5:])
                if avg_biofilm < optimal_thickness * params['biofilm_threshold_ratio']:
                    if current_flow_rate > params['flow_penalty_threshold']:
                        flow_penalty = -params['flow_penalty_multiplier'] * (current_flow_rate - params['flow_penalty_threshold']) / 10.0

            # 7. COMBINED PENALTY
            combined_penalty = 0
            if (power_change < 0 and substrate_change < 0 and
                error_improvement < 0 and biofilm_deviation > deviation_threshold):
                combined_penalty = -200.0

            # Total reward
            total_reward = (power_reward + power_base +
                           substrate_reward + substrate_base +
                           biofilm_reward +
                           concentration_reward + concentration_base +
                           stability_bonus + flow_penalty + combined_penalty)

            return total_reward

        # Replace reward function
        sim.unified_controller.calculate_unified_reward = optimized_reward_function

        # Update Q-learning parameters
        sim.unified_controller.learning_rate = params['learning_rate']
        sim.unified_controller.discount_factor = params['discount_factor']
        sim.unified_controller.epsilon = params['initial_epsilon']
        sim.unified_controller.epsilon_decay = params['epsilon_decay']
        sim.unified_controller.epsilon_min = params['epsilon_min']

        # Update action space based on parameters
        self.update_action_space(sim, params)

        return sim

    def update_action_space(self, sim, params: dict):
        """Update action space based on optimized parameters"""

        # Create optimized flow actions
        flow_actions = list(range(params['max_flow_decrease'], params['max_flow_increase'] + 1))

        # Create optimized substrate actions based on fineness
        if params['substrate_increment_fineness'] == 'coarse':
            substrate_step = 1.0
        elif params['substrate_increment_fineness'] == 'medium':
            substrate_step = 0.5
        else:  # fine
            substrate_step = 0.25

        substrate_actions = []
        val = params['max_substrate_decrease']
        while val <= params['max_substrate_increase']:
            substrate_actions.append(val)
            val += substrate_step

        # Update controller actions
        sim.unified_controller.actions = []
        for flow_adj in flow_actions:
            for substr_adj in substrate_actions:
                sim.unified_controller.actions.append((flow_adj, substr_adj))

        self.logger.info(f"Updated action space: {len(flow_actions)} flow × {len(substrate_actions)} substrate = {len(sim.unified_controller.actions)} total actions")

    def objective(self, trial) -> float:
        """
        Objective function for Optuna optimization

        Args:
            trial: Optuna trial object

        Returns:
            Objective value to minimize (lower is better)
        """
        try:
            # Get hyperparameters for this trial
            params = self.define_search_space(trial)

            # Create modified simulation
            sim = self.create_modified_simulation(params)

            self.logger.info(f"Trial {trial.number}: Starting simulation with {len(sim.unified_controller.actions)} actions")

            # Run simulation
            start_time = time.time()
            sim.run_simulation()  # This method doesn't return anything
            elapsed_time = time.time() - start_time

            # Extract key metrics directly from simulation object
            energy_total = float(np.trapezoid(sim.stack_powers, dx=sim.dt/3600))
            final_power = float(sim.stack_powers[-1])

            # Calculate control metrics
            valid_errors = sim.concentration_errors[sim.concentration_errors != 0]
            if len(valid_errors) > 0:
                control_rmse = float(np.sqrt(np.mean(valid_errors**2)))
                control_mae = float(np.mean(np.abs(valid_errors)))
            else:
                control_rmse = 100.0
                control_mae = 100.0

            substrate_utilization = float(sim.substrate_utilizations[-1])

            # Get Q-learning reward from controller
            q_reward = float(np.sum(sim.q_rewards))

            # Calculate biofilm deviation (if available)
            final_biofilm = getattr(sim, 'biofilm_thickness', np.array([[0.5]]))
            if hasattr(final_biofilm, 'shape') and len(final_biofilm.shape) > 1:
                avg_final_biofilm = np.mean(final_biofilm[-1, :])
            else:
                avg_final_biofilm = 0.5

            biofilm_error = abs(avg_final_biofilm - self.target_biofilm)

            # Multi-objective optimization: weighted combination
            # Primary objectives with weights
            control_objective = control_rmse * 0.4  # 40% weight on concentration control
            biofilm_objective = biofilm_error * 20.0 * 0.3  # 30% weight on biofilm control
            energy_objective = (10.0 - energy_total) * 0.2  # 20% weight on energy (minimize negative)
            stability_objective = abs(q_reward / 1e6) * 0.1  # 10% weight on learning stability

            total_objective = control_objective + biofilm_objective + energy_objective + stability_objective

            # Log trial results
            self.logger.info(f"Trial {trial.number} completed in {elapsed_time:.1f}s:")
            self.logger.info(f"  Energy: {energy_total:.3f} Wh, Power: {final_power:.4f} W")
            self.logger.info(f"  Control RMSE: {control_rmse:.3f}, MAE: {control_mae:.3f}")
            self.logger.info(f"  Biofilm: {avg_final_biofilm:.3f} (target: {self.target_biofilm})")
            self.logger.info(f"  Substrate util: {substrate_utilization:.3f}%")
            self.logger.info(f"  Q-reward: {q_reward:.0f}")
            self.logger.info(f"  Objective: {total_objective:.6f}")

            # Store additional metrics for analysis
            trial.set_user_attr('energy_total', energy_total)
            trial.set_user_attr('final_power', final_power)
            trial.set_user_attr('control_rmse', control_rmse)
            trial.set_user_attr('control_mae', control_mae)
            trial.set_user_attr('biofilm_error', biofilm_error)
            trial.set_user_attr('avg_final_biofilm', avg_final_biofilm)
            trial.set_user_attr('substrate_utilization', substrate_utilization)
            trial.set_user_attr('q_reward', q_reward)
            trial.set_user_attr('simulation_time', elapsed_time)
            trial.set_user_attr('n_actions', len(sim.unified_controller.actions))

            return total_objective

        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {str(e)}")
            return 1000.0  # High penalty for failed trials

    def run_optimization(self) -> optuna.Study:
        """
        Run the complete optimization process

        Returns:
            Completed Optuna study
        """
        self.logger.info(f"Starting MFC Q-Learning optimization with {self.n_trials} trials")
        self.logger.info(f"Target biofilm thickness: {self.target_biofilm}")
        self.logger.info(f"Target outlet concentration: {self.target_outlet_conc} mmol/L")

        # Create study
        if self.storage:
            study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage,
                direction='minimize',
                load_if_exists=True
            )
        else:
            study = optuna.create_study(
                study_name=self.study_name,
                direction='minimize'
            )

        # Set up pruner for early stopping of unpromising trials (optimized for 14-thread)
        study.pruner = optuna.pruners.MedianPruner(
            n_startup_trials=28,  # 2 trials per thread for warmup
            n_warmup_steps=7,     # Faster warmup for parallel execution
            interval_steps=3      # More frequent pruning checks
        )

        # Run optimization
        try:
            if self.n_jobs > 1:
                self.logger.info(f"Running optimization with {self.n_jobs} parallel jobs")
                study.optimize(
                    self.objective,
                    n_trials=self.n_trials,
                    n_jobs=self.n_jobs,
                    timeout=self.timeout * self.n_trials
                )
            else:
                self.logger.info("Running optimization sequentially")
                study.optimize(
                    self.objective,
                    n_trials=self.n_trials,
                    timeout=self.timeout * self.n_trials
                )

        except KeyboardInterrupt:
            self.logger.info("Optimization interrupted by user")

        # Save results
        self.save_results(study)

        return study

    def save_results(self, study: optuna.Study):
        """Save optimization results to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save best parameters
        best_params_file = self.results_dir / f"best_parameters_{timestamp}.json"
        with open(best_params_file, 'w') as f:
            json.dump(study.best_params, f, indent=2)

        # Save study summary
        summary = {
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
            'study_name': study.study_name,
            'optimization_target': {
                'biofilm_thickness': self.target_biofilm,
                'outlet_concentration': self.target_outlet_conc
            }
        }

        # Add best trial user attributes
        best_trial = study.best_trial
        if best_trial.user_attrs:
            summary['best_trial_metrics'] = best_trial.user_attrs

        summary_file = self.results_dir / f"optimization_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Save all trials data
        trials_data = []
        for trial in study.trials:
            trial_data = {
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name,
                'user_attrs': trial.user_attrs
            }
            trials_data.append(trial_data)

        trials_file = self.results_dir / f"all_trials_{timestamp}.json"
        with open(trials_file, 'w') as f:
            json.dump(trials_data, f, indent=2)

        self.logger.info(f"Results saved to {self.results_dir}")
        self.logger.info(f"Best value: {study.best_value:.6f}")
        self.logger.info(f"Best parameters saved to: {best_params_file}")

    def print_optimization_summary(self, study: optuna.Study):
        """Print comprehensive optimization summary"""
        print("\n" + "="*80)
        print("MFC Q-LEARNING OPTUNA OPTIMIZATION RESULTS")
        print("="*80)

        print(f"Study name: {study.study_name}")
        print(f"Number of finished trials: {len(study.trials)}")
        print(f"Best objective value: {study.best_value:.6f}")

        print("\nBest parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        if study.best_trial.user_attrs:
            print("\nBest trial performance metrics:")
            attrs = study.best_trial.user_attrs
            print(f"  Energy: {attrs.get('energy_total', 'N/A'):.3f} Wh")
            print(f"  Final Power: {attrs.get('final_power', 'N/A'):.4f} W")
            print(f"  Control RMSE: {attrs.get('control_rmse', 'N/A'):.3f} mmol/L")
            print(f"  Control MAE: {attrs.get('control_mae', 'N/A'):.3f} mmol/L")
            print(f"  Biofilm Deviation: {attrs.get('biofilm_error', 'N/A'):.3f}")
            print(f"  Final Biofilm: {attrs.get('avg_final_biofilm', 'N/A'):.3f}")
            print(f"  Substrate Utilization: {attrs.get('substrate_utilization', 'N/A'):.3f}%")
            print(f"  Q-Learning Reward: {attrs.get('q_reward', 'N/A'):.0f}")
            print(f"  Simulation Time: {attrs.get('simulation_time', 'N/A'):.1f}s")
            print(f"  Number of Actions: {attrs.get('n_actions', 'N/A')}")

        print("\n" + "="*80)

    def get_top_configurations(self, study: optuna.Study, n_configs: int = 14) -> list[dict]:
        """
        Extract top N configurations from completed study

        Args:
            study: Completed Optuna study
            n_configs: Number of top configurations to extract

        Returns:
            List of parameter dictionaries for top configurations
        """
        # Sort trials by objective value (lower is better)
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        sorted_trials = sorted(completed_trials, key=lambda x: x.value)

        # Extract top N configurations
        top_configs = []
        for i, trial in enumerate(sorted_trials[:n_configs]):
            config = {
                'trial_number': trial.number,
                'objective_value': trial.value,
                'params': trial.params.copy(),
                'original_metrics': trial.user_attrs.copy() if trial.user_attrs else {}
            }
            top_configs.append(config)

            self.logger.info(f"Top config {i+1}: Trial {trial.number}, Objective: {trial.value:.6f}")

        return top_configs

    def run_extended_validation(self, configs: list[dict]) -> list[dict]:
        """
        Run extended 600-hour simulations on top configurations in parallel

        Args:
            configs: List of configuration dictionaries

        Returns:
            List of validation results
        """
        self.logger.info(f"Starting extended validation of {len(configs)} configurations")
        self.logger.info("Each configuration will run for 600 hours (vs 120h during optimization)")

        validation_results = []

        # Use multiprocessing for parallel validation
        if len(configs) <= 14:  # Run all in parallel if we have enough threads
            import concurrent.futures

            with concurrent.futures.ProcessPoolExecutor(max_workers=min(14, len(configs))) as executor:
                # Submit all validation jobs
                future_to_config = {}
                for i, config in enumerate(configs):
                    future = executor.submit(self._run_single_extended_validation, config, i+1)
                    future_to_config[future] = config

                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_config):
                    config = future_to_config[future]
                    try:
                        result = future.result(timeout=3600)  # 1 hour timeout per validation
                        validation_results.append(result)
                        self.logger.info(f"Validation completed for config {config['trial_number']}")
                    except Exception as e:
                        self.logger.error(f"Validation failed for config {config['trial_number']}: {str(e)}")
                        # Add failed result
                        validation_results.append({
                            'config': config,
                            'status': 'failed',
                            'error': str(e)
                        })
        else:
            # Sequential execution if too many configs
            for i, config in enumerate(configs):
                try:
                    result = self._run_single_extended_validation(config, i+1)
                    validation_results.append(result)
                except Exception as e:
                    self.logger.error(f"Validation failed for config {config['trial_number']}: {str(e)}")
                    validation_results.append({
                        'config': config,
                        'status': 'failed',
                        'error': str(e)
                    })

        # Sort by performance and save results
        successful_results = [r for r in validation_results if r.get('status') != 'failed']
        if successful_results:
            successful_results.sort(key=lambda x: x['extended_objective'])

        self._save_validation_results(validation_results)
        self._print_validation_summary(validation_results)

        return validation_results

    def _run_single_extended_validation(self, config: dict, config_num: int) -> dict:
        """
        Run single extended validation simulation

        Args:
            config: Configuration dictionary
            config_num: Configuration number for logging

        Returns:
            Validation result dictionary
        """
        self.logger.info(f"Starting extended validation {config_num}/14 (Trial {config['trial_number']})")

        try:
            # Create extended simulation (600 hours)
            sim = MFCUnifiedQLearningSimulation(
                use_gpu=False,
                target_outlet_conc=self.target_outlet_conc
            )

            # Override simulation parameters for extended validation
            sim.total_time = 600 * 3600  # 600 hours in seconds for extended validation
            sim.num_steps = int(sim.total_time / sim.dt)

            # Reinitialize arrays with new dimensions for extended simulation
            array_func = np.zeros if not sim.use_gpu else (lambda shape: np.zeros(shape))

            # Cell state arrays [time_step, cell_index]
            sim.cell_voltages = array_func((sim.num_steps, sim.num_cells))
            sim.biofilm_thickness = array_func((sim.num_steps, sim.num_cells))
            sim.acetate_concentrations = array_func((sim.num_steps, sim.num_cells))
            sim.current_densities = array_func((sim.num_steps, sim.num_cells))
            sim.power_outputs = array_func((sim.num_steps, sim.num_cells))
            sim.substrate_consumptions = array_func((sim.num_steps, sim.num_cells))

            # Stack-level arrays [time_step]
            sim.stack_voltages = array_func(sim.num_steps)
            sim.stack_powers = array_func(sim.num_steps)
            sim.flow_rates = array_func(sim.num_steps)
            sim.objective_values = array_func(sim.num_steps)
            sim.substrate_utilizations = array_func(sim.num_steps)
            sim.q_rewards = array_func(sim.num_steps)
            sim.q_actions = array_func(sim.num_steps)

            # Unified control arrays
            sim.inlet_concentrations = array_func(sim.num_steps)
            sim.outlet_concentrations = array_func(sim.num_steps)
            sim.concentration_errors = array_func(sim.num_steps)

            # Reinitialize starting conditions
            sim.biofilm_thickness[0, :] = 1.0
            sim.acetate_concentrations[0, :] = 20.0
            sim.flow_rates[0] = 0.010  # 10 mL/h in L/h
            sim.inlet_concentrations[0] = 20.0

            # Apply the optimized parameters (reuse the same logic from optimization)
            self._apply_parameters_to_simulation(sim, config['params'])

            # Run extended simulation
            start_time = time.time()
            sim.run_simulation()  # This method doesn't return anything
            elapsed_time = time.time() - start_time

            # Calculate extended metrics directly from simulation object
            energy_total = float(np.trapezoid(sim.stack_powers, dx=sim.dt/3600))
            final_power = float(sim.stack_powers[-1])

            # Calculate control metrics
            valid_errors = sim.concentration_errors[sim.concentration_errors != 0]
            if len(valid_errors) > 0:
                control_rmse = float(np.sqrt(np.mean(valid_errors**2)))
                control_mae = float(np.mean(np.abs(valid_errors)))
            else:
                control_rmse = 100.0
                control_mae = 100.0

            substrate_utilization = float(sim.substrate_utilizations[-1])

            # Get Q-learning reward from controller
            q_reward = float(np.sum(sim.q_rewards))

            # Calculate biofilm performance
            final_biofilm = getattr(sim, 'biofilm_thickness', np.array([[0.5]]))
            if hasattr(final_biofilm, 'shape') and len(final_biofilm.shape) > 1:
                avg_final_biofilm = np.mean(final_biofilm[-1, :])
                biofilm_stability = np.std(final_biofilm[-100:, :])  # Last 100 measurements
            else:
                avg_final_biofilm = 0.5
                biofilm_stability = 0.0

            biofilm_error = abs(avg_final_biofilm - self.target_biofilm)

            # Extended objective calculation (same weights as optimization)
            control_objective = control_rmse * 0.4
            biofilm_objective = biofilm_error * 20.0 * 0.3
            energy_objective = (15.0 - energy_total) * 0.2  # Higher energy target for 600h
            stability_objective = abs(q_reward / 1e6) * 0.1

            extended_objective = control_objective + biofilm_objective + energy_objective + stability_objective

            # Prepare result
            validation_result = {
                'config': config,
                'status': 'completed',
                'extended_simulation_time': elapsed_time,
                'extended_objective': extended_objective,
                'extended_metrics': {
                    'energy_total': energy_total,
                    'final_power': final_power,
                    'control_rmse': control_rmse,
                    'control_mae': control_mae,
                    'biofilm_error': biofilm_error,
                    'avg_final_biofilm': avg_final_biofilm,
                    'biofilm_stability': biofilm_stability,
                    'substrate_utilization': substrate_utilization,
                    'q_reward': q_reward,
                    'simulation_hours': 600
                },
                'improvement_vs_short': {
                    'energy_ratio': energy_total / config['original_metrics'].get('energy_total', 1),
                    'rmse_ratio': control_rmse / config['original_metrics'].get('control_rmse', 100),
                    'biofilm_ratio': biofilm_error / config['original_metrics'].get('biofilm_error', 1)
                }
            }

            self.logger.info(f"Extended validation {config_num} completed: Obj={extended_objective:.6f}, "
                           f"Energy={energy_total:.3f}Wh, RMSE={control_rmse:.3f}, "
                           f"Biofilm={avg_final_biofilm:.3f}, Time={elapsed_time:.1f}s")

            return validation_result

        except Exception as e:
            self.logger.error(f"Extended validation {config_num} failed: {str(e)}")
            raise

    def _apply_parameters_to_simulation(self, sim, params: dict):
        """Apply optimized parameters to simulation (reuse optimization logic)"""
        # This reuses the same parameter application logic from create_modified_simulation
        # but without recreating the simulation object

        # Override reward function with optimized parameters
        def optimized_reward_function(power, biofilm_deviation, substrate_utilization,
                                    outlet_conc, prev_power, prev_biofilm_dev,
                                    prev_substrate_util, prev_outlet_conc,
                                    biofilm_thickness_history=None):

            # Calculate changes
            power_change = power - prev_power
            substrate_change = substrate_utilization - prev_substrate_util
            error_improvement = abs(prev_outlet_conc - self.target_outlet_conc) - abs(outlet_conc - self.target_outlet_conc)

            # Apply all optimized reward components (same as in optimization)
            # [Full reward function implementation - same as in create_modified_simulation]

            # 1. POWER COMPONENT
            if power_change > 0:
                power_reward = power_change * params['power_increase_multiplier']
            else:
                power_reward = power_change * params['power_decrease_multiplier']

            power_base = 10.0 if power > 0.005 else -5.0

            # 2. SUBSTRATE COMPONENT
            if substrate_change > 0:
                substrate_reward = substrate_change * params['substrate_increase_multiplier']
            else:
                substrate_reward = substrate_change * params['substrate_decrease_multiplier']

            substrate_base = 5.0 if substrate_utilization > 10.0 else -2.0

            # 3. BIOFILM COMPONENT
            optimal_thickness = self.target_biofilm
            deviation_threshold = 0.05 * optimal_thickness

            if biofilm_deviation <= deviation_threshold:
                biofilm_reward = params['biofilm_base_reward'] - (biofilm_deviation / deviation_threshold) * 15.0

                if biofilm_thickness_history is not None and len(biofilm_thickness_history) >= 3:
                    recent_thickness = biofilm_thickness_history[-3:]
                    if len(recent_thickness) >= 2:
                        growth_rate = abs(recent_thickness[-1] - recent_thickness[-2])
                        if growth_rate < 0.01:
                            biofilm_reward += params['biofilm_steady_bonus']
            else:
                excess_deviation = biofilm_deviation - deviation_threshold
                biofilm_reward = -params['biofilm_penalty_multiplier'] * (excess_deviation / deviation_threshold)

            # 4. CONCENTRATION CONTROL
            outlet_error = abs(outlet_conc - self.target_outlet_conc)

            if outlet_error <= 0.5:
                concentration_reward = params['conc_precise_reward'] - (outlet_error * 10.0)
            elif outlet_error <= 2.0:
                concentration_reward = params['conc_acceptable_reward'] - (outlet_error * 2.5)
            else:
                concentration_reward = params['conc_poor_penalty'] - (outlet_error * 5.0)

            concentration_base = 3.0 if error_improvement > 0 else -1.0

            # 5. STABILITY BONUS
            stability_bonus = 0
            if (abs(power_change) < 0.001 and abs(substrate_change) < 0.5 and
                outlet_error < 1.0 and biofilm_deviation <= deviation_threshold):
                stability_bonus = 30.0

            # 6. FLOW PENALTY
            flow_penalty = 0
            current_flow_rate = getattr(sim.unified_controller, 'current_flow_rate', 10.0)
            if biofilm_thickness_history is not None and len(biofilm_thickness_history) > 0:
                avg_biofilm = np.mean(biofilm_thickness_history[-5:])
                if avg_biofilm < optimal_thickness * params['biofilm_threshold_ratio']:
                    if current_flow_rate > params['flow_penalty_threshold']:
                        flow_penalty = -params['flow_penalty_multiplier'] * (current_flow_rate - params['flow_penalty_threshold']) / 10.0

            # 7. COMBINED PENALTY
            combined_penalty = 0
            if (power_change < 0 and substrate_change < 0 and
                error_improvement < 0 and biofilm_deviation > deviation_threshold):
                combined_penalty = -200.0

            # Total reward
            total_reward = (power_reward + power_base +
                           substrate_reward + substrate_base +
                           biofilm_reward +
                           concentration_reward + concentration_base +
                           stability_bonus + flow_penalty + combined_penalty)

            return total_reward

        # Apply reward function
        sim.unified_controller.calculate_unified_reward = optimized_reward_function

        # Apply Q-learning parameters
        sim.unified_controller.learning_rate = params['learning_rate']
        sim.unified_controller.discount_factor = params['discount_factor']
        sim.unified_controller.epsilon = params['initial_epsilon']
        sim.unified_controller.epsilon_decay = params['epsilon_decay']
        sim.unified_controller.epsilon_min = params['epsilon_min']

        # Apply action space
        self.update_action_space(sim, params)

    def _save_validation_results(self, validation_results: list[dict]):
        """Save extended validation results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save detailed validation results
        validation_file = self.results_dir / f"extended_validation_{timestamp}.json"

        # Prepare serializable data
        serializable_results = []
        for result in validation_results:
            serializable_result = result.copy()
            # Convert any numpy types to Python types
            if 'extended_metrics' in serializable_result:
                metrics = serializable_result['extended_metrics']
                for key, value in metrics.items():
                    if hasattr(value, 'item'):  # numpy scalar
                        metrics[key] = value.item()
            serializable_results.append(serializable_result)

        with open(validation_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        self.logger.info(f"Extended validation results saved to: {validation_file}")

    def _print_validation_summary(self, validation_results: list[dict]):
        """Print comprehensive validation summary"""
        print("\n" + "="*80)
        print("EXTENDED VALIDATION RESULTS (600 HOURS)")
        print("="*80)

        successful_results = [r for r in validation_results if r.get('status') != 'failed']
        failed_count = len(validation_results) - len(successful_results)

        print(f"Successful validations: {len(successful_results)}/14")
        if failed_count > 0:
            print(f"Failed validations: {failed_count}")

        if successful_results:
            # Sort by extended objective (lower is better)
            successful_results.sort(key=lambda x: x['extended_objective'])

            print("\nBEST VALIDATED CONFIGURATION:")
            best = successful_results[0]
            config = best['config']
            metrics = best['extended_metrics']

            print(f"  Original Trial: {config['trial_number']}")
            print(f"  Extended Objective: {best['extended_objective']:.6f}")
            print(f"  Energy: {metrics['energy_total']:.3f} Wh")
            print(f"  Final Power: {metrics['final_power']:.4f} W")
            print(f"  Control RMSE: {metrics['control_rmse']:.3f} mmol/L")
            print(f"  Control MAE: {metrics['control_mae']:.3f} mmol/L")
            print(f"  Final Biofilm: {metrics['avg_final_biofilm']:.3f} (target: {self.target_biofilm})")
            print(f"  Biofilm Stability: {metrics['biofilm_stability']:.4f}")
            print(f"  Substrate Utilization: {metrics['substrate_utilization']:.3f}%")
            print(f"  Simulation Time: {best['extended_simulation_time']:.1f}s")

            print("\nTOP 3 CONFIGURATIONS SUMMARY:")
            for i, result in enumerate(successful_results[:3]):
                config = result['config']
                metrics = result['extended_metrics']
                print(f"  {i+1}. Trial {config['trial_number']}: "
                      f"Obj={result['extended_objective']:.4f}, "
                      f"Energy={metrics['energy_total']:.1f}Wh, "
                      f"RMSE={metrics['control_rmse']:.3f}, "
                      f"Biofilm={metrics['avg_final_biofilm']:.3f}")

        print("="*80)


def main():
    """Main optimization execution"""
    print("MFC Unified Q-Learning Hyperparameter Optimization with Optuna")
    print("==============================================================")

    # Configuration
    N_TRIALS = 140  # Increased for 14-thread optimization
    N_JOBS = min(14, mp.cpu_count())  # Use up to 14 threads
    STUDY_NAME = "mfc_qlearning_optimization_v1"

    print("Configuration:")
    print(f"  Trials: {N_TRIALS}")
    print(f"  Parallel jobs: {N_JOBS}")
    print(f"  Available CPU cores: {mp.cpu_count()}")
    print(f"  Study name: {STUDY_NAME}")
    print("  Simulation duration: 120h per trial")
    print("  Timeout per trial: 300s (5 min)")
    print(f"  Expected total time: ~{(N_TRIALS * 300 / 60 / N_JOBS):.1f} minutes")
    print("  Extended validation: Top 14 configs × 600h each")

    # Create optimizer
    optimizer = MFCOptunaOptimizer(
        n_trials=N_TRIALS,
        n_jobs=N_JOBS,
        study_name=STUDY_NAME,
        timeout=300  # 5 minutes per trial max (optimized for 14-thread)
    )

    # Run optimization
    try:
        study = optimizer.run_optimization()

        # Print results
        optimizer.print_optimization_summary(study)

        print(f"\nOptimization completed! Results saved in: {optimizer.results_dir}")

        # Run extended validation on best configurations
        print("\n" + "="*80)
        print("RUNNING EXTENDED VALIDATION ON TOP 14 CONFIGURATIONS")
        print("="*80)

        best_configs = optimizer.get_top_configurations(study, n_configs=14)
        optimizer.run_extended_validation(best_configs)  # Results printed internally

        print("Extended validation completed!")
        print("Use the best validated parameters to create an optimized MFC controller.")

    except Exception as e:
        print(f"Optimization failed: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
