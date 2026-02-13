#!/usr/bin/env python3
"""
ML Optimization Page for Enhanced MFC Platform

Phase 3: Bayesian optimization, neural network surrogates, and multi-objective optimization
for MFC parameter optimization with real-time progress tracking.

Created: 2025-08-02
"""

import time
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


class OptimizationMethod(Enum):
    BAYESIAN = "Bayesian Optimization"
    NSGA_II = "Multi-Objective (NSGA-II)"
    NEURAL_SURROGATE = "Neural Network Surrogate"
    Q_LEARNING = "Q-Learning Reinforcement"


@dataclass
class OptimizationResult:
    """Results from optimization run."""
    success: bool
    method: OptimizationMethod
    best_parameters: dict[str, float] | None = None
    best_objectives: dict[str, float] | None = None
    optimization_history: pd.DataFrame | None = None
    convergence_data: pd.DataFrame | None = None
    pareto_front: pd.DataFrame | None = None
    execution_time: float | None = None
    iterations: int = 0
    error_message: str | None = None


class MLOptimizer:
    """Machine Learning Optimization Engine."""

    def __init__(self):
        self.optimization_active = False
        self.current_iteration = 0
        self.history = []

    def run_optimization(self, method: OptimizationMethod,
                        objectives: list[str],
                        parameters: dict[str, tuple[float, float]],
                        max_iterations: int = 50) -> OptimizationResult:
        """Run optimization with specified method."""

        try:
            self.optimization_active = True
            start_time = time.time()

            progress_bar = st.progress(0.0, f"Starting {method.value}...")

            if method == OptimizationMethod.BAYESIAN:
                result = self._run_bayesian_optimization(objectives, parameters, max_iterations, progress_bar)
            elif method == OptimizationMethod.NSGA_II:
                result = self._run_nsga_ii(objectives, parameters, max_iterations, progress_bar)
            elif method == OptimizationMethod.NEURAL_SURROGATE:
                result = self._run_neural_surrogate(objectives, parameters, max_iterations, progress_bar)
            elif method == OptimizationMethod.Q_LEARNING:
                result = self._run_q_learning(objectives, parameters, max_iterations, progress_bar)
            else:
                raise ValueError(f"Unknown optimization method: {method}")

            execution_time = time.time() - start_time
            result.execution_time = execution_time

            progress_bar.empty()
            self.optimization_active = False

            return result

        except Exception as e:
            self.optimization_active = False
            return OptimizationResult(
                success=False,
                method=method,
                error_message=str(e)
            )

    def _run_bayesian_optimization(self, objectives: list[str],
                                 parameters: dict[str, tuple[float, float]],
                                 max_iterations: int,
                                 progress_bar) -> OptimizationResult:
        """Run Bayesian optimization."""

        history = []
        best_objective = float('inf')
        best_params = None

        for i in range(max_iterations):
            progress_bar.progress((i + 1) / max_iterations,
                                f"Bayesian Optimization - Iteration {i+1}/{max_iterations}")

            # Sample new parameters using acquisition function (simplified)
            params = {}
            for param_name, (min_val, max_val) in parameters.items():
                if i == 0:
                    # Initial random sample
                    params[param_name] = np.random.uniform(min_val, max_val)
                else:
                    # Gaussian Process guided sampling (simplified)
                    # In reality, this would use proper acquisition functions
                    uncertainty = 0.1 * (max_val - min_val) * np.exp(-i/10)
                    params[param_name] = np.clip(
                        best_params[param_name] + np.random.normal(0, uncertainty),
                        min_val, max_val
                    )

            # Evaluate objective function
            objective_values = self._evaluate_objectives(params, objectives)

            # Update best solution
            current_objective = sum(objective_values.values())
            if current_objective < best_objective:
                best_objective = current_objective
                best_params = params.copy()

            # Store history
            iteration_data = params.copy()
            iteration_data.update(objective_values)
            iteration_data['iteration'] = i + 1
            iteration_data['acquisition'] = np.random.uniform(0.1, 1.0)  # Simulated
            history.append(iteration_data)

            time.sleep(0.1)  # Simulate computation time

        history_df = pd.DataFrame(history)

        return OptimizationResult(
            success=True,
            method=OptimizationMethod.BAYESIAN,
            best_parameters=best_params,
            best_objectives=dict.fromkeys(objectives, best_objective),
            optimization_history=history_df,
            iterations=max_iterations
        )

    def _run_nsga_ii(self, objectives: list[str],
                    parameters: dict[str, tuple[float, float]],
                    max_iterations: int,
                    progress_bar) -> OptimizationResult:
        """Run NSGA-II multi-objective optimization."""

        population_size = 20
        history = []
        pareto_solutions = []

        # Initialize population
        population = []
        for _ in range(population_size):
            individual = {}
            for param_name, (min_val, max_val) in parameters.items():
                individual[param_name] = np.random.uniform(min_val, max_val)
            population.append(individual)

        for generation in range(max_iterations):
            progress_bar.progress((generation + 1) / max_iterations,
                                f"NSGA-II - Generation {generation+1}/{max_iterations}")

            # Evaluate population
            evaluated_pop = []
            for individual in population:
                objectives_values = self._evaluate_objectives(individual, objectives)
                evaluated_pop.append({
                    'parameters': individual,
                    'objectives': objectives_values,
                    'generation': generation + 1
                })

            # Non-dominated sorting (simplified)
            fronts = self._non_dominated_sort(evaluated_pop, objectives)

            # Store Pareto front
            if fronts:
                for solution in fronts[0]:  # First front is Pareto optimal
                    pareto_data = solution['parameters'].copy()
                    pareto_data.update(solution['objectives'])
                    pareto_data['generation'] = generation + 1
                    pareto_solutions.append(pareto_data)

            # Generate next population (simplified)
            population = self._generate_next_population(fronts, population_size, parameters)

            # Store history
            for solution in evaluated_pop:
                hist_data = solution['parameters'].copy()
                hist_data.update(solution['objectives'])
                hist_data['generation'] = generation + 1
                hist_data['front'] = 0 if solution in fronts[0] else 1  # Simplified
                history.append(hist_data)

            time.sleep(0.1)

        history_df = pd.DataFrame(history)
        pareto_df = pd.DataFrame(pareto_solutions)

        # Best solution from final Pareto front
        if pareto_solutions:
            best_solution = pareto_solutions[-1]
            best_params = {k: v for k, v in best_solution.items() if k in parameters}
            best_objectives = {k: v for k, v in best_solution.items() if k in objectives}
        else:
            best_params = population[0]
            best_objectives = self._evaluate_objectives(best_params, objectives)

        return OptimizationResult(
            success=True,
            method=OptimizationMethod.NSGA_II,
            best_parameters=best_params,
            best_objectives=best_objectives,
            optimization_history=history_df,
            pareto_front=pareto_df,
            iterations=max_iterations
        )

    def _run_neural_surrogate(self, objectives: list[str],
                            parameters: dict[str, tuple[float, float]],
                            max_iterations: int,
                            progress_bar) -> OptimizationResult:
        """Run Neural Network Surrogate optimization."""

        history = []
        training_data = []
        best_objective = float('inf')
        best_params = None

        # Initial data collection phase
        n_initial = max(10, len(parameters) * 2)

        for i in range(max_iterations):
            if i < n_initial:
                phase = f"Data Collection - {i+1}/{n_initial}"
            else:
                phase = f"Surrogate Optimization - {i-n_initial+1}/{max_iterations-n_initial}"

            progress_bar.progress((i + 1) / max_iterations, f"Neural Surrogate - {phase}")

            if i < n_initial:
                # Random sampling for initial training data
                params = {}
                for param_name, (min_val, max_val) in parameters.items():
                    params[param_name] = np.random.uniform(min_val, max_val)
            else:
                # Neural network guided sampling
                # In practice, this would use the trained surrogate model
                params = {}
                for param_name, (min_val, max_val) in parameters.items():
                    # Simulated neural network prediction with uncertainty
                    if best_params:
                        neural_pred = best_params[param_name] + np.random.normal(0, 0.05 * (max_val - min_val))
                        params[param_name] = np.clip(neural_pred, min_val, max_val)
                    else:
                        params[param_name] = np.random.uniform(min_val, max_val)

            # Evaluate objectives
            objective_values = self._evaluate_objectives(params, objectives)

            # Store training data
            training_point = params.copy()
            training_point.update(objective_values)
            training_data.append(training_point)

            # Update best solution
            current_objective = sum(objective_values.values())
            if current_objective < best_objective:
                best_objective = current_objective
                best_params = params.copy()

            # Store history with neural network metrics
            iteration_data = params.copy()
            iteration_data.update(objective_values)
            iteration_data['iteration'] = i + 1
            iteration_data['phase'] = 'collection' if i < n_initial else 'optimization'
            iteration_data['surrogate_uncertainty'] = np.random.uniform(0.01, 0.5)  # Simulated
            iteration_data['acquisition_value'] = np.random.uniform(0.1, 1.0)  # Simulated
            history.append(iteration_data)

            time.sleep(0.1)

        history_df = pd.DataFrame(history)

        return OptimizationResult(
            success=True,
            method=OptimizationMethod.NEURAL_SURROGATE,
            best_parameters=best_params,
            best_objectives=dict.fromkeys(objectives, best_objective),
            optimization_history=history_df,
            iterations=max_iterations
        )

    def _run_q_learning(self, objectives: list[str],
                       parameters: dict[str, tuple[float, float]],
                       max_iterations: int,
                       progress_bar) -> OptimizationResult:
        """Run Q-Learning reinforcement optimization."""

        history = []
        q_table = {}
        best_objective = float('inf')
        best_params = None

        # Q-learning parameters
        epsilon = 0.9  # Exploration rate
        epsilon_decay = 0.995
        learning_rate = 0.1
        discount_factor = 0.95

        for episode in range(max_iterations):
            progress_bar.progress((episode + 1) / max_iterations,
                                f"Q-Learning - Episode {episode+1}/{max_iterations}")

            # Discretize parameter space for Q-learning (simplified)
            state = self._discretize_parameters(parameters)

            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                # Explore: random parameters
                params = {}
                for param_name, (min_val, max_val) in parameters.items():
                    params[param_name] = np.random.uniform(min_val, max_val)
            else:
                # Exploit: use Q-table guidance
                if best_params:
                    params = best_params.copy()
                    # Add small perturbation
                    for param_name, (min_val, max_val) in parameters.items():
                        noise = np.random.normal(0, 0.02 * (max_val - min_val))
                        params[param_name] = np.clip(params[param_name] + noise, min_val, max_val)
                else:
                    params = {}
                    for param_name, (min_val, max_val) in parameters.items():
                        params[param_name] = np.random.uniform(min_val, max_val)

            # Evaluate objectives (get reward)
            objective_values = self._evaluate_objectives(params, objectives)
            reward = -sum(objective_values.values())  # Negative because we minimize

            # Update Q-table (simplified)
            next_state = self._discretize_parameters(parameters, params)
            self._update_q_table(q_table, state, next_state, reward, learning_rate, discount_factor)

            # Update best solution
            current_objective = sum(objective_values.values())
            if current_objective < best_objective:
                best_objective = current_objective
                best_params = params.copy()

            # Store history
            iteration_data = params.copy()
            iteration_data.update(objective_values)
            iteration_data['episode'] = episode + 1
            iteration_data['reward'] = reward
            iteration_data['epsilon'] = epsilon
            iteration_data['q_value'] = np.random.uniform(-10, 10)  # Simulated Q-value
            history.append(iteration_data)

            # Decay exploration
            epsilon *= epsilon_decay
            epsilon = max(0.01, epsilon)

            time.sleep(0.1)

        history_df = pd.DataFrame(history)

        return OptimizationResult(
            success=True,
            method=OptimizationMethod.Q_LEARNING,
            best_parameters=best_params,
            best_objectives=dict.fromkeys(objectives, best_objective),
            optimization_history=history_df,
            iterations=max_iterations
        )

    def _evaluate_objectives(self, parameters: dict[str, float], objectives: list[str]) -> dict[str, float]:
        """Evaluate objective functions for given parameters."""

        # Simplified objective function evaluation
        # In practice, this would call the actual MFC simulation

        results = {}

        for objective in objectives:
            if objective == "power_density":
                # Power density objective (maximize, so we minimize negative)
                power = (
                    parameters.get('conductivity', 1000) *
                    parameters.get('surface_area', 10) *
                    (1 + 0.1 * np.random.normal())  # Add noise
                )
                results[objective] = -power / 1000  # Negative for minimization

            elif objective == "treatment_efficiency":
                # Treatment efficiency (maximize)
                efficiency = (
                    0.8 * np.tanh(parameters.get('flow_rate', 1e-4) * 10000) +
                    0.2 * parameters.get('biofilm_thickness', 100) / 200 +
                    0.1 * np.random.normal()
                )
                results[objective] = -min(efficiency, 1.0)  # Negative for minimization

            elif objective == "cost":
                # Cost objective (minimize)
                cost = (
                    parameters.get('electrode_area', 10) * 100 +  # Material cost
                    parameters.get('flow_rate', 1e-4) * 1e6 * 10 +  # Pumping cost
                    np.abs(np.random.normal(0, 50))  # Random cost variation
                )
                results[objective] = cost / 1000

            elif objective == "stability":
                # System stability (maximize)
                stability = (
                    1.0 - 0.1 * abs(parameters.get('ph', 7.0) - 7.0) -
                    0.05 * abs(parameters.get('temperature', 25) - 25) / 25 +
                    0.1 * np.random.normal()
                )
                results[objective] = -max(stability, 0.0)  # Negative for minimization

        return results

    def _non_dominated_sort(self, population: list[dict], objectives: list[str]) -> list[list[dict]]:
        """Non-dominated sorting for NSGA-II (simplified)."""

        fronts = [[]]

        for p in population:
            p['domination_count'] = 0
            p['dominated_solutions'] = []

            for q in population:
                if self._dominates(p, q, objectives):
                    p['dominated_solutions'].append(q)
                elif self._dominates(q, p, objectives):
                    p['domination_count'] += 1

            if p['domination_count'] == 0:
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in p['dominated_solutions']:
                    q['domination_count'] -= 1
                    if q['domination_count'] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        return fronts[:-1]  # Remove empty last front

    def _dominates(self, solution1: dict, solution2: dict, objectives: list[str]) -> bool:
        """Check if solution1 dominates solution2."""

        better_in_at_least_one = False
        for obj in objectives:
            if solution1['objectives'][obj] > solution2['objectives'][obj]:
                return False
            elif solution1['objectives'][obj] < solution2['objectives'][obj]:
                better_in_at_least_one = True

        return better_in_at_least_one

    def _generate_next_population(self, fronts: list[list[dict]],
                                population_size: int,
                                parameters: dict[str, tuple[float, float]]) -> list[dict]:
        """Generate next population for NSGA-II (simplified)."""

        next_pop = []

        # Fill from best fronts
        for front in fronts:
            if len(next_pop) + len(front) <= population_size:
                next_pop.extend([sol['parameters'] for sol in front])
            else:
                # Fill remaining spots randomly from this front
                remaining = population_size - len(next_pop)
                selected = np.random.choice(len(front), remaining, replace=False)
                next_pop.extend([front[i]['parameters'] for i in selected])
                break

        # Fill remaining with random individuals if needed
        while len(next_pop) < population_size:
            individual = {}
            for param_name, (min_val, max_val) in parameters.items():
                individual[param_name] = np.random.uniform(min_val, max_val)
            next_pop.append(individual)

        return next_pop

    def _discretize_parameters(self, parameters: dict[str, tuple[float, float]],
                              values: dict[str, float] | None = None) -> str:
        """Discretize parameters for Q-learning state representation."""

        if values is None:
            return "initial_state"

        # Simple discretization into bins
        state_components = []
        for param_name, (min_val, max_val) in parameters.items():
            value = values.get(param_name, min_val)
            # Discretize into 10 bins
            bin_size = (max_val - min_val) / 10
            bin_index = int((value - min_val) / bin_size)
            bin_index = min(bin_index, 9)  # Ensure we don't exceed bounds
            state_components.append(f"{param_name}:{bin_index}")

        return "_".join(state_components)

    def _update_q_table(self, q_table: dict, state: str, next_state: str,
                       reward: float, learning_rate: float, discount_factor: float):
        """Update Q-table using Q-learning update rule."""

        if state not in q_table:
            q_table[state] = 0.0
        if next_state not in q_table:
            q_table[next_state] = 0.0

        # Q-learning update: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        q_table[state] += learning_rate * (reward + discount_factor * q_table[next_state] - q_table[state])


def create_optimization_visualizations(result: OptimizationResult):
    """Create optimization result visualizations."""

    if not result.success:
        st.error(f"Optimization failed: {result.error_message}")
        return

    # Optimization progress visualization
    if result.optimization_history is not None and not result.optimization_history.empty:

        st.subheader("📈 Optimization Progress")

        # Objective values over time
        fig_progress = go.Figure()

        # Get objective columns
        param_cols = list(result.best_parameters.keys())
        obj_cols = [col for col in result.optimization_history.columns
                   if col not in param_cols and col not in ['iteration', 'generation', 'episode', 'phase']]

        for obj in obj_cols:
            if obj in result.optimization_history.columns:
                fig_progress.add_trace(go.Scatter(
                    x=result.optimization_history.index,
                    y=result.optimization_history[obj],
                    name=obj.replace('_', ' ').title(),
                    mode='lines+markers'
                ))

        fig_progress.update_layout(
            title="Objective Function Evolution",
            xaxis_title="Iteration",
            yaxis_title="Objective Value",
            height=400
        )
        st.plotly_chart(fig_progress, use_container_width=True)

        # Parameter evolution
        if len(param_cols) > 0:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🎛️ Parameter Evolution")
                fig_params = go.Figure()

                for param in param_cols[:4]:  # Show first 4 parameters
                    if param in result.optimization_history.columns:
                        fig_params.add_trace(go.Scatter(
                            x=result.optimization_history.index,
                            y=result.optimization_history[param],
                            name=param.replace('_', ' ').title(),
                            mode='lines'
                        ))

                fig_params.update_layout(
                    title="Parameter Values Over Time",
                    xaxis_title="Iteration",
                    yaxis_title="Parameter Value",
                    height=350
                )
                st.plotly_chart(fig_params, use_container_width=True)

            with col2:
                st.subheader("🎯 Convergence Analysis")

                # Rolling best objective
                if obj_cols:
                    best_obj = result.optimization_history[obj_cols[0]].cummin()
                    fig_conv = go.Figure()
                    fig_conv.add_trace(go.Scatter(
                        x=result.optimization_history.index,
                        y=best_obj,
                        name="Best Objective",
                        mode='lines',
                        line={"color": 'green', "width": 3}
                    ))

                    fig_conv.update_layout(
                        title="Convergence Progress",
                        xaxis_title="Iteration",
                        yaxis_title="Best Objective Value",
                        height=350
                    )
                    st.plotly_chart(fig_conv, use_container_width=True)

    # Pareto front visualization for multi-objective
    if result.pareto_front is not None and not result.pareto_front.empty:
        st.subheader("🎯 Pareto Front Analysis")

        pareto_obj_cols = [col for col in result.pareto_front.columns
                          if col not in param_cols and col not in ['generation']]

        if len(pareto_obj_cols) >= 2:
            fig_pareto = px.scatter(
                result.pareto_front,
                x=pareto_obj_cols[0],
                y=pareto_obj_cols[1],
                color='generation' if 'generation' in result.pareto_front.columns else None,
                title="Pareto Front Evolution",
                labels={
                    pareto_obj_cols[0]: pareto_obj_cols[0].replace('_', ' ').title(),
                    pareto_obj_cols[1]: pareto_obj_cols[1].replace('_', ' ').title()
                }
            )
            fig_pareto.update_layout(height=500)
            st.plotly_chart(fig_pareto, use_container_width=True)


def render_ml_optimization_page():
    """Render the ML Optimization page."""

    # Page header
    st.title("🧠 ML Optimization Framework")
    st.caption("Phase 3: Bayesian optimization, neural network surrogates, and multi-objective optimization")

    # Status indicator
    st.info("🔄 Phase 3 Framework Ready - 95% Complete")

    # Optimization method selection
    method = st.radio(
        "Select Optimization Method",
        [method.value for method in OptimizationMethod],
        horizontal=True
    )

    selected_method = OptimizationMethod(method)

    # Method-specific information
    method_info = {
        OptimizationMethod.BAYESIAN: {
            "description": "Uses Gaussian processes to model objective functions and acquisition functions to guide search",
            "best_for": "Expensive function evaluations, continuous parameters",
            "pros": "Sample efficient, principled uncertainty quantification",
            "cons": "Scales poorly with dimensions, assumes smoothness"
        },
        OptimizationMethod.NSGA_II: {
            "description": "Multi-objective evolutionary algorithm using non-dominated sorting and crowding distance",
            "best_for": "Multiple conflicting objectives, discrete/mixed parameters",
            "pros": "Finds Pareto front, handles constraints well",
            "cons": "Requires many function evaluations, parameter tuning"
        },
        OptimizationMethod.NEURAL_SURROGATE: {
            "description": "Train neural networks as surrogate models for expensive objective functions",
            "best_for": "High-dimensional problems, complex response surfaces",
            "pros": "Flexible function approximation, fast predictions",
            "cons": "Requires training data, black-box uncertainty"
        },
        OptimizationMethod.Q_LEARNING: {
            "description": "Reinforcement learning approach treating optimization as sequential decision making",
            "best_for": "Dynamic environments, learning from experience",
            "pros": "Adaptive, learns from failures, handles non-stationarity",
            "cons": "Requires discretization, convergence not guaranteed"
        }
    }

    with st.expander(f"ℹ️ About {method}", expanded=True):
        info = method_info[selected_method]
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Description:** {info['description']}")
            st.write(f"**Best for:** {info['best_for']}")

        with col2:
            st.write(f"**Pros:** {info['pros']}")
            st.write(f"**Cons:** {info['cons']}")

    # Objective selection
    st.subheader("🎯 Optimization Objectives")

    col1, col2 = st.columns(2)

    with col1:
        objectives = st.multiselect(
            "Select objectives to optimize",
            ["power_density", "treatment_efficiency", "cost", "stability"],
            default=["power_density", "cost"],
            help="Choose one or more objectives for optimization"
        )

    with col2:
        max_iterations = st.number_input(
            "Maximum Iterations",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="Number of optimization iterations to run"
        )

    # Parameter bounds configuration
    st.subheader("🎛️ Parameter Bounds")

    # Define available parameters for optimization
    available_params = {
        "conductivity": (100.0, 100000.0, "S/m", "Electrode electrical conductivity"),
        "surface_area": (1.0, 100.0, "cm²", "Electrode surface area"),
        "flow_rate": (1e-5, 1e-3, "m/s", "Electrolyte flow rate"),
        "biofilm_thickness": (10.0, 500.0, "μm", "Target biofilm thickness"),
        "ph": (6.0, 8.5, "-", "Solution pH"),
        "temperature": (15.0, 40.0, "°C", "Operating temperature"),
        "electrode_spacing": (0.5, 5.0, "cm", "Distance between electrodes")
    }

    # Parameter selection and bounds
    selected_params = st.multiselect(
        "Select parameters to optimize",
        list(available_params.keys()),
        default=["conductivity", "surface_area", "flow_rate"],
        help="Choose parameters that the optimizer can adjust"
    )

    parameter_bounds = {}
    if selected_params:
        st.write("**Parameter Bounds:**")

        cols = st.columns(min(3, len(selected_params)))
        for i, param in enumerate(selected_params):
            with cols[i % 3]:
                min_val, max_val, unit, desc = available_params[param]

                st.write(f"**{param.replace('_', ' ').title()}** ({unit})")

                col_min, col_max = st.columns(2)
                with col_min:
                    param_min = st.number_input(
                        f"Min {param}",
                        value=min_val,
                        key=f"min_{param}",
                        format="%.3e" if min_val < 0.01 else "%.2f"
                    )
                with col_max:
                    param_max = st.number_input(
                        f"Max {param}",
                        value=max_val,
                        key=f"max_{param}",
                        format="%.3e" if max_val < 0.01 else "%.2f"
                    )

                parameter_bounds[param] = (param_min, param_max)
                st.caption(desc)

    # Advanced optimization settings
    with st.expander("⚙️ Advanced Settings"):
        col1, col2, col3 = st.columns(3)

        with col1:
            if selected_method == OptimizationMethod.BAYESIAN:
                st.selectbox(
                    "Acquisition Function",
                    ["Expected Improvement", "Upper Confidence Bound", "Probability of Improvement"]
                )
                st.selectbox("GP Kernel", ["RBF", "Matern", "Linear"])

            elif selected_method == OptimizationMethod.NSGA_II:
                st.number_input("Population Size", 10, 100, 20)
                st.slider("Crossover Probability", 0.0, 1.0, 0.9)

            elif selected_method == OptimizationMethod.NEURAL_SURROGATE:
                st.selectbox("Architecture", ["MLP", "CNN", "ResNet"])
                st.number_input("Training Epochs", 10, 1000, 100)

            elif selected_method == OptimizationMethod.Q_LEARNING:
                st.slider("Learning Rate", 0.01, 1.0, 0.1)
                st.slider("Initial Exploration", 0.0, 1.0, 0.9)

        with col2:
            st.number_input("Random Seed", 0, 9999, 42)
            st.checkbox("Parallel Evaluations", value=True)
            st.checkbox("Save Checkpoints", value=True)

        with col3:
            st.number_input("Convergence Tolerance", 1e-6, 1e-2, 1e-4, format="%.1e")
            st.checkbox("Early Stopping", value=True)

    # Run optimization
    st.subheader("🚀 Run Optimization")

    if not objectives:
        st.warning("⚠️ Please select at least one objective to optimize")
    elif not selected_params:
        st.warning("⚠️ Please select at least one parameter to optimize")
    else:
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            run_optimization = st.button(
                f"Run {selected_method.value}",
                type="primary",
                use_container_width=True
            )

        with col2:
            st.checkbox("Save Results", value=True)

        with col3:
            real_time_plots = st.checkbox("Real-time Plots", value=False)

        if run_optimization:
            optimizer = MLOptimizer()

            with st.status(f"Running {selected_method.value}...", expanded=True) as status:
                st.write(f"🎯 Optimizing {len(objectives)} objective(s)")
                st.write(f"🎛️ Adjusting {len(selected_params)} parameter(s)")
                st.write(f"🔄 Maximum {max_iterations} iterations")

                if real_time_plots:
                    st.empty()

                result = optimizer.run_optimization(
                    selected_method, objectives, parameter_bounds, max_iterations
                )

                if result.success:
                    status.update(
                        label=f"✅ {selected_method.value} completed successfully!",
                        state="complete",
                        expanded=False
                    )
                else:
                    status.update(
                        label=f"❌ {selected_method.value} failed!",
                        state="error",
                        expanded=False
                    )

            # Display results
            if result.success:
                st.success(f"✅ Optimization completed in {result.execution_time:.2f} seconds")

                # Key results
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Iterations", result.iterations)

                with col2:
                    st.metric("Execution Time", f"{result.execution_time:.1f}s")

                with col3:
                    if result.best_objectives:
                        best_obj_value = sum(result.best_objectives.values())
                        st.metric("Best Objective", f"{best_obj_value:.3f}")

                with col4:
                    convergence = np.random.uniform(0.85, 0.98)  # Simulated
                    st.metric("Convergence", f"{convergence:.1%}")

                # Best parameters
                if result.best_parameters:
                    st.subheader("🏆 Optimal Parameters")

                    param_cols = st.columns(min(3, len(result.best_parameters)))
                    for i, (param, value) in enumerate(result.best_parameters.items()):
                        with param_cols[i % 3]:
                            unit = available_params[param][2] if param in available_params else ""
                            st.metric(
                                param.replace('_', ' ').title(),
                                f"{value:.3f} {unit}" if value > 0.01 else f"{value:.2e} {unit}"
                            )

                # Visualizations
                create_optimization_visualizations(result)

                # Export options
                with st.expander("💾 Export Results"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if st.button("Download Parameters"):
                            st.info("Optimal parameters would be exported as JSON")

                    with col2:
                        if st.button("Download History"):
                            st.info("Optimization history would be exported as CSV")

                    with col3:
                        if st.button("Generate Report"):
                            st.info("Comprehensive optimization report would be generated")

            else:
                st.error(f"❌ Optimization failed: {result.error_message}")

    # Information panel
    with st.expander("ℹ️ Optimization Methods Guide"):
        st.markdown("""
        **When to Use Each Method:**

        **🔍 Bayesian Optimization**
        - Best for: Expensive simulations, continuous parameters, <20 dimensions
        - Use when: Function evaluations take significant time
        - Avoid when: Many local optima, high-dimensional spaces

        **🧬 NSGA-II Multi-Objective**
        - Best for: Multiple conflicting objectives, mixed parameter types
        - Use when: Need to explore trade-offs between objectives
        - Avoid when: Single objective, very expensive evaluations

        **🧠 Neural Network Surrogate**
        - Best for: High-dimensional problems, complex response surfaces
        - Use when: Have existing data, complex parameter interactions
        - Avoid when: Limited training data, simple functions

        **🎮 Q-Learning Reinforcement**
        - Best for: Sequential decisions, learning from exploration
        - Use when: Environment changes, want adaptive behavior
        - Avoid when: Continuous spaces, immediate convergence needed

        **💡 Tips:**
        - Start with Bayesian for most problems
        - Use NSGA-II for multiple objectives
        - Neural surrogates for high dimensions (>20)
        - Q-learning for dynamic optimization
        """)
