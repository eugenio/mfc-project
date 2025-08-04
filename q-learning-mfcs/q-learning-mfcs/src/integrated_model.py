#!/usr/bin/env python3
"""
Integrated Model for Cross-System MFC Integration

This module provides a high-level integration framework that combines:
- Physics models (biofilm, metabolic, electrochemical)  
- Machine learning controllers (Q-learning, transformer, etc.)
- Control systems (flow, substrate, temperature)
- Hardware interfaces (sensors, actuators)
- Data processing and validation

Key classes:
- IntegratedSystemState: Unified state representation
- IntegratedModelManager: Main orchestration class
- PhysicsMLBridge: Physics-ML interaction handler
- ControlSystemBridge: Control system integration
- DataIntegrationPipeline: Data flow management
"""

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config.config_manager import ConfigManager
from .controller_models.real_time_controller import RealTimeController

# Import key components for integration
from .integrated_mfc_model import IntegratedMFCModel, IntegratedMFCState
from .ml.physics_ml_integration import PhysicsMLConfig, PhysicsMLIntegrator
from .monitoring.dashboard_api import DashboardAPI
from .sensing_models.sensor_fusion import SensorFusion
from .validation.experimental_validation import MFCExperimentalDatabase

logger = logging.getLogger(__name__)


@dataclass
class SystemPerformanceMetrics:
    """Comprehensive system performance tracking"""
    power_output: float = 0.0
    efficiency: float = 0.0
    biofilm_health: float = 0.0
    substrate_utilization: float = 0.0
    control_stability: float = 0.0
    prediction_accuracy: float = 0.0
    hardware_reliability: float = 1.0
    overall_score: float = 0.0

    # Detailed breakdowns
    physics_metrics: dict[str, float] = field(default_factory=dict)
    ml_metrics: dict[str, float] = field(default_factory=dict)
    control_metrics: dict[str, float] = field(default_factory=dict)
    sensor_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class IntegratedSystemState:
    """Unified state representation across all subsystems"""
    # Time and simulation info
    timestamp: datetime = field(default_factory=datetime.now)
    simulation_time: float = 0.0
    step_count: int = 0

    # Physics state
    mfc_state: IntegratedMFCState | None = None
    biofilm_thickness: float = 0.0
    substrate_concentration: float = 0.0
    ph_level: float = 7.0
    temperature: float = 298.15  # Kelvin

    # ML state
    q_values: np.ndarray | None = None
    policy_state: dict[str, Any] | None = None
    learning_rate: float = 0.01
    exploration_rate: float = 0.1

    # Control state
    flow_rate: float = 0.0
    substrate_feed_rate: float = 0.0
    recirculation_rate: float = 0.0
    pump_speed: float = 0.0

    # Sensor state
    voltage: float = 0.0
    current: float = 0.0
    impedance: float = float('inf')
    biofilm_mass: float = 0.0

    # Performance tracking
    metrics: SystemPerformanceMetrics = field(default_factory=SystemPerformanceMetrics)

    # Integration flags
    physics_active: bool = True
    ml_active: bool = True
    control_active: bool = True
    sensors_active: bool = True


class PhysicsMLBridge:
    """Handles bidirectional communication between physics and ML components"""

    def __init__(self, physics_model: IntegratedMFCModel, ml_integrator: PhysicsMLIntegrator):
        self.physics_model = physics_model
        self.ml_integrator = ml_integrator
        self.sync_frequency = 10  # steps
        self.last_sync_step = 0

    def sync_models(self, system_state: IntegratedSystemState) -> IntegratedSystemState:
        """Synchronize physics and ML model states"""
        if system_state.step_count - self.last_sync_step >= self.sync_frequency:
            # Update ML model with physics insights
            physics_features = self._extract_physics_features(system_state)
            self.ml_integrator.update_physics_context(physics_features)

            # Update physics model with ML predictions
            ml_predictions = self.ml_integrator.predict_next_state(system_state)
            system_state = self._apply_ml_predictions(system_state, ml_predictions)

            self.last_sync_step = system_state.step_count

        return system_state

    def _extract_physics_features(self, state: IntegratedSystemState) -> dict[str, float]:
        """Extract relevant physics features for ML"""
        return {
            'biofilm_thickness': state.biofilm_thickness,
            'substrate_concentration': state.substrate_concentration,
            'power_density': state.voltage * state.current,
            'efficiency': state.metrics.efficiency,
            'ph_level': state.ph_level,
            'temperature': state.temperature
        }

    def _apply_ml_predictions(self, state: IntegratedSystemState,
                             predictions: dict[str, Any]) -> IntegratedSystemState:
        """Apply ML predictions to system state"""
        if 'optimal_flow_rate' in predictions:
            state.flow_rate = predictions['optimal_flow_rate']
        if 'substrate_feed_rate' in predictions:
            state.substrate_feed_rate = predictions['substrate_feed_rate']
        return state


class ControlSystemBridge:
    """Integrates multiple control subsystems"""

    def __init__(self, config: dict[str, Any]):
        self.controllers = {}
        try:
            self.real_time_controller = RealTimeController(config.get('control', {}))
            self.controllers['real_time'] = self.real_time_controller
        except Exception as e:
            logger.warning(f"Failed to initialize real-time controller: {e}")

        self.control_history = []

    def update_control_systems(self, system_state: IntegratedSystemState) -> IntegratedSystemState:
        """Update all control systems based on current state"""
        control_inputs = {
            'voltage': system_state.voltage,
            'current': system_state.current,
            'flow_rate': system_state.flow_rate,
            'substrate_concentration': system_state.substrate_concentration,
            'biofilm_thickness': system_state.biofilm_thickness
        }

        # Get control outputs from each controller
        control_outputs = {}
        for name, controller in self.controllers.items():
            try:
                outputs = controller.compute_control_action(control_inputs)
                control_outputs[name] = outputs
            except Exception as e:
                logger.error(f"Controller {name} failed: {e}")
                continue

        # Merge control outputs (with priority handling)
        system_state = self._merge_control_outputs(system_state, control_outputs)

        # Record control history
        self.control_history.append({
            'timestamp': system_state.timestamp,
            'inputs': control_inputs,
            'outputs': control_outputs
        })

        return system_state

    def _merge_control_outputs(self, state: IntegratedSystemState,
                              outputs: dict[str, dict[str, float]]) -> IntegratedSystemState:
        """Merge outputs from multiple controllers"""
        # Simple averaging for now - could implement more sophisticated arbitration
        if outputs:
            flow_rates = [out.get('flow_rate', state.flow_rate)
                         for out in outputs.values() if 'flow_rate' in out]
            if flow_rates:
                state.flow_rate = np.mean(flow_rates)

            substrate_rates = [out.get('substrate_feed_rate', state.substrate_feed_rate)
                              for out in outputs.values() if 'substrate_feed_rate' in out]
            if substrate_rates:
                state.substrate_feed_rate = np.mean(substrate_rates)

        return state


class DataIntegrationPipeline:
    """Manages data flow between all system components"""

    def __init__(self, config: dict[str, Any]):
        self.data_buffer = []
        self.validation_database = None
        self.dashboard_api = None

        try:
            self.validation_database = MFCExperimentalDatabase(
                config.get('validation', {})
            )
        except Exception as e:
            logger.warning(f"Failed to initialize validation database: {e}")

        try:
            self.dashboard_api = DashboardAPI(config.get('monitoring', {}))
        except Exception as e:
            logger.warning(f"Failed to initialize dashboard API: {e}")

    def process_data_flow(self, system_state: IntegratedSystemState) -> IntegratedSystemState:
        """Process data flow through integration pipeline"""
        # Buffer current state
        self.data_buffer.append(self._serialize_state(system_state))

        # Validate against experimental data
        if self.validation_database:
            validation_score = self._validate_against_experiments(system_state)
            system_state.metrics.prediction_accuracy = validation_score

        # Update monitoring dashboard
        if self.dashboard_api:
            self._update_dashboard(system_state)

        # Periodic data cleanup
        if len(self.data_buffer) > 1000:
            self.data_buffer = self.data_buffer[-500:]  # Keep last 500 entries

        return system_state

    def _serialize_state(self, state: IntegratedSystemState) -> dict[str, Any]:
        """Serialize state for storage/transmission"""
        return {
            'timestamp': state.timestamp.isoformat(),
            'simulation_time': state.simulation_time,
            'step_count': state.step_count,
            'voltage': state.voltage,
            'current': state.current,
            'power': state.voltage * state.current,
            'efficiency': state.metrics.efficiency,
            'biofilm_thickness': state.biofilm_thickness,
            'substrate_concentration': state.substrate_concentration,
            'flow_rate': state.flow_rate
        }

    def _validate_against_experiments(self, state: IntegratedSystemState) -> float:
        """Validate predictions against experimental data"""
        try:
            experimental_data = self.validation_database.get_matching_conditions({
                'substrate_concentration': state.substrate_concentration,
                'temperature': state.temperature,
                'ph': state.ph_level
            })

            if experimental_data:
                predicted_power = state.voltage * state.current
                actual_power = experimental_data.get('power_density', predicted_power)
                error = abs(predicted_power - actual_power) / max(actual_power, 0.001)
                return max(0.0, 1.0 - error)

        except Exception as e:
            logger.error(f"Validation failed: {e}")

        return 0.5  # Default moderate score if validation fails

    def _update_dashboard(self, state: IntegratedSystemState):
        """Update monitoring dashboard with current state"""
        try:
            dashboard_data = {
                'power_output': state.voltage * state.current,
                'efficiency': state.metrics.efficiency,
                'biofilm_health': state.metrics.biofilm_health,
                'control_stability': state.metrics.control_stability,
                'timestamp': state.timestamp.isoformat()
            }
            self.dashboard_api.update_real_time_data(dashboard_data)
        except Exception as e:
            logger.error(f"Dashboard update failed: {e}")


class IntegratedModelManager:
    """Main orchestration class for the integrated MFC system"""

    def __init__(self, config_path: str | None = None):
        # Load configuration
        self.config_manager = ConfigManager()
        if config_path:
            self.config = self.config_manager.load_config(config_path)
        else:
            self.config = self.config_manager.get_default_config()

        # Initialize system state
        self.system_state = IntegratedSystemState()

        # Initialize core components
        self._initialize_core_components()

        # Initialize integration bridges
        self._initialize_integration_bridges()

        # Performance tracking
        self.performance_history = []
        self.simulation_results = {}

    def _initialize_core_components(self):
        """Initialize all core system components"""
        try:
            # Physics model
            self.physics_model = IntegratedMFCModel(
                n_cells=self.config.get('n_cells', 5),
                species='geobacter_sulfurreducens',
                substrate='acetate',
                membrane_type='proton_exchange',
                use_gpu=self.config.get('use_gpu', False),
                simulation_hours=self.config.get('simulation_hours', 24)
            )

            # ML integrator
            ml_config = PhysicsMLConfig(**self.config.get('ml_integration', {}))
            self.ml_integrator = PhysicsMLIntegrator(ml_config)

            # Sensor fusion
            self.sensor_fusion = SensorFusion(self.config.get('sensors', {}))

        except Exception as e:
            logger.error(f"Failed to initialize core components: {e}")
            raise

    def _initialize_integration_bridges(self):
        """Initialize integration bridge components"""
        try:
            # Physics-ML bridge
            self.physics_ml_bridge = PhysicsMLBridge(
                self.physics_model, self.ml_integrator
            )

            # Control system bridge
            self.control_bridge = ControlSystemBridge(self.config)

            # Data integration pipeline
            self.data_pipeline = DataIntegrationPipeline(self.config)

        except Exception as e:
            logger.error(f"Failed to initialize integration bridges: {e}")
            raise

    def step(self) -> IntegratedSystemState:
        """Execute one integration step across all subsystems"""
        try:
            # Update simulation time
            self.system_state.simulation_time += self.config.get('dt', 0.1)
            self.system_state.step_count += 1
            self.system_state.timestamp = datetime.now()

            # Physics model step
            if self.system_state.physics_active:
                self.system_state = self._step_physics()

            # ML integration step
            if self.system_state.ml_active:
                self.system_state = self.physics_ml_bridge.sync_models(self.system_state)

            # Control system step
            if self.system_state.control_active:
                self.system_state = self.control_bridge.update_control_systems(self.system_state)

            # Sensor integration step
            if self.system_state.sensors_active:
                self.system_state = self._step_sensors()

            # Data integration step
            self.system_state = self.data_pipeline.process_data_flow(self.system_state)

            # Update performance metrics
            self.system_state.metrics = self._calculate_integrated_metrics()

            # Record performance history
            self.performance_history.append({
                'step': self.system_state.step_count,
                'timestamp': self.system_state.timestamp,
                'metrics': self.system_state.metrics
            })

            return self.system_state

        except Exception as e:
            logger.error(f"Integration step failed: {e}")
            raise

    def _step_physics(self) -> IntegratedSystemState:
        """Execute physics model step"""
        try:
            # Convert integrated state to MFC model format
            mfc_action = {
                'flow_rate': self.system_state.flow_rate,
                'substrate_concentration': self.system_state.substrate_concentration
            }

            # Step the physics model
            mfc_state, reward, done, info = self.physics_model.step_integrated_dynamics(
                mfc_action, self.system_state.simulation_time
            )

            # Update integrated state with physics results
            self.system_state.mfc_state = mfc_state
            self.system_state.voltage = info.get('voltage', self.system_state.voltage)
            self.system_state.current = info.get('current', self.system_state.current)
            self.system_state.biofilm_thickness = info.get('biofilm_thickness',
                                                          self.system_state.biofilm_thickness)

            return self.system_state

        except Exception as e:
            logger.error(f"Physics step failed: {e}")
            return self.system_state

    def _step_sensors(self) -> IntegratedSystemState:
        """Execute sensor integration step"""
        try:
            # Collect sensor data
            sensor_data = {
                'voltage': self.system_state.voltage,
                'current': self.system_state.current,
                'temperature': self.system_state.temperature,
                'ph': self.system_state.ph_level,
                'biofilm_thickness': self.system_state.biofilm_thickness
            }

            # Process through sensor fusion
            fused_data = self.sensor_fusion.fuse_sensor_data(sensor_data)

            # Update state with fused sensor data
            self.system_state.impedance = fused_data.get('impedance', self.system_state.impedance)
            self.system_state.biofilm_mass = fused_data.get('biofilm_mass', self.system_state.biofilm_mass)

            return self.system_state

        except Exception as e:
            logger.error(f"Sensor step failed: {e}")
            return self.system_state

    def _calculate_integrated_metrics(self) -> SystemPerformanceMetrics:
        """Calculate comprehensive system performance metrics"""
        metrics = SystemPerformanceMetrics()

        # Basic power metrics
        power = self.system_state.voltage * self.system_state.current
        metrics.power_output = power

        # Efficiency calculation
        if self.system_state.substrate_concentration > 0:
            theoretical_max = self.system_state.substrate_concentration * 0.1  # Simplified
            metrics.efficiency = min(power / max(theoretical_max, 0.001), 1.0)

        # Biofilm health (based on thickness and activity)
        optimal_thickness = 0.1  # mm, example value
        thickness_score = 1.0 - abs(self.system_state.biofilm_thickness - optimal_thickness) / optimal_thickness
        metrics.biofilm_health = max(0.0, thickness_score)

        # Substrate utilization
        if hasattr(self.physics_model, 'initial_substrate_concentration'):
            initial_conc = self.physics_model.initial_substrate_concentration
            current_conc = self.system_state.substrate_concentration
            metrics.substrate_utilization = max(0.0, 1.0 - current_conc / max(initial_conc, 0.001))

        # Control stability (based on control action variance)
        if len(self.control_bridge.control_history) > 10:
            recent_actions = [h['outputs'] for h in self.control_bridge.control_history[-10:]]
            if recent_actions:
                # Calculate variance in control actions as stability metric
                flow_rates = [a.get('real_time', {}).get('flow_rate', 0) for a in recent_actions]
                if flow_rates:
                    variance = np.var(flow_rates)
                    metrics.control_stability = max(0.0, 1.0 - variance / 10.0)  # Normalize

        # Overall score (weighted average)
        weights = {
            'power_output': 0.3,
            'efficiency': 0.25,
            'biofilm_health': 0.2,
            'substrate_utilization': 0.15,
            'control_stability': 0.1
        }

        metrics.overall_score = (
            weights['power_output'] * min(metrics.power_output / 10.0, 1.0) +
            weights['efficiency'] * metrics.efficiency +
            weights['biofilm_health'] * metrics.biofilm_health +
            weights['substrate_utilization'] * metrics.substrate_utilization +
            weights['control_stability'] * metrics.control_stability
        )

        return metrics

    def run_simulation(self, steps: int, save_interval: int = 100) -> dict[str, Any]:
        """Run integrated simulation for specified number of steps"""
        logger.info(f"Starting integrated simulation for {steps} steps")

        start_time = datetime.now()

        try:
            for step in range(steps):
                self.step()

                # Periodic saving
                if step % save_interval == 0:
                    self.save_checkpoint(f"checkpoint_step_{step}.pkl")
                    logger.info(f"Completed step {step}/{steps} - Overall score: {self.system_state.metrics.overall_score:.3f}")

        except Exception as e:
            logger.error(f"Simulation failed at step {self.system_state.step_count}: {e}")
            raise

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Compile final results
        self.simulation_results = {
            'total_steps': steps,
            'simulation_time': self.system_state.simulation_time,
            'duration_seconds': duration,
            'final_state': self.system_state,
            'performance_history': self.performance_history,
            'average_performance': np.mean([h['metrics'].overall_score for h in self.performance_history]),
            'peak_performance': max([h['metrics'].overall_score for h in self.performance_history]),
            'final_metrics': self.system_state.metrics
        }

        logger.info(f"Simulation completed in {duration:.1f}s - Average performance: {self.simulation_results['average_performance']:.3f}")

        return self.simulation_results

    def save_checkpoint(self, filename: str):
        """Save current state as checkpoint"""
        checkpoint_data = {
            'system_state': self.system_state,
            'performance_history': self.performance_history,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }

        with open(filename, 'wb') as f:
            pickle.dump(checkpoint_data, f)

    def load_checkpoint(self, filename: str):
        """Load state from checkpoint"""
        with open(filename, 'rb') as f:
            checkpoint_data = pickle.load(f)

        self.system_state = checkpoint_data['system_state']
        self.performance_history = checkpoint_data['performance_history']
        logger.info(f"Loaded checkpoint from {checkpoint_data['timestamp']}")

    def export_results(self, output_dir: str = "results"):
        """Export comprehensive simulation results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Export simulation results
        with open(output_path / "simulation_results.json", 'w') as f:
            # Convert non-serializable objects
            serializable_results = self._make_serializable(self.simulation_results)
            json.dump(serializable_results, f, indent=2)

        # Export performance history
        with open(output_path / "performance_history.json", 'w') as f:
            serializable_history = [self._make_serializable(h) for h in self.performance_history]
            json.dump(serializable_history, f, indent=2)

        # Export final state
        with open(output_path / "final_state.json", 'w') as f:
            serializable_state = self._make_serializable(self.system_state)
            json.dump(serializable_state, f, indent=2)

        logger.info(f"Results exported to {output_path}")

    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format"""
        if hasattr(obj, '__dict__'):
            result = {}
            for key, value in obj.__dict__.items():
                if isinstance(value, (int, float, str, bool, type(None))):
                    result[key] = value
                elif isinstance(value, datetime):
                    result[key] = value.isoformat()
                elif isinstance(value, np.ndarray):
                    result[key] = value.tolist()
                elif isinstance(value, (list, tuple)):
                    result[key] = [self._make_serializable(item) for item in value]
                elif isinstance(value, dict):
                    result[key] = {k: self._make_serializable(v) for k, v in value.items()}
                else:
                    result[key] = str(value)  # Fallback to string representation
            return result
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        else:
            return obj


def main():
    """Example usage of the integrated model system"""
    logging.basicConfig(level=logging.INFO)

    # Create integrated model manager
    manager = IntegratedModelManager()

    # Run simulation
    results = manager.run_simulation(steps=1000, save_interval=100)

    # Export results
    manager.export_results("integrated_model_results")

    print(f"Simulation completed with average performance: {results['average_performance']:.3f}")
    print(f"Peak performance: {results['peak_performance']:.3f}")
    print(f"Final overall score: {results['final_metrics'].overall_score:.3f}")


if __name__ == "__main__":
    main()
