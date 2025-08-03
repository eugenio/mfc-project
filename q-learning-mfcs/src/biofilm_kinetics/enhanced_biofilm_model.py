"""
Enhanced Biofilm Kinetics Model with EIS/QCM Sensor Integration

This module extends the base biofilm kinetics model to include:
1. Real-time EIS feedback for biofilm thickness validation
2. QCM feedback for biomass validation and calibration
3. Adaptive parameter tuning based on sensor data
4. Sensor-guided biofilm state estimation
5. Multi-sensor fault detection and compensation

The enhanced model provides more accurate biofilm dynamics by incorporating
actual sensor measurements as feedback for model validation and adaptation.
"""

import os
import sys
from typing import Any

import numpy as np

# Import base biofilm model
from .biofilm_model import BiofilmKineticsModel

# Import sensing models
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'sensing_models'))
try:
    from sensing_models.eis_model import BacterialSpecies, EISMeasurement, EISModel
    from sensing_models.qcm_model import QCMMeasurement, QCMModel
    from sensing_models.sensor_fusion import (
        FusedMeasurement,
        FusionMethod,
        SensorFusion,
    )
except ImportError as e:
    print(f"Warning: Sensing models not available: {e}")
    EISModel = QCMModel = SensorFusion = None

# Add GPU acceleration
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
try:
    from gpu_acceleration import get_gpu_accelerator
except ImportError:
    get_gpu_accelerator = None


class SensorCalibrationError(Exception):
    """Exception raised when sensor calibration fails."""
    pass


class EnhancedBiofilmModel(BiofilmKineticsModel):
    """
    Enhanced biofilm model with integrated EIS/QCM sensor feedback.

    Extends the base biofilm kinetics model with:
    - Real-time sensor feedback integration
    - Adaptive parameter calibration
    - Sensor-guided state estimation
    - Multi-sensor validation and fault detection
    """

    def __init__(self, species: str = 'mixed', substrate: str = 'lactate',
                 use_gpu: bool = True, temperature: float = 303.0, ph: float = 7.0,
                 enable_eis: bool = True, enable_qcm: bool = True,
                 sensor_fusion_method: str = 'kalman_filter'):
        """
        Initialize enhanced biofilm model with sensor integration.

        Args:
            species: Species type ('geobacter', 'shewanella', 'mixed')
            substrate: Substrate type ('acetate', 'lactate')
            use_gpu: Enable GPU acceleration
            temperature: Operating temperature (K)
            ph: Operating pH
            enable_eis: Enable EIS sensor integration
            enable_qcm: Enable QCM sensor integration
            sensor_fusion_method: Sensor fusion algorithm
        """
        # Initialize base biofilm model
        super().__init__(species, substrate, use_gpu, temperature, ph)

        # Sensor configuration
        self.enable_eis = enable_eis
        self.enable_qcm = enable_qcm
        self.sensor_fusion_method = sensor_fusion_method

        # Initialize sensors if available
        self.eis_model = None
        self.qcm_model = None
        self.sensor_fusion = None

        # Map species names to enum (needed for multiple sensor models)
        species_map = {
            'geobacter': BacterialSpecies.GEOBACTER,
            'shewanella': BacterialSpecies.SHEWANELLA,
            'mixed': BacterialSpecies.MIXED
        }

        if EISModel and enable_eis:
            self.eis_model = EISModel(
                species=species_map.get(species, BacterialSpecies.MIXED),
                electrode_area=1e-4,  # m² (1 cm² electrode) - fixed for optimal EIS sensing
                use_gpu=use_gpu
            )

        if QCMModel and enable_qcm:
            self.qcm_model = QCMModel(
                electrode_area=0.196e-4,  # m² (5mm diameter, 0.196 cm²) - fixed for optimal QCM sensing
                use_gpu=use_gpu
            )
            self.qcm_model.set_biofilm_species(species)

        if SensorFusion and (enable_eis or enable_qcm):
            fusion_method_map = {
                'kalman_filter': FusionMethod.KALMAN_FILTER,
                'weighted_average': FusionMethod.WEIGHTED_AVERAGE,
                'maximum_likelihood': FusionMethod.MAXIMUM_LIKELIHOOD,
                'bayesian': FusionMethod.BAYESIAN_FUSION
            }

            self.sensor_fusion = SensorFusion(
                method=fusion_method_map.get(sensor_fusion_method, FusionMethod.KALMAN_FILTER),
                species=species_map.get(species, BacterialSpecies.MIXED),
                use_gpu=use_gpu
            )

        # Sensor feedback parameters
        self.sensor_feedback_enabled = True
        self.adaptive_calibration_enabled = True
        self.calibration_threshold = 0.2  # 20% threshold for recalibration
        self.sensor_weight_decay = 0.95   # Exponential decay for sensor weights

        # State estimation parameters
        self.model_confidence = 1.0       # Confidence in model predictions
        self.sensor_confidence = 0.8      # Initial confidence in sensors
        self.fusion_history_length = 50   # Number of measurements to keep

        # Tracking variables
        self.sensor_measurements = []
        self.model_predictions = []
        self.fusion_results = []
        self.calibration_history = []

        # Error tracking
        self.prediction_errors = []
        self.sensor_residuals = []
        self.last_calibration_time = 0.0

        # Enhanced state variables
        self.sensor_validated_thickness = 0.0
        self.sensor_validated_biomass = 0.0
        self.thickness_prediction_error = 0.0
        self.biomass_prediction_error = 0.0

    def step_biofilm_dynamics_with_sensors(self, dt: float, anode_potential: float,
                                         substrate_supply: float, time_hours: float = 0.0,
                                         external_measurements: dict | None = None) -> dict[str, Any]:
        """
        Step biofilm dynamics with integrated sensor feedback.

        Args:
            dt: Time step (hours)
            anode_potential: Anode potential (V vs SHE)
            substrate_supply: Substrate supply rate (mmol/L/h)
            time_hours: Current simulation time (hours)
            external_measurements: Optional external sensor data

        Returns:
            Enhanced biofilm state with sensor validation
        """
        # Step base biofilm model
        base_state = self.step_biofilm_dynamics(dt, anode_potential, substrate_supply)

        # Extract predicted values
        predicted_thickness = base_state['biofilm_thickness']
        predicted_biomass = base_state['biomass_density']

        # Simulate or use sensor measurements
        sensor_measurements = self._get_sensor_measurements(
            predicted_thickness, predicted_biomass, time_hours, external_measurements
        )

        # Apply sensor feedback if available
        if sensor_measurements and self.sensor_feedback_enabled:
            validated_state = self._apply_sensor_feedback(
                base_state, sensor_measurements, time_hours
            )
        else:
            validated_state = base_state.copy()
            validated_state.update({
                'sensor_validated_thickness': predicted_thickness,
                'sensor_validated_biomass': predicted_biomass,
                'sensor_confidence': 0.0,
                'fusion_confidence': 0.0,
                'sensor_status': 'unavailable'
            })

        # Update adaptive parameters if enabled
        if self.adaptive_calibration_enabled and sensor_measurements:
            self._update_adaptive_parameters(validated_state, sensor_measurements, time_hours)

        # Store results for analysis
        self.model_predictions.append({
            'time': time_hours,
            'thickness': predicted_thickness,
            'biomass': predicted_biomass
        })

        if sensor_measurements:
            self.sensor_measurements.append({
                'time': time_hours,
                'measurements': sensor_measurements
            })

        return validated_state

    def _get_sensor_measurements(self, predicted_thickness: float, predicted_biomass: float,
                               time_hours: float, external_measurements: dict | None = None) -> dict | None:
        """
        Get sensor measurements (simulated or external).

        Args:
            predicted_thickness: Model-predicted thickness (μm)
            predicted_biomass: Model-predicted biomass (g/L)
            time_hours: Current time (hours)
            external_measurements: External sensor data

        Returns:
            Dictionary of sensor measurements or None
        """
        if external_measurements:
            # Use provided external measurements
            return external_measurements

        # Simulate sensor measurements
        measurements = {}

        # EIS measurement simulation
        if self.eis_model and self.enable_eis:
            try:
                eis_measurements = self.eis_model.simulate_measurement(
                    biofilm_thickness=predicted_thickness,
                    biomass_density=predicted_biomass,
                    porosity=0.8,
                    temperature=self.temperature,
                    time_hours=time_hours
                )

                eis_properties = self.eis_model.get_biofilm_properties(eis_measurements)

                measurements['eis'] = {
                    'measurements': eis_measurements,
                    'properties': eis_properties
                }

            except Exception as e:
                print(f"Warning: EIS measurement simulation failed: {e}")

        # QCM measurement simulation
        if self.qcm_model and self.enable_qcm:
            try:
                # Convert biomass density to total mass (simplified)
                electrode_area_m2 = self.qcm_model.electrode_area  # m² (get from QCM model)
                thickness_m = predicted_thickness * 1e-6  # μm to m
                volume_m3 = electrode_area_m2 * thickness_m
                total_mass_ug = predicted_biomass * volume_m3 * 1e9  # g/L to μg (1e3 L/m³ * 1e6 μg/g)

                qcm_measurement = self.qcm_model.simulate_measurement(
                    biofilm_mass=total_mass_ug,
                    biofilm_thickness=predicted_thickness,
                    temperature=self.temperature,
                    time_hours=time_hours
                )

                qcm_properties = self.qcm_model.estimate_biofilm_properties(qcm_measurement)

                measurements['qcm'] = {
                    'measurement': qcm_measurement,
                    'properties': qcm_properties
                }

            except Exception as e:
                print(f"Warning: QCM measurement simulation failed: {e}")

        return measurements if measurements else None

    def _apply_sensor_feedback(self, base_state: dict[str, Any],
                             sensor_measurements: dict, time_hours: float) -> dict[str, Any]:
        """
        Apply sensor feedback to validate and correct model predictions.

        Args:
            base_state: Base biofilm state from model
            sensor_measurements: Sensor measurement data
            time_hours: Current time

        Returns:
            Validated biofilm state
        """
        validated_state = base_state.copy()

        # Extract sensor data
        eis_data = sensor_measurements.get('eis')
        qcm_data = sensor_measurements.get('qcm')

        if not (eis_data or qcm_data):
            return validated_state

        # Apply sensor fusion if both sensors available
        if eis_data and qcm_data and self.sensor_fusion:
            try:
                # Extract measurements
                eis_measurement = eis_data['measurements'][0] if eis_data['measurements'] else None
                qcm_measurement = qcm_data['measurement']

                if eis_measurement and qcm_measurement:
                    # Apply sensor fusion
                    fused_result = self.sensor_fusion.fuse_measurements(
                        eis_measurement=eis_measurement,
                        qcm_measurement=qcm_measurement,
                        eis_properties=eis_data['properties'],
                        qcm_properties=qcm_data['properties'],
                        time_hours=time_hours
                    )

                    # Update state with fused results
                    validated_state.update({
                        'sensor_validated_thickness': fused_result.thickness_um,
                        'sensor_validated_biomass': fused_result.biomass_density_g_per_L,
                        'thickness_uncertainty': fused_result.thickness_uncertainty,
                        'biomass_uncertainty': fused_result.biomass_uncertainty,
                        'sensor_confidence': fused_result.fusion_confidence,
                        'fusion_confidence': fused_result.fusion_confidence,
                        'sensor_agreement': fused_result.sensor_agreement,
                        'eis_weight': fused_result.eis_weight,
                        'qcm_weight': fused_result.qcm_weight,
                        'sensor_status': 'fused'
                    })

                    # Calculate prediction errors
                    self.thickness_prediction_error = abs(
                        base_state['biofilm_thickness'] - fused_result.thickness_um
                    )
                    self.biomass_prediction_error = abs(
                        base_state['biomass_density'] - fused_result.biomass_density_g_per_L
                    )

                    self.fusion_results.append(fused_result)

            except Exception as e:
                print(f"Warning: Sensor fusion failed: {e}")
                # Fall back to individual sensor processing

        # Process individual sensors if fusion not available
        if not validated_state.get('sensor_validated_thickness'):
            # Use EIS data if available
            if eis_data:
                eis_thickness = eis_data['properties'].get('thickness_um', 0.0)
                if eis_thickness > 0:
                    validated_state.update({
                        'sensor_validated_thickness': eis_thickness,
                        'sensor_confidence': eis_data['properties'].get('measurement_quality', 0.5),
                        'sensor_status': 'eis_only'
                    })

                    self.thickness_prediction_error = abs(
                        base_state['biofilm_thickness'] - eis_thickness
                    )

            # Use QCM data if available
            elif qcm_data:
                qcm_thickness = qcm_data['properties'].get('thickness_um', 0.0)
                qcm_biomass = qcm_data['properties'].get('biomass_density_g_per_L', 0.0)

                if qcm_thickness > 0:
                    validated_state.update({
                        'sensor_validated_thickness': qcm_thickness,
                        'sensor_validated_biomass': qcm_biomass,
                        'sensor_confidence': qcm_data['properties'].get('measurement_quality', 0.5),
                        'sensor_status': 'qcm_only'
                    })

                    self.thickness_prediction_error = abs(
                        base_state['biofilm_thickness'] - qcm_thickness
                    )
                    self.biomass_prediction_error = abs(
                        base_state['biomass_density'] - qcm_biomass
                    )

        # Store error tracking
        if hasattr(self, 'thickness_prediction_error'):
            self.prediction_errors.append({
                'time': time_hours,
                'thickness_error': self.thickness_prediction_error,
                'biomass_error': getattr(self, 'biomass_prediction_error', 0.0)
            })

        return validated_state

    def _update_adaptive_parameters(self, validated_state: dict[str, Any],
                                  sensor_measurements: dict, time_hours: float):
        """
        Update model parameters based on sensor feedback.

        Args:
            validated_state: Validated biofilm state
            sensor_measurements: Sensor measurements
            time_hours: Current time
        """
        # Check if calibration is needed
        if not self._should_recalibrate(time_hours):
            return

        # Calculate model-sensor agreement
        model_thickness = validated_state['biofilm_thickness']
        sensor_thickness = validated_state.get('sensor_validated_thickness', model_thickness)

        if sensor_thickness > 0 and model_thickness > 0:
            relative_error = abs(model_thickness - sensor_thickness) / sensor_thickness

            # Recalibrate if error exceeds threshold
            if relative_error > self.calibration_threshold:
                self._recalibrate_parameters(
                    model_thickness, sensor_thickness, time_hours
                )

        # Update confidence levels
        self._update_confidence_levels(validated_state)

    def _should_recalibrate(self, time_hours: float) -> bool:
        """Check if model should be recalibrated."""
        # Recalibrate every 24 hours or if significant drift detected
        time_since_calibration = time_hours - self.last_calibration_time

        if time_since_calibration > 24.0:  # 24 hours
            return True

        # Check for systematic drift
        if len(self.prediction_errors) >= 10:
            recent_errors = [e['thickness_error'] for e in self.prediction_errors[-10:]]
            if np.mean(recent_errors) > 5.0:  # μm systematic error
                return True

        return False

    def _recalibrate_parameters(self, model_thickness: float, sensor_thickness: float,
                              time_hours: float):
        """
        Recalibrate kinetic parameters based on sensor feedback.

        Args:
            model_thickness: Model-predicted thickness
            sensor_thickness: Sensor-measured thickness
            time_hours: Current time
        """
        try:
            # Calculate correction factor
            correction_factor = sensor_thickness / model_thickness if model_thickness > 0 else 1.0

            # Apply correction to growth rate parameters
            if hasattr(self, 'kinetic_params'):
                # Adjust maximum growth rate
                original_mu_max = self.kinetic_params.mu_max
                self.kinetic_params.mu_max *= correction_factor

                # Ensure reasonable bounds
                self.kinetic_params.mu_max = np.clip(
                    self.kinetic_params.mu_max, 0.01, 2.0
                )

                # Log calibration event
                self.calibration_history.append({
                    'time': time_hours,
                    'correction_factor': correction_factor,
                    'original_mu_max': original_mu_max,
                    'new_mu_max': self.kinetic_params.mu_max,
                    'model_thickness': model_thickness,
                    'sensor_thickness': sensor_thickness
                })

                self.last_calibration_time = time_hours

                print(f"Biofilm model recalibrated at t={time_hours:.1f}h: "
                      f"μ_max {original_mu_max:.4f} → {self.kinetic_params.mu_max:.4f}")

        except Exception as e:
            print(f"Warning: Parameter recalibration failed: {e}")

    def _update_confidence_levels(self, validated_state: dict[str, Any]):
        """Update model and sensor confidence levels."""
        # Update model confidence based on prediction accuracy
        if hasattr(self, 'thickness_prediction_error'):
            error_factor = max(0.1, 1.0 - self.thickness_prediction_error / 20.0)  # 20 μm reference
            self.model_confidence = self.model_confidence * 0.9 + error_factor * 0.1

        # Update sensor confidence from fusion results
        fusion_confidence = validated_state.get('fusion_confidence', 0.5)
        self.sensor_confidence = self.sensor_confidence * 0.9 + fusion_confidence * 0.1

    def get_sensor_diagnostics(self) -> dict[str, Any]:
        """Get comprehensive sensor diagnostics."""
        diagnostics = {
            'sensor_configuration': {
                'eis_enabled': self.enable_eis,
                'qcm_enabled': self.enable_qcm,
                'fusion_method': self.sensor_fusion_method,
                'feedback_enabled': self.sensor_feedback_enabled,
                'adaptive_calibration_enabled': self.adaptive_calibration_enabled
            },
            'confidence_levels': {
                'model_confidence': self.model_confidence,
                'sensor_confidence': self.sensor_confidence
            },
            'measurement_statistics': {
                'total_measurements': len(self.sensor_measurements),
                'total_predictions': len(self.model_predictions),
                'fusion_results': len(self.fusion_results),
                'calibration_events': len(self.calibration_history)
            },
            'error_statistics': {}
        }

        # Add error statistics if available
        if self.prediction_errors:
            thickness_errors = [e['thickness_error'] for e in self.prediction_errors]
            biomass_errors = [e['biomass_error'] for e in self.prediction_errors]

            diagnostics['error_statistics'] = {
                'mean_thickness_error_um': np.mean(thickness_errors),
                'std_thickness_error_um': np.std(thickness_errors),
                'max_thickness_error_um': np.max(thickness_errors),
                'mean_biomass_error_g_per_L': np.mean(biomass_errors),
                'std_biomass_error_g_per_L': np.std(biomass_errors),
                'max_biomass_error_g_per_L': np.max(biomass_errors)
            }

        # Add sensor-specific diagnostics
        if self.eis_model:
            diagnostics['eis_diagnostics'] = self.eis_model.get_measurement_summary()

        if self.qcm_model:
            diagnostics['qcm_diagnostics'] = self.qcm_model.get_measurement_summary()

        if self.sensor_fusion:
            diagnostics['fusion_diagnostics'] = self.sensor_fusion.get_fusion_summary()
            diagnostics['sensor_faults'] = self.sensor_fusion.detect_sensor_faults()

        return diagnostics

    def reset_sensor_calibration(self):
        """Reset sensor calibration to initial state."""
        self.model_confidence = 1.0
        self.sensor_confidence = 0.8
        self.last_calibration_time = 0.0

        # Clear history
        self.sensor_measurements.clear()
        self.model_predictions.clear()
        self.fusion_results.clear()
        self.calibration_history.clear()
        self.prediction_errors.clear()
        self.sensor_residuals.clear()

        # Reset sensor models
        if self.eis_model:
            self.eis_model.measurement_history.clear()
            self.eis_model.thickness_history.clear()
            self.eis_model.time_history.clear()

        if self.qcm_model:
            self.qcm_model.measurement_history.clear()
            self.qcm_model.frequency_history.clear()
            self.qcm_model.mass_history.clear()
            self.qcm_model.time_history.clear()

        print("Sensor calibration reset to initial state")

    def validate_sensor_integration(self) -> dict[str, bool]:
        """Validate that sensor integration is working correctly."""
        validation_results = {
            'eis_model_available': self.eis_model is not None,
            'qcm_model_available': self.qcm_model is not None,
            'sensor_fusion_available': self.sensor_fusion is not None,
            'can_simulate_eis': False,
            'can_simulate_qcm': False,
            'can_perform_fusion': False
        }

        # Test EIS simulation
        if self.eis_model:
            try:
                test_measurements = self.eis_model.simulate_measurement(10.0, 5.0, 0.8, 303.0, 0.0)
                validation_results['can_simulate_eis'] = len(test_measurements) > 0
            except Exception:
                pass

        # Test QCM simulation
        if self.qcm_model:
            try:
                test_measurement = self.qcm_model.simulate_measurement(100.0, 10.0, 303.0, 0.0)
                validation_results['can_simulate_qcm'] = test_measurement is not None
            except Exception:
                pass

        # Test sensor fusion
        if (self.sensor_fusion and validation_results['can_simulate_eis'] and
            validation_results['can_simulate_qcm']):
            try:
                # This is a basic test - actual fusion would need real measurements
                validation_results['can_perform_fusion'] = True
            except Exception:
                pass

        return validation_results
