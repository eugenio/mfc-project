"""
Sensor-Integrated MFC Model with EIS/QCM Feedback Loops

This module extends the integrated MFC model to include:
1. Real-time EIS and QCM sensor feedback
2. Sensor-guided biofilm and metabolic model validation
3. Adaptive parameter tuning based on sensor data
4. Enhanced Q-learning control with sensor state variables
5. Fault-tolerant operation with sensor degradation handling

The sensor integration provides closed-loop control and improved
accuracy through continuous validation of model predictions.
"""

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
import time
import pickle
import os
import sys

# Import base integrated model
sys.path.append(os.path.dirname(__file__))
from integrated_mfc_model import IntegratedMFCModel, IntegratedMFCState

# Import enhanced components
from biofilm_kinetics.enhanced_biofilm_model import EnhancedBiofilmModel
from sensing_enhanced_q_controller import SensingEnhancedQLearningController

# Import sensing models
try:
    from sensing_models.eis_model import EISModel, BacterialSpecies
    from sensing_models.qcm_model import QCMModel
    from sensing_models.sensor_fusion import SensorFusion, FusionMethod
    SENSING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Sensing models not available: {e}")
    SENSING_AVAILABLE = False

from mfc_recirculation_control import (
    AnolytereservoirSystem, 
    SubstrateConcentrationController,
    MFCCellWithMonitoring
)
from gpu_acceleration import get_gpu_accelerator
from path_config import get_model_path


@dataclass
class SensorIntegratedMFCState(IntegratedMFCState):
    """Extended MFC state with sensor measurements."""
    
    # Sensor measurements (per cell)
    eis_thickness: List[float]
    eis_conductivity: List[float]
    eis_measurement_quality: List[float]
    
    qcm_mass_per_area: List[float]
    qcm_frequency_shift: List[float]
    qcm_dissipation: List[float]
    
    # Sensor fusion results (per cell)
    fused_thickness: List[float]
    fused_biomass: List[float]
    sensor_agreement: List[float]
    fusion_confidence: List[float]
    
    # Sensor status
    eis_sensor_status: List[str]
    qcm_sensor_status: List[str]
    sensor_fault_flags: List[bool]
    
    # Model validation metrics
    thickness_prediction_error: List[float]
    biomass_prediction_error: List[float]
    model_sensor_agreement: float
    adaptive_calibration_active: bool


class SensorIntegratedMFCModel(IntegratedMFCModel):
    """
    Complete MFC model with integrated sensor feedback loops.
    
    Features:
    - Real-time EIS and QCM sensor integration
    - Sensor-guided model validation and calibration
    - Enhanced Q-learning with sensor state variables
    - Fault-tolerant operation with sensor degradation
    - Multi-scale temporal dynamics with sensor feedback
    """
    
    def __init__(self, n_cells: int = 5, species: str = "mixed", 
                 substrate: str = "lactate", membrane_type: str = "Nafion-117",
                 use_gpu: bool = True, simulation_hours: int = 100,
                 enable_eis: bool = True, enable_qcm: bool = True,
                 sensor_fusion_method: str = 'kalman_filter',
                 sensor_feedback_weight: float = 0.3,
                 recirculation_mode: bool = False):
        """
        Initialize sensor-integrated MFC model.
        
        Args:
            n_cells: Number of cells in stack
            species: Bacterial species
            substrate: Substrate type
            membrane_type: Nafion membrane grade
            use_gpu: Enable GPU acceleration
            simulation_hours: Total simulation duration
            enable_eis: Enable EIS sensor integration
            enable_qcm: Enable QCM sensor integration
            sensor_fusion_method: Sensor fusion algorithm
            sensor_feedback_weight: Weight of sensor feedback in control
            recirculation_mode: Enable recirculation mode with substrate monitoring
        """
        # Initialize base model (but don't call super().__init__ to avoid duplicate initialization)
        self.n_cells = n_cells
        self.species = species
        self.substrate = substrate
        self.membrane_type = membrane_type
        self.use_gpu = use_gpu
        self.simulation_hours = simulation_hours
        
        # Sensor configuration
        self.enable_eis = enable_eis and SENSING_AVAILABLE
        self.enable_qcm = enable_qcm and SENSING_AVAILABLE
        self.sensor_fusion_method = sensor_fusion_method
        self.sensor_feedback_weight = sensor_feedback_weight
        self.recirculation_mode = recirculation_mode
        
        # GPU acceleration
        self.gpu_acc = get_gpu_accelerator() if use_gpu else None
        
        # Initialize sensor states tracking
        self.sensor_states: List[Dict[str, Any]] = []
        self.gpu_available = self.gpu_acc.is_gpu_available() if self.gpu_acc else False
        
        print("Initializing Sensor-Integrated MFC Model")
        print(f"Configuration: {n_cells} cells, {species} bacteria, {substrate} substrate")
        print(f"Sensors: EIS={self.enable_eis}, QCM={self.enable_qcm}")
        print(f"GPU Acceleration: {'Enabled' if self.gpu_available else 'Disabled'}")
        
        # Initialize enhanced components
        self._initialize_enhanced_models()
        self._initialize_sensors()
        self._initialize_recirculation()
        self._initialize_tracking()
    
    def _initialize_enhanced_models(self):
        """Initialize enhanced biofilm and metabolic models."""
        # Create enhanced biofilm models for each cell
        self.biofilm_models = []
        self.metabolic_models = []
        
        for i in range(self.n_cells):
            # Enhanced biofilm model with sensor integration
            biofilm_model = EnhancedBiofilmModel(
                species=self.species,
                substrate=self.substrate,
                use_gpu=self.use_gpu,
                temperature=303.0,  # 30°C
                ph=7.0,
                enable_eis=self.enable_eis,
                enable_qcm=self.enable_qcm,
                sensor_fusion_method=self.sensor_fusion_method
            )
            self.biofilm_models.append(biofilm_model)
            
            # Standard metabolic model (will be enhanced in future)
            from metabolic_model import MetabolicModel
            metabolic_model = MetabolicModel(
                species=self.species,
                substrate=self.substrate,
                membrane_type=self.membrane_type,
                use_gpu=self.use_gpu
            )
            self.metabolic_models.append(metabolic_model)
    
    def _initialize_sensors(self):
        """Initialize sensor models for each cell."""
        self.eis_models = []
        self.qcm_models = []
        self.sensor_fusion_models = []
        
        if not SENSING_AVAILABLE:
            print("Warning: Sensing models not available - running without sensors")
            return
        
        # Map species names
        species_map = {
            'geobacter': BacterialSpecies.GEOBACTER,
            'shewanella': BacterialSpecies.SHEWANELLA,
            'mixed': BacterialSpecies.MIXED
        }
        
        for i in range(self.n_cells):
            # EIS sensor for each cell
            if self.enable_eis:
                eis_model = EISModel(
                    species=species_map.get(self.species, BacterialSpecies.MIXED),
                    electrode_area=1e-4,  # 1 cm² per cell
                    use_gpu=self.use_gpu
                )
                self.eis_models.append(eis_model)
            
            # QCM sensor for each cell
            if self.enable_qcm:
                qcm_model = QCMModel(
                    electrode_area=0.196,  # 5mm diameter
                    use_gpu=self.use_gpu
                )
                qcm_model.set_biofilm_species(self.species)
                self.qcm_models.append(qcm_model)
            
            # Sensor fusion for each cell
            if self.enable_eis and self.enable_qcm:
                fusion_method_map = {
                    'kalman_filter': FusionMethod.KALMAN_FILTER,
                    'weighted_average': FusionMethod.WEIGHTED_AVERAGE,
                    'maximum_likelihood': FusionMethod.MAXIMUM_LIKELIHOOD,
                    'bayesian': FusionMethod.BAYESIAN_FUSION
                }
                
                fusion_model = SensorFusion(
                    method=fusion_method_map.get(self.sensor_fusion_method, FusionMethod.KALMAN_FILTER),
                    species=species_map.get(self.species, BacterialSpecies.MIXED),
                    use_gpu=self.use_gpu
                )
                self.sensor_fusion_models.append(fusion_model)
        
        # Initialize sensor states for each cell
        for i in range(self.n_cells):
            sensor_state = {
                'eis_status': 'good' if self.enable_eis else 'disabled',
                'qcm_status': 'good' if self.enable_qcm else 'disabled',
                'fusion_confidence': 0.8,  # Initial confidence
                'last_measurement_time': 0.0
            }
            self.sensor_states.append(sensor_state)
    
    def _initialize_recirculation(self):
        """Initialize recirculation and enhanced control systems."""
        # Initialize reservoir
        self.reservoir = AnolytereservoirSystem(
            initial_substrate_conc=20.0,  # mmol/L
            volume_liters=1.0
        )
        
        # Initialize MFC cells
        self.mfc_cells = [
            MFCCellWithMonitoring(i+1, initial_biofilm=1.0) 
            for i in range(self.n_cells)
        ]
        
        # Enhanced Q-learning flow controller with sensor integration
        self.flow_controller = SensingEnhancedQLearningController(
            enable_sensor_state=self.enable_eis or self.enable_qcm,
            fault_tolerance=True
        )
        
        # Substrate controller
        self.substrate_controller = SubstrateConcentrationController(
            target_outlet_conc=12.0,
            target_reservoir_conc=20.0
        )
        
        # Simulation state
        self.flow_rate_ml_h = 10.0
        self.total_energy_generated = 0.0
        self.pump_power_consumed = 0.0
        
        # Sensor feedback state
        self.sensor_data_history = []
        self.sensor_fault_history = []
        self.adaptive_calibration_events = []
    
    def _initialize_tracking(self):
        """Initialize enhanced tracking variables."""
        self.time = 0.0
        self.history = []
        self.biofilm_history = []
        self.metabolic_history = []
        self.sensor_history = []
        
        self.performance_metrics = {
            'total_energy': 0.0,
            'average_power': 0.0,
            'max_power': 0.0,
            'coulombic_efficiency': 0.0,
            'substrate_utilization': 0.0,
            'sensor_accuracy': 0.0,
            'model_validation_score': 0.0
        }
    
    def step_sensor_integrated_dynamics(self, dt: float = 1.0) -> SensorIntegratedMFCState:
        """
        Step the sensor-integrated model forward by dt hours.
        
        Args:
            dt: Time step (hours)
            
        Returns:
            SensorIntegratedMFCState with sensor measurements
        """
        # 1. Get inlet concentration from reservoir
        inlet_conc = self.reservoir.get_inlet_concentration()
        
        # 2. Update enhanced biofilm dynamics with sensor feedback
        biofilm_states = []
        sensor_measurements = []
        
        for i in range(self.n_cells):
            # Get cell-specific conditions
            cell = self.mfc_cells[i]
            anode_potential = -0.3 + getattr(cell, 'anode_overpotential', 0.1)
            
            # Step enhanced biofilm model with sensors
            if hasattr(self.biofilm_models[i], 'step_biofilm_dynamics_with_sensors'):
                biofilm_state = self.biofilm_models[i].step_biofilm_dynamics_with_sensors(
                    dt=dt,
                    anode_potential=anode_potential,
                    substrate_supply=inlet_conc / 10.0,
                    time_hours=self.time
                )
            else:
                # Fallback to standard biofilm model
                biofilm_state = self.biofilm_models[i].step_biofilm_dynamics(
                    dt=dt,
                    anode_potential=anode_potential,
                    substrate_supply=inlet_conc / 10.0
                )
                
                # Add empty sensor data
                biofilm_state.update({
                    'sensor_validated_thickness': biofilm_state['biofilm_thickness'],
                    'sensor_validated_biomass': biofilm_state['biomass_density'],
                    'sensor_confidence': 0.0,
                    'fusion_confidence': 0.0,
                    'sensor_status': 'unavailable'
                })
            
            biofilm_states.append(biofilm_state)
            
            # Collect sensor measurements if available
            if SENSING_AVAILABLE and (self.enable_eis or self.enable_qcm):
                cell_sensor_data = self._collect_cell_sensor_data(
                    i, biofilm_state, self.time
                )
                sensor_measurements.append(cell_sensor_data)
            else:
                sensor_measurements.append({})
        
        # 3. Update metabolic dynamics
        metabolic_states = []
        for i in range(self.n_cells):
            # Use sensor-validated biofilm parameters if available
            biomass = biofilm_states[i].get('sensor_validated_biomass', 
                                           biofilm_states[i]['biomass_density'])
            growth_rate = biofilm_states[i]['growth_rate']
            
            # Step metabolic model
            metabolic_state = self.metabolic_models[i].step_metabolism(
                dt=dt,
                biomass=biomass,
                growth_rate=growth_rate,
                anode_potential=anode_potential,
                substrate_supply=inlet_conc / 20.0,
                cathode_o2_conc=0.25,  # mol/m³
                membrane_area=0.01,    # m²
                volume=0.1,           # L per cell
                electrode_area=0.01   # m²  
            )
            metabolic_states.append(metabolic_state)
        
        # 4. Enhanced Q-learning control with sensor data
        current_concentrations = [cell.substrate_concentration for cell in self.mfc_cells]
        outlet_conc = current_concentrations[-1] if current_concentrations else inlet_conc
        
        # Prepare base state for Q-learning
        avg_biofilm = np.mean([bs['biofilm_thickness'] for bs in biofilm_states])
        biofilm_deviation = abs(avg_biofilm - 30.0)  # 30 μm target
        substrate_utilization = ((inlet_conc - outlet_conc) / 
                               inlet_conc * 100 if inlet_conc > 0 else 0)
        
        base_state = self.flow_controller.discretize_enhanced_state(
            sum([0.35 * bs['biofilm_thickness'] * bs['biomass_density'] * 0.002 
                for bs in biofilm_states]),  # Total power estimate
            biofilm_deviation,
            substrate_utilization,
            self.reservoir.substrate_concentration,
            min(current_concentrations) if current_concentrations else 0,
            abs(outlet_conc - 12.0),  # Target outlet concentration
            self.time
        )
        
        # Prepare sensor data for Q-learning
        aggregated_sensor_data = self._aggregate_sensor_data(sensor_measurements)
        
        # Choose action with sensor enhancement
        if hasattr(self.flow_controller, 'choose_action_with_sensors'):
            action = self.flow_controller.choose_action_with_sensors(
                base_state, aggregated_sensor_data
            )
        else:
            # Fallback to standard action selection
            action = np.random.choice(len(self.flow_controller.actions))
        
        # Update flow rate
        flow_adjustment = self.flow_controller.actions[action]
        self.flow_rate_ml_h = max(5.0, min(50.0, self.flow_rate_ml_h + flow_adjustment))
        
        # 5. Update MFC cells with enhanced current calculations
        enhanced_currents = []
        cell_voltages = []
        
        for i in range(self.n_cells):
            cell = self.mfc_cells[i]
            
            # Process cell with monitoring
            if hasattr(cell, 'update_concentrations'):
                cell.update_concentrations(inlet_conc, self.flow_rate_ml_h, dt)
            
            # Calculate enhanced current with sensor validation
            base_current = getattr(cell, 'current', 0.001)  # Default current (1 mA)
            
            # Apply biofilm enhancement
            biofilm_current = self._calculate_biofilm_current_enhancement(
                biofilm_states[i], sensor_measurements[i]
            )
            
            # Apply metabolic enhancement
            metabolic_current = metabolic_states[i].fluxes.get('GSU_R004', 0.0) * 0.001
            
            # Total enhanced current
            total_current = base_current + biofilm_current + metabolic_current
            enhanced_currents.append(total_current)
            
            # Cell voltage (simplified)
            cell_voltage = 0.35  # Acetate-specific potential
            cell_voltages.append(cell_voltage)
        
        # 6. Update reservoir with recirculation
        self.reservoir.circulate_anolyte(
            flow_rate_ml_h=self.flow_rate_ml_h,
            stack_outlet_conc=outlet_conc,
            dt_hours=dt
        )
        
        # 7. Substrate addition control
        cell_concentrations = [cell.substrate_concentration for cell in self.mfc_cells]
        substrate_addition, halt_flag = self.substrate_controller.calculate_substrate_addition(
            outlet_conc=outlet_conc,
            reservoir_conc=self.reservoir.substrate_concentration,
            cell_concentrations=cell_concentrations,
            reservoir_sensors=self.reservoir.get_sensor_readings(),
            dt_hours=dt
        )
        
        self.reservoir.add_substrate(substrate_addition, dt)
        
        # 8. Update energy and power tracking
        total_power = sum(v * i for v, i in zip(cell_voltages, enhanced_currents))
        self.total_energy_generated += total_power * dt
        self.pump_power_consumed += 0.001 * self.flow_rate_ml_h * dt
        
        # 9. Calculate model validation metrics
        model_validation_score = self._calculate_model_validation_score(
            biofilm_states, sensor_measurements
        )
        
        # 10. Handle sensor faults if detected
        self._handle_sensor_faults(sensor_measurements)
        
        # 11. Update time and tracking
        self.time += dt
        
        # 12. Create sensor-integrated state
        integrated_state = self._create_sensor_integrated_state(
            biofilm_states, metabolic_states, sensor_measurements,
            enhanced_currents, cell_voltages, total_power, model_validation_score
        )
        
        # Store history
        self.history.append(integrated_state)
        self.sensor_history.append(sensor_measurements)
        
        return integrated_state
    
    def _collect_cell_sensor_data(self, cell_index: int, biofilm_state: Dict, 
                                 time_hours: float) -> Dict:
        """Collect sensor data for a specific cell."""
        sensor_data: Dict[str, Any] = {
            'eis': {},
            'qcm': {},
            'fusion': {}
        }
        
        # Collect EIS data
        if self.enable_eis and cell_index < len(self.eis_models):
            try:
                thickness = biofilm_state.get('sensor_validated_thickness', 
                                             biofilm_state['biofilm_thickness'])
                biomass = biofilm_state.get('sensor_validated_biomass',
                                          biofilm_state['biomass_density'])
                
                eis_measurements = self.eis_models[cell_index].simulate_measurement(
                    biofilm_thickness=thickness,
                    biomass_density=biomass,
                    time_hours=time_hours
                )
                
                eis_properties = self.eis_models[cell_index].get_biofilm_properties(eis_measurements)
                
                sensor_data['eis'] = {
                    'measurements': eis_measurements,
                    'properties': eis_properties,
                    'status': 'good' if eis_properties.get('measurement_quality', 0) > 0.7 else 'degraded'
                }
                
            except Exception as e:
                print(f"Warning: EIS measurement failed for cell {cell_index}: {e}")
                sensor_data['eis']['status'] = 'failed'
        
        # Collect QCM data
        if self.enable_qcm and cell_index < len(self.qcm_models):
            try:
                thickness = biofilm_state.get('sensor_validated_thickness',
                                             biofilm_state['biofilm_thickness'])
                biomass = biofilm_state.get('sensor_validated_biomass',
                                          biofilm_state['biomass_density'])
                
                # Estimate mass from biomass and thickness
                electrode_area = 0.196  # cm²
                thickness_cm = thickness * 1e-4
                volume_cm3 = electrode_area * thickness_cm
                mass_ug = biomass * volume_cm3 * 1e3  # g/L to μg
                
                qcm_measurement = self.qcm_models[cell_index].simulate_measurement(
                    biofilm_mass=mass_ug,
                    biofilm_thickness=thickness,
                    time_hours=time_hours
                )
                
                qcm_properties = self.qcm_models[cell_index].estimate_biofilm_properties(qcm_measurement)
                
                sensor_data['qcm'] = {
                    'measurement': qcm_measurement,
                    'properties': qcm_properties,
                    'status': 'good' if qcm_properties.get('measurement_quality', 0) > 0.7 else 'degraded'
                }
                
            except Exception as e:
                print(f"Warning: QCM measurement failed for cell {cell_index}: {e}")
                sensor_data['qcm']['status'] = 'failed'
        
        # Perform sensor fusion if both sensors available
        if (sensor_data['eis'].get('measurements') and sensor_data['qcm'].get('measurement') and
            cell_index < len(self.sensor_fusion_models)):
            try:
                fused_result = self.sensor_fusion_models[cell_index].fuse_measurements(
                    eis_measurement=sensor_data['eis']['measurements'][0],
                    qcm_measurement=sensor_data['qcm']['measurement'],
                    eis_properties=sensor_data['eis']['properties'],
                    qcm_properties=sensor_data['qcm']['properties'],
                    time_hours=time_hours
                )
                
                sensor_data['fusion'] = {
                    'fused_result': fused_result,
                    'thickness': fused_result.thickness_um,
                    'biomass': fused_result.biomass_density_g_per_L,
                    'sensor_agreement': fused_result.sensor_agreement,
                    'fusion_confidence': fused_result.fusion_confidence,
                    'status': 'good' if fused_result.fusion_confidence > 0.7 else 'degraded'
                }
                
            except Exception as e:
                print(f"Warning: Sensor fusion failed for cell {cell_index}: {e}")
                sensor_data['fusion']['status'] = 'failed'
        
        return sensor_data
    
    def _aggregate_sensor_data(self, sensor_measurements: List[Dict]) -> Dict:
        """Aggregate sensor data from all cells for Q-learning."""
        if not sensor_measurements or not any(sensor_measurements):
            return {}
        
        # Aggregate EIS data
        eis_thicknesses = []
        eis_conductivities = []
        eis_qualities = []
        
        # Aggregate QCM data
        qcm_masses = []
        qcm_freq_shifts = []
        qcm_dissipations = []
        
        # Aggregate fusion data
        sensor_agreements = []
        fusion_confidences = []
        
        for sensor_data in sensor_measurements:
            # EIS data
            if 'eis' in sensor_data and 'properties' in sensor_data['eis']:
                eis_props = sensor_data['eis']['properties']
                eis_thicknesses.append(eis_props.get('thickness_um', 0))
                eis_conductivities.append(eis_props.get('conductivity_S_per_m', 0))
                eis_qualities.append(eis_props.get('measurement_quality', 0))
            
            # QCM data
            if 'qcm' in sensor_data and 'properties' in sensor_data['qcm']:
                qcm_props = sensor_data['qcm']['properties']
                qcm_masses.append(qcm_props.get('mass_per_area_ng_per_cm2', 0))
                
                if 'measurement' in sensor_data['qcm']:
                    qcm_meas = sensor_data['qcm']['measurement']
                    qcm_freq_shifts.append(abs(qcm_meas.frequency_shift))
                    qcm_dissipations.append(qcm_meas.dissipation)
            
            # Fusion data
            if 'fusion' in sensor_data and 'sensor_agreement' in sensor_data['fusion']:
                sensor_agreements.append(sensor_data['fusion']['sensor_agreement'])
                fusion_confidences.append(sensor_data['fusion']['fusion_confidence'])
        
        # Calculate aggregated metrics
        aggregated = {}
        
        if eis_thicknesses:
            aggregated['eis'] = {
                'thickness_um': np.mean(eis_thicknesses),
                'conductivity_S_per_m': np.mean(eis_conductivities),
                'measurement_quality': np.mean(eis_qualities),
                'status': 'good' if np.mean(eis_qualities) > 0.7 else 'degraded'
            }
        
        if qcm_masses:
            aggregated['qcm'] = {
                'mass_per_area_ng_per_cm2': np.mean(qcm_masses),
                'frequency_shift_Hz': np.mean(qcm_freq_shifts) if qcm_freq_shifts else 0,
                'dissipation': np.mean(qcm_dissipations) if qcm_dissipations else 0,
                'measurement_quality': 0.8,  # Simplified
                'status': 'good'
            }
        
        if sensor_agreements:
            aggregated['fusion'] = {
                'sensor_agreement': np.mean(sensor_agreements),
                'fusion_confidence': np.mean(fusion_confidences),
                'status': 'good' if np.mean(fusion_confidences) > 0.7 else 'degraded'
            }
        
        return aggregated
    
    def _calculate_biofilm_current_enhancement(self, biofilm_state: Dict, 
                                             sensor_data: Dict) -> float:
        """Calculate biofilm current enhancement based on sensor data."""
        # Use sensor-validated thickness if available
        if 'fusion' in sensor_data and 'thickness' in sensor_data['fusion']:
            thickness = sensor_data['fusion']['thickness']
            confidence = sensor_data['fusion'].get('fusion_confidence', 0.5)
        elif 'eis' in sensor_data and 'properties' in sensor_data['eis']:
            thickness = sensor_data['eis']['properties'].get('thickness_um', 0)
            confidence = sensor_data['eis']['properties'].get('measurement_quality', 0.5)
        else:
            thickness = biofilm_state['biofilm_thickness']
            confidence = 0.5
        
        # Calculate current density based on validated thickness
        base_current_density = 0.0001  # A/cm²
        thickness_factor = min(thickness / 30.0, 2.0)  # Normalize to 30 μm optimal
        confidence_factor = 0.5 + 0.5 * confidence  # Scale by sensor confidence
        
        enhanced_current = base_current_density * thickness_factor * confidence_factor
        
        return enhanced_current
    
    def _calculate_model_validation_score(self, biofilm_states: List[Dict],
                                        sensor_measurements: List[Dict]) -> float:
        """Calculate model validation score based on sensor agreement."""
        if not sensor_measurements or not any(sensor_measurements):
            return 0.5  # Neutral score without sensors
        
        validation_scores = []
        
        for i, (biofilm_state, sensor_data) in enumerate(zip(biofilm_states, sensor_measurements)):
            model_thickness = biofilm_state['biofilm_thickness']
            model_biomass = biofilm_state['biomass_density']
            
            # Compare with sensor measurements
            if 'fusion' in sensor_data and 'thickness' in sensor_data['fusion']:
                sensor_thickness = sensor_data['fusion']['thickness']
                sensor_biomass = sensor_data['fusion']['biomass']
                
                # Calculate relative errors
                thickness_error = abs(model_thickness - sensor_thickness) / max(sensor_thickness, 1.0)
                biomass_error = abs(model_biomass - sensor_biomass) / max(sensor_biomass, 1.0)
                
                # Convert to validation score (lower error = higher score)
                thickness_score = max(0, 1.0 - thickness_error)
                biomass_score = max(0, 1.0 - biomass_error)
                
                cell_score = (thickness_score + biomass_score) / 2
                validation_scores.append(cell_score)
        
        return np.mean(validation_scores) if validation_scores else 0.5
    
    def _handle_sensor_faults(self, sensor_measurements: List[Dict]):
        """Handle detected sensor faults."""
        for i, sensor_data in enumerate(sensor_measurements):
            # Check EIS faults
            if 'eis' in sensor_data and sensor_data['eis'].get('status') == 'failed':
                if hasattr(self.flow_controller, 'handle_sensor_fault'):
                    self.flow_controller.handle_sensor_fault('failed', 'eis')
                self.sensor_fault_history.append({
                    'time': self.time,
                    'cell': i,
                    'sensor': 'eis',
                    'fault_type': 'failed'
                })
            
            # Check QCM faults
            if 'qcm' in sensor_data and sensor_data['qcm'].get('status') == 'failed':
                if hasattr(self.flow_controller, 'handle_sensor_fault'):
                    self.flow_controller.handle_sensor_fault('failed', 'qcm')
                self.sensor_fault_history.append({
                    'time': self.time,
                    'cell': i,
                    'sensor': 'qcm',
                    'fault_type': 'failed'
                })
    
    def _create_sensor_integrated_state(self, biofilm_states: List[Dict],
                                      metabolic_states: List, sensor_measurements: List[Dict],
                                      enhanced_currents: List[float], cell_voltages: List[float],
                                      total_power: float, model_validation_score: float) -> SensorIntegratedMFCState:
        """Create comprehensive sensor-integrated state."""
        # Extract sensor data
        eis_thickness = []
        eis_conductivity = []
        eis_quality = []
        eis_status = []
        
        qcm_mass = []
        qcm_freq_shift = []
        qcm_dissipation = []
        qcm_status = []
        
        fused_thickness = []
        fused_biomass = []
        sensor_agreement = []
        fusion_confidence = []
        sensor_faults = []
        
        thickness_errors = []
        biomass_errors = []
        
        for i, (biofilm_state, sensor_data) in enumerate(zip(biofilm_states, sensor_measurements)):
            # EIS data
            if 'eis' in sensor_data and 'properties' in sensor_data['eis']:
                eis_props = sensor_data['eis']['properties']
                eis_thickness.append(eis_props.get('thickness_um', 0))
                eis_conductivity.append(eis_props.get('conductivity_S_per_m', 0))
                eis_quality.append(eis_props.get('measurement_quality', 0))
                eis_status.append(sensor_data['eis'].get('status', 'unavailable'))
            else:
                eis_thickness.append(0)
                eis_conductivity.append(0)
                eis_quality.append(0)
                eis_status.append('unavailable')
            
            # QCM data
            if 'qcm' in sensor_data and 'measurement' in sensor_data['qcm']:
                qcm_meas = sensor_data['qcm']['measurement']
                qcm_props = sensor_data['qcm']['properties']
                qcm_mass.append(qcm_props.get('mass_per_area_ng_per_cm2', 0))
                qcm_freq_shift.append(qcm_meas.frequency_shift)
                qcm_dissipation.append(qcm_meas.dissipation)
                qcm_status.append(sensor_data['qcm'].get('status', 'unavailable'))
            else:
                qcm_mass.append(0)
                qcm_freq_shift.append(0)
                qcm_dissipation.append(0)
                qcm_status.append('unavailable')
            
            # Fusion data
            if 'fusion' in sensor_data and 'thickness' in sensor_data['fusion']:
                fused_thickness.append(sensor_data['fusion']['thickness'])
                fused_biomass.append(sensor_data['fusion']['biomass'])
                sensor_agreement.append(sensor_data['fusion']['sensor_agreement'])
                fusion_confidence.append(sensor_data['fusion']['fusion_confidence'])
            else:
                fused_thickness.append(biofilm_state['biofilm_thickness'])
                fused_biomass.append(biofilm_state['biomass_density'])
                sensor_agreement.append(0.5)
                fusion_confidence.append(0.0)
            
            # Prediction errors
            model_thickness = biofilm_state['biofilm_thickness']
            model_biomass = biofilm_state['biomass_density']
            sensor_thickness = fused_thickness[-1]
            sensor_biomass = fused_biomass[-1]
            
            thickness_errors.append(abs(model_thickness - sensor_thickness))
            biomass_errors.append(abs(model_biomass - sensor_biomass))
            
            # Sensor faults
            has_fault = (eis_status[-1] == 'failed' or qcm_status[-1] == 'failed')
            sensor_faults.append(has_fault)
        
        # Create state
        return SensorIntegratedMFCState(
            # Base state
            time=self.time,
            total_energy=self.total_energy_generated,
            average_power=self.total_energy_generated / (self.time + 1e-6),
            coulombic_efficiency=np.mean([ms.coulombic_efficiency for ms in metabolic_states]),
            biofilm_thickness=[bs['biofilm_thickness'] for bs in biofilm_states],
            biomass_density=[bs['biomass_density'] for bs in biofilm_states],
            attachment_fraction=[0.5] * self.n_cells,  # Simplified
            substrate_concentration=[ms.metabolites[self.substrate] for ms in metabolic_states],
            nadh_ratio=[ms.metabolites['nadh']/(ms.metabolites['nadh']+ms.metabolites['nad_plus']) 
                       for ms in metabolic_states],
            atp_level=[ms.metabolites['atp'] for ms in metabolic_states],
            electron_flux=[ms.electron_production for ms in metabolic_states],
            cell_voltages=cell_voltages,
            current_densities=[c/0.01 for c in enhanced_currents],
            anode_potentials=[-0.3] * self.n_cells,
            reservoir_concentration=self.reservoir.substrate_concentration,
            flow_rate=self.flow_rate_ml_h,
            pump_power=self.pump_power_consumed,
            epsilon=self.flow_controller.epsilon,
            q_table_size=len(self.flow_controller.q_table),
            learning_progress=1.0 - self.flow_controller.epsilon/0.3,
            
            # Sensor measurements
            eis_thickness=eis_thickness,
            eis_conductivity=eis_conductivity,
            eis_measurement_quality=eis_quality,
            qcm_mass_per_area=qcm_mass,
            qcm_frequency_shift=qcm_freq_shift,
            qcm_dissipation=qcm_dissipation,
            
            # Sensor fusion results
            fused_thickness=fused_thickness,
            fused_biomass=fused_biomass,
            sensor_agreement=sensor_agreement,
            fusion_confidence=fusion_confidence,
            
            # Sensor status
            eis_sensor_status=eis_status,
            qcm_sensor_status=qcm_status,
            sensor_fault_flags=sensor_faults,
            
            # Model validation
            thickness_prediction_error=thickness_errors,
            biomass_prediction_error=biomass_errors,
            model_sensor_agreement=np.mean(sensor_agreement),
            adaptive_calibration_active=len(self.adaptive_calibration_events) > 0
        )
    
    def run_sensor_integrated_simulation(self, dt: float = 1.0, 
                                       save_interval: int = 10) -> Dict[str, Any]:
        """
        Run complete sensor-integrated simulation.
        
        Args:
            dt: Time step (hours)
            save_interval: Save results every N hours
            
        Returns:
            Dictionary with enhanced simulation results
        """
        print("\
Starting Sensor-Integrated MFC Simulation")
        print(f"Duration: {self.simulation_hours} hours")
        print(f"Time step: {dt} hours")
        print(f"Sensors: EIS={self.enable_eis}, QCM={self.enable_qcm}")
        
        start_time = time.time()
        
        # Main simulation loop
        for hour in range(int(self.simulation_hours / dt)):
            # Step sensor-integrated dynamics
            state = self.step_sensor_integrated_dynamics(dt)
            
            # Progress update
            if hour % 10 == 0:
                sensor_info = f"Sensor Agreement={state.model_sensor_agreement:.2%}" if SENSING_AVAILABLE else "No Sensors"
                print(f"Hour {self.time:.1f}: Power={state.average_power:.3f}W, "
                      f"CE={state.coulombic_efficiency:.2%}, "
                      f"Biofilm={np.mean(state.biofilm_thickness):.1f}μm, "
                      f"{sensor_info}")
            
            # Save checkpoint
            if hour % save_interval == 0 and hour > 0:
                self._save_sensor_checkpoint(hour)
        
        # Final statistics
        computation_time = time.time() - start_time
        
        results = self._compile_sensor_results()
        results['computation_time'] = computation_time
        
        print("\
Sensor-Integrated Simulation Complete!")
        print(f"Total Energy: {results['total_energy']:.2f} Wh")
        print(f"Average Power: {results['average_power']:.3f} W")
        print(f"Model Validation Score: {results.get('average_model_validation', 0):.2%}")
        if SENSING_AVAILABLE:
            print(f"Sensor Integration: {results.get('sensor_integration_score', 0):.2%}")
        print(f"Computation Time: {computation_time:.1f} seconds")
        
        return results
    
    def _save_sensor_checkpoint(self, hour: int):
        """Save sensor-enhanced checkpoint data."""
        checkpoint = {
            'hour': hour,
            'time': self.time,
            'history': self.history[-100:],
            'sensor_history': self.sensor_history[-100:],
            'sensor_fault_history': self.sensor_fault_history,
            'adaptive_calibration_events': self.adaptive_calibration_events,
            'q_table': dict(self.flow_controller.q_table) if hasattr(self.flow_controller, 'q_table') else {},
            'performance_metrics': self.performance_metrics
        }
        
        filename = get_model_path(f'sensor_integrated_checkpoint_h{hour}.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    def _compile_sensor_results(self) -> Dict[str, Any]:
        """Compile sensor-enhanced simulation results."""
        if not self.history:
            return {}
        
        # Base results
        results = super()._compile_results() if hasattr(super(), '_compile_results') else {}
        
        # Add sensor-specific results
        if SENSING_AVAILABLE and self.history:
            # Extract sensor data from history
            sensor_agreements = [getattr(s, 'model_sensor_agreement', 0.5) for s in self.history]
            fusion_confidences = [np.mean(getattr(s, 'fusion_confidence', [0.5])) for s in self.history]
            thickness_errors = [np.mean(getattr(s, 'thickness_prediction_error', [0])) for s in self.history]
            
            results.update({
                'sensor_integration_score': np.mean(sensor_agreements),
                'average_fusion_confidence': np.mean(fusion_confidences),
                'average_thickness_error': np.mean(thickness_errors),
                'sensor_fault_count': len(self.sensor_fault_history),
                'adaptive_calibration_events': len(self.adaptive_calibration_events),
                'average_model_validation': np.mean(sensor_agreements)
            })
        
        # Controller performance
        if hasattr(self.flow_controller, 'get_controller_performance_summary'):
            controller_performance = self.flow_controller.get_controller_performance_summary()
            results['controller_performance'] = controller_performance
        
        return results
    
    def get_sensor_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive sensor diagnostics."""
        diagnostics = {
            'sensor_availability': {
                'sensing_models_available': SENSING_AVAILABLE,
                'eis_enabled': self.enable_eis,
                'qcm_enabled': self.enable_qcm,
                'sensor_fusion_enabled': self.enable_eis and self.enable_qcm,
                'n_eis_models': len(self.eis_models) if hasattr(self, 'eis_models') else 0,
                'n_qcm_models': len(self.qcm_models) if hasattr(self, 'qcm_models') else 0,
                'n_fusion_models': len(self.sensor_fusion_models) if hasattr(self, 'sensor_fusion_models') else 0
            },
            'sensor_performance': {
                'total_measurements': len(self.sensor_history),
                'fault_events': len(self.sensor_fault_history),
                'calibration_events': len(self.adaptive_calibration_events)
            }
        }
        
        # Add individual sensor diagnostics
        if hasattr(self, 'biofilm_models'):
            for i, model in enumerate(self.biofilm_models):
                if hasattr(model, 'get_sensor_diagnostics'):
                    diagnostics[f'cell_{i}_biofilm_sensor_diagnostics'] = model.get_sensor_diagnostics()
        
        # Add controller diagnostics
        if hasattr(self.flow_controller, 'get_controller_performance_summary'):
            diagnostics['controller_diagnostics'] = self.flow_controller.get_controller_performance_summary()
        
        return diagnostics


def main():
    """Main function to run sensor-integrated simulation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sensor-Integrated MFC Model Simulation')
    parser.add_argument('--cells', type=int, default=5, help='Number of cells')
    parser.add_argument('--species', choices=['geobacter', 'shewanella', 'mixed'], 
                       default='mixed', help='Bacterial species')
    parser.add_argument('--substrate', choices=['acetate', 'lactate'], 
                       default='lactate', help='Substrate type')
    parser.add_argument('--hours', type=int, default=100, help='Simulation duration')
    parser.add_argument('--gpu', action='store_true', help='Enable GPU acceleration')
    parser.add_argument('--eis', action='store_true', default=True, help='Enable EIS sensors')
    parser.add_argument('--qcm', action='store_true', default=True, help='Enable QCM sensors')
    parser.add_argument('--fusion', choices=['kalman_filter', 'weighted_average', 'maximum_likelihood', 'bayesian'],
                       default='kalman_filter', help='Sensor fusion method')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    
    args = parser.parse_args()
    
    # Create and run model
    model = SensorIntegratedMFCModel(
        n_cells=args.cells,
        species=args.species,
        substrate=args.substrate,
        use_gpu=args.gpu,
        simulation_hours=args.hours,
        enable_eis=args.eis,
        enable_qcm=args.qcm,
        sensor_fusion_method=args.fusion
    )
    
    # Run simulation
    results = model.run_sensor_integrated_simulation()
    
    # Save results
    if hasattr(model, 'save_results'):
        model.save_results(results, prefix="sensor_integrated")
    
    # Generate plots
    if args.plot and hasattr(model, 'plot_results'):
        model.plot_results(results)
    
    # Print diagnostics
    diagnostics = model.get_sensor_diagnostics()
    print("\
=== Sensor Diagnostics ===")
    for key, value in diagnostics.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()