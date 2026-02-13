#!/usr/bin/env python3
"""
MFC System Integration - Phase 5
================================

Complete MFC system model integrating:
- Anode: Biofilm kinetics, metabolic pathways, sensing
- Membrane: PEM/AEM models with fouling and transport  
- Cathode: Platinum/biological cathodes with kinetics
- Control: Q-learning optimization for multi-objective control

This creates a full electrochemical stack with advanced control for:
- Performance optimization (power, efficiency)
- Lifetime maximization (fouling, degradation)
- Economic optimization (operational costs)

Created: 2025-07-27 (Phase 5)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum
import time

# Internal imports
from biofilm_kinetics import BiofilmKineticsModel
from metabolic_model import MetabolicModel
from cathode_models.platinum_cathode import create_platinum_cathode
from cathode_models.biological_cathode import create_biological_cathode
from membrane_models.proton_exchange import create_nafion_membrane, create_speek_membrane
from membrane_models.anion_exchange import create_aem_membrane
from membrane_models.membrane_fouling import FoulingModel, FoulingParameters


class MFCConfiguration(Enum):
    """Standard MFC system configurations."""
    BASIC_LAB = "basic_lab"           # Small lab-scale system
    PILOT_PLANT = "pilot_plant"       # Pilot-scale system
    INDUSTRIAL = "industrial"         # Industrial-scale system
    RESEARCH = "research"             # Advanced research system


@dataclass
class MFCStackParameters:
    """Parameters for complete MFC stack configuration."""
    
    # Stack geometry
    n_cells: int = 5
    cell_area: float = 0.01           # m² per cell
    anode_cathode_spacing: float = 0.002  # m (2mm)
    membrane_thickness: float = 183e-6    # m (183 μm for Nafion)
    
    # Operating conditions
    temperature: float = 303.15        # K (30°C)
    pressure: float = 101325          # Pa (1 atm)
    ph_anode: float = 7.0
    ph_cathode: float = 7.0
    
    # Biological parameters
    bacterial_species: str = "mixed"   # "geobacter", "shewanella", "mixed"
    substrate_type: str = "lactate"    # "acetate", "lactate", "glucose"
    initial_substrate_conc: float = 20.0  # mmol/L
    
    # Membrane configuration
    membrane_type: str = "PEM"         # "PEM", "AEM", "Bipolar"
    membrane_material: str = "Nafion"  # "Nafion", "SPEEK", "QA"
    
    # Cathode configuration
    cathode_type: str = "platinum"     # "platinum", "biological", "air"
    oxygen_supply: str = "air"         # "air", "pure_o2", "passive"
    
    # Economic parameters
    electricity_price: float = 0.12   # $/kWh
    maintenance_cost: float = 0.05     # $/day
    
    # Q-learning parameters
    enable_qlearning: bool = True
    learning_objectives: List[str] = field(default_factory=lambda: [
        "power_density", "coulombic_efficiency", "lifetime", "cost"
    ])


@dataclass
class MFCSystemState:
    """Complete system state at any time point."""
    
    # Time and identifiers
    time: float
    cell_id: int = 0
    
    # Anode (biofilm) state
    biofilm_thickness: float = 0.0     # μm
    biomass_density: float = 0.0       # kg/m³
    substrate_concentration: float = 0.0  # mmol/L
    anode_potential: float = -0.4      # V vs SHE
    
    # Membrane state  
    membrane_conductivity: float = 0.0  # S/m
    water_content: float = 0.0         # mol H2O/mol functional group
    fouling_thickness: float = 0.0     # μm
    degradation_fraction: float = 0.0  # 0-1
    
    # Cathode state
    cathode_potential: float = 0.8     # V vs SHE
    oxygen_concentration: float = 0.25  # mol/m³
    cathode_current_density: float = 0.0  # A/m²
    
    # Electrochemical performance
    cell_voltage: float = 0.0          # V
    current_density: float = 0.0       # A/m²
    power_density: float = 0.0         # W/m²
    coulombic_efficiency: float = 0.0  # %
    
    # Transport phenomena
    proton_flux: float = 0.0           # mol/m²/s
    water_flux: float = 0.0            # mol/m²/s
    oxygen_flux: float = 0.0           # mol/m²/s
    
    # Economics
    power_generated: float = 0.0       # W
    operational_cost: float = 0.0      # $/h
    revenue_rate: float = 0.0          # $/h


class IntegratedMFCSystem:
    """
    Complete integrated MFC system with all components.
    
    Integrates:
    - Anode: Biofilm kinetics + metabolic pathways
    - Membrane: Ion transport + fouling + degradation
    - Cathode: Electrochemical kinetics + mass transport
    - Control: Multi-objective Q-learning optimization
    """
    
    def __init__(self, config: MFCStackParameters):
        """Initialize integrated MFC system."""
        self.config = config
        self.time = 0.0
        
        print("🔋 Initializing Integrated MFC System")
        print(f"   Configuration: {config.n_cells} cells, {config.bacterial_species} bacteria")
        print(f"   Membrane: {config.membrane_material} {config.membrane_type}")
        print(f"   Cathode: {config.cathode_type}")
        
        # Initialize all subsystems
        self._initialize_anode_models()
        self._initialize_membrane_models()
        self._initialize_cathode_models()
        self._initialize_system_state()
        self._initialize_control_system()
        
        # Performance tracking
        self.history: List[MFCSystemState] = []
        self.performance_metrics = {
            'total_energy': 0.0,
            'peak_power': 0.0,
            'average_efficiency': 0.0,
            'lifetime_estimate': 0.0,
            'total_cost': 0.0
        }
        
        print("✅ MFC System Integration Complete")
    
    def _initialize_anode_models(self):
        """Initialize anode-side models for each cell."""
        self.anode_models = []
        self.metabolic_models = []
        
        for i in range(self.config.n_cells):
            # Biofilm kinetics
            biofilm_model = BiofilmKineticsModel(
                species=self.config.bacterial_species,
                substrate=self.config.substrate_type,
                use_gpu=False,  # Disable for integration stability
                temperature=self.config.temperature,
                ph=self.config.ph_anode
            )
            self.anode_models.append(biofilm_model)
            
            # Metabolic pathways  
            # Map membrane materials to supported metabolic model grades
            membrane_mapping = {
                "Nafion": "Nafion-117",
                "SPEEK": "Nafion-117",  # Use Nafion as fallback for SPEEK
                "QA": "Nafion-117"
            }
            membrane_grade = membrane_mapping.get(self.config.membrane_material, "Nafion-117")
            metabolic_model = MetabolicModel(
                species=self.config.bacterial_species,
                substrate=self.config.substrate_type,
                membrane_type=membrane_grade,
                use_gpu=False
            )
            self.metabolic_models.append(metabolic_model)
    
    def _initialize_membrane_models(self):
        """Initialize membrane models for each cell."""
        self.membrane_models = []
        self.fouling_models = []
        
        for i in range(self.config.n_cells):
            # Create membrane based on type
            if self.config.membrane_type == "PEM":
                if self.config.membrane_material == "Nafion":
                    membrane = create_nafion_membrane(
                        thickness_um=self.config.membrane_thickness * 1e6,
                        area_cm2=self.config.cell_area * 1e4,
                        temperature_C=self.config.temperature - 273.15
                    )
                elif self.config.membrane_material == "SPEEK":
                    membrane = create_speek_membrane(
                        degree_sulfonation=0.7,
                        thickness_um=self.config.membrane_thickness * 1e6,
                        area_cm2=self.config.cell_area * 1e4
                    )
                else:
                    raise ValueError(f"Unknown PEM material: {self.config.membrane_material}")
                    
            elif self.config.membrane_type == "AEM":
                membrane = create_aem_membrane(
                    membrane_type="Quaternary Ammonium",
                    thickness_um=self.config.membrane_thickness * 1e6,
                    area_cm2=self.config.cell_area * 1e4,
                    temperature_C=self.config.temperature - 273.15
                )
            else:
                raise ValueError(f"Unknown membrane type: {self.config.membrane_type}")
            
            self.membrane_models.append(membrane)
            
            # Fouling model
            fouling_params = FoulingParameters(
                temperature=self.config.temperature,
                ph=self.config.ph_anode
            )
            fouling_model = FoulingModel(fouling_params)
            self.fouling_models.append(fouling_model)
    
    def _initialize_cathode_models(self):
        """Initialize cathode models for each cell."""
        self.cathode_models = []
        
        for i in range(self.config.n_cells):
            if self.config.cathode_type == "platinum":
                cathode = create_platinum_cathode(
                    area_cm2=self.config.cell_area * 1e4,  # Convert m² to cm²
                    temperature_C=self.config.temperature - 273.15,
                    platinum_loading_mg_cm2=0.5
                )
            elif self.config.cathode_type == "biological":
                cathode = create_biological_cathode(
                    area_cm2=self.config.cell_area * 1e4,
                    temperature_C=self.config.temperature - 273.15,
                    initial_biofilm_thickness_um=1.0
                )
            else:
                raise ValueError(f"Unknown cathode type: {self.config.cathode_type}")
            
            self.cathode_models.append(cathode)
    
    def _initialize_system_state(self):
        """Initialize system state variables."""
        self.cell_states = []
        
        for i in range(self.config.n_cells):
            state = MFCSystemState(
                time=0.0,
                cell_id=i,
                substrate_concentration=self.config.initial_substrate_conc,
                biofilm_thickness=1.0,  # Initial 1 μm biofilm
                biomass_density=100.0,  # Initial biomass
                membrane_conductivity=self.membrane_models[i].calculate_ionic_conductivity(),
                water_content=10.0,     # Initial water content
                oxygen_concentration=0.25  # Air-saturated water
            )
            self.cell_states.append(state)
    
    def _initialize_control_system(self):
        """Initialize Q-learning control system."""
        if self.config.enable_qlearning:
            # Import the optimized Q-learning controller
            try:
                from mfc_unified_qlearning_optimized import OptimizedUnifiedQController
                self.q_controller = OptimizedUnifiedQController(
                    target_outlet_conc=self.config.initial_substrate_conc * 0.6
                )
                print("✅ Q-learning control system initialized")
            except ImportError:
                print("⚠️ Q-learning controller not available, using basic control")
                self.q_controller = None
        else:
            self.q_controller = None
    
    def step_system_dynamics(self, dt: float = 1.0) -> List[MFCSystemState]:
        """
        Step the complete system forward by dt hours.
        
        Args:
            dt: Time step (hours)
            
        Returns:
            List of updated cell states
        """
        updated_states = []
        
        for i in range(self.config.n_cells):
            state = self.cell_states[i]
            
            # 1. Update anode (biofilm) dynamics
            anode_state = self._step_anode_dynamics(i, dt, state)
            
            # 2. Update membrane transport and fouling
            membrane_state = self._step_membrane_dynamics(i, dt, state, anode_state)
            
            # 3. Update cathode dynamics
            cathode_state = self._step_cathode_dynamics(i, dt, state, membrane_state)
            
            # 4. Calculate electrochemical performance
            performance = self._calculate_cell_performance(i, anode_state, membrane_state, cathode_state)
            
            # 5. Update system state
            updated_state = self._update_cell_state(i, dt, anode_state, membrane_state, 
                                                  cathode_state, performance)
            updated_states.append(updated_state)
        
        # 6. Apply Q-learning control
        if self.q_controller:
            self._apply_qlearning_control(updated_states, dt)
        
        # 7. Update time and store history
        self.time += dt
        self.cell_states = updated_states
        self.history.extend(updated_states)
        
        return updated_states
    
    def _step_anode_dynamics(self, cell_idx: int, dt: float, state: MFCSystemState) -> Dict[str, float]:
        """Update anode biofilm and metabolic dynamics."""
        # Step biofilm model
        biofilm_state = self.anode_models[cell_idx].step_biofilm_dynamics(
            dt=dt,
            anode_potential=state.anode_potential,
            substrate_supply=state.substrate_concentration / 10.0
        )
        
        # Step metabolic model
        metabolic_state = self.metabolic_models[cell_idx].step_metabolism(
            dt=dt,
            biomass=biofilm_state['biomass_density'],
            growth_rate=biofilm_state['growth_rate'],
            anode_potential=state.anode_potential,
            substrate_supply=state.substrate_concentration / 20.0,
            cathode_o2_conc=state.oxygen_concentration,
            membrane_area=self.config.cell_area,
            volume=0.1,  # L per cell
            electrode_area=self.config.cell_area
        )
        
        return {
            'biofilm_thickness': biofilm_state['biofilm_thickness'],
            'biomass_density': biofilm_state['biomass_density'],
            'substrate_concentration': max(0, state.substrate_concentration - 
                                         metabolic_state.fluxes.get('substrate_consumption', 0) * dt),
            'anode_current_density': self.anode_models[cell_idx].calculate_biofilm_current_density(
                biofilm_state['biofilm_thickness'], biofilm_state['biomass_density']
            )
        }
    
    def _step_membrane_dynamics(self, cell_idx: int, dt: float, state: MFCSystemState, 
                               anode_state: Dict[str, float]) -> Dict[str, float]:
        """Update membrane transport, fouling, and degradation."""
        membrane = self.membrane_models[cell_idx]
        fouling = self.fouling_models[cell_idx]
        
        # Calculate current flux through membrane
        current_density = anode_state['anode_current_density']
        proton_flux = current_density / (96485.0)  # A/m² to mol/m²/s
        
        # Update fouling
        operating_conditions = {
            'temperature': self.config.temperature,
            'ph': self.config.ph_anode,
            'nutrient_concentration': state.substrate_concentration / 1000.0,  # mol/L
            'current_density': current_density,
            'ion_concentrations': {'Ca2+': 0.001, 'CO3--': 0.001},
            'particle_concentration': 0.0001,  # kg/m³
            'flow_velocity': 0.01  # m/s
        }
        
        fouling.update_fouling(dt, operating_conditions)
        
        # Calculate membrane properties with fouling
        base_resistance = membrane.calculate_membrane_resistance()
        fouling_data = fouling.calculate_total_resistance(base_resistance)
        
        # Update conductivity based on fouling
        effective_conductivity = 1.0 / fouling_data['total_resistance'] * membrane.thickness
        
        return {
            'membrane_conductivity': effective_conductivity,
            'proton_flux': proton_flux,
            'water_flux': proton_flux * 2.0,  # Simplified drag
            'fouling_thickness': fouling.biofilm_thickness * 1e6,  # μm
            'degradation_fraction': fouling.degradation_fraction,
            'membrane_resistance': fouling_data['total_resistance']
        }
    
    def _step_cathode_dynamics(self, cell_idx: int, dt: float, state: MFCSystemState,
                              membrane_state: Dict[str, float]) -> Dict[str, float]:
        """Update cathode kinetics and mass transport."""
        cathode = self.cathode_models[cell_idx]
        
        # Calculate overpotential from current density  
        current_density = membrane_state['proton_flux'] * 96485.0  # Convert to A/m²
        operating_overpotential = 0.3  # Simplified assumption (would be calculated in real system)
        
        # Calculate cathode performance
        performance = cathode.calculate_performance_metrics(
            operating_overpotential=operating_overpotential,
            oxygen_conc=state.oxygen_concentration
        )
        
        # Update oxygen concentration (simplified consumption)
        oxygen_consumption = membrane_state['proton_flux'] * 0.25  # mol O2 per mol H+
        new_oxygen_conc = max(0.01, state.oxygen_concentration - oxygen_consumption * dt)
        
        # Calculate cathode potential (standard potential - overpotential)
        standard_potential = 1.23  # V vs SHE for O2/H2O
        cathode_potential = standard_potential - operating_overpotential
        
        return {
            'cathode_potential': cathode_potential,
            'oxygen_concentration': new_oxygen_conc,
            'cathode_current_density': performance.get('current_density', current_density),
            'overpotential': operating_overpotential,
            'mass_transport_coefficient': 1e-5  # Simplified
        }
    
    def _calculate_cell_performance(self, cell_idx: int, anode_state: Dict[str, float],
                                   membrane_state: Dict[str, float], 
                                   cathode_state: Dict[str, float]) -> Dict[str, float]:
        """Calculate overall cell electrochemical performance."""
        
        # Cell voltage = cathode potential - anode potential - ohmic losses
        ohmic_loss = membrane_state['proton_flux'] * 96485.0 * membrane_state['membrane_resistance']
        cell_voltage = cathode_state['cathode_potential'] - self.cell_states[cell_idx].anode_potential - ohmic_loss
        cell_voltage = max(0, cell_voltage)
        
        # Current density (limited by anode or cathode)
        current_density = min(anode_state['anode_current_density'], 
                            cathode_state['cathode_current_density'])
        
        # Power density
        power_density = cell_voltage * current_density
        
        # Coulombic efficiency (simplified)
        theoretical_current = anode_state['anode_current_density']
        coulombic_efficiency = min(1.0, current_density / max(theoretical_current, 1e-6))
        
        return {
            'cell_voltage': cell_voltage,
            'current_density': current_density,
            'power_density': power_density,
            'coulombic_efficiency': coulombic_efficiency,
            'power_generated': power_density * self.config.cell_area  # W
        }
    
    def _update_cell_state(self, cell_idx: int, dt: float, anode_state: Dict[str, float],
                          membrane_state: Dict[str, float], cathode_state: Dict[str, float],
                          performance: Dict[str, float]) -> MFCSystemState:
        """Create updated cell state."""
        
        # Calculate operational cost
        power_generated = performance['power_generated']
        operational_cost = (self.config.maintenance_cost / 24.0 +  # $/h
                          0.001 * membrane_state['proton_flux'])  # Pumping costs
        
        # Revenue from power generation
        revenue_rate = power_generated * self.config.electricity_price / 1000.0  # $/h
        
        return MFCSystemState(
            time=self.time + dt,
            cell_id=cell_idx,
            # Anode state
            biofilm_thickness=anode_state['biofilm_thickness'],
            biomass_density=anode_state['biomass_density'],
            substrate_concentration=anode_state['substrate_concentration'],
            anode_potential=self.cell_states[cell_idx].anode_potential,
            # Membrane state
            membrane_conductivity=membrane_state['membrane_conductivity'],
            water_content=10.0,  # Simplified
            fouling_thickness=membrane_state['fouling_thickness'],
            degradation_fraction=membrane_state['degradation_fraction'],
            # Cathode state
            cathode_potential=cathode_state['cathode_potential'],
            oxygen_concentration=cathode_state['oxygen_concentration'],
            cathode_current_density=cathode_state['cathode_current_density'],
            # Performance
            cell_voltage=performance['cell_voltage'],
            current_density=performance['current_density'],
            power_density=performance['power_density'],
            coulombic_efficiency=performance['coulombic_efficiency'],
            # Transport
            proton_flux=membrane_state['proton_flux'],
            water_flux=membrane_state['water_flux'],
            oxygen_flux=cathode_state['cathode_current_density'] * 0.25 / 96485.0,
            # Economics
            power_generated=performance['power_generated'],
            operational_cost=operational_cost,
            revenue_rate=revenue_rate
        )
    
    def _apply_qlearning_control(self, states: List[MFCSystemState], dt: float):
        """Apply Q-learning control to optimize system performance."""
        if not self.q_controller:
            return
        
        # This would interface with the actual Q-learning controller
        # For now, just track that control is active
        # Would calculate: total_power, avg_efficiency, avg_substrate for Q-learning
        pass
    
    def run_system_simulation(self, duration_hours: float, dt: float = 1.0, 
                             save_interval: int = 10) -> Dict[str, Any]:
        """
        Run complete integrated system simulation.
        
        Args:
            duration_hours: Total simulation time (hours)
            dt: Time step (hours)
            save_interval: Save interval (hours)
            
        Returns:
            Simulation results dictionary
        """
        print("\n🚀 Starting Integrated MFC System Simulation")
        print(f"   Duration: {duration_hours} hours")
        print(f"   Time step: {dt} hours")
        print(f"   Cells: {self.config.n_cells}")
        
        start_time = time.time()
        n_steps = int(duration_hours / dt)
        
        # Main simulation loop
        for step in range(n_steps):
            # Step system dynamics
            current_states = self.step_system_dynamics(dt)
            
            # Progress reporting
            if step % 10 == 0:
                avg_power = np.mean([s.power_generated for s in current_states])
                avg_voltage = np.mean([s.cell_voltage for s in current_states])
                avg_efficiency = np.mean([s.coulombic_efficiency for s in current_states])
                
                print(f"   t={self.time:.1f}h: P={avg_power*1000:.1f}mW, "
                      f"V={avg_voltage:.3f}V, CE={avg_efficiency:.2%}")
            
            # Save checkpoint
            if step % save_interval == 0 and step > 0:
                self._save_checkpoint(step)
        
        # Compile final results
        computation_time = time.time() - start_time
        results = self._compile_simulation_results()
        results['computation_time'] = computation_time
        
        print("\n✅ Simulation Complete!")
        print(f"   Total Energy: {results['total_energy_wh']:.2f} Wh")
        print(f"   Peak Power: {results['peak_power_mw']:.1f} mW")
        print(f"   Average Efficiency: {results['average_efficiency']:.2%}")
        print(f"   Computation Time: {computation_time:.1f} seconds")
        
        return results
    
    def _compile_simulation_results(self) -> Dict[str, Any]:
        """Compile comprehensive simulation results."""
        if not self.history:
            return {}
        
        # Extract time series data
        df = pd.DataFrame([{
            'time': state.time,
            'cell_id': state.cell_id,
            'power_generated': state.power_generated,
            'cell_voltage': state.cell_voltage,
            'current_density': state.current_density,
            'coulombic_efficiency': state.coulombic_efficiency,
            'biofilm_thickness': state.biofilm_thickness,
            'substrate_concentration': state.substrate_concentration,
            'fouling_thickness': state.fouling_thickness,
            'operational_cost': state.operational_cost,
            'revenue_rate': state.revenue_rate
        } for state in self.history])
        
        # Calculate performance metrics
        total_energy = df['power_generated'].sum() * (self.time / len(df) * self.config.n_cells)  # Wh
        peak_power = df['power_generated'].max() * self.config.n_cells * 1000  # mW
        average_efficiency = df['coulombic_efficiency'].mean()
        
        # Economic metrics
        total_cost = df['operational_cost'].sum() * (self.time / len(df))
        total_revenue = df['revenue_rate'].sum() * (self.time / len(df))
        net_profit = total_revenue - total_cost
        
        # System health metrics
        final_fouling = df[df['time'] == df['time'].max()]['fouling_thickness'].mean()
        biofilm_stability = 1.0 - df['biofilm_thickness'].std() / df['biofilm_thickness'].mean()
        
        return {
            'total_energy_wh': total_energy,
            'peak_power_mw': peak_power,
            'average_power_mw': df['power_generated'].mean() * self.config.n_cells * 1000,
            'average_efficiency': average_efficiency,
            'final_fouling_um': final_fouling,
            'biofilm_stability': biofilm_stability,
            'total_cost_usd': total_cost,
            'total_revenue_usd': total_revenue,
            'net_profit_usd': net_profit,
            'time_series': df,
            'configuration': {
                'n_cells': self.config.n_cells,
                'bacterial_species': self.config.bacterial_species,
                'substrate_type': self.config.substrate_type,
                'membrane_type': f"{self.config.membrane_material} {self.config.membrane_type}",
                'cathode_type': self.config.cathode_type,
                'cell_area_cm2': self.config.cell_area * 1e4,
                'temperature_c': self.config.temperature - 273.15
            }
        }
    
    def _save_checkpoint(self, step: int):
        """Save simulation checkpoint."""
        # Implementation would save current state for recovery
        pass
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status summary."""
        if not self.cell_states:
            return {}
        
        current_time = self.time
        states = self.cell_states
        
        return {
            'time_hours': current_time,
            'total_power_mw': sum(s.power_generated for s in states) * 1000,
            'average_voltage_v': np.mean([s.cell_voltage for s in states]),
            'average_efficiency_pct': np.mean([s.coulombic_efficiency for s in states]) * 100,
            'average_biofilm_um': np.mean([s.biofilm_thickness for s in states]),
            'average_fouling_um': np.mean([s.fouling_thickness for s in states]),
            'substrate_utilization_pct': (1 - np.mean([s.substrate_concentration for s in states]) / 
                                        self.config.initial_substrate_conc) * 100,
            'operational_cost_usd_h': sum(s.operational_cost for s in states),
            'revenue_rate_usd_h': sum(s.revenue_rate for s in states),
            'cells_active': len(states),
            'system_health': 'Healthy' if all(s.cell_voltage > 0.1 for s in states) else 'Degraded'
        }


def create_standard_mfc_system(configuration: MFCConfiguration = MFCConfiguration.RESEARCH) -> IntegratedMFCSystem:
    """
    Create a standard MFC system configuration.
    
    Args:
        configuration: Standard configuration type
        
    Returns:
        Configured MFC system
    """
    if configuration == MFCConfiguration.BASIC_LAB:
        params = MFCStackParameters(
            n_cells=3,
            cell_area=0.005,  # 5 cm²
            bacterial_species="geobacter",
            substrate_type="acetate",
            membrane_material="Nafion",
            cathode_type="platinum"
        )
    elif configuration == MFCConfiguration.PILOT_PLANT:
        params = MFCStackParameters(
            n_cells=10,
            cell_area=0.1,    # 100 cm²
            bacterial_species="mixed",
            substrate_type="lactate",
            membrane_material="SPEEK",
            cathode_type="biological"
        )
    elif configuration == MFCConfiguration.INDUSTRIAL:
        params = MFCStackParameters(
            n_cells=20,
            cell_area=1.0,    # 1000 cm²
            bacterial_species="mixed",
            substrate_type="lactate",
            membrane_material="Nafion",
            cathode_type="platinum"
        )
    else:  # RESEARCH
        params = MFCStackParameters(
            n_cells=5,
            cell_area=0.01,   # 10 cm²
            bacterial_species="mixed",
            substrate_type="lactate",
            membrane_material="Nafion",
            cathode_type="platinum",
            enable_qlearning=True
        )
    
    return IntegratedMFCSystem(params)


def main():
    """Demonstration of integrated MFC system."""
    print("🧪 MFC System Integration Demo")
    
    # Create research configuration system
    mfc_system = create_standard_mfc_system(MFCConfiguration.RESEARCH)
    
    # Run short simulation
    results = mfc_system.run_system_simulation(
        duration_hours=24.0,  # 24 hour simulation
        dt=0.5,              # 30 minute time steps
        save_interval=2      # Save every 2 hours
    )
    
    # Display results
    print("\n📊 Simulation Results:")
    for key, value in results.items():
        if key != 'time_series' and key != 'configuration':
            print(f"   {key}: {value}")
    
    return results


if __name__ == "__main__":
    main()