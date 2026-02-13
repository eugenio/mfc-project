#!/usr/bin/env python3
"""
MFC-Hydraulic System Integration - Phase 6
===========================================

Integration of hydraulic system with the main MFC system from Phase 5.
Adds hydraulic modeling to the complete MFC simulation including:
- Flow distribution across cells
- Pump power consumption in system energy balance
- Hydraulic control integration with Q-learning
- Cost analysis including hydraulic components

Created: 2025-07-27 (Phase 6)
"""

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
import time

# Import MFC system components from Phase 5
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from mfc_system_integration import (
    IntegratedMFCSystem, MFCStackParameters, MFCSystemState
)

# Import hydraulic system components
from hydraulic_models import (
    HydraulicController, create_standard_hydraulic_system,
    calculate_hydraulic_costs, PumpParameters, PeristalticPump, CentrifugalPump, DiaphragmPump
)


@dataclass
class HydraulicMFCParameters:
    """Extended MFC parameters including hydraulic system configuration."""
    
    # Base MFC parameters
    mfc_params: MFCStackParameters
    
    # Hydraulic system parameters
    enable_recirculation: bool = True
    recirculation_rate: float = 100.0  # mL/min per cell
    substrate_delivery_rate: float = 10.0  # mL/min per cell
    aeration_rate: float = 500.0  # mL/min per cell (air)
    cleaning_frequency: float = 168.0  # hours (1 week)
    
    # Flow distribution
    flow_distribution_strategy: str = "uniform"  # "uniform", "weighted", "cascade"
    residence_time_target: float = 2.0  # hours
    
    # Pump specifications
    substrate_pump_type: str = "peristaltic"  # precise dosing
    recirculation_pump_type: str = "centrifugal"  # high flow
    aeration_pump_type: str = "diaphragm"  # gas handling


class IntegratedMFCHydraulicSystem:
    """
    Complete MFC system with integrated hydraulic modeling.
    
    Combines the electrochemical MFC system with detailed hydraulic modeling
    including pumps, flow distribution, and power consumption.
    """
    
    def __init__(self, config: HydraulicMFCParameters):
        self.config = config
        
        print("🔧 Initializing Integrated MFC-Hydraulic System")
        print(f"   Cells: {config.mfc_params.n_cells}")
        print(f"   Recirculation: {config.enable_recirculation}")
        print(f"   Flow rates: {config.substrate_delivery_rate} mL/min substrate")
        
        # Initialize base MFC system
        self.mfc_system = IntegratedMFCSystem(config.mfc_params)
        
        # Initialize hydraulic system
        self._initialize_hydraulic_system()
        
        # Initialize integration components
        self._initialize_flow_controllers()
        self._initialize_cost_tracking()
        
        print("✅ MFC-Hydraulic Integration Complete")
    
    def _initialize_hydraulic_system(self):
        """Initialize the hydraulic system."""
        # Create hydraulic network
        self.hydraulic_network = create_standard_hydraulic_system(
            n_cells=self.config.mfc_params.n_cells
        )
        
        # Customize pumps based on configuration
        self._configure_pumps()
        
        # Create hydraulic controller
        self.hydraulic_controller = HydraulicController(self.hydraulic_network)
        
        # Set initial flow setpoints
        self._set_initial_flow_setpoints()
    
    def _configure_pumps(self):
        """Configure pumps based on system requirements."""
        # Clear existing pumps
        self.hydraulic_network.pumps.clear()
        
        n_cells = self.config.mfc_params.n_cells
        
        # Substrate delivery pump
        substrate_params = PumpParameters(
            max_flow_rate=self.config.substrate_delivery_rate * n_cells * 2,  # 2x safety factor
            max_pressure=50000.0,  # 0.5 bar
            efficiency=0.8,
            power_rating=3.0,
            cost=200.0
        )
        substrate_pump = PeristalticPump("substrate_delivery", substrate_params)
        self.hydraulic_network.add_pump(substrate_pump, "substrate")
        
        # Recirculation pump (if enabled)
        if self.config.enable_recirculation:
            recirculation_params = PumpParameters(
                max_flow_rate=self.config.recirculation_rate * n_cells * 1.5,
                max_pressure=30000.0,  # 0.3 bar
                efficiency=0.85,
                power_rating=8.0,
                cost=300.0
            )
            recirculation_pump = CentrifugalPump("recirculation", recirculation_params)
            self.hydraulic_network.add_pump(recirculation_pump, "recirculation")
        
        # Aeration pump
        aeration_params = PumpParameters(
            max_flow_rate=self.config.aeration_rate * n_cells,
            max_pressure=20000.0,  # 0.2 bar
            efficiency=0.7,
            power_rating=5.0,
            cost=150.0
        )
        aeration_pump = DiaphragmPump("aeration", aeration_params)
        self.hydraulic_network.add_pump(aeration_pump, "aeration")
        
        # Cleaning pump (periodic)
        cleaning_params = PumpParameters(
            max_flow_rate=50.0,  # mL/min
            max_pressure=100000.0,  # 1 bar for cleaning
            efficiency=0.75,
            power_rating=4.0,
            cost=180.0
        )
        cleaning_pump = PeristalticPump("cleaning", cleaning_params)
        self.hydraulic_network.add_pump(cleaning_pump, "cleaning")
    
    def _set_initial_flow_setpoints(self):
        """Set initial flow rate setpoints."""
        n_cells = self.config.mfc_params.n_cells
        
        # Substrate delivery
        self.hydraulic_controller.set_flow_setpoint(
            "substrate", self.config.substrate_delivery_rate * n_cells
        )
        
        # Recirculation
        if self.config.enable_recirculation:
            self.hydraulic_controller.set_flow_setpoint(
                "recirculation", self.config.recirculation_rate * n_cells
            )
        
        # Aeration
        self.hydraulic_controller.set_flow_setpoint(
            "aeration", self.config.aeration_rate * n_cells
        )
        
        # Cleaning (initially off)
        self.hydraulic_controller.set_flow_setpoint("cleaning", 0.0)
    
    def _initialize_flow_controllers(self):
        """Initialize flow control integration."""
        self.flow_control_enabled = True
        self.last_cleaning_time = 0.0
        self.flow_adjustments = {
            "substrate": 1.0,
            "recirculation": 1.0,
            "aeration": 1.0
        }
    
    def _initialize_cost_tracking(self):
        """Initialize cost tracking for hydraulic components."""
        self.hydraulic_costs = {
            'total_energy_kwh': 0.0,
            'total_cost_usd': 0.0,
            'pump_maintenance_due': []
        }
    
    def start_hydraulic_system(self):
        """Start all hydraulic pumps."""
        for pump_name, pump in self.hydraulic_network.pumps.items():
            if pump_name != "cleaning":  # Don't start cleaning pump by default
                pump.start_pump()
        
        print("🚀 Hydraulic system started")
    
    def stop_hydraulic_system(self):
        """Stop all hydraulic pumps."""
        for pump in self.hydraulic_network.pumps.values():
            pump.stop_pump()
        
        print("⏹️ Hydraulic system stopped")
    
    def update_flow_control(self, system_states: List[MFCSystemState], dt: float):
        """Update hydraulic flow control based on MFC system state."""
        if not self.flow_control_enabled:
            return
        
        # Calculate average system performance
        avg_power = np.mean([s.power_density for s in system_states])
        avg_efficiency = np.mean([s.coulombic_efficiency for s in system_states])
        avg_substrate = np.mean([s.substrate_concentration for s in system_states])
        avg_fouling = np.mean([s.fouling_thickness for s in system_states])
        
        # Adjust substrate delivery based on concentration
        if avg_substrate < 5.0:  # Low substrate
            self.flow_adjustments["substrate"] = min(1.5, self.flow_adjustments["substrate"] + 0.1)
        elif avg_substrate > 30.0:  # High substrate
            self.flow_adjustments["substrate"] = max(0.5, self.flow_adjustments["substrate"] - 0.1)
        
        # Adjust recirculation based on performance
        if self.config.enable_recirculation:
            if avg_efficiency < 0.5:  # Low efficiency
                self.flow_adjustments["recirculation"] = min(1.3, self.flow_adjustments["recirculation"] + 0.05)
            elif avg_efficiency > 0.8:  # High efficiency
                self.flow_adjustments["recirculation"] = max(0.7, self.flow_adjustments["recirculation"] - 0.02)
        
        # Adjust aeration based on power output
        if avg_power < 0.2:  # Low power
            self.flow_adjustments["aeration"] = min(1.2, self.flow_adjustments["aeration"] + 0.05)
        
        # Update flow setpoints
        n_cells = self.config.mfc_params.n_cells
        
        self.hydraulic_controller.set_flow_setpoint(
            "substrate", 
            self.config.substrate_delivery_rate * n_cells * self.flow_adjustments["substrate"]
        )
        
        if self.config.enable_recirculation:
            self.hydraulic_controller.set_flow_setpoint(
                "recirculation", 
                self.config.recirculation_rate * n_cells * self.flow_adjustments["recirculation"]
            )
        
        self.hydraulic_controller.set_flow_setpoint(
            "aeration", 
            self.config.aeration_rate * n_cells * self.flow_adjustments["aeration"]
        )
        
        # Check for cleaning cycle
        current_time = self.mfc_system.time
        if (current_time - self.last_cleaning_time) > self.config.cleaning_frequency:
            if avg_fouling > 20.0:  # Significant fouling
                self._initiate_cleaning_cycle()
    
    def _initiate_cleaning_cycle(self):
        """Initiate cleaning cycle."""
        print(f"🧽 Initiating cleaning cycle at t={self.mfc_system.time:.1f}h")
        
        # Set cleaning flow
        self.hydraulic_controller.set_flow_setpoint("cleaning", 30.0)  # mL/min
        
        # Start cleaning pump
        self.hydraulic_network.pumps["cleaning"].start_pump()
        
        # Update last cleaning time
        self.last_cleaning_time = self.mfc_system.time
        
        # Schedule cleaning stop (would be implemented in real system)
        # For simulation, cleaning runs for 1 hour
    
    def _stop_cleaning_cycle(self):
        """Stop cleaning cycle."""
        self.hydraulic_controller.set_flow_setpoint("cleaning", 0.0)
        self.hydraulic_network.pumps["cleaning"].stop_pump()
        print("🧽 Cleaning cycle complete")
    
    def step_integrated_system(self, dt: float) -> List[MFCSystemState]:
        """Step the complete integrated MFC-hydraulic system."""
        # Update MFC system dynamics
        mfc_states = self.mfc_system.step_system_dynamics(dt)
        
        # Update hydraulic flow control
        self.update_flow_control(mfc_states, dt)
        
        # Update hydraulic system
        target_flows = self.hydraulic_controller.update_control(dt)
        self.hydraulic_network.update_hydraulic_system(dt, target_flows)
        
        # Update cost tracking
        self._update_cost_tracking(dt)
        
        # Handle cleaning cycle timing
        if self.hydraulic_network.pumps["cleaning"].is_running:
            if (self.mfc_system.time - self.last_cleaning_time) > 1.0:  # 1 hour cleaning
                self._stop_cleaning_cycle()
        
        return mfc_states
    
    def _update_cost_tracking(self, dt: float):
        """Update hydraulic cost tracking."""
        # Energy consumption
        power_kw = self.hydraulic_network.total_power_consumption / 1000.0
        energy_kwh = power_kw * dt
        self.hydraulic_costs['total_energy_kwh'] += energy_kwh
        
        # Check for maintenance due
        self.hydraulic_costs['pump_maintenance_due'] = []
        for pump_name, pump in self.hydraulic_network.pumps.items():
            if pump.maintenance_due:
                self.hydraulic_costs['pump_maintenance_due'].append(pump_name)
    
    def run_integrated_simulation(self, duration_hours: float, dt: float = 0.5) -> Dict[str, Any]:
        """Run complete integrated MFC-hydraulic simulation."""
        print("\n🚀 Starting Integrated MFC-Hydraulic Simulation")
        print(f"   Duration: {duration_hours} hours")
        print(f"   Time step: {dt} hours")
        print(f"   Hydraulic pumps: {len(self.hydraulic_network.pumps)}")
        
        # Start hydraulic system
        self.start_hydraulic_system()
        
        start_time = time.time()
        n_steps = int(duration_hours / dt)
        
        # Tracking data
        hydraulic_data = []
        
        # Main simulation loop
        for step in range(n_steps):
            # Step integrated system
            mfc_states = self.step_integrated_system(dt)
            
            # Collect hydraulic data
            hydraulic_status = self.hydraulic_network.get_network_status()
            hydraulic_data.append({
                'time': self.mfc_system.time,
                'total_hydraulic_power_w': hydraulic_status['total_power_w'],
                'network_efficiency': hydraulic_status['network_efficiency'],
                'substrate_flow': hydraulic_status['flow_rates'].get('substrate', 0),
                'recirculation_flow': hydraulic_status['flow_rates'].get('recirculation', 0),
                'aeration_flow': hydraulic_status['flow_rates'].get('aeration', 0)
            })
            
            # Progress reporting
            if step % 20 == 0:
                avg_power = np.mean([s.power_generated for s in mfc_states]) * 1000  # mW
                hydraulic_power = hydraulic_status['total_power_w']
                net_power = avg_power - hydraulic_power
                
                print(f"   t={self.mfc_system.time:.1f}h: "
                      f"MFC={avg_power:.1f}mW, Hydraulic={hydraulic_power:.1f}W, "
                      f"Net={net_power:.1f}mW")
        
        # Compile final results
        computation_time = time.time() - start_time
        
        # Get MFC results
        mfc_results = self.mfc_system._compile_simulation_results()
        
        # Calculate hydraulic costs
        hydraulic_costs = calculate_hydraulic_costs(
            self.hydraulic_network, 
            duration_hours,
            self.config.mfc_params.electricity_price
        )
        
        # Compile integrated results
        integrated_results = {
            **mfc_results,
            'hydraulic_data': hydraulic_data,
            'hydraulic_costs': hydraulic_costs,
            'total_system_cost_usd': mfc_results.get('total_cost_usd', 0) + hydraulic_costs['total_cost_usd'],
            'hydraulic_power_consumption_kwh': hydraulic_costs['power_consumption_kwh'],
            'net_energy_production_kwh': max(0, mfc_results.get('total_energy_wh', 0)/1000 - hydraulic_costs['power_consumption_kwh']),
            'system_efficiency_including_hydraulics': self._calculate_system_efficiency(mfc_results, hydraulic_costs),
            'computation_time': computation_time
        }
        
        print("\n✅ Integrated Simulation Complete!")
        print(f"   MFC Energy: {mfc_results.get('total_energy_wh', 0):.2f} Wh")
        print(f"   Hydraulic Energy: {hydraulic_costs['power_consumption_kwh']*1000:.2f} Wh")
        print(f"   Net Energy: {integrated_results['net_energy_production_kwh']*1000:.2f} Wh")
        print(f"   System Efficiency: {integrated_results['system_efficiency_including_hydraulics']:.2%}")
        
        return integrated_results
    
    def _calculate_system_efficiency(self, mfc_results: Dict, hydraulic_costs: Dict) -> float:
        """Calculate overall system efficiency including hydraulics."""
        mfc_energy_wh = mfc_results.get('total_energy_wh', 0)
        hydraulic_energy_wh = hydraulic_costs['power_consumption_kwh'] * 1000
        
        if hydraulic_energy_wh == 0:
            return 1.0
        
        net_energy = mfc_energy_wh - hydraulic_energy_wh
        total_input_energy = hydraulic_energy_wh  # Simplified - could include substrate energy content
        
        if total_input_energy == 0:
            return 0.0
        
        return max(0, net_energy / total_input_energy)
    
    def get_integrated_status(self) -> Dict[str, Any]:
        """Get complete integrated system status."""
        mfc_status = self.mfc_system.get_system_status()
        hydraulic_status = self.hydraulic_network.get_network_status()
        
        return {
            'mfc_system': mfc_status,
            'hydraulic_system': hydraulic_status,
            'flow_adjustments': self.flow_adjustments.copy(),
            'last_cleaning_time': self.last_cleaning_time,
            'hydraulic_costs': self.hydraulic_costs.copy(),
            'net_power_mw': mfc_status.get('total_power_mw', 0) - hydraulic_status.get('total_power_w', 0) / 1000
        }


def create_integrated_mfc_hydraulic_system(
    mfc_config: MFCStackParameters,
    enable_recirculation: bool = True,
    substrate_rate: float = 10.0,
    recirculation_rate: float = 100.0
) -> IntegratedMFCHydraulicSystem:
    """Create an integrated MFC-hydraulic system."""
    
    hydraulic_config = HydraulicMFCParameters(
        mfc_params=mfc_config,
        enable_recirculation=enable_recirculation,
        substrate_delivery_rate=substrate_rate,
        recirculation_rate=recirculation_rate
    )
    
    return IntegratedMFCHydraulicSystem(hydraulic_config)


def main():
    """Demonstration of integrated MFC-hydraulic system."""
    print("🔧🔋 Integrated MFC-Hydraulic System Demo")
    
    # Create MFC configuration
    from mfc_system_integration import MFCStackParameters
    
    mfc_config = MFCStackParameters(
        n_cells=3,
        cell_area=0.01,  # 10 cm²
        bacterial_species="mixed",
        substrate_type="lactate",
        membrane_material="Nafion",
        cathode_type="platinum",
        enable_qlearning=False  # Disable for hydraulic demo
    )
    
    # Create integrated system
    integrated_system = create_integrated_mfc_hydraulic_system(
        mfc_config=mfc_config,
        enable_recirculation=True,
        substrate_rate=15.0,  # mL/min per cell
        recirculation_rate=80.0  # mL/min per cell
    )
    
    # Run simulation
    results = integrated_system.run_integrated_simulation(
        duration_hours=12.0,  # 12 hour simulation
        dt=0.25  # 15 minute time steps
    )
    
    # Display key results
    print("\n📊 Integration Results:")
    print(f"   Total System Cost: ${results['total_system_cost_usd']:.2f}")
    print(f"   Net Energy: {results['net_energy_production_kwh']*1000:.1f} Wh")
    print(f"   System Efficiency: {results['system_efficiency_including_hydraulics']:.1%}")
    
    return results


if __name__ == "__main__":
    main()