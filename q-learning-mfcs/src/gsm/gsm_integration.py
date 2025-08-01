#!/usr/bin/env python3
"""
Phase 4: Genome-Scale Metabolic (GSM) Model Integration for MFC Optimization

This module integrates genome-scale metabolic models with the electrode optimization
system to enable organism-specific parameter tuning and performance prediction.

Based on Shewanella oneidensis MR-1 metabolic network analysis.

Created: 2025-08-01
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any, List

@dataclass
class MetabolicReaction:
    """Represents a metabolic reaction with stoichiometry and constraints."""
    
    id: str
    name: str
    stoichiometry: Dict[str, float]  # metabolite_id: coefficient
    lower_bound: float = 0.0  # mmol/g AFDW/h
    upper_bound: float = 1000.0  # mmol/g AFDW/h
    objective_coefficient: float = 0.0
    subsystem: str = ""
    ec_number: str = ""
    gene_reaction_rule: str = ""

@dataclass 
class Metabolite:
    """Represents a metabolite in the network."""
    
    id: str
    name: str
    formula: str = ""
    charge: int = 0
    compartment: str = "c"  # cytoplasm default
    boundary: bool = False

@dataclass
class GSMModelConfig:
    """Configuration for GSM model parameters."""
    
    # Model identification
    organism: str = "Shewanella oneidensis MR-1"
    model_id: str = "iSO783"
    
    # Growth parameters
    max_growth_rate: float = 0.085  # h⁻¹
    maintenance_atp: float = 1.03   # mmol ATP/(g AFDW·h)
    growth_atp: float = 220.22      # mmol ATP/g AFDW
    
    # Substrate utilization
    max_lactate_uptake: float = 4.11  # mmol/g AFDW/h
    max_oxygen_uptake: float = 10.0   # mmol/g AFDW/h
    max_riboflavin_export: float = 0.01  # mmol/g AFDW/h
    
    # Electron transfer parameters
    flavin_transfer_efficiency: float = 0.7  # 70% efficiency
    direct_transfer_efficiency: float = 0.3  # 30% efficiency
    
    # Environmental conditions  
    temperature: float = 30.0  # °C
    ph: float = 7.0
    electrode_potential: float = 0.2  # V vs SHE

class ShewanellaGSMModel:
    """
    Genome-scale metabolic model for Shewanella oneidensis MR-1.
    
    Implements core metabolic pathways based on constraint-based modeling
    and integrates with electrode optimization framework.
    """
    
    def __init__(self, config: Optional[GSMModelConfig] = None):
        self.config = config or GSMModelConfig()
        
        # Initialize metabolic network
        self.metabolites: Dict[str, Metabolite] = {}
        self.reactions: Dict[str, MetabolicReaction] = {}
        self.flux_solutions: Dict[str, Dict[str, float]] = {}
        
        # Build core metabolic network
        self._build_metabolic_network()
        
        # Current metabolic state
        self.current_fluxes: Dict[str, float] = {}
        self.current_growth_rate = 0.0
        self.current_electron_production = 0.0
        
    def _build_metabolic_network(self):
        """Build the core metabolic network for S. oneidensis MR-1."""
        
        # Define key metabolites
        self.metabolites = {
            # External metabolites
            'lactate_ext': Metabolite('lactate_ext', 'Lactate (external)', 'C3H6O3', compartment='e', boundary=True),
            'oxygen_ext': Metabolite('oxygen_ext', 'Oxygen (external)', 'O2', compartment='e', boundary=True),
            'co2_ext': Metabolite('co2_ext', 'Carbon dioxide (external)', 'CO2', compartment='e', boundary=True),
            'acetate_ext': Metabolite('acetate_ext', 'Acetate (external)', 'C2H4O2', compartment='e', boundary=True),
            'riboflavin_ext': Metabolite('riboflavin_ext', 'Riboflavin (external)', 'C17H20N4O6', compartment='e', boundary=True),
            
            # Internal metabolites
            'lactate_c': Metabolite('lactate_c', 'Lactate (cytoplasm)', 'C3H6O3', compartment='c'),
            'pyruvate_c': Metabolite('pyruvate_c', 'Pyruvate', 'C3H4O3', compartment='c'),
            'acetyl_coa_c': Metabolite('acetyl_coa_c', 'Acetyl-CoA', 'C23H38N7O17P3S', compartment='c'),
            'acetate_c': Metabolite('acetate_c', 'Acetate (cytoplasm)', 'C2H4O2', compartment='c'),
            'co2_c': Metabolite('co2_c', 'Carbon dioxide (cytoplasm)', 'CO2', compartment='c'),
            
            # Energy metabolites
            'atp_c': Metabolite('atp_c', 'ATP', 'C10H16N5O13P3', compartment='c'),
            'adp_c': Metabolite('adp_c', 'ADP', 'C10H15N5O10P2', compartment='c'),
            'nad_c': Metabolite('nad_c', 'NAD+', 'C21H29N7O14P2', compartment='c'),
            'nadh_c': Metabolite('nadh_c', 'NADH', 'C21H30N7O14P2', compartment='c'),
            
            # Electron transfer
            'quinone_c': Metabolite('quinone_c', 'Quinone', compartment='c'),
            'quinol_c': Metabolite('quinol_c', 'Quinol', compartment='c'),
            'cytc_ox_p': Metabolite('cytc_ox_p', 'Cytochrome c (oxidized)', compartment='p'),
            'cytc_red_p': Metabolite('cytc_red_p', 'Cytochrome c (reduced)', compartment='p'),
            'riboflavin_c': Metabolite('riboflavin_c', 'Riboflavin', 'C17H20N4O6', compartment='c'),
            'riboflavin_red_c': Metabolite('riboflavin_red_c', 'Riboflavin (reduced)', compartment='c'),
            
            # Biomass
            'biomass_c': Metabolite('biomass_c', 'Biomass', compartment='c')
        }
        
        # Define core metabolic reactions
        self.reactions = {
            # Substrate uptake
            'lactate_uptake': MetabolicReaction(
                id='lactate_uptake',
                name='Lactate uptake',
                stoichiometry={'lactate_ext': -1, 'lactate_c': 1},
                upper_bound=self.config.max_lactate_uptake,
                subsystem='Transport'
            ),
            
            'oxygen_uptake': MetabolicReaction(
                id='oxygen_uptake', 
                name='Oxygen uptake',
                stoichiometry={'oxygen_ext': -1},
                upper_bound=self.config.max_oxygen_uptake,
                subsystem='Transport'
            ),
            
            # Core metabolism
            'lactate_dehydrogenase': MetabolicReaction(
                id='lactate_dehydrogenase',
                name='Lactate dehydrogenase',
                stoichiometry={'lactate_c': -1, 'nad_c': -1, 'pyruvate_c': 1, 'nadh_c': 1},
                upper_bound=self.config.max_lactate_uptake,
                subsystem='Central Carbon Metabolism',
                ec_number='1.1.1.27'
            ),
            
            'pyruvate_dehydrogenase': MetabolicReaction(
                id='pyruvate_dehydrogenase',
                name='Pyruvate dehydrogenase complex',
                stoichiometry={'pyruvate_c': -1, 'nad_c': -1, 'acetyl_coa_c': 1, 'co2_c': 1, 'nadh_c': 1},
                upper_bound=self.config.max_lactate_uptake,
                subsystem='Central Carbon Metabolism',
                ec_number='1.2.4.1'
            ),
            
            'acetyl_coa_hydrolysis': MetabolicReaction(
                id='acetyl_coa_hydrolysis',
                name='Acetyl-CoA hydrolysis',
                stoichiometry={'acetyl_coa_c': -1, 'acetate_c': 1},
                upper_bound=self.config.max_lactate_uptake,
                subsystem='Central Carbon Metabolism'
            ),
            
            # Electron transport chain
            'nadh_quinone_oxidoreductase': MetabolicReaction(
                id='nadh_quinone_oxidoreductase',
                name='NADH:quinone oxidoreductase (Complex I)',
                stoichiometry={'nadh_c': -1, 'quinone_c': -1, 'nad_c': 1, 'quinol_c': 1},
                upper_bound=1000.0,
                subsystem='Electron Transport',
                ec_number='1.6.5.3'
            ),
            
            'bc1_complex': MetabolicReaction(
                id='bc1_complex',
                name='Cytochrome bc1 complex',
                stoichiometry={'quinol_c': -1, 'cytc_ox_p': -2, 'quinone_c': 1, 'cytc_red_p': 2},
                upper_bound=1000.0,
                subsystem='Electron Transport',
                ec_number='1.10.2.2'
            ),
            
            'cytochrome_oxidase': MetabolicReaction(
                id='cytochrome_oxidase',
                name='Cytochrome c oxidase',
                stoichiometry={'cytc_red_p': -4, 'oxygen_ext': -1, 'cytc_ox_p': 4, 'atp_c': 2.8},
                upper_bound=1000.0,
                subsystem='Electron Transport',
                ec_number='1.9.3.1'
            ),
            
            # Flavin-mediated electron transfer
            'riboflavin_synthesis': MetabolicReaction(
                id='riboflavin_synthesis',
                name='Riboflavin biosynthesis',
                stoichiometry={'atp_c': -5, 'riboflavin_c': 1},
                upper_bound=self.config.max_riboflavin_export * 2,
                subsystem='Flavin Metabolism',
                ec_number='2.5.1.9'
            ),
            
            'riboflavin_reduction': MetabolicReaction(
                id='riboflavin_reduction',
                name='Riboflavin reduction',
                stoichiometry={'riboflavin_c': -1, 'nadh_c': -1, 'riboflavin_red_c': 1, 'nad_c': 1},
                upper_bound=self.config.max_riboflavin_export,
                subsystem='Flavin Metabolism'
            ),
            
            'riboflavin_export': MetabolicReaction(
                id='riboflavin_export',
                name='Riboflavin export',
                stoichiometry={'riboflavin_c': -1, 'riboflavin_ext': 1},
                upper_bound=self.config.max_riboflavin_export,
                subsystem='Transport'
            ),
            
            'electrode_electron_transfer': MetabolicReaction(
                id='electrode_electron_transfer',
                name='Electrode electron transfer',
                stoichiometry={'riboflavin_red_c': -1, 'riboflavin_c': 1},
                upper_bound=self.config.max_riboflavin_export,
                subsystem='Electron Transfer',
                objective_coefficient=1.0  # This is what we want to maximize
            ),
            
            # Export reactions
            'co2_export': MetabolicReaction(
                id='co2_export',
                name='CO2 export',
                stoichiometry={'co2_c': -1, 'co2_ext': 1},
                upper_bound=1000.0,
                subsystem='Transport'
            ),
            
            'acetate_export': MetabolicReaction(
                id='acetate_export',
                name='Acetate export',
                stoichiometry={'acetate_c': -1, 'acetate_ext': 1},
                upper_bound=1000.0,
                subsystem='Transport'
            ),
            
            # ATP maintenance
            'atp_maintenance': MetabolicReaction(
                id='atp_maintenance',
                name='ATP maintenance',
                stoichiometry={'atp_c': -1, 'adp_c': 1},
                lower_bound=self.config.maintenance_atp,
                upper_bound=self.config.maintenance_atp,
                subsystem='Energy'
            ),
            
            # Growth (biomass formation)
            'biomass_reaction': MetabolicReaction(
                id='biomass_reaction',
                name='Biomass formation',
                stoichiometry={
                    'atp_c': -self.config.growth_atp/1000,  # Convert to reasonable scale
                    'acetyl_coa_c': -0.1,  # Simplified biomass precursors
                    'biomass_c': 1
                },
                upper_bound=self.config.max_growth_rate,
                subsystem='Growth'
            )
        }
    
    def solve_fba(self, objective_reaction: str = 'electrode_electron_transfer') -> Dict[str, float]:
        """
        Solve flux balance analysis to find optimal metabolic fluxes.
        
        This is a simplified FBA implementation focusing on electron production.
        """
        
        # Simplified flux calculation based on stoichiometric constraints
        fluxes = {}
        
        # Set substrate uptake based on availability
        lactate_flux = min(self.config.max_lactate_uptake, 4.0)  # Typical uptake rate
        fluxes['lactate_uptake'] = lactate_flux
        
        # Central metabolism fluxes (assuming complete lactate oxidation)
        fluxes['lactate_dehydrogenase'] = lactate_flux
        fluxes['pyruvate_dehydrogenase'] = lactate_flux * 0.8  # 80% goes to acetyl-CoA
        fluxes['acetyl_coa_hydrolysis'] = lactate_flux * 0.6   # 60% goes to acetate
        
        # NADH production (2 NADH per lactate)
        nadh_production = lactate_flux * 2.0
        
        # Electron transport
        fluxes['nadh_quinone_oxidoreductase'] = nadh_production
        fluxes['bc1_complex'] = nadh_production
        
        # Determine electron acceptor usage
        if self.config.max_oxygen_uptake > 0:
            # Aerobic conditions - some electrons go to oxygen
            oxygen_electrons = min(nadh_production * 0.5, self.config.max_oxygen_uptake * 4)
            fluxes['cytochrome_oxidase'] = oxygen_electrons / 4
            fluxes['oxygen_uptake'] = oxygen_electrons / 4
        else:
            oxygen_electrons = 0
            fluxes['cytochrome_oxidase'] = 0
            fluxes['oxygen_uptake'] = 0
        
        # Remaining electrons for electrode transfer via flavins
        remaining_electrons = nadh_production * 2 - oxygen_electrons
        electrode_flux = remaining_electrons * self.config.flavin_transfer_efficiency / 2
        
        fluxes['riboflavin_synthesis'] = electrode_flux * 1.2  # Slight excess
        fluxes['riboflavin_reduction'] = electrode_flux
        fluxes['electrode_electron_transfer'] = electrode_flux
        fluxes['riboflavin_export'] = electrode_flux * 0.1  # Small export
        
        # Export fluxes
        fluxes['co2_export'] = lactate_flux * 0.8  # Most pyruvate → CO2
        fluxes['acetate_export'] = lactate_flux * 0.6  # Acetate export
        
        # ATP and maintenance
        atp_from_respiration = oxygen_electrons * 0.7  # P/O ratio ~2.8
        fluxes['atp_maintenance'] = self.config.maintenance_atp
        
        # Growth (if sufficient ATP)
        available_atp = atp_from_respiration - self.config.maintenance_atp
        if available_atp > 0:
            growth_flux = min(available_atp / self.config.growth_atp * 1000, self.config.max_growth_rate)
            fluxes['biomass_reaction'] = max(0, growth_flux)
        else:
            fluxes['biomass_reaction'] = 0
        
        # Store current state
        self.current_fluxes = fluxes
        self.current_growth_rate = fluxes.get('biomass_reaction', 0)
        self.current_electron_production = fluxes.get('electrode_electron_transfer', 0)
        
        return fluxes
    
    def calculate_metabolic_objectives(self) -> Dict[str, float]:
        """Calculate metabolic objectives for optimization integration."""
        
        # Solve current metabolic state
        fluxes = self.solve_fba()
        
        objectives = {
            # Primary objectives
            'maximize_electron_production': self.current_electron_production,
            'maximize_growth_rate': self.current_growth_rate,
            'maximize_substrate_utilization': fluxes.get('lactate_uptake', 0) / self.config.max_lactate_uptake,
            
            # Secondary objectives
            'minimize_substrate_waste': 1.0 - (fluxes.get('acetate_export', 0) / fluxes.get('lactate_uptake', 1)),
            'maximize_energy_efficiency': fluxes.get('electrode_electron_transfer', 0) / max(fluxes.get('lactate_uptake', 1), 0.1),
            'maximize_flavin_utilization': fluxes.get('riboflavin_reduction', 0) / max(fluxes.get('riboflavin_synthesis', 1), 0.1),
            
            # Metabolic health indicators
            'metabolic_burden': fluxes.get('atp_maintenance', 0) / max(fluxes.get('lactate_uptake', 1), 0.1),
            'electron_transfer_efficiency': self.config.flavin_transfer_efficiency,
            'cofactor_balance': min(fluxes.get('nadh_quinone_oxidoreductase', 0) / max(fluxes.get('lactate_dehydrogenase', 1), 0.1), 1.0)
        }
        
        return objectives
    
    def update_environmental_conditions(self, 
                                      substrate_concentration: float,
                                      oxygen_availability: float,
                                      electrode_potential: float,
                                      temperature: Optional[float] = None,
                                      ph: Optional[float] = None):
        """Update environmental conditions affecting metabolism."""
        
        # Update substrate availability 
        max_possible = min(4.11, substrate_concentration * 0.1)  # Scale appropriately
        if max_possible != self.config.max_lactate_uptake:
            self.config.max_lactate_uptake = max_possible
        
        # Update oxygen availability
        self.config.max_oxygen_uptake = oxygen_availability * 10.0
        
        # Update electrode potential (affects electron transfer efficiency)
        self.config.electrode_potential = electrode_potential
        
        # Simple potential-dependent efficiency (could be more sophisticated)
        if electrode_potential > 0.1:
            self.config.flavin_transfer_efficiency = min(0.9, 0.5 + electrode_potential)
        else:
            self.config.flavin_transfer_efficiency = max(0.1, 0.5 + electrode_potential)
        
        # Update temperature and pH if provided
        if temperature is not None:
            self.config.temperature = temperature
            
        if ph is not None:
            self.config.ph = ph
    
    def get_metabolic_summary(self) -> Dict[str, Any]:
        """Get comprehensive metabolic summary."""
        
        objectives = self.calculate_metabolic_objectives()
        
        return {
            'organism': self.config.organism,
            'model_id': self.config.model_id,
            'current_fluxes': self.current_fluxes,
            'objectives': objectives,
            'environmental_conditions': {
                'temperature': self.config.temperature,
                'ph': self.config.ph,
                'electrode_potential': self.config.electrode_potential,
                'max_lactate_uptake': self.config.max_lactate_uptake,
                'max_oxygen_uptake': self.config.max_oxygen_uptake
            },
            'performance_metrics': {
                'electron_production_rate': self.current_electron_production,
                'growth_rate': self.current_growth_rate,
                'substrate_utilization_efficiency': objectives['maximize_substrate_utilization'],
                'energy_efficiency': objectives['maximize_energy_efficiency']
            }
        }

class GSMPhysicsIntegrator:
    """
    Integrator that connects GSM models with physics-based electrode models.
    
    This class bridges metabolic predictions with physical electrode performance.
    """
    
    def __init__(self, gsm_model: ShewanellaGSMModel):
        self.gsm_model = gsm_model
        self.integration_history: List[Dict[str, Any]] = []
    
    def integrate_with_electrode_model(self, electrode_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Integrate GSM predictions with electrode model results.
        
        Args:
            electrode_results: Results from advanced electrode model
            
        Returns:
            Integrated optimization objectives
        """
        
        # Extract relevant conditions from electrode model
        performance_metrics = electrode_results.get('performance_metrics', {})
        
        # Update GSM model with electrode conditions
        substrate_conc = performance_metrics.get('avg_substrate_mM', 25.0)
        oxygen_availability = 0.0  # Assume anaerobic MFC conditions
        
        # Get electrode potential from results (simplified)
        electrode_potential = 0.2  # Default, could be calculated from current density
        
        # Update GSM environmental conditions
        self.gsm_model.update_environmental_conditions(
            substrate_concentration=substrate_conc,
            oxygen_availability=oxygen_availability,
            electrode_potential=electrode_potential
        )
        
        # Get metabolic objectives
        metabolic_objectives = self.gsm_model.calculate_metabolic_objectives()
        
        # Get physics objectives (from electrode model)
        physics_objectives = {
            'current_density': performance_metrics.get('avg_current_density_A_m2', 0.0),
            'substrate_utilization': 1.0 - (substrate_conc / 25.0),  # Normalized utilization
            'biofilm_density': performance_metrics.get('avg_biofilm_density_kg_m3', 0.0)
        }
        
        # Integrate objectives with weighting
        integrated_objectives = {
            # Primary: Electron production (GSM) × Current density (Physics)
            'maximize_bioelectrochemical_performance': (
                metabolic_objectives['maximize_electron_production'] * 
                physics_objectives['current_density'] * 10  # Scale factor
            ),
            
            # Substrate efficiency: Both models should agree
            'maximize_integrated_substrate_efficiency': (
                0.5 * metabolic_objectives['maximize_substrate_utilization'] +
                0.5 * physics_objectives['substrate_utilization']
            ),
            
            # Growth-biofilm coupling
            'maximize_biofilm_metabolic_activity': (
                metabolic_objectives['maximize_growth_rate'] * 
                physics_objectives['biofilm_density'] / 100.0  # Normalize
            ),
            
            # Energy efficiency integration
            'maximize_integrated_energy_efficiency': (
                0.7 * metabolic_objectives['maximize_energy_efficiency'] +
                0.3 * physics_objectives['current_density']
            ),
            
            # System stability
            'maximize_system_stability': (
                0.4 * metabolic_objectives['cofactor_balance'] +
                0.3 * (1.0 - metabolic_objectives['metabolic_burden']) +
                0.3 * min(physics_objectives['biofilm_density'] / 50.0, 1.0)  # Optimal biofilm
            ),
            
            # Minimize waste and inefficiencies
            'minimize_integrated_losses': (
                metabolic_objectives['minimize_substrate_waste'] * 
                physics_objectives['substrate_utilization']
            )
        }
        
        # Store integration history
        self.integration_history.append({
            'metabolic_objectives': metabolic_objectives,
            'physics_objectives': physics_objectives,
            'integrated_objectives': integrated_objectives,
            'gsm_summary': self.gsm_model.get_metabolic_summary()
        })
        
        return integrated_objectives
    
    def get_optimization_targets_gsm(self) -> Dict[str, float]:
        """Get GSM-enhanced optimization targets for electrode optimization."""
        
        if not self.integration_history:
            # Run with default conditions if no integration has occurred
            default_results = {
                'performance_metrics': {
                    'avg_substrate_mM': 25.0,
                    'avg_current_density_A_m2': 0.0,
                    'avg_biofilm_density_kg_m3': 10.0
                }
            }
            return self.integrate_with_electrode_model(default_results)
        
        # Return most recent integrated objectives
        return self.integration_history[-1]['integrated_objectives']

if __name__ == "__main__":
    # Example usage
    print("🧬 Phase 4: GSM Integration Example")
    print("=" * 50)
    
    # Create GSM model
    config = GSMModelConfig()
    gsm_model = ShewanellaGSMModel(config)
    
    # Solve metabolic network
    fluxes = gsm_model.solve_fba()
    print(f"Electron production rate: {gsm_model.current_electron_production:.3f} mmol/g AFDW/h")
    print(f"Growth rate: {gsm_model.current_growth_rate:.3f} h⁻¹")
    
    # Get metabolic objectives
    objectives = gsm_model.calculate_metabolic_objectives()
    print("\nMetabolic objectives:")
    for name, value in objectives.items():
        print(f"  {name}: {value:.3f}")
    
    # Create integrator
    integrator = GSMPhysicsIntegrator(gsm_model)
    
    # Test integration with dummy electrode results
    dummy_electrode_results = {
        'performance_metrics': {
            'avg_substrate_mM': 20.0,
            'avg_current_density_A_m2': 0.5,
            'avg_biofilm_density_kg_m3': 15.0
        }
    }
    
    # Integrate with electrode model
    integrated_objectives = integrator.integrate_with_electrode_model(dummy_electrode_results)
    print("\nIntegrated objectives:")
    for name, value in integrated_objectives.items():
        print(f"  {name}: {value:.3f}")
    
    print("\n✅ GSM integration example completed!")