"""
Electron shuttle model for mediator-based electron transfer.

This module implements electron shuttle dynamics including flavins,
cytochromes, and other mediators used by exoelectrogenic bacteria
for extracellular electron transfer.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ShuttleType(Enum):
    """Types of electron shuttles."""
    FLAVIN_MONONUCLEOTIDE = "FMN"
    RIBOFLAVIN = "RF"
    FLAVIN_ADENINE_DINUCLEOTIDE = "FAD"
    CYTOCHROME_C = "CYT_C"
    PHENAZINE = "PHZ"
    HUMIC_ACID = "HA"


@dataclass
class ElectronShuttle:
    """Container for electron shuttle properties."""
    
    name: str                    # Shuttle name
    type: ShuttleType           # Shuttle type
    molecular_weight: float      # g/mol
    
    # Electrochemical properties
    redox_potential: float       # V vs SHE
    electrons_transferred: int   # e- per molecule
    diffusion_coefficient: float # m²/s
    
    # Kinetic properties
    production_rate: float       # mmol/gDW/h - max production
    degradation_rate: float      # 1/h - degradation constant
    binding_constant: float      # μM - binding to cells/electrode
    
    # Species specificity
    producing_species: List[str]  # Species that produce this shuttle
    utilization_efficiency: Dict[str, float]  # Species-specific efficiency


class ElectronShuttleModel:
    """
    Model for electron shuttle dynamics in MFC systems.
    
    Features:
    - Multiple shuttle types with different properties
    - Production and degradation kinetics
    - Diffusion and mass transport
    - Species-specific shuttle production
    - Electrode interaction kinetics
    """
    
    def __init__(self):
        """Initialize electron shuttle model with literature data."""
        self.shuttles = self._load_shuttle_database()
        self.shuttle_concentrations = {}  # Current concentrations
        self.reset_concentrations()
    
    def _load_shuttle_database(self) -> Dict[ShuttleType, ElectronShuttle]:
        """Load electron shuttle properties from literature."""
        
        shuttle_db = {
            ShuttleType.FLAVIN_MONONUCLEOTIDE: ElectronShuttle(
                name="Flavin mononucleotide",
                type=ShuttleType.FLAVIN_MONONUCLEOTIDE,
                molecular_weight=456.34,
                redox_potential=-0.20,  # V vs SHE
                electrons_transferred=2,
                diffusion_coefficient=4.3e-10,  # m²/s
                production_rate=0.05,  # mmol/gDW/h (S. oneidensis)
                degradation_rate=0.01,  # 1/h
                binding_constant=10.0,  # μM
                producing_species=["shewanella_oneidensis"],
                utilization_efficiency={
                    "shewanella_oneidensis": 0.9,
                    "geobacter_sulfurreducens": 0.3
                }
            ),
            
            ShuttleType.RIBOFLAVIN: ElectronShuttle(
                name="Riboflavin",
                type=ShuttleType.RIBOFLAVIN,
                molecular_weight=376.36,
                redox_potential=-0.21,  # V vs SHE
                electrons_transferred=2,
                diffusion_coefficient=5.0e-10,  # m²/s
                production_rate=0.08,  # mmol/gDW/h (S. oneidensis)
                degradation_rate=0.02,  # 1/h
                binding_constant=5.0,   # μM
                producing_species=["shewanella_oneidensis"],
                utilization_efficiency={
                    "shewanella_oneidensis": 0.95,
                    "geobacter_sulfurreducens": 0.4
                }
            ),
            
            ShuttleType.FLAVIN_ADENINE_DINUCLEOTIDE: ElectronShuttle(
                name="Flavin adenine dinucleotide",
                type=ShuttleType.FLAVIN_ADENINE_DINUCLEOTIDE,
                molecular_weight=785.55,
                redox_potential=-0.22,  # V vs SHE
                electrons_transferred=2,
                diffusion_coefficient=3.5e-10,  # m²/s
                production_rate=0.02,  # mmol/gDW/h
                degradation_rate=0.015,  # 1/h
                binding_constant=20.0,   # μM
                producing_species=["shewanella_oneidensis"],
                utilization_efficiency={
                    "shewanella_oneidensis": 0.85,
                    "geobacter_sulfurreducens": 0.25
                }
            ),
            
            ShuttleType.CYTOCHROME_C: ElectronShuttle(
                name="c-type cytochrome",
                type=ShuttleType.CYTOCHROME_C,
                molecular_weight=12000,  # Da
                redox_potential=0.25,   # V vs SHE
                electrons_transferred=1,
                diffusion_coefficient=1e-11,  # m²/s (large protein)
                production_rate=0.001,  # mmol/gDW/h (mostly membrane-bound)
                degradation_rate=0.001,  # 1/h (stable)
                binding_constant=0.1,    # μM (tight binding)
                producing_species=["geobacter_sulfurreducens", "shewanella_oneidensis"],
                utilization_efficiency={
                    "geobacter_sulfurreducens": 0.95,
                    "shewanella_oneidensis": 0.8
                }
            ),
            
            ShuttleType.PHENAZINE: ElectronShuttle(
                name="Phenazine",
                type=ShuttleType.PHENAZINE,
                molecular_weight=180.21,
                redox_potential=-0.08,  # V vs SHE
                electrons_transferred=2,
                diffusion_coefficient=6e-10,  # m²/s
                production_rate=0.0,    # Not produced by our species
                degradation_rate=0.05,  # 1/h
                binding_constant=50.0,  # μM
                producing_species=[],   # Produced by Pseudomonas
                utilization_efficiency={
                    "geobacter_sulfurreducens": 0.5,
                    "shewanella_oneidensis": 0.6
                }
            )
        }
        
        return shuttle_db
    
    def reset_concentrations(self):
        """Reset shuttle concentrations to initial values."""
        self.shuttle_concentrations = {
            shuttle_type: 0.0 for shuttle_type in ShuttleType
        }
    
    def calculate_shuttle_production(self, species: str, biomass: float, 
                                   growth_rate: float, dt: float) -> Dict[ShuttleType, float]:
        """
        Calculate electron shuttle production rates.
        
        Args:
            species: Bacterial species name
            biomass: Biomass concentration (g/L)
            growth_rate: Specific growth rate (1/h)
            dt: Time step (h)
            
        Returns:
            Dictionary of shuttle production (mmol/L)
        """
        production = {}
        
        for shuttle_type, shuttle in self.shuttles.items():
            if species in shuttle.producing_species:
                # Production proportional to biomass and growth
                base_production = shuttle.production_rate * biomass
                
                # Growth-associated production
                growth_factor = 1.0 + 0.5 * growth_rate
                
                # Calculate production
                production[shuttle_type] = base_production * growth_factor * dt
            else:
                production[shuttle_type] = 0.0
        
        return production
    
    def calculate_shuttle_degradation(self, dt: float) -> Dict[ShuttleType, float]:
        """
        Calculate shuttle degradation/consumption.
        
        Args:
            dt: Time step (h)
            
        Returns:
            Dictionary of degradation amounts (mmol/L)
        """
        degradation = {}
        
        for shuttle_type, shuttle in self.shuttles.items():
            current_conc = self.shuttle_concentrations[shuttle_type]
            
            # First-order degradation
            degradation[shuttle_type] = (shuttle.degradation_rate * 
                                       current_conc * dt)
        
        return degradation
    
    def calculate_electron_transfer_rate(self, shuttle_type: ShuttleType,
                                       concentration: float,
                                       electrode_potential: float) -> float:
        """
        Calculate electron transfer rate via shuttle.
        
        Args:
            shuttle_type: Type of electron shuttle
            concentration: Shuttle concentration (mmol/L)
            electrode_potential: Electrode potential (V vs SHE)
            
        Returns:
            Electron transfer rate (mmol e-/L/h)
        """
        shuttle = self.shuttles[shuttle_type]
        
        # Nernst equation for driving force
        driving_force = electrode_potential - shuttle.redox_potential
        
        # Butler-Volmer kinetics (simplified)
        if driving_force > 0:  # Oxidation at electrode
            # Rate proportional to reduced shuttle concentration
            k_et = 10.0  # 1/h - electron transfer rate constant
            
            # Michaelis-Menten type kinetics
            rate = (k_et * concentration / 
                   (shuttle.binding_constant * 1e-3 + concentration))
            
            # Account for driving force
            rate *= float(1 - np.exp(-driving_force / 0.025))  # 25 mV at room temp
            
            # Electrons per shuttle molecule
            electron_rate = rate * shuttle.electrons_transferred
        else:
            electron_rate = 0.0
        
        return electron_rate
    
    def calculate_shuttle_diffusion(self, concentration_gradient: float,
                                  shuttle_type: ShuttleType,
                                  distance: float) -> float:
        """
        Calculate shuttle diffusion flux.
        
        Args:
            concentration_gradient: Concentration difference (mmol/L)
            shuttle_type: Type of electron shuttle
            distance: Diffusion distance (m)
            
        Returns:
            Diffusion flux (mmol/m²/s)
        """
        shuttle = self.shuttles[shuttle_type]
        
        # Fick's law: J = -D * (dC/dx)
        flux = shuttle.diffusion_coefficient * concentration_gradient / distance
        
        return flux
    
    def update_shuttle_concentrations(self, production: Dict[ShuttleType, float],
                                    degradation: Dict[ShuttleType, float],
                                    consumption: Dict[ShuttleType, float]):
        """
        Update shuttle concentrations based on production/consumption.
        
        Args:
            production: Production rates by type (mmol/L)
            degradation: Degradation amounts by type (mmol/L)
            consumption: Consumption at electrode (mmol/L)
        """
        for shuttle_type in ShuttleType:
            # Mass balance
            net_change = (production.get(shuttle_type, 0.0) - 
                         degradation.get(shuttle_type, 0.0) -
                         consumption.get(shuttle_type, 0.0))
            
            # Update concentration
            self.shuttle_concentrations[shuttle_type] += net_change
            
            # Ensure non-negative
            self.shuttle_concentrations[shuttle_type] = max(0.0, 
                self.shuttle_concentrations[shuttle_type])
    
    def get_total_electron_flux(self, electrode_potential: float) -> float:
        """
        Calculate total electron flux from all shuttles.
        
        Args:
            electrode_potential: Electrode potential (V vs SHE)
            
        Returns:
            Total electron flux (mmol e-/L/h)
        """
        total_flux = 0.0
        
        for shuttle_type, concentration in self.shuttle_concentrations.items():
            if concentration > 0:
                flux = self.calculate_electron_transfer_rate(
                    shuttle_type, concentration, electrode_potential
                )
                total_flux += flux
        
        return total_flux
    
    def get_dominant_shuttle(self) -> Optional[ShuttleType]:
        """Get the dominant electron shuttle by concentration."""
        if not any(self.shuttle_concentrations.values()):
            return None
        
        return max(self.shuttle_concentrations.items(), 
                  key=lambda x: x[1])[0]
    
    def calculate_shuttle_current_contribution(self, volume: float,
                                             electrode_area: float) -> float:
        """
        Calculate current contribution from electron shuttles.
        
        Args:
            volume: System volume (L)
            electrode_area: Electrode area (m²)
            
        Returns:
            Current density contribution (A/m²)
        """
        # Get total electron flux
        total_flux = self.get_total_electron_flux(-0.2)  # Typical anode potential
        
        # Convert to current
        # flux in mmol e-/L/h -> mol e-/m³/s
        flux_si = total_flux * 1e-3 / 3600 * 1000  # mol e-/m³/s
        
        # Current = flux * volume * F / area
        F = 96485  # C/mol
        current_density = flux_si * volume * F / electrode_area
        
        return current_density
    
    def get_shuttle_properties(self, shuttle_type: ShuttleType) -> Dict[str, Any]:
        """Get properties of specific shuttle."""
        shuttle = self.shuttles[shuttle_type]
        
        return {
            "name": shuttle.name,
            "molecular_weight": shuttle.molecular_weight,
            "redox_potential": shuttle.redox_potential,
            "electrons_transferred": shuttle.electrons_transferred,
            "diffusion_coefficient": shuttle.diffusion_coefficient,
            "current_concentration": self.shuttle_concentrations[shuttle_type]
        }
    
    def get_species_shuttle_efficiency(self, species: str) -> Dict[ShuttleType, float]:
        """Get shuttle utilization efficiency for a species."""
        efficiencies = {}
        
        for shuttle_type, shuttle in self.shuttles.items():
            if species in shuttle.utilization_efficiency:
                efficiencies[shuttle_type] = shuttle.utilization_efficiency[species]
            else:
                efficiencies[shuttle_type] = 0.0
        
        return efficiencies
    
    def estimate_optimal_shuttle_concentration(self, target_current: float,
                                             electrode_potential: float,
                                             volume: float,
                                             area: float) -> Dict[ShuttleType, float]:
        """
        Estimate optimal shuttle concentrations for target current.
        
        Args:
            target_current: Target current (A)
            electrode_potential: Electrode potential (V)
            volume: System volume (L)
            area: Electrode area (m²)
            
        Returns:
            Optimal concentrations by shuttle type (mmol/L)
        """
        F = 96485  # C/mol
        target_flux = target_current * area / (volume * F) * 3600 * 1000  # mmol e-/L/h
        
        optimal_conc = {}
        
        # Distribute target flux among available shuttles
        # Prefer shuttles with better kinetics
        total_weight = sum(1.0 / shuttle.binding_constant 
                         for shuttle in self.shuttles.values())
        
        for shuttle_type, shuttle in self.shuttles.items():
            # Weight by inverse of binding constant (lower Km = better)
            weight = (1.0 / shuttle.binding_constant) / total_weight
            
            # Target flux for this shuttle
            shuttle_target_flux = target_flux * weight
            
            # Back-calculate concentration
            # Simplified from Michaelis-Menten
            if shuttle_target_flux > 0:
                k_et = 10.0  # Rate constant
                driving_force = electrode_potential - shuttle.redox_potential
                
                if driving_force > 0:
                    factor = float(1 - np.exp(-driving_force / 0.025))
                    if factor > 0:
                        # Solve for concentration
                        # rate = k_et * C / (Km + C) * factor * n_electrons
                        # Rearranging: C = rate * Km / (k_et * factor * n - rate)
                        denominator = (k_et * factor * shuttle.electrons_transferred - 
                                     shuttle_target_flux)
                        
                        if denominator > 0:
                            optimal_conc[shuttle_type] = (
                                shuttle_target_flux * shuttle.binding_constant * 1e-3 / 
                                denominator
                            )
                        else:
                            optimal_conc[shuttle_type] = 100.0  # Max reasonable
                    else:
                        optimal_conc[shuttle_type] = 0.0
                else:
                    optimal_conc[shuttle_type] = 0.0
            else:
                optimal_conc[shuttle_type] = 0.0
        
        return optimal_conc