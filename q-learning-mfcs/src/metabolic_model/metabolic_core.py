"""
Core metabolic model integrating pathways, membrane transport, and electron shuttles.

This module provides the main interface for metabolic modeling in MFC systems,
combining all components into a comprehensive simulation framework.
"""

import numpy as np
from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass
import sys
import os

# Add path for GPU acceleration
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from gpu_acceleration import get_gpu_accelerator

from .pathway_database import PathwayDatabase, Species, Substrate, MetabolicReaction
from .membrane_transport import MembraneTransport
from .electron_shuttles import ElectronShuttleModel, ShuttleType

# Import biological configuration
try:
    from config.biological_config import (
        SpeciesMetabolicConfig, BacterialSpecies,
        get_geobacter_config, get_shewanella_config
    )
    from config.substrate_config import (
        ComprehensiveSubstrateConfig, SubstrateType,
        DEFAULT_SUBSTRATE_CONFIGS
    )
    from config.biological_validation import (
        validate_species_metabolic_config, validate_comprehensive_substrate_config
    )
except ImportError:
    # Fallback if configuration modules are not available
    SpeciesMetabolicConfig = None
    ComprehensiveSubstrateConfig = None


@dataclass
class MetabolicState:
    """Container for metabolic system state."""

    # Metabolite concentrations (mmol/L)
    metabolites: Dict[str, float]

    # Reaction fluxes (mmol/gDW/h)
    fluxes: Dict[str, float]

    # Energy state
    atp_production: float      # mmol/gDW/h
    nadh_production: float     # mmol/gDW/h
    electron_production: float  # mmol e-/gDW/h

    # Oxygen and proton fluxes
    oxygen_consumption: float   # mmol/L/h
    proton_production: float    # mmol/L/h

    # Performance metrics
    substrate_utilization: float  # fraction
    coulombic_efficiency: float   # fraction
    energy_efficiency: float      # fraction


class MetabolicModel:
    """
    Comprehensive metabolic model for MFC systems.
    
    Features:
    - Species-specific metabolic pathways
    - Substrate metabolism with accurate stoichiometry
    - Electron shuttle production and dynamics
    - Oxygen crossover effects
    - Real-time flux balance analysis
    - GPU acceleration support
    """

    def __init__(self, species: str = "mixed", substrate: str = "lactate",
                 membrane_type: str = "Nafion-117", use_gpu: bool = True,
                 species_config: Optional[SpeciesMetabolicConfig] = None,
                 substrate_config: Optional[ComprehensiveSubstrateConfig] = None):
        """
        Initialize metabolic model.
        
        Args:
            species: Bacterial species ("geobacter", "shewanella", "mixed")
            substrate: Substrate type ("acetate", "lactate")
            membrane_type: Nafion membrane grade
            use_gpu: Enable GPU acceleration
            species_config: Optional species metabolic configuration
            substrate_config: Optional substrate configuration
        """
        self.species_str = species
        self.substrate_str = substrate
        self.use_gpu = use_gpu

        # Store configuration objects
        self.species_config = species_config
        self.substrate_config = substrate_config

        # Load default configurations if not provided
        if SpeciesMetabolicConfig and self.species_config is None:
            if species == "geobacter":
                self.species_config = get_geobacter_config()
            elif species == "shewanella":
                self.species_config = get_shewanella_config()
            # For "mixed" or other cases, could blend configs or use default

        if ComprehensiveSubstrateConfig and self.substrate_config is None:
            substrate_type = SubstrateType.ACETATE if substrate == "acetate" else SubstrateType.LACTATE
            self.substrate_config = DEFAULT_SUBSTRATE_CONFIGS.get(substrate_type)

        # Validate configurations if available
        if self.species_config and validate_species_metabolic_config:
            validate_species_metabolic_config(self.species_config)
        if self.substrate_config and validate_comprehensive_substrate_config:
            validate_comprehensive_substrate_config(self.substrate_config)

        # Map string inputs to enums
        self._map_species_substrate()

        # Initialize components
        self.pathway_db = PathwayDatabase()
        self.membrane_model = MembraneTransport(membrane_type, use_gpu)
        self.shuttle_model = ElectronShuttleModel()

        # GPU acceleration
        self.gpu_acc = get_gpu_accelerator() if use_gpu else None
        self.gpu_available = self.gpu_acc.is_gpu_available() if self.gpu_acc else False

        # Load metabolic pathway
        self._load_pathway()

        # Initialize state
        self.reset_state()

    def _map_species_substrate(self):
        """Map string inputs to enum types."""
        species_map = {
            "geobacter": Species.GEOBACTER,
            "shewanella": Species.SHEWANELLA,
            "mixed": Species.GEOBACTER  # Default to Geobacter for mixed
        }

        substrate_map = {
            "acetate": Substrate.ACETATE,
            "lactate": Substrate.LACTATE
        }

        self.species = species_map.get(self.species_str, Species.GEOBACTER)
        self.substrate = substrate_map.get(self.substrate_str, Substrate.LACTATE)

        # For mixed cultures, we'll handle both species
        self.is_mixed = (self.species_str == "mixed")

    def _load_pathway(self):
        """Load metabolic pathway for current species/substrate."""
        if self.is_mixed:
            # Load both pathways for mixed culture
            self.pathway_geobacter = self.pathway_db.get_pathway(
                Species.GEOBACTER, self.substrate
            )
            self.pathway_shewanella = self.pathway_db.get_pathway(
                Species.SHEWANELLA, self.substrate
            )
            self.current_pathway = self.pathway_geobacter  # Default
        else:
            self.current_pathway = self.pathway_db.get_pathway(
                self.species, self.substrate
            )

    def reset_state(self):
        """Reset metabolic state to initial conditions."""
        # Initialize metabolite concentrations from configuration or defaults
        if self.species_config and hasattr(self.species_config, 'metabolite_concentrations'):
            self.metabolites = self.species_config.metabolite_concentrations.copy()
        else:
            # Fallback to hardcoded values
            self.metabolites = {
                "acetate": 10.0 if self.substrate == Substrate.ACETATE else 0.0,
                "lactate": 10.0 if self.substrate == Substrate.LACTATE else 0.0,
                "pyruvate": 0.0,
                "acetyl_coa": 0.0,
                "citrate": 0.0,
                "co2": 0.0,
                "nadh": 0.1,
                "nad_plus": 1.0,
                "atp": 5.0,
                "adp": 5.0,
                "cytochrome_c_red": 0.1,
                "cytochrome_c_ox": 0.9,
                "o2": 0.001,  # Low oxygen (anaerobic)
                "h_plus": 1e-7 * 1000  # pH 7.0 in mmol/L
            }

        # Initialize reaction fluxes
        self.fluxes = {
            reaction.id: 0.0
            for reaction in self.current_pathway.reactions
        }

        # Reset shuttle concentrations
        self.shuttle_model.reset_concentrations()

        # Initialize performance metrics
        self.coulombic_efficiency = 0.0
        self.energy_efficiency = 0.0

    def calculate_metabolic_fluxes(self, biomass: float, growth_rate: float,
                                  anode_potential: float, substrate_supply: float) -> Dict[str, float]:
        """
        Calculate metabolic fluxes using simplified FBA.
        
        Args:
            biomass: Biomass concentration (g/L)
            growth_rate: Specific growth rate (1/h)
            anode_potential: Anode potential (V)
            substrate_supply: Substrate supply rate (mmol/L/h)
            
        Returns:
            Dictionary of reaction fluxes
        """
        fluxes = {}

        # Get pathway reactions
        reactions = self.current_pathway.reactions

        for reaction in reactions:
            # Get kinetic parameters from configuration or fallback to reaction
            if (self.species_config and hasattr(self.species_config, 'reactions') and
                len(self.species_config.reactions) > 0):
                # Find matching reaction configuration
                reaction_config = next((r for r in self.species_config.reactions
                                      if r.id == reaction.id), None)
                if reaction_config:
                    vmax = reaction_config.kinetics.vmax
                    km = reaction_config.kinetics.km
                    ki = reaction_config.kinetics.ki
                else:
                    vmax = reaction.vmax
                    km = getattr(reaction, 'km_values', {})
                    ki = getattr(reaction, 'ki_values', {})
            else:
                vmax = reaction.vmax
                km = getattr(reaction, 'km_values', {})
                ki = getattr(reaction, 'ki_values', {})

            # Base flux from Vmax and biomass
            flux = vmax * biomass

            # Apply Michaelis-Menten kinetics for substrates
            if isinstance(km, dict):
                for metabolite, km_val in km.items():
                    if metabolite in self.metabolites:
                        conc = self.metabolites[metabolite]
                        flux *= conc / (km_val + conc)
            elif isinstance(km, float) and km > 0:
                # Use primary substrate concentration
                substrate_key = "acetate" if self.substrate == Substrate.ACETATE else "lactate"
                if substrate_key in self.metabolites:
                    conc = self.metabolites[substrate_key]
                    flux *= conc / (km + conc)

            # Apply inhibition kinetics
            if isinstance(ki, dict):
                for metabolite, ki_val in ki.items():
                    if metabolite in self.metabolites:
                        conc = self.metabolites[metabolite]
                        flux *= ki_val / (ki_val + conc)
            elif isinstance(ki, float) and ki > 0:
                # Apply general inhibition if available
                if "o2" in self.metabolites:
                    conc = self.metabolites["o2"]
                    flux *= ki / (ki + conc)

            # Growth rate dependency for biosynthetic reactions
            if "synthetase" in reaction.name.lower():
                flux *= (1 + growth_rate)

            # Thermodynamic constraints
            if reaction.reversible:
                # Calculate actual ΔG based on concentrations
                delta_g = self._calculate_reaction_delta_g(reaction)
                if delta_g > 0:
                    flux *= np.exp(-delta_g / (8.314 * 303))  # RT at 30°C

            # Electron transport dependency on anode potential
            if "electron" in reaction.name.lower():
                potential_factor = 1 / (1 + np.exp(-(anode_potential + 0.3) / 0.05))
                flux *= potential_factor

            # Apply bounds
            flux = max(reaction.lb, min(reaction.ub, flux))

            fluxes[reaction.id] = flux

        return fluxes

    def _calculate_reaction_delta_g(self, reaction: MetabolicReaction) -> float:
        """Calculate actual Gibbs free energy for reaction."""
        # Standard ΔG
        delta_g = reaction.delta_g0

        # Concentration correction (simplified)
        RT = 8.314 * 303 / 1000  # kJ/mol

        # Product/reactant ratio
        Q = 1.0
        for metabolite, coeff in reaction.stoichiometry.items():
            if metabolite in self.metabolites:
                conc = max(1e-6, self.metabolites[metabolite])  # Avoid log(0)
                if coeff > 0:  # Product
                    Q *= conc ** abs(coeff)
                else:  # Reactant
                    Q /= conc ** abs(coeff)

        # Actual ΔG = ΔG° + RT ln(Q)
        if Q > 0:
            delta_g += RT * np.log(Q)

        return delta_g

    def update_metabolite_concentrations(self, fluxes: Dict[str, float],
                                       dt: float, volume: float):
        """
        Update metabolite concentrations based on fluxes.
        
        Args:
            fluxes: Reaction fluxes (mmol/gDW/h)
            dt: Time step (h)
            volume: System volume (L)
        """
        # Calculate net production/consumption for each metabolite
        net_rates = {}

        for reaction_id, flux in fluxes.items():
            # Find reaction
            reaction = next((r for r in self.current_pathway.reactions
                           if r.id == reaction_id), None)

            if reaction:
                for metabolite, coeff in reaction.stoichiometry.items():
                    if metabolite in self.metabolites:
                        # Net rate = flux * stoichiometric coefficient
                        rate = flux * coeff
                        net_rates[metabolite] = net_rates.get(metabolite, 0) + rate

        # Update concentrations
        for metabolite, net_rate in net_rates.items():
            # dC/dt = rate (assuming well-mixed)
            self.metabolites[metabolite] += net_rate * dt

            # Ensure non-negative
            self.metabolites[metabolite] = max(0.0, self.metabolites[metabolite])

    def calculate_oxygen_crossover_effects(self, cathode_o2_conc: float,
                                         membrane_area: float,
                                         temperature: float = 303.0) -> float:
        """
        Calculate oxygen crossover and its metabolic effects.
        
        Args:
            cathode_o2_conc: Cathode oxygen concentration (mol/m³)
            membrane_area: Membrane area (m²)
            temperature: Temperature (K)
            
        Returns:
            Oxygen consumption rate due to crossover (mmol/L/h)
        """
        # Calculate oxygen flux through membrane
        anode_o2_conc = self.metabolites["o2"] / 1000  # Convert to mol/m³

        o2_flux = self.membrane_model.calculate_oxygen_crossover(
            anode_o2_conc, cathode_o2_conc, temperature
        )

        # Convert flux to concentration rate
        # flux in mol/m²/s -> mmol/L/h
        o2_rate = o2_flux * membrane_area * 3600 * 1000  # mmol/h

        return o2_rate

    def calculate_electron_shuttle_dynamics(self, biomass: float, growth_rate: float,
                                          electrode_potential: float, dt: float):
        """
        Calculate electron shuttle production and consumption.
        
        Args:
            biomass: Biomass concentration (g/L)
            growth_rate: Specific growth rate (1/h)
            electrode_potential: Electrode potential (V)
            dt: Time step (h)
        """
        # Get species name for shuttle production
        species_name = self.species.value if not self.is_mixed else "shewanella_oneidensis"

        # Calculate shuttle production
        production = self.shuttle_model.calculate_shuttle_production(
            species_name, biomass, growth_rate, dt
        )

        # Calculate degradation
        degradation = self.shuttle_model.calculate_shuttle_degradation(dt)

        # Calculate consumption at electrode
        consumption = {}
        for shuttle_type in ShuttleType:
            conc = self.shuttle_model.shuttle_concentrations[shuttle_type]
            if conc > 0:
                # Electron transfer rate
                et_rate = self.shuttle_model.calculate_electron_transfer_rate(
                    shuttle_type, conc, electrode_potential
                )
                # Consumption = et_rate / electrons_per_shuttle * dt
                shuttle = self.shuttle_model.shuttles[shuttle_type]
                consumption[shuttle_type] = et_rate / shuttle.electrons_transferred * dt

        # Update concentrations
        self.shuttle_model.update_shuttle_concentrations(
            production, degradation, consumption
        )

    def calculate_current_output(self, biomass: float, volume: float,
                               electrode_area: float) -> Tuple[float, float]:
        """
        Calculate total current output from metabolism.
        
        Args:
            biomass: Biomass concentration (g/L)
            volume: System volume (L)
            electrode_area: Electrode area (m²)
            
        Returns:
            Tuple of (direct_current, mediated_current) in A/m²
        """
        # Get electron transport efficiency from configuration
        if self.species_config:
            et_efficiency = self.species_config.electron_transport_efficiency
        else:
            et_efficiency = 0.85  # Default fallback

        # Direct electron transfer (cytochrome-based)
        cytochrome_flux = self.fluxes.get("GSU_R004", 0.0)  # Electron transport reaction

        if cytochrome_flux > 0:
            # flux in mmol e-/gDW/h -> A/m²
            # flux * biomass * volume * F * efficiency / (3600 * area)
            F = 96485  # C/mol
            direct_current = (cytochrome_flux * biomass * volume * F * et_efficiency) / (3600 * electrode_area)
        else:
            direct_current = 0.0

        # Mediated electron transfer (shuttles)
        mediated_current = self.shuttle_model.calculate_shuttle_current_contribution(
            volume, electrode_area
        )

        return direct_current, mediated_current

    def calculate_coulombic_efficiency(self, current_output: float,
                                     substrate_consumed: float,
                                     dt: float) -> float:
        """
        Calculate coulombic efficiency.
        
        Args:
            current_output: Total current (A)
            substrate_consumed: Substrate consumption (mmol)
            dt: Time period (h)
            
        Returns:
            Coulombic efficiency (fraction)
        """
        if substrate_consumed <= 0:
            return 0.0

        # Get theoretical electron yield from configuration
        if (self.substrate_config and hasattr(self.substrate_config, 'degradation_pathways') and
            len(self.substrate_config.degradation_pathways) > 0):
            electrons_per_substrate = self.substrate_config.degradation_pathways[0].electron_yield
        else:
            electrons_per_substrate = self.current_pathway.electron_yield

        theoretical_electrons = substrate_consumed * electrons_per_substrate  # mmol e-

        # Actual electrons from current
        F = 96485  # C/mol
        actual_electrons = (current_output * dt * 3600) / F * 1000  # mmol e-

        efficiency = actual_electrons / theoretical_electrons

        # Use yield coefficient from configuration for more realistic bounds
        if self.species_config:
            max_efficiency = min(0.9, self.species_config.yield_coefficient * 10)  # Scale yield coefficient
            min_efficiency = max(0.05, self.species_config.yield_coefficient * 0.5)
        else:
            max_efficiency = 0.9
            min_efficiency = 0.1

        # For realistic MFC operation, apply species-specific bounds
        if efficiency > 1.5 or efficiency < 0.001:
            # Use substrate utilization as a proxy for efficiency
            substrate_utilization = min(1.0, substrate_consumed / 0.1)  # Normalize to 0.1 mmol baseline
            efficiency = min_efficiency + (max_efficiency - min_efficiency) * substrate_utilization

        return min(max_efficiency, max(min_efficiency, efficiency))

    def step_metabolism(self, dt: float, biomass: float, growth_rate: float,
                      anode_potential: float, substrate_supply: float,
                      cathode_o2_conc: float, membrane_area: float,
                      volume: float, electrode_area: float) -> MetabolicState:
        """
        Step metabolic model forward by dt.
        
        Args:
            dt: Time step (h)
            biomass: Biomass concentration (g/L)
            growth_rate: Specific growth rate (1/h)
            anode_potential: Anode potential (V)
            substrate_supply: Substrate supply rate (mmol/L/h)
            cathode_o2_conc: Cathode O2 concentration (mol/m³)
            membrane_area: Membrane area (m²)
            volume: System volume (L)
            electrode_area: Electrode area (m²)
            
        Returns:
            MetabolicState with current system state
        """
        # Add substrate supply
        if self.substrate == Substrate.ACETATE:
            self.metabolites["acetate"] += substrate_supply * dt
        else:
            self.metabolites["lactate"] += substrate_supply * dt

        # Calculate metabolic fluxes
        self.fluxes = self.calculate_metabolic_fluxes(
            biomass, growth_rate, anode_potential, substrate_supply
        )

        # Update metabolite concentrations
        self.update_metabolite_concentrations(self.fluxes, dt, volume)

        # Calculate oxygen crossover effects
        o2_crossover = self.calculate_oxygen_crossover_effects(
            cathode_o2_conc, membrane_area
        )
        self.metabolites["o2"] += o2_crossover * dt

        # Update electron shuttles
        self.calculate_electron_shuttle_dynamics(
            biomass, growth_rate, anode_potential, dt
        )

        # Calculate current output
        direct_current_density, mediated_current_density = self.calculate_current_output(
            biomass, volume, electrode_area
        )
        # Convert current density (A/m²) to actual current (A)
        total_current = (direct_current_density + mediated_current_density) * electrode_area

        # Calculate actual substrate consumption based on metabolic flux
        substrate_key = "acetate" if self.substrate == Substrate.ACETATE else "lactate"

        # Calculate actual substrate consumption from pathway flux
        actual_substrate_flux = 0.0
        for reaction in self.current_pathway.reactions:
            if substrate_key in reaction.stoichiometry and reaction.stoichiometry[substrate_key] < 0:
                # Negative stoichiometry means consumption
                substrate_flux_contribution = abs(reaction.stoichiometry[substrate_key]) * self.fluxes.get(reaction.id, 0.0)
                actual_substrate_flux += substrate_flux_contribution

        # Convert flux to consumption (mmol substrate consumed)
        substrate_consumed = actual_substrate_flux * biomass * dt


        # Fallback: if no flux calculated, use simplified approximation based on current
        if substrate_consumed <= 0:
            # Estimate from current output and theoretical yield
            electrons_from_current = (total_current * dt * 3600) / 96485 * 1000  # mmol e-
            electrons_per_substrate = self.current_pathway.electron_yield
            substrate_consumed = electrons_from_current / electrons_per_substrate if electrons_per_substrate > 0 else 0

        # Calculate efficiencies
        self.coulombic_efficiency = self.calculate_coulombic_efficiency(
            total_current, substrate_consumed, dt
        )

        # Energy calculations
        atp_flux = sum(flux * reaction.stoichiometry.get("atp", 0)
                      for reaction, flux in zip(self.current_pathway.reactions,
                                              self.fluxes.values()))

        nadh_flux = sum(flux * reaction.stoichiometry.get("nadh", 0)
                       for reaction, flux in zip(self.current_pathway.reactions,
                                               self.fluxes.values()))

        electron_flux = sum(flux * reaction.stoichiometry.get("electron_anode", 0)
                          for reaction, flux in zip(self.current_pathway.reactions,
                                                  self.fluxes.values()))

        # Create state summary
        state = MetabolicState(
            metabolites=self.metabolites.copy(),
            fluxes=self.fluxes.copy(),
            atp_production=atp_flux,
            nadh_production=nadh_flux,
            electron_production=electron_flux,
            oxygen_consumption=o2_crossover,
            proton_production=self.metabolites["h_plus"],
            substrate_utilization=1.0 - (self.metabolites[substrate_key] / 10.0),
            coulombic_efficiency=self.coulombic_efficiency,
            energy_efficiency=self.coulombic_efficiency * 0.4  # Simplified
        )

        return state

    def get_metabolic_summary(self) -> Dict[str, Any]:
        """Get summary of current metabolic state."""
        substrate_key = "acetate" if self.substrate == Substrate.ACETATE else "lactate"

        # Dominant shuttle
        dominant_shuttle = self.shuttle_model.get_dominant_shuttle()

        return {
            "species": self.species_str,
            "substrate": self.substrate_str,
            "is_mixed_culture": self.is_mixed,
            "substrate_concentration": self.metabolites[substrate_key],
            "nadh_ratio": self.metabolites["nadh"] / (self.metabolites["nadh"] +
                                                     self.metabolites["nad_plus"]),
            "atp_level": self.metabolites["atp"],
            "oxygen_level": self.metabolites["o2"],
            "dominant_shuttle": dominant_shuttle.value if dominant_shuttle else "None",
            "coulombic_efficiency": self.coulombic_efficiency,
            "gpu_acceleration": self.gpu_available
        }

    def optimize_for_current_production(self, target_current: float,
                                      constraints: Dict[str, float]) -> Dict[str, float]:
        """
        Optimize metabolic fluxes for maximum current production.
        
        Args:
            target_current: Target current density (A/m²)
            constraints: Constraints on metabolites/fluxes
            
        Returns:
            Optimal flux distribution
        """
        # Simplified optimization - maximize electron-producing reactions
        optimal_fluxes = {}

        for reaction in self.current_pathway.reactions:
            # Check if reaction produces electrons
            if reaction.stoichiometry.get("electron_anode", 0) > 0:
                # Maximize within bounds
                optimal_fluxes[reaction.id] = reaction.ub
            else:
                # Set to moderate level
                optimal_fluxes[reaction.id] = (reaction.lb + reaction.ub) / 2

        return optimal_fluxes
