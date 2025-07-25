"""
Biofilm Kinetics Model Implementation

This module implements the comprehensive biofilm formation and growth model
for exoelectrogenic bacteria with species selection, substrate selection,
and environmental compensation.
"""

import numpy as np
from typing import Dict, Optional, Any
import sys
import os

# Add path for GPU acceleration
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from gpu_acceleration import get_gpu_accelerator

from .species_params import SpeciesParameters
from .substrate_params import SubstrateParameters

# Import biological configuration
try:
    from config.biological_config import (
        SpeciesMetabolicConfig, BiofilmKineticsConfig, BacterialSpecies,
        get_geobacter_config, get_shewanella_config, get_default_biofilm_config
    )
    from config.substrate_config import (
        ComprehensiveSubstrateConfig, SubstrateType,
        DEFAULT_SUBSTRATE_CONFIGS
    )
    from config.biological_validation import (
        validate_species_metabolic_config, validate_biofilm_kinetics_config
    )
except ImportError:
    # Fallback if configuration modules are not available
    SpeciesMetabolicConfig = None
    BiofilmKineticsConfig = None
    ComprehensiveSubstrateConfig = None


class BiofilmKineticsModel:
    """
    Comprehensive biofilm kinetics model with species and substrate selection.
    
    Features:
    - Species selection: G. sulfurreducens, S. oneidensis, mixed cultures
    - Substrate selection: acetate, lactate (default)
    - pH and temperature compensation
    - GPU acceleration support
    - Real-time parameter adaptation
    """
    
    def __init__(self, species: str = 'mixed', substrate: str = 'lactate',
                 use_gpu: bool = True, temperature: float = 303.0, ph: float = 7.0,
                 species_config: Optional[SpeciesMetabolicConfig] = None,
                 biofilm_config: Optional[BiofilmKineticsConfig] = None,
                 substrate_config: Optional[ComprehensiveSubstrateConfig] = None):
        """
        Initialize biofilm kinetics model.
        
        Args:
            species: Species type ('geobacter', 'shewanella', 'mixed')
            substrate: Substrate type ('acetate', 'lactate')
            use_gpu: Enable GPU acceleration
            temperature: Operating temperature (K)
            ph: Operating pH
            species_config: Optional species metabolic configuration
            biofilm_config: Optional biofilm kinetics configuration
            substrate_config: Optional substrate configuration
        """
        self.species = species
        self.substrate = substrate
        self.temperature = temperature
        self.ph = ph
        
        # Store configuration objects
        self.species_config = species_config
        self.biofilm_config = biofilm_config
        self.substrate_config = substrate_config
        
        # Load default configurations if not provided
        if SpeciesMetabolicConfig and self.species_config is None:
            if species == "geobacter":
                self.species_config = get_geobacter_config()
            elif species == "shewanella":
                self.species_config = get_shewanella_config()
            # For "mixed" or other cases, could use default or blend configs
        
        if BiofilmKineticsConfig and self.biofilm_config is None:
            self.biofilm_config = get_default_biofilm_config()
            
        if ComprehensiveSubstrateConfig and self.substrate_config is None:
            substrate_type = SubstrateType.ACETATE if substrate == "acetate" else SubstrateType.LACTATE
            self.substrate_config = DEFAULT_SUBSTRATE_CONFIGS.get(substrate_type)
        
        # Validate configurations if available
        if self.species_config and validate_species_metabolic_config:
            validate_species_metabolic_config(self.species_config)
        if self.biofilm_config and validate_biofilm_kinetics_config:
            validate_biofilm_kinetics_config(self.biofilm_config)
        
        # Initialize parameter databases (fallback)
        self.species_db = SpeciesParameters()
        self.substrate_db = SubstrateParameters()
        
        # GPU acceleration setup
        self.use_gpu = use_gpu
        self.gpu_acc = get_gpu_accelerator() if use_gpu else None
        self.gpu_available = self.gpu_acc.is_gpu_available() if self.gpu_acc else False
        
        # Load and compensate parameters
        self._load_parameters()
        
        # Initialize state variables
        self.reset_state()
    
    def _load_parameters(self):
        """Load and apply environmental compensation to parameters."""
        # Get base parameters from configuration or fallback
        if self.species_config and self.biofilm_config:
            # Use configuration parameters
            self.mu_max = self.species_config.max_growth_rate
            self.K_s = self.biofilm_config.monod_kinetics['half_saturation']
            self.Y_xs = self.biofilm_config.monod_kinetics['yield_coefficient']
            self.j_max = self.biofilm_config.nernst_monod['electron_transfer_rate'] * 10.0  # Convert to mA/cm²
            self.sigma_biofilm = self.biofilm_config.nernst_monod['biofilm_conductivity']
            self.E_ka = self.biofilm_config.nernst_monod['standard_potential']
            self.E_an = -0.5  # Typical anode potential
            self.attachment_prob = self.species_config.attachment_rate
            self.biofilm_thickness_max = self.species_config.max_biofilm_thickness
        else:
            # Fallback to legacy parameter database
            self.kinetic_params = self.species_db.get_parameters(self.species)
            self.mu_max = getattr(self.kinetic_params, 'mu_max', 0.3)
            self.K_s = getattr(self.kinetic_params, 'K_s', 0.5)
            self.Y_xs = getattr(self.kinetic_params, 'Y_xs', 0.1)
            self.j_max = getattr(self.kinetic_params, 'j_max', 10.0)
            self.sigma_biofilm = getattr(self.kinetic_params, 'sigma_biofilm', 0.005)
            self.E_ka = getattr(self.kinetic_params, 'E_ka', -0.3)
            self.E_an = getattr(self.kinetic_params, 'E_an', -0.5)
            self.attachment_prob = getattr(self.kinetic_params, 'attachment_prob', 0.1)
            self.biofilm_thickness_max = getattr(self.kinetic_params, 'biofilm_thickness_max', 100.0)
        
        # Get substrate properties
        if self.substrate_config:
            self.substrate_molecular_weight = self.substrate_config.molecular_weight
        else:
            self.substrate_props = self.substrate_db.get_substrate_properties(self.substrate)
            self.substrate_molecular_weight = getattr(self.substrate_props, 'molecular_weight', 90.08)
        
        # Apply temperature compensation (Arrhenius equation)
        if self.temperature != 303.0:  # Reference temperature
            temp_factor = np.exp(-50000 / 8.314 * (1/self.temperature - 1/303.0))  # Ea = 50 kJ/mol
            self.mu_max *= temp_factor
        
        # Apply pH compensation (simple Gaussian)
        if self.ph != 7.0:  # Reference pH
            ph_optimal = 7.0
            ph_factor = np.exp(-0.5 * ((self.ph - ph_optimal) / 1.0)**2)
            self.mu_max *= ph_factor
    
    def reset_state(self):
        """Reset biofilm state variables."""
        self.biofilm_thickness = 0.1  # μm - initial thickness
        self.biomass_density = 0.01   # kg/m³ - initial density
        self.substrate_concentration = 10.0  # mmol/L - initial concentration
        self.attached_fraction = 0.0  # Fraction of cells attached
        self.current_density = 0.0    # A/m² - current output
        self.time = 0.0              # h - simulation time
    
    def calculate_nernst_monod_growth_rate(self, substrate_conc: float, 
                                         anode_potential: float) -> float:
        """
        Calculate growth rate using Nernst-Monod kinetics.
        
        Args:
            substrate_conc: Substrate concentration (mmol/L)
            anode_potential: Anode potential (V)
            
        Returns:
            Specific growth rate (1/h)
        """
        # Nernst-Monod model: μ = μ_max * (S/(K_s + S)) * (E_a - E_ka)/(E_ka - E_an)
        
        # Substrate limitation term
        substrate_term = substrate_conc / (self.K_s + substrate_conc)
        
        # Electrochemical limitation term
        potential_numerator = anode_potential - self.E_ka
        potential_denominator = self.E_ka - self.E_an
        
        # Avoid division by zero
        if abs(potential_denominator) < 1e-6:
            electrochemical_term = 1.0
        else:
            electrochemical_term = max(0.0, potential_numerator / potential_denominator)
        
        growth_rate = (self.mu_max * substrate_term * electrochemical_term)
        
        return max(0.0, growth_rate)
    
    def calculate_stochastic_attachment(self, cell_density: float, 
                                      surface_area: float) -> float:
        """
        Calculate stochastic cell attachment using probability matrix.
        
        Args:
            cell_density: Planktonic cell density (cells/m³)
            surface_area: Available surface area (m²)
            
        Returns:
            Attachment rate (cells/(m²·h))
        """
        # Base attachment probability from configuration
        base_prob = self.attachment_prob
        
        # Environmental corrections
        if hasattr(self, 'substrate_db'):
            ph_correction = self.substrate_db.apply_ph_correction(
                self.substrate, 1.0, self.ph
            )
        else:
            # Simple pH correction
            ph_correction = np.exp(-0.5 * ((self.ph - 7.0) / 1.0)**2)
        
        # Surface coverage effect (reduced attachment as biofilm grows)
        coverage_factor = 1.0 - (self.biofilm_thickness / self.biofilm_thickness_max)
        coverage_factor = max(0.1, coverage_factor)  # Minimum 10% attachment rate
        
        # Calculate attachment rate
        attachment_rate = (base_prob * ph_correction * coverage_factor * cell_density)
        
        return attachment_rate
    
    def calculate_biofilm_current_density(self, thickness: float, 
                                        biomass_density: float) -> float:
        """
        Calculate current density from biofilm parameters.
        
        Args:
            thickness: Biofilm thickness (μm)
            biomass_density: Biomass density (kg/m³)
            
        Returns:
            Current density (A/m²)
        """
        # Convert thickness to meters
        thickness_m = thickness * 1e-6
        
        # Current density with biofilm resistance effects
        # j = j_max * (biomass/max_biomass) * (1 - R_biofilm * thickness)
        
        max_biomass = 50.0  # kg/m³ typical maximum
        biomass_factor = min(1.0, biomass_density / max_biomass)
        
        # Biofilm resistance effect using configuration parameters
        resistance_factor = 1.0 / (1.0 + thickness_m / (self.sigma_biofilm * 1e-3))
        
        # Calculate current density (convert from mA/cm² to A/m²)
        current_density = (self.j_max * 1e-3 * 1e4 * biomass_factor * resistance_factor)
        
        return current_density
    
    def calculate_substrate_consumption(self, growth_rate: float, 
                                      biomass: float) -> float:
        """
        Calculate substrate consumption rate.
        
        Args:
            growth_rate: Specific growth rate (1/h)
            biomass: Biomass concentration (kg/m³)
            
        Returns:
            Substrate consumption rate (mmol/(L·h))
        """
        # Consumption = growth_rate * biomass / yield_coefficient
        consumption_rate = growth_rate * biomass / self.Y_xs
        
        # Convert to mmol/(L·h) using molecular weight from configuration
        consumption_mmol = consumption_rate * 1000 / self.substrate_molecular_weight  # mmol/(L·h)
        
        return consumption_mmol
    
    def calculate_mixed_culture_synergy(self, geobacter_current: float, 
                                      shewanella_current: float) -> float:
        """
        Calculate synergy effect in mixed cultures.
        
        Args:
            geobacter_current: Current from G. sulfurreducens (A/m²)
            shewanella_current: Current from S. oneidensis (A/m²)
            
        Returns:
            Enhanced total current with synergy (A/m²)
        """
        if self.species != 'mixed':
            return geobacter_current + shewanella_current
        
        # Mixed culture synergy: j_mixed = j_Gs + α * j_So * f_synergy
        alpha = self.species_db.get_synergy_coefficient('geobacter', 'shewanella')
        
        # Synergy efficiency factor (depends on species ratio and conditions)
        ph_factor = self.substrate_db.apply_ph_correction(self.substrate, 1.0, self.ph)
        temp_factor = max(1.0, min(1.5, self.temperature / 303.0))  # Temperature enhancement
        
        synergy_efficiency = ph_factor * temp_factor
        
        # Use total current with synergy enhancement
        total_individual = geobacter_current + shewanella_current
        enhanced_current = total_individual * alpha * synergy_efficiency
        
        return enhanced_current
    
    def step_biofilm_dynamics(self, dt: float, anode_potential: float, 
                            substrate_supply: float = 0.0) -> Dict[str, float]:
        """
        Step biofilm dynamics forward by time dt.
        
        Args:
            dt: Time step (h)
            anode_potential: Anode potential (V)
            substrate_supply: Substrate supply rate (mmol/(L·h))
            
        Returns:
            Dictionary of state variables and outputs
        """
        # Use GPU arrays if available
        if self.gpu_available:
            # Convert to GPU arrays for computation
            thickness_gpu = self.gpu_acc.array([self.biofilm_thickness])
            biomass_gpu = self.gpu_acc.array([self.biomass_density])
            substrate_gpu = self.gpu_acc.array([self.substrate_concentration])
        
        # Calculate growth rate
        growth_rate = self.calculate_nernst_monod_growth_rate(
            self.substrate_concentration, anode_potential
        )
        
        # Calculate current density
        self.current_density = self.calculate_biofilm_current_density(
            self.biofilm_thickness, self.biomass_density
        )
        
        # Calculate substrate consumption
        consumption_rate = self.calculate_substrate_consumption(
            growth_rate, self.biomass_density
        )
        
        # Update substrate concentration
        dC_substrate_dt = substrate_supply - consumption_rate
        self.substrate_concentration += dC_substrate_dt * dt
        self.substrate_concentration = max(0.01, self.substrate_concentration)  # Minimum
        
        # Update biomass density
        # dX/dt = μ * X - k_death * X
        death_rate = 0.01  # 1/h typical death rate
        dX_dt = growth_rate * self.biomass_density - death_rate * self.biomass_density
        self.biomass_density += dX_dt * dt
        self.biomass_density = max(0.001, self.biomass_density)  # Minimum
        
        # Update biofilm thickness
        # Thickness growth proportional to biomass growth and attachment
        attachment_rate = self.calculate_stochastic_attachment(
            cell_density=1e12,  # cells/m³ typical planktonic density
            surface_area=1.0    # m² reference area
        )
        
        # More conservative thickness growth to avoid runaway growth
        thickness_growth_rate = min(0.05 * growth_rate, 0.5)  # μm/h with cap
        if self.biomass_density > 1.0:  # Only grow when sufficient biomass
            self.biofilm_thickness += thickness_growth_rate * dt
        self.biofilm_thickness = min(self.biofilm_thickness, self.biofilm_thickness_max)
        
        # Update time
        self.time += dt
        
        # Return current state
        return {
            'time': self.time,
            'biofilm_thickness': self.biofilm_thickness,
            'biomass_density': self.biomass_density,
            'substrate_concentration': self.substrate_concentration,
            'current_density': self.current_density,
            'growth_rate': growth_rate,
            'consumption_rate': consumption_rate,
            'anode_potential': anode_potential
        }
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """Get current model parameters for inspection."""
        return {
            'species': self.species,
            'substrate': self.substrate,
            'temperature': self.temperature,
            'ph': self.ph,
            'kinetic_params': self.kinetic_params.__dict__,
            'substrate_props': self.substrate_props.__dict__,
            'gpu_available': self.gpu_available
        }
    
    def set_environmental_conditions(self, temperature: Optional[float] = None, 
                                   ph: Optional[float] = None):
        """
        Update environmental conditions and recompute parameters.
        
        Args:
            temperature: New temperature (K)
            ph: New pH
        """
        if temperature is not None:
            self.temperature = temperature
        if ph is not None:
            self.ph = ph
        
        # Reload parameters with new conditions
        self._load_parameters()
    
    def calculate_theoretical_maximum_current(self) -> float:
        """
        Calculate theoretical maximum current density.
        
        Returns:
            Maximum theoretical current density (A/m²)
        """
        # Theoretical current from substrate consumption at maximum rate
        max_consumption = (self.kinetic_params.mu_max * 50.0 /  # max biomass
                         self.kinetic_params.Y_xs)
        
        theoretical_current = self.substrate_db.calculate_theoretical_current(
            self.substrate, max_consumption
        )
        
        return theoretical_current
    
    def get_mass_balance_equations(self) -> Dict[str, str]:
        """Get mass balance equations for current configuration."""
        return {
            'substrate_equation': self.substrate_db.get_mass_balance_equation(self.substrate),
            'biomass_balance': r"\frac{dX}{dt} = \mu \cdot X - k_{death} \cdot X",
            'biofilm_thickness': r"\frac{dL}{dt} = k_{growth} \cdot \mu \cdot r_{attachment}",
            'nernst_monod': r"\mu = \mu_{max} \cdot \frac{S}{K_s + S} \cdot \frac{E_a - E_{ka}}{E_{ka} - E_{an}}"
        }