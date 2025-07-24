"""
Species-specific parameters for biofilm kinetics modeling.

This module contains experimental parameters for G. sulfurreducens,
S. oneidensis MR-1, and mixed cultures based on literature values
from the MATHEMATICAL_MODELS_CELL_ATTACHMENT.md documentation.
"""

import numpy as np
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class KineticParameters:
    """Container for species-specific kinetic parameters."""
    
    # Growth kinetics
    mu_max: float  # Maximum specific growth rate (1/h)
    K_s: float     # Half-saturation constant (mmol/L)
    Y_xs: float    # Yield coefficient (g biomass/g substrate)
    
    # Electrochemical parameters
    j_max: float   # Maximum current density (mA/cm²)
    sigma_biofilm: float  # Biofilm conductivity (S/m)
    E_ka: float    # Half-saturation potential (V)
    E_an: float    # Potential at which growth ceases (V)
    
    # Biofilm formation
    attachment_prob: float  # Base attachment probability
    biofilm_thickness_max: float  # Maximum biofilm thickness (μm)
    diffusion_coeff: float  # Substrate diffusion coefficient in biofilm (m²/s)
    
    # Temperature compensation
    activation_energy: float  # Activation energy (kJ/mol)
    temp_ref: float  # Reference temperature (K)


class SpeciesParameters:
    """Species-specific parameter database with literature values."""
    
    def __init__(self):
        """Initialize parameter database with literature values."""
        self._parameters = self._load_species_parameters()
    
    def _load_species_parameters(self) -> Dict[str, KineticParameters]:
        """Load species-specific parameters from literature."""
        
        # G. sulfurreducens parameters (from literature compilation)
        geobacter_params = KineticParameters(
            # Growth kinetics
            mu_max=0.15,  # 1/h (Nernst-Monod model)
            K_s=0.5,      # mmol/L acetate (half-saturation)
            Y_xs=0.083,   # g biomass/g acetate (yield coefficient)
            
            # Electrochemical parameters  
            j_max=0.39,   # mA/cm² (maximum current density ± 0.09)
            sigma_biofilm=1e-4,  # S/m (biofilm conductivity range 10⁻⁵ to 10⁻³)
            E_ka=-0.3,    # V (half-saturation potential)
            E_an=-0.5,    # V (potential at which growth ceases)
            
            # Biofilm formation
            attachment_prob=0.7,  # Base attachment probability on surfaces
            biofilm_thickness_max=69.0,  # μm (pure culture)
            diffusion_coeff=1e-9,  # m²/s (substrate diffusion in biofilm)
            
            # Temperature compensation
            activation_energy=65.0,  # kJ/mol (typical for bacterial growth)
            temp_ref=303.0  # K (30°C reference temperature)
        )
        
        # S. oneidensis MR-1 parameters (from literature compilation)
        shewanella_params = KineticParameters(
            # Growth kinetics
            mu_max=0.12,  # 1/h (lower than G. sulfurreducens)
            K_s=1.0,      # mmol/L lactate (half-saturation)
            Y_xs=0.45,    # g biomass/g lactate (higher yield than G. sulfurreducens)
            
            # Electrochemical parameters
            j_max=0.034,  # mA/cm² (±0.011, much lower than G. sulfurreducens)
            sigma_biofilm=5e-5,  # S/m (lower conductivity)
            E_ka=-0.25,   # V (slightly higher potential)
            E_an=-0.4,    # V (higher threshold)
            
            # Biofilm formation
            attachment_prob=0.5,  # Lower attachment probability (less stable biofilms)
            biofilm_thickness_max=35.0,  # μm (thinner biofilms)
            diffusion_coeff=1.2e-9,  # m²/s (slightly higher diffusion)
            
            # Temperature compensation
            activation_energy=58.0,  # kJ/mol (slightly lower activation energy)
            temp_ref=303.0  # K (30°C reference temperature)
        )
        
        # Mixed culture parameters (G. sulfurreducens + S. oneidensis)
        mixed_params = KineticParameters(
            # Growth kinetics (weighted average with synergy)
            mu_max=0.14,  # 1/h (between species values)
            K_s=0.75,     # mmol/L (compromise between acetate and lactate)
            Y_xs=0.25,    # g biomass/g substrate (intermediate yield)
            
            # Electrochemical parameters (synergy factor α = 1.38)
            j_max=0.54,   # mA/cm² (38% enhancement from synergy)
            sigma_biofilm=8e-5,  # S/m (intermediate conductivity)
            E_ka=-0.275,  # V (average of species values)
            E_an=-0.45,   # V (compromise threshold)
            
            # Biofilm formation
            attachment_prob=0.6,  # Intermediate attachment
            biofilm_thickness_max=93.0,  # μm (enhanced thickness in mixed culture)
            diffusion_coeff=1.1e-9,  # m²/s (intermediate diffusion)
            
            # Temperature compensation
            activation_energy=61.5,  # kJ/mol (average activation energy)
            temp_ref=303.0  # K (30°C reference temperature)
        )
        
        return {
            'geobacter': geobacter_params,
            'shewanella': shewanella_params,
            'mixed': mixed_params
        }
    
    def get_parameters(self, species: str) -> KineticParameters:
        """
        Get kinetic parameters for specified species.
        
        Args:
            species: Species name ('geobacter', 'shewanella', 'mixed')
            
        Returns:
            KineticParameters object with species-specific values
            
        Raises:
            ValueError: If species not recognized
        """
        if species not in self._parameters:
            available = list(self._parameters.keys())
            raise ValueError(f"Species '{species}' not recognized. Available: {available}")
        
        return self._parameters[species]
    
    def get_synergy_coefficient(self, species_1: str, species_2: str) -> float:
        """
        Get synergy coefficient for mixed cultures.
        
        Args:
            species_1: Primary species name
            species_2: Secondary species name
            
        Returns:
            Synergy coefficient (α = 1.38 for G. sulfurreducens + S. oneidensis)
        """
        mixed_pairs = {
            ('geobacter', 'shewanella'): 1.38,
            ('shewanella', 'geobacter'): 1.38
        }
        
        pair = (species_1, species_2)
        return mixed_pairs.get(pair, 1.0)  # Default to no synergy
    
    def apply_temperature_compensation(self, params: KineticParameters, 
                                     temperature: float) -> KineticParameters:
        """
        Apply Arrhenius temperature compensation to kinetic parameters.
        
        Args:
            params: Base kinetic parameters at reference temperature
            temperature: Operating temperature (K)
            
        Returns:
            Temperature-compensated parameters
        """
        # Arrhenius equation: k(T) = k_ref * exp(-Ea/R * (1/T - 1/T_ref))
        R = 8.314  # J/(mol·K) - Gas constant
        
        temp_factor = np.exp(-params.activation_energy * 1000 / R * 
                           (1/temperature - 1/params.temp_ref))
        
        # Apply temperature compensation to rate parameters
        compensated_params = KineticParameters(
            mu_max=params.mu_max * temp_factor,
            K_s=params.K_s,  # Saturation constant unchanged
            Y_xs=params.Y_xs,  # Yield coefficient unchanged
            j_max=params.j_max * temp_factor,
            sigma_biofilm=params.sigma_biofilm * temp_factor,
            E_ka=params.E_ka,  # Electrochemical potentials unchanged
            E_an=params.E_an,
            attachment_prob=params.attachment_prob * min(temp_factor, 1.0),
            biofilm_thickness_max=params.biofilm_thickness_max,
            diffusion_coeff=params.diffusion_coeff * temp_factor,
            activation_energy=params.activation_energy,
            temp_ref=params.temp_ref
        )
        
        return compensated_params
    
    def apply_ph_compensation(self, params: KineticParameters, pH: float) -> KineticParameters:
        """
        Apply pH compensation to electrochemical parameters.
        
        Args:
            params: Base kinetic parameters
            pH: Operating pH
            
        Returns:
            pH-compensated parameters
        """
        # Nernst equation pH compensation: E = E° - (RT/nF) * ln([H+])
        # For pH effects: ΔE = -0.059 * (pH - pH_ref) at 25°C
        pH_ref = 7.0  # Reference pH
        nernst_factor = -0.059  # V/pH unit at 25°C
        
        ph_compensation = nernst_factor * (pH - pH_ref)
        
        # Also apply pH effects to growth rate (optimal pH around 7.0-7.2)
        ph_deviation = abs(pH - 7.1)  # Optimal pH
        ph_growth_factor = np.exp(-0.5 * (ph_deviation / 1.5)**2)  # Gaussian response
        
        # Apply pH compensation to electrochemical parameters
        compensated_params = KineticParameters(
            mu_max=params.mu_max * ph_growth_factor,  # Growth affected by pH
            K_s=params.K_s,
            Y_xs=params.Y_xs,
            j_max=params.j_max * ph_growth_factor,  # Current also affected
            sigma_biofilm=params.sigma_biofilm,
            E_ka=params.E_ka + ph_compensation,
            E_an=params.E_an + ph_compensation,
            attachment_prob=params.attachment_prob,
            biofilm_thickness_max=params.biofilm_thickness_max,
            diffusion_coeff=params.diffusion_coeff,
            activation_energy=params.activation_energy,
            temp_ref=params.temp_ref
        )
        
        return compensated_params
    
    def list_available_species(self) -> list:
        """Return list of available species."""
        return list(self._parameters.keys())