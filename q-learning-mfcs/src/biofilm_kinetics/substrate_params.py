"""
Substrate-specific parameters for biofilm kinetics modeling.

This module contains parameters for different substrates (acetate, lactate)
and their utilization by exoelectrogenic bacteria.
"""

import numpy as np
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class SubstrateProperties:
    """Container for substrate-specific properties."""
    
    # Chemical properties
    molecular_weight: float  # g/mol
    density: float          # g/mL
    diffusivity: float      # m²/s in water at 25°C
    
    # Thermodynamic properties
    standard_potential: float  # V (standard reduction potential)
    gibbs_free_energy: float   # kJ/mol (ΔG° of oxidation)
    electrons_per_mole: int    # e⁻/mol substrate
    
    # Kinetic properties
    base_consumption_rate: float  # mol/(m³·h) baseline consumption
    mass_transfer_coeff: float    # m/s (liquid-biofilm interface)
    
    # pH dependencies
    ph_optimum: float         # Optimal pH for utilization
    ph_sensitivity: float     # pH sensitivity factor


class SubstrateParameters:
    """Substrate parameter database with thermodynamic and kinetic data."""
    
    def __init__(self):
        """Initialize substrate parameter database."""
        self._substrates = self._load_substrate_parameters()
        self._default_substrate = 'lactate'  # As specified in requirements
    
    def _load_substrate_parameters(self) -> Dict[str, SubstrateProperties]:
        """Load substrate-specific parameters."""
        
        # Acetate parameters (preferred by G. sulfurreducens)
        acetate_params = SubstrateProperties(
            # Chemical properties
            molecular_weight=82.03,    # g/mol (sodium acetate)
            density=1.528,             # g/mL
            diffusivity=1.09e-9,       # m²/s in water at 25°C
            
            # Thermodynamic properties (acetate oxidation)
            # CH₃COO⁻ + 4H₂O → 2CO₂ + 7H⁺ + 8e⁻
            standard_potential=-0.296,  # V vs SHE (acetate/CO₂)
            gibbs_free_energy=-844.5,   # kJ/mol (acetate oxidation)
            electrons_per_mole=8,       # 8 e⁻ per mole acetate
            
            # Kinetic properties
            base_consumption_rate=0.5,   # mol/(m³·h) typical consumption
            mass_transfer_coeff=1e-5,    # m/s at biofilm interface
            
            # pH dependencies
            ph_optimum=7.0,             # Optimal pH for acetate utilization
            ph_sensitivity=0.15         # pH sensitivity factor
        )
        
        # Lactate parameters (preferred by S. oneidensis, default substrate)
        lactate_params = SubstrateProperties(
            # Chemical properties
            molecular_weight=112.06,    # g/mol (sodium lactate)
            density=1.33,               # g/mL
            diffusivity=0.95e-9,        # m²/s in water at 25°C (slightly lower than acetate)
            
            # Thermodynamic properties (lactate oxidation)
            # C₃H₅O₃⁻ + 3H₂O → C₃H₃O₃⁻ + 5H⁺ + 4e⁻ (lactate → pyruvate)
            standard_potential=-0.190,  # V vs SHE (lactate/pyruvate)
            gibbs_free_energy=-516.2,   # kJ/mol (lactate oxidation)
            electrons_per_mole=4,       # 4 e⁻ per mole lactate
            
            # Kinetic properties
            base_consumption_rate=0.8,   # mol/(m³·h) higher consumption rate
            mass_transfer_coeff=1.2e-5,  # m/s slightly higher mass transfer
            
            # pH dependencies
            ph_optimum=7.2,             # Slightly higher optimal pH
            ph_sensitivity=0.12         # Less pH sensitive than acetate
        )
        
        return {
            'acetate': acetate_params,
            'lactate': lactate_params
        }
    
    def get_substrate_properties(self, substrate: str) -> SubstrateProperties:
        """
        Get properties for specified substrate.
        
        Args:
            substrate: Substrate name ('acetate', 'lactate')
            
        Returns:
            SubstrateProperties object
            
        Raises:
            ValueError: If substrate not recognized
        """
        if substrate not in self._substrates:
            available = list(self._substrates.keys())
            raise ValueError(f"Substrate '{substrate}' not recognized. Available: {available}")
        
        return self._substrates[substrate]
    
    def get_default_substrate(self) -> str:
        """Get default substrate (lactate as specified)."""
        return self._default_substrate
    
    def calculate_nernst_potential(self, substrate: str, concentration: float, 
                                 ph: float = 7.0, temperature: float = 298.15) -> float:
        """
        Calculate Nernst potential for substrate at given conditions.
        
        Args:
            substrate: Substrate name
            concentration: Substrate concentration (mol/L)
            ph: Solution pH
            temperature: Temperature (K)
            
        Returns:
            Nernst potential (V)
        """
        props = self.get_substrate_properties(substrate)
        
        # Nernst equation: E = E° - (RT/nF) * ln(Q)
        R = 8.314  # J/(mol·K)
        F = 96485  # C/mol
        
        # For substrate oxidation: Q = [products]/[reactants]
        # Simplified: Q ≈ [H⁺]^n / [substrate]
        h_concentration = 10**(-ph)
        
        nernst_term = (R * temperature) / (props.electrons_per_mole * F)
        
        if substrate == 'acetate':
            # CH₃COO⁻ + 4H₂O → 2CO₂ + 7H⁺ + 8e⁻
            # For oxidation reaction, Q should be inverted
            Q = concentration / (h_concentration**7)
        elif substrate == 'lactate':
            # C₃H₅O₃⁻ + 3H₂O → C₃H₃O₃⁻ + 5H⁺ + 4e⁻  
            # For oxidation reaction, Q should be inverted
            Q = concentration / (h_concentration**5)
        else:
            Q = 1.0  # Fallback
        
        # Ensure Q is positive and not zero
        Q = max(Q, 1e-10)
        
        nernst_potential = props.standard_potential - nernst_term * np.log(Q)
        
        return nernst_potential
    
    def calculate_theoretical_current(self, substrate: str, consumption_rate: float) -> float:
        """
        Calculate theoretical current from substrate consumption rate.
        
        Args:
            substrate: Substrate name
            consumption_rate: Consumption rate (mol/(m³·h))
            
        Returns:
            Theoretical current density (A/m²)
        """
        props = self.get_substrate_properties(substrate)
        
        # Current = n * F * consumption_rate
        # Convert h to s: consumption_rate / 3600
        F = 96485  # C/mol
        
        current_density = (props.electrons_per_mole * F * consumption_rate) / 3600
        
        return current_density
    
    def apply_ph_correction(self, substrate: str, value: float, 
                           current_ph: float) -> float:
        """
        Apply pH correction to kinetic parameters.
        
        Args:
            substrate: Substrate name
            value: Parameter value to correct
            current_ph: Current pH
            
        Returns:
            pH-corrected value
        """
        props = self.get_substrate_properties(substrate)
        
        # Gaussian pH response: correction = exp(-0.5 * ((pH - pH_opt)/sensitivity)²)
        ph_deviation = (current_ph - props.ph_optimum) / props.ph_sensitivity
        ph_correction = np.exp(-0.5 * ph_deviation**2)
        
        return value * ph_correction
    
    def get_stoichiometric_coefficients(self, substrate: str) -> Dict[str, float]:
        """
        Get stoichiometric coefficients for substrate oxidation.
        
        Args:
            substrate: Substrate name
            
        Returns:
            Dictionary of stoichiometric coefficients
        """
        if substrate == 'acetate':
            # CH₃COO⁻ + 4H₂O → 2CO₂ + 7H⁺ + 8e⁻
            return {
                'substrate': -1.0,
                'water': -4.0,
                'co2': 2.0,
                'protons': 7.0,
                'electrons': 8.0
            }
        elif substrate == 'lactate':
            # C₃H₅O₃⁻ + 3H₂O → C₃H₃O₃⁻ + 5H⁺ + 4e⁻
            return {
                'substrate': -1.0,
                'water': -3.0,
                'pyruvate': 1.0,
                'protons': 5.0,
                'electrons': 4.0
            }
        else:
            return {}
    
    def list_available_substrates(self) -> list:
        """Return list of available substrates."""
        return list(self._substrates.keys())
    
    def get_mass_balance_equation(self, substrate: str) -> str:
        """
        Get mass balance equation string for substrate.
        
        Args:
            substrate: Substrate name
            
        Returns:
            LaTeX-formatted mass balance equation
        """
        if substrate == 'acetate':
            return r"CH_3COO^- + 4H_2O \rightarrow 2CO_2 + 7H^+ + 8e^-"
        elif substrate == 'lactate':
            return r"C_3H_5O_3^- + 3H_2O \rightarrow C_3H_3O_3^- + 5H^+ + 4e^-"
        else:
            return "Unknown substrate equation"