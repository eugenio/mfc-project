#!/usr/bin/env python3
"""
Membrane configuration for MFC simulations

Defines membrane materials, properties, and configuration options
including literature-based parameters for common MFC membranes.

Created: 2025-08-03
"""

from dataclasses import dataclass
from enum import Enum


class MembraneMaterial(Enum):
    """Common membrane materials used in MFCs."""
    NAFION_117 = "nafion_117"
    NAFION_112 = "nafion_112"
    NAFION_115 = "nafion_115"
    ULTREX_CMI_7000 = "ultrex_cmi_7000"
    FUMASEP_FKE = "fumasep_fke"
    FUMASEP_FAA = "fumasep_faa"
    CELLULOSE_ACETATE = "cellulose_acetate"
    BIPOLAR_MEMBRANE = "bipolar_membrane"
    CERAMIC_SEPARATOR = "ceramic_separator"
    J_CLOTH = "j_cloth"
    CUSTOM = "custom"


@dataclass
class MembraneProperties:
    """
    Membrane material properties with literature-based defaults.
    
    All properties should be based on published MFC literature.
    """

    # Ion transport properties
    proton_conductivity: float  # S/cm - proton conductivity at 25°C
    ion_exchange_capacity: float  # meq/g - ion exchange capacity
    permselectivity: float  # dimensionless (0-1) - cation selectivity

    # Physical properties
    thickness: float  # μm - membrane thickness
    water_uptake: float  # % - water uptake percentage
    density: float  # g/cm³ - dry membrane density

    # Transport resistances
    area_resistance: float  # Ω·cm² - area-specific resistance
    oxygen_permeability: float  # cm²/s - oxygen crossover coefficient
    substrate_permeability: float  # cm²/s - substrate crossover coefficient

    # Mechanical properties
    tensile_strength: float | None = None  # MPa - mechanical strength
    max_operating_temp: float = 60.0  # °C - maximum temperature

    # Cost and lifetime
    cost_per_m2: float | None = None  # $/m² - material cost
    expected_lifetime: float = 1000.0  # hours - operational lifetime

    # Literature reference
    reference: str = "User specified"


@dataclass
class MembraneConfiguration:
    """Complete membrane configuration."""

    material: MembraneMaterial
    properties: MembraneProperties
    area: float  # m² - membrane active area

    # Operational conditions
    operating_temperature: float = 25.0  # °C
    ph_anode: float = 7.0  # pH at anode side
    ph_cathode: float = 7.0  # pH at cathode side

    def calculate_resistance(self) -> float:
        """
        Calculate membrane resistance based on area and properties.
        
        Returns:
            Resistance in Ω
        """
        # Convert area resistance from Ω·cm² to Ω·m²
        area_resistance_m2 = self.properties.area_resistance * 1e-4
        return area_resistance_m2 / self.area

    def calculate_proton_flux(self, current_density: float) -> float:
        """
        Calculate proton flux through membrane.
        
        Args:
            current_density: Current density in A/m²
            
        Returns:
            Proton flux in mol/m²/s
        """
        faraday = 96485  # C/mol
        return current_density / faraday

    def estimate_lifetime_factor(self, current_density: float) -> float:
        """
        Estimate lifetime reduction factor based on operating conditions.
        
        Args:
            current_density: Operating current density in A/m²
            
        Returns:
            Lifetime factor (0-1), where 1 is full expected lifetime
        """
        # Higher current density reduces lifetime
        current_factor = 1.0 / (1.0 + current_density / 1000.0)

        # Temperature effect (every 10°C doubles degradation rate)
        temp_factor = 2.0 ** ((25.0 - self.operating_temperature) / 10.0)

        # pH gradient effect
        ph_gradient = abs(self.ph_anode - self.ph_cathode)
        ph_factor = 1.0 / (1.0 + ph_gradient / 4.0)

        return current_factor * temp_factor * ph_factor


# Literature-based membrane properties database
MEMBRANE_PROPERTIES_DATABASE = {
    MembraneMaterial.NAFION_117: MembraneProperties(
        proton_conductivity=0.10,  # S/cm - Kim et al. (2007) Environ. Sci. Technol.
        ion_exchange_capacity=0.9,  # meq/g - DuPont specification
        permselectivity=0.95,  # High cation selectivity
        thickness=175,  # μm - Standard thickness
        water_uptake=38,  # % - at 25°C
        density=1.98,  # g/cm³ - dry density
        area_resistance=1.5,  # Ω·cm² - in 0.5M NaCl
        oxygen_permeability=1.8e-12,  # cm²/s - Chae et al. (2008)
        substrate_permeability=5e-14,  # cm²/s - acetate crossover
        tensile_strength=43,  # MPa
        max_operating_temp=80,  # °C
        cost_per_m2=700,  # $/m² - approximate 2024 price
        expected_lifetime=5000,  # hours
        reference="Kim, J.R. et al. (2007) Environ. Sci. Technol. 41, 1004-1009"
    ),

    MembraneMaterial.NAFION_112: MembraneProperties(
        proton_conductivity=0.08,  # S/cm - Thinner, slightly lower conductivity
        ion_exchange_capacity=0.9,  # meq/g
        permselectivity=0.94,
        thickness=50,  # μm - Thinner variant
        water_uptake=35,  # %
        density=1.98,  # g/cm³
        area_resistance=0.6,  # Ω·cm² - Lower due to thickness
        oxygen_permeability=2.5e-12,  # cm²/s - Higher due to thickness
        substrate_permeability=8e-14,  # cm²/s
        tensile_strength=35,  # MPa
        max_operating_temp=80,  # °C
        cost_per_m2=500,  # $/m²
        expected_lifetime=3000,  # hours - shorter due to thickness
        reference="Chae, K.J. et al. (2008) Water Res. 42, 1501-1510"
    ),

    MembraneMaterial.ULTREX_CMI_7000: MembraneProperties(
        proton_conductivity=0.02,  # S/cm - Lower than Nafion
        ion_exchange_capacity=1.4,  # meq/g - Higher IEC
        permselectivity=0.98,  # Very high selectivity
        thickness=450,  # μm - Thick membrane
        water_uptake=25,  # % - Lower water uptake
        density=1.5,  # g/cm³
        area_resistance=8.0,  # Ω·cm² - Higher resistance
        oxygen_permeability=5e-13,  # cm²/s - Lower permeability
        substrate_permeability=1e-14,  # cm²/s - Very low
        tensile_strength=50,  # MPa
        max_operating_temp=60,  # °C
        cost_per_m2=150,  # $/m² - Much cheaper than Nafion
        expected_lifetime=8000,  # hours - Longer lifetime
        reference="Zhang, F. et al. (2011) Energy Environ. Sci. 4, 4340-4346"
    ),

    MembraneMaterial.FUMASEP_FKE: MembraneProperties(
        proton_conductivity=0.05,  # S/cm
        ion_exchange_capacity=1.2,  # meq/g
        permselectivity=0.96,
        thickness=125,  # μm
        water_uptake=30,  # %
        density=1.6,  # g/cm³
        area_resistance=2.5,  # Ω·cm²
        oxygen_permeability=8e-13,  # cm²/s
        substrate_permeability=3e-14,  # cm²/s
        tensile_strength=45,  # MPa
        max_operating_temp=70,  # °C
        cost_per_m2=300,  # $/m²
        expected_lifetime=6000,  # hours
        reference="Sleutels, T. et al. (2013) Int. J. Hydrogen Energy 38, 7201-7208"
    ),

    MembraneMaterial.J_CLOTH: MembraneProperties(
        proton_conductivity=0.001,  # S/cm - Very low, not ion-selective
        ion_exchange_capacity=0.0,  # meq/g - No ion exchange
        permselectivity=0.5,  # No selectivity
        thickness=700,  # μm - Cloth thickness
        water_uptake=150,  # % - High water absorption
        density=0.8,  # g/cm³ - Low density fabric
        area_resistance=0.2,  # Ω·cm² - Very low resistance
        oxygen_permeability=1e-8,  # cm²/s - High permeability
        substrate_permeability=1e-9,  # cm²/s - High crossover
        tensile_strength=20,  # MPa
        max_operating_temp=40,  # °C
        cost_per_m2=5,  # $/m² - Very cheap
        expected_lifetime=500,  # hours - Short lifetime
        reference="Fan, Y. et al. (2007) Environ. Sci. Technol. 41, 8154-8158"
    )
}


def create_membrane_config(
    material: MembraneMaterial,
    area: float,
    custom_properties: MembraneProperties | None = None
) -> MembraneConfiguration:
    """
    Create membrane configuration with material selection.
    
    Args:
        material: Membrane material type
        area: Membrane active area in m²
        custom_properties: Optional custom properties (for CUSTOM material)
        
    Returns:
        Complete membrane configuration
    """
    if material == MembraneMaterial.CUSTOM:
        if custom_properties is None:
            raise ValueError("Custom properties required for CUSTOM membrane material")
        properties = custom_properties
    else:
        properties = MEMBRANE_PROPERTIES_DATABASE.get(material)
        if properties is None:
            raise ValueError(f"No properties defined for {material.value}")

    return MembraneConfiguration(
        material=material,
        properties=properties,
        area=area
    )
