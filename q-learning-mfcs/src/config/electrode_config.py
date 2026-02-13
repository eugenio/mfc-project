#!/usr/bin/env python3
"""
Electrode Configuration Classes for MFC Modeling

This module defines comprehensive electrode configurations including:
- Material-specific properties (conductance, surface charge, hydrophobicity)
- Geometry-based surface area calculations
- Microbial attachment characteristics
- Charge transfer properties

Created: 2025-08-01
Literature References:
1. Logan, B.E. (2008). "Microbial Fuel Cells: Methodology and Technology"
2. Wei, J. et al. (2011). "A comprehensive study of electrode materials for microbial fuel cells"
3. Santoro, C. et al. (2017). "Microbial fuel cells: From fundamentals to applications"
4. Erable, B. et al. (2012). "Anode materials for microbial fuel cells"
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ElectrodeMaterial(Enum):
    """Electrode material types with specific properties."""
    GRAPHITE_PLATE = "graphite_plate"
    GRAPHITE_ROD = "graphite_rod"
    CARBON_FELT = "carbon_felt"
    CARBON_CLOTH = "carbon_cloth"
    CARBON_PAPER = "carbon_paper"
    STAINLESS_STEEL = "stainless_steel"
    PLATINUM = "platinum"
    GOLD = "gold"
    CUSTOM = "custom"


class ElectrodeGeometry(Enum):
    """Electrode geometry types."""
    RECTANGULAR_PLATE = "rectangular_plate"
    CYLINDRICAL_ROD = "cylindrical_rod"
    CYLINDRICAL_TUBE = "cylindrical_tube"
    SPHERICAL = "spherical"
    CUSTOM = "custom"


@dataclass
class MaterialProperties:
    """Material-specific electrode properties from literature."""

    # Electrical properties
    specific_conductance: float  # S/m - electrical conductivity
    contact_resistance: float  # Ω·cm² - electrode-electrolyte interface resistance

    # Surface properties
    surface_charge_density: float  # C/m² - surface charge at neutral pH
    hydrophobicity_angle: float  # degrees - water contact angle
    surface_roughness: float  # dimensionless - relative to smooth surface

    # Microbial attachment properties
    biofilm_adhesion_coefficient: float  # dimensionless - relative to graphite
    attachment_energy: float  # kJ/mol - microbial attachment energy

    # Surface area properties
    specific_surface_area: float | None = None  # m²/m³ - for porous materials
    porosity: float | None = None  # dimensionless - void fraction for porous materials

    # Material density
    density: float | None = None  # kg/m³ - material density

    # Literature reference
    reference: str = "User specified"


@dataclass
class ElectrodeGeometrySpec:
    """Electrode geometry specifications."""

    geometry_type: ElectrodeGeometry

    # Dimensions (in meters)
    length: float | None = None  # m
    width: float | None = None   # m
    height: float | None = None  # m
    diameter: float | None = None  # m
    thickness: float | None = None  # m

    # For custom geometries
    specific_surface_area: float | None = None  # m²/g - manually specified
    total_surface_area: float | None = None  # m² - manually specified

    # Material density
    density: float | None = None  # kg/m³ - material density
    def calculate_specific_surface_area(self) -> float:
        """Calculate specific surface area based on geometry type."""
        if self.specific_surface_area is not None:
            return self.specific_surface_area

        if self.geometry_type == ElectrodeGeometry.RECTANGULAR_PLATE:
            if self.length and self.width:
                return self.length * self.width

        elif self.geometry_type == ElectrodeGeometry.CYLINDRICAL_ROD:
            if self.diameter and self.length:
                return math.pi * (self.diameter / 2) ** 2

        elif self.geometry_type == ElectrodeGeometry.CYLINDRICAL_TUBE:
            if self.diameter and self.length:
                return math.pi * (self.diameter / 2) ** 2  # Cross-sectional area

        elif self.geometry_type == ElectrodeGeometry.SPHERICAL:
            if self.diameter:
                return math.pi * (self.diameter / 2) ** 2  # Great circle area

        raise ValueError(f"Insufficient dimensions for {self.geometry_type}")

    def calculate_mass(self) -> float:
        """Calculate electrode mass based on volume and density."""
        if self.density is None:
            raise ValueError("Density not specified for mass calculation")

        volume = self.calculate_volume()
        return volume * self.density  # kg

    def calculate_total_surface_area(self) -> float:
        """Calculate total surface area available for microbial colonization."""
        if self.total_surface_area is not None:
            return self.total_surface_area

        if self.geometry_type == ElectrodeGeometry.RECTANGULAR_PLATE:
            if self.length and self.width and self.thickness:
                # All surfaces exposed
                return 2 * (self.length * self.width +
                          self.length * self.thickness +
                          self.width * self.thickness)

        elif self.geometry_type == ElectrodeGeometry.CYLINDRICAL_ROD:
            if self.diameter and self.length:
                # Cylindrical surface + end caps
                return (math.pi * self.diameter * self.length +
                        2 * math.pi * (self.diameter / 2) ** 2)

        elif self.geometry_type == ElectrodeGeometry.CYLINDRICAL_TUBE:
            if self.diameter and self.length and self.thickness:
                # Inner and outer cylindrical surfaces + annular ends
                outer_area = math.pi * self.diameter * self.length
                inner_area = math.pi * (self.diameter - 2 * self.thickness) * self.length
                end_area = 2 * math.pi * ((self.diameter / 2) ** 2 -
                                        ((self.diameter - 2 * self.thickness) / 2) ** 2)
                return outer_area + inner_area + end_area

        elif self.geometry_type == ElectrodeGeometry.SPHERICAL:
            if self.diameter:
                return 4 * math.pi * (self.diameter / 2) ** 2

        raise ValueError(f"Insufficient dimensions for {self.geometry_type}")

    def calculate_volume(self) -> float:
        """Calculate electrode volume."""
        if self.geometry_type == ElectrodeGeometry.RECTANGULAR_PLATE:
            if self.length and self.width and self.thickness:
                return self.length * self.width * self.thickness

        elif self.geometry_type == ElectrodeGeometry.CYLINDRICAL_ROD:
            if self.diameter and self.length:
                return math.pi * (self.diameter / 2) ** 2 * self.length

        elif self.geometry_type == ElectrodeGeometry.CYLINDRICAL_TUBE:
            if self.diameter and self.length and self.thickness:
                outer_vol = math.pi * (self.diameter / 2) ** 2 * self.length
                inner_vol = math.pi * ((self.diameter - 2 * self.thickness) / 2) ** 2 * self.length
                return outer_vol - inner_vol

        elif self.geometry_type == ElectrodeGeometry.SPHERICAL:
            if self.diameter:
                return (4/3) * math.pi * (self.diameter / 2) ** 3

        raise ValueError(f"Insufficient dimensions for {self.geometry_type}")


@dataclass
class ElectrodeConfiguration:
    """Complete electrode configuration with material and geometry."""

    material: ElectrodeMaterial
    geometry: ElectrodeGeometrySpec
    material_properties: MaterialProperties

    # Operational parameters
    operating_potential: float = -0.4  # V vs SHE - typical anode potential
    surface_treatment: str = "none"  # Surface treatment applied
    age_hours: float = 0.0  # Hours of operation (affects biofilm development)

    def calculate_effective_surface_area(self) -> float:
        """
        Calculate effective surface area for microbial colonization.
        Accounts for material-specific surface area enhancement.
        """
        specific_surface_area = self.geometry.calculate_specific_surface_area()

        # For porous materials, use specific surface area
        if self.material_properties.specific_surface_area is not None:
            volume = self.geometry.calculate_volume()
            return specific_surface_area + (self.material_properties.specific_surface_area * volume)
        else:
            # For non-porous materials, use geometric surface area
            total_area = self.geometry.calculate_total_surface_area()
            # Apply surface roughness factor
            return total_area * self.material_properties.surface_roughness

    def calculate_biofilm_capacity(self) -> float:
        """Calculate maximum biofilm capacity based on electrode properties."""
        effective_area = self.calculate_effective_surface_area()

        # Base biofilm thickness (literature: 10-200 μm for MFC)
        base_thickness_m = 50e-6  # 50 μm

        # Adjust based on material properties
        adhesion_factor = self.material_properties.biofilm_adhesion_coefficient
        hydrophobicity_factor = 1.0 - (self.material_properties.hydrophobicity_angle - 60) / 180

        effective_thickness = base_thickness_m * adhesion_factor * max(0.1, hydrophobicity_factor)

        return effective_area * effective_thickness  # m³ biofilm volume

    def calculate_charge_transfer_coefficient(self) -> float:
        """Calculate charge transfer coefficient based on material properties."""
        # Base charge transfer coefficient (dimensionless, 0-1)
        base_coefficient = 0.5

        # Adjust based on conductance (higher conductance = better charge transfer)
        conductance_factor = min(1.0, self.material_properties.specific_conductance / 1000)  # Normalize to 1000 S/m

        # Adjust based on contact resistance (lower resistance = better transfer)
        resistance_factor = max(0.1, 1.0 / (1.0 + self.material_properties.contact_resistance))

        return base_coefficient * conductance_factor * resistance_factor

    def get_configuration_summary(self) -> dict[str, Any]:
        """Get a summary of electrode configuration for display."""
        return {
            'material': self.material.value,
            'geometry': self.geometry.geometry_type.value,
            'projected_area_cm2': self.geometry.calculate_specific_surface_area() * 10000,  # Convert to cm²
            'effective_area_cm2': self.calculate_effective_surface_area() * 10000,  # Convert to cm²
            'biofilm_capacity_ul': self.calculate_biofilm_capacity() * 1e9,  # Convert to μL
            'charge_transfer_coeff': self.calculate_charge_transfer_coefficient(),
            'specific_conductance_S_per_m': self.material_properties.specific_conductance,
            'hydrophobicity_angle_deg': self.material_properties.hydrophobicity_angle,
            'surface_roughness': self.material_properties.surface_roughness
        }


# Predefined material properties from literature
MATERIAL_PROPERTIES_DATABASE = {
    ElectrodeMaterial.GRAPHITE_PLATE: MaterialProperties(
        specific_conductance=25000,  # S/m - Graphite conductivity
        contact_resistance=0.1,  # Ω·cm² - Low contact resistance
        surface_charge_density=-0.05,  # C/m² - Slightly negative
        hydrophobicity_angle=75,  # degrees - Moderately hydrophobic
        surface_roughness=1.2,  # 20% rougher than smooth
        biofilm_adhesion_coefficient=1.0,  # Reference material
        attachment_energy=-12.5,
        density=2200,  # kg/m³
        reference="Logan, B.E. (2008). Microbial Fuel Cells"
    ),

    ElectrodeMaterial.GRAPHITE_ROD: MaterialProperties(
        specific_conductance=25000,  # S/m - Same as plate
        contact_resistance=0.12,  # Ω·cm² - Slightly higher due to geometry
        surface_charge_density=-0.05,  # C/m²
        hydrophobicity_angle=75,  # degrees
        surface_roughness=1.1,  # Smoother than plate
        biofilm_adhesion_coefficient=0.95,  # Slightly lower due to geometry
        attachment_energy=-12.0,
        density=2200,  # kg/m³
        reference="Logan, B.E. (2008). Microbial Fuel Cells"
    ),

    ElectrodeMaterial.CARBON_FELT: MaterialProperties(
        specific_conductance=500,  # S/m - Lower than graphite
        contact_resistance=0.8,  # Ω·cm² - Higher due to fiber structure
        surface_charge_density=-0.08,  # C/m² - More negative
        hydrophobicity_angle=85,  # degrees - More hydrophobic
        surface_roughness=15.0,  # Very high surface area
        biofilm_adhesion_coefficient=2.5,  # Excellent for biofilm
        attachment_energy=-18.0,
        density=120,  # kg/m³
        specific_surface_area=1500,  # m²/m³ - High specific surface area
        porosity=0.95,  # 95% void space
        reference="Wei, J. et al. (2011). Biosens. Bioelectron."
    ),

    ElectrodeMaterial.CARBON_CLOTH: MaterialProperties(
        specific_conductance=800,  # S/m
        contact_resistance=0.6,  # Ω·cm²
        surface_charge_density=-0.06,  # C/m²
        hydrophobicity_angle=80,  # degrees
        surface_roughness=8.0,  # High surface area
        biofilm_adhesion_coefficient=2.0,  # Very good
        attachment_energy=-15.5,
        density=400,  # kg/m³
        specific_surface_area=800,  # m²/m³
        porosity=0.85,  # 85% void space
        reference="Santoro, C. et al. (2017). Chem. Soc. Rev."
    ),

    ElectrodeMaterial.CARBON_PAPER: MaterialProperties(
        specific_conductance=1200,  # S/m
        contact_resistance=0.4,  # Ω·cm²
        surface_charge_density=-0.04,  # C/m²
        hydrophobicity_angle=90,  # degrees - Hydrophobic
        surface_roughness=5.0,  # Moderate surface area
        biofilm_adhesion_coefficient=1.5,  # Good
        attachment_energy=-14.0,
        density=450,  # kg/m³
        specific_surface_area=400,  # m²/m³
        porosity=0.70,  # 70% void space
        reference="Erable, B. et al. (2012). Electrochem. Commun."
    ),

    ElectrodeMaterial.STAINLESS_STEEL: MaterialProperties(
        specific_conductance=1400000,  # S/m - Very high conductivity
        contact_resistance=0.05,  # Ω·cm² - Very low
        surface_charge_density=0.02,  # C/m² - Slightly positive
        hydrophobicity_angle=45,  # degrees - Hydrophilic
        surface_roughness=1.0,  # Smooth
        biofilm_adhesion_coefficient=0.3,  # Poor biofilm adhesion
        attachment_energy=-5.0,
        density=7850,  # kg/m³
        reference="Torres, C.I. et al. (2010). Environ. Sci. Technol."
    ),

    ElectrodeMaterial.PLATINUM: MaterialProperties(
        specific_conductance=9600000,  # S/m - Excellent conductivity
        contact_resistance=0.01,  # Ω·cm² - Extremely low
        surface_charge_density=0.0,  # C/m² - Neutral
        hydrophobicity_angle=40,  # degrees - Hydrophilic
        surface_roughness=1.0,  # Smooth
        biofilm_adhesion_coefficient=0.2,  # Poor for biofilm
        attachment_energy=-3.0,
        density=21450,  # kg/m³
        reference="Rabaey, K. & Verstraete, W. (2005). Trends Biotechnol."
    )
}


def create_electrode_config(
    material: ElectrodeMaterial,
    geometry_type: ElectrodeGeometry,
    dimensions: dict[str, float],
    custom_properties: MaterialProperties | None = None
) -> ElectrodeConfiguration:
    """
    Create an electrode configuration with specified material and geometry.

    Args:
        material: Electrode material type
        geometry_type: Electrode geometry type
        dimensions: Dictionary of dimensions (length, width, height, diameter, thickness)
        custom_properties: Custom material properties (optional)

    Returns:
        Complete electrode configuration
    """
    # Get material properties
    if custom_properties is not None:
        material_props = custom_properties
    else:
        material_props_lookup = MATERIAL_PROPERTIES_DATABASE.get(material)
        if material_props_lookup is None:
            raise ValueError(f"No predefined properties for material {material}")
        material_props = material_props_lookup

    # Create geometry specification
    geometry = ElectrodeGeometrySpec(
        geometry_type=geometry_type,
        length=dimensions.get('length'),
        width=dimensions.get('width'),
        height=dimensions.get('height'),
        diameter=dimensions.get('diameter'),
        thickness=dimensions.get('thickness'),
        specific_surface_area=dimensions.get('specific_surface_area'),
        total_surface_area=dimensions.get('total_surface_area')
    )

    return ElectrodeConfiguration(
        material=material,
        geometry=geometry,
        material_properties=material_props
    )


# Default electrode configurations for common setups
DEFAULT_GRAPHITE_PLATE_CONFIG = create_electrode_config(
    material=ElectrodeMaterial.GRAPHITE_PLATE,
    geometry_type=ElectrodeGeometry.RECTANGULAR_PLATE,
    dimensions={
        'length': 0.05,    # 5 cm
        'width': 0.05,     # 5 cm
        'thickness': 0.005  # 5 mm
    }
)

DEFAULT_CARBON_FELT_CONFIG = create_electrode_config(
    material=ElectrodeMaterial.CARBON_FELT,
    geometry_type=ElectrodeGeometry.RECTANGULAR_PLATE,
    dimensions={
        'length': 0.03,    # 3 cm
        'width': 0.03,     # 3 cm
        'thickness': 0.01   # 1 cm
    }
)
