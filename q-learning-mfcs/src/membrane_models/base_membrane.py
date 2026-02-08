#!/usr/bin/env python3
"""Base membrane model for MFC simulations.

Provides abstract base class for all membrane types with common transport
mechanisms, selectivity calculations, and performance metrics.

Key transport mechanisms:
- Ion transport (migration, diffusion, convection)
- Water transport (electro-osmotic drag, hydraulic permeation)
- Gas crossover (O2, H2, CO2)
- Multi-ion competition and selectivity

Created: 2025-07-27
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

try:
    import jax.numpy as jnp
except (ImportError, ModuleNotFoundError):
    import numpy as jnp  # noqa: ICN001


class IonType(Enum):
    """Types of ions that can transport through membranes."""

    PROTON = "H+"
    HYDROXIDE = "OH-"
    SODIUM = "Na+"
    POTASSIUM = "K+"
    CHLORIDE = "Cl-"
    BICARBONATE = "HCO3-"
    CARBONATE = "CO3--"
    SULFATE = "SO4--"
    PHOSPHATE = "PO4---"


class TransportMechanism(Enum):
    """Transport mechanisms through membranes."""

    MIGRATION = "migration"  # Electric field driven
    DIFFUSION = "diffusion"  # Concentration gradient driven
    CONVECTION = "convection"  # Pressure/flow driven
    ELECTROOSMOTIC = "electroosmotic"  # Water drag with ions


@dataclass
class MembraneParameters:
    """Base parameters for all membrane types."""

    # Physical properties
    thickness: float = 100e-6  # m - membrane thickness
    area: float = 1e-4  # m² - membrane area
    porosity: float = 0.3  # - void fraction
    tortuosity: float = 2.5  # - pore tortuosity
    pore_size: float = 1e-9  # m - average pore size

    # Operating conditions
    temperature: float = 298.15  # K - operating temperature
    pressure_anode: float = 101325  # Pa - anode pressure
    pressure_cathode: float = 101325  # Pa - cathode pressure

    # Ion exchange capacity
    ion_exchange_capacity: float = 1.2  # mol/kg - IEC
    water_content: float = 10.0  # mol H2O/mol functional group
    fixed_charge_density: float = 1200  # mol/m³

    # Physical constants
    faraday_constant: float = 96485.0  # C/mol
    gas_constant: float = 8.314  # J/(mol·K)

    # Degradation parameters
    initial_conductivity: float = 0.1  # S/cm - initial ionic conductivity
    degradation_rate: float = 1e-6  # h⁻¹ - conductivity loss rate
    fouling_resistance: float = 0.0  # Ω·m² - additional fouling resistance


@dataclass
class IonTransportMechanisms:
    """Ion transport properties and mechanisms."""

    ion_type: IonType
    diffusion_coefficient: float  # m²/s - in membrane
    mobility: float  # m²/(V·s) - ion mobility
    partition_coefficient: float  # - membrane/solution partition
    hydration_number: float  # - water molecules per ion
    charge: int  # - ion charge
    stokes_radius: float  # m - hydrated ion radius


class BaseMembraneModel(ABC):
    """Abstract base class for membrane models in MFC simulations.

    Provides common functionality for all membrane types:
    - Multi-ion transport calculations
    - Selectivity and permeability
    - Water transport
    - Gas crossover
    - Membrane resistance
    - Degradation effects
    """

    def __init__(self, parameters: MembraneParameters) -> None:
        self.params = parameters
        self.thickness = parameters.thickness
        self.area = parameters.area
        self.temperature = parameters.temperature
        self.operating_hours = 0.0

        # Setup ion transport mechanisms
        self._setup_ion_transport()

        # Calculate initial properties
        self._calculate_membrane_properties()

    @abstractmethod
    def _setup_ion_transport(self):
        """Setup membrane-specific ion transport mechanisms."""

    @abstractmethod
    def _calculate_membrane_properties(self):
        """Calculate membrane-specific properties."""

    def calculate_nernst_planck_flux(
        self,
        ion: IonType,
        concentration_anode: float,
        concentration_cathode: float,
        potential_gradient: float,
    ) -> float:
        """Calculate ion flux using Nernst-Planck equation.

        J = -D * (dC/dx) - z * F * D * C * (dφ/dx) / (R*T) + C * v

        Args:
            ion: Ion type
            concentration_anode: Anode side concentration (mol/m³)
            concentration_cathode: Cathode side concentration (mol/m³)
            potential_gradient: Electric potential gradient (V/m)

        Returns:
            Ion flux (mol/m²/s)

        """
        if ion not in self.ion_transport:
            return 0.0

        transport = self.ion_transport[ion]
        D = transport.diffusion_coefficient
        z = transport.charge
        F = self.params.faraday_constant
        R = self.params.gas_constant
        T = self.temperature

        # Effective diffusion in porous membrane
        D_eff = D * self.params.porosity / self.params.tortuosity

        # Average concentration for migration term
        C_avg = (concentration_anode + concentration_cathode) / 2

        # Concentration gradient
        dC_dx = (concentration_cathode - concentration_anode) / self.thickness

        # Diffusion flux
        flux_diffusion = -D_eff * dC_dx

        # Migration flux (electric field driven)
        flux_migration = -z * F * D_eff * C_avg * potential_gradient / (R * T)

        # Total flux
        return flux_diffusion + flux_migration

    def calculate_donnan_potential(
        self,
        ion_concentrations_anode: dict[IonType, float],
        ion_concentrations_cathode: dict[IonType, float],
    ) -> float:
        """Calculate Donnan potential at membrane interfaces.

        Args:
            ion_concentrations_anode: Ion concentrations at anode (mol/m³)
            ion_concentrations_cathode: Ion concentrations at cathode (mol/m³)

        Returns:
            Donnan potential (V)

        """
        R = self.params.gas_constant
        T = self.temperature
        F = self.params.faraday_constant

        # Calculate ionic strength on both sides
        ionic_strength_anode = 0.0
        ionic_strength_cathode = 0.0

        for ion, conc in ion_concentrations_anode.items():
            if ion in self.ion_transport:
                z = self.ion_transport[ion].charge
                ionic_strength_anode += 0.5 * conc * z**2

        for ion, conc in ion_concentrations_cathode.items():
            if ion in self.ion_transport:
                z = self.ion_transport[ion].charge
                ionic_strength_cathode += 0.5 * conc * z**2

        # Simplified Donnan potential calculation
        if ionic_strength_anode > 0 and ionic_strength_cathode > 0:
            donnan_potential = (R * T / F) * jnp.log(
                ionic_strength_cathode / ionic_strength_anode,
            )
        else:
            donnan_potential = 0.0

        return float(donnan_potential)

    def calculate_water_transport(
        self,
        current_density: float,
        primary_ion: IonType,
    ) -> float:
        """Calculate water transport through electro-osmotic drag.

        Args:
            current_density: Current density (A/m²)
            primary_ion: Primary charge carrier ion

        Returns:
            Water flux (mol/m²/s)

        """
        if primary_ion not in self.ion_transport:
            return 0.0

        # Ion flux from current
        z = abs(self.ion_transport[primary_ion].charge)
        ion_flux = current_density / (z * self.params.faraday_constant)

        # Water drag coefficient
        hydration_number = self.ion_transport[primary_ion].hydration_number

        # Account for membrane water content
        water_activity = min(
            1.0,
            self.params.water_content / 14.0,
        )  # Normalized to Nafion
        effective_hydration = hydration_number * water_activity

        # Water flux
        return effective_hydration * ion_flux

    def calculate_gas_permeability(
        self,
        gas_type: str,
        pressure_difference: float,
    ) -> float:
        """Calculate gas permeability through membrane.

        Args:
            gas_type: Type of gas ("O2", "H2", "CO2", etc.)
            pressure_difference: Pressure difference (Pa)

        Returns:
            Gas flux (mol/m²/s)

        """
        # Literature permeability values (mol·m/(m²·s·Pa))
        permeability_data = {
            "O2": 1.8e-15,  # Oxygen
            "H2": 3.5e-15,  # Hydrogen
            "CO2": 5.2e-15,  # Carbon dioxide
            "N2": 0.9e-15,  # Nitrogen
            "CH4": 2.1e-15,  # Methane
        }

        if gas_type not in permeability_data:
            return 0.0

        # Base permeability
        P_base = permeability_data[gas_type]

        # Temperature correction (Arrhenius)
        T_ref = 298.15
        activation_energy = 20000  # J/mol (typical)
        temp_factor = jnp.exp(
            -activation_energy
            / self.params.gas_constant
            * (1 / self.temperature - 1 / T_ref),
        )

        # Water content effect (hydrated membranes have higher permeability)
        water_factor = 1.0 + 0.1 * self.params.water_content

        # Effective permeability
        P_eff = P_base * temp_factor * water_factor

        # Gas flux (mol/m²/s)
        gas_flux = P_eff * pressure_difference / self.thickness

        return float(gas_flux)

    def calculate_membrane_resistance(
        self,
        ionic_conductivity: float | None = None,
    ) -> float:
        """Calculate membrane resistance including degradation effects.

        Args:
            ionic_conductivity: Override conductivity (S/m)

        Returns:
            Membrane resistance (Ω)

        """
        if ionic_conductivity is None:
            # Use degraded conductivity
            degradation_factor = jnp.exp(
                -self.params.degradation_rate * self.operating_hours,
            )
            ionic_conductivity = (
                self.params.initial_conductivity * degradation_factor * 100
            )  # S/m

        # Resistance = thickness / (conductivity * area)
        resistance = self.thickness / (ionic_conductivity * self.area)

        # Add fouling resistance
        resistance += self.params.fouling_resistance / self.area

        return float(resistance)

    def calculate_selectivity(
        self,
        ion1: IonType,
        ion2: IonType,
        concentration_ratio: float = 1.0,
    ) -> float:
        """Calculate membrane selectivity between two ions.

        Selectivity = (P1/P2) * (D1/D2) * (K1/K2)

        Args:
            ion1: First ion type
            ion2: Second ion type
            concentration_ratio: C1/C2 ratio

        Returns:
            Selectivity coefficient

        """
        if ion1 not in self.ion_transport or ion2 not in self.ion_transport:
            return 1.0

        transport1 = self.ion_transport[ion1]
        transport2 = self.ion_transport[ion2]

        # Diffusion selectivity
        D_ratio = transport1.diffusion_coefficient / transport2.diffusion_coefficient

        # Partition selectivity
        K_ratio = transport1.partition_coefficient / transport2.partition_coefficient

        # Charge selectivity (Donnan exclusion)
        z1 = transport1.charge
        z2 = transport2.charge

        if z1 * z2 < 0:  # Opposite charges
            charge_selectivity = 10.0  # High selectivity
        elif z1 == z2:  # Same charge
            charge_selectivity = 1.0
        else:  # Different magnitudes
            charge_selectivity = abs(z1 / z2)

        # Total selectivity
        selectivity = D_ratio * K_ratio * charge_selectivity

        return float(selectivity)

    def calculate_transport_number(
        self,
        ion: IonType,
        all_ion_concentrations: dict[IonType, float],
        current_density: float,
    ) -> float:
        """Calculate transport number for specific ion.

        t_i = |z_i| * u_i * C_i / Σ(|z_j| * u_j * C_j)

        Args:
            ion: Ion of interest
            all_ion_concentrations: All ion concentrations (mol/m³)
            current_density: Current density (A/m²)

        Returns:
            Transport number (0-1)

        """
        if ion not in self.ion_transport or ion not in all_ion_concentrations:
            return 0.0

        # Calculate conductivity contribution of each ion
        total_conductivity = 0.0
        ion_conductivity = 0.0

        F = self.params.faraday_constant

        for ion_type, concentration in all_ion_concentrations.items():
            if ion_type in self.ion_transport:
                transport = self.ion_transport[ion_type]
                z = abs(transport.charge)
                u = transport.mobility

                conductivity_contribution = z * F * u * concentration
                total_conductivity += conductivity_contribution

                if ion_type == ion:
                    ion_conductivity = conductivity_contribution

        if total_conductivity > 0:
            transport_number = ion_conductivity / total_conductivity
        else:
            transport_number = 0.0

        return float(transport_number)

    @abstractmethod
    def calculate_ionic_conductivity(
        self,
        temperature: float | None = None,
        water_content: float | None = None,
    ) -> float:
        """Calculate ionic conductivity specific to membrane type.

        Args:
            temperature: Temperature (K)
            water_content: Water content (mol H2O/mol functional group)

        Returns:
            Ionic conductivity (S/m)

        """

    def update_operating_conditions(
        self,
        temperature: float | None = None,
        pressure_anode: float | None = None,
        pressure_cathode: float | None = None,
    ) -> None:
        """Update membrane operating conditions."""
        if temperature is not None:
            self.temperature = temperature
            self.params.temperature = temperature

        if pressure_anode is not None:
            self.params.pressure_anode = pressure_anode

        if pressure_cathode is not None:
            self.params.pressure_cathode = pressure_cathode

        # Recalculate properties
        self._calculate_membrane_properties()

    def update_degradation(self, operating_hours: float) -> None:
        """Update membrane degradation state."""
        self.operating_hours = operating_hours

    def add_fouling_resistance(self, additional_resistance: float) -> None:
        """Add fouling resistance to membrane."""
        self.params.fouling_resistance += additional_resistance

    def get_performance_metrics(
        self,
        current_density: float,
        ion_concentrations: dict[IonType, float],
    ) -> dict[str, float]:
        """Calculate comprehensive membrane performance metrics.

        Args:
            current_density: Operating current density (A/m²)
            ion_concentrations: Ion concentrations (mol/m³)

        Returns:
            Dictionary of performance metrics

        """
        # Calculate conductivity
        conductivity = self.calculate_ionic_conductivity()

        # Calculate resistance
        resistance = self.calculate_membrane_resistance(conductivity)

        # Voltage drop
        voltage_drop = current_density * self.area * resistance

        # Power loss
        power_loss = current_density * voltage_drop * self.area

        # Transport numbers for major ions
        transport_numbers = {}
        for ion in ion_concentrations:
            if ion in self.ion_transport:
                t_ion = self.calculate_transport_number(
                    ion,
                    ion_concentrations,
                    current_density,
                )
                transport_numbers[f"transport_number_{ion.value}"] = t_ion

        return {
            "ionic_conductivity_S_m": float(conductivity),
            "membrane_resistance_ohm": float(resistance),
            "voltage_drop_V": float(voltage_drop),
            "power_loss_W": float(power_loss),
            "thickness_um": self.thickness * 1e6,
            "area_cm2": self.area * 1e4,
            "operating_hours": self.operating_hours,
            "fouling_resistance_ohm_m2": self.params.fouling_resistance,
            **transport_numbers,
        }

    def get_transport_properties(self) -> dict[str, Any]:
        """Get all transport properties for inspection."""
        properties = {
            "membrane_type": self.__class__.__name__,
            "thickness_m": self.thickness,
            "area_m2": self.area,
            "porosity": self.params.porosity,
            "tortuosity": self.params.tortuosity,
            "ion_exchange_capacity_mol_kg": self.params.ion_exchange_capacity,
            "water_content": self.params.water_content,
            "temperature_K": self.temperature,
            "ion_transport_mechanisms": {},
        }

        # Add ion transport details
        for ion, transport in self.ion_transport.items():
            properties["ion_transport_mechanisms"][ion.value] = {
                "diffusion_coefficient_m2_s": transport.diffusion_coefficient,
                "mobility_m2_V_s": transport.mobility,
                "partition_coefficient": transport.partition_coefficient,
                "hydration_number": transport.hydration_number,
                "charge": transport.charge,
            }

        return properties

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"thickness={self.thickness * 1e6:.1f} μm, "
            f"area={self.area * 1e4:.1f} cm², "
            f"T={self.temperature:.1f} K)"
        )


class IonTransportDatabase:
    """Database of ion transport properties in various membrane types."""

    @staticmethod
    def get_proton_transport() -> IonTransportMechanisms:
        """Get proton transport properties."""
        return IonTransportMechanisms(
            ion_type=IonType.PROTON,
            diffusion_coefficient=9.3e-9,  # m²/s in water
            mobility=3.6e-7,  # m²/(V·s)
            partition_coefficient=1.0,  # For PEM
            hydration_number=3.0,  # H3O+
            charge=1,
            stokes_radius=2.8e-10,  # m
        )

    @staticmethod
    def get_hydroxide_transport() -> IonTransportMechanisms:
        """Get hydroxide transport properties."""
        return IonTransportMechanisms(
            ion_type=IonType.HYDROXIDE,
            diffusion_coefficient=5.3e-9,  # m²/s in water
            mobility=2.0e-7,  # m²/(V·s)
            partition_coefficient=0.8,  # For AEM
            hydration_number=3.0,
            charge=-1,
            stokes_radius=3.0e-10,  # m
        )

    @staticmethod
    def get_sodium_transport() -> IonTransportMechanisms:
        """Get sodium transport properties."""
        return IonTransportMechanisms(
            ion_type=IonType.SODIUM,
            diffusion_coefficient=1.3e-9,  # m²/s in water
            mobility=5.2e-8,  # m²/(V·s)
            partition_coefficient=0.3,  # Lower for ion exchange membranes
            hydration_number=4.0,
            charge=1,
            stokes_radius=3.6e-10,  # m
        )
