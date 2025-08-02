#!/usr/bin/env python3
"""
Example configuration for mixed culture systems.

This example demonstrates:
- Multi-species configuration management
- Substrate competition modeling
- Synergistic interactions
- Dynamic species ratios
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


from config.biological_config import (
    BacterialSpecies,
    get_default_biofilm_config,
    get_geobacter_config,
    get_shewanella_config,
)
from config.substrate_config import (
    SubstrateKineticsConfig,
    SubstrateType,
    get_acetate_config,
    get_lactate_config,
    get_pyruvate_config,
)


class MixedCultureConfig:
    """Configuration manager for mixed culture systems."""

    def __init__(self, species_ratios: dict[BacterialSpecies, float]):
        """
        Initialize mixed culture configuration.

        Args:
            species_ratios: Dictionary of species to their fractional abundance
        """
        self.species_ratios = species_ratios
        self.species_configs = {}
        self.substrate_configs = {}
        self.biofilm_config = None

        # Validate ratios sum to 1.0
        total_ratio = sum(species_ratios.values())
        if abs(total_ratio - 1.0) > 0.01:
            raise ValueError(f"Species ratios must sum to 1.0, got {total_ratio}")

        self._load_species_configurations()
        self._load_substrate_configurations()
        self._load_biofilm_configuration()

    def _load_species_configurations(self):
        """Load individual species configurations."""
        for species in self.species_ratios.keys():
            if species == BacterialSpecies.GEOBACTER_SULFURREDUCENS:
                self.species_configs[species] = get_geobacter_config()
            elif species == BacterialSpecies.SHEWANELLA_ONEIDENSIS:
                self.species_configs[species] = get_shewanella_config()
            else:
                raise ValueError(f"Unsupported species: {species}")

    def _load_substrate_configurations(self):
        """Load substrate configurations for all relevant substrates."""
        self.substrate_configs = {
            SubstrateType.ACETATE: get_acetate_config(),
            SubstrateType.LACTATE: get_lactate_config(),
            SubstrateType.PYRUVATE: get_pyruvate_config()
        }

    def _load_biofilm_configuration(self):
        """Load and customize biofilm configuration for mixed culture."""
        self.biofilm_config = get_default_biofilm_config()

        # Adjust biofilm properties based on species composition
        geobacter_fraction = self.species_ratios.get(BacterialSpecies.GEOBACTER_SULFURREDUCENS, 0.0)
        shewanella_fraction = self.species_ratios.get(BacterialSpecies.SHEWANELLA_ONEIDENSIS, 0.0)

        # Mixed culture biofilm is typically more porous and conductive
        self.biofilm_config.porosity = 0.75 + 0.1 * shewanella_fraction  # Shewanella creates more porous biofilms
        self.biofilm_config.nernst_monod['biofilm_conductivity'] = (
            0.005 * geobacter_fraction + 0.003 * shewanella_fraction  # Geobacter more conductive
        )

        # Adjust growth parameters as weighted average
        geobacter_growth = 0.25
        shewanella_growth = 0.35
        mixed_growth = geobacter_growth * geobacter_fraction + shewanella_growth * shewanella_fraction
        self.biofilm_config.monod_kinetics['max_growth_rate'] = mixed_growth

    def get_effective_substrate_kinetics(self, substrate: SubstrateType) -> SubstrateKineticsConfig:
        """
        Calculate effective substrate kinetics for mixed culture.

        Args:
            substrate: Substrate type

        Returns:
            Effective kinetics combining all species
        """
        substrate_config = self.substrate_configs[substrate]

        # Calculate weighted average of kinetic parameters
        total_uptake = 0.0
        weighted_km = 0.0
        weighted_ki = 0.0

        for species, fraction in self.species_ratios.items():
            if species in substrate_config.species_kinetics:
                kinetics = substrate_config.species_kinetics[species]
                uptake_contribution = kinetics.max_uptake_rate * fraction
                total_uptake += uptake_contribution

                # Weight Km and Ki by uptake rate contribution
                if uptake_contribution > 0:
                    weight = uptake_contribution / (kinetics.max_uptake_rate * fraction)
                    weighted_km += kinetics.half_saturation_constant * weight * fraction
                    if kinetics.substrate_inhibition_constant:
                        weighted_ki += kinetics.substrate_inhibition_constant * weight * fraction

        # Create effective kinetics configuration
        effective_kinetics = SubstrateKineticsConfig(
            max_uptake_rate=total_uptake,
            half_saturation_constant=weighted_km if weighted_km > 0 else 0.5,
            substrate_inhibition_constant=weighted_ki if weighted_ki > 0 else None,
            temperature_coefficient=1.08,  # Average
            ph_optimum=7.1,  # Compromise pH
            ph_tolerance_range=(6.5, 8.0),
            activation_energy=45.0,  # Average
            enthalpy_change=-125.0,
            entropy_change=-0.15
        )

        return effective_kinetics

    def calculate_substrate_competition(self, substrate_concentrations: dict[SubstrateType, float]) -> dict[SubstrateType, dict[BacterialSpecies, float]]:
        """
        Calculate substrate uptake rates considering competition.

        Args:
            substrate_concentrations: Current substrate concentrations (mmol/L)

        Returns:
            Dictionary of substrate -> species -> uptake rate
        """
        uptake_rates = {}

        for substrate, concentration in substrate_concentrations.items():
            uptake_rates[substrate] = {}

            for species, fraction in self.species_ratios.items():
                if species in self.substrate_configs[substrate].species_kinetics:
                    kinetics = self.substrate_configs[substrate].species_kinetics[species]

                    # Base Monod kinetics
                    base_rate = (kinetics.max_uptake_rate * concentration /
                               (kinetics.half_saturation_constant + concentration))

                    # Apply competition effects
                    competition_factor = 1.0
                    for other_substrate, other_conc in substrate_concentrations.items():
                        if other_substrate != substrate and other_conc > 0:
                            # Simple competitive inhibition
                            if other_substrate.value in kinetics.competitive_inhibition:
                                ki = kinetics.competitive_inhibition[other_substrate.value]
                                competition_factor *= ki / (ki + other_conc)

                    # Final uptake rate
                    uptake_rates[substrate][species] = base_rate * competition_factor * fraction
                else:
                    uptake_rates[substrate][species] = 0.0

        return uptake_rates

    def calculate_synergy_factor(self, current_densities: dict[BacterialSpecies, float]) -> float:
        """
        Calculate synergy factor for mixed culture current production.

        Args:
            current_densities: Current production by each species (A/m²)

        Returns:
            Synergy enhancement factor
        """
        geobacter_current = current_densities.get(BacterialSpecies.GEOBACTER_SULFURREDUCENS, 0.0)
        shewanella_current = current_densities.get(BacterialSpecies.SHEWANELLA_ONEIDENSIS, 0.0)

        if geobacter_current > 0 and shewanella_current > 0:
            # Synergy occurs when both species are active
            # Shewanella flavins can enhance Geobacter electron transfer
            synergy_factor = 1.0 + 0.3 * min(geobacter_current, shewanella_current) / max(geobacter_current, shewanella_current)
        else:
            synergy_factor = 1.0

        return synergy_factor

    def get_configuration_summary(self) -> dict[str, any]:
        """Get summary of mixed culture configuration."""
        return {
            'species_composition': {species.value: ratio for species, ratio in self.species_ratios.items()},
            'dominant_species': max(self.species_ratios, key=self.species_ratios.get).value,
            'biofilm_porosity': self.biofilm_config.porosity,
            'biofilm_conductivity': self.biofilm_config.nernst_monod['biofilm_conductivity'],
            'mixed_growth_rate': self.biofilm_config.monod_kinetics['max_growth_rate'],
            'supported_substrates': [substrate.value for substrate in self.substrate_configs.keys()]
        }

def create_geobacter_dominant_culture():
    """Create a Geobacter-dominant mixed culture configuration."""
    species_ratios = {
        BacterialSpecies.GEOBACTER_SULFURREDUCENS: 0.7,
        BacterialSpecies.SHEWANELLA_ONEIDENSIS: 0.3
    }

    return MixedCultureConfig(species_ratios)

def create_balanced_mixed_culture():
    """Create a balanced mixed culture configuration."""
    species_ratios = {
        BacterialSpecies.GEOBACTER_SULFURREDUCENS: 0.5,
        BacterialSpecies.SHEWANELLA_ONEIDENSIS: 0.5
    }

    return MixedCultureConfig(species_ratios)

def demonstrate_substrate_competition():
    """Demonstrate substrate competition in mixed cultures."""
    print("\n=== Substrate Competition Analysis ===")

    mixed_config = create_balanced_mixed_culture()

    # Simulate different substrate conditions
    scenarios = [
        {"name": "Acetate-rich", "acetate": 10.0, "lactate": 1.0, "pyruvate": 0.5},
        {"name": "Lactate-rich", "acetate": 1.0, "lactate": 10.0, "pyruvate": 0.5},
        {"name": "Balanced", "acetate": 5.0, "lactate": 5.0, "pyruvate": 2.0},
        {"name": "Substrate-limited", "acetate": 0.5, "lactate": 0.3, "pyruvate": 0.1}
    ]

    print(f"{'Scenario':<16} {'Species':<12} {'Acetate':<8} {'Lactate':<8} {'Pyruvate':<8}")
    print("-" * 60)

    for scenario in scenarios:
        substrate_concs = {
            SubstrateType.ACETATE: scenario["acetate"],
            SubstrateType.LACTATE: scenario["lactate"],
            SubstrateType.PYRUVATE: scenario["pyruvate"]
        }

        uptake_rates = mixed_config.calculate_substrate_competition(substrate_concs)

        for species in [BacterialSpecies.GEOBACTER_SULFURREDUCENS, BacterialSpecies.SHEWANELLA_ONEIDENSIS]:
            species_name = species.value.split('_')[0].capitalize()
            acetate_rate = uptake_rates[SubstrateType.ACETATE][species]
            lactate_rate = uptake_rates[SubstrateType.LACTATE][species]
            pyruvate_rate = uptake_rates[SubstrateType.PYRUVATE][species]

            print(f"{scenario['name'] if species == BacterialSpecies.GEOBACTER_SULFURREDUCENS else '':<16} "
                  f"{species_name:<12} {acetate_rate:<8.2f} {lactate_rate:<8.2f} {pyruvate_rate:<8.2f}")

        print()

def analyze_culture_compositions():
    """Analyze different mixed culture compositions."""
    print("\n=== Mixed Culture Composition Analysis ===")

    compositions = [
        {"name": "Geobacter-dominant", "geobacter": 0.8, "shewanella": 0.2},
        {"name": "Balanced", "geobacter": 0.5, "shewanella": 0.5},
        {"name": "Shewanella-dominant", "geobacter": 0.3, "shewanella": 0.7}
    ]

    print(f"{'Composition':<18} {'G.sulf':<8} {'S.onei':<8} {'Porosity':<10} {'Conductivity':<12} {'Growth Rate':<12}")
    print("-" * 80)

    for comp in compositions:
        species_ratios = {
            BacterialSpecies.GEOBACTER_SULFURREDUCENS: comp["geobacter"],
            BacterialSpecies.SHEWANELLA_ONEIDENSIS: comp["shewanella"]
        }

        mixed_config = MixedCultureConfig(species_ratios)
        summary = mixed_config.get_configuration_summary()

        print(f"{comp['name']:<18} {comp['geobacter']:<8.1f} {comp['shewanella']:<8.1f} "
              f"{summary['biofilm_porosity']:<10.3f} {summary['biofilm_conductivity']:<12.5f} "
              f"{summary['mixed_growth_rate']:<12.3f}")

def validate_mixed_configurations():
    """Validate mixed culture configurations."""
    print("\n=== Mixed Culture Validation ===")

    try:
        # Test different compositions
        compositions = [
            create_geobacter_dominant_culture(),
            create_balanced_mixed_culture()
        ]

        for i, config in enumerate(compositions):
            print(f"Validating composition {i+1}...")

            # Validate individual species configs
            for _species, species_config in config.species_configs.items():
                validate_species_metabolic_config(species_config)

            print(f"✓ Composition {i+1} validation passed")

        return True

    except Exception as e:
        print(f"✗ Mixed culture validation failed: {e}")
        return False

if __name__ == "__main__":
    print("Mixed Culture Configuration Example")
    print("=" * 50)

    # Validate configurations
    if not validate_mixed_configurations():
        print("\n✗ Configuration validation failed!")
        sys.exit(1)

    # Analyze different compositions
    analyze_culture_compositions()

    # Demonstrate substrate competition
    demonstrate_substrate_competition()

    print("\n✓ Mixed culture configuration example completed successfully!")
    print("\nKey features demonstrated:")
    print("  - Species ratio management")
    print("  - Substrate competition modeling")
    print("  - Biofilm property blending")
    print("  - Synergistic interactions")
    print("  - Dynamic composition effects")
