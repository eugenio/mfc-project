#!/usr/bin/env python3
"""
Example configuration for Geobacter sulfurreducens with acetate substrate.

This example demonstrates how to create and customize biological configurations
for MFC modeling with specific species and substrate combinations.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.biological_config import (
    BacterialSpecies, MetabolicReactionConfig,
    KineticParameters, LiteratureReference, get_geobacter_config
)
from config.substrate_config import (
    get_acetate_config
)
from config.biological_validation import (
    validate_species_metabolic_config, validate_comprehensive_substrate_config
)

def create_custom_geobacter_acetate_config():
    """
    Create a customized configuration for Geobacter sulfurreducens growing on acetate.
    This shows how to modify default configurations for specific experimental conditions.
    """

    # Start with default Geobacter configuration
    geobacter_config = get_geobacter_config()

    # Customize for specific experimental conditions
    # Example: Lab strain with enhanced electron transport
    geobacter_config.electron_transport_efficiency = 0.92  # Higher efficiency
    geobacter_config.cytochrome_content = 0.18  # Enhanced cytochrome expression
    geobacter_config.max_growth_rate = 0.28  # Optimized strain

    # Add additional metabolic reactions
    citrate_synthase_kinetics = KineticParameters(
        vmax=12.0,  # mmol/gDW/h
        km=0.3,     # mmol/L for acetyl-CoA
        ea=48.0,    # kJ/mol
        ph_optimal=7.0,
        reference=LiteratureReference(
            authors="Lovley, D.R., Phillips, E.J.P.",
            title="Novel mode of microbial energy metabolism: organic carbon oxidation coupled to dissimilatory reduction of iron or manganese",
            journal="Applied and Environmental Microbiology",
            year=1988,
            doi="10.1128/aem.54.6.1472-1480.1988"
        )
    )

    citrate_synthase = MetabolicReactionConfig(
        id="GSU_R002",
        name="Citrate synthase",
        equation="Acetyl-CoA + Oxaloacetate + H2O → Citrate + CoA",
        stoichiometry={
            "acetyl_coa": -1.0, "oxaloacetate": -1.0, "h2o": -1.0,
            "citrate": 1.0, "coa": 1.0
        },
        enzyme_name="Citrate synthase",
        ec_number="EC 2.3.3.1",
        kegg_id="R00351",
        kinetics=citrate_synthase_kinetics,
        delta_g0=-31.5,  # kJ/mol
        reversible=False,
        flux_lower_bound=0.0,
        flux_upper_bound=15.0
    )

    geobacter_config.reactions.append(citrate_synthase)

    # Get acetate configuration
    acetate_config = get_acetate_config()

    # Customize acetate kinetics for this specific strain
    geobacter_acetate_kinetics = acetate_config.species_kinetics[BacterialSpecies.GEOBACTER_SULFURREDUCENS]
    geobacter_acetate_kinetics.max_uptake_rate = 25.0  # Enhanced uptake
    geobacter_acetate_kinetics.half_saturation_constant = 0.4  # Improved affinity

    return geobacter_config, acetate_config

def validate_and_demonstrate_config():
    """Validate the configuration and demonstrate its usage."""

    print("Creating custom Geobacter-acetate configuration...")
    geobacter_config, acetate_config = create_custom_geobacter_acetate_config()

    # Validate configurations
    print("\nValidating species configuration...")
    try:
        validate_species_metabolic_config(geobacter_config)
        print("✓ Species configuration is valid")
    except Exception as e:
        print(f"✗ Species configuration validation failed: {e}")
        return False

    print("\nValidating substrate configuration...")
    try:
        validate_comprehensive_substrate_config(acetate_config)
        print("✓ Substrate configuration is valid")
    except Exception as e:
        print(f"✗ Substrate configuration validation failed: {e}")
        return False

    # Demonstrate configuration usage
    print("\n=== Configuration Summary ===")
    print(f"Species: {geobacter_config.species.value}")
    print(f"Substrate: {acetate_config.substrate_type.value}")
    print(f"Max growth rate: {geobacter_config.max_growth_rate:.3f} h⁻¹")
    print(f"Electron transport efficiency: {geobacter_config.electron_transport_efficiency:.1%}")
    print(f"Cytochrome content: {geobacter_config.cytochrome_content:.3f} mmol/gDW")
    print(f"Max biofilm thickness: {geobacter_config.max_biofilm_thickness:.1f} μm")

    print("\n=== Acetate Properties ===")
    print(f"Molecular weight: {acetate_config.molecular_weight:.2f} g/mol")
    print(f"Chemical formula: {acetate_config.chemical_formula}")
    print(f"Water solubility: {acetate_config.water_solubility:.1f} g/L")

    geobacter_kinetics = acetate_config.species_kinetics[BacterialSpecies.GEOBACTER_SULFURREDUCENS]
    print(f"Max uptake rate: {geobacter_kinetics.max_uptake_rate:.1f} mmol/gDW/h")
    print(f"Half-saturation constant: {geobacter_kinetics.half_saturation_constant:.2f} mmol/L")

    print("\n=== Metabolic Reactions ===")
    for reaction in geobacter_config.reactions:
        print(f"- {reaction.name} (ID: {reaction.id})")
        print(f"  Vmax: {reaction.kinetics.vmax:.1f} mmol/gDW/h")
        print(f"  Km: {reaction.kinetics.km:.2f} mmol/L")
        print(f"  ΔG°: {reaction.delta_g0:.1f} kJ/mol")

    print("\n=== Literature References ===")
    for ref in geobacter_config.references:
        print(f"- {ref}")

    return True

def demonstrate_model_integration():
    """Show how to use the configuration with the metabolic model."""

    print("\n=== Model Integration Example ===")

    # This would be used with the actual models like this:
    print("""
# Example integration with metabolic model:
from metabolic_model.metabolic_core import MetabolicModel

geobacter_config, acetate_config = create_custom_geobacter_acetate_config()

# Initialize model with custom configuration
model = MetabolicModel(
    species="geobacter",
    substrate="acetate", 
    species_config=geobacter_config,
    substrate_config=acetate_config
)

# The model will now use the custom parameters from configuration
model.reset_state()  # Uses metabolite concentrations from config
fluxes = model.calculate_metabolic_fluxes(
    biomass=1.0, growth_rate=0.2, 
    anode_potential=-0.2, substrate_supply=5.0
)
    """)

if __name__ == "__main__":
    print("Geobacter sulfurreducens + Acetate Configuration Example")
    print("=" * 60)

    success = validate_and_demonstrate_config()

    if success:
        demonstrate_model_integration()
        print("\n✓ Configuration example completed successfully!")
    else:
        print("\n✗ Configuration validation failed!")
        sys.exit(1)
