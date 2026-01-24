#!/usr/bin/env python3
"""Example configuration for Geobacter sulfurreducens with acetate substrate.

This example demonstrates how to create and customize biological configurations
for MFC modeling with specific species and substrate combinations.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from config.biological_config import (
    BacterialSpecies,
    KineticParameters,
    LiteratureReference,
    MetabolicReactionConfig,
    get_geobacter_config,
)
from config.biological_validation import (
    validate_comprehensive_substrate_config,
    validate_species_metabolic_config,
)
from config.substrate_config import get_acetate_config


def create_custom_geobacter_acetate_config():
    """Create a customized configuration for Geobacter sulfurreducens growing on acetate.
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
        km=0.3,  # mmol/L for acetyl-CoA
        ea=48.0,  # kJ/mol
        ph_optimal=7.0,
        reference=LiteratureReference(
            authors="Lovley, D.R., Phillips, E.J.P.",
            title="Novel mode of microbial energy metabolism: organic carbon oxidation coupled to dissimilatory reduction of iron or manganese",
            journal="Applied and Environmental Microbiology",
            year=1988,
            doi="10.1128/aem.54.6.1472-1480.1988",
        ),
    )

    citrate_synthase = MetabolicReactionConfig(
        id="GSU_R002",
        name="Citrate synthase",
        equation="Acetyl-CoA + Oxaloacetate + H2O → Citrate + CoA",
        stoichiometry={
            "acetyl_coa": -1.0,
            "oxaloacetate": -1.0,
            "h2o": -1.0,
            "citrate": 1.0,
            "coa": 1.0,
        },
        enzyme_name="Citrate synthase",
        ec_number="EC 2.3.3.1",
        kegg_id="R00351",
        kinetics=citrate_synthase_kinetics,
        delta_g0=-31.5,  # kJ/mol
        reversible=False,
        flux_lower_bound=0.0,
        flux_upper_bound=15.0,
    )

    geobacter_config.reactions.append(citrate_synthase)

    # Get acetate configuration
    acetate_config = get_acetate_config()

    # Customize acetate kinetics for this specific strain
    geobacter_acetate_kinetics = acetate_config.species_kinetics[
        BacterialSpecies.GEOBACTER_SULFURREDUCENS
    ]
    geobacter_acetate_kinetics.max_uptake_rate = 25.0  # Enhanced uptake
    geobacter_acetate_kinetics.half_saturation_constant = 0.4  # Improved affinity

    return geobacter_config, acetate_config


def validate_and_demonstrate_config() -> bool:
    """Validate the configuration and demonstrate its usage."""
    geobacter_config, acetate_config = create_custom_geobacter_acetate_config()

    # Validate configurations
    try:
        validate_species_metabolic_config(geobacter_config)
    except Exception:
        return False

    try:
        validate_comprehensive_substrate_config(acetate_config)
    except Exception:
        return False

    # Demonstrate configuration usage

    acetate_config.species_kinetics[BacterialSpecies.GEOBACTER_SULFURREDUCENS]

    for _reaction in geobacter_config.reactions:
        pass

    for _ref in geobacter_config.references:
        pass

    return True


def demonstrate_model_integration() -> None:
    """Show how to use the configuration with the metabolic model."""
    # This would be used with the actual models like this:


if __name__ == "__main__":
    success = validate_and_demonstrate_config()

    if success:
        demonstrate_model_integration()
    else:
        sys.exit(1)
