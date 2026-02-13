#!/usr/bin/env python3
"""
Example configuration for Shewanella oneidensis with lactate substrate.

This example demonstrates advanced configuration features including:
- Custom degradation pathways
- Environmental condition optimization
- Transport mechanism configuration
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.biological_config import (
    BacterialSpecies,
    KineticParameters,
    LiteratureReference,
    MetabolicReactionConfig,
    get_shewanella_config,
)
from config.biological_validation import (
    validate_comprehensive_substrate_config,
    validate_species_metabolic_config,
)
from config.substrate_config import (
    SubstrateDegradationPathway,
    SubstrateKineticsConfig,
    SubstrateTransportConfig,
    SubstrateType,
    get_lactate_config,
)


def create_enhanced_shewanella_lactate_config():
    """
    Create an enhanced configuration for Shewanella oneidensis with optimized lactate utilization.
    This example shows advanced customization for specific research applications.
    """

    # Start with default Shewanella configuration
    shewanella_config = get_shewanella_config()

    # Enhance for marine/high-salt conditions
    shewanella_config.salinity_tolerance = 1.2  # Higher salt tolerance
    shewanella_config.temperature_range = (275.0, 318.0)  # Extended temperature range
    shewanella_config.electron_transport_efficiency = 0.78  # Account for flavin mediators

    # Add pyruvate dehydrogenase complex reaction
    pdh_kinetics = KineticParameters(
        vmax=18.0,  # mmol/gDW/h
        km=0.25,    # mmol/L for pyruvate
        ea=40.0,    # kJ/mol - lower activation energy
        ph_optimal=7.2,
        ph_tolerance=1.2,
        reference=LiteratureReference(
            authors="Pinchuk, G.E., Hill, E.A., Geydebrekht, O.V., et al.",
            title="Constraint-based model of Shewanella oneidensis MR-1 metabolism: a tool for data analysis and hypothesis generation",
            journal="PLoS Computational Biology",
            year=2010,
            doi="10.1371/journal.pcbi.1000822"
        )
    )

    pyruvate_dehydrogenase = MetabolicReactionConfig(
        id="MR1_R002",
        name="Pyruvate dehydrogenase complex",
        equation="Pyruvate + NAD+ + CoA → Acetyl-CoA + NADH + CO2",
        stoichiometry={
            "pyruvate": -1.0, "nad_plus": -1.0, "coa": -1.0,
            "acetyl_coa": 1.0, "nadh": 1.0, "co2": 1.0
        },
        enzyme_name="Pyruvate dehydrogenase complex",
        ec_number="EC 1.2.4.1",
        kegg_id="R00209",
        kinetics=pdh_kinetics,
        delta_g0=-33.4,  # kJ/mol
        reversible=False,
        flux_lower_bound=0.0,
        flux_upper_bound=25.0
    )

    shewanella_config.reactions.append(pyruvate_dehydrogenase)

    # Get lactate configuration and enhance it
    lactate_config = get_lactate_config()

    # Create enhanced lactate degradation pathway with intermediate tracking
    enhanced_lactate_pathway = SubstrateDegradationPathway(
        pathway_name="Enhanced lactate oxidation with flavin mediation",
        substrate=SubstrateType.LACTATE,
        intermediates=["pyruvate", "acetyl_coa", "flavin_reduced", "flavin_oxidized"],
        final_products=["co2", "h2o", "acetate"],
        substrate_stoichiometry=1.0,
        electron_yield=12.0,  # Full oxidation yield
        biomass_yield=0.15,   # Enhanced growth yield
        atp_yield=2.8,        # Higher ATP yield with electron transport
        nadh_yield=5.2,       # Increased NADH production
        co2_yield=3.0,
        regulatory_metabolites={
            "oxygen": "inhibition",
            "acetate": "product_inhibition",
            "flavin_oxidized": "activation"
        },
        allosteric_effectors={
            "amp": 1.5,    # AMP activation
            "atp": 0.7,    # ATP inhibition
            "nadh": 0.8    # NADH inhibition
        },
        optimal_conditions={
            'temperature': 308.0,  # K - marine optimal
            'ph': 7.3,            # Slightly alkaline
            'ionic_strength': 0.2, # M - marine salinity
            'redox_potential': -0.15  # V vs SHE - optimized for flavins
        }
    )

    # Replace the default pathway
    lactate_config.degradation_pathways = [enhanced_lactate_pathway]

    # Enhanced kinetics for marine Shewanella strain
    enhanced_shewanella_kinetics = SubstrateKineticsConfig(
        max_uptake_rate=35.0,  # mmol/gDW/h - higher for enhanced strain
        half_saturation_constant=0.25,  # mmol/L - improved affinity
        substrate_inhibition_constant=90.0,  # mmol/L - higher tolerance
        temperature_coefficient=1.12,  # Enhanced temperature response
        ph_optimum=7.3,
        ph_tolerance_range=(6.8, 8.2),  # Broader pH range
        activation_energy=38.0,  # kJ/mol - lower energy barrier
        enthalpy_change=-140.0,  # kJ/mol
        entropy_change=-0.22,    # kJ/mol/K
        reference=LiteratureReference(
            authors="Marsili, E., Baron, D.B., Shikhare, I.D., et al.",
            title="Shewanella secretes flavins that mediate extracellular electron transfer",
            journal="Proceedings of the National Academy of Sciences",
            year=2008,
            doi="10.1073/pnas.0710525105"
        )
    )

    # Update species kinetics
    lactate_config.species_kinetics[BacterialSpecies.SHEWANELLA_ONEIDENSIS] = enhanced_shewanella_kinetics

    # Enhanced transport configuration for active lactate uptake
    enhanced_transport = SubstrateTransportConfig(
        transport_type="secondary_active_transport",  # Na+/lactate symporter
        max_transport_rate=60.0,  # mmol/gDW/h - high capacity
        transport_km=0.12,        # mmol/L - high affinity
        atp_cost=0.3,            # Indirect ATP cost via Na+/K+ ATPase
        pmf_cost=1.5,            # Na+ coupling
        transport_regulation={
            "sodium": 1.2,       # Na+ activation
            "potassium": 0.9,    # K+ slight inhibition
            "camp": 1.3          # cAMP activation
        },
        competitive_substrates=["pyruvate", "malate", "succinate"],
        competition_factors={"pyruvate": 0.85, "malate": 0.75, "succinate": 0.92}
    )

    lactate_config.transport_config = enhanced_transport

    return shewanella_config, lactate_config

def demonstrate_environmental_adaptation():
    """Show how configuration adapts to different environmental conditions."""

    print("\n=== Environmental Adaptation Example ===")

    shewanella_config, lactate_config = create_enhanced_shewanella_lactate_config()

    # Simulate different environmental conditions
    conditions = [
        {"name": "Freshwater", "temp": 298.0, "ph": 7.0, "salinity": 0.01},
        {"name": "Brackish", "temp": 303.0, "ph": 7.5, "salinity": 0.5},
        {"name": "Marine", "temp": 288.0, "ph": 8.1, "salinity": 0.6},
        {"name": "Extreme", "temp": 313.0, "ph": 6.8, "salinity": 1.0}
    ]

    print(f"{'Environment':<12} {'Temp (°C)':<10} {'pH':<5} {'Salinity (M)':<12} {'Feasible':<10}")
    print("-" * 60)

    for condition in conditions:
        temp_c = condition["temp"] - 273.15
        temp_ok = shewanella_config.temperature_range[0] <= condition["temp"] <= shewanella_config.temperature_range[1]
        ph_ok = shewanella_config.ph_range[0] <= condition["ph"] <= shewanella_config.ph_range[1]
        salt_ok = condition["salinity"] <= shewanella_config.salinity_tolerance

        feasible = "✓" if (temp_ok and ph_ok and salt_ok) else "✗"

        print(f"{condition['name']:<12} {temp_c:<10.1f} {condition['ph']:<5.1f} {condition['salinity']:<12.2f} {feasible:<10}")

    return shewanella_config, lactate_config

def analyze_metabolic_pathway():
    """Analyze the enhanced metabolic pathway configuration."""

    print("\n=== Enhanced Lactate Pathway Analysis ===")

    _, lactate_config = create_enhanced_shewanella_lactate_config()

    pathway = lactate_config.degradation_pathways[0]

    print(f"Pathway: {pathway.pathway_name}")
    print(f"Substrate stoichiometry: {pathway.substrate_stoichiometry:.1f}")
    print(f"Electron yield: {pathway.electron_yield:.1f} e⁻/lactate")
    print(f"Biomass yield: {pathway.biomass_yield:.3f} gDW/mmol")
    print(f"ATP yield: {pathway.atp_yield:.1f} mol/mol")
    print(f"NADH yield: {pathway.nadh_yield:.1f} mol/mol")

    print("\nMetabolic intermediates:")
    for intermediate in pathway.intermediates:
        print(f"  - {intermediate}")

    print("\nFinal products:")
    for product in pathway.final_products:
        print(f"  - {product}")

    print("\nRegulatory mechanisms:")
    for metabolite, effect in pathway.regulatory_metabolites.items():
        print(f"  - {metabolite}: {effect}")

    print("\nAllosteric effectors:")
    for effector, factor in pathway.allosteric_effectors.items():
        factor_text = "activation" if factor > 1.0 else "inhibition"
        print(f"  - {effector}: {factor:.1f}x ({factor_text})")

    print("\nOptimal conditions:")
    optimal = pathway.optimal_conditions
    print(f"  - Temperature: {optimal['temperature'] - 273.15:.1f}°C")
    print(f"  - pH: {optimal['ph']:.1f}")
    print(f"  - Ionic strength: {optimal['ionic_strength']:.2f} M")
    print(f"  - Redox potential: {optimal['redox_potential']:+.2f} V vs SHE")

def validate_enhanced_configuration():
    """Validate the enhanced configuration."""

    print("\n=== Configuration Validation ===")

    shewanella_config, lactate_config = create_enhanced_shewanella_lactate_config()

    try:
        validate_species_metabolic_config(shewanella_config)
        print("✓ Enhanced Shewanella configuration is valid")
    except Exception as e:
        print(f"✗ Shewanella configuration validation failed: {e}")
        return False

    try:
        validate_comprehensive_substrate_config(lactate_config)
        print("✓ Enhanced lactate configuration is valid")
    except Exception as e:
        print(f"✗ Lactate configuration validation failed: {e}")
        return False

    return True

if __name__ == "__main__":
    print("Enhanced Shewanella oneidensis + Lactate Configuration Example")
    print("=" * 70)

    # Validate configuration
    if not validate_enhanced_configuration():
        print("\n✗ Configuration validation failed!")
        sys.exit(1)

    # Demonstrate environmental adaptation
    demonstrate_environmental_adaptation()

    # Analyze metabolic pathway
    analyze_metabolic_pathway()

    print("\n✓ Enhanced configuration example completed successfully!")
    print("\nThis configuration can be used with:")
    print("  - MetabolicModel for detailed flux analysis")
    print("  - BiofilmKineticsModel for growth simulation")
    print("  - Sensor-integrated models for real-time monitoring")
