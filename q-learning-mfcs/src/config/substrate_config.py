"""
Substrate-specific configuration classes for MFC applications.
Includes detailed substrate properties, kinetics, and environmental effects.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from .biological_config import (
    SubstrateType, BacterialSpecies, LiteratureReference, 
    LITERATURE_REFERENCES
)


@dataclass
class SubstrateKineticsConfig:
    """Detailed kinetic configuration for substrate utilization."""
    
    # Basic Monod kinetics
    max_uptake_rate: float  # mmol/gDW/h - maximum substrate uptake rate
    half_saturation_constant: float  # mmol/L - Monod half-saturation constant
    
    # Inhibition kinetics
    substrate_inhibition_constant: Optional[float] = None  # mmol/L - substrate inhibition
    product_inhibition: Dict[str, float] = field(default_factory=dict)  # {product: Ki}
    competitive_inhibition: Dict[str, float] = field(default_factory=dict)  # {inhibitor: Ki}
    
    # Environmental effects
    temperature_coefficient: float = 1.07  # Q10 temperature coefficient
    ph_optimum: float = 7.0
    ph_tolerance_range: Tuple[float, float] = (6.0, 8.0)
    
    # Thermodynamic parameters
    activation_energy: float = 50.0  # kJ/mol - Arrhenius activation energy
    enthalpy_change: float = -120.0  # kJ/mol - reaction enthalpy
    entropy_change: float = -0.15  # kJ/mol/K - reaction entropy
    
    # Literature reference
    reference: Optional[LiteratureReference] = None


@dataclass
class SubstrateDegradationPathway:
    """Configuration for substrate degradation pathway."""
    
    pathway_name: str
    substrate: SubstrateType
    
    # Degradation steps
    intermediates: List[str]  # Intermediate metabolites
    final_products: List[str]  # Final products
    
    # Stoichiometry
    substrate_stoichiometry: float = 1.0  # moles substrate consumed
    electron_yield: float = 8.0  # electrons produced per substrate
    biomass_yield: float = 0.1  # gDW per mmol substrate
    
    # Energetics
    atp_yield: float = 2.0  # moles ATP per substrate
    nadh_yield: float = 4.0  # moles NADH per substrate
    co2_yield: float = 2.0  # moles CO2 per substrate
    
    # Pathway regulation
    regulatory_metabolites: Dict[str, str] = field(default_factory=dict)  # {metabolite: effect}
    allosteric_effectors: Dict[str, float] = field(default_factory=dict)  # {effector: factor}
    
    # Environmental conditions for optimal pathway flux
    optimal_conditions: Dict[str, float] = field(default_factory=lambda: {
        'temperature': 303.0,  # K
        'ph': 7.0,
        'ionic_strength': 0.1,  # M
        'redox_potential': -0.2  # V vs SHE
    })


@dataclass
class SubstrateTransportConfig:
    """Configuration for substrate transport across cell membrane."""
    
    # Transport mechanism
    transport_type: str = "facilitated_diffusion"  # "active", "passive", "facilitated_diffusion"
    
    # Transport kinetics
    max_transport_rate: float = 50.0  # mmol/gDW/h
    transport_km: float = 0.1  # mmol/L
    
    # Energy requirements
    atp_cost: float = 0.0  # ATP molecules per substrate transported
    pmf_cost: float = 1.0  # Protons per substrate (if applicable)
    
    # Regulation
    transport_regulation: Dict[str, float] = field(default_factory=dict)  # {regulator: effect}
    
    # Competition with other substrates
    competitive_substrates: List[str] = field(default_factory=list)
    competition_factors: Dict[str, float] = field(default_factory=dict)


@dataclass
class ComprehensiveSubstrateConfig:
    """Comprehensive configuration for a specific substrate."""
    
    substrate_type: SubstrateType
    
    # Chemical properties
    molecular_weight: float  # g/mol
    chemical_formula: str
    cas_number: str
    iupac_name: str
    
    # Physical properties
    density: float = 1.0  # g/cm³
    melting_point: float = 273.0  # K
    boiling_point: float = 373.0  # K
    vapor_pressure: float = 0.0  # Pa at 25°C
    
    # Solubility properties
    water_solubility: float = 1000.0  # g/L at 25°C
    log_kow: float = 0.0  # Octanol-water partition coefficient
    
    # Diffusion properties
    diffusion_coefficient_water: float = 1.0e-5  # cm²/s at 25°C
    diffusion_coefficient_biofilm: float = 5.0e-6  # cm²/s in biofilm
    
    # Biodegradation kinetics (species-specific)
    species_kinetics: Dict[BacterialSpecies, SubstrateKineticsConfig] = field(default_factory=dict)
    
    # Degradation pathways
    degradation_pathways: List[SubstrateDegradationPathway] = field(default_factory=list)
    
    # Transport configuration
    transport_config: SubstrateTransportConfig = field(default_factory=SubstrateTransportConfig)
    
    # Toxicity and inhibition
    toxicity_threshold: Optional[float] = None  # mmol/L - toxic concentration
    inhibitory_effects: Dict[str, float] = field(default_factory=dict)  # {process: IC50}
    
    # Environmental fate
    biodegradability: float = 1.0  # Fraction biodegradable (0-1)
    half_life_aerobic: float = 168.0  # hours - aerobic biodegradation half-life
    half_life_anaerobic: float = 720.0  # hours - anaerobic biodegradation half-life
    
    # Analytical properties
    detection_methods: List[str] = field(default_factory=list)
    quantification_limit: float = 0.001  # mmol/L - analytical detection limit
    
    # Literature references
    references: List[LiteratureReference] = field(default_factory=list)


# Pre-configured substrate configurations
def get_acetate_config() -> ComprehensiveSubstrateConfig:
    """Get comprehensive configuration for acetate."""
    
    # Geobacter acetate kinetics
    geobacter_kinetics = SubstrateKineticsConfig(
        max_uptake_rate=20.0,  # mmol/gDW/h
        half_saturation_constant=0.5,  # mmol/L
        substrate_inhibition_constant=50.0,  # mmol/L
        temperature_coefficient=1.08,
        ph_optimum=7.0,
        ph_tolerance_range=(6.0, 8.5),
        activation_energy=45.0,  # kJ/mol
        enthalpy_change=-125.0,  # kJ/mol
        entropy_change=-0.12,  # kJ/mol/K
        reference=LITERATURE_REFERENCES['lovley_2003']
    )
    
    # Shewanella acetate kinetics (limited)
    shewanella_kinetics = SubstrateKineticsConfig(
        max_uptake_rate=8.0,  # mmol/gDW/h - lower efficiency
        half_saturation_constant=1.2,  # mmol/L
        substrate_inhibition_constant=30.0,  # mmol/L
        temperature_coefficient=1.06,
        ph_optimum=7.2,
        ph_tolerance_range=(6.5, 8.0),
        activation_energy=48.0,  # kJ/mol
        enthalpy_change=-110.0,  # kJ/mol
        entropy_change=-0.10,  # kJ/mol/K
        reference=LITERATURE_REFERENCES['marsili_2008']
    )
    
    # Acetate degradation pathway
    acetate_pathway = SubstrateDegradationPathway(
        pathway_name="Acetate oxidation to CO2",
        substrate=SubstrateType.ACETATE,
        intermediates=["acetyl_coa", "citrate", "oxaloacetate"],
        final_products=["co2", "h2o"],
        substrate_stoichiometry=1.0,
        electron_yield=8.0,  # electrons per acetate
        biomass_yield=0.08,  # gDW per mmol acetate
        atp_yield=1.0,  # Net ATP yield
        nadh_yield=4.0,  # NADH produced
        co2_yield=2.0,  # CO2 produced
        optimal_conditions={
            'temperature': 303.0,  # K
            'ph': 7.0,
            'ionic_strength': 0.15,  # M
            'redox_potential': -0.25  # V vs SHE
        }
    )
    
    # Transport configuration
    transport_config = SubstrateTransportConfig(
        transport_type="facilitated_diffusion",
        max_transport_rate=40.0,  # mmol/gDW/h
        transport_km=0.2,  # mmol/L
        atp_cost=0.0,  # No energy required
        pmf_cost=1.0,  # One proton per acetate
        competitive_substrates=["propionate", "butyrate"],
        competition_factors={"propionate": 0.8, "butyrate": 0.6}
    )
    
    return ComprehensiveSubstrateConfig(
        substrate_type=SubstrateType.ACETATE,
        molecular_weight=60.05,  # g/mol
        chemical_formula="C2H4O2",
        cas_number="64-19-7",
        iupac_name="acetic acid",
        density=1.049,  # g/cm³
        melting_point=289.8,  # K
        boiling_point=391.0,  # K
        vapor_pressure=1533.0,  # Pa at 25°C
        water_solubility=1000.0,  # g/L - completely miscible
        log_kow=-0.17,  # Hydrophilic
        diffusion_coefficient_water=1.29e-5,  # cm²/s at 25°C
        diffusion_coefficient_biofilm=6.5e-6,  # cm²/s in biofilm
        species_kinetics={
            BacterialSpecies.GEOBACTER_SULFURREDUCENS: geobacter_kinetics,
            BacterialSpecies.SHEWANELLA_ONEIDENSIS: shewanella_kinetics
        },
        degradation_pathways=[acetate_pathway],
        transport_config=transport_config,
        toxicity_threshold=500.0,  # mmol/L - high tolerance
        biodegradability=0.98,  # Readily biodegradable
        half_life_aerobic=24.0,  # hours
        half_life_anaerobic=72.0,  # hours
        detection_methods=["GC-MS", "HPLC", "IC"],
        quantification_limit=0.01,  # mmol/L
        references=[
            LITERATURE_REFERENCES['lovley_2003'],
            LITERATURE_REFERENCES['bond_2002']
        ]
    )


def get_lactate_config() -> ComprehensiveSubstrateConfig:
    """Get comprehensive configuration for lactate."""
    
    # Shewanella lactate kinetics (preferred substrate)
    shewanella_kinetics = SubstrateKineticsConfig(
        max_uptake_rate=30.0,  # mmol/gDW/h
        half_saturation_constant=0.3,  # mmol/L
        substrate_inhibition_constant=80.0,  # mmol/L
        temperature_coefficient=1.09,
        ph_optimum=7.2,
        ph_tolerance_range=(6.5, 8.0),
        activation_energy=42.0,  # kJ/mol
        enthalpy_change=-135.0,  # kJ/mol
        entropy_change=-0.18,  # kJ/mol/K
        reference=LITERATURE_REFERENCES['marsili_2008']
    )
    
    # Geobacter lactate kinetics (limited capability)
    geobacter_kinetics = SubstrateKineticsConfig(
        max_uptake_rate=8.0,  # mmol/gDW/h - limited utilization
        half_saturation_constant=2.0,  # mmol/L
        substrate_inhibition_constant=25.0,  # mmol/L
        temperature_coefficient=1.06,
        ph_optimum=7.0,
        ph_tolerance_range=(6.0, 8.0),
        activation_energy=52.0,  # kJ/mol
        enthalpy_change=-115.0,  # kJ/mol
        entropy_change=-0.14,  # kJ/mol/K
        reference=LITERATURE_REFERENCES['lovley_2003']
    )
    
    # Lactate degradation pathway
    lactate_pathway = SubstrateDegradationPathway(
        pathway_name="Lactate oxidation via pyruvate",
        substrate=SubstrateType.LACTATE,
        intermediates=["pyruvate", "acetyl_coa", "acetate"],
        final_products=["co2", "h2o", "acetate"],
        substrate_stoichiometry=1.0,
        electron_yield=12.0,  # electrons per lactate
        biomass_yield=0.12,  # gDW per mmol lactate
        atp_yield=2.5,  # Net ATP yield
        nadh_yield=5.0,  # NADH produced
        co2_yield=3.0,  # CO2 produced
        optimal_conditions={
            'temperature': 308.0,  # K - slightly higher optimal
            'ph': 7.2,
            'ionic_strength': 0.12,  # M
            'redox_potential': -0.18  # V vs SHE
        }
    )
    
    # Transport configuration
    transport_config = SubstrateTransportConfig(
        transport_type="active_transport",
        max_transport_rate=50.0,  # mmol/gDW/h
        transport_km=0.15,  # mmol/L
        atp_cost=0.5,  # ATP required for active transport
        pmf_cost=1.0,  # One proton per lactate
        competitive_substrates=["malate", "succinate"],
        competition_factors={"malate": 0.7, "succinate": 0.9}
    )
    
    return ComprehensiveSubstrateConfig(
        substrate_type=SubstrateType.LACTATE,
        molecular_weight=90.08,  # g/mol
        chemical_formula="C3H6O3",
        cas_number="50-21-5",
        iupac_name="2-hydroxypropanoic acid",
        density=1.206,  # g/cm³
        melting_point=326.0,  # K
        boiling_point=395.0,  # K (decomposes)
        vapor_pressure=0.1,  # Pa at 25°C
        water_solubility=1200.0,  # g/L - very soluble
        log_kow=-0.62,  # Hydrophilic
        diffusion_coefficient_water=1.04e-5,  # cm²/s at 25°C
        diffusion_coefficient_biofilm=5.2e-6,  # cm²/s in biofilm
        species_kinetics={
            BacterialSpecies.SHEWANELLA_ONEIDENSIS: shewanella_kinetics,
            BacterialSpecies.GEOBACTER_SULFURREDUCENS: geobacter_kinetics
        },
        degradation_pathways=[lactate_pathway],
        transport_config=transport_config,
        toxicity_threshold=200.0,  # mmol/L
        biodegradability=0.95,  # Readily biodegradable
        half_life_aerobic=18.0,  # hours
        half_life_anaerobic=48.0,  # hours
        detection_methods=["HPLC", "GC-MS", "enzymatic assay"],
        quantification_limit=0.005,  # mmol/L
        references=[
            LITERATURE_REFERENCES['marsili_2008'],
            LITERATURE_REFERENCES['torres_2010']
        ]
    )


def get_pyruvate_config() -> ComprehensiveSubstrateConfig:
    """Get comprehensive configuration for pyruvate."""
    
    # Universal pyruvate kinetics (both species can use it well)
    universal_kinetics = SubstrateKineticsConfig(
        max_uptake_rate=35.0,  # mmol/gDW/h
        half_saturation_constant=0.4,  # mmol/L
        substrate_inhibition_constant=60.0,  # mmol/L
        temperature_coefficient=1.10,
        ph_optimum=7.1,
        ph_tolerance_range=(6.5, 8.0),
        activation_energy=38.0,  # kJ/mol - lower barrier
        enthalpy_change=-140.0,  # kJ/mol
        entropy_change=-0.20,  # kJ/mol/K
        reference=LITERATURE_REFERENCES['torres_2010']
    )
    
    # Pyruvate degradation pathway
    pyruvate_pathway = SubstrateDegradationPathway(
        pathway_name="Pyruvate oxidation to acetyl-CoA",
        substrate=SubstrateType.PYRUVATE,
        intermediates=["acetyl_coa", "formate"],
        final_products=["co2", "h2o", "acetate"],
        substrate_stoichiometry=1.0,
        electron_yield=10.0,  # electrons per pyruvate
        biomass_yield=0.15,  # gDW per mmol pyruvate
        atp_yield=3.0,  # Net ATP yield
        nadh_yield=4.0,  # NADH produced
        co2_yield=3.0,  # CO2 produced
        optimal_conditions={
            'temperature': 305.0,  # K
            'ph': 7.1,
            'ionic_strength': 0.13,  # M
            'redox_potential': -0.20  # V vs SHE
        }
    )
    
    # Transport configuration
    transport_config = SubstrateTransportConfig(
        transport_type="facilitated_diffusion",
        max_transport_rate=60.0,  # mmol/gDW/h
        transport_km=0.25,  # mmol/L
        atp_cost=0.0,  # No energy required
        pmf_cost=1.0,  # One proton per pyruvate
        competitive_substrates=["lactate", "malate"],
        competition_factors={"lactate": 0.8, "malate": 0.7}
    )
    
    return ComprehensiveSubstrateConfig(
        substrate_type=SubstrateType.PYRUVATE,
        molecular_weight=88.06,  # g/mol
        chemical_formula="C3H4O3",
        cas_number="127-17-3",
        iupac_name="2-oxopropanoic acid",
        density=1.250,  # g/cm³
        melting_point=284.0,  # K
        boiling_point=438.0,  # K (decomposes)
        vapor_pressure=0.01,  # Pa at 25°C
        water_solubility=1000.0,  # g/L - very soluble
        log_kow=-1.15,  # Very hydrophilic
        diffusion_coefficient_water=0.98e-5,  # cm²/s at 25°C
        diffusion_coefficient_biofilm=4.9e-6,  # cm²/s in biofilm
        species_kinetics={
            BacterialSpecies.GEOBACTER_SULFURREDUCENS: universal_kinetics,
            BacterialSpecies.SHEWANELLA_ONEIDENSIS: universal_kinetics
        },
        degradation_pathways=[pyruvate_pathway],
        transport_config=transport_config,
        toxicity_threshold=150.0,  # mmol/L
        biodegradability=0.99,  # Readily biodegradable
        half_life_aerobic=12.0,  # hours
        half_life_anaerobic=36.0,  # hours
        detection_methods=["HPLC", "enzymatic assay", "GC-MS"],
        quantification_limit=0.002,  # mmol/L
        references=[
            LITERATURE_REFERENCES['torres_2010'],
            LITERATURE_REFERENCES['marcus_2007']
        ]
    )


def get_glucose_config() -> ComprehensiveSubstrateConfig:
    """Get comprehensive configuration for glucose (for complex substrates)."""
    
    # Glucose is typically not directly used by exoelectrogens but can be fermented
    fermentation_kinetics = SubstrateKineticsConfig(
        max_uptake_rate=15.0,  # mmol/gDW/h - lower for complex substrate
        half_saturation_constant=1.0,  # mmol/L
        substrate_inhibition_constant=100.0,  # mmol/L
        temperature_coefficient=1.12,
        ph_optimum=6.8,
        ph_tolerance_range=(6.0, 8.0),
        activation_energy=55.0,  # kJ/mol
        enthalpy_change=-150.0,  # kJ/mol
        entropy_change=-0.25,  # kJ/mol/K
    )
    
    # Glucose fermentation pathway
    glucose_pathway = SubstrateDegradationPathway(
        pathway_name="Glucose fermentation to organic acids",
        substrate=SubstrateType.GLUCOSE,
        intermediates=["glucose_6_phosphate", "pyruvate", "acetyl_coa"],
        final_products=["acetate", "lactate", "co2", "h2"],
        substrate_stoichiometry=1.0,
        electron_yield=24.0,  # electrons per glucose (theoretical)
        biomass_yield=0.2,  # gDW per mmol glucose
        atp_yield=2.0,  # Net ATP from fermentation
        nadh_yield=2.0,  # NADH produced
        co2_yield=2.0,  # CO2 produced
        optimal_conditions={
            'temperature': 310.0,  # K
            'ph': 6.8,
            'ionic_strength': 0.10,  # M
            'redox_potential': -0.30  # V vs SHE
        }
    )
    
    # Transport configuration
    transport_config = SubstrateTransportConfig(
        transport_type="active_transport",
        max_transport_rate=25.0,  # mmol/gDW/h
        transport_km=0.8,  # mmol/L
        atp_cost=1.0,  # ATP required for phosphorylation
        pmf_cost=0.0,  # No proton coupling
        competitive_substrates=["fructose", "galactose"],
        competition_factors={"fructose": 0.9, "galactose": 0.7}
    )
    
    return ComprehensiveSubstrateConfig(
        substrate_type=SubstrateType.GLUCOSE,
        molecular_weight=180.16,  # g/mol
        chemical_formula="C6H12O6",
        cas_number="50-99-7",
        iupac_name="D-glucose",
        density=1.544,  # g/cm³
        melting_point=419.0,  # K
        boiling_point=623.0,  # K (decomposes)
        vapor_pressure=0.0,  # Pa at 25°C (negligible)
        water_solubility=900.0,  # g/L at 25°C
        log_kow=-3.24,  # Very hydrophilic
        diffusion_coefficient_water=0.67e-5,  # cm²/s at 25°C
        diffusion_coefficient_biofilm=3.4e-6,  # cm²/s in biofilm
        species_kinetics={
            BacterialSpecies.GEOBACTER_SULFURREDUCENS: fermentation_kinetics,
            BacterialSpecies.SHEWANELLA_ONEIDENSIS: fermentation_kinetics,
            BacterialSpecies.MIXED_CULTURE: fermentation_kinetics
        },
        degradation_pathways=[glucose_pathway],
        transport_config=transport_config,
        toxicity_threshold=1000.0,  # mmol/L - very high tolerance
        biodegradability=1.0,  # Completely biodegradable
        half_life_aerobic=8.0,  # hours
        half_life_anaerobic=24.0,  # hours
        detection_methods=["HPLC", "enzymatic assay", "GC-MS"],
        quantification_limit=0.01,  # mmol/L
        references=[
            LITERATURE_REFERENCES['torres_2010']
        ]
    )


# Default substrate configurations
DEFAULT_SUBSTRATE_CONFIGS = {
    SubstrateType.ACETATE: get_acetate_config(),
    SubstrateType.LACTATE: get_lactate_config(),
    SubstrateType.PYRUVATE: get_pyruvate_config(),
    SubstrateType.GLUCOSE: get_glucose_config()
}