"""
Biological configuration classes for metabolic and biofilm models.
Includes species-specific parameters, substrate properties, and literature references.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum


class BacterialSpecies(Enum):
    """Supported bacterial species for MFC applications."""
    GEOBACTER_SULFURREDUCENS = "geobacter_sulfurreducens"
    SHEWANELLA_ONEIDENSIS = "shewanella_oneidensis"
    MIXED_CULTURE = "mixed_culture"


class SubstrateType(Enum):
    """Supported substrate types."""
    ACETATE = "acetate"
    LACTATE = "lactate"
    PYRUVATE = "pyruvate"
    GLUCOSE = "glucose"


@dataclass
class LiteratureReference:
    """Container for literature references."""
    authors: str
    title: str
    journal: str
    year: int
    doi: Optional[str] = None
    pmid: Optional[str] = None
    pages: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.authors} ({self.year}). {self.title}. {self.journal}."


@dataclass
class KineticParameters:
    """Kinetic parameters for enzymatic reactions with literature references."""

    # Michaelis-Menten parameters
    vmax: float  # Maximum reaction rate (mmol/gDW/h)
    km: float    # Michaelis constant (mmol/L)
    ki: Optional[float] = None  # Inhibition constant (mmol/L)

    # Temperature dependence
    ea: float = 50.0  # Activation energy (kJ/mol)
    temperature_ref: float = 303.0  # Reference temperature (K)

    # pH dependence
    ph_optimal: float = 7.0  # Optimal pH
    ph_tolerance: float = 1.0  # pH tolerance range

    # Literature reference
    reference: Optional[LiteratureReference] = None


@dataclass
class MetabolicReactionConfig:
    """Configuration for a single metabolic reaction."""

    id: str  # Reaction identifier
    name: str  # Reaction name
    equation: str  # Balanced chemical equation

    # Stoichiometry
    stoichiometry: Dict[str, float]  # {metabolite: coefficient}

    # Enzyme information
    enzyme_name: str

    # Kinetic parameters
    kinetics: KineticParameters

    # Thermodynamics
    delta_g0: float  # Standard Gibbs free energy (kJ/mol)

    # Optional fields with defaults
    ec_number: Optional[str] = None
    kegg_id: Optional[str] = None
    reversible: bool = True

    # Flux constraints
    flux_lower_bound: float = -1000.0  # mmol/gDW/h
    flux_upper_bound: float = 1000.0   # mmol/gDW/h


@dataclass
class SpeciesMetabolicConfig:
    """Species-specific metabolic configuration."""

    species: BacterialSpecies

    # Central metabolism reactions
    reactions: List[MetabolicReactionConfig] = field(default_factory=list)

    # Key metabolite concentrations (mmol/L)
    metabolite_concentrations: Dict[str, float] = field(default_factory=dict)

    # Electron transport parameters
    electron_transport_efficiency: float = 0.85  # Efficiency of electron transport to electrode
    cytochrome_content: float = 0.1  # mmol cytochrome c per gDW

    # Growth parameters
    max_growth_rate: float = 0.3  # 1/h - maximum specific growth rate
    maintenance_coefficient: float = 0.05  # 1/h - maintenance energy coefficient
    yield_coefficient: float = 0.1  # gDW/mmol substrate

    # Biofilm formation parameters
    attachment_rate: float = 0.1  # 1/h - cell attachment rate
    detachment_rate: float = 0.01  # 1/h - cell detachment rate
    max_biofilm_thickness: float = 100.0  # μm - maximum sustainable thickness

    # Environmental tolerances
    temperature_range: Tuple[float, float] = (273.0, 333.0)  # K
    ph_range: Tuple[float, float] = (5.0, 9.0)
    salinity_tolerance: float = 0.5  # M NaCl

    # Literature references
    references: List[LiteratureReference] = field(default_factory=list)


@dataclass
class SubstrateProperties:
    """Properties of organic substrates."""

    substrate: SubstrateType

    # Chemical properties
    molecular_weight: float  # g/mol
    formula: str  # Chemical formula

    # Thermodynamic properties
    delta_g_formation: float  # kJ/mol - standard Gibbs free energy of formation
    delta_h_formation: float  # kJ/mol - standard enthalpy of formation

    # Solubility and transport
    solubility: float  # g/L - water solubility at 25°C
    diffusion_coefficient: float  # cm²/s - diffusion coefficient in water

    # Biodegradation properties
    electron_equivalents: float  # electrons per molecule
    theoretical_cod: float  # mg COD/mg substrate

    # Uptake kinetics (species-specific)
    uptake_kinetics: Dict[BacterialSpecies, KineticParameters] = field(default_factory=dict)

    # Literature reference
    reference: Optional[LiteratureReference] = None


@dataclass
class BiofilmKineticsConfig:
    """Configuration for biofilm formation and growth kinetics."""

    # Physical properties
    biofilm_density: float = 1050.0  # kg/m³ - typical biofilm density
    porosity: float = 0.8  # Void fraction in biofilm
    tortuosity: float = 1.5  # Diffusion tortuosity factor

    # Growth kinetics
    monod_kinetics: Dict[str, float] = field(default_factory=lambda: {
        'max_growth_rate': 0.3,  # 1/h
        'half_saturation': 0.5,  # mmol/L
        'yield_coefficient': 0.1,  # gDW/mmol
        'decay_rate': 0.001  # 1/h
    })

    # Nernst-Monod parameters for electroactive biofilms
    nernst_monod: Dict[str, float] = field(default_factory=lambda: {
        'standard_potential': -0.3,  # V vs SHE
        'electron_transfer_rate': 1.0,  # 1/s
        'biofilm_conductivity': 0.005,  # S/m
        'double_layer_capacitance': 1e-6  # F/cm²
    })

    # Mass transfer coefficients
    mass_transfer: Dict[str, float] = field(default_factory=lambda: {
        'boundary_layer_thickness': 0.1,  # mm
        'substrate_diffusivity': 1.0e-9,  # m²/s
        'oxygen_diffusivity': 2.0e-9,  # m²/s
        'product_diffusivity': 0.8e-9  # m²/s
    })

    # Biofilm structure parameters
    structure: Dict[str, float] = field(default_factory=lambda: {
        'critical_thickness': 50.0,  # μm - critical thickness for layering
        'detachment_shear_stress': 0.5,  # Pa
        'compaction_factor': 0.9,  # Compaction with age
        'roughness_factor': 1.2  # Surface roughness multiplier
    })

    # Literature references
    references: List[LiteratureReference] = field(default_factory=list)


@dataclass
class ElectrochemicalConfig:
    """Configuration for electrochemical parameters."""

    # Fundamental constants
    faraday_constant: float = 96485.0  # C/mol
    gas_constant: float = 8.314  # J/mol/K
    avogadro_number: float = 6.022e23  # 1/mol

    # Standard electrode potentials (V vs SHE)
    standard_potentials: Dict[str, float] = field(default_factory=lambda: {
        'acetate_co2': -0.28,  # Acetate/CO2 couple
        'lactate_pyruvate': -0.19,  # Lactate/pyruvate couple
        'nad_nadh': -0.32,  # NAD+/NADH couple
        'cytochrome_c': 0.22,  # Cytochrome c (ox/red)
        'fumarate_succinate': 0.03  # Fumarate/succinate couple
    })

    # Electrode materials
    electrode_properties: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'carbon_cloth': {
            'surface_area': 0.5,  # m²/g
            'conductivity': 25000.0,  # S/m
            'porosity': 0.75,
            'roughness_factor': 10.0
        },
        'graphite_rod': {
            'surface_area': 0.001,  # m²/g
            'conductivity': 100000.0,  # S/m
            'porosity': 0.0,
            'roughness_factor': 1.0
        }
    })

    # Membrane properties
    membrane_properties: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'nafion_117': {
            'thickness': 0.175,  # mm
            'conductivity': 0.1,  # S/cm
            'permeability_o2': 3.1e-10,  # mol/m/s/Pa
            'water_uptake': 14.0  # H2O/SO3H
        },
        'cationic_exchange': {
            'thickness': 0.2,  # mm
            'conductivity': 0.05,  # S/cm
            'permeability_o2': 1.5e-10,  # mol/m/s/Pa
            'water_uptake': 10.0  # H2O/SO3H
        }
    })

    # Literature references
    references: List[LiteratureReference] = field(default_factory=list)


# Literature references database
LITERATURE_REFERENCES = {
    'lovley_2003': LiteratureReference(
        authors="Lovley, D.R.",
        title="Cleaning up with genomics: applying molecular biology to bioremediation",
        journal="Nature Reviews Microbiology",
        year=2003,
        doi="10.1038/nrmicro731"
    ),
    'bond_2002': LiteratureReference(
        authors="Bond, D.R., Holmes, D.E., Tender, L.M., Lovley, D.R.",
        title="Electrode-reducing microorganisms that harvest energy from marine sediments",
        journal="Science",
        year=2002,
        doi="10.1126/science.1067535"
    ),
    'reguera_2005': LiteratureReference(
        authors="Reguera, G., McCarthy, K.D., Mehta, T., Nicoll, J.S., Tuominen, M.T., Lovley, D.R.",
        title="Extracellular electron transfer via microbial nanowires",
        journal="Nature",
        year=2005,
        doi="10.1038/nature03661"
    ),
    'marsili_2008': LiteratureReference(
        authors="Marsili, E., Baron, D.B., Shikhare, I.D., Coursolle, D., Gralnick, J.A., Bond, D.R.",
        title="Shewanella secretes flavins that mediate extracellular electron transfer",
        journal="Proceedings of the National Academy of Sciences",
        year=2008,
        doi="10.1073/pnas.0710525105"
    ),
    'torres_2010': LiteratureReference(
        authors="Torres, C.I., Marcus, A.K., Lee, H.S., Parameswaran, P., Krajmalnik-Brown, R., Rittmann, B.E.",
        title="A kinetic perspective on extracellular electron transfer by anode-respiring bacteria",
        journal="FEMS Microbiology Reviews",
        year=2010,
        doi="10.1111/j.1574-6976.2009.00209.x"
    ),
    'marcus_2007': LiteratureReference(
        authors="Marcus, A.K., Torres, C.I., Rittmann, B.E.",
        title="Conduction-based modeling of the biofilm anode of a microbial fuel cell",
        journal="Biotechnology and Bioengineering",
        year=2007,
        doi="10.1002/bit.21533"
    )
}


# Default configurations for common species
def get_geobacter_config() -> SpeciesMetabolicConfig:
    """Get default configuration for Geobacter sulfurreducens."""

    # Acetyl-CoA synthetase kinetics
    acetyl_coa_synthetase_kinetics = KineticParameters(
        vmax=15.0,  # mmol/gDW/h
        km=0.5,     # mmol/L for acetate
        ea=45.0,    # kJ/mol
        ph_optimal=7.0,
        reference=LITERATURE_REFERENCES['lovley_2003']
    )

    # Define acetyl-CoA synthetase reaction
    acetyl_coa_synthetase = MetabolicReactionConfig(
        id="GSU_R001",
        name="Acetyl-CoA synthetase",
        equation="Acetate + CoA + ATP → Acetyl-CoA + AMP + PPi",
        stoichiometry={
            "acetate": -1.0, "coa": -1.0, "atp": -1.0,
            "acetyl_coa": 1.0, "amp": 1.0, "ppi": 1.0
        },
        enzyme_name="Acetyl-CoA synthetase",
        ec_number="EC 6.2.1.1",
        kegg_id="R00235",
        kinetics=acetyl_coa_synthetase_kinetics,
        delta_g0=-31.4,  # kJ/mol
        reversible=False,
        flux_lower_bound=0.0,
        flux_upper_bound=20.0
    )

    metabolite_concentrations = {
        "acetate": 10.0,  # mmol/L
        "lactate": 0.0,
        "pyruvate": 0.0,
        "acetyl_coa": 0.0,
        "citrate": 0.0,
        "co2": 0.0,
        "nadh": 0.1,
        "nad_plus": 1.0,
        "atp": 5.0,
        "adp": 5.0,
        "cytochrome_c_red": 0.1,
        "cytochrome_c_ox": 0.9,
        "o2": 0.001,  # Low oxygen (anaerobic)
        "h_plus": 1e-4  # pH 7.0 in mol/L
    }

    references = [
        LITERATURE_REFERENCES['lovley_2003'],
        LITERATURE_REFERENCES['bond_2002'],
        LITERATURE_REFERENCES['reguera_2005']
    ]

    return SpeciesMetabolicConfig(
        species=BacterialSpecies.GEOBACTER_SULFURREDUCENS,
        reactions=[acetyl_coa_synthetase],  # Add more reactions as needed
        metabolite_concentrations=metabolite_concentrations,
        electron_transport_efficiency=0.85,
        cytochrome_content=0.15,  # Higher for Geobacter
        max_growth_rate=0.25,  # 1/h
        maintenance_coefficient=0.03,
        yield_coefficient=0.08,  # Lower yield, more energy to electrode
        attachment_rate=0.15,  # Good biofilm former
        detachment_rate=0.008,
        max_biofilm_thickness=120.0,  # μm
        temperature_range=(283.0, 318.0),  # Mesophilic
        ph_range=(6.0, 8.5),
        salinity_tolerance=0.3,  # M NaCl
        references=references
    )


def get_shewanella_config() -> SpeciesMetabolicConfig:
    """Get default configuration for Shewanella oneidensis MR-1."""

    # Lactate dehydrogenase kinetics
    lactate_dehydrogenase_kinetics = KineticParameters(
        vmax=25.0,  # mmol/gDW/h
        km=0.3,     # mmol/L for lactate
        ea=42.0,    # kJ/mol
        ph_optimal=7.2,
        reference=LITERATURE_REFERENCES['marsili_2008']
    )

    # Define lactate dehydrogenase reaction
    lactate_dehydrogenase = MetabolicReactionConfig(
        id="MR1_R001",
        name="Lactate dehydrogenase",
        equation="Lactate + NAD+ → Pyruvate + NADH + H+",
        stoichiometry={
            "lactate": -1.0, "nad_plus": -1.0,
            "pyruvate": 1.0, "nadh": 1.0, "h_plus": 1.0
        },
        enzyme_name="L-lactate dehydrogenase",
        ec_number="EC 1.1.1.27",
        kegg_id="R00703",
        kinetics=lactate_dehydrogenase_kinetics,
        delta_g0=-25.1,  # kJ/mol
        reversible=True,
        flux_lower_bound=-50.0,
        flux_upper_bound=50.0
    )

    metabolite_concentrations = {
        "acetate": 0.0,
        "lactate": 10.0,  # mmol/L
        "pyruvate": 0.1,
        "acetyl_coa": 0.0,
        "citrate": 0.0,
        "co2": 0.0,
        "nadh": 0.2,
        "nad_plus": 1.2,
        "atp": 4.5,
        "adp": 5.5,
        "cytochrome_c_red": 0.15,
        "cytochrome_c_ox": 0.85,
        "o2": 0.002,  # Slightly more oxygen tolerant
        "h_plus": 1e-4  # pH 7.0 in mol/L
    }

    references = [
        LITERATURE_REFERENCES['marsili_2008'],
        LITERATURE_REFERENCES['torres_2010']
    ]

    return SpeciesMetabolicConfig(
        species=BacterialSpecies.SHEWANELLA_ONEIDENSIS,
        reactions=[lactate_dehydrogenase],  # Add more reactions as needed
        metabolite_concentrations=metabolite_concentrations,
        electron_transport_efficiency=0.75,  # Uses flavins as mediators
        cytochrome_content=0.12,
        max_growth_rate=0.35,  # 1/h - faster growing
        maintenance_coefficient=0.04,
        yield_coefficient=0.12,  # Higher growth yield
        attachment_rate=0.08,   # Moderate biofilm former
        detachment_rate=0.012,
        max_biofilm_thickness=80.0,  # μm - thinner biofilms
        temperature_range=(278.0, 313.0),  # Psychrotrophic
        ph_range=(6.5, 8.0),
        salinity_tolerance=0.8,  # M NaCl - more salt tolerant
        references=references
    )


def get_default_substrate_properties() -> Dict[SubstrateType, SubstrateProperties]:
    """Get default substrate properties for common substrates."""

    acetate_uptake = {
        BacterialSpecies.GEOBACTER_SULFURREDUCENS: KineticParameters(
            vmax=20.0, km=0.5, ea=45.0,
            reference=LITERATURE_REFERENCES['lovley_2003']
        ),
        BacterialSpecies.SHEWANELLA_ONEIDENSIS: KineticParameters(
            vmax=8.0, km=1.2, ea=48.0,  # Less efficient with acetate
            reference=LITERATURE_REFERENCES['marsili_2008']
        )
    }

    lactate_uptake = {
        BacterialSpecies.GEOBACTER_SULFURREDUCENS: KineticParameters(
            vmax=5.0, km=2.0, ea=52.0,  # Limited lactate utilization
            reference=LITERATURE_REFERENCES['lovley_2003']
        ),
        BacterialSpecies.SHEWANELLA_ONEIDENSIS: KineticParameters(
            vmax=30.0, km=0.3, ea=42.0,  # Preferred substrate
            reference=LITERATURE_REFERENCES['marsili_2008']
        )
    }

    return {
        SubstrateType.ACETATE: SubstrateProperties(
            substrate=SubstrateType.ACETATE,
            molecular_weight=60.05,  # g/mol
            formula="C2H4O2",
            delta_g_formation=-369.3,  # kJ/mol
            delta_h_formation=-484.5,  # kJ/mol
            solubility=1000.0,  # g/L - very soluble
            diffusion_coefficient=1.29e-5,  # cm²/s in water at 25°C
            electron_equivalents=8.0,  # electrons per acetate
            theoretical_cod=1.07,  # mg COD/mg acetate
            uptake_kinetics=acetate_uptake
        ),

        SubstrateType.LACTATE: SubstrateProperties(
            substrate=SubstrateType.LACTATE,
            molecular_weight=90.08,  # g/mol
            formula="C3H6O3",
            delta_g_formation=-516.6,  # kJ/mol
            delta_h_formation=-694.0,  # kJ/mol
            solubility=1200.0,  # g/L
            diffusion_coefficient=1.04e-5,  # cm²/s in water at 25°C
            electron_equivalents=12.0,  # electrons per lactate
            theoretical_cod=1.07,  # mg COD/mg lactate
            uptake_kinetics=lactate_uptake
        )
    }


def get_default_biofilm_config() -> BiofilmKineticsConfig:
    """Get default biofilm kinetics configuration."""

    references = [
        LITERATURE_REFERENCES['marcus_2007'],
        LITERATURE_REFERENCES['torres_2010']
    ]

    return BiofilmKineticsConfig(
        biofilm_density=1050.0,  # kg/m³
        porosity=0.8,
        tortuosity=1.5,
        references=references
    )


def get_default_electrochemical_config() -> ElectrochemicalConfig:
    """Get default electrochemical configuration."""

    references = [
        LITERATURE_REFERENCES['bond_2002'],
        LITERATURE_REFERENCES['torres_2010']
    ]

    return ElectrochemicalConfig(references=references)
