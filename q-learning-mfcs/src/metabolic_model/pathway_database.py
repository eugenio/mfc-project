"""Metabolic pathway database with KEGG-based pathways and literature values.

This module contains comprehensive metabolic pathway data for G. sulfurreducens
and S. oneidensis MR-1, including stoichiometric coefficients, enzyme kinetics,
and flux constraints derived from KEGG and literature sources.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Species(Enum):
    """Supported bacterial species."""

    GEOBACTER = "geobacter_sulfurreducens"
    SHEWANELLA = "shewanella_oneidensis"


class Substrate(Enum):
    """Supported substrate types."""

    ACETATE = "acetate"
    LACTATE = "lactate"


@dataclass
class MetabolicReaction:
    """Container for metabolic reaction data."""

    id: str  # Reaction identifier
    name: str  # Reaction name
    equation: str  # Reaction equation
    stoichiometry: dict[str, float]  # Stoichiometric coefficients
    enzyme: str  # Enzyme name/EC number
    kegg_id: str  # KEGG reaction ID

    # Kinetic parameters
    vmax: float  # Maximum reaction rate (mmol/gDW/h)
    km_values: dict[str, float]  # Michaelis constants (mmol/L)
    ki_values: dict[str, float]  # Inhibition constants (mmol/L)

    # Thermodynamic parameters
    delta_g0: float  # Standard Gibbs free energy (kJ/mol)
    reversible: bool  # Is reaction reversible?

    # Flux constraints
    lb: float  # Lower bound (mmol/gDW/h)
    ub: float  # Upper bound (mmol/gDW/h)


@dataclass
class MetabolicPathway:
    """Container for complete metabolic pathway."""

    name: str  # Pathway name
    reactions: list[MetabolicReaction]  # List of reactions
    key_metabolites: list[str]  # Important metabolites
    electron_yield: float  # Total electrons per substrate
    energy_yield: float  # ATP yield per substrate
    byproducts: dict[str, float]  # Byproduct stoichiometry


class PathwayDatabase:
    """Database of metabolic pathways for exoelectrogenic bacteria."""

    def __init__(self) -> None:
        """Initialize pathway database with KEGG and literature data."""
        self._pathways = self._load_pathways()
        self._metabolites = self._load_metabolites()
        self._electron_carriers = self._load_electron_carriers()

    def _load_pathways(self) -> dict[tuple[Species, Substrate], MetabolicPathway]:
        """Load metabolic pathways from KEGG and literature."""
        pathways = {}

        # G. sulfurreducens acetate metabolism
        geobacter_acetate = self._create_geobacter_acetate_pathway()
        pathways[(Species.GEOBACTER, Substrate.ACETATE)] = geobacter_acetate

        # G. sulfurreducens lactate metabolism (limited capability)
        geobacter_lactate = self._create_geobacter_lactate_pathway()
        pathways[(Species.GEOBACTER, Substrate.LACTATE)] = geobacter_lactate

        # S. oneidensis lactate metabolism
        shewanella_lactate = self._create_shewanella_lactate_pathway()
        pathways[(Species.SHEWANELLA, Substrate.LACTATE)] = shewanella_lactate

        # S. oneidensis acetate metabolism (limited capability)
        shewanella_acetate = self._create_shewanella_acetate_pathway()
        pathways[(Species.SHEWANELLA, Substrate.ACETATE)] = shewanella_acetate

        return pathways

    def _create_geobacter_acetate_pathway(self) -> MetabolicPathway:
        """Create G. sulfurreducens acetate metabolism pathway."""
        # Reaction 1: Acetyl-CoA synthetase (KEGG: R00235)
        acetyl_coa_synthetase = MetabolicReaction(
            id="GSU_R001",
            name="Acetyl-CoA synthetase",
            equation="Acetate + CoA + ATP → Acetyl-CoA + AMP + PPi",
            stoichiometry={
                "acetate": -1.0,
                "coa": -1.0,
                "atp": -1.0,
                "acetyl_coa": 1.0,
                "amp": 1.0,
                "ppi": 1.0,
            },
            enzyme="Acetyl-CoA synthetase (EC 6.2.1.1)",
            kegg_id="R00235",
            vmax=15.0,  # mmol/gDW/h
            km_values={"acetate": 0.5, "coa": 0.05, "atp": 0.1},
            ki_values={},
            delta_g0=-31.4,  # kJ/mol
            reversible=False,
            lb=0.0,
            ub=20.0,
        )

        # Reaction 2: Citrate synthase (KEGG: R00351)
        citrate_synthase = MetabolicReaction(
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
            enzyme="Citrate synthase (EC 2.3.3.1)",
            kegg_id="R00351",
            vmax=12.0,
            km_values={"acetyl_coa": 0.02, "oxaloacetate": 0.01},
            ki_values={},
            delta_g0=-32.2,
            reversible=False,
            lb=0.0,
            ub=15.0,
        )

        # Reaction 3: CO2 reduction (overall)
        co2_reduction = MetabolicReaction(
            id="GSU_R003",
            name="CO2 reduction to acetate",
            equation="2CO2 + 8H+ + 8e- → Acetate + 2H2O",
            stoichiometry={
                "co2": -2.0,
                "h_plus": -8.0,
                "electron": -8.0,
                "acetate": 1.0,
                "h2o": 2.0,
            },
            enzyme="CO2 reductase complex",
            kegg_id="R00014",
            vmax=8.0,
            km_values={"co2": 0.1},
            ki_values={},
            delta_g0=47.6,  # kJ/mol (unfavorable without electron acceptor)
            reversible=True,
            lb=-10.0,
            ub=10.0,
        )

        # Reaction 4: Electron transport to electrode
        electron_transport = MetabolicReaction(
            id="GSU_R004",
            name="Electron transport to anode",
            equation="8 Cyt_c_red → 8 Cyt_c_ox + 8e- (to anode)",
            stoichiometry={
                "cytochrome_c_red": -8.0,
                "cytochrome_c_ox": 8.0,
                "electron_anode": 8.0,
            },
            enzyme="Outer membrane cytochromes",
            kegg_id="custom",
            vmax=20.0,
            km_values={"cytochrome_c_red": 0.001},
            ki_values={},
            delta_g0=-15.0,  # Depends on anode potential
            reversible=False,
            lb=0.0,
            ub=25.0,
        )

        reactions = [
            acetyl_coa_synthetase,
            citrate_synthase,
            co2_reduction,
            electron_transport,
        ]

        return MetabolicPathway(
            name="Geobacter sulfurreducens acetate metabolism",
            reactions=reactions,
            key_metabolites=["acetate", "acetyl_coa", "co2", "cytochrome_c_red"],
            electron_yield=8.0,  # electrons per acetate
            energy_yield=0.5,  # ATP per acetate (net)
            byproducts={"co2": 2.0, "h2o": 2.0},
        )

    def _create_geobacter_lactate_pathway(self) -> MetabolicPathway:
        """Create G. sulfurreducens lactate metabolism (limited)."""
        # G. sulfurreducens has limited lactate utilization
        lactate_oxidation = MetabolicReaction(
            id="GSU_L001",
            name="Lactate oxidation (limited)",
            equation="Lactate + 2H2O → Pyruvate + 4H+ + 4e-",
            stoichiometry={
                "lactate": -1.0,
                "h2o": -2.0,
                "pyruvate": 1.0,
                "h_plus": 4.0,
                "electron": 4.0,
            },
            enzyme="Lactate dehydrogenase (limited activity)",
            kegg_id="R00703",
            vmax=2.0,  # Much lower than acetate
            km_values={"lactate": 5.0},  # Higher Km (lower affinity)
            ki_values={},
            delta_g0=-25.1,
            reversible=True,
            lb=-5.0,
            ub=5.0,
        )

        return MetabolicPathway(
            name="Geobacter sulfurreducens lactate metabolism (limited)",
            reactions=[lactate_oxidation],
            key_metabolites=["lactate", "pyruvate"],
            electron_yield=4.0,
            energy_yield=0.1,  # Very low ATP yield
            byproducts={"pyruvate": 1.0},
        )

    def _create_shewanella_lactate_pathway(self) -> MetabolicPathway:
        """Create S. oneidensis lactate metabolism pathway."""
        # Reaction 1: Lactate dehydrogenase (KEGG: R00703)
        lactate_dehydrogenase = MetabolicReaction(
            id="SON_R001",
            name="Lactate dehydrogenase",
            equation="L-Lactate + NAD+ → Pyruvate + NADH + H+",
            stoichiometry={
                "lactate": -1.0,
                "nad_plus": -1.0,
                "pyruvate": 1.0,
                "nadh": 1.0,
                "h_plus": 1.0,
            },
            enzyme="L-lactate dehydrogenase (EC 1.1.1.27)",
            kegg_id="R00703",
            vmax=25.0,  # mmol/gDW/h (high activity)
            km_values={"lactate": 0.8, "nad_plus": 0.05},
            ki_values={},
            delta_g0=-25.1,
            reversible=True,
            lb=-30.0,
            ub=30.0,
        )

        # Reaction 2: Pyruvate dehydrogenase complex (KEGG: R00014)
        pyruvate_dehydrogenase = MetabolicReaction(
            id="SON_R002",
            name="Pyruvate dehydrogenase",
            equation="Pyruvate + NAD+ + CoA → Acetyl-CoA + NADH + CO2",
            stoichiometry={
                "pyruvate": -1.0,
                "nad_plus": -1.0,
                "coa": -1.0,
                "acetyl_coa": 1.0,
                "nadh": 1.0,
                "co2": 1.0,
            },
            enzyme="Pyruvate dehydrogenase complex (EC 1.2.4.1)",
            kegg_id="R00014",
            vmax=18.0,
            km_values={"pyruvate": 0.3, "nad_plus": 0.05},
            ki_values={},
            delta_g0=-33.4,
            reversible=False,
            lb=0.0,
            ub=25.0,
        )

        # Reaction 2b: Acetate secretion (Shewanella cannot utilize acetate efficiently)
        acetate_secretion = MetabolicReaction(
            id="SON_R002b",
            name="Acetate secretion",
            equation="Acetyl-CoA + H2O → Acetate + CoA + H+",
            stoichiometry={
                "acetyl_coa": -0.6,  # ~60% of acetyl-CoA is secreted as acetate
                "h2o": -0.6,
                "acetate": 0.6,  # Acetate secretion as byproduct
                "coa": 0.6,
                "h_plus": 0.6,
            },
            enzyme="Acetyl-CoA hydrolase / Phosphotransacetylase-Acetate kinase",
            kegg_id="R00315",  # Acetyl-CoA hydrolysis
            vmax=12.0,  # High rate of acetate secretion
            km_values={"acetyl_coa": 0.1},
            ki_values={},
            delta_g0=-7.5,  # Thermodynamically favorable
            reversible=False,
            lb=0.0,
            ub=15.0,
        )

        # Reaction 3: NADH electron transport
        nadh_transport = MetabolicReaction(
            id="SON_R003",
            name="NADH electron transport",
            equation="NADH + H+ + 0.5O2 → NAD+ + H2O (or to flavins)",
            stoichiometry={
                "nadh": -1.0,
                "h_plus": -1.0,
                "o2": -0.5,
                "nad_plus": 1.0,
                "h2o": 1.0,
            },
            enzyme="NADH dehydrogenase/flavin reductase",
            kegg_id="R00100",
            vmax=22.0,
            km_values={"nadh": 0.02},
            ki_values={},
            delta_g0=-52.6,
            reversible=False,
            lb=0.0,
            ub=30.0,
        )

        # Reaction 4: Flavin-mediated electron transfer
        flavin_transport = MetabolicReaction(
            id="SON_R004",
            name="Flavin-mediated electron transfer",
            equation="Flavin_red → Flavin_ox + e- (to anode)",
            stoichiometry={
                "flavin_reduced": -1.0,
                "flavin_oxidized": 1.0,
                "electron_anode": 1.0,
            },
            enzyme="Outer membrane cytochromes + flavins",
            kegg_id="custom",
            vmax=15.0,
            km_values={"flavin_reduced": 0.005},
            ki_values={},
            delta_g0=-12.0,
            reversible=False,
            lb=0.0,
            ub=20.0,
        )

        reactions = [
            lactate_dehydrogenase,
            pyruvate_dehydrogenase,
            acetate_secretion,
            nadh_transport,
            flavin_transport,
        ]

        return MetabolicPathway(
            name="Shewanella oneidensis lactate metabolism with acetate secretion",
            reactions=reactions,
            key_metabolites=[
                "lactate",
                "pyruvate",
                "acetyl_coa",
                "nadh",
                "flavin_reduced",
                "acetate",
            ],
            electron_yield=4.0,  # electrons per lactate via flavins
            energy_yield=1.5,  # ATP per lactate
            byproducts={
                "co2": 1.0,
                "h2o": 1.0,
                "acetate": 0.6,
            },  # Acetate as major byproduct
        )

    def _create_shewanella_acetate_pathway(self) -> MetabolicPathway:
        """Create S. oneidensis acetate metabolism (limited)."""
        # S. oneidensis can use acetate but with lower efficiency
        acetate_activation = MetabolicReaction(
            id="SON_A001",
            name="Acetate activation (limited)",
            equation="Acetate + ATP + CoA → Acetyl-CoA + AMP + PPi",
            stoichiometry={
                "acetate": -1.0,
                "atp": -1.0,
                "coa": -1.0,
                "acetyl_coa": 1.0,
                "amp": 1.0,
                "ppi": 1.0,
            },
            enzyme="Acetyl-CoA synthetase (lower activity)",
            kegg_id="R00235",
            vmax=5.0,  # Much lower than Geobacter
            km_values={"acetate": 2.0},  # Higher Km
            ki_values={},
            delta_g0=-31.4,
            reversible=False,
            lb=0.0,
            ub=8.0,
        )

        return MetabolicPathway(
            name="Shewanella oneidensis acetate metabolism (limited)",
            reactions=[acetate_activation],
            key_metabolites=["acetate", "acetyl_coa"],
            electron_yield=8.0,  # theoretical
            energy_yield=0.3,  # Low efficiency
            byproducts={"co2": 2.0},
        )

    def _load_metabolites(self) -> dict[str, dict[str, Any]]:
        """Load metabolite properties."""
        return {
            "acetate": {
                "formula": "C2H3O2-",
                "mw": 59.04,  # g/mol
                "charge": -1,
                "kegg_id": "C00033",
            },
            "lactate": {
                "formula": "C3H5O3-",
                "mw": 89.07,
                "charge": -1,
                "kegg_id": "C00186",
            },
            "pyruvate": {
                "formula": "C3H3O3-",
                "mw": 87.05,
                "charge": -1,
                "kegg_id": "C00022",
            },
            "acetyl_coa": {
                "formula": "C23H34N7O17P3S",
                "mw": 809.57,
                "charge": -4,
                "kegg_id": "C00024",
            },
            "nadh": {
                "formula": "C21H27N7O14P2",
                "mw": 663.43,
                "charge": -2,
                "kegg_id": "C00004",
            },
            "nad_plus": {
                "formula": "C21H26N7O14P2",
                "mw": 662.42,
                "charge": -1,
                "kegg_id": "C00003",
            },
        }

    def _load_electron_carriers(self) -> dict[str, dict[str, Any]]:
        """Load electron carrier properties."""
        return {
            "cytochrome_c": {
                "redox_potential": 0.25,  # V vs SHE
                "molecular_weight": 12000,  # Da
                "location": "periplasm",
            },
            "flavin_mononucleotide": {
                "redox_potential": -0.20,  # V vs SHE
                "molecular_weight": 456.34,  # Da
                "location": "extracellular",
            },
            "riboflavin": {
                "redox_potential": -0.21,  # V vs SHE
                "molecular_weight": 376.36,  # Da
                "location": "extracellular",
            },
        }

    def get_pathway(self, species: Species, substrate: Substrate) -> MetabolicPathway:
        """Get metabolic pathway for species-substrate combination.

        Args:
            species: Bacterial species
            substrate: Substrate type

        Returns:
            MetabolicPathway object

        Raises:
            KeyError: If combination not available

        """
        key = (species, substrate)
        if key not in self._pathways:
            available = list(self._pathways.keys())
            msg = (
                f"Pathway for {species.value} + {substrate.value} not available. "
                f"Available: {available}"
            )
            raise KeyError(
                msg,
            )

        return self._pathways[key]

    def get_metabolite_properties(self, metabolite: str) -> dict[str, Any]:
        """Get properties for specific metabolite."""
        if metabolite not in self._metabolites:
            msg = f"Metabolite '{metabolite}' not found in database"
            raise KeyError(msg)
        return self._metabolites[metabolite]

    def get_electron_carrier_properties(self, carrier: str) -> dict[str, Any]:
        """Get properties for specific electron carrier."""
        if carrier not in self._electron_carriers:
            msg = f"Electron carrier '{carrier}' not found in database"
            raise KeyError(msg)
        return self._electron_carriers[carrier]

    def calculate_pathway_stoichiometry(
        self,
        species: Species,
        substrate: Substrate,
    ) -> dict[str, float]:
        """Calculate overall pathway stoichiometry.

        Args:
            species: Bacterial species
            substrate: Substrate type

        Returns:
            Dictionary of net stoichiometric coefficients

        """
        pathway = self.get_pathway(species, substrate)

        # Sum stoichiometry across all reactions
        net_stoichiometry: dict[str, float] = {}
        for reaction in pathway.reactions:
            for metabolite, coeff in reaction.stoichiometry.items():
                net_stoichiometry[metabolite] = (
                    net_stoichiometry.get(metabolite, 0) + coeff
                )

        return net_stoichiometry

    def get_available_combinations(self) -> list[tuple[Species, Substrate]]:
        """Get list of available species-substrate combinations."""
        return list(self._pathways.keys())

    def get_kegg_pathway_ids(self, species: Species) -> dict[str, str]:
        """Get KEGG pathway IDs for species."""
        kegg_ids = {
            Species.GEOBACTER: {
                "central_metabolism": "gsu00020",  # TCA cycle
                "acetate_metabolism": "gsu00640",  # Propanoate metabolism
                "electron_transport": "gsu00190",  # Oxidative phosphorylation
            },
            Species.SHEWANELLA: {
                "central_metabolism": "son00020",  # TCA cycle
                "lactate_metabolism": "son00620",  # Pyruvate metabolism
                "electron_transport": "son00190",  # Oxidative phosphorylation
            },
        }

        return kegg_ids.get(species, {})
