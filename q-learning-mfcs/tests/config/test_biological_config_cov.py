"""Tests for config/biological_config.py - targeting 98%+ coverage."""
import importlib.util
import os
import sys

import pytest

_src = os.path.join(os.path.dirname(__file__), "..", "..", "src")
_spec = importlib.util.spec_from_file_location(
    "config.biological_config",
    os.path.join(_src, "config", "biological_config.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["config.biological_config"] = _mod
_spec.loader.exec_module(_mod)

BacterialSpecies = _mod.BacterialSpecies
SubstrateType = _mod.SubstrateType
LiteratureReference = _mod.LiteratureReference
KineticParameters = _mod.KineticParameters
MetabolicReactionConfig = _mod.MetabolicReactionConfig
SpeciesMetabolicConfig = _mod.SpeciesMetabolicConfig
SubstrateProperties = _mod.SubstrateProperties
BiofilmKineticsConfig = _mod.BiofilmKineticsConfig
ElectrochemicalConfig = _mod.ElectrochemicalConfig
LITERATURE_REFERENCES = _mod.LITERATURE_REFERENCES
get_geobacter_config = _mod.get_geobacter_config
get_shewanella_config = _mod.get_shewanella_config
get_default_substrate_properties = _mod.get_default_substrate_properties
get_default_biofilm_config = _mod.get_default_biofilm_config
get_default_electrochemical_config = _mod.get_default_electrochemical_config


class TestEnums:
    def test_bacterial_species_values(self):
        assert BacterialSpecies.GEOBACTER_SULFURREDUCENS.value == "geobacter_sulfurreducens"
        assert BacterialSpecies.SHEWANELLA_ONEIDENSIS.value == "shewanella_oneidensis"
        assert BacterialSpecies.MIXED_CULTURE.value == "mixed_culture"

    def test_substrate_type_values(self):
        assert SubstrateType.ACETATE.value == "acetate"
        assert SubstrateType.LACTATE.value == "lactate"
        assert SubstrateType.PYRUVATE.value == "pyruvate"
        assert SubstrateType.GLUCOSE.value == "glucose"


class TestLiteratureReference:
    def test_required_fields(self):
        ref = LiteratureReference(
            authors="Smith, J.",
            title="A Study",
            journal="Nature",
            year=2020,
        )
        assert ref.authors == "Smith, J."
        assert ref.year == 2020

    def test_optional_fields(self):
        ref = LiteratureReference(
            authors="Smith, J.",
            title="A Study",
            journal="Nature",
            year=2020,
            doi="10.1234/test",
            pmid="12345",
            pages="1-10",
        )
        assert ref.doi == "10.1234/test"
        assert ref.pmid == "12345"
        assert ref.pages == "1-10"

    def test_optional_defaults(self):
        ref = LiteratureReference(authors="A", title="B", journal="C", year=2000)
        assert ref.doi is None
        assert ref.pmid is None
        assert ref.pages is None

    def test_str(self):
        ref = LiteratureReference(
            authors="Smith, J.",
            title="A Study",
            journal="Nature",
            year=2020,
        )
        s = str(ref)
        assert "Smith, J." in s
        assert "2020" in s
        assert "A Study" in s
        assert "Nature" in s


class TestKineticParameters:
    def test_defaults(self):
        kp = KineticParameters(vmax=10.0, km=0.5)
        assert kp.ki is None
        assert kp.ea == 50.0
        assert kp.temperature_ref == 303.0
        assert kp.ph_optimal == 7.0
        assert kp.ph_tolerance == 1.0
        assert kp.reference is None

    def test_custom(self):
        ref = LiteratureReference(authors="A", title="B", journal="C", year=2000)
        kp = KineticParameters(vmax=15.0, km=0.3, ki=100.0, reference=ref)
        assert kp.vmax == 15.0
        assert kp.ki == 100.0
        assert kp.reference is ref


class TestMetabolicReactionConfig:
    def test_required_and_defaults(self):
        kp = KineticParameters(vmax=10.0, km=0.5)
        rxn = MetabolicReactionConfig(
            id="R001",
            name="Test Reaction",
            equation="A -> B",
            stoichiometry={"A": -1.0, "B": 1.0},
            enzyme_name="TestEnzyme",
            kinetics=kp,
            delta_g0=-30.0,
        )
        assert rxn.id == "R001"
        assert rxn.ec_number is None
        assert rxn.kegg_id is None
        assert rxn.reversible is True
        assert rxn.flux_lower_bound == -1000.0
        assert rxn.flux_upper_bound == 1000.0


class TestSpeciesMetabolicConfig:
    def test_defaults(self):
        cfg = SpeciesMetabolicConfig(species=BacterialSpecies.GEOBACTER_SULFURREDUCENS)
        assert cfg.reactions == []
        assert cfg.metabolite_concentrations == {}
        assert cfg.electron_transport_efficiency == 0.85
        assert cfg.max_growth_rate == 0.3
        assert cfg.temperature_range == (273.0, 333.0)
        assert cfg.ph_range == (5.0, 9.0)
        assert cfg.references == []


class TestSubstrateProperties:
    def test_required_fields(self):
        sp = SubstrateProperties(
            substrate=SubstrateType.ACETATE,
            molecular_weight=60.05,
            formula="C2H4O2",
            delta_g_formation=-369.3,
            delta_h_formation=-484.5,
            solubility=1000.0,
            diffusion_coefficient=1.29e-5,
            electron_equivalents=8.0,
            theoretical_cod=1.07,
        )
        assert sp.substrate == SubstrateType.ACETATE
        assert sp.uptake_kinetics == {}
        assert sp.reference is None


class TestBiofilmKineticsConfig:
    def test_defaults(self):
        cfg = BiofilmKineticsConfig()
        assert cfg.biofilm_density == 1050.0
        assert cfg.porosity == 0.8
        assert cfg.tortuosity == 1.5
        assert "max_growth_rate" in cfg.monod_kinetics
        assert "standard_potential" in cfg.nernst_monod
        assert "boundary_layer_thickness" in cfg.mass_transfer
        assert "critical_thickness" in cfg.structure
        assert cfg.references == []


class TestElectrochemicalConfig:
    def test_defaults(self):
        cfg = ElectrochemicalConfig()
        assert cfg.faraday_constant == 96485.0
        assert cfg.gas_constant == 8.314
        assert "acetate_co2" in cfg.standard_potentials
        assert "carbon_cloth" in cfg.electrode_properties
        assert "nafion_117" in cfg.membrane_properties
        assert cfg.references == []


class TestLiteratureReferencesDB:
    def test_all_references_present(self):
        assert "lovley_2003" in LITERATURE_REFERENCES
        assert "bond_2002" in LITERATURE_REFERENCES
        assert "reguera_2005" in LITERATURE_REFERENCES
        assert "marsili_2008" in LITERATURE_REFERENCES
        assert "torres_2010" in LITERATURE_REFERENCES
        assert "marcus_2007" in LITERATURE_REFERENCES

    def test_reference_types(self):
        for ref in LITERATURE_REFERENCES.values():
            assert isinstance(ref, LiteratureReference)


class TestGetGeobacterConfig:
    def test_returns_correct_species(self):
        cfg = get_geobacter_config()
        assert cfg.species == BacterialSpecies.GEOBACTER_SULFURREDUCENS

    def test_has_reactions(self):
        cfg = get_geobacter_config()
        assert len(cfg.reactions) >= 1
        assert cfg.reactions[0].id == "GSU_R001"

    def test_has_metabolites(self):
        cfg = get_geobacter_config()
        assert "acetate" in cfg.metabolite_concentrations

    def test_has_references(self):
        cfg = get_geobacter_config()
        assert len(cfg.references) >= 1

    def test_specific_parameters(self):
        cfg = get_geobacter_config()
        assert cfg.electron_transport_efficiency == 0.85
        assert cfg.max_growth_rate == 0.25
        assert cfg.attachment_rate == 0.15
        assert cfg.max_biofilm_thickness == 120.0


class TestGetShewanellaConfig:
    def test_returns_correct_species(self):
        cfg = get_shewanella_config()
        assert cfg.species == BacterialSpecies.SHEWANELLA_ONEIDENSIS

    def test_has_reactions(self):
        cfg = get_shewanella_config()
        assert len(cfg.reactions) >= 1
        assert cfg.reactions[0].id == "MR1_R001"

    def test_has_metabolites(self):
        cfg = get_shewanella_config()
        assert "lactate" in cfg.metabolite_concentrations

    def test_has_references(self):
        cfg = get_shewanella_config()
        assert len(cfg.references) >= 1

    def test_specific_parameters(self):
        cfg = get_shewanella_config()
        assert cfg.electron_transport_efficiency == 0.75
        assert cfg.max_growth_rate == 0.35
        assert cfg.max_biofilm_thickness == 80.0


class TestGetDefaultSubstrateProperties:
    def test_returns_dict(self):
        props = get_default_substrate_properties()
        assert isinstance(props, dict)

    def test_has_acetate(self):
        props = get_default_substrate_properties()
        assert SubstrateType.ACETATE in props
        assert props[SubstrateType.ACETATE].molecular_weight == 60.05

    def test_has_lactate(self):
        props = get_default_substrate_properties()
        assert SubstrateType.LACTATE in props
        assert props[SubstrateType.LACTATE].molecular_weight == 90.08

    def test_uptake_kinetics(self):
        props = get_default_substrate_properties()
        acetate = props[SubstrateType.ACETATE]
        assert BacterialSpecies.GEOBACTER_SULFURREDUCENS in acetate.uptake_kinetics
        assert BacterialSpecies.SHEWANELLA_ONEIDENSIS in acetate.uptake_kinetics

        lactate = props[SubstrateType.LACTATE]
        assert BacterialSpecies.GEOBACTER_SULFURREDUCENS in lactate.uptake_kinetics
        assert BacterialSpecies.SHEWANELLA_ONEIDENSIS in lactate.uptake_kinetics


class TestGetDefaultBiofilmConfig:
    def test_returns_correct_type(self):
        cfg = get_default_biofilm_config()
        assert isinstance(cfg, BiofilmKineticsConfig)

    def test_has_references(self):
        cfg = get_default_biofilm_config()
        assert len(cfg.references) >= 1

    def test_default_values(self):
        cfg = get_default_biofilm_config()
        assert cfg.biofilm_density == 1050.0
        assert cfg.porosity == 0.8
        assert cfg.tortuosity == 1.5


class TestGetDefaultElectrochemicalConfig:
    def test_returns_correct_type(self):
        cfg = get_default_electrochemical_config()
        assert isinstance(cfg, ElectrochemicalConfig)

    def test_has_references(self):
        cfg = get_default_electrochemical_config()
        assert len(cfg.references) >= 1
