"""Tests for config/biological_validation.py - targeting 98%+ coverage."""
import importlib.util
import os
import sys
from copy import deepcopy
from unittest.mock import MagicMock

import pytest

_mock_np = MagicMock()
_mock_np.isclose = lambda a, b, rtol=1e-5: abs(a - b) <= rtol * abs(b)
sys.modules.setdefault("numpy", _mock_np)

_src = os.path.join(os.path.dirname(__file__), "..", "..", "src")

# Load deps
for mod_name, mod_file in [
    ("config.electrode_config", "electrode_config.py"),
    ("config.qlearning_config", "qlearning_config.py"),
    ("config.sensor_config", "sensor_config.py"),
    ("config.parameter_validation", "parameter_validation.py"),
    ("config.biological_config", "biological_config.py"),
    ("config.substrate_config", "substrate_config.py"),
]:
    _s = importlib.util.spec_from_file_location(
        mod_name, os.path.join(_src, "config", mod_file)
    )
    _m = importlib.util.module_from_spec(_s)
    sys.modules.setdefault(mod_name, _m)
    _s.loader.exec_module(_m)

# Load biological_validation
_spec = importlib.util.spec_from_file_location(
    "config.biological_validation",
    os.path.join(_src, "config", "biological_validation.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["config.biological_validation"] = _mod
_spec.loader.exec_module(_mod)

validate_kinetic_parameters = _mod.validate_kinetic_parameters
validate_metabolic_reaction = _mod.validate_metabolic_reaction
validate_species_metabolic_config = _mod.validate_species_metabolic_config
validate_biofilm_kinetics_config = _mod.validate_biofilm_kinetics_config
validate_substrate_kinetics_config = _mod.validate_substrate_kinetics_config
validate_substrate_degradation_pathway = _mod.validate_substrate_degradation_pathway
validate_comprehensive_substrate_config = _mod.validate_comprehensive_substrate_config
validate_electrochemical_config = _mod.validate_electrochemical_config
validate_all_biological_configs = _mod.validate_all_biological_configs
ConfigValidationError = sys.modules["config.parameter_validation"].ConfigValidationError

# Import config classes for constructing test data
_bc = sys.modules["config.biological_config"]
_sc = sys.modules["config.substrate_config"]
KineticParameters = _bc.KineticParameters
MetabolicReactionConfig = _bc.MetabolicReactionConfig
SpeciesMetabolicConfig = _bc.SpeciesMetabolicConfig
BiofilmKineticsConfig = _bc.BiofilmKineticsConfig
ElectrochemicalConfig = _bc.ElectrochemicalConfig
BacterialSpecies = _bc.BacterialSpecies
SubstrateType = _bc.SubstrateType
get_geobacter_config = _bc.get_geobacter_config
get_default_biofilm_config = _bc.get_default_biofilm_config
get_default_electrochemical_config = _bc.get_default_electrochemical_config
SubstrateKineticsConfig = _sc.SubstrateKineticsConfig
SubstrateDegradationPathway = _sc.SubstrateDegradationPathway
ComprehensiveSubstrateConfig = _sc.ComprehensiveSubstrateConfig
get_acetate_config = _sc.get_acetate_config


class TestValidateKineticParameters:
    def test_valid_defaults(self):
        kp = KineticParameters(vmax=10.0, km=0.5)
        assert validate_kinetic_parameters(kp) is True

    def test_with_ki(self):
        kp = KineticParameters(vmax=10.0, km=0.5, ki=100.0)
        assert validate_kinetic_parameters(kp) is True

    def test_invalid_vmax(self):
        kp = KineticParameters(vmax=0.0, km=0.5)
        with pytest.raises(ConfigValidationError):
            validate_kinetic_parameters(kp)

    def test_invalid_km(self):
        kp = KineticParameters(vmax=10.0, km=0.0)
        with pytest.raises(ConfigValidationError):
            validate_kinetic_parameters(kp)

    def test_invalid_ki(self):
        kp = KineticParameters(vmax=10.0, km=0.5, ki=0.0)
        with pytest.raises(ConfigValidationError):
            validate_kinetic_parameters(kp)

    def test_invalid_ea(self):
        kp = KineticParameters(vmax=10.0, km=0.5, ea=5.0)
        with pytest.raises(ConfigValidationError):
            validate_kinetic_parameters(kp)

    def test_invalid_temp_ref(self):
        kp = KineticParameters(vmax=10.0, km=0.5, temperature_ref=200.0)
        with pytest.raises(ConfigValidationError):
            validate_kinetic_parameters(kp)

    def test_invalid_ph_optimal(self):
        kp = KineticParameters(vmax=10.0, km=0.5, ph_optimal=15.0)
        with pytest.raises(ConfigValidationError):
            validate_kinetic_parameters(kp)

    def test_invalid_ph_tolerance(self):
        kp = KineticParameters(vmax=10.0, km=0.5, ph_tolerance=0.0)
        with pytest.raises(ConfigValidationError):
            validate_kinetic_parameters(kp)


class TestValidateMetabolicReaction:
    def _valid_reaction(self):
        kp = KineticParameters(vmax=10.0, km=0.5)
        return MetabolicReactionConfig(
            id="R001", name="Test", equation="A -> B",
            stoichiometry={"A": -1.0, "B": 1.0},
            enzyme_name="E", kinetics=kp, delta_g0=-30.0,
        )

    def test_valid(self):
        assert validate_metabolic_reaction(self._valid_reaction()) is True

    def test_short_id(self):
        r = self._valid_reaction()
        r.id = "AB"
        with pytest.raises(ConfigValidationError, match="at least 3"):
            validate_metabolic_reaction(r)

    def test_empty_id(self):
        r = self._valid_reaction()
        r.id = ""
        with pytest.raises(ConfigValidationError):
            validate_metabolic_reaction(r)

    def test_empty_stoichiometry(self):
        r = self._valid_reaction()
        r.stoichiometry = {}
        with pytest.raises(ConfigValidationError, match="cannot be empty"):
            validate_metabolic_reaction(r)

    def test_large_stoichiometry(self):
        r = self._valid_reaction()
        r.stoichiometry = {"A": -200.0}
        with pytest.raises(ConfigValidationError, match="between -100 and 100"):
            validate_metabolic_reaction(r)

    def test_invalid_delta_g0(self):
        r = self._valid_reaction()
        r.delta_g0 = -600.0
        with pytest.raises(ConfigValidationError):
            validate_metabolic_reaction(r)

    def test_flux_bounds_inverted(self):
        r = self._valid_reaction()
        r.flux_lower_bound = 100.0
        r.flux_upper_bound = 10.0
        with pytest.raises(ConfigValidationError, match="Lower bound"):
            validate_metabolic_reaction(r)

    def test_flux_bounds_too_large(self):
        r = self._valid_reaction()
        r.flux_upper_bound = 20000.0
        with pytest.raises(ConfigValidationError):
            validate_metabolic_reaction(r)


class TestValidateSpeciesMetabolicConfig:
    def test_geobacter_valid(self):
        cfg = get_geobacter_config()
        assert validate_species_metabolic_config(cfg) is True

    def test_invalid_metabolite_concentration(self):
        cfg = get_geobacter_config()
        cfg.metabolite_concentrations["acetate"] = -1.0
        with pytest.raises(ConfigValidationError):
            validate_species_metabolic_config(cfg)

    def test_invalid_electron_transport_efficiency(self):
        cfg = get_geobacter_config()
        cfg.electron_transport_efficiency = 0.0
        with pytest.raises(ConfigValidationError):
            validate_species_metabolic_config(cfg)

    def test_invalid_cytochrome_content(self):
        cfg = get_geobacter_config()
        cfg.cytochrome_content = 0.0
        with pytest.raises(ConfigValidationError):
            validate_species_metabolic_config(cfg)

    def test_invalid_max_growth_rate(self):
        cfg = get_geobacter_config()
        cfg.max_growth_rate = 0.0
        with pytest.raises(ConfigValidationError):
            validate_species_metabolic_config(cfg)

    def test_invalid_maintenance_coefficient(self):
        cfg = get_geobacter_config()
        cfg.maintenance_coefficient = -1.0
        with pytest.raises(ConfigValidationError):
            validate_species_metabolic_config(cfg)

    def test_invalid_yield_coefficient(self):
        cfg = get_geobacter_config()
        cfg.yield_coefficient = 0.0
        with pytest.raises(ConfigValidationError):
            validate_species_metabolic_config(cfg)

    def test_invalid_attachment_rate(self):
        cfg = get_geobacter_config()
        cfg.attachment_rate = 0.0
        with pytest.raises(ConfigValidationError):
            validate_species_metabolic_config(cfg)

    def test_invalid_detachment_rate(self):
        cfg = get_geobacter_config()
        cfg.detachment_rate = 0.0
        with pytest.raises(ConfigValidationError):
            validate_species_metabolic_config(cfg)

    def test_invalid_max_biofilm_thickness(self):
        cfg = get_geobacter_config()
        cfg.max_biofilm_thickness = 0.0
        with pytest.raises(ConfigValidationError):
            validate_species_metabolic_config(cfg)

    def test_invalid_temperature_range(self):
        cfg = get_geobacter_config()
        cfg.temperature_range = (350.0, 300.0)  # inverted
        with pytest.raises(ConfigValidationError, match="Temperature range"):
            validate_species_metabolic_config(cfg)

    def test_invalid_temperature_range_out_of_bounds(self):
        cfg = get_geobacter_config()
        cfg.temperature_range = (100.0, 300.0)  # below 200
        with pytest.raises(ConfigValidationError):
            validate_species_metabolic_config(cfg)

    def test_invalid_ph_range(self):
        cfg = get_geobacter_config()
        cfg.ph_range = (8.0, 6.0)  # inverted
        with pytest.raises(ConfigValidationError, match="pH range"):
            validate_species_metabolic_config(cfg)

    def test_invalid_ph_range_out_of_bounds(self):
        cfg = get_geobacter_config()
        cfg.ph_range = (-1.0, 8.0)  # below 0
        with pytest.raises(ConfigValidationError):
            validate_species_metabolic_config(cfg)

    def test_invalid_salinity_tolerance(self):
        cfg = get_geobacter_config()
        cfg.salinity_tolerance = -1.0
        with pytest.raises(ConfigValidationError):
            validate_species_metabolic_config(cfg)


class TestValidateBiofilmKineticsConfig:
    def test_default_valid(self):
        cfg = get_default_biofilm_config()
        assert validate_biofilm_kinetics_config(cfg) is True

    def test_invalid_density(self):
        cfg = get_default_biofilm_config()
        cfg.biofilm_density = 100.0
        with pytest.raises(ConfigValidationError):
            validate_biofilm_kinetics_config(cfg)

    def test_invalid_porosity(self):
        cfg = get_default_biofilm_config()
        cfg.porosity = 0.0
        with pytest.raises(ConfigValidationError):
            validate_biofilm_kinetics_config(cfg)

    def test_invalid_tortuosity(self):
        cfg = get_default_biofilm_config()
        cfg.tortuosity = 0.5
        with pytest.raises(ConfigValidationError):
            validate_biofilm_kinetics_config(cfg)

    def test_invalid_monod_growth_rate(self):
        cfg = get_default_biofilm_config()
        cfg.monod_kinetics["max_growth_rate"] = 0.0
        with pytest.raises(ConfigValidationError):
            validate_biofilm_kinetics_config(cfg)

    def test_invalid_monod_half_saturation(self):
        cfg = get_default_biofilm_config()
        cfg.monod_kinetics["half_saturation"] = 0.0
        with pytest.raises(ConfigValidationError):
            validate_biofilm_kinetics_config(cfg)

    def test_invalid_monod_yield(self):
        cfg = get_default_biofilm_config()
        cfg.monod_kinetics["yield_coefficient"] = 0.0
        with pytest.raises(ConfigValidationError):
            validate_biofilm_kinetics_config(cfg)

    def test_invalid_monod_decay(self):
        cfg = get_default_biofilm_config()
        cfg.monod_kinetics["decay_rate"] = -1.0
        with pytest.raises(ConfigValidationError):
            validate_biofilm_kinetics_config(cfg)

    def test_invalid_nernst_potential(self):
        cfg = get_default_biofilm_config()
        cfg.nernst_monod["standard_potential"] = -2.0
        with pytest.raises(ConfigValidationError):
            validate_biofilm_kinetics_config(cfg)

    def test_invalid_nernst_rate(self):
        cfg = get_default_biofilm_config()
        cfg.nernst_monod["electron_transfer_rate"] = 0.0
        with pytest.raises(ConfigValidationError):
            validate_biofilm_kinetics_config(cfg)

    def test_invalid_nernst_conductivity(self):
        cfg = get_default_biofilm_config()
        cfg.nernst_monod["biofilm_conductivity"] = 0.0
        with pytest.raises(ConfigValidationError):
            validate_biofilm_kinetics_config(cfg)

    def test_invalid_nernst_capacitance(self):
        cfg = get_default_biofilm_config()
        cfg.nernst_monod["double_layer_capacitance"] = 0.0
        with pytest.raises(ConfigValidationError):
            validate_biofilm_kinetics_config(cfg)

    def test_invalid_boundary_layer(self):
        cfg = get_default_biofilm_config()
        cfg.mass_transfer["boundary_layer_thickness"] = 0.0
        with pytest.raises(ConfigValidationError):
            validate_biofilm_kinetics_config(cfg)

    def test_invalid_diffusivities(self):
        cfg = get_default_biofilm_config()
        cfg.mass_transfer["substrate_diffusivity"] = 0.0
        with pytest.raises(ConfigValidationError):
            validate_biofilm_kinetics_config(cfg)

    def test_invalid_critical_thickness(self):
        cfg = get_default_biofilm_config()
        cfg.structure["critical_thickness"] = 0.0
        with pytest.raises(ConfigValidationError):
            validate_biofilm_kinetics_config(cfg)

    def test_invalid_detachment_shear(self):
        cfg = get_default_biofilm_config()
        cfg.structure["detachment_shear_stress"] = 0.0
        with pytest.raises(ConfigValidationError):
            validate_biofilm_kinetics_config(cfg)

    def test_invalid_compaction(self):
        cfg = get_default_biofilm_config()
        cfg.structure["compaction_factor"] = 0.1
        with pytest.raises(ConfigValidationError):
            validate_biofilm_kinetics_config(cfg)

    def test_invalid_roughness(self):
        cfg = get_default_biofilm_config()
        cfg.structure["roughness_factor"] = 0.5
        with pytest.raises(ConfigValidationError):
            validate_biofilm_kinetics_config(cfg)


class TestValidateSubstrateKineticsConfig:
    def _valid(self):
        return SubstrateKineticsConfig(max_uptake_rate=20.0, half_saturation_constant=0.5)

    def test_valid(self):
        assert validate_substrate_kinetics_config(self._valid()) is True

    def test_with_inhibition_constant(self):
        cfg = self._valid()
        cfg.substrate_inhibition_constant = 50.0
        assert validate_substrate_kinetics_config(cfg) is True

    def test_invalid_inhibition_constant(self):
        cfg = self._valid()
        cfg.substrate_inhibition_constant = 0.0
        with pytest.raises(ConfigValidationError):
            validate_substrate_kinetics_config(cfg)

    def test_invalid_temp_coefficient(self):
        cfg = self._valid()
        cfg.temperature_coefficient = 0.5
        with pytest.raises(ConfigValidationError):
            validate_substrate_kinetics_config(cfg)

    def test_invalid_ph_optimum(self):
        cfg = self._valid()
        cfg.ph_optimum = 15.0
        with pytest.raises(ConfigValidationError):
            validate_substrate_kinetics_config(cfg)

    def test_invalid_ph_tolerance_range(self):
        cfg = self._valid()
        cfg.ph_tolerance_range = (8.0, 6.0)
        with pytest.raises(ConfigValidationError, match="pH tolerance range"):
            validate_substrate_kinetics_config(cfg)

    def test_invalid_activation_energy(self):
        cfg = self._valid()
        cfg.activation_energy = 5.0
        with pytest.raises(ConfigValidationError):
            validate_substrate_kinetics_config(cfg)

    def test_invalid_enthalpy(self):
        cfg = self._valid()
        cfg.enthalpy_change = 200.0
        with pytest.raises(ConfigValidationError):
            validate_substrate_kinetics_config(cfg)

    def test_invalid_entropy(self):
        cfg = self._valid()
        cfg.entropy_change = 5.0
        with pytest.raises(ConfigValidationError):
            validate_substrate_kinetics_config(cfg)


class TestValidateSubstrateDegradationPathway:
    def _valid(self):
        return SubstrateDegradationPathway(
            pathway_name="Test", substrate=SubstrateType.ACETATE,
            intermediates=["X"], final_products=["CO2"],
        )

    def test_valid(self):
        assert validate_substrate_degradation_pathway(self._valid()) is True

    def test_invalid_stoichiometry(self):
        p = self._valid()
        p.substrate_stoichiometry = 0.0
        with pytest.raises(ConfigValidationError):
            validate_substrate_degradation_pathway(p)

    def test_invalid_electron_yield(self):
        p = self._valid()
        p.electron_yield = 0.0
        with pytest.raises(ConfigValidationError):
            validate_substrate_degradation_pathway(p)

    def test_invalid_biomass_yield(self):
        p = self._valid()
        p.biomass_yield = 0.0
        with pytest.raises(ConfigValidationError):
            validate_substrate_degradation_pathway(p)

    def test_invalid_atp_yield(self):
        p = self._valid()
        p.atp_yield = -1.0
        with pytest.raises(ConfigValidationError):
            validate_substrate_degradation_pathway(p)

    def test_invalid_nadh_yield(self):
        p = self._valid()
        p.nadh_yield = -1.0
        with pytest.raises(ConfigValidationError):
            validate_substrate_degradation_pathway(p)

    def test_invalid_co2_yield(self):
        p = self._valid()
        p.co2_yield = -1.0
        with pytest.raises(ConfigValidationError):
            validate_substrate_degradation_pathway(p)

    def test_invalid_optimal_temperature(self):
        p = self._valid()
        p.optimal_conditions["temperature"] = 100.0
        with pytest.raises(ConfigValidationError):
            validate_substrate_degradation_pathway(p)

    def test_invalid_optimal_ph(self):
        p = self._valid()
        p.optimal_conditions["ph"] = 15.0
        with pytest.raises(ConfigValidationError):
            validate_substrate_degradation_pathway(p)

    def test_invalid_optimal_ionic_strength(self):
        p = self._valid()
        p.optimal_conditions["ionic_strength"] = 0.0
        with pytest.raises(ConfigValidationError):
            validate_substrate_degradation_pathway(p)

    def test_invalid_optimal_redox(self):
        p = self._valid()
        p.optimal_conditions["redox_potential"] = -2.0
        with pytest.raises(ConfigValidationError):
            validate_substrate_degradation_pathway(p)


class TestValidateComprehensiveSubstrateConfig:
    def test_acetate_valid(self):
        cfg = get_acetate_config()
        assert validate_comprehensive_substrate_config(cfg) is True

    def test_invalid_molecular_weight(self):
        cfg = get_acetate_config()
        cfg.molecular_weight = 5.0
        with pytest.raises(ConfigValidationError):
            validate_comprehensive_substrate_config(cfg)

    def test_empty_formula(self):
        cfg = get_acetate_config()
        cfg.chemical_formula = ""
        with pytest.raises(ConfigValidationError, match="cannot be empty"):
            validate_comprehensive_substrate_config(cfg)

    def test_invalid_density(self):
        cfg = get_acetate_config()
        cfg.density = 0.1
        with pytest.raises(ConfigValidationError):
            validate_comprehensive_substrate_config(cfg)

    def test_invalid_melting_point(self):
        cfg = get_acetate_config()
        cfg.melting_point = 100.0
        with pytest.raises(ConfigValidationError):
            validate_comprehensive_substrate_config(cfg)

    def test_boiling_below_melting(self):
        cfg = get_acetate_config()
        cfg.boiling_point = cfg.melting_point - 10.0
        with pytest.raises(ConfigValidationError, match="higher than melting"):
            validate_comprehensive_substrate_config(cfg)

    def test_invalid_solubility(self):
        cfg = get_acetate_config()
        cfg.water_solubility = 0.0
        with pytest.raises(ConfigValidationError):
            validate_comprehensive_substrate_config(cfg)

    def test_invalid_log_kow(self):
        cfg = get_acetate_config()
        cfg.log_kow = -10.0
        with pytest.raises(ConfigValidationError):
            validate_comprehensive_substrate_config(cfg)

    def test_invalid_diffusion_water(self):
        cfg = get_acetate_config()
        cfg.diffusion_coefficient_water = 0.0
        with pytest.raises(ConfigValidationError):
            validate_comprehensive_substrate_config(cfg)

    def test_invalid_diffusion_biofilm(self):
        cfg = get_acetate_config()
        cfg.diffusion_coefficient_biofilm = 0.0
        with pytest.raises(ConfigValidationError):
            validate_comprehensive_substrate_config(cfg)

    def test_with_toxicity_threshold(self):
        cfg = get_acetate_config()
        cfg.toxicity_threshold = 100.0
        assert validate_comprehensive_substrate_config(cfg) is True

    def test_invalid_toxicity_threshold(self):
        cfg = get_acetate_config()
        cfg.toxicity_threshold = 0.0
        with pytest.raises(ConfigValidationError):
            validate_comprehensive_substrate_config(cfg)

    def test_invalid_biodegradability(self):
        cfg = get_acetate_config()
        cfg.biodegradability = -1.0
        with pytest.raises(ConfigValidationError):
            validate_comprehensive_substrate_config(cfg)

    def test_invalid_half_life_aerobic(self):
        cfg = get_acetate_config()
        cfg.half_life_aerobic = 0.0
        with pytest.raises(ConfigValidationError):
            validate_comprehensive_substrate_config(cfg)

    def test_invalid_half_life_anaerobic(self):
        cfg = get_acetate_config()
        cfg.half_life_anaerobic = 0.0
        with pytest.raises(ConfigValidationError):
            validate_comprehensive_substrate_config(cfg)

    def test_invalid_quantification_limit(self):
        cfg = get_acetate_config()
        cfg.quantification_limit = 0.0
        with pytest.raises(ConfigValidationError):
            validate_comprehensive_substrate_config(cfg)


class TestValidateElectrochemicalConfig:
    def test_default_valid(self):
        cfg = get_default_electrochemical_config()
        assert validate_electrochemical_config(cfg) is True

    def test_invalid_faraday(self):
        cfg = get_default_electrochemical_config()
        cfg.faraday_constant = 90000.0
        with pytest.raises(ConfigValidationError, match="Faraday"):
            validate_electrochemical_config(cfg)

    def test_invalid_gas_constant(self):
        cfg = get_default_electrochemical_config()
        cfg.gas_constant = 9.0
        with pytest.raises(ConfigValidationError, match="Gas constant"):
            validate_electrochemical_config(cfg)

    def test_invalid_standard_potential(self):
        cfg = get_default_electrochemical_config()
        cfg.standard_potentials["acetate_co2"] = 3.0
        with pytest.raises(ConfigValidationError):
            validate_electrochemical_config(cfg)

    def test_invalid_electrode_surface_area(self):
        cfg = get_default_electrochemical_config()
        cfg.electrode_properties["carbon_cloth"]["surface_area"] = 0.0
        with pytest.raises(ConfigValidationError):
            validate_electrochemical_config(cfg)

    def test_invalid_electrode_conductivity(self):
        cfg = get_default_electrochemical_config()
        cfg.electrode_properties["carbon_cloth"]["conductivity"] = 0.0
        with pytest.raises(ConfigValidationError):
            validate_electrochemical_config(cfg)

    def test_invalid_electrode_porosity(self):
        cfg = get_default_electrochemical_config()
        cfg.electrode_properties["carbon_cloth"]["porosity"] = -1.0
        with pytest.raises(ConfigValidationError):
            validate_electrochemical_config(cfg)

    def test_invalid_membrane_thickness(self):
        cfg = get_default_electrochemical_config()
        cfg.membrane_properties["nafion_117"]["thickness"] = 0.0
        with pytest.raises(ConfigValidationError):
            validate_electrochemical_config(cfg)

    def test_invalid_membrane_conductivity(self):
        cfg = get_default_electrochemical_config()
        cfg.membrane_properties["nafion_117"]["conductivity"] = 0.0
        with pytest.raises(ConfigValidationError):
            validate_electrochemical_config(cfg)


class TestValidateAllBiologicalConfigs:
    def test_all_valid(self):
        species = get_geobacter_config()
        biofilm = get_default_biofilm_config()
        echem = get_default_electrochemical_config()
        acetate = get_acetate_config()
        substrates = {SubstrateType.ACETATE: acetate}
        assert validate_all_biological_configs(species, biofilm, substrates, echem) is True

    def test_missing_species_kinetics(self):
        species = get_geobacter_config()
        biofilm = get_default_biofilm_config()
        echem = get_default_electrochemical_config()
        acetate = get_acetate_config()
        # Remove geobacter kinetics from acetate
        acetate.species_kinetics = {}
        substrates = {SubstrateType.ACETATE: acetate}
        with pytest.raises(ConfigValidationError, match="No kinetics"):
            validate_all_biological_configs(species, biofilm, substrates, echem)

    def test_growth_rate_mismatch(self):
        species = get_geobacter_config()
        biofilm = get_default_biofilm_config()
        echem = get_default_electrochemical_config()
        acetate = get_acetate_config()
        substrates = {SubstrateType.ACETATE: acetate}
        # Make growth rates very different
        biofilm.monod_kinetics["max_growth_rate"] = 1.5
        species.max_growth_rate = 0.25  # > 0.5 difference
        with pytest.raises(ConfigValidationError, match="growth_rate_consistency"):
            validate_all_biological_configs(species, biofilm, substrates, echem)
