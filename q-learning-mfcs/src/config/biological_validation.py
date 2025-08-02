"""
Validation functions for biological configuration parameters.
Ensures biological plausibility and parameter consistency.
"""


from .biological_config import (
    BiofilmKineticsConfig,
    ElectrochemicalConfig,
    KineticParameters,
    MetabolicReactionConfig,
    SpeciesMetabolicConfig,
)
from .parameter_validation import ConfigValidationError, validate_range
from .substrate_config import (
    ComprehensiveSubstrateConfig,
    SubstrateDegradationPathway,
    SubstrateKineticsConfig,
    SubstrateType,
)


def validate_kinetic_parameters(params: KineticParameters, parameter_name: str = "kinetic_parameters") -> bool:
    """
    Validate kinetic parameters for biological plausibility.

    Args:
        params: KineticParameters to validate
        parameter_name: Name for error messages

    Returns:
        True if validation passes

    Raises:
        ConfigValidationError: If any parameter is biologically implausible
    """
    # Validate Vmax (maximum reaction rate)
    validate_range(params.vmax, 0.001, 1000.0, f"{parameter_name}.vmax",
                  "Maximum reaction rate must be between 0.001-1000 mmol/gDW/h")

    # Validate Km (Michaelis constant)
    validate_range(params.km, 0.0001, 1000.0, f"{parameter_name}.km",
                  "Michaelis constant must be between 0.0001-1000 mmol/L")

    # Validate Ki (inhibition constant) if present
    if params.ki is not None:
        validate_range(params.ki, 0.001, 10000.0, f"{parameter_name}.ki",
                      "Inhibition constant must be between 0.001-10000 mmol/L")

    # Validate activation energy
    validate_range(params.ea, 10.0, 200.0, f"{parameter_name}.ea",
                  "Activation energy must be between 10-200 kJ/mol for biological reactions")

    # Validate reference temperature
    validate_range(params.temperature_ref, 273.0, 373.0, f"{parameter_name}.temperature_ref",
                  "Reference temperature must be between 273-373 K")

    # Validate pH parameters
    validate_range(params.ph_optimal, 3.0, 11.0, f"{parameter_name}.ph_optimal",
                  "Optimal pH must be between 3-11")
    validate_range(params.ph_tolerance, 0.1, 5.0, f"{parameter_name}.ph_tolerance",
                  "pH tolerance must be between 0.1-5 units")

    return True


def validate_metabolic_reaction(reaction: MetabolicReactionConfig,
                              reaction_name: str = "metabolic_reaction") -> bool:
    """
    Validate metabolic reaction configuration.

    Args:
        reaction: MetabolicReactionConfig to validate
        reaction_name: Name for error messages

    Returns:
        True if validation passes

    Raises:
        ConfigValidationError: If reaction is invalid
    """
    # Validate reaction ID format
    if not reaction.id or len(reaction.id) < 3:
        raise ConfigValidationError(f"{reaction_name}.id", reaction.id,
                                  "Reaction ID must be at least 3 characters long")

    # Validate stoichiometry balance (basic check)
    if not reaction.stoichiometry:
        raise ConfigValidationError(f"{reaction_name}.stoichiometry", reaction.stoichiometry,
                                  "Stoichiometry dictionary cannot be empty")

    # Check for reasonable stoichiometric coefficients
    for metabolite, coeff in reaction.stoichiometry.items():
        if abs(coeff) > 100:
            raise ConfigValidationError(f"{reaction_name}.stoichiometry[{metabolite}]", coeff,
                                      "Stoichiometric coefficients should be between -100 and 100")

    # Validate kinetic parameters
    validate_kinetic_parameters(reaction.kinetics, f"{reaction_name}.kinetics")

    # Validate thermodynamic parameters
    validate_range(reaction.delta_g0, -500.0, 500.0, f"{reaction_name}.delta_g0",
                  "Standard Gibbs free energy must be between -500 to 500 kJ/mol")

    # Validate flux bounds
    if reaction.flux_lower_bound >= reaction.flux_upper_bound:
        raise ConfigValidationError(f"{reaction_name}.flux_bounds",
                                  (reaction.flux_lower_bound, reaction.flux_upper_bound),
                                  "Lower bound must be less than upper bound")

    validate_range(abs(reaction.flux_lower_bound), 0, 10000.0, f"{reaction_name}.flux_lower_bound",
                  "Flux bounds must be reasonable (< 10000 mmol/gDW/h)")
    validate_range(reaction.flux_upper_bound, 0, 10000.0, f"{reaction_name}.flux_upper_bound",
                  "Flux bounds must be reasonable (< 10000 mmol/gDW/h)")

    return True


def validate_species_metabolic_config(config: SpeciesMetabolicConfig,
                                    config_name: str = "species_config") -> bool:
    """
    Validate species-specific metabolic configuration.

    Args:
        config: SpeciesMetabolicConfig to validate
        config_name: Name for error messages

    Returns:
        True if validation passes

    Raises:
        ConfigValidationError: If configuration is invalid
    """
    # Validate reactions
    for i, reaction in enumerate(config.reactions):
        validate_metabolic_reaction(reaction, f"{config_name}.reactions[{i}]")

    # Validate metabolite concentrations
    for metabolite, conc in config.metabolite_concentrations.items():
        validate_range(conc, 0.0, 1000.0, f"{config_name}.metabolite_concentrations[{metabolite}]",
                      "Metabolite concentrations must be between 0-1000 mmol/L")

    # Validate electron transport efficiency
    validate_range(config.electron_transport_efficiency, 0.1, 1.0,
                  f"{config_name}.electron_transport_efficiency",
                  "Electron transport efficiency must be between 0.1-1.0")

    # Validate cytochrome content
    validate_range(config.cytochrome_content, 0.001, 1.0, f"{config_name}.cytochrome_content",
                  "Cytochrome content must be between 0.001-1.0 mmol/gDW")

    # Validate growth parameters
    validate_range(config.max_growth_rate, 0.001, 2.0, f"{config_name}.max_growth_rate",
                  "Maximum growth rate must be between 0.001-2.0 h⁻¹")

    validate_range(config.maintenance_coefficient, 0.0, 0.5, f"{config_name}.maintenance_coefficient",
                  "Maintenance coefficient must be between 0-0.5 h⁻¹")

    validate_range(config.yield_coefficient, 0.01, 1.0, f"{config_name}.yield_coefficient",
                  "Yield coefficient must be between 0.01-1.0 gDW/mmol")

    # Validate biofilm parameters
    validate_range(config.attachment_rate, 0.001, 1.0, f"{config_name}.attachment_rate",
                  "Attachment rate must be between 0.001-1.0 h⁻¹")

    validate_range(config.detachment_rate, 0.0001, 0.5, f"{config_name}.detachment_rate",
                  "Detachment rate must be between 0.0001-0.5 h⁻¹")

    validate_range(config.max_biofilm_thickness, 1.0, 1000.0, f"{config_name}.max_biofilm_thickness",
                  "Maximum biofilm thickness must be between 1-1000 μm")

    # Validate environmental ranges
    temp_min, temp_max = config.temperature_range
    if temp_min >= temp_max or temp_min < 200 or temp_max > 400:
        raise ConfigValidationError(f"{config_name}.temperature_range", config.temperature_range,
                                  "Temperature range must be reasonable (200-400 K) with min < max")

    ph_min, ph_max = config.ph_range
    if ph_min >= ph_max or ph_min < 0 or ph_max > 14:
        raise ConfigValidationError(f"{config_name}.ph_range", config.ph_range,
                                  "pH range must be between 0-14 with min < max")

    validate_range(config.salinity_tolerance, 0.0, 5.0, f"{config_name}.salinity_tolerance",
                  "Salinity tolerance must be between 0-5 M NaCl")

    return True


def validate_biofilm_kinetics_config(config: BiofilmKineticsConfig,
                                    config_name: str = "biofilm_config") -> bool:
    """
    Validate biofilm kinetics configuration.

    Args:
        config: BiofilmKineticsConfig to validate
        config_name: Name for error messages

    Returns:
        True if validation passes

    Raises:
        ConfigValidationError: If configuration is invalid
    """
    # Validate physical properties
    validate_range(config.biofilm_density, 800.0, 1500.0, f"{config_name}.biofilm_density",
                  "Biofilm density must be between 800-1500 kg/m³")

    validate_range(config.porosity, 0.1, 0.95, f"{config_name}.porosity",
                  "Biofilm porosity must be between 0.1-0.95")

    validate_range(config.tortuosity, 1.0, 10.0, f"{config_name}.tortuosity",
                  "Biofilm tortuosity must be between 1.0-10.0")

    # Validate Monod kinetics
    monod = config.monod_kinetics
    validate_range(monod['max_growth_rate'], 0.001, 2.0, f"{config_name}.monod_kinetics.max_growth_rate",
                  "Maximum growth rate must be between 0.001-2.0 h⁻¹")

    validate_range(monod['half_saturation'], 0.001, 100.0, f"{config_name}.monod_kinetics.half_saturation",
                  "Half-saturation constant must be between 0.001-100 mmol/L")

    validate_range(monod['yield_coefficient'], 0.01, 1.0, f"{config_name}.monod_kinetics.yield_coefficient",
                  "Yield coefficient must be between 0.01-1.0 gDW/mmol")

    validate_range(monod['decay_rate'], 0.0, 0.1, f"{config_name}.monod_kinetics.decay_rate",
                  "Decay rate must be between 0-0.1 h⁻¹")

    # Validate Nernst-Monod parameters
    nernst = config.nernst_monod
    validate_range(nernst['standard_potential'], -1.0, 1.0, f"{config_name}.nernst_monod.standard_potential",
                  "Standard potential must be between -1.0 to 1.0 V vs SHE")

    validate_range(nernst['electron_transfer_rate'], 0.001, 1000.0,
                  f"{config_name}.nernst_monod.electron_transfer_rate",
                  "Electron transfer rate must be between 0.001-1000 s⁻¹")

    validate_range(nernst['biofilm_conductivity'], 1e-6, 1.0,
                  f"{config_name}.nernst_monod.biofilm_conductivity",
                  "Biofilm conductivity must be between 1e-6 to 1.0 S/m")

    validate_range(nernst['double_layer_capacitance'], 1e-8, 1e-3,
                  f"{config_name}.nernst_monod.double_layer_capacitance",
                  "Double layer capacitance must be between 1e-8 to 1e-3 F/cm²")

    # Validate mass transfer coefficients
    mass_transfer = config.mass_transfer
    validate_range(mass_transfer['boundary_layer_thickness'], 0.001, 10.0,
                  f"{config_name}.mass_transfer.boundary_layer_thickness",
                  "Boundary layer thickness must be between 0.001-10 mm")

    for param_name in ['substrate_diffusivity', 'oxygen_diffusivity', 'product_diffusivity']:
        validate_range(mass_transfer[param_name], 1e-12, 1e-6,
                      f"{config_name}.mass_transfer.{param_name}",
                      f"{param_name} must be between 1e-12 to 1e-6 m²/s")

    # Validate structure parameters
    structure = config.structure
    validate_range(structure['critical_thickness'], 1.0, 500.0,
                  f"{config_name}.structure.critical_thickness",
                  "Critical thickness must be between 1-500 μm")

    validate_range(structure['detachment_shear_stress'], 0.001, 10.0,
                  f"{config_name}.structure.detachment_shear_stress",
                  "Detachment shear stress must be between 0.001-10 Pa")

    validate_range(structure['compaction_factor'], 0.5, 1.0,
                  f"{config_name}.structure.compaction_factor",
                  "Compaction factor must be between 0.5-1.0")

    validate_range(structure['roughness_factor'], 1.0, 10.0,
                  f"{config_name}.structure.roughness_factor",
                  "Roughness factor must be between 1.0-10.0")

    return True


def validate_substrate_kinetics_config(config: SubstrateKineticsConfig,
                                     config_name: str = "substrate_kinetics") -> bool:
    """
    Validate substrate kinetics configuration.

    Args:
        config: SubstrateKineticsConfig to validate
        config_name: Name for error messages

    Returns:
        True if validation passes
    """
    # Validate basic kinetic parameters
    validate_range(config.max_uptake_rate, 0.1, 200.0, f"{config_name}.max_uptake_rate",
                  "Maximum uptake rate must be between 0.1-200 mmol/gDW/h")

    validate_range(config.half_saturation_constant, 0.001, 100.0,
                  f"{config_name}.half_saturation_constant",
                  "Half-saturation constant must be between 0.001-100 mmol/L")

    # Validate inhibition constants if present
    if config.substrate_inhibition_constant is not None:
        validate_range(config.substrate_inhibition_constant, 0.1, 1000.0,
                      f"{config_name}.substrate_inhibition_constant",
                      "Substrate inhibition constant must be between 0.1-1000 mmol/L")

    # Validate environmental parameters
    validate_range(config.temperature_coefficient, 1.01, 2.0,
                  f"{config_name}.temperature_coefficient",
                  "Temperature coefficient must be between 1.01-2.0")

    validate_range(config.ph_optimum, 3.0, 11.0, f"{config_name}.ph_optimum",
                  "pH optimum must be between 3-11")

    ph_min, ph_max = config.ph_tolerance_range
    if ph_min >= ph_max or ph_min < 0 or ph_max > 14:
        raise ConfigValidationError(f"{config_name}.ph_tolerance_range",
                                  config.ph_tolerance_range,
                                  "pH tolerance range must be valid (0-14) with min < max")

    # Validate thermodynamic parameters
    validate_range(config.activation_energy, 10.0, 200.0, f"{config_name}.activation_energy",
                  "Activation energy must be between 10-200 kJ/mol")

    validate_range(config.enthalpy_change, -300.0, 100.0, f"{config_name}.enthalpy_change",
                  "Enthalpy change must be between -300 to 100 kJ/mol")

    validate_range(config.entropy_change, -1.0, 1.0, f"{config_name}.entropy_change",
                  "Entropy change must be between -1.0 to 1.0 kJ/mol/K")

    return True


def validate_substrate_degradation_pathway(pathway: SubstrateDegradationPathway,
                                         pathway_name: str = "degradation_pathway") -> bool:
    """
    Validate substrate degradation pathway.

    Args:
        pathway: SubstrateDegradationPathway to validate
        pathway_name: Name for error messages

    Returns:
        True if validation passes
    """
    # Validate stoichiometry
    validate_range(pathway.substrate_stoichiometry, 0.1, 10.0,
                  f"{pathway_name}.substrate_stoichiometry",
                  "Substrate stoichiometry must be between 0.1-10")

    validate_range(pathway.electron_yield, 1.0, 50.0, f"{pathway_name}.electron_yield",
                  "Electron yield must be between 1-50 electrons per substrate")

    validate_range(pathway.biomass_yield, 0.01, 2.0, f"{pathway_name}.biomass_yield",
                  "Biomass yield must be between 0.01-2.0 gDW/mmol")

    # Validate energy yields
    validate_range(pathway.atp_yield, 0.0, 10.0, f"{pathway_name}.atp_yield",
                  "ATP yield must be between 0-10 moles per substrate")

    validate_range(pathway.nadh_yield, 0.0, 20.0, f"{pathway_name}.nadh_yield",
                  "NADH yield must be between 0-20 moles per substrate")

    validate_range(pathway.co2_yield, 0.0, 20.0, f"{pathway_name}.co2_yield",
                  "CO2 yield must be between 0-20 moles per substrate")

    # Validate optimal conditions
    optimal = pathway.optimal_conditions
    validate_range(optimal['temperature'], 273.0, 373.0,
                  f"{pathway_name}.optimal_conditions.temperature",
                  "Optimal temperature must be between 273-373 K")

    validate_range(optimal['ph'], 3.0, 11.0, f"{pathway_name}.optimal_conditions.ph",
                  "Optimal pH must be between 3-11")

    validate_range(optimal['ionic_strength'], 0.001, 2.0,
                  f"{pathway_name}.optimal_conditions.ionic_strength",
                  "Optimal ionic strength must be between 0.001-2.0 M")

    validate_range(optimal['redox_potential'], -1.0, 1.0,
                  f"{pathway_name}.optimal_conditions.redox_potential",
                  "Optimal redox potential must be between -1.0 to 1.0 V vs SHE")

    return True


def validate_comprehensive_substrate_config(config: ComprehensiveSubstrateConfig,
                                          config_name: str = "substrate_config") -> bool:
    """
    Validate comprehensive substrate configuration.

    Args:
        config: ComprehensiveSubstrateConfig to validate
        config_name: Name for error messages

    Returns:
        True if validation passes
    """
    # Validate chemical properties
    validate_range(config.molecular_weight, 10.0, 1000.0, f"{config_name}.molecular_weight",
                  "Molecular weight must be between 10-1000 g/mol")

    if not config.chemical_formula:
        raise ConfigValidationError(f"{config_name}.chemical_formula", config.chemical_formula,
                                  "Chemical formula cannot be empty")

    # Validate physical properties
    validate_range(config.density, 0.5, 5.0, f"{config_name}.density",
                  "Density must be between 0.5-5.0 g/cm³")

    validate_range(config.melting_point, 200.0, 500.0, f"{config_name}.melting_point",
                  "Melting point must be between 200-500 K")

    if config.boiling_point <= config.melting_point:
        raise ConfigValidationError(f"{config_name}.boiling_point", config.boiling_point,
                                  "Boiling point must be higher than melting point")

    # Validate solubility properties
    validate_range(config.water_solubility, 0.001, 2000.0, f"{config_name}.water_solubility",
                  "Water solubility must be between 0.001-2000 g/L")

    validate_range(config.log_kow, -5.0, 10.0, f"{config_name}.log_kow",
                  "Log Kow must be between -5.0 to 10.0")

    # Validate diffusion coefficients
    validate_range(config.diffusion_coefficient_water, 1e-7, 1e-3,
                  f"{config_name}.diffusion_coefficient_water",
                  "Water diffusion coefficient must be between 1e-7 to 1e-3 cm²/s")

    validate_range(config.diffusion_coefficient_biofilm, 1e-8, 1e-4,
                  f"{config_name}.diffusion_coefficient_biofilm",
                  "Biofilm diffusion coefficient must be between 1e-8 to 1e-4 cm²/s")

    # Validate species-specific kinetics
    for species, kinetics in config.species_kinetics.items():
        validate_substrate_kinetics_config(kinetics, f"{config_name}.species_kinetics[{species.value}]")

    # Validate degradation pathways
    for i, pathway in enumerate(config.degradation_pathways):
        validate_substrate_degradation_pathway(pathway, f"{config_name}.degradation_pathways[{i}]")

    # Validate toxicity parameters
    if config.toxicity_threshold is not None:
        validate_range(config.toxicity_threshold, 0.1, 10000.0,
                      f"{config_name}.toxicity_threshold",
                      "Toxicity threshold must be between 0.1-10000 mmol/L")

    # Validate environmental fate
    validate_range(config.biodegradability, 0.0, 1.0, f"{config_name}.biodegradability",
                  "Biodegradability must be between 0-1")

    validate_range(config.half_life_aerobic, 0.1, 10000.0, f"{config_name}.half_life_aerobic",
                  "Aerobic half-life must be between 0.1-10000 hours")

    validate_range(config.half_life_anaerobic, 0.1, 50000.0, f"{config_name}.half_life_anaerobic",
                  "Anaerobic half-life must be between 0.1-50000 hours")

    # Validate analytical properties
    validate_range(config.quantification_limit, 1e-6, 10.0, f"{config_name}.quantification_limit",
                  "Quantification limit must be between 1e-6 to 10 mmol/L")

    return True


def validate_electrochemical_config(config: ElectrochemicalConfig,
                                   config_name: str = "electrochemical_config") -> bool:
    """
    Validate electrochemical configuration.

    Args:
        config: ElectrochemicalConfig to validate
        config_name: Name for error messages

    Returns:
        True if validation passes
    """
    # Validate fundamental constants (should be close to accepted values)
    if abs(config.faraday_constant - 96485.0) > 1.0:
        raise ConfigValidationError(f"{config_name}.faraday_constant", config.faraday_constant,
                                  "Faraday constant should be approximately 96485 C/mol")

    if abs(config.gas_constant - 8.314) > 0.01:
        raise ConfigValidationError(f"{config_name}.gas_constant", config.gas_constant,
                                  "Gas constant should be approximately 8.314 J/mol/K")

    # Validate standard potentials
    for couple, potential in config.standard_potentials.items():
        validate_range(potential, -2.0, 2.0, f"{config_name}.standard_potentials[{couple}]",
                      "Standard potential must be between -2.0 to 2.0 V vs SHE")

    # Validate electrode properties
    for electrode, props in config.electrode_properties.items():
        validate_range(props['surface_area'], 0.0001, 100.0,
                      f"{config_name}.electrode_properties[{electrode}].surface_area",
                      "Surface area must be between 0.0001-100 m²/g")

        validate_range(props['conductivity'], 1.0, 1000000.0,
                      f"{config_name}.electrode_properties[{electrode}].conductivity",
                      "Conductivity must be between 1-1000000 S/m")

        validate_range(props['porosity'], 0.0, 0.99,
                      f"{config_name}.electrode_properties[{electrode}].porosity",
                      "Porosity must be between 0-0.99")

    # Validate membrane properties
    for membrane, props in config.membrane_properties.items():
        validate_range(props['thickness'], 0.01, 10.0,
                      f"{config_name}.membrane_properties[{membrane}].thickness",
                      "Membrane thickness must be between 0.01-10 mm")

        validate_range(props['conductivity'], 0.001, 1.0,
                      f"{config_name}.membrane_properties[{membrane}].conductivity",
                      "Membrane conductivity must be between 0.001-1.0 S/cm")

    return True


# Comprehensive validation function
def validate_all_biological_configs(species_config: SpeciesMetabolicConfig,
                                   biofilm_config: BiofilmKineticsConfig,
                                   substrate_configs: dict[SubstrateType, ComprehensiveSubstrateConfig],
                                   electrochemical_config: ElectrochemicalConfig) -> bool:
    """
    Validate all biological configurations together for consistency.

    Args:
        species_config: Species metabolic configuration
        biofilm_config: Biofilm kinetics configuration
        substrate_configs: Dictionary of substrate configurations
        electrochemical_config: Electrochemical configuration

    Returns:
        True if all validations pass

    Raises:
        ConfigValidationError: If any configuration is invalid
    """
    # Validate individual configurations
    validate_species_metabolic_config(species_config)
    validate_biofilm_kinetics_config(biofilm_config)
    validate_electrochemical_config(electrochemical_config)

    for substrate_type, substrate_config in substrate_configs.items():
        validate_comprehensive_substrate_config(substrate_config, f"substrate_configs[{substrate_type.value}]")

    # Cross-validation checks

    # Check that species has kinetics for configured substrates
    for substrate_type in substrate_configs.keys():
        if species_config.species not in substrate_configs[substrate_type].species_kinetics:
            raise ConfigValidationError(
                f"substrate_configs[{substrate_type.value}].species_kinetics",
                list(substrate_configs[substrate_type].species_kinetics.keys()),
                f"No kinetics defined for species {species_config.species.value} with substrate {substrate_type.value}"
            )

    # Check consistency between biofilm and species growth parameters
    if abs(biofilm_config.monod_kinetics['max_growth_rate'] - species_config.max_growth_rate) > 0.5:
        raise ConfigValidationError(
            "growth_rate_consistency",
            (biofilm_config.monod_kinetics['max_growth_rate'], species_config.max_growth_rate),
            "Maximum growth rates in biofilm and species configs should be similar (within 0.5 h⁻¹)"
        )

    return True
