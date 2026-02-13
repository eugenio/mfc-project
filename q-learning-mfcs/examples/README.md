# Configuration Examples

This directory contains comprehensive examples demonstrating how to use the biological configuration system for MFC modeling. The configuration system provides species-specific, substrate-specific, and literature-referenced parameters for accurate biological modeling.

## Overview

The biological configuration system replaces hardcoded values with:

- **Species-specific configurations**: Metabolic parameters for different bacterial species
- **Substrate-specific configurations**: Kinetic and thermodynamic properties for various substrates
- **Literature references**: All parameters backed by peer-reviewed research
- **Comprehensive validation**: Ensures biological plausibility of all parameters

## Example Files

### 1. `example_geobacter_acetate_config.py`

**Focus**: Geobacter sulfurreducens with acetate substrate

**Key Features**:

- Custom metabolic reaction definitions
- Enhanced electron transport parameters
- Literature-referenced kinetic constants
- Integration with metabolic models

**Usage**:

```bash
python example_geobacter_acetate_config.py
```

**When to use**:

- Modeling acetate-fed MFCs
- Studying direct electron transfer mechanisms
- Optimizing biofilm conductivity
- Research on exoelectrogenic biofilms

### 2. `example_shewanella_lactate_config.py`

**Focus**: Shewanella oneidensis with lactate substrate (enhanced marine strain)

**Key Features**:

- Advanced degradation pathway configuration
- Environmental condition optimization
- Flavin-mediated electron transfer
- Marine/high-salt adaptations
- Complex substrate transport mechanisms

**Usage**:

```bash
python example_shewanella_lactate_config.py
```

**When to use**:

- Modeling lactate utilization
- Marine/brackish water applications
- Studying flavin-mediated electron transfer
- Environmental adaptation research
- Complex substrate degradation pathways

### 3. `example_mixed_culture_config.py`

**Focus**: Mixed culture systems with substrate competition

**Key Features**:

- Multi-species configuration management
- Substrate competition modeling
- Synergistic interaction calculations
- Dynamic species ratio effects
- Biofilm property blending

**Usage**:

```bash
python example_mixed_culture_config.py
```

**When to use**:

- Mixed culture MFC systems
- Substrate competition studies
- Synergy effect analysis
- Dynamic community modeling
- Industrial wastewater treatment

## Configuration System Architecture

### Core Components

1. **Species Configurations** (`config/biological_config.py`)

   - `SpeciesMetabolicConfig`: Species-specific metabolic parameters
   - `BiofilmKineticsConfig`: Biofilm formation and growth parameters
   - `ElectrochemicalConfig`: Electrochemical constants and electrode properties

1. **Substrate Configurations** (`config/substrate_config.py`)

   - `ComprehensiveSubstrateConfig`: Complete substrate characterization
   - `SubstrateKineticsConfig`: Uptake and degradation kinetics
   - `SubstrateDegradationPathway`: Metabolic pathway definitions

1. **Validation System** (`config/biological_validation.py`)

   - Parameter range validation
   - Biological plausibility checks
   - Cross-validation between configurations
   - Literature reference verification

### Key Features

- **Literature-backed parameters**: All values referenced to peer-reviewed sources
- **Environmental compensation**: Temperature and pH adjustments
- **Comprehensive validation**: Ensures parameter biological plausibility
- **Modular design**: Easy to extend with new species/substrates
- **Integration ready**: Direct compatibility with existing models

## Usage Patterns

### Basic Configuration Usage

```python
from config.biological_config import get_geobacter_config
from config.substrate_config import get_acetate_config

# Load default configurations
species_config = get_geobacter_config()
substrate_config = get_acetate_config()

# Use with models
model = MetabolicModel(
    species="geobacter",
    substrate="acetate",
    species_config=species_config,
    substrate_config=substrate_config
)
```

### Custom Configuration

```python
# Customize parameters for specific experiments
species_config = get_geobacter_config()
species_config.max_growth_rate = 0.32  # Enhanced strain
species_config.electron_transport_efficiency = 0.92

# Add custom reactions
custom_reaction = MetabolicReactionConfig(
    id="CUSTOM_R001",
    name="Custom enzyme",
    # ... other parameters
)
species_config.reactions.append(custom_reaction)
```

### Mixed Culture Configuration

```python
from examples.example_mixed_culture_config import MixedCultureConfig

# Define species composition
species_ratios = {
    BacterialSpecies.GEOBACTER_SULFURREDUCENS: 0.6,
    BacterialSpecies.SHEWANELLA_ONEIDENSIS: 0.4
}

# Create mixed culture configuration
mixed_config = MixedCultureConfig(species_ratios)

# Calculate effective kinetics
effective_kinetics = mixed_config.get_effective_substrate_kinetics(
    SubstrateType.ACETATE
)
```

## Parameter Sources and References

All parameters are derived from peer-reviewed literature:

### Key References

- **Lovley (2003)**: Geobacter metabolism and electron transfer
- **Bond et al. (2002)**: Electrode-reducing microorganisms
- **Marsili et al. (2008)**: Shewanella flavin-mediated electron transfer
- **Torres et al. (2010)**: Kinetic perspective on extracellular electron transfer
- **Marcus et al. (2007)**: Biofilm anode modeling
- **Reguera et al. (2005)**: Microbial nanowires

### Parameter Categories

- **Kinetic constants**: Vmax, Km, Ki from enzymatic studies
- **Thermodynamic values**: ΔG°, ΔH°, ΔS° from calorimetry
- **Growth parameters**: μmax, Y from batch culture experiments
- **Biofilm properties**: Density, porosity from microscopy studies
- **Electrochemical values**: Standard potentials from electrochemistry

## Integration with Models

### Metabolic Models

```python
from metabolic_model.metabolic_core import MetabolicModel

model = MetabolicModel(
    species_config=your_species_config,
    substrate_config=your_substrate_config
)
```

### Biofilm Models

```python
from biofilm_kinetics.biofilm_model import BiofilmKineticsModel

model = BiofilmKineticsModel(
    species_config=your_species_config,
    biofilm_config=your_biofilm_config,
    substrate_config=your_substrate_config
)
```

### Sensor-Integrated Models

```python
from sensor_integrated_mfc_model import SensorIntegratedMFCModel

model = SensorIntegratedMFCModel(
    species_config=your_species_config,
    substrate_config=your_substrate_config
)
```

## Best Practices

### 1. Always Validate Configurations

```python
from config.biological_validation import validate_species_metabolic_config

validate_species_metabolic_config(your_config)
```

### 2. Use Literature References

```python
from config.biological_config import LITERATURE_REFERENCES

# Reference parameters to literature
kinetics.reference = LITERATURE_REFERENCES['lovley_2003']
```

### 3. Environment-Specific Adaptations

```python
# Adjust for temperature
if temperature != 303.0:  # Reference temperature
    temp_factor = np.exp(-Ea/R * (1/T - 1/T_ref))
    kinetics.vmax *= temp_factor
```

### 4. Validate Parameter Ranges

```python
from config.parameter_validation import validate_range

validate_range(growth_rate, 0.001, 2.0, "growth_rate", 
               "Growth rate must be biologically realistic")
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the `src` directory is in your Python path
1. **Validation Failures**: Check parameter ranges against biological limits
1. **Missing References**: Verify literature reference completeness
1. **Configuration Conflicts**: Use cross-validation for multi-species systems

### Debug Tips

- Run examples with verbose output: `python -v example_file.py`
- Check validation error messages for specific parameter issues
- Verify literature references are complete
- Test configurations with simple models first

## Contributing

To add new configurations:

1. Define new species/substrate parameters in appropriate config files
1. Add literature references to `LITERATURE_REFERENCES`
1. Create validation functions for new parameter types
1. Write example configurations demonstrating usage
1. Update this README with new examples

## License

These examples are part of the MFC Q-learning project and follow the same license terms.
