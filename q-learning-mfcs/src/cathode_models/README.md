# Cathode Models for MFC Simulation

## Overview
This directory contains cathode models for microbial fuel cell (MFC) simulations, including both platinum-based and biological cathode models based on recent literature (2023-2024).

## Literature Review Summary

### Platinum-Based Cathode Models

**Key Kinetic Parameters (from literature):**
- Exchange current density (i₀): 3.0 × 10⁻⁹ A/cm²
- Tafel slope: 60 mV/decade (low overpotential), 120 mV/decade (high overpotential)
- Oxygen reduction reaction (ORR) follows Butler-Volmer kinetics
- Temperature dependency affects kinetics, activation energy, and electrode potentials

**Performance Characteristics:**
- Commercial Pt/C reference performance: 358-458 mW/m²
- Platinum loading typically: 0.5 mg/cm²
- Major limitation: sluggish ORR kinetics
- Cost-effective alternatives (Ag-Fe-N/C) can outperform Pt/C

### Biological Cathode (Biocathode) Models

**Key Features:**
- Microbial communities: Sphingobacterium, Acinetobacter, Acidovorax sp.
- Direct electron transfer from cathode to microorganisms
- Oxygen reduction catalyzed by electroactive bacteria
- Biofilm formation crucial for performance

**Performance Characteristics:**
- Reduced charge-transfer resistance compared to abiotic cathodes
- Mass transfer limitation of oxygen is significant bottleneck
- Performance depends on: cathode potential, oxygen mass transfer, biofilm thickness
- Can operate for extended periods (>150 days reported)

**Applications:**
- Wastewater treatment with heavy metals removal
- Self-powered electrochemical bioremediation
- Plant microbial fuel cells (P-MFCs)

## Proposed Model Implementation

### 1. Unified Cathode Model Structure
```python
class CathodeModel:
    def __init__(self, cathode_type='platinum'):
        self.cathode_type = cathode_type
        self.area = None  # m²
        self.load_parameters()
    
    def calculate_overpotential(self, current_density, oxygen_conc, temperature):
        """Calculate cathode overpotential using Butler-Volmer kinetics"""
        pass
    
    def calculate_power_consumption(self, overpotential, current):
        """Calculate power loss at cathode"""
        pass
```

### 2. Platinum Cathode Model
- Butler-Volmer kinetics with literature-based parameters
- Temperature-dependent exchange current density
- Dual Tafel slope regions (low/high overpotential)
- Oxygen concentration dependency (Nernst equation)

### 3. Biological Cathode Model
- Combined Butler-Volmer + Monod kinetics
- Biofilm growth modeling
- Microbial community dynamics
- Oxygen mass transfer limitations
- pH dependency for microbial activity

### 4. Mathematical Framework

**Butler-Volmer Equation for ORR:**
```
i = i₀ * [C_O₂/C_O₂_ref] * [exp(-αF*η/RT) - exp((1-α)F*η/RT)]
```

**Monod Kinetics for Biocathode:**
```
μ = μ_max * (C_O₂/(K_O₂ + C_O₂)) * (electrode_potential - E_min)/(E_opt - E_min)
```

**Biofilm Growth:**
```
dX_biofilm/dt = μ * X_biofilm - k_decay * X_biofilm - k_detachment * X_biofilm
```

## Integration with Existing MFC Model

### Current System Enhancement
1. **Separate cathode area from anode area** ✓ (already implemented)
2. **Add cathode-specific power calculations**
3. **Include cathode overpotential in total cell voltage**
4. **Implement oxygen mass transport to cathode**
5. **Add cathode biofilm growth (for biocathode option)**

### Model Parameters to Add
- Cathode type selection (platinum/biological)
- Cathode-specific kinetic parameters
- Oxygen solubility and diffusion coefficients
- Biofilm parameters (for biological cathodes)
- Temperature effects on all parameters

## Expected Benefits

1. **More Accurate Power Predictions**: Account for cathode losses
2. **Design Optimization**: Optimize cathode size vs anode size
3. **Technology Comparison**: Compare platinum vs biological cathodes
4. **Cost Analysis**: Include cathode material costs in economics
5. **Environmental Impact**: Model sustainability of different cathode types

## Implementation Priority

1. **High Priority**: Platinum cathode model with Butler-Volmer kinetics
2. **Medium Priority**: Temperature dependency and oxygen mass transfer
3. **Low Priority**: Biological cathode model with biofilm dynamics

## References

1. Khan et al. (2024) - Techno-economic perspective of MFCs
2. Frontiers Research (2023) - Biological cathodes for heavy metal treatment
3. Environmental Science & Technology - Biocathode performance studies
4. Multiple studies on Butler-Volmer kinetics in MFC systems

---
*Created: 2025-07-26*
*Last Updated: 2025-07-26*