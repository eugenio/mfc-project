# QCM Biofilm Monitoring Literature Summary

## Key Literature Sources

### Geobacter sulfurreducens Biofilm Thickness Studies

1. **"The effect of biofilm thickness on electrochemical activity of Geobacter sulfurreducens"**
   - Journal: International Journal of Hydrogen Energy
   - DOI: 10.1016/j.ijhydene.2015.04.137
   - Key findings: 
     - Optimal thickness: ~20 μm for maximum electrochemical activity
     - Maximum thickness: ~45 μm (growth ceases)
     - Viable biofilms: up to 80 μm under certain conditions
     - Current density relationship: 2-8 μA/μg protein

2. **"Detecting Excess Biofilm Thickness in Microbial Electrolysis Cells"**
   - Recent study on real-time biofilm monitoring
   - Maximum non-limited thickness: 55 μm for mixed cultures
   - Bulk acetate concentration dependency: 8 mmol/L conditions

### QCM Physics Models

3. **"A Practical Model of Quartz Crystal Microbalance in Actual Applications"**
   - PMC Article ID: PMC5579555
   - DOI: 10.3390/s17081785
   - Key equations:
     - Original Sauerbrey: Δm = −CQCM × Δf
     - Practical model: Δf = −CQCM* × Δm
     - Mass sensitivity: Sf(r,θ) = |A(r,θ)|²/[2π∫0∞r|A(r,θ)|²dr] × Cf
   - Gaussian distribution of mass sensitivity
   - Electrode size effects on sensitivity

4. **"Advances in the Mass Sensitivity Distribution of Quartz Crystal Microbalances"**
   - MDPI Sensors, 2022
   - DOI: 10.3390/s22145112
   - Comprehensive review of mass sensitivity factors:
     - Electrode shape, diameter, thickness effects
     - Different materials (Au, Ag, Al)
     - Frequency dependencies

### QCM-Biofilm Applications

5. **"Tracking Biofilms Using an Electrochemical Quartz Crystal Microbalance"**
   - Gamry Instruments Application Note
   - Geobacter sulfurreducens specific studies
   - Protocol: 10 MHz Au-coated crystal, 0 V vs Ag/AgCl
   - Current-frequency correlation for biofilm growth
   - Reduced Q indicates biofilm viscosity

6. **"Real-time monitoring of electrochemically active biofilm"**
   - ScienceDirect article on EQCM and ATR/FTIR
   - DOI: 10.1016/j.snb.2014.12.027
   - Multi-technique approach for biofilm characterization

## Literature-Derived Parameters

### Geobacter sulfurreducens Biofilm Properties
- **Optimal thickness**: 20 μm
- **Maximum viable thickness**: 45-80 μm  
- **Density**: ~1.1 g/cm³ (typical biofilm density)
- **Young's modulus**: 10-100 kPa (soft biological material)
- **Current density**: 172 ± 29 μA/cm²

### QCM Sensor Specifications
- **Fundamental frequency**: 5-10 MHz typical
- **Mass sensitivity**: ~17.7 ng/(cm²·Hz) at 5 MHz
- **Q factor**: 10,000-100,000 in air
- **Temperature coefficient**: -20 ppm/°C
- **Electrode material**: Gold (most common)

## Implementation Notes

1. **Sauerbrey equation limitations**: Not valid for soft, viscous biofilms
2. **Viscoelastic corrections**: Necessary for accurate biofilm mass determination
3. **Real-time monitoring**: Current generation is better proxy than total mass
4. **Calibration**: Species-specific calibration required for accurate thickness
5. **Environmental factors**: Temperature, pH, ionic strength affect measurements