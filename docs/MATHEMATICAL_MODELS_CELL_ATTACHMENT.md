# Mathematical Models of Cell Attachment for Exoelectrogenic Bacteria

## Overview

This document compiles mathematical models for cell attachment and biofilm formation kinetics in exoelectrogenic bacteria, with focus on *Geobacter sulfurreducens* and *Shewanella oneidensis* MR-1. These models are essential for understanding biofilm initiation, growth dynamics, and electron transfer in microbial fuel cells.

## 1. General Biofilm Formation Models

### 1.1 Stochastic Attachment Model

A probabilistic approach to bacterial cell attachment on surfaces:

**Attachment Probabilities:**

- `p_s`: Attachment probability on surface
- `p_b1`: Probability of horizontal attachment on bacteria sides
- `p_b2`: Probability of vertical attachment above/below bacteria

**Attachment Probability Matrix:**

```math
M_pr(i,j) = f(p_s, p_b1, p_b2, n_adjacent)
```

where `n_adjacent` is the number of occupied adjacent grid elements.

**Algorithm Steps:**

1. Randomly select possible attachment element
1. Perform Bernoulli test based on `M_pr(i,j)` value
1. Mark grid element as occupied if test succeeds
1. Repeat until desired bacterial cell density achieved

### 1.2 Monod Kinetics for Biofilm Growth

**Mass Balance Equation with Monod Kinetics:**

```math
d²S_f/dz² = (k·X_f·S_f)/(D_f·(K + S_f))
```

Where:

- `S_f`: Substrate concentration in biofilm (mmol/L)
- `k`: Reaction rate constant (1/h)
- `X_f`: Biomass concentration in biofilm (g/L)
- `D_f`: Diffusion coefficient in biofilm (m²/s)
- `K`: Monod half-saturation constant (mmol/L)
- `z`: Spatial coordinate (m)

**Boundary Conditions:**

- At attachment surface (z = 0): `dS_f0/dz = 0` (no flux)
- At biofilm/water interface (z = L_e): `J = D_f·(dS_f/dz) = D·(S_0 - S_s)/L`

## 2. Geobacter sulfurreducens Specific Models

### 2.1 Nernst-Monod Kinetic Model

**Growth Rate Expression:**

```math
μ = μ_max · (S/(K_s + S)) · (E_a - E_ka)/(E_ka - E_an)
```

Where:

- `μ`: Specific growth rate (1/h)
- `μ_max`: Maximum specific growth rate (1/h)
- `S`: Substrate concentration (mmol/L)
- `K_s`: Half-saturation constant (mmol/L)
- `E_a`: Anode potential (V)
- `E_ka`: Half-saturation potential (V)
- `E_an`: Potential at which growth ceases (V)

### 2.2 Biofilm Conductivity Model

**Current Density Expression:**

```math
j = σ_biofilm · (dE/dx)
```

Where:

- `j`: Current density (A/m²)
- `σ_biofilm`: Biofilm conductivity (S/m)
- `dE/dx`: Electric field gradient (V/m)

**Biofilm Resistance:**

```math
R_biofilm = L_biofilm/(σ_biofilm · A)
```

Where:

- `R_biofilm`: Biofilm resistance (Ω)
- `L_biofilm`: Biofilm thickness (m)
- `A`: Electrode surface area (m²)

### 2.3 Extracellular Electron Transfer (EET) Kinetics

**Rate-Limiting Steps During Growth:**

1. **Early biofilm**: Irreversible acetate turnover

   ```math
   r_acetate = k_acetate · [Acetate] · X_biofilm
   ```

1. **Mature biofilm**: Electron transfer to extracellular redox cofactors

   ```math
   r_EET = k_EET · [ERC] · (E_donor - E_acceptor)
   ```

Where:

- `[ERC]`: Concentration of extracellular redox cofactors
- `k_EET`: Electron transfer rate constant
- `E_donor`, `E_acceptor`: Redox potentials of donor and acceptor

## 3. Shewanella oneidensis MR-1 Models

### 3.1 Planktonic vs. Biofilm Contribution

**Total Current Generation:**

```math
I_total = I_biofilm + I_planktonic
```

**Biofilm Current Density:**

```math
j_biofilm = n_F · k_biofilm · X_biofilm · η
```

Where:

- `n_F`: Number of electrons transferred × Faraday constant
- `k_biofilm`: Biofilm-specific rate constant
- `η`: Overpotential efficiency factor

**Planktonic Current Contribution:**

```math
j_planktonic = n_F · k_planktonic · X_planktonic · f_transport
```

Where:

- `f_transport`: Transport efficiency to electrode

### 3.2 Mixed Culture Synergy Model

**Enhanced Performance in G. sulfurreducens-S. oneidensis Mixtures:**

```math
j_mixed = j_Gs + α · j_So · f_synergy
```

Where:

- `j_Gs`: Current from G. sulfurreducens alone
- `j_So`: Current from S. oneidensis alone
- `α`: Synergy coefficient (α = 1.38 based on experimental data)
- `f_synergy`: Synergy efficiency factor

## 4. Differential Equation Systems

### 4.1 Multi-Species Biofilm Model

**Coupled PDE System:**

```math
∂X_i/∂t = ∇·(D_i∇X_i) + μ_i·X_i - k_death,i·X_i
∂S_j/∂t = ∇·(D_s,j∇S_j) - Σ(Y_i,j·μ_i·X_i)
```

Where:

- `X_i`: Biomass density of species i
- `S_j`: Concentration of substrate j
- `D_i`: Biomass diffusion coefficient
- `Y_i,j`: Yield coefficient

### 4.2 Biofilm Growth with Detachment

**Mass Balance with Detachment:**

```math
∂X/∂t = μ·X - k_det·X^n
```

Where:

- `k_det`: Detachment rate constant
- `n`: Detachment order (typically n = 1 or 2)

### 4.3 EPS Production Model

**EPS Formation Rate:**

```math
dEPS/dt = k_EPS · μ · X - k_decay · EPS
```

Where:

- `k_EPS`: EPS production coefficient
- `k_decay`: EPS decay rate constant

## 5. Parameter Values from Literature

### 5.1 Geobacter sulfurreducens

- Maximum current density: 0.39 ± 0.09 mA/cm²
- Biofilm thickness: 69 μm (pure culture), 93 μm (mixed culture)
- Biofilm conductivity: 10⁻⁵ to 10⁻³ S/m
- Half-saturation constant (acetate): 0.1-1.0 mM

### 5.2 Shewanella oneidensis MR-1

- Maximum current density: 0.034 ± 0.011 mA/cm²
- Biofilm formation: Less stable than G. sulfurreducens
- Planktonic contribution: Significant in mixed cultures
- Half-saturation constant (lactate): 0.5-2.0 mM

### 5.3 Mixed Cultures

- Enhanced current density: 0.54 ± 0.07 mA/cm² (38% increase)
- Synergy coefficient: α = 1.38
- Optimal ratio: G. sulfurreducens:S. oneidensis ≈ 3:1

## 6. Model Applications

### 6.1 MFC Design Optimization

- Electrode spacing optimization
- Biofilm thickness control
- Current density prediction

### 6.2 Operational Parameter Control

- Substrate feeding strategies
- pH and temperature optimization
- Hydraulic retention time

### 6.3 Performance Prediction

- Long-term stability assessment
- Mixed culture optimization
- Scale-up considerations

## 7. Model Limitations and Future Directions

### 7.1 Current Limitations

- Limited spatial resolution in stochastic models
- Simplified EET kinetics
- Assumption of homogeneous biofilm properties
- Limited multi-species interaction modeling

### 7.2 Future Developments

- 3D biofilm structure modeling
- Molecular-scale EET mechanisms
- Real-time parameter adaptation
- Machine learning integration for parameter optimization

## References

Based on scientific literature search conducted on 2025-01-24, including studies on:

- Nernst-Monod kinetic models for anode-respiring bacteria
- Biofilm formation mathematics with Monod kinetics
- Multi-skilled mathematical models of bacterial attachment
- EET kinetics in G. sulfurreducens biofilms
- Mixed culture dynamics in bioelectrochemical systems

## Implementation Notes

These mathematical models can be implemented in simulation frameworks for:

1. **Biofilm growth prediction** in MFC systems
1. **Current generation optimization** through parameter tuning
1. **Mixed culture design** for enhanced performance
1. **Scale-up analysis** for practical applications

The models require computational solutions using numerical methods such as finite difference, finite element, or Monte Carlo approaches depending on the specific model complexity and spatial dimensions.
