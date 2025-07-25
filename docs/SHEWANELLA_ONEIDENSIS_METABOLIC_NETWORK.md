# Shewanella oneidensis MR-1 Metabolic Network Analysis

## Executive Summary

This document presents a comprehensive analysis of Shewanella oneidensis MR-1 metabolic pathways based on KEGG database entries, constraint-based metabolic models, and recent literature. The analysis focuses on central carbon metabolism, lactate utilization, electron transfer mechanisms, and comparison with Geobacter sulfurreducens.

## 1. KEGG Pathway Analysis

### 1.1 Organism Classification

- **KEGG Organism Code**: son
- **Full Name**: Shewanella oneidensis MR-1
- **Genome Size**: 5,131,449 bp
- **Gene Count**: 4,758 genes

### 1.2 Central Carbon Metabolism Pathways

#### Glycolysis/Gluconeogenesis (KEGG: son00010)

- **Pathway Components**: EMP pathway, gluconeogenesis
- **Key Enzymes**: Phosphofructokinase, pyruvate kinase, glucose-6-phosphate dehydrogenase
- **Module Coverage**: M00001 (Glycolysis), M00002 (Glycolysis core module)

#### Citrate Cycle (KEGG: son00020)

- **Pathway Status**: Branched TCA cycle under anaerobic conditions
- **Key Feature**: Incomplete TCA cycle during anaerobic respiration
- **Module Coverage**: M00009 (Citrate cycle), M00010 (Citrate cycle, second carbon oxidation), M00011 (Citrate cycle, oxaloacetate ⇒ 2-oxoglutarate)

#### Pentose Phosphate Pathway (KEGG: son00030)

- **Function**: NADPH generation, pentose synthesis
- **Module Coverage**: M00004 (Pentose phosphate pathway), M00006 (Pentose phosphate pathway, oxidative phase), M00007 (Pentose phosphate pathway, non-oxidative phase)

#### Entner-Doudoroff Pathway (KEGG: son00051)

- **Alternative**: Alternative glucose catabolism pathway
- **Significance**: Provides metabolic flexibility

### 1.3 Lactate Metabolism Pathway

#### Lactate Oxidation

```
Lactate → Pyruvate + 2e⁻ + 2H⁺
Pyruvate → Acetyl-CoA + CO₂ + 2e⁻ + 2H⁺ (via pyruvate dehydrogenase)
```

#### Key Metabolic Features

- **Lactate Consumption Rate**: 4.06-4.11 mmol/g AFDW/h
- **Incomplete TCA Utilization**: Under anaerobic conditions, acetyl-CoA not fully oxidized
- **Electron Yield**: 4 electrons per lactate molecule (partial oxidation)

## 2. Constraint-Based Metabolic Models

### 2.1 iSO783 Model (2010)

- **Reactions**: 774
- **Genes**: 783
- **Metabolites**: 634
- **Coverage**: Biosynthesis pathways for all cell constituents

### 2.2 iLJ1162 Model (2022)

- **Reactions**: 2,084
- **Genes**: 1,162
- **Metabolites**: 1,818
- **Validation Accuracy**: 86.9% agreement with experimental results

### 2.3 Stoichiometric Coefficients and Flux Constraints

#### Growth Parameters

- **Maximum Growth Rate**: 0.085 h⁻¹
- **ATP Requirements**:
  - Non-growth associated (NGAR): 1.03 mmol ATP/(g AFDW·h)
  - Growth associated (GAR): 220.22 mmol ATP/g AFDW

#### Electron Transport Chain Constraints

- **Proton Translocation Ratios**:
  - Cytochrome c oxidases (Cco, Cox): 6H⁺/2e⁻
  - Cytochrome d ubiquinol oxidase (Cyd): 2H⁺/2e⁻
  - Overall ubiquinol to O₂: 2.8H⁺/2e⁻

## 3. Electron Transfer Mechanisms

### 3.1 Flavin-Mediated Electron Transfer

#### Flavin Production

- **Primary Compounds**: Riboflavin, riboflavin-5'-phosphate
- **Concentration**: Few µM effective concentration
- **Function**: Soluble redox shuttle
- **Impact**: >70% reduction in electron transfer rate when removed

#### Mechanism

```
Flavin + 2H⁺ + 2e⁻ ⇌ Flavin-H₂ (reduced form)
Flavin-H₂ → Electrode + 2H⁺ + 2e⁻
```

### 3.2 Cytochrome Network

#### Periplasmic Cytochrome Network

- **Inner Membrane Quinol Oxidases**:
  - CymA (tetraheme cytochrome)
  - bc₁ complex
  - TorC (pentaheme cytochrome)

#### Outer Membrane Components

- **Mtr System**: MtrCAB complex
- **OmcA**: Outer membrane cytochrome A
- **Function**: Direct electron transfer to external acceptors

### 3.3 Nanowire Structure

- **Composition**: Outer membrane and periplasmic extensions
- **Components**: Include multiheme cytochromes
- **Function**: Long-range electron transfer

## 4. Respiratory Chain Components

### 4.1 Terminal Oxidases

#### Aerobic Respiration

1. **Cytochrome c oxidase (Cco)**

   - Electron acceptor: O₂
   - Proton pumping: 4H⁺/2e⁻

1. **Cytochrome c oxidase (Cox)**

   - Electron acceptor: O₂
   - Proton pumping: 4H⁺/2e⁻

1. **Cytochrome d ubiquinol oxidase (Cyd)**

   - Electron acceptor: O₂
   - Proton pumping: 2H⁺/2e⁻

#### Anaerobic Respiration

- **Fumarate reductase**: Fumarate → Succinate
- **Nitrate reductase**: NO₃⁻ → NO₂⁻
- **TMAO reductase**: Trimethylamine N-oxide reduction
- **Metal oxide reductases**: Fe(III), Mn(IV) reduction

### 4.2 Electron Transport Chain Organization

```
NADH → Complex I → Quinone Pool → bc₁ Complex → Cytochrome c → Terminal Oxidases
                ↓
            Alternative pathways to external acceptors via Mtr system
```

## 5. Comparison with Geobacter sulfurreducens

### 5.1 Key Differences

| Feature | S. oneidensis MR-1 | G. sulfurreducens |
|---------|-------------------|-------------------|
| **Electron Shuttles** | Flavin-mediated (riboflavin, FMN) | Minimal shuttle dependence |
| **Nanowire Composition** | Outer membrane extensions with cytochromes | Conductive pili with aromatic amino acids |
| **Metabolic Strategy** | Facultative anaerobe, diverse substrates | Obligate anaerobe, limited substrates |
| **Electron Transfer** | Mixed: Direct + mediated | Primarily direct transfer |
| **TCA Cycle** | Branched/incomplete under anaerobic conditions | Complete TCA cycle |

### 5.2 Metabolic Reaction Differences

#### S. oneidensis MR-1 (Anaerobic lactate metabolism)

```
Lactate → Pyruvate → Acetyl-CoA → (incomplete TCA)
Net: 1 Lactate → 1 NADH + 1 Formate + 4e⁻
```

#### G. sulfurreducens (Complete oxidation)

```
Acetate → Acetyl-CoA → (complete TCA cycle)
Net: 1 Acetate → 8e⁻ + 8H⁺ + 2CO₂
```

## 6. Structured Metabolic Reaction Network

### 6.1 Core Metabolic Reactions

#### Lactate Utilization

```
R1: Lactate + NAD⁺ → Pyruvate + NADH + H⁺
    ΔG°' = -25.1 kJ/mol

R2: Pyruvate + CoA + NAD⁺ → Acetyl-CoA + CO₂ + NADH + H⁺
    ΔG°' = -33.5 kJ/mol

R3: Acetyl-CoA + 2H₂O → Acetate + CoA + H⁺
    ΔG°' = -31.4 kJ/mol
```

#### Electron Transport

```
R4: NADH + H⁺ + Q → NAD⁺ + QH₂
    (Complex I, ΔG°' = -69.5 kJ/mol)

R5: QH₂ + 2Cyt c(ox) → Q + 2Cyt c(red) + 2H⁺
    (bc₁ complex, ΔG°' = -31.8 kJ/mol)

R6: 2Cyt c(red) + ½O₂ + 2H⁺ → 2Cyt c(ox) + H₂O
    (Terminal oxidase, ΔG°' = -112.0 kJ/mol)
```

#### Flavin-Mediated Transfer

```
R7: Riboflavin + NADH + H⁺ → Riboflavin-H₂ + NAD⁺
    ΔG°' = -12.5 kJ/mol

R8: Riboflavin-H₂ + Electrode → Riboflavin + 2H⁺ + 2e⁻
    E°' = -0.21 V
```

### 6.2 Flux Constraints

#### Uptake Rates (mmol/g AFDW/h)

- Lactate: 4.06-4.11
- Oxygen: Variable based on availability
- Maximum growth rate: 0.085 h⁻¹

#### Production Rates

- CO₂: ~2.0 mmol/g AFDW/h
- Acetate: ~1.5 mmol/g AFDW/h (anaerobic)
- Riboflavin: ~0.001-0.01 mmol/g AFDW/h

### 6.3 Energy Balance

#### ATP Yield per Lactate

- **Aerobic conditions**: ~15 ATP/lactate
- **Anaerobic conditions**: ~2-3 ATP/lactate
- **With external electron acceptor**: ~5-8 ATP/lactate

## 7. Bioelectrochemical Applications

### 7.1 Microbial Fuel Cell Performance

- **Current Density**: 0.5-2.0 A/m²
- **Power Density**: 0.1-0.5 W/m²
- **Coulombic Efficiency**: 15-45%

### 7.2 Metabolic Engineering Targets

1. **Enhanced flavin production**: Overexpression of riboflavin synthesis genes
1. **Cytochrome optimization**: Improved electron transfer chain efficiency
1. **Metabolic pathway redirection**: Increased electron flux to electrode

## 8. Implementation Framework

### 8.1 Computational Model Structure

```python
# Stoichiometric matrix format
reactions = {
    'lactate_uptake': {'lactate_ext': -1, 'lactate_int': 1},
    'lactate_oxidation': {'lactate_int': -1, 'pyruvate': 1, 'nadh': 1},
    'pyruvate_decarboxylation': {'pyruvate': -1, 'acetyl_coa': 1, 'co2': 1, 'nadh': 1},
    'electron_transport': {'nadh': -1, 'nad': 1, 'h_ext': 2.8, 'atp': 2.5},
    'flavin_synthesis': {'riboflavin': 1, 'atp': -5},
    'extracellular_transfer': {'riboflavin_red': -1, 'riboflavin_ox': 1, 'electrons': 2}
}
```

### 8.2 Constraint Definitions

```python
flux_bounds = {
    'lactate_uptake': (0, 4.11),  # mmol/g AFDW/h
    'growth': (0, 0.085),         # h⁻¹
    'maintenance_atp': (1.03, 1.03),  # mmol ATP/(g AFDW·h)
    'riboflavin_export': (0, 0.01)    # mmol/g AFDW/h
}
```

## 9. Conclusions

S. oneidensis MR-1 presents a unique metabolic architecture optimized for diverse electron acceptor utilization through:

1. **Flexible electron transfer mechanisms** combining direct and mediated pathways
1. **Adaptive central metabolism** with branched TCA cycle under anaerobic conditions
1. **Sophisticated electron transport network** with multiple terminal oxidases
1. **Flavin-based electron shuttling** enabling efficient extracellular electron transfer

Key differences from G. sulfurreducens include greater metabolic versatility, flavin dependence, and mixed electron transfer strategies, making S. oneidensis MR-1 particularly suitable for variable environmental conditions and bioelectrochemical applications.

## References

1. Pinchuk, G.E., et al. (2010). Constraint-based model of Shewanella oneidensis MR-1 metabolism. PLoS Comput. Biol. 6(6): e1000822.

1. Liu, J., et al. (2022). Reconstruction of a genome-scale metabolic network for Shewanella oneidensis MR-1. Front. Bioeng. Biotechnol. 10: 913077.

1. Marsili, E., et al. (2008). Shewanella secretes flavins that mediate extracellular electron transfer. PNAS 105(10): 3968-3973.

1. Coursolle, D., et al. (2010). The Mtr respiratory pathway is essential for reducing flavins and electrodes in Shewanella oneidensis. J. Bacteriol. 192(2): 467-474.

1. KEGG Database: Shewanella oneidensis MR-1 (son) - https://www.kegg.jp/kegg-bin/show_organism?org=son
