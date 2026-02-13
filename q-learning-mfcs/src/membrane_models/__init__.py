"""Membrane models for MFC simulations.

This package provides comprehensive membrane models for microbial fuel cells,
including proton exchange membranes (PEM), anion exchange membranes (AEM),
and bipolar membranes with transport mechanisms and degradation modeling.

Available models:
- BaseMembraneModel: Abstract base class for all membrane types
- ProtonExchangeMembrane: PEM models (Nafion, SPEEK, etc.)
- AnionExchangeMembrane: AEM models with hydroxide transport
- BipolarMembrane: Combined PEM/AEM functionality
- CeramicMembrane: Ceramic and composite membranes

Key features:
- Ion transport mechanisms (protons, hydroxides, other ions)
- Water transport and electro-osmotic drag
- Membrane fouling and degradation
- Temperature and humidity effects
- Multi-ion transport and selectivity

Created: 2025-07-27
"""

from .anion_exchange import AEMParameters, AnionExchangeMembrane, create_aem_membrane
from .base_membrane import BaseMembraneModel, IonTransportMechanisms, MembraneParameters
from .bipolar_membrane import (
    BipolarMembrane,
    BipolarParameters,
    create_bipolar_membrane,
)
from .ceramic_membrane import (
    CeramicMembrane,
    CeramicParameters,
    create_ceramic_membrane,
)
from .membrane_fouling import (
    FoulingModel,
    FoulingParameters,
    calculate_fouling_resistance,
)
from .proton_exchange import (
    PEMParameters,
    ProtonExchangeMembrane,
    create_nafion_membrane,
    create_speek_membrane,
)

__all__ = [
    "AEMParameters",
    # AEM
    "AnionExchangeMembrane",
    # Base classes
    "BaseMembraneModel",
    # Bipolar
    "BipolarMembrane",
    "BipolarParameters",
    # Ceramic
    "CeramicMembrane",
    "CeramicParameters",
    # Fouling
    "FoulingModel",
    "FoulingParameters",
    "IonTransportMechanisms",
    "MembraneParameters",
    "PEMParameters",
    # PEM
    "ProtonExchangeMembrane",
    "calculate_fouling_resistance",
    "create_aem_membrane",
    "create_bipolar_membrane",
    "create_ceramic_membrane",
    "create_nafion_membrane",
    "create_speek_membrane",
]

__version__ = "1.0.0"
