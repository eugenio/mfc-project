"""
Cathode models for MFC simulations

This package provides cathode models for microbial fuel cell simulations,
including platinum-based and biological cathode models with literature-based
parameters and comprehensive performance analysis.

Available models:
- BaseCathodeModel: Abstract base class with Butler-Volmer kinetics
- PlatinumCathodeModel: Platinum cathode with temperature dependency  
- BiologicalCathodeModel: Biological cathode with biofilm dynamics

Convenience functions:
- create_platinum_cathode(): Create platinum cathode with standard parameters
- create_biological_cathode(): Create biological cathode with standard parameters

Created: 2025-07-26
"""

from .base_cathode import BaseCathodeModel, ButlerVolmerKinetics, CathodeParameters
from .biological_cathode import (
    BiologicalCathodeModel,
    BiologicalParameters,
    create_biological_cathode,
)
from .platinum_cathode import (
    PlatinumCathodeModel,
    PlatinumParameters,
    create_platinum_cathode,
)

__all__ = [
    'BaseCathodeModel',
    'CathodeParameters',
    'ButlerVolmerKinetics',
    'PlatinumCathodeModel',
    'PlatinumParameters',
    'create_platinum_cathode',
    'BiologicalCathodeModel',
    'BiologicalParameters',
    'create_biological_cathode'
]

__version__ = '1.0.0'
