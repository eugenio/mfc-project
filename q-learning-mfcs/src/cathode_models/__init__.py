"""
Cathode models for microbial fuel cell simulations

This package provides cathode models for MFC simulations including:
- Platinum-based cathodes with Butler-Volmer kinetics
- Biological cathodes with biofilm dynamics
- Unified interface for cathode performance prediction
"""

from .base_cathode import BaseCathodeModel
from .platinum_cathode import PlatinumCathodeModel
# from .biological_cathode import BiologicalCathodeModel  # Future implementation

__all__ = [
    'BaseCathodeModel',
    'PlatinumCathodeModel',
    # 'BiologicalCathodeModel',
]

__version__ = '1.0.0'
__author__ = 'MFC Simulation Team'