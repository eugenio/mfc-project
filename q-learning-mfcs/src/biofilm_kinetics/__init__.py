"""
Biofilm Kinetics Module

This module implements biofilm formation kinetics for exoelectrogenic bacteria
in microbial fuel cells, supporting G. sulfurreducens, S. oneidensis MR-1,
and mixed cultures with pH and temperature compensation.

Key Features:
- Species-specific kinetic parameters
- Substrate selection (acetate, lactate)
- pH and temperature compensation
- Mixed culture synergy modeling
- GPU acceleration support
"""

from .biofilm_model import BiofilmKineticsModel
from .species_params import SpeciesParameters
from .substrate_params import SubstrateParameters

__version__ = "1.0.0"
__author__ = "MFC Simulation Project"

__all__ = [
    'BiofilmKineticsModel',
    'SpeciesParameters',
    'SubstrateParameters'
]
