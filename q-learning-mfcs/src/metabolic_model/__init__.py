"""
Metabolic Model Module

This module implements metabolic pathway models for exoelectrogenic bacteria
including acetate and lactate metabolism, electron shuttle production,
and oxygen crossover calculations through Nafion membranes.

Key Features:
- Species-specific metabolic pathways (G. sulfurreducens, S. oneidensis)
- Substrate metabolism (acetate, lactate) with stoichiometric accuracy
- Electron shuttle and cytochrome modeling
- Nafion membrane oxygen crossover calculations
- GPU acceleration support
- KEGG pathway integration
"""

from .metabolic_core import MetabolicModel
from .pathway_database import PathwayDatabase
from .membrane_transport import MembraneTransport
from .electron_shuttles import ElectronShuttleModel

__version__ = "1.0.0"
__author__ = "MFC Simulation Project"

__all__ = [
    'MetabolicModel',
    'PathwayDatabase',
    'MembraneTransport',
    'ElectronShuttleModel'
]