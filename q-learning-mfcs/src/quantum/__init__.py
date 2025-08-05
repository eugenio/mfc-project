"""
Quantum Computing Module for MFC Optimization

This module provides quantum computing capabilities for optimizing
Microbial Fuel Cell (MFC) parameters and control strategies.

Key Features:
- Quantum Approximate Optimization Algorithm (QAOA) for parameter optimization
- Variational Quantum Eigensolver (VQE) for energy state calculations
- Quantum Machine Learning for biofilm pattern recognition
- Hybrid quantum-classical algorithms for real-time control
"""

__version__ = "1.0.0"
__author__ = "TDD Agent 50"

# Import quantum modules
from .hybrid_algorithms import HybridQuantumClassicalController
from .quantum_circuits import CircuitValidator, QuantumCircuit, QuantumGate
from .quantum_ml import QuantumBiofilmClassifier, QuantumMLModel
from .quantum_optimization import QAOAOptimizer, QuantumParameterOptimizer, VQEOptimizer

__all__ = [
    "QuantumCircuit",
    "QuantumGate",
    "CircuitValidator",
    "QAOAOptimizer",
    "VQEOptimizer",
    "QuantumParameterOptimizer",
    "QuantumMLModel",
    "QuantumBiofilmClassifier",
    "HybridQuantumClassicalController"
]
