"""Cross-Phase Integration Module.

Provides seamless integration between all MFC system phases.
"""

from .cross_phase_integrator import (
    CrossPhaseIntegrator,
    IntegrationConfig,
    IntegrationResult,
)

__all__ = ["CrossPhaseIntegrator", "IntegrationConfig", "IntegrationResult"]
