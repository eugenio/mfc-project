"""Simplified MFC Real-time Monitoring Dashboard API.

A minimal but functional monitoring API for MFC systems.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class SystemMetrics(BaseModel):
    """System metrics response."""

    timestamp: str
    status: str
    uptime_hours: float
    power_output_w: float
    efficiency_pct: float
    temperature_c: float
    ph_level: float
    pressure_bar: float
    flow_rate_ml_min: float
    cell_voltages: list[float]


class ControlCommand(BaseModel):
    """Control command request."""

    command: str
    parameters: dict[str, Any] | None = None
