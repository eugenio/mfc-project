"""
API models for MFC monitoring system
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SystemStatus(BaseModel):
    """System status response"""
    status: str = Field(..., description="System status: running, stopped, error")
    timestamp: datetime = Field(default_factory=datetime.now)
    message: str | None = None
    details: dict[str, Any] | None = None


class SystemMetrics(BaseModel):
    """System metrics response"""
    timestamp: str
    status: str
    uptime_hours: float
    power_output_w: float
    efficiency_pct: float
    temperature_c: float
    ph_level: float
    voltage_v: float | None = None
    current_a: float | None = None
    substrate_concentration: float | None = None


class AlertMessage(BaseModel):
    """Alert message model"""
    timestamp: datetime = Field(default_factory=datetime.now)
    level: str = Field(..., description="Alert level: info, warning, error, critical")
    message: str
    source: str | None = None
    details: dict[str, Any] | None = None
