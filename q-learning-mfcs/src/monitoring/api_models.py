"""
API models for MFC monitoring system
"""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class SystemStatus(BaseModel):
    """System status response"""
    status: str = Field(..., description="System status: running, stopped, error")
    timestamp: datetime = Field(default_factory=datetime.now)
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class SystemMetrics(BaseModel):
    """System metrics response"""
    timestamp: str
    status: str
    uptime_hours: float
    power_output_w: float
    efficiency_pct: float
    temperature_c: float
    ph_level: float
    voltage_v: Optional[float] = None
    current_a: Optional[float] = None
    substrate_concentration: Optional[float] = None


class AlertMessage(BaseModel):
    """Alert message model"""
    timestamp: datetime = Field(default_factory=datetime.now)
    level: str = Field(..., description="Alert level: info, warning, error, critical")
    message: str
    source: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
