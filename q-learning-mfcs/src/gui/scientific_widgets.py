#!/usr/bin/env python3
"""
Scientific Parameter Widgets for Enhanced MFC Platform

Provides scientific parameter input widgets with real-time validation
and literature-backed feedback for electrode configuration.

Created: 2025-08-02
"""

import streamlit as st
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class ParameterSpec:
    """Scientific parameter specification with validation."""
    name: str
    unit: str
    min_value: float
    max_value: float
    typical_range: Tuple[float, float]
    literature_refs: str
    description: str
    

class ScientificParameterWidget:
    """Scientific parameter input widget with validation."""
    
    def __init__(self, param_spec: ParameterSpec, key: str):
        self.param_spec = param_spec
        self.key = key
    
    def render(self, label: str, value: float) -> float:
        """Render parameter input with validation."""
        
        # Input widget
        input_value = st.number_input(
            label,
            min_value=self.param_spec.min_value,
            max_value=self.param_spec.max_value,
            value=value,
            key=self.key,
            help=f"{self.param_spec.description} ({self.param_spec.unit})"
        )
        
        # Validation feedback
        self._show_validation_feedback(input_value)
        
        return input_value
    
    def _show_validation_feedback(self, value: float):
        """Show real-time validation feedback."""
        
        min_typical, max_typical = self.param_spec.typical_range
        
        if min_typical <= value <= max_typical:
            st.success(f"✅ Typical range ({self.param_spec.unit})")
        elif value < min_typical:
            st.warning(f"⚠️ Below typical range ({min_typical}-{max_typical} {self.param_spec.unit})")
        else:
            st.warning(f"⚠️ Above typical range ({min_typical}-{max_typical} {self.param_spec.unit})")
        
        # Literature reference
        if self.param_spec.literature_refs:
            with st.expander("📚 Literature Reference"):
                st.info(self.param_spec.literature_refs)


def create_parameter_section(title: str, parameters: Dict[str, Any]) -> Dict[str, float]:
    """Create a section of scientific parameters."""
    
    st.subheader(title)
    values = {}
    
    for param_name, param_config in parameters.items():
        widget = ScientificParameterWidget(param_config["spec"], f"{title}_{param_name}")
        values[param_name] = widget.render(
            param_config["label"],
            param_config["default"]
        )
    
    return values


# MFC Electrode Parameter Specifications
MFC_ELECTRODE_PARAMETERS = {
    "conductivity": ParameterSpec(
        name="Electrical Conductivity",
        unit="S/m",
        min_value=0.1,
        max_value=10000000.0,
        typical_range=(100.0, 100000.0),
        literature_refs="Logan, B.E. (2008). Microbial Fuel Cells: Methodology and Technology",
        description="Electrical conductivity of electrode material"
    )
}