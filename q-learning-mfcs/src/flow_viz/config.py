"""Flow visualization configuration.

Central dataclass controlling colormaps, rendering options,
and output format for all flow-viz modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class OutputFormat(Enum):
    """Supported output formats for visualizations."""

    HTML = "html"
    PNG = "png"
    SVG = "svg"


class ColormapPreset(Enum):
    """Named colormap presets for flow variables."""

    VELOCITY = "coolwarm"
    PRESSURE = "viridis"
    STREAMLINE = "plasma"


@dataclass(frozen=True)
class FlowVizConfig:
    """Configuration for hydraulic flow visualization.

    Controls rendering parameters, colormaps, and output paths
    shared across schematic, mesh, and render modules.
    """

    # Fluid properties (MFC anolyte, not water)
    fluid_density: float = 1020.0  # kg/m^3 (dilute wastewater)
    fluid_viscosity: float = 1.1e-3  # Pa.s (slightly above water)
    fluid_name: str = "MFC Anolyte"

    # Arrow / annotation scaling for schematics
    arrow_scale: float = 1.0
    node_size: float = 10.0
    pipe_width_scale: float = 2.0

    # Colormap settings
    velocity_cmap: str = ColormapPreset.VELOCITY.value
    pressure_cmap: str = ColormapPreset.PRESSURE.value
    streamline_cmap: str = ColormapPreset.STREAMLINE.value

    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("data/flow_viz"))
    output_format: OutputFormat = OutputFormat.HTML
    dpi: int = 150

    # 3D rendering
    background_color: str = "#1a1a2e"
    mesh_opacity: float = 0.3
    streamline_tube_radius: float = 0.5  # mm
    decimation_target: int = 50000  # max triangles for HTML export

    def output_path(self, name: str) -> Path:
        """Build an output file path for the given name."""
        ext = self.output_format.value
        return self.output_dir / f"{name}.{ext}"
