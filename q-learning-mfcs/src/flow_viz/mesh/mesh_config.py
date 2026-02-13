"""Mesh generation configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MeshConfig:
    """Configuration for tetrahedral mesh generation.

    Controls element sizes, boundary layer refinement,
    and output paths for the gmsh meshing pipeline.
    """

    # Global element size (metres)
    element_size_min: float = 0.0005  # 0.5 mm
    element_size_max: float = 0.002  # 2 mm

    # Boundary layer settings
    boundary_layer_thickness: float = 0.0002  # 0.2 mm first layer
    boundary_layer_ratio: float = 1.3
    boundary_layer_count: int = 5

    # Mesh algorithm (1=MeshAdapt, 5=Delaunay, 6=Frontal-Delaunay)
    mesh_algorithm_2d: int = 6
    mesh_algorithm_3d: int = 1  # Delaunay

    # Quality targets
    min_quality: float = 0.3  # minimum element quality (0-1)

    # Output
    output_dir: Path = Path("data/flow_viz/mesh")
    mesh_format: str = "msh"  # gmsh native format

    # STL repair options
    remove_duplicate_triangles: bool = True
    tolerance: float = 1e-8

    def stl_path(self, name: str) -> Path:
        """STL file path for a given component name."""
        return self.output_dir / f"{name}.stl"

    def mesh_path(self, name: str) -> Path:
        """Mesh file path for a given component name."""
        return self.output_dir / f"{name}.{self.mesh_format}"
