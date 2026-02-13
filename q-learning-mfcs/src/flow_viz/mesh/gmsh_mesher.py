"""STL -> tetrahedral volume mesh using gmsh.

Runs in the **flow-viz** environment (Python 3.12, gmsh available).
Reads STL files produced by ``stl_export.py`` and generates
volume meshes suitable for OpenFOAM or PyVista rendering.
"""

from __future__ import annotations

import math
from pathlib import Path

from flow_viz.mesh.mesh_config import MeshConfig

GMSH_TETRAHEDRON = 4


def mesh_stl(
    stl_path: Path | str,
    output_path: Path | str | None = None,
    config: MeshConfig | None = None,
) -> Path:
    """Generate a tetrahedral volume mesh from an STL file.

    Parameters
    ----------
    stl_path : Path
        Input STL file.
    output_path : Path, optional
        Output mesh file. Defaults to same name with .msh extension.
    config : MeshConfig, optional
        Mesh generation settings. Uses defaults if None.

    Returns
    -------
    Path
        Path to the generated mesh file.

    """
    import gmsh  # noqa: PLC0415

    if config is None:
        config = MeshConfig()

    stl_path = Path(stl_path)
    if output_path is None:
        output_path = stl_path.with_suffix(f".{config.mesh_format}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)

        # STL repair options
        if config.remove_duplicate_triangles:
            gmsh.option.setNumber(
                "Mesh.StlRemoveDuplicateTriangles", 1,
            )

        # Merge STL as triangulation
        gmsh.merge(str(stl_path))

        # Classify surface mesh into geometric surfaces
        angle = math.pi / 4
        gmsh.model.mesh.classifySurfaces(
            angle, True, False, config.tolerance,  # noqa: FBT003
        )

        # Create geometry from the classified mesh
        gmsh.model.mesh.createGeometry()

        # Get all surfaces and build a volume
        surfaces = gmsh.model.getEntities(dim=2)
        if surfaces:
            surface_tags = [tag for _, tag in surfaces]
            sl = gmsh.model.geo.addSurfaceLoop(surface_tags)
            gmsh.model.geo.addVolume([sl])
            gmsh.model.geo.synchronize()

        # Mesh size settings (config values in metres, STL in mm)
        gmsh.option.setNumber(
            "Mesh.CharacteristicLengthMin",
            config.element_size_min * 1000,
        )
        gmsh.option.setNumber(
            "Mesh.CharacteristicLengthMax",
            config.element_size_max * 1000,
        )
        gmsh.option.setNumber(
            "Mesh.Algorithm", config.mesh_algorithm_2d,
        )
        gmsh.option.setNumber(
            "Mesh.Algorithm3D", config.mesh_algorithm_3d,
        )
        gmsh.option.setNumber(
            "Mesh.QualityType", 2,
        )

        # Generate 3D mesh
        gmsh.model.mesh.generate(3)

        # Write output
        gmsh.write(str(output_path))
    finally:
        gmsh.finalize()

    return output_path


def get_mesh_stats(mesh_path: Path | str) -> dict:
    """Read basic mesh statistics from a gmsh file.

    Returns
    -------
    dict
        Keys: num_nodes, num_elements, num_tetrahedra

    """
    import gmsh  # noqa: PLC0415

    mesh_path = Path(mesh_path)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(mesh_path))

        nodes = gmsh.model.mesh.getNodes()
        num_nodes = len(nodes[0])

        # Count element types
        elem_types, elem_tags, _ = gmsh.model.mesh.getElements()
        num_elements = 0
        num_tets = 0
        for i, etype in enumerate(elem_types):
            count = len(elem_tags[i])
            num_elements += count
            if etype == GMSH_TETRAHEDRON:
                num_tets = count
    finally:
        gmsh.finalize()

    return {
        "num_nodes": num_nodes,
        "num_elements": num_elements,
        "num_tetrahedra": num_tets,
    }
