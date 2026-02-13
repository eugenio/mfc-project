"""Generate OpenFOAM case directory structure."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from flow_viz.cfd.boundary_conditions import (
    BoundaryConditions,
    compute_boundary_conditions,
)
from flow_viz.cfd.solver_config import SolverConfig
from flow_viz.config import FlowVizConfig

if TYPE_CHECKING:
    from cad.cad_config import StackCADConfig


def create_case_directory(
    case_dir: Path | str,
    cad_config: StackCADConfig,
    viz_config: FlowVizConfig | None = None,
    solver_config: SolverConfig | None = None,
) -> Path:
    """Create an OpenFOAM case directory with 0/, constant/, system/.

    Parameters
    ----------
    case_dir : Path
        Root directory for the OpenFOAM case.
    cad_config : StackCADConfig
        MFC stack configuration.
    viz_config : FlowVizConfig, optional
        Flow visualization config.
    solver_config : SolverConfig, optional
        Solver settings.

    Returns
    -------
    Path
        Path to the created case directory.

    """
    if viz_config is None:
        viz_config = FlowVizConfig()
    if solver_config is None:
        solver_config = SolverConfig()

    case_dir = Path(case_dir)
    bc = compute_boundary_conditions(cad_config, viz_config)

    _write_0_dir(case_dir / "0", bc)
    _write_constant_dir(case_dir / "constant", bc)
    _write_system_dir(case_dir / "system", solver_config)

    return case_dir


def _write_0_dir(dir_path: Path, bc: BoundaryConditions) -> None:
    """Write initial condition files (U, p)."""
    dir_path.mkdir(parents=True, exist_ok=True)

    u_content = _foam_header("volVectorField", "U")
    u_content += f"""
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);
boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform (0 0 {bc.inlet_velocity:.6e});
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    walls
    {{
        type            {bc.wall_type};
    }}
}}
"""
    (dir_path / "U").write_text(u_content)

    p_content = _foam_header("volScalarField", "p")
    p_content += f"""
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform {bc.outlet_pressure};
boundaryField
{{
    inlet
    {{
        type            zeroGradient;
    }}
    outlet
    {{
        type            fixedValue;
        value           uniform {bc.outlet_pressure};
    }}
    walls
    {{
        type            zeroGradient;
    }}
}}
"""
    (dir_path / "p").write_text(p_content)


def _write_constant_dir(dir_path: Path, bc: BoundaryConditions) -> None:
    """Write transportProperties."""
    dir_path.mkdir(parents=True, exist_ok=True)

    tp = _foam_header("dictionary", "transportProperties")
    tp += f"""
transportModel  Newtonian;
nu              [0 2 -1 0 0 0 0] {bc.kinematic_viscosity:.6e};
"""
    (dir_path / "transportProperties").write_text(tp)


def _write_system_dir(
    dir_path: Path, solver_config: SolverConfig,
) -> None:
    """Write controlDict, fvSchemes, fvSolution."""
    dir_path.mkdir(parents=True, exist_ok=True)

    # controlDict
    app = solver_config.solver.value
    cd = _foam_header("dictionary", "controlDict")
    cd += f"""
application     {app};
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {solver_config.end_time};
deltaT          {solver_config.delta_t};
writeControl    timeStep;
writeInterval   {solver_config.write_interval};
purgeWrite      3;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""
    (dir_path / "controlDict").write_text(cd)

    # fvSchemes
    fs = _foam_header("dictionary", "fvSchemes")
    fs += f"""
ddtSchemes
{{
    default         steadyState;
}}
gradSchemes
{{
    default         {solver_config.grad_scheme};
}}
divSchemes
{{
    default         none;
    div(phi,U)      {solver_config.div_scheme};
}}
laplacianSchemes
{{
    default         {solver_config.laplacian_scheme};
}}
interpolationSchemes
{{
    default         linear;
}}
snGradSchemes
{{
    default         corrected;
}}
"""
    (dir_path / "fvSchemes").write_text(fs)

    # fvSolution
    fv = _foam_header("dictionary", "fvSolution")
    fv += f"""
solvers
{{
    p
    {{
        solver          GAMG;
        tolerance       {solver_config.residual_target};
        relTol          0.01;
        smoother        GaussSeidel;
    }}
    U
    {{
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       {solver_config.residual_target};
        relTol          0.1;
    }}
}}
SIMPLE
{{
    nNonOrthogonalCorrectors 1;
    residualControl
    {{
        p               {solver_config.residual_target};
        U               {solver_config.residual_target};
    }}
}}
relaxationFactors
{{
    fields
    {{
        p               {solver_config.relaxation_p};
    }}
    equations
    {{
        U               {solver_config.relaxation_u};
    }}
}}
"""
    (dir_path / "fvSolution").write_text(fv)


def _foam_header(class_name: str, object_name: str) -> str:
    """Generate FoamFile header."""
    return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       {class_name};
    object      {object_name};
}}
"""
