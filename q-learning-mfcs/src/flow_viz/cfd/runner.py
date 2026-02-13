"""Execute OpenFOAM solver."""

from __future__ import annotations

from pathlib import Path

from flow_viz.cfd.solver_config import SolverConfig


def build_solver_command(
    case_dir: Path | str,
    solver_config: SolverConfig | None = None,
) -> list[str]:
    """Build the solver command line arguments.

    Returns a list of strings suitable for subprocess.run.
    """
    if solver_config is None:
        solver_config = SolverConfig()
    case_dir = Path(case_dir)
    app = solver_config.solver.value
    if solver_config.num_processors > 1:
        return [
            "mpirun", "-np",
            str(solver_config.num_processors),
            app, "-parallel",
            "-case", str(case_dir),
        ]
    return [app, "-case", str(case_dir)]


def check_convergence(
    case_dir: Path | str,
    log_file: str = "log.solver",
    target_residual: float = 1e-4,
) -> bool:
    """Check if the solver converged by parsing the log."""
    case_dir = Path(case_dir)
    log_path = case_dir / log_file
    if not log_path.exists():
        return False
    text = log_path.read_text()
    if "solution converged" in text.lower():
        return True
    lines = text.strip().splitlines()
    for line in reversed(lines):
        if "Final residual" in line:
            parts = line.split("Final residual = ")
            if len(parts) > 1:
                try:
                    res = float(parts[1].split(",")[0])
                    if res < target_residual:
                        return True
                except ValueError:
                    continue
            break
    return False
