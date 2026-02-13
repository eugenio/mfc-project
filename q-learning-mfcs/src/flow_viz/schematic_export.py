"""CLI entry point: export MFC flow schematics to HTML.

Usage::

    python -m src.flow_viz.schematic_export [--cells N] [--circuit anode|cathode]
"""

from __future__ import annotations

import argparse

from cad.cad_config import StackCADConfig
from flow_viz.config import FlowVizConfig
from flow_viz.schematic_plot import export_schematic_html


def main(argv: list[str] | None = None) -> None:
    """CLI main for schematic export."""
    parser = argparse.ArgumentParser(
        description="Export MFC flow schematic to HTML",
    )
    parser.add_argument(
        "--cells", type=int, default=10,
        help="Number of cells (default: 10)",
    )
    parser.add_argument(
        "--circuit", choices=["anode", "cathode"],
        default="anode",
        help="Circuit to visualize (default: anode)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output HTML file path",
    )
    args = parser.parse_args(argv)

    cad_cfg = StackCADConfig(num_cells=args.cells)
    viz_cfg = FlowVizConfig()

    export_schematic_html(
        cad_cfg, viz_cfg,
        circuit=args.circuit,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
