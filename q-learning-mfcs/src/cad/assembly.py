"""Full MFC stack assembly.

Positions all components along the Z-axis (compression axis)
and returns a coloured ``cq.Assembly`` for visualisation and export.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cadquery as cq

from .cad_config import StackCADConfig
from .components import (
    anode_frame,
    cathode_frame,
    cathode_frame_gas,
    current_collector,
    electrode_placeholder,
    end_plate,
    membrane_gasket,
    tie_rod,
)

if TYPE_CHECKING:
    pass

# Colour palette  (name -> RGB floats 0-1 for cq.Color)
_COLOURS: dict[str, tuple[float, float, float]] = {
    "anode_frame": (0.8, 0.2, 0.2),  # red
    "cathode_frame": (0.2, 0.4, 0.8),  # blue
    "gas_cathode": (0.2, 0.67, 0.33),  # green
    "membrane": (0.87, 0.87, 0.2),  # yellow
    "electrode_anode": (0.33, 0.33, 0.33),  # dark grey
    "electrode_cathode": (0.4, 0.4, 0.4),  # slightly lighter
    "end_plate": (0.67, 0.67, 0.67),  # grey
    "tie_rod": (0.27, 0.27, 0.27),  # dark
    "nut": (0.33, 0.33, 0.33),
    "washer": (0.47, 0.47, 0.47),
    "collector": (0.75, 0.75, 0.75),  # silver
}


def _mm(m: float) -> float:
    return m * 1000.0


class MFCStackAssembly:
    """Builder for the complete MFC stack assembly."""

    def __init__(
        self,
        config: StackCADConfig,
        *,
        gas_cathode_cells: set[int] | None = None,
    ) -> None:
        self.config = config
        # Cell indices (0-based) that use the gas cathode variant
        self.gas_cathode_cells: set[int] = gas_cathode_cells or set()

    def build(self) -> cq.Assembly:
        """Assemble all components and return a ``cq.Assembly``."""
        asm = cq.Assembly()
        cfg = self.config
        z = 0.0  # current Z offset in mm

        # --- inlet end plate ---
        ep_thickness = _mm(cfg.end_plate.thickness)
        asm.add(
            end_plate.build(cfg, is_inlet=True),
            name="inlet_end_plate",
            loc=cq.Location(cq.Vector(0, 0, z + ep_thickness / 2)),
            color=cq.Color(*_COLOURS["end_plate"]),
        )
        z += ep_thickness

        # --- repeating cell units ---
        for i in range(cfg.num_cells):
            anode_depth = _mm(cfg.semi_cell.depth)
            gasket_thick = _mm(cfg.gasket_membrane_thickness)

            # Anode frame
            asm.add(
                anode_frame.build(cfg),
                name=f"anode_frame_{i}",
                loc=cq.Location(cq.Vector(0, 0, z + anode_depth / 2)),
                color=cq.Color(*_COLOURS["anode_frame"]),
            )

            # Anode electrode placeholder
            elec_thick = _mm(cfg.electrode.thickness)
            asm.add(
                electrode_placeholder.build(cfg),
                name=f"anode_electrode_{i}",
                loc=cq.Location(cq.Vector(0, 0, z + anode_depth / 2)),
                color=cq.Color(*_COLOURS["electrode_anode"]),
            )
            z += anode_depth

            # Membrane gasket
            asm.add(
                membrane_gasket.build(cfg),
                name=f"membrane_gasket_{i}",
                loc=cq.Location(cq.Vector(0, 0, z + gasket_thick / 2)),
                color=cq.Color(*_COLOURS["membrane"]),
            )
            z += gasket_thick

            # Cathode frame (liquid or gas variant)
            if i in self.gas_cathode_cells:
                cathode_depth = _mm(
                    cfg.semi_cell.depth + cfg.gas_cathode.headspace_depth,
                )
                asm.add(
                    cathode_frame_gas.build(cfg),
                    name=f"gas_cathode_frame_{i}",
                    loc=cq.Location(cq.Vector(0, 0, z + cathode_depth / 2)),
                    color=cq.Color(*_COLOURS["gas_cathode"]),
                )
            else:
                cathode_depth = _mm(cfg.semi_cell.depth)
                asm.add(
                    cathode_frame.build(cfg),
                    name=f"cathode_frame_{i}",
                    loc=cq.Location(cq.Vector(0, 0, z + cathode_depth / 2)),
                    color=cq.Color(*_COLOURS["cathode_frame"]),
                )

            # Cathode electrode placeholder
            asm.add(
                electrode_placeholder.build(cfg),
                name=f"cathode_electrode_{i}",
                loc=cq.Location(cq.Vector(0, 0, z + cathode_depth / 2)),
                color=cq.Color(*_COLOURS["electrode_cathode"]),
            )
            z += cathode_depth

        # --- outlet end plate ---
        asm.add(
            end_plate.build(cfg, is_inlet=False),
            name="outlet_end_plate",
            loc=cq.Location(cq.Vector(0, 0, z + ep_thickness / 2)),
            color=cq.Color(*_COLOURS["end_plate"]),
        )
        z += ep_thickness

        # --- tie rods + nuts + washers ---
        rod_solid = tie_rod.build_rod(cfg)
        nut_solid = tie_rod.build_nut(cfg)
        washer_solid = tie_rod.build_washer(cfg)

        for idx, (x, y) in enumerate(cfg.tie_rod_positions):
            xm = _mm(x)
            ym = _mm(y)
            asm.add(
                rod_solid,
                name=f"tie_rod_{idx}",
                loc=cq.Location(cq.Vector(xm, ym, 0)),
                color=cq.Color(*_COLOURS["tie_rod"]),
            )
            # Bottom nut + washer
            asm.add(
                washer_solid,
                name=f"washer_bot_{idx}",
                loc=cq.Location(
                    cq.Vector(xm, ym, -_mm(cfg.tie_rod.washer_thickness)),
                ),
                color=cq.Color(*_COLOURS["washer"]),
            )
            asm.add(
                nut_solid,
                name=f"nut_bot_{idx}",
                loc=cq.Location(
                    cq.Vector(
                        xm,
                        ym,
                        -(
                            _mm(cfg.tie_rod.washer_thickness)
                            + _mm(cfg.tie_rod.nut_height)
                        ),
                    ),
                ),
                color=cq.Color(*_COLOURS["nut"]),
            )
            # Top nut + washer
            asm.add(
                washer_solid,
                name=f"washer_top_{idx}",
                loc=cq.Location(cq.Vector(xm, ym, z)),
                color=cq.Color(*_COLOURS["washer"]),
            )
            asm.add(
                nut_solid,
                name=f"nut_top_{idx}",
                loc=cq.Location(
                    cq.Vector(xm, ym, z + _mm(cfg.tie_rod.washer_thickness)),
                ),
                color=cq.Color(*_COLOURS["nut"]),
            )

        # --- current-collector rods ---
        collector_solid = current_collector.build(cfg)
        for cell_i in range(cfg.num_cells):
            for rod_j, (cx, cy) in enumerate(cfg.collector_positions):
                asm.add(
                    collector_solid,
                    name=f"collector_{cell_i}_{rod_j}",
                    loc=cq.Location(cq.Vector(_mm(cx), _mm(cy), 0)),
                    color=cq.Color(*_COLOURS["collector"]),
                )

        return asm

    @property
    def expected_part_count(self) -> int:
        """Expected number of named parts in the assembly."""
        n = self.config.num_cells
        n_gas = len(self.gas_cathode_cells)
        n_collectors = self.config.current_collector.count_per_electrode

        parts = 0
        parts += 2  # end plates
        parts += n  # anode frames
        parts += n  # anode electrodes
        parts += n  # membrane gaskets
        parts += n  # cathode frames (liquid or gas)
        parts += n  # cathode electrodes
        parts += 4  # tie rods
        parts += 8  # nuts (top + bottom × 4)
        parts += 8  # washers (top + bottom × 4)
        parts += n * n_collectors  # current collector rods
        return parts
