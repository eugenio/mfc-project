"""Full MFC stack assembly.

Positions all components along the Z-axis (compression axis)
and returns a coloured ``cq.Assembly`` for visualisation and export.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cadquery as cq

from .cad_config import CathodeType, FlowConfiguration, StackCADConfig
from .components import (
    anode_frame,
    barb_fitting,
    cathode_frame,
    cathode_frame_gas,
    current_collector,
    electrode_placeholder,
    end_plate,
    manifold,
    membrane_gasket,
    port_label,
    pump_head,
    reservoir,
    support_feet,
    tie_rod,
    tubing,
)
from .hydraulics import compute_port_positions, compute_series_flow_path

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
    "support_foot": (0.5, 0.5, 0.5),
    "barb_fitting": (0.85, 0.85, 0.85),
    "tubing_anode": (0.9, 0.4, 0.3),
    "tubing_cathode": (0.3, 0.4, 0.9),
    "tubing_gas": (0.3, 0.8, 0.3),
    "manifold": (0.7, 0.7, 0.7),
    "reservoir": (0.3, 0.7, 0.9),
    "pump": (0.6, 0.6, 0.3),
    "label": (1.0, 1.0, 1.0),
}


def _mm(m: float) -> float:
    return m * 1000.0


class MFCStackAssembly:
    """Builder for the complete MFC stack assembly.

    Parameters
    ----------
    config : StackCADConfig
        Master parametric configuration.
    cathode_type : CathodeType
        ``LIQUID`` for all-liquid cathodes, ``GAS`` for all-gas.
    gas_cathode_cells : set[int] | None
        Legacy: explicit cell indices for gas cathodes.
        If ``cathode_type`` is set, this is derived automatically.
    include_supports : bool
        Add U-cradle support feet.
    include_labels : bool
        Add 3D port label plates.
    include_hydraulics : bool
        Add barb fittings and tubing.
    include_peripherals : bool
        Add reservoir and pump head.
    """

    def __init__(
        self,
        config: StackCADConfig,
        *,
        cathode_type: CathodeType = CathodeType.LIQUID,
        gas_cathode_cells: set[int] | None = None,
        include_supports: bool = False,
        include_labels: bool = False,
        include_hydraulics: bool = False,
        include_peripherals: bool = False,
    ) -> None:
        self.config = config
        self.cathode_type = cathode_type
        self.include_supports = include_supports
        self.include_labels = include_labels
        self.include_hydraulics = include_hydraulics
        self.include_peripherals = include_peripherals

        # Derive gas_cathode_cells from cathode_type if not explicit
        if gas_cathode_cells is not None:
            self.gas_cathode_cells: set[int] = gas_cathode_cells
        elif cathode_type == CathodeType.GAS:
            self.gas_cathode_cells = set(range(config.num_cells))
        else:
            self.gas_cathode_cells = set()

    @classmethod
    def all_liquid(cls, config: StackCADConfig, **kwargs) -> "MFCStackAssembly":
        """Factory for all-liquid cathode assembly."""
        return cls(config, cathode_type=CathodeType.LIQUID, **kwargs)

    @classmethod
    def all_gas(cls, config: StackCADConfig, **kwargs) -> "MFCStackAssembly":
        """Factory for all-gas cathode assembly."""
        return cls(config, cathode_type=CathodeType.GAS, **kwargs)

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

        # --- support feet ---
        if self.include_supports:
            self._add_supports(asm, cfg, z)

        # --- port labels ---
        if self.include_labels:
            self._add_labels(asm, cfg)

        # --- barb fittings + tubing ---
        if self.include_hydraulics:
            self._add_hydraulics(asm, cfg)

        # --- reservoir + pump ---
        if self.include_peripherals:
            self._add_peripherals(asm, cfg, z)

        return asm

    def _add_supports(
        self, asm: cq.Assembly, cfg: StackCADConfig, stack_z_end: float,
    ) -> None:
        """Add two U-cradle support feet near each end plate."""
        foot_solid = support_feet.build(cfg)
        ep = _mm(cfg.end_plate.thickness)
        half_outer = _mm(cfg.outer_side / 2)
        foot_h = _mm(cfg.support_feet.foot_height)

        # Foot near inlet end plate
        asm.add(
            foot_solid,
            name="support_foot_inlet",
            loc=cq.Location(cq.Vector(0, -half_outer - foot_h / 2, ep / 2)),
            color=cq.Color(*_COLOURS["support_foot"]),
        )
        # Foot near outlet end plate
        asm.add(
            foot_solid,
            name="support_foot_outlet",
            loc=cq.Location(cq.Vector(
                0,
                -half_outer - foot_h / 2,
                stack_z_end - ep / 2,
            )),
            color=cq.Color(*_COLOURS["support_foot"]),
        )

    def _add_labels(self, asm: cq.Assembly, cfg: StackCADConfig) -> None:
        """Add 3D text label plates near ports."""
        half_outer = _mm(cfg.outer_side / 2)
        ep = _mm(cfg.end_plate.thickness)
        label_offset = 5.0  # mm offset from surface

        label_specs = [
            ("AN IN", 0, -(half_outer + label_offset), ep + _mm(cfg.semi_cell.depth / 2)),
            ("AN OUT", 0, half_outer + label_offset, ep + _mm(cfg.semi_cell.depth / 2)),
            ("CA IN", -(half_outer + label_offset), 0, ep + _mm(cfg.cell_thickness * 0.75)),
            ("CA OUT", half_outer + label_offset, 0, ep + _mm(cfg.cell_thickness * 0.75)),
            ("IN", 0, 0, -label_offset),
            ("OUT", 0, 0, _mm(cfg.stack_length) + label_offset),
        ]

        for text, lx, ly, lz in label_specs:
            try:
                solid = port_label.build_label(text, cfg)
                asm.add(
                    solid,
                    name=f"label_{text.replace(' ', '_').replace('+', 'p').replace('-', 'm')}",
                    loc=cq.Location(cq.Vector(lx, ly, lz)),
                    color=cq.Color(*_COLOURS["label"]),
                )
            except Exception:
                pass  # Skip label if rendering fails

        # CC+/CC- labels near collector rod ends
        for sign, label_text in [(1, "CC+"), (-1, "CC-")]:
            try:
                solid = port_label.build_label(label_text, cfg)
                asm.add(
                    solid,
                    name=f"label_{label_text.replace('+', 'p').replace('-', 'm')}",
                    loc=cq.Location(cq.Vector(
                        sign * (half_outer + label_offset),
                        0,
                        _mm(cfg.stack_length / 2),
                    )),
                    color=cq.Color(*_COLOURS["label"]),
                )
            except Exception:
                pass

    def _add_hydraulics(self, asm: cq.Assembly, cfg: StackCADConfig) -> None:
        """Add barb fittings on all ports and tubing between cells."""
        # Barb fittings on every port
        ports = compute_port_positions(cfg)
        fitting_solid = barb_fitting.build(cfg)
        fitting_len = _mm(
            cfg.barb_fitting.thread_length
            + cfg.barb_fitting.hex_height
            + cfg.barb_fitting.barb_length,
        )

        for p in ports:
            if "end_plate" in p.label:
                continue  # Skip end plate flow ports for fittings
            nx, ny, nz = p.normal
            asm.add(
                fitting_solid,
                name=f"fitting_{p.label}",
                loc=cq.Location(cq.Vector(p.x, p.y, p.z)),
                color=cq.Color(*_COLOURS["barb_fitting"]),
            )

        # Series U-tube tubing between adjacent cells (anode circuit)
        if cfg.flow_config == FlowConfiguration.SERIES:
            self._add_series_tubing(asm, cfg, ports)
        else:
            self._add_parallel_manifolds(asm, cfg, ports)

    def _add_series_tubing(
        self,
        asm: cq.Assembly,
        cfg: StackCADConfig,
        ports: list,
    ) -> None:
        """Add series U-tube segments between adjacent cells."""
        clearance_mm = _mm(cfg.utube_clearance)

        for circuit, colour_key in [("anode", "tubing_anode"), ("cathode", "tubing_cathode")]:
            # Get outlet/inlet ports sorted by Z
            outlets = sorted(
                [p for p in ports if circuit in p.label and "outlet" in p.label],
                key=lambda p: p.z,
            )
            inlets = sorted(
                [p for p in ports if circuit in p.label and "inlet" in p.label],
                key=lambda p: p.z,
            )
            for i in range(len(outlets) - 1):
                try:
                    out_p = outlets[i]
                    in_p = inlets[i + 1]
                    solid, _ = tubing.build_utube(
                        port_a=(out_p.x, out_p.y, out_p.z),
                        port_b=(in_p.x, in_p.y, in_p.z),
                        clearance_mm=clearance_mm,
                        normal=out_p.normal,
                        config=cfg,
                    )
                    asm.add(
                        solid,
                        name=f"utube_{circuit}_{i}",
                        loc=cq.Location(cq.Vector(0, 0, 0)),
                        color=cq.Color(*_COLOURS[colour_key]),
                    )
                except Exception:
                    pass  # Skip segment if geometry fails

    def _add_parallel_manifolds(
        self,
        asm: cq.Assembly,
        cfg: StackCADConfig,
        ports: list,
    ) -> None:
        """Add parallel-flow manifold headers."""
        standoff = _mm(cfg.manifold_standoff)
        half_outer = _mm(cfg.outer_side / 2)
        header_solid = manifold.build_header(cfg)

        # Supply manifold (anode inlet side, -Y)
        asm.add(
            header_solid,
            name="manifold_anode_supply",
            loc=cq.Location(cq.Vector(0, -(half_outer + standoff), 0)),
            color=cq.Color(*_COLOURS["manifold"]),
        )
        # Return manifold (anode outlet side, +Y)
        asm.add(
            header_solid,
            name="manifold_anode_return",
            loc=cq.Location(cq.Vector(0, half_outer + standoff, 0)),
            color=cq.Color(*_COLOURS["manifold"]),
        )
        # Cathode supply (-X)
        asm.add(
            header_solid,
            name="manifold_cathode_supply",
            loc=cq.Location(cq.Vector(-(half_outer + standoff), 0, 0)),
            color=cq.Color(*_COLOURS["manifold"]),
        )
        # Cathode return (+X)
        asm.add(
            header_solid,
            name="manifold_cathode_return",
            loc=cq.Location(cq.Vector(half_outer + standoff, 0, 0)),
            color=cq.Color(*_COLOURS["manifold"]),
        )

    def _add_peripherals(
        self, asm: cq.Assembly, cfg: StackCADConfig, stack_z_end: float,
    ) -> None:
        """Add reservoir and pump head beside the stack."""
        half_outer = _mm(cfg.outer_side / 2)
        res_od = _mm(cfg.reservoir.outer_diameter)

        # Reservoir placed beside stack (+Y, centred along Z)
        asm.add(
            reservoir.build(cfg),
            name="reservoir",
            loc=cq.Location(cq.Vector(
                0,
                half_outer + res_od / 2 + 50,  # 50 mm gap
                stack_z_end / 2 - _mm(cfg.reservoir.outer_height) / 2,
            )),
            color=cq.Color(*_COLOURS["reservoir"]),
        )

        # Pump head placed beside reservoir
        asm.add(
            pump_head.build(cfg),
            name="pump_head",
            loc=cq.Location(cq.Vector(
                0,
                half_outer + res_od + 100,  # further out
                stack_z_end / 2,
            )),
            color=cq.Color(*_COLOURS["pump"]),
        )

    @property
    def expected_part_count(self) -> int:
        """Expected number of named parts in the assembly."""
        n = self.config.num_cells
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

        if self.include_supports:
            parts += 2  # two support feet

        if self.include_labels:
            parts += 8  # up to 8 label plates (AN IN/OUT, CA IN/OUT, IN, OUT, CC+, CC-)

        if self.include_hydraulics:
            # Barb fittings: 4 per cell (anode in/out + cathode in/out)
            parts += n * 4
            if self.config.flow_config == FlowConfiguration.SERIES:
                # U-tubes: (n-1) per circuit × 2 circuits
                parts += (n - 1) * 2
            else:
                # 4 manifold headers
                parts += 4

        if self.include_peripherals:
            parts += 2  # reservoir + pump head

        return parts
