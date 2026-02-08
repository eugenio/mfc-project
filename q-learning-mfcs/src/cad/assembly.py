"""Full MFC stack assembly.

Positions all components along the Z-axis (compression axis)
and returns a coloured ``cq.Assembly`` for visualisation and export.
"""

from __future__ import annotations

from typing import Any

import cadquery as cq

from .cad_config import (
    CathodeType,
    FlowConfiguration,
    ReservoirRole,
    StackCADConfig,
)
from .components import (
    anode_frame,
    barb_fitting,
    cathode_frame,
    cathode_frame_gas,
    conical_bottom,
    current_collector,
    electrode_placeholder,
    electrovalve,
    end_plate,
    gas_diffusion,
    manifold,
    membrane_gasket,
    port_label,
    pump_head,
    pump_support,
    reservoir,
    reservoir_feet,
    reservoir_lid,
    stirring_motor,
    support_feet,
    tie_rod,
    tubing,
)
from .hydraulics import compute_port_positions

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
    "reservoir_anolyte": (0.3, 0.7, 0.9),
    "reservoir_catholyte": (0.4, 0.6, 0.8),
    "reservoir_nutrient": (0.6, 0.8, 0.4),
    "reservoir_buffer": (0.8, 0.6, 0.4),
    "reservoir_lid": (0.5, 0.5, 0.55),
    "conical_bottom": (0.35, 0.65, 0.85),
    "reservoir_feet": (0.45, 0.45, 0.45),
    "stirring_motor": (0.4, 0.4, 0.5),
    "gas_diffuser": (0.7, 0.7, 0.8),
    "electrovalve": (0.55, 0.55, 0.6),
    "pump_support": (0.5, 0.5, 0.5),
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
    def all_liquid(cls, config: StackCADConfig, **kwargs: Any) -> MFCStackAssembly:
        """Factory for all-liquid cathode assembly."""
        return cls(config, cathode_type=CathodeType.LIQUID, **kwargs)

    @classmethod
    def all_gas(cls, config: StackCADConfig, **kwargs: Any) -> MFCStackAssembly:
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
            for rod_j, (cx, cy) in enumerate(cfg.anode_collector_positions):
                asm.add(
                    collector_solid,
                    name=f"collector_anode_{cell_i}_{rod_j}",
                    loc=cq.Location(cq.Vector(_mm(cx), _mm(cy), 0)),
                    color=cq.Color(*_COLOURS["collector"]),
                )
            for rod_j, (cx, cy) in enumerate(cfg.cathode_collector_positions):
                asm.add(
                    collector_solid,
                    name=f"collector_cathode_{cell_i}_{rod_j}",
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
        gap = 5.0  # mm gap between stack face and label plate edge

        def _label_offset(text: str) -> float:
            """Dynamic offset: half plate width + gap."""
            pw, _ = port_label.plate_dimensions(text, cfg)
            return pw / 2 + gap

        an_z = ep + _mm(cfg.semi_cell.depth / 2)
        ca_z = ep + _mm(cfg.cell_thickness * 0.75)
        label_specs = [
            ("AN IN", -(half_outer + _label_offset("AN IN")), 0, an_z),
            ("AN OUT", half_outer + _label_offset("AN OUT"), 0, an_z),
            ("CA IN", 0, -(half_outer + _label_offset("CA IN")), ca_z),
            ("CA OUT", 0, half_outer + _label_offset("CA OUT"), ca_z),
            ("IN", 0, 0, -_label_offset("IN")),
            ("OUT", 0, 0, _mm(cfg.stack_length) + _label_offset("OUT")),
        ]

        for text, lx, ly, lz in label_specs:
            try:
                solid = port_label.build_label(text, cfg)
                asm.add(
                    solid,
                    name=f"label_{text.replace(' ', '_').replace('+', 'p').replace('-', 'm')}",  # noqa: E501
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
                        sign * (half_outer + _label_offset(label_text)),
                        0,
                        _mm(cfg.stack_length / 2),
                    )),
                    color=cq.Color(*_COLOURS["label"]),
                )
            except Exception:
                pass

    def _add_hydraulics(self, asm: cq.Assembly, cfg: StackCADConfig) -> None:
        """Add barb fittings on all ports and tubing between cells."""
        # Barb fittings on every port, oriented along port normal
        ports = compute_port_positions(cfg)

        for p in ports:
            if "end_plate" in p.label:
                continue  # Skip end plate flow ports for fittings
            nx, ny, nz = p.normal
            oriented = barb_fitting.build_oriented(cfg, p.normal)
            asm.add(
                oriented,
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

        circuits = [("anode", "tubing_anode"), ("cathode", "tubing_cathode")]
        for circuit, colour_key in circuits:
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

        # Supply manifold (anode inlet side, -X)
        asm.add(
            header_solid,
            name="manifold_anode_supply",
            loc=cq.Location(cq.Vector(-(half_outer + standoff), 0, 0)),
            color=cq.Color(*_COLOURS["manifold"]),
        )
        # Return manifold (anode outlet side, +X)
        asm.add(
            header_solid,
            name="manifold_anode_return",
            loc=cq.Location(cq.Vector(half_outer + standoff, 0, 0)),
            color=cq.Color(*_COLOURS["manifold"]),
        )
        # Cathode supply (-Y)
        asm.add(
            header_solid,
            name="manifold_cathode_supply",
            loc=cq.Location(cq.Vector(0, -(half_outer + standoff), 0)),
            color=cq.Color(*_COLOURS["manifold"]),
        )
        # Cathode return (+Y)
        asm.add(
            header_solid,
            name="manifold_cathode_return",
            loc=cq.Location(cq.Vector(0, half_outer + standoff, 0)),
            color=cq.Color(*_COLOURS["manifold"]),
        )

    def _add_peripherals(
        self, asm: cq.Assembly, cfg: StackCADConfig, stack_z_end: float,
    ) -> None:
        """Add 4 reservoir circuits with pumps, valves, motors, feet.

        Layout (top-down, looking at outlet face, Z into page):
            Anolyte(-X)  |  STACK  |  Catholyte(+X)
            Nutrient near anolyte (-X, -Z inlet side)
            Buffer near anolyte (-X, +Z outlet side)
        """
        half_outer = _mm(cfg.outer_side / 2)
        stack_foot_h = _mm(cfg.support_feet.foot_height)
        gap = 80.0  # mm gap between stack edge and reservoir centre
        stack_z_mid = stack_z_end / 2  # midpoint along Z

        # Role -> (dx, dz) on XZ plane; reservoirs stand vertically on Y
        layout: dict[ReservoirRole, tuple[int, int]] = {
            ReservoirRole.ANOLYTE: (-1, 0),    # -X (left)
            ReservoirRole.CATHOLYTE: (1, 0),   # +X (right)
            ReservoirRole.NUTRIENT: (-1, -1),  # -X, inlet side
            ReservoirRole.BUFFER: (-1, 1),     # -X, outlet side
        }

        for role, (dx, dz) in layout.items():
            self._add_reservoir_circuit(
                asm, cfg, role, dx, dz,
                gap, half_outer, stack_foot_h, stack_z_mid,
            )

    def _add_reservoir_circuit(
        self,
        asm: cq.Assembly,
        cfg: StackCADConfig,
        role: ReservoirRole,
        dx: int,
        dz: int,
        gap: float,
        half_outer: float,
        stack_foot_h: float,
        stack_z_mid: float,
    ) -> None:
        """Add one reservoir circuit (reservoir+cone+lid+motor+pump+valve+feet).

        Reservoirs stand vertically (local Z -> assembly +Y) using Rx(-90).
        Ground plane Y = -(half_outer + stack_foot_h).
        """
        spec = cfg.reservoir_spec_for_role(role)
        name = role.value
        res_od = _mm(spec.outer_diameter)
        res_oh = _mm(spec.outer_height)
        cone_h = _mm(cfg.conical_bottom.cone_height)
        boss_l = _mm(cfg.conical_bottom.drain_boss_length)
        lid_t = _mm(cfg.reservoir_lid.thickness)
        feet_h = _mm(cfg.reservoir_feet.foot_height)
        ring_h = _mm(cfg.reservoir_feet.ring_height)

        # Ground plane: bottom of stack feet
        ground_y = -(half_outer + stack_foot_h)

        # XZ position on the horizontal plane
        cx = dx * (half_outer + gap + res_od / 2)
        z_offset = res_od + 30.0  # mm offset along Z (diameter + 30mm clearance)
        cz = stack_z_mid + dz * z_offset

        # Vertical (Y) positions for each component, stacking from ground up
        # Rotation Rx(-90) maps local Z -> assembly +Y
        rx_neg90 = cq.Vector(1, 0, 0)
        rx_angle = -90.0

        colour_key = f"reservoir_{name}"
        if colour_key not in _COLOURS:
            colour_key = "reservoir"

        # 1. Reservoir feet — bottom on ground, ring top at feet_base_y
        feet_base_y = ground_y + feet_h + ring_h
        asm.add(
            reservoir_feet.build(cfg, reservoir_spec=spec),
            name=f"feet_{name}",
            loc=cq.Location(
                cq.Vector(cx, feet_base_y, cz), rx_neg90, rx_angle,
            ),
            color=cq.Color(*_COLOURS["reservoir_feet"]),
        )

        # 2. Conical bottom — hangs below cylinder (cone top = cyl base)
        cone_base_y = feet_base_y - cone_h
        asm.add(
            conical_bottom.build(cfg, reservoir_spec=spec),
            name=f"cone_{name}",
            loc=cq.Location(
                cq.Vector(cx, cone_base_y, cz), rx_neg90, rx_angle,
            ),
            color=cq.Color(*_COLOURS["conical_bottom"]),
        )

        # 3. Reservoir cylinder body — base sits in ring at feet_base_y
        cyl_base_y = feet_base_y
        asm.add(
            reservoir.build(cfg, role=role),
            name=f"reservoir_{name}",
            loc=cq.Location(
                cq.Vector(cx, cyl_base_y, cz), rx_neg90, rx_angle,
            ),
            color=cq.Color(*_COLOURS[colour_key]),
        )

        # 4. Lid on top of cylinder
        lid_y = cyl_base_y + res_oh
        asm.add(
            reservoir_lid.build(cfg, reservoir_spec=spec),
            name=f"lid_{name}",
            loc=cq.Location(
                cq.Vector(cx, lid_y, cz), rx_neg90, rx_angle,
            ),
            color=cq.Color(*_COLOURS["reservoir_lid"]),
        )

        # 5. Stirring motor on top of lid
        motor_y = lid_y + lid_t
        shaft_override = 0.100 if role == ReservoirRole.NUTRIENT else None
        asm.add(
            stirring_motor.build(cfg, shaft_length_override=shaft_override),
            name=f"motor_{name}",
            loc=cq.Location(
                cq.Vector(cx, motor_y, cz), rx_neg90, rx_angle,
            ),
            color=cq.Color(*_COLOURS["stirring_motor"]),
        )

        # 6. Gas diffusion element (catholyte only)
        if role == ReservoirRole.CATHOLYTE:
            gas_y = cyl_base_y + _mm(cfg.gas_diffusion.port_height_from_bottom)
            asm.add(
                gas_diffusion.build(cfg),
                name=f"gas_diffuser_{name}",
                loc=cq.Location(cq.Vector(cx + res_od / 2, gas_y, cz)),
                color=cq.Color(*_COLOURS["gas_diffuser"]),
            )

        # 7. Pump horizontal on ground plane beside reservoir
        pump_support_h = _mm(cfg.pump_support.foot_height)
        platform_t = _mm(cfg.pump_support.platform_thickness)
        pump_body_h = _mm(cfg.pump_head.body_height)
        pump_body_w = _mm(cfg.pump_head.body_width)
        valve_body_w = _mm(cfg.electrovalve.body_width)
        valve_body_h = _mm(cfg.electrovalve.body_height)
        # At least 100mm clearance between reservoir edge and pump edge
        pump_x = (
            cx + dx * (res_od / 2 + 100 + pump_body_w / 2)
            if dx != 0
            else cx + res_od / 2 + 100 + pump_body_w / 2
        )
        pump_z = cz
        pump_y = ground_y + pump_support_h + platform_t + pump_body_h / 2

        # Pump support on ground (rotated Rx(-90) so feet point down)
        pump_support_y = ground_y + pump_support_h + platform_t
        asm.add(
            pump_support.build(cfg),
            name=f"pump_support_{name}",
            loc=cq.Location(
                cq.Vector(pump_x, pump_support_y, pump_z), rx_neg90, rx_angle,
            ),
            color=cq.Color(*_COLOURS["pump_support"]),
        )

        # Pump body (horizontal, rotated Rx(-90) so it lies flat)
        asm.add(
            pump_head.build(cfg),
            name=f"pump_{name}",
            loc=cq.Location(
                cq.Vector(pump_x, pump_y, pump_z), rx_neg90, rx_angle,
            ),
            color=cq.Color(*_COLOURS["pump"]),
        )

        # 8. Electrovalve — 100mm gap between pump edge and valve edge
        valve_offset = 100.0 + pump_body_w / 2 + valve_body_w / 2
        valve_x = pump_x + (dx * valve_offset if dx != 0 else valve_offset)
        valve_z = pump_z
        valve_y = ground_y + pump_support_h + platform_t + valve_body_h / 2
        asm.add(
            electrovalve.build(cfg),
            name=f"valve_{name}",
            loc=cq.Location(cq.Vector(valve_x, valve_y, valve_z)),
            color=cq.Color(*_COLOURS["electrovalve"]),
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
        parts += n * n_collectors * 2  # current collector rods (anode + cathode)

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
            # Per circuit: reservoir+cone+feet+lid+motor+pump_support+pump+valve
            # 4 circuits = 32
            parts += 4 * 8
            # Catholyte gets 1 extra gas diffuser
            parts += 1

        return parts
