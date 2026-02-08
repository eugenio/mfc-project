"""Parametric CAD configuration for MFC stack design.

All dimensions are in SI units (metres) unless otherwise noted.
The dataclasses mirror the physical structure of a dual-chamber
series MFC stack with tie-rod assembly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ElectrodeDimensions:
    """Square porous electrode (carbon felt)."""

    side_length: float = 0.10  # m  (10 cm)
    thickness: float = 0.005  # m  (5 mm)

    @property
    def area(self) -> float:
        """Projected active area [m²]."""
        return self.side_length**2

    @property
    def volume(self) -> float:
        """Bulk volume [m³]."""
        return self.area * self.thickness


@dataclass(frozen=True)
class SemiCellDimensions:
    """A single semi-cell frame (anode or cathode side)."""

    inner_side: float = 0.10  # m  — chamber opening matches electrode
    depth: float = 0.025  # m  (2.5 cm) — chamber depth
    wall_thickness: float = 0.015  # m  (1.5 cm) — frame wall around chamber

    # Flow ports
    flow_port_diameter: float = 0.008  # m  (8 mm G1/4 bore)
    flow_port_offset: float = 0.015  # m  from chamber edge to port centre

    @property
    def outer_side(self) -> float:
        """Outer dimension of the frame plate [m]."""
        return self.inner_side + 2 * self.wall_thickness

    @property
    def chamber_volume(self) -> float:
        """Chamber volume [m³]."""
        return self.inner_side**2 * self.depth


@dataclass(frozen=True)
class GasCathodeDimensions:
    """Extra dimensions for the gas-collection cathode variant."""

    headspace_depth: float = 0.015  # m  (1.5 cm) above electrode
    gas_port_diameter: float = 0.008  # m  (8 mm)
    gas_port_offset_top: float = 0.005  # m  from top face to port centre


@dataclass(frozen=True)
class MembraneDimensions:
    """Proton-exchange membrane and sealing gasket."""

    thickness: float = 1.78e-4  # m  Nafion 117 (178 µm)
    gasket_thickness: float = 0.002  # m  (2 mm silicone gasket)
    active_side: float = 0.10  # m  — active area opening


@dataclass(frozen=True)
class ORingSpec:
    """O-ring specification following ISO 3601.

    Default: CS 3.53 mm (ISO 3601 series 3).
    Groove dimensions target ~25 % compression for NBR 70 Shore A.
    """

    cross_section_diameter: float = 0.00353  # m  (3.53 mm CS)
    material_shore_a: int = 70
    compression_ratio: float = 0.25

    @property
    def groove_depth(self) -> float:
        """Groove depth for face-seal application [m]."""
        return self.cross_section_diameter * (1.0 - self.compression_ratio)

    @property
    def groove_width(self) -> float:
        """Groove width (~1.35 × CS for face seal) [m]."""
        return self.cross_section_diameter * 1.35

    @property
    def compressed_height(self) -> float:
        """O-ring height after compression [m]."""
        return self.cross_section_diameter * (1.0 - self.compression_ratio)


@dataclass(frozen=True)
class RodSealORingSpec:
    """Smaller O-ring for sealing Ti current-collector rods."""

    cross_section_diameter: float = 0.00178  # m  (1.78 mm CS)
    compression_ratio: float = 0.25

    @property
    def groove_depth(self) -> float:
        return self.cross_section_diameter * (1.0 - self.compression_ratio)

    @property
    def groove_width(self) -> float:
        return self.cross_section_diameter * 1.35


@dataclass(frozen=True)
class TieRodSpec:
    """M10 stainless-steel tie rod."""

    diameter: float = 0.010  # m  (M10)
    clearance_hole_diameter: float = 0.011  # m  (11 mm clearance)
    nut_af: float = 0.017  # m  across-flats for M10 hex nut
    nut_height: float = 0.008  # m
    washer_od: float = 0.021  # m
    washer_thickness: float = 0.002  # m
    material: str = "SS316"

    @property
    def inset(self) -> float:
        """Distance from plate edge to tie-rod centre [m]."""
        return self.washer_od / 2 + 0.003  # 3 mm clearance from edge


@dataclass(frozen=True)
class CurrentCollectorSpec:
    """Titanium current-collector rod passing through the electrode."""

    diameter: float = 0.006  # m  (6 mm Ti Grade 2)
    clearance_hole_diameter: float = 0.007  # m
    seal_oring: RodSealORingSpec = field(default_factory=RodSealORingSpec)
    count_per_electrode: int = 3
    material: str = "Ti Grade 2"

    @property
    def rod_length(self) -> float:
        """Rod protrudes through frame wall on both sides [m]."""
        # frame_depth + 2 * wall_thickness (approximate)
        return 0.070  # 70 mm — trimmed to fit


@dataclass(frozen=True)
class EndPlateSpec:
    """Thick end plate with recessed nut pockets and flow ports."""

    thickness: float = 0.025  # m  (2.5 cm)
    nut_pocket_depth: float = 0.012  # m
    material: str = "Polypropylene"


@dataclass
class StackCADConfig:
    """Master parametric configuration for the entire MFC stack.

    Aggregates all sub-configs and exposes derived stack-level
    properties used by the assembly module.
    """

    num_cells: int = 10

    electrode: ElectrodeDimensions = field(default_factory=ElectrodeDimensions)
    semi_cell: SemiCellDimensions = field(default_factory=SemiCellDimensions)
    gas_cathode: GasCathodeDimensions = field(default_factory=GasCathodeDimensions)
    membrane: MembraneDimensions = field(default_factory=MembraneDimensions)
    face_oring: ORingSpec = field(default_factory=ORingSpec)
    rod_oring: RodSealORingSpec = field(default_factory=RodSealORingSpec)
    tie_rod: TieRodSpec = field(default_factory=TieRodSpec)
    current_collector: CurrentCollectorSpec = field(
        default_factory=CurrentCollectorSpec,
    )
    end_plate: EndPlateSpec = field(default_factory=EndPlateSpec)

    # --- derived properties ---------------------------------------------------

    @property
    def gasket_membrane_thickness(self) -> float:
        """Combined gasket + membrane layer [m]."""
        return self.membrane.gasket_thickness

    @property
    def cell_thickness(self) -> float:
        """Thickness of one repeating cell unit [m].

        anode_frame + gasket/membrane + cathode_frame
        """
        return (
            self.semi_cell.depth
            + self.gasket_membrane_thickness
            + self.semi_cell.depth
        )

    @property
    def stack_length(self) -> float:
        """Total stack length along the compression axis [m]."""
        return (
            2 * self.end_plate.thickness + self.num_cells * self.cell_thickness
        )

    @property
    def outer_side(self) -> float:
        """Outer square dimension of frame plates [m]."""
        return self.semi_cell.outer_side

    @property
    def tie_rod_positions(self) -> list[tuple[float, float]]:
        """(x, y) centres of the four corner tie rods [m].

        Origin at plate centre; inset from each edge.
        """
        inset = self.tie_rod.inset
        half = self.outer_side / 2
        dx = half - inset
        dy = half - inset
        return [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]

    @property
    def collector_positions(self) -> list[tuple[float, float]]:
        """(x, y) centres of Ti collector rods across one electrode.

        Three rods evenly spaced along the electrode centre line.
        """
        n = self.current_collector.count_per_electrode
        side = self.semi_cell.inner_side
        spacing = side / (n + 1)
        return [(spacing * (i + 1) - side / 2, 0.0) for i in range(n)]

    @property
    def tie_rod_length(self) -> float:
        """Total tie rod length including nut engagement [m]."""
        extra = 2 * (
            self.tie_rod.nut_height + self.tie_rod.washer_thickness + 0.005
        )
        return self.stack_length + extra

    @property
    def total_anode_volume(self) -> float:
        """Total anode chamber volume for all cells [m³]."""
        return self.num_cells * self.semi_cell.chamber_volume

    @property
    def total_cathode_volume(self) -> float:
        """Total cathode chamber volume for all cells [m³]."""
        return self.num_cells * self.semi_cell.chamber_volume

    @property
    def active_membrane_area(self) -> float:
        """Active membrane area per cell [m²]."""
        return self.membrane.active_side**2

    def validate(self) -> list[str]:
        """Return a list of validation warnings (empty = OK)."""
        warnings: list[str] = []

        if self.num_cells < 1:
            warnings.append("num_cells must be >= 1")

        if self.semi_cell.inner_side != self.electrode.side_length:
            warnings.append(
                "Semi-cell inner_side does not match electrode side_length"
            )

        if self.membrane.active_side != self.electrode.side_length:
            warnings.append(
                "Membrane active_side does not match electrode side_length"
            )

        # Tie rod must fit within frame wall
        max_inset = self.semi_cell.wall_thickness - self.tie_rod.washer_od / 2
        if max_inset < 0:
            warnings.append(
                "Tie-rod washer extends beyond frame wall thickness"
            )

        # O-ring groove must fit in wall thickness
        if self.face_oring.groove_depth > self.semi_cell.wall_thickness:
            warnings.append("Face O-ring groove deeper than wall thickness")

        # Electrode must fit in chamber
        if self.electrode.thickness > self.semi_cell.depth:
            warnings.append("Electrode thicker than semi-cell depth")

        # Gasket must be thicker than membrane
        if self.membrane.gasket_thickness < self.membrane.thickness:
            warnings.append("Gasket thinner than membrane — no compression")

        return warnings
