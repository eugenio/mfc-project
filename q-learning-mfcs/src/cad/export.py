"""STEP / STL / BOM export utilities.

Can be run as ``python -m src.cad.export`` to generate the
full stack and write files to ``data/cad_models/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .cad_config import StackCADConfig
from .components.oring import total_oring_count


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Component / Assembly export (requires CadQuery)
# ---------------------------------------------------------------------------

def export_component(
    solid: Any,
    name: str,
    output_dir: Path,
    formats: list[str] | None = None,
) -> list[Path]:
    """Export a CadQuery solid to STEP and/or STL.

    Parameters
    ----------
    solid : cq.Workplane
        The CadQuery workplane containing the solid.
    name : str
        Base filename without extension.
    output_dir : Path
        Directory to write files into.
    formats : list[str], optional
        Export formats, defaults to ``["step", "stl"]``.

    Returns
    -------
    list[Path]
        Paths of exported files.
    """
    import cadquery as cq

    if formats is None:
        formats = ["step", "stl"]

    _ensure_dir(output_dir)
    paths: list[Path] = []

    for fmt in formats:
        ext = fmt.lower()
        out = output_dir / f"{name}.{ext}"
        if ext == "step":
            cq.exporters.export(solid, str(out), cq.exporters.ExportTypes.STEP)
        elif ext == "stl":
            cq.exporters.export(solid, str(out), cq.exporters.ExportTypes.STL)
        else:
            msg = f"Unsupported format: {fmt}"
            raise ValueError(msg)
        paths.append(out)

    return paths


def export_assembly(
    assembly: Any,
    output_dir: Path,
    name: str = "mfc_stack_assembly",
) -> Path:
    """Export a CadQuery Assembly to a STEP file.

    Returns the path of the written file.
    """
    import cadquery as cq

    _ensure_dir(output_dir)
    out = output_dir / f"{name}.step"
    assembly.save(str(out))
    return out


# ---------------------------------------------------------------------------
# Bill of Materials (pure Python — no CadQuery needed)
# ---------------------------------------------------------------------------

def generate_bom(config: StackCADConfig) -> dict[str, Any]:
    """Generate a bill of materials for the stack.

    Returns a JSON-serialisable dictionary.
    """
    n = config.num_cells
    cc = config.current_collector
    oring_counts = total_oring_count(n, cc.count_per_electrode)

    outer_mm = config.outer_side * 1000
    inner_mm = config.semi_cell.inner_side * 1000
    depth_mm = config.semi_cell.depth * 1000
    ep_mm = config.end_plate.thickness * 1000
    gasket_mm = config.membrane.gasket_thickness * 1000
    rod_len_mm = config.tie_rod_length * 1000
    collector_len_mm = cc.rod_length * 1000

    parts = [
        {
            "item": 1,
            "part": "Inlet End Plate",
            "qty": 1,
            "material": config.end_plate.material,
            "dimensions_mm": f"{outer_mm:.0f}x{outer_mm:.0f}x{ep_mm:.0f}",
        },
        {
            "item": 2,
            "part": "Outlet End Plate",
            "qty": 1,
            "material": config.end_plate.material,
            "dimensions_mm": f"{outer_mm:.0f}x{outer_mm:.0f}x{ep_mm:.0f}",
        },
        {
            "item": 3,
            "part": "Anode Frame",
            "qty": n,
            "material": "Polypropylene",
            "dimensions_mm": f"{outer_mm:.0f}x{outer_mm:.0f}x{depth_mm:.0f}",
        },
        {
            "item": 4,
            "part": "Cathode Frame (liquid)",
            "qty": n,
            "material": "Polypropylene",
            "dimensions_mm": f"{outer_mm:.0f}x{outer_mm:.0f}x{depth_mm:.0f}",
        },
        {
            "item": 5,
            "part": "Membrane Gasket",
            "qty": n,
            "material": "Silicone",
            "dimensions_mm": f"{outer_mm:.0f}x{outer_mm:.0f}x{gasket_mm:.0f}",
        },
        {
            "item": 6,
            "part": "Nafion 117 Membrane",
            "qty": n,
            "material": "Nafion",
            "dimensions_mm": (
                f"{inner_mm:.0f}x{inner_mm:.0f}"
                f"x{config.membrane.thickness * 1000:.3f}"
            ),
        },
        {
            "item": 7,
            "part": "Face Seal O-Ring",
            "qty": oring_counts["face_seal"],
            "material": "NBR 70A",
            "dimensions_mm": (
                f"CS {config.face_oring.cross_section_diameter * 1000:.2f}"
            ),
        },
        {
            "item": 8,
            "part": "Rod Seal O-Ring",
            "qty": oring_counts["rod_seal"],
            "material": "NBR 70A",
            "dimensions_mm": (
                f"CS {config.rod_oring.cross_section_diameter * 1000:.2f}"
            ),
        },
        {
            "item": 9,
            "part": f"Tie Rod M{config.tie_rod.diameter * 1000:.0f}",
            "qty": 4,
            "material": config.tie_rod.material,
            "dimensions_mm": (
                f"M{config.tie_rod.diameter * 1000:.0f} x {rod_len_mm:.0f}"
            ),
        },
        {
            "item": 10,
            "part": f"Hex Nut M{config.tie_rod.diameter * 1000:.0f}",
            "qty": 8,
            "material": config.tie_rod.material,
            "dimensions_mm": f"M{config.tie_rod.diameter * 1000:.0f}",
        },
        {
            "item": 11,
            "part": f"Flat Washer M{config.tie_rod.diameter * 1000:.0f}",
            "qty": 8,
            "material": config.tie_rod.material,
            "dimensions_mm": f"OD {config.tie_rod.washer_od * 1000:.0f}",
        },
        {
            "item": 12,
            "part": "Ti Current Rod",
            "qty": n * cc.count_per_electrode * 2,  # anode + cathode
            "material": cc.material,
            "dimensions_mm": (
                f"{cc.diameter * 1000:.0f} mm x {collector_len_mm:.0f}"
            ),
        },
        {
            "item": 13,
            "part": "Anode Electrode",
            "qty": n,
            "material": "Carbon felt",
            "dimensions_mm": (
                f"{inner_mm:.0f}x{inner_mm:.0f}"
                f"x{config.electrode.thickness * 1000:.0f}"
            ),
        },
        {
            "item": 14,
            "part": "Cathode Electrode",
            "qty": n,
            "material": "Carbon felt",
            "dimensions_mm": (
                f"{inner_mm:.0f}x{inner_mm:.0f}"
                f"x{config.electrode.thickness * 1000:.0f}"
            ),
        },
    ]

    # --- extended parts (hydraulics + peripherals) ---
    barb = config.barb_fitting
    parts.append({
        "item": 15,
        "part": "Barb Fitting",
        "qty": n * 4,  # 4 per cell (anode in/out + cathode in/out)
        "material": "Brass",
        "dimensions_mm": (
            f"Barb OD {barb.barb_od * 1000:.0f}, "
            f"Hex AF {barb.hex_af * 1000:.0f}"
        ),
    })

    res = config.reservoir
    parts.append({
        "item": 16,
        "part": "Anolyte Reservoir",
        "qty": 1,
        "material": "HDPE",
        "dimensions_mm": (
            f"OD {res.outer_diameter * 1000:.0f} x "
            f"H {res.outer_height * 1000:.0f}"
        ),
    })

    pump = config.pump_head
    parts.append({
        "item": 17,
        "part": "Peristaltic Pump Head",
        "qty": 1,
        "material": "Polycarbonate",
        "dimensions_mm": (
            f"{pump.body_width * 1000:.0f}x"
            f"{pump.body_depth * 1000:.0f}x"
            f"{pump.body_height * 1000:.0f}"
        ),
    })

    tubing_spec = config.tubing
    parts.append({
        "item": 18,
        "part": "Silicone Tubing",
        "qty": 1,
        "material": "Silicone (platinum-cured)",
        "dimensions_mm": (
            f"ID {tubing_spec.inner_diameter * 1000:.0f} / "
            f"OD {tubing_spec.outer_diameter * 1000:.0f}"
        ),
    })

    sf = config.support_feet
    parts.append({
        "item": 19,
        "part": "Support Foot (U-cradle)",
        "qty": 2,
        "material": "Aluminium",
        "dimensions_mm": (
            f"{sf.foot_depth * 1000:.0f}x"
            f"{sf.foot_width * 1000:.0f}x"
            f"{sf.foot_height * 1000:.0f}"
        ),
    })

    return {
        "title": f"MFC Stack BOM — {n}-cell series stack",
        "num_cells": n,
        "stack_length_mm": config.stack_length * 1000,
        "outer_dimensions_mm": f"{outer_mm:.0f} x {outer_mm:.0f}",
        "parts": parts,
    }


def write_bom_json(config: StackCADConfig, output_dir: Path) -> Path:
    """Write BOM to a JSON file and return the path."""
    _ensure_dir(output_dir)
    bom = generate_bom(config)
    out = output_dir / "bom.json"
    out.write_text(json.dumps(bom, indent=2))
    return out


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate the full stack and export to data/cad_models/."""
    from pathlib import Path as _Path

    from .assembly import MFCStackAssembly

    cfg = StackCADConfig()
    warnings = cfg.validate()
    if warnings:
        for w in warnings:
            print(f"  WARNING: {w}")

    # Determine output directory
    try:
        from path_config import get_cad_model_path

        base = _Path(get_cad_model_path("", subdir="")).parent
    except ImportError:
        base = _Path(__file__).resolve().parent.parent.parent / "data" / "cad_models"

    comp_dir = base / "components"
    asm_dir = base / "assemblies"

    # Export individual components
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

    component_builders = {
        "anode_frame": lambda: anode_frame.build(cfg),
        "cathode_frame": lambda: cathode_frame.build(cfg),
        "cathode_frame_gas": lambda: cathode_frame_gas.build(cfg),
        "membrane_gasket": lambda: membrane_gasket.build(cfg),
        "end_plate_inlet": lambda: end_plate.build(cfg, is_inlet=True),
        "end_plate_outlet": lambda: end_plate.build(cfg, is_inlet=False),
        "tie_rod": lambda: tie_rod.build_rod(cfg),
        "tie_rod_nut": lambda: tie_rod.build_nut(cfg),
        "tie_rod_washer": lambda: tie_rod.build_washer(cfg),
        "current_collector": lambda: current_collector.build(cfg),
        "electrode_placeholder": lambda: electrode_placeholder.build(cfg),
    }

    print(f"Exporting {len(component_builders)} components ...")
    for name, builder in component_builders.items():
        solid = builder()
        paths = export_component(solid, name, comp_dir)
        for p in paths:
            print(f"  {p}")

    # Export full assembly
    print("Building full stack assembly ...")
    asm = MFCStackAssembly(cfg).build()
    asm_path = export_assembly(asm, asm_dir)
    print(f"  {asm_path}")

    # Write BOM
    bom_path = write_bom_json(cfg, base)
    print(f"  {bom_path}")
    print("Done.")


if __name__ == "__main__":
    main()
