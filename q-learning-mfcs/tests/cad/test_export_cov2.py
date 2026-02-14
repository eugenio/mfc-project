"""Coverage tests for cad/export.py -- target 99%+.

Covers: main() function with mocked CadQuery, export_component with
all format branches, export_assembly, write_bom_json.
"""
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Snapshot sys.modules before mocking
_original_modules = dict(sys.modules)

_mock_cq = MagicMock()
sys.modules.setdefault("cadquery", _mock_cq)


from cad.cad_config import StackCADConfig
from cad.export import (
    _ensure_dir,
    export_assembly,
    export_component,
    generate_bom,
    main,
    write_bom_json,
)

# --- Restore sys.modules to prevent mock leakage ---
for _mock_key in list(sys.modules):
    if _mock_key not in _original_modules:
        if isinstance(sys.modules[_mock_key], MagicMock):
            del sys.modules[_mock_key]
    elif isinstance(sys.modules[_mock_key], MagicMock):
        sys.modules[_mock_key] = _original_modules[_mock_key]


@pytest.mark.coverage_extra
class TestMainFunction:
    """Cover the main() CLI entry point with mocked CQ components."""

    def test_main_with_path_config(self, tmp_path):
        """Cover main with path_config importable."""
        mock_asm_cls = MagicMock()
        mock_asm = MagicMock()
        mock_asm_cls.return_value.build.return_value = mock_asm

        # Mock all component builders
        mock_components = MagicMock()

        with patch("cad.export.MFCStackAssembly", mock_asm_cls, create=True):
            with patch(
                "cad.export.get_cad_model_path",
                side_effect=ImportError("no path_config"),
                create=True,
            ):
                # Patch the from-import inside main
                with patch.dict(
                    sys.modules,
                    {
                        "path_config": None,  # Force ImportError
                    },
                ):
                    with patch(
                        "cad.export.export_component"
                    ) as mock_ec, patch(
                        "cad.export.export_assembly"
                    ) as mock_ea, patch(
                        "cad.export.write_bom_json"
                    ) as mock_wbj:
                        # Mock component imports
                        mock_comp_mods = {}
                        comp_names = [
                            "anode_frame",
                            "cathode_frame",
                            "cathode_frame_gas",
                            "membrane_gasket",
                            "end_plate",
                            "tie_rod",
                            "current_collector",
                            "electrode_placeholder",
                            "reservoir",
                            "conical_bottom",
                            "reservoir_lid",
                            "stirring_motor",
                            "gas_diffusion",
                            "electrovalve",
                            "reservoir_feet",
                            "pump_support",
                            "pump_head",
                        ]
                        for cn in comp_names:
                            mock_mod = MagicMock()
                            mock_mod.build.return_value = MagicMock()
                            if cn == "tie_rod":
                                mock_mod.build_rod.return_value = MagicMock()
                                mock_mod.build_nut.return_value = MagicMock()
                                mock_mod.build_washer.return_value = MagicMock()
                            if cn == "end_plate":
                                mock_mod.build.return_value = MagicMock()
                            mock_comp_mods[
                                f"cad.components.{cn}"
                            ] = mock_mod

                        with patch.dict(sys.modules, mock_comp_mods):
                            main()

                        assert mock_ec.call_count >= 1
                        mock_ea.assert_called_once()
                        mock_wbj.assert_called_once()

    def test_main_with_warnings(self, tmp_path, capsys):
        """Cover main when config.validate() returns warnings."""
        cfg = StackCADConfig()

        with patch(
            "cad.export.StackCADConfig", return_value=cfg
        ) as mock_cfg_cls:
            with patch.object(
                cfg, "validate", return_value=["Test warning 1"]
            ):
                with patch.dict(sys.modules, {"path_config": None}):
                    with patch(
                        "cad.export.export_component"
                    ), patch(
                        "cad.export.export_assembly"
                    ), patch(
                        "cad.export.write_bom_json"
                    ), patch(
                        "cad.export.MFCStackAssembly",
                        MagicMock(),
                        create=True,
                    ):
                        # Mock component imports
                        comp_names = [
                            "anode_frame",
                            "cathode_frame",
                            "cathode_frame_gas",
                            "membrane_gasket",
                            "end_plate",
                            "tie_rod",
                            "current_collector",
                            "electrode_placeholder",
                            "reservoir",
                            "conical_bottom",
                            "reservoir_lid",
                            "stirring_motor",
                            "gas_diffusion",
                            "electrovalve",
                            "reservoir_feet",
                            "pump_support",
                            "pump_head",
                        ]
                        for cn in comp_names:
                            sys.modules.setdefault(
                                f"cad.components.{cn}", MagicMock()
                            )

                        main()

                        captured = capsys.readouterr()
                        assert "WARNING" in captured.out


@pytest.mark.coverage_extra
class TestExportComponentFormats:
    def test_step_and_stl(self, tmp_path):
        solid = MagicMock()
        paths = export_component(solid, "dual", tmp_path)
        assert len(paths) == 2

    def test_stl_only(self, tmp_path):
        solid = MagicMock()
        paths = export_component(solid, "stl_only", tmp_path, formats=["stl"])
        assert len(paths) == 1
        assert str(paths[0]).endswith(".stl")

    def test_step_only(self, tmp_path):
        solid = MagicMock()
        paths = export_component(solid, "step_only", tmp_path, formats=["step"])
        assert len(paths) == 1
        assert str(paths[0]).endswith(".step")


@pytest.mark.coverage_extra
class TestExportAssembly:
    def test_custom_name(self, tmp_path):
        asm = MagicMock()
        result = export_assembly(asm, tmp_path, name="custom_asm")
        assert "custom_asm.step" in str(result)
        asm.save.assert_called_once()


@pytest.mark.coverage_extra
class TestWriteBomJson:
    def test_creates_subdir(self, tmp_path):
        cfg = StackCADConfig(num_cells=3)
        sub = tmp_path / "bom_out"
        result = write_bom_json(cfg, sub)
        assert result.exists()
        data = json.loads(result.read_text())
        assert data["num_cells"] == 3


@pytest.mark.coverage_extra
class TestGenerateBomExtended:
    def test_all_extended_parts_present(self):
        cfg = StackCADConfig()
        bom = generate_bom(cfg)
        part_names = [p["part"] for p in bom["parts"]]
        expected = [
            "Barb Fitting",
            "Anolyte Reservoir (10L)",
            "Catholyte Reservoir (10L)",
            "Nutrient Reservoir (1L)",
            "Buffer Reservoir (5L)",
            "Reservoir Lid",
            "Conical Bottom",
            "Reservoir Feet Assembly",
            "Stirring Motor Assembly",
            "Peristaltic Pump Head",
            "Pump Support Platform",
            "3-Way Electrovalve",
            "Gas Diffusion Element",
            "Silicone Tubing",
            "Support Foot (U-cradle)",
        ]
        for name in expected:
            assert name in part_names, f"Missing BOM part: {name}"
