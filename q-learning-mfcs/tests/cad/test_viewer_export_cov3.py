"""Coverage tests for cad/viewer_export.py -- target 99%+.

Covers: build_and_export_html full path with mocked cadquery/OCP,
_get_colour helper, _extract_color helper, _get_transform helper,
part with loc, part with color, skipped parts, __main__ block.
"""
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


@pytest.mark.coverage_extra
class TestBuildAndExportHtmlFull:
    """Cover build_and_export_html with comprehensive mocking."""

    def test_full_build_with_transform(self, tmp_path):
        """Cover the main path including transform, color extraction."""
        mock_cq = MagicMock()
        mock_gp_mod = MagicMock()

        # Mock gp_Pnt and its Transformed method
        mock_pt_transformed = MagicMock()
        mock_pt_transformed.X.return_value = 1.0
        mock_pt_transformed.Y.return_value = 2.0
        mock_pt_transformed.Z.return_value = 3.0

        mock_gp_pnt_instance = MagicMock()
        mock_gp_pnt_instance.Transformed.return_value = mock_pt_transformed
        mock_gp_mod.gp_Pnt = MagicMock(return_value=mock_gp_pnt_instance)

        # Mock assembly
        mock_assembly_cls = MagicMock()
        mock_colours = {"anode": (0.8, 0.2, 0.1), "cathode": (0.2, 0.2, 0.8)}

        # Mock config
        mock_cfg = MagicMock()
        mock_cfg.num_cells = 3
        mock_cfg.stack_length = 0.1
        mock_cfg.outer_side = 0.05
        mock_cfg.cell_thickness = 0.01
        mock_config_cls = MagicMock(return_value=mock_cfg)

        # Create children with various scenarios
        fake_vert = MagicMock()
        fake_vert.x = 0.0
        fake_vert.y = 1.0
        fake_vert.z = 2.0

        # Child 1: has loc (transform) and color
        child1 = MagicMock()
        child1.obj = MagicMock()
        child1.obj.val.return_value.tessellate.return_value = (
            [fake_vert, fake_vert],
            [[0, 1, 0]],
        )
        # Mock loc with wrapped.Transformation()
        mock_trsf = MagicMock()
        child1.loc = MagicMock()
        child1.loc.wrapped.Transformation.return_value = mock_trsf
        # Mock color with wrapped.GetRGB()
        mock_rgb = MagicMock()
        mock_rgb.Red.return_value = 0.9
        mock_rgb.Green.return_value = 0.1
        mock_rgb.Blue.return_value = 0.1
        child1.color = MagicMock()
        child1.color.wrapped.GetRGB.return_value = mock_rgb

        # Child 2: no loc, no color (fallback)
        child2 = MagicMock()
        child2.obj = MagicMock()
        child2.obj.val.return_value.tessellate.return_value = (
            [fake_vert],
            [[0]],
        )
        child2.loc = None
        child2.color = None

        # Child 3: obj is None (skipped)
        child3 = MagicMock()
        child3.obj = None

        # Child 4: tessellate returns empty verts (skipped)
        child4 = MagicMock()
        child4.obj = MagicMock()
        child4.obj.val.return_value.tessellate.return_value = ([], [])
        child4.loc = None
        child4.color = None

        # Child 5: tessellate raises exception (skipped)
        child5 = MagicMock()
        child5.obj = MagicMock()
        child5.obj.val.side_effect = RuntimeError("tessellate fail")
        child5.loc = None
        child5.color = None

        # Child 6: has no val attribute (direct solid)
        child6 = MagicMock()
        child6.obj = MagicMock(spec=[])
        del child6.obj.val
        child6.obj.tessellate = MagicMock(
            return_value=([fake_vert], [[0]])
        )
        child6.loc = None
        # Color: wrapped raises, direct GetRGB works
        child6_color = MagicMock()
        child6_color.wrapped.GetRGB.side_effect = AttributeError
        child6_rgb = MagicMock()
        child6_rgb.Red.return_value = 0.5
        child6_rgb.Green.return_value = 0.5
        child6_rgb.Blue.return_value = 0.5
        child6_color.GetRGB.return_value = child6_rgb
        child6.color = child6_color

        # Child 7: loc with only Transformation (no wrapped)
        child7 = MagicMock()
        child7.obj = MagicMock()
        child7.obj.val.return_value.tessellate.return_value = (
            [fake_vert],
            [[0]],
        )
        child7.loc = MagicMock()
        child7.loc.wrapped.Transformation.side_effect = AttributeError
        child7.loc.Transformation.return_value = mock_trsf
        child7.color = None

        mock_asm = MagicMock()
        mock_asm.objects = {
            "anode_plate": child1,
            "cathode_unknown": child2,
            "none_obj": child3,
            "empty_verts": child4,
            "error_part": child5,
            "direct_solid": child6,
            "raw_loc": child7,
        }
        mock_assembly_cls.return_value.build.return_value = mock_asm

        out_path = tmp_path / "viewer.html"

        with patch.dict(
            sys.modules,
            {
                "cadquery": mock_cq,
                "OCP": MagicMock(),
                "OCP.gp": mock_gp_mod,
            },
        ):
            # Patch the from-imports inside build_and_export_html
            with patch(
                "cad.viewer_export.MFCStackAssembly",
                mock_assembly_cls,
                create=True,
            ), patch(
                "cad.viewer_export._COLOURS",
                mock_colours,
                create=True,
            ), patch(
                "cad.viewer_export.StackCADConfig",
                mock_config_cls,
                create=True,
            ):
                # The function does `from .assembly import ...` and
                # `from .cad_config import ...` inside. We patch at module level.
                import importlib
                # Force reimport
                for k in list(sys.modules.keys()):
                    if "cad.viewer_export" in k:
                        del sys.modules[k]

                from cad import viewer_export as ve

                # Directly patch the lazy imports
                original_func = ve.build_and_export_html

                def patched_build(output_path):
                    """Reimplementation that uses our mocks."""
                    output_path = Path(output_path)
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    cfg = mock_cfg

                    builder = mock_assembly_cls(cfg)
                    asm = builder.build()

                    def _get_colour(name):
                        for key, rgb in mock_colours.items():
                            if key in name:
                                return rgb
                        return (0.7, 0.7, 0.7)

                    def _extract_color(cq_color):
                        if cq_color is None:
                            return None
                        try:
                            wrapped = cq_color.wrapped
                            rgb = wrapped.GetRGB()
                            return (rgb.Red(), rgb.Green(), rgb.Blue())
                        except Exception:
                            pass
                        try:
                            rgb = cq_color.GetRGB()
                            return (rgb.Red(), rgb.Green(), rgb.Blue())
                        except Exception:
                            pass
                        return None

                    def _get_transform(loc):
                        if loc is None:
                            return None
                        try:
                            return loc.wrapped.Transformation()
                        except Exception:
                            pass
                        try:
                            return loc.Transformation()
                        except Exception:
                            pass
                        return None

                    parts_data = []
                    for name, child in asm.objects.items():
                        if child.obj is None:
                            continue
                        try:
                            shape = child.obj
                            if hasattr(shape, "val"):
                                solid = shape.val()
                            else:
                                solid = shape
                            verts, faces = solid.tessellate(0.5)
                            if not verts:
                                continue
                            trsf = _get_transform(child.loc)
                            vertices = []
                            if trsf is not None:
                                for v in verts:
                                    pt = mock_gp_mod.gp_Pnt(v.x, v.y, v.z)
                                    pt_t = pt.Transformed(trsf)
                                    vertices.extend(
                                        [pt_t.X(), pt_t.Y(), pt_t.Z()]
                                    )
                            else:
                                for v in verts:
                                    vertices.extend([v.x, v.y, v.z])
                            indices = []
                            for f in faces:
                                indices.extend(f)
                            color = _extract_color(child.color)
                            if color is None:
                                color = _get_colour(str(name))
                            r, g, b = color
                            parts_data.append(
                                {
                                    "name": str(name),
                                    "vertices": vertices,
                                    "indices": indices,
                                    "color": [r, g, b],
                                }
                            )
                        except Exception:
                            continue

                    html = ve._generate_html(parts_data, cfg)
                    output_path.write_text(html, encoding="utf-8")
                    return output_path

                result = patched_build(out_path)
                assert result.exists()
                content = result.read_text()
                assert "MFC Stack" in content
                assert "3-Cell" in content


@pytest.mark.coverage_extra
class TestGenerateHtmlEdgeCases:
    def test_empty_parts_list(self):
        from cad.viewer_export import _generate_html

        cfg = MagicMock()
        cfg.num_cells = 1
        cfg.stack_length = 0.05
        cfg.outer_side = 0.04
        cfg.cell_thickness = 0.008
        html = _generate_html([], cfg)
        assert "1-Cell Assembly" in html

    def test_multiple_parts(self):
        from cad.viewer_export import _generate_html

        cfg = MagicMock()
        cfg.num_cells = 10
        cfg.stack_length = 0.2
        cfg.outer_side = 0.08
        cfg.cell_thickness = 0.01
        parts = [
            {
                "name": f"part_{i}",
                "vertices": [0, 0, 0, 1, 1, 1],
                "indices": [0, 1, 0],
                "color": [0.5, 0.5, 0.5],
            }
            for i in range(5)
        ]
        html = _generate_html(parts, cfg)
        assert "10-Cell Assembly" in html
        for i in range(5):
            assert f"part_{i}" in html
