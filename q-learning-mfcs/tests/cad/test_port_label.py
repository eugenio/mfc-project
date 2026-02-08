"""Tests for port_label.py — 3D text label plates."""

from __future__ import annotations

import pytest

from cad.cad_config import StackCADConfig

pytest.importorskip("cadquery", reason="CadQuery not installed")


class TestPortLabel:
    def test_build_label_returns_solid(self, default_config: StackCADConfig) -> None:
        from cad.components.port_label import build_label

        result = build_label("AN IN", default_config)
        assert result is not None
        assert len(result.solids().vals()) >= 1

    def test_plate_thickness(self, default_config: StackCADConfig) -> None:
        """Label plate Z thickness should match spec."""
        from cad.components.port_label import build_label

        result = build_label("TEST", default_config)
        bb = result.val().BoundingBox()
        spec = default_config.port_label
        expected_mm = (spec.plate_thickness + spec.text_depth) * 1000
        assert bb.zlen == pytest.approx(expected_mm, rel=0.3)

    def test_different_texts(self, default_config: StackCADConfig) -> None:
        """Different labels should all build successfully."""
        from cad.components.port_label import build_label

        for text in ["AN IN", "AN OUT", "CA IN", "CA OUT", "IN", "OUT", "CC+", "CC-"]:
            result = build_label(text, default_config)
            assert result is not None

    def test_fallback_plain_plate(self, default_config: StackCADConfig) -> None:
        """Should build a plain plate even if text rendering fails."""
        from cad.components.port_label import build_label

        # Empty text should still produce a valid plate
        result = build_label("", default_config)
        assert result is not None
        assert len(result.solids().vals()) >= 1
