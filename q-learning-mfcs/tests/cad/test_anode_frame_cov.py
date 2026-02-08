"""Coverage tests for cad.components.anode_frame module."""
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.modules.setdefault("cadquery", MagicMock())

import pytest

from cad.cad_config import StackCADConfig
from cad.components.anode_frame import _mm, build


class TestMmHelper:
    def test_conversion(self):
        assert _mm(0.025) == pytest.approx(25.0)


class TestBuild:
    def test_returns_workplane(self):
        cfg = StackCADConfig()
        result = build(cfg)
        assert result is not None

    def test_single_cell(self):
        cfg = StackCADConfig(num_cells=1)
        result = build(cfg)
        assert result is not None

    def test_custom_semi_cell(self):
        from cad.cad_config import ElectrodeDimensions, MembraneDimensions, SemiCellDimensions
        cfg = StackCADConfig(
            electrode=ElectrodeDimensions(side_length=0.05),
            semi_cell=SemiCellDimensions(inner_side=0.05, depth=0.015),
            membrane=MembraneDimensions(active_side=0.05),
        )
        result = build(cfg)
        assert result is not None
