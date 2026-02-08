"""Coverage tests for cad.components.cathode_frame_gas module."""
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.modules.setdefault("cadquery", MagicMock())

import pytest

from cad.cad_config import StackCADConfig
from cad.components.cathode_frame_gas import _mm, build


class TestMmHelper:
    def test_conversion(self):
        assert _mm(0.1) == pytest.approx(100.0)


class TestBuild:
    def test_returns_workplane(self):
        cfg = StackCADConfig()
        result = build(cfg)
        assert result is not None

    def test_single_cell_config(self):
        cfg = StackCADConfig(num_cells=1)
        result = build(cfg)
        assert result is not None

    def test_custom_gas_cathode(self):
        from cad.cad_config import GasCathodeDimensions
        cfg = StackCADConfig(
            gas_cathode=GasCathodeDimensions(headspace_depth=0.020)
        )
        result = build(cfg)
        assert result is not None
