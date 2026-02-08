"""Comprehensive coverage tests for cad.assembly module."""
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Mock cadquery and all CQ-dependent component modules before import
_mock_cq = MagicMock()
sys.modules.setdefault("cadquery", _mock_cq)

import pytest

from cad.cad_config import (
    CathodeType,
    FlowConfiguration,
    StackCADConfig,
)


# Patch component modules that use cadquery at import time
@pytest.fixture(autouse=True)
def _mock_cq_modules():
    """Ensure cadquery stays mocked throughout tests."""
    sys.modules["cadquery"] = _mock_cq
    yield


class TestMFCStackAssemblyInit:
    def test_default_liquid(self):
        from cad.assembly import MFCStackAssembly
        cfg = StackCADConfig()
        asm = MFCStackAssembly(cfg)
        assert asm.cathode_type == CathodeType.LIQUID
        assert asm.gas_cathode_cells == set()

    def test_gas_cathode(self):
        from cad.assembly import MFCStackAssembly
        cfg = StackCADConfig(num_cells=3)
        asm = MFCStackAssembly(cfg, cathode_type=CathodeType.GAS)
        assert asm.gas_cathode_cells == {0, 1, 2}

    def test_explicit_gas_cells(self):
        from cad.assembly import MFCStackAssembly
        cfg = StackCADConfig(num_cells=5)
        asm = MFCStackAssembly(cfg, gas_cathode_cells={1, 3})
        assert asm.gas_cathode_cells == {1, 3}

    def test_include_flags(self):
        from cad.assembly import MFCStackAssembly
        cfg = StackCADConfig()
        asm = MFCStackAssembly(
            cfg,
            include_supports=True,
            include_labels=True,
            include_hydraulics=True,
            include_peripherals=True,
        )
        assert asm.include_supports is True
        assert asm.include_labels is True
        assert asm.include_hydraulics is True
        assert asm.include_peripherals is True


class TestFactoryMethods:
    def test_all_liquid(self):
        from cad.assembly import MFCStackAssembly
        cfg = StackCADConfig(num_cells=3)
        asm = MFCStackAssembly.all_liquid(cfg)
        assert asm.cathode_type == CathodeType.LIQUID
        assert asm.gas_cathode_cells == set()

    def test_all_gas(self):
        from cad.assembly import MFCStackAssembly
        cfg = StackCADConfig(num_cells=3)
        asm = MFCStackAssembly.all_gas(cfg)
        assert asm.cathode_type == CathodeType.GAS
        assert asm.gas_cathode_cells == {0, 1, 2}

    def test_all_liquid_with_kwargs(self):
        from cad.assembly import MFCStackAssembly
        cfg = StackCADConfig()
        asm = MFCStackAssembly.all_liquid(cfg, include_supports=True)
        assert asm.include_supports is True


class TestMmHelper:
    def test_mm_conversion(self):
        from cad.assembly import _mm
        assert _mm(1.0) == pytest.approx(1000.0)
        assert _mm(0.025) == pytest.approx(25.0)


class TestExpectedPartCount:
    def test_base_count(self):
        from cad.assembly import MFCStackAssembly
        cfg = StackCADConfig(num_cells=2)
        asm = MFCStackAssembly(cfg)
        # 2 end plates + 2 anode + 2 anode elec + 2 membrane + 2 cathode + 2 cathode elec
        # + 4 tie rods + 8 nuts + 8 washers + 2*3 collectors = 32
        expected = 2 + 2 + 2 + 2 + 2 + 2 + 4 + 8 + 8 + 6
        assert asm.expected_part_count == expected

    def test_with_supports(self):
        from cad.assembly import MFCStackAssembly
        cfg = StackCADConfig(num_cells=1)
        asm_no = MFCStackAssembly(cfg, include_supports=False)
        asm_yes = MFCStackAssembly(cfg, include_supports=True)
        assert asm_yes.expected_part_count == asm_no.expected_part_count + 2

    def test_with_labels(self):
        from cad.assembly import MFCStackAssembly
        cfg = StackCADConfig(num_cells=1)
        asm_no = MFCStackAssembly(cfg, include_labels=False)
        asm_yes = MFCStackAssembly(cfg, include_labels=True)
        assert asm_yes.expected_part_count == asm_no.expected_part_count + 8

    def test_with_hydraulics_series(self):
        from cad.assembly import MFCStackAssembly
        cfg = StackCADConfig(num_cells=3, flow_config=FlowConfiguration.SERIES)
        asm = MFCStackAssembly(cfg, include_hydraulics=True)
        base = MFCStackAssembly(cfg, include_hydraulics=False)
        # 3 cells * 4 fittings + (3-1)*2 u-tubes = 12 + 4 = 16
        assert asm.expected_part_count == base.expected_part_count + 16

    def test_with_hydraulics_parallel(self):
        from cad.assembly import MFCStackAssembly
        cfg = StackCADConfig(num_cells=3, flow_config=FlowConfiguration.PARALLEL)
        asm = MFCStackAssembly(cfg, include_hydraulics=True)
        base = MFCStackAssembly(cfg, include_hydraulics=False)
        # 3*4 fittings + 4 manifolds = 16
        assert asm.expected_part_count == base.expected_part_count + 16

    def test_with_peripherals(self):
        from cad.assembly import MFCStackAssembly
        cfg = StackCADConfig(num_cells=1)
        asm = MFCStackAssembly(cfg, include_peripherals=True)
        base = MFCStackAssembly(cfg, include_peripherals=False)
        assert asm.expected_part_count == base.expected_part_count + 2


class TestBuildAssembly:
    def test_build_returns_assembly(self):
        from cad.assembly import MFCStackAssembly
        cfg = StackCADConfig(num_cells=1)
        asm = MFCStackAssembly(cfg)
        result = asm.build()
        assert result is not None

    def test_build_with_gas_cathode(self):
        from cad.assembly import MFCStackAssembly
        cfg = StackCADConfig(num_cells=1)
        asm = MFCStackAssembly(cfg, cathode_type=CathodeType.GAS)
        result = asm.build()
        assert result is not None

    def test_build_with_all_options(self):
        from cad.assembly import MFCStackAssembly
        cfg = StackCADConfig(num_cells=2)
        asm = MFCStackAssembly(
            cfg,
            include_supports=True,
            include_labels=True,
            include_hydraulics=True,
            include_peripherals=True,
        )
        result = asm.build()
        assert result is not None

    def test_build_parallel_hydraulics(self):
        from cad.assembly import MFCStackAssembly
        cfg = StackCADConfig(num_cells=2, flow_config=FlowConfiguration.PARALLEL)
        asm = MFCStackAssembly(cfg, include_hydraulics=True)
        result = asm.build()
        assert result is not None


class TestColours:
    def test_colours_dict(self):
        from cad.assembly import _COLOURS
        assert isinstance(_COLOURS, dict)
        assert "anode_frame" in _COLOURS
        assert "end_plate" in _COLOURS
        assert len(_COLOURS) > 10
        for key, val in _COLOURS.items():
            assert len(val) == 3
            assert all(0.0 <= v <= 1.0 for v in val)
