"""Tests for assembly.py — full MFC stack assembly."""

from __future__ import annotations

import pytest

from cad.cad_config import StackCADConfig


# ---------------------------------------------------------------------------
# Part-count tests (no CadQuery needed)
# ---------------------------------------------------------------------------
class TestExpectedPartCount:
    """Test the part-count arithmetic independently of CadQuery."""

    @staticmethod
    def _part_count(num_cells: int, n_collectors: int = 3) -> int:
        """Manual expected-part-count formula."""
        parts = 0
        parts += 2  # end plates
        parts += num_cells * 5  # anode frame + elec + gasket + cathode frame + elec
        parts += 4  # tie rods
        parts += 8  # nuts (top + bot × 4)
        parts += 8  # washers
        parts += num_cells * n_collectors  # collector rods
        return parts

    def test_10_cell_standard(self) -> None:
        expected = self._part_count(10)
        assert expected == 2 + 50 + 4 + 8 + 8 + 30  # = 102

    def test_1_cell(self) -> None:
        expected = self._part_count(1)
        assert expected == 2 + 5 + 4 + 8 + 8 + 3  # = 30

    def test_matches_assembly_class(self) -> None:
        """Verify that MFCStackAssembly.expected_part_count matches
        our manual formula (imports assembly lazily to avoid CadQuery)."""
        try:
            from cad.assembly import MFCStackAssembly
        except ImportError:
            pytest.skip("CadQuery not installed — cannot import assembly")

        for n in (1, 5, 10):
            cfg = StackCADConfig(num_cells=n)
            builder = MFCStackAssembly(cfg)
            assert builder.expected_part_count == self._part_count(n)


# ---------------------------------------------------------------------------
# Full assembly tests (require CadQuery)
# ---------------------------------------------------------------------------
class TestMFCStackAssembly:
    @pytest.fixture(autouse=True)
    def _require_cadquery(self) -> None:
        pytest.importorskip("cadquery", reason="CadQuery not installed")

    def test_build_returns_assembly(self, default_config: StackCADConfig) -> None:
        from cad.assembly import MFCStackAssembly

        asm = MFCStackAssembly(default_config).build()
        assert asm is not None

    def test_expected_part_count(self, default_config: StackCADConfig) -> None:
        from cad.assembly import MFCStackAssembly

        builder = MFCStackAssembly(default_config)
        asm = builder.build()
        actual = len(asm.objects)
        assert actual == builder.expected_part_count

    def test_single_cell(self, single_cell_config: StackCADConfig) -> None:
        from cad.assembly import MFCStackAssembly

        builder = MFCStackAssembly(single_cell_config)
        asm = builder.build()
        assert len(asm.objects) == builder.expected_part_count

    def test_gas_cathode_variant(self, default_config: StackCADConfig) -> None:
        from cad.assembly import MFCStackAssembly

        builder = MFCStackAssembly(
            default_config,
            gas_cathode_cells={0, 5},
        )
        asm = builder.build()
        assert len(asm.objects) == builder.expected_part_count
