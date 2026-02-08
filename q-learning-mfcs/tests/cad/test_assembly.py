"""Tests for assembly.py — full MFC stack assembly."""

from __future__ import annotations

import pytest

from cad.cad_config import CathodeType, FlowConfiguration, StackCADConfig


# ---------------------------------------------------------------------------
# Part-count tests (no CadQuery needed)
# ---------------------------------------------------------------------------
class TestExpectedPartCount:
    """Test the part-count arithmetic independently of CadQuery."""

    @staticmethod
    def _base_parts(num_cells: int, n_collectors: int = 3) -> int:
        """Manual expected-part-count formula (base stack only)."""
        parts = 0
        parts += 2  # end plates
        parts += num_cells * 5  # anode frame + elec + gasket + cathode frame + elec
        parts += 4  # tie rods
        parts += 8  # nuts (top + bot x 4)
        parts += 8  # washers
        parts += num_cells * n_collectors * 2  # collector rods (anode + cathode)
        return parts

    def test_10_cell_standard(self) -> None:
        expected = self._base_parts(10)
        assert expected == 2 + 50 + 4 + 8 + 8 + 60  # = 132

    def test_1_cell(self) -> None:
        expected = self._base_parts(1)
        assert expected == 2 + 5 + 4 + 8 + 8 + 6  # = 33

    def test_matches_assembly_class(self) -> None:
        """Verify base part count matches."""
        try:
            from cad.assembly import MFCStackAssembly
        except ImportError:
            pytest.skip("CadQuery not installed -- cannot import assembly")

        for n in (1, 5, 10):
            cfg = StackCADConfig(num_cells=n)
            builder = MFCStackAssembly(cfg)
            assert builder.expected_part_count == self._base_parts(n)

    def test_peripherals_count(self) -> None:
        """Peripherals: 4 circuits x 8 parts + 1 gas diffuser = 33."""
        try:
            from cad.assembly import MFCStackAssembly
        except ImportError:
            pytest.skip("CadQuery not installed -- cannot import assembly")

        cfg = StackCADConfig(num_cells=1)
        builder = MFCStackAssembly(cfg, include_peripherals=True)
        base = self._base_parts(1)
        assert builder.expected_part_count == base + 33


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
        # asm.objects includes the root assembly entry (UUID key), subtract 1
        actual = len(asm.objects) - 1
        assert actual == builder.expected_part_count

    def test_single_cell(self, single_cell_config: StackCADConfig) -> None:
        from cad.assembly import MFCStackAssembly

        builder = MFCStackAssembly(single_cell_config)
        asm = builder.build()
        assert len(asm.objects) - 1 == builder.expected_part_count

    def test_gas_cathode_variant(self, default_config: StackCADConfig) -> None:
        from cad.assembly import MFCStackAssembly

        builder = MFCStackAssembly(
            default_config,
            gas_cathode_cells={0, 5},
        )
        asm = builder.build()
        assert len(asm.objects) - 1 == builder.expected_part_count

    def test_all_liquid_factory(self, single_cell_config: StackCADConfig) -> None:
        from cad.assembly import MFCStackAssembly

        builder = MFCStackAssembly.all_liquid(single_cell_config)
        assert builder.cathode_type == CathodeType.LIQUID
        assert len(builder.gas_cathode_cells) == 0
        asm = builder.build()
        assert len(asm.objects) - 1 == builder.expected_part_count

    def test_all_gas_factory(self, single_cell_config: StackCADConfig) -> None:
        from cad.assembly import MFCStackAssembly

        builder = MFCStackAssembly.all_gas(single_cell_config)
        assert builder.cathode_type == CathodeType.GAS
        assert len(builder.gas_cathode_cells) == 1
        asm = builder.build()
        assert len(asm.objects) - 1 == builder.expected_part_count

    def test_with_supports(self, single_cell_config: StackCADConfig) -> None:
        from cad.assembly import MFCStackAssembly

        builder = MFCStackAssembly(single_cell_config, include_supports=True)
        asm = builder.build()
        assert len(asm.objects) - 1 == builder.expected_part_count

    def test_with_labels(self, single_cell_config: StackCADConfig) -> None:
        from cad.assembly import MFCStackAssembly

        builder = MFCStackAssembly(single_cell_config, include_labels=True)
        asm = builder.build()
        assert len(asm.objects) - 1 == builder.expected_part_count

    def test_with_peripherals(self, single_cell_config: StackCADConfig) -> None:
        from cad.assembly import MFCStackAssembly

        builder = MFCStackAssembly(single_cell_config, include_peripherals=True)
        asm = builder.build()
        assert len(asm.objects) - 1 == builder.expected_part_count

    def test_with_hydraulics_series(self, single_cell_config: StackCADConfig) -> None:
        from cad.assembly import MFCStackAssembly

        builder = MFCStackAssembly(single_cell_config, include_hydraulics=True)
        asm = builder.build()
        assert len(asm.objects) - 1 == builder.expected_part_count

    def test_with_hydraulics_parallel(self, single_cell_config: StackCADConfig) -> None:
        from cad.assembly import MFCStackAssembly

        cfg = StackCADConfig(
            num_cells=1,
            flow_config=FlowConfiguration.PARALLEL,
        )
        builder = MFCStackAssembly(cfg, include_hydraulics=True)
        asm = builder.build()
        assert len(asm.objects) - 1 == builder.expected_part_count

    def test_full_assembly_all_options(self, single_cell_config: StackCADConfig) -> None:
        from cad.assembly import MFCStackAssembly

        builder = MFCStackAssembly(
            single_cell_config,
            include_supports=True,
            include_labels=True,
            include_hydraulics=True,
            include_peripherals=True,
        )
        asm = builder.build()
        assert len(asm.objects) - 1 == builder.expected_part_count
