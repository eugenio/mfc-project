import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest
from config.unit_converter import UnitConverter, UnitDefinition, UNIT_CONVERTER


class TestUnitDefinition:
    def test_basic_creation(self):
        ud = UnitDefinition("meters", "m", "length", 1.0, "m")
        assert ud.name == "meters"
        assert ud.symbol == "m"
        assert ud.common_aliases == []

    def test_creation_with_aliases(self):
        ud = UnitDefinition("meters", "m", "length", 1.0, "m", ["meter"])
        assert ud.common_aliases == ["meter"]

    def test_post_init_none_aliases(self):
        ud = UnitDefinition("meters", "m", "length", 1.0, "m", None)
        assert ud.common_aliases == []


class TestUnitConverter:
    @pytest.fixture
    def converter(self):
        return UnitConverter()

    def test_init(self, converter):
        assert converter.units is not None
        assert converter.dimension_map is not None
        assert len(converter.units) > 0

    def test_define_units_keys(self, converter):
        assert "A/m\u00b2" in converter.units
        assert "mA/cm\u00b2" in converter.units
        assert "mM" in converter.units
        assert "V" in converter.units
        assert "s" in converter.units

    def test_dimension_map_structure(self, converter):
        assert "current_density" in converter.dimension_map
        assert "concentration" in converter.dimension_map
        assert "length" in converter.dimension_map

    def test_normalize_unit_direct(self, converter):
        assert converter.normalize_unit("V") == "V"
        assert converter.normalize_unit("mM") == "mM"

    def test_normalize_unit_alias(self, converter):
        assert converter.normalize_unit("mA/cm2") == "mA/cm\u00b2"
        assert converter.normalize_unit("mol/L") == "M"
        assert converter.normalize_unit("uM") == "\u03bcM"

    def test_normalize_unit_case_insensitive_direct(self, converter):
        result = converter.normalize_unit("v")
        assert result == "V"

    def test_normalize_unit_case_insensitive_alias(self, converter):
        result = converter.normalize_unit("MA/CM2")
        assert result == "mA/cm\u00b2"

    def test_normalize_unit_not_found(self, converter):
        assert converter.normalize_unit("unknown_unit") is None

    def test_convert_same_unit(self, converter):
        result = converter.convert(100.0, "mM", "mM")
        assert result == pytest.approx(100.0)

    def test_convert_concentration(self, converter):
        result = converter.convert(1.0, "M", "mM")
        assert result == pytest.approx(1000.0)

    def test_convert_length(self, converter):
        result = converter.convert(1.0, "m", "cm")
        assert result == pytest.approx(100.0)

    def test_convert_time(self, converter):
        result = converter.convert(1.0, "h", "min")
        assert result == pytest.approx(60.0)

    def test_convert_incompatible_dimensions(self, converter):
        result = converter.convert(1.0, "V", "mM")
        assert result is None

    def test_convert_unknown_from(self, converter):
        result = converter.convert(1.0, "xyz", "mM")
        assert result is None

    def test_convert_unknown_to(self, converter):
        result = converter.convert(1.0, "mM", "xyz")
        assert result is None

    def test_get_compatible_units(self, converter):
        units = converter.get_compatible_units("V")
        assert "V" in units
        assert "mV" in units

    def test_get_compatible_units_unknown(self, converter):
        result = converter.get_compatible_units("xyz")
        assert result == []

    def test_validate_unit_valid(self, converter):
        valid, normalized = converter.validate_unit("V")
        assert valid is True
        assert normalized == "V"

    def test_validate_unit_invalid(self, converter):
        valid, normalized = converter.validate_unit("xyz")
        assert valid is False
        assert normalized is None

    def test_validate_unit_correct_dimension(self, converter):
        valid, normalized = converter.validate_unit("V", "voltage")
        assert valid is True
        assert normalized == "V"

    def test_validate_unit_wrong_dimension(self, converter):
        valid, normalized = converter.validate_unit("V", "length")
        assert valid is False
        assert normalized is None

    def test_format_value_large(self, converter):
        result = converter.format_value_with_unit(1500.0, "V")
        assert "e" in result.lower()
        assert "V" in result

    def test_format_value_small(self, converter):
        result = converter.format_value_with_unit(0.005, "V")
        assert "e" in result.lower()
        assert "V" in result

    def test_format_value_normal(self, converter):
        result = converter.format_value_with_unit(1.5, "V", precision=2)
        assert "V" in result

    def test_format_value_zero(self, converter):
        result = converter.format_value_with_unit(0, "V")
        assert "V" in result

    def test_suggest_unit_known_dimension(self, converter):
        result = converter.suggest_unit_for_dimension("voltage")
        assert result == "V"

    def test_suggest_unit_unknown_dimension(self, converter):
        result = converter.suggest_unit_for_dimension("unknown_dim")
        assert result is None

    def test_suggest_unit_no_prefer_common(self, converter):
        result = converter.suggest_unit_for_dimension("voltage", prefer_common=False)
        assert result is not None

    def test_suggest_unit_prefer_common_current_density(self, converter):
        result = converter.suggest_unit_for_dimension("current_density")
        assert result == "mA/cm\u00b2"

    def test_suggest_unit_prefer_common_no_preference(self, converter):
        result = converter.suggest_unit_for_dimension("dimensionless")
        assert result is not None

    def test_global_instance(self):
        assert UNIT_CONVERTER is not None
        assert isinstance(UNIT_CONVERTER, UnitConverter)

    def test_convert_via_alias(self, converter):
        result = converter.convert(1.0, "mol/L", "mM")
        assert result == pytest.approx(1000.0)

    def test_voltage_reference_convert(self, converter):
        result = converter.convert(1.0, "V vs Ag/AgCl", "V")
        assert result is not None
        assert result == pytest.approx(1.197)

    def test_format_negative_large(self, converter):
        result = converter.format_value_with_unit(-1500.0, "V")
        assert "V" in result
        assert "e" in result.lower()

    def test_format_negative_small(self, converter):
        result = converter.format_value_with_unit(-0.005, "V")
        assert "V" in result
