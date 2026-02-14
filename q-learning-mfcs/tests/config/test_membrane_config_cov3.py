"""Tests for membrane_config.py - coverage part 3.

Targets missing line 238: dead code after assert in create_membrane_config.
"""
import sys
import os
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from config.membrane_config import (
    MEMBRANE_PROPERTIES_DATABASE,
    MembraneConfiguration,
    MembraneMaterial,
    MembraneProperties,
    create_membrane_config,
)


@pytest.mark.coverage_extra
class TestCreateMembraneConfigMaterialNotInDB:
    """Cover line 238: properties is None after database lookup."""

    def test_material_not_in_database_raises(self):
        """When a non-CUSTOM material is not in the database, should raise.

        Line 238 is dead code because the assert on line 236 fires first.
        We need to bypass the assert to hit line 238.
        """
        # The assert on line 236 will fire before line 238
        # Use a material not in the database - but all non-CUSTOM materials
        # in the enum ARE in the database except some.
        # Let's patch the database to be empty to trigger line 236 assertion
        with patch.dict(
            "config.membrane_config.MEMBRANE_PROPERTIES_DATABASE",
            {},
            clear=True,
        ):
            with pytest.raises(AssertionError):
                create_membrane_config(MembraneMaterial.NAFION_117, 0.001)


@pytest.mark.coverage_extra
class TestCreateMembraneConfigVariants:
    """Test all materials in the database for coverage."""

    def test_nafion_115_not_in_db(self):
        """NAFION_115 is not in the database."""
        with pytest.raises(AssertionError):
            create_membrane_config(MembraneMaterial.NAFION_115, 0.001)

    def test_fumasep_faa_not_in_db(self):
        """FUMASEP_FAA is not in the database."""
        with pytest.raises(AssertionError):
            create_membrane_config(MembraneMaterial.FUMASEP_FAA, 0.001)

    def test_cellulose_acetate_not_in_db(self):
        """CELLULOSE_ACETATE is not in the database."""
        with pytest.raises(AssertionError):
            create_membrane_config(MembraneMaterial.CELLULOSE_ACETATE, 0.001)

    def test_bipolar_membrane_not_in_db(self):
        """BIPOLAR_MEMBRANE is not in the database."""
        with pytest.raises(AssertionError):
            create_membrane_config(MembraneMaterial.BIPOLAR_MEMBRANE, 0.001)

    def test_ceramic_separator_not_in_db(self):
        """CERAMIC_SEPARATOR is not in the database."""
        with pytest.raises(AssertionError):
            create_membrane_config(MembraneMaterial.CERAMIC_SEPARATOR, 0.001)


@pytest.mark.coverage_extra
class TestMembraneEnumValues:
    """Cover all MembraneMaterial enum values."""

    def test_all_values(self):
        assert MembraneMaterial.NAFION_117.value == "nafion_117"
        assert MembraneMaterial.NAFION_112.value == "nafion_112"
        assert MembraneMaterial.NAFION_115.value == "nafion_115"
        assert MembraneMaterial.ULTREX_CMI_7000.value == "ultrex_cmi_7000"
        assert MembraneMaterial.FUMASEP_FKE.value == "fumasep_fke"
        assert MembraneMaterial.FUMASEP_FAA.value == "fumasep_faa"
        assert MembraneMaterial.CELLULOSE_ACETATE.value == "cellulose_acetate"
        assert MembraneMaterial.BIPOLAR_MEMBRANE.value == "bipolar_membrane"
        assert MembraneMaterial.CERAMIC_SEPARATOR.value == "ceramic_separator"
        assert MembraneMaterial.J_CLOTH.value == "j_cloth"
        assert MembraneMaterial.CUSTOM.value == "custom"
