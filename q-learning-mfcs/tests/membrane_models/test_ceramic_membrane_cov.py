"""Coverage tests for ceramic_membrane.py targeting 98%+ coverage."""
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
if "jax" not in sys.modules:
    _jax = MagicMock()
    _jax.numpy = MagicMock()
    for _a in dir(np):
        if not _a.startswith("_"):
            setattr(_jax.numpy, _a, getattr(np, _a))
    sys.modules["jax"] = _jax
    sys.modules["jax.numpy"] = _jax.numpy

from membrane_models.ceramic_membrane import (
    CeramicMembrane,
    CeramicParameters,
    create_ceramic_membrane,
)
from membrane_models.base_membrane import IonType


class TestCeramicParameters(unittest.TestCase):
    def test_defaults(self):
        p = CeramicParameters()
        self.assertEqual(p.ceramic_type, "Zirconia")
        self.assertAlmostEqual(p.dopant_concentration, 0.08)
        self.assertAlmostEqual(p.grain_size, 1e-6)
        self.assertAlmostEqual(p.max_operating_temp, 1273.15)
        self.assertAlmostEqual(p.thermal_expansion, 10e-6)
        self.assertAlmostEqual(p.material_cost_per_m2, 1500.0)


class TestCeramicMembrane(unittest.TestCase):
    def test_init(self):
        params = CeramicParameters()
        cm = CeramicMembrane(params)
        self.assertIn(IonType.PROTON, cm.ion_transport)

    def test_ionic_conductivity_default(self):
        cm = create_ceramic_membrane()
        cond = cm.calculate_ionic_conductivity()
        self.assertGreater(cond, 0.0)

    def test_ionic_conductivity_high_temp(self):
        cm = create_ceramic_membrane()
        cond_low = cm.calculate_ionic_conductivity(temperature=298.15)
        cond_high = cm.calculate_ionic_conductivity(temperature=1000.0)
        self.assertGreater(cond_high, cond_low)


class TestCreateCeramicMembrane(unittest.TestCase):
    def test_default(self):
        cm = create_ceramic_membrane()
        self.assertAlmostEqual(cm.thickness, 500e-6)
        self.assertAlmostEqual(cm.area, 1e-4)

    def test_custom(self):
        cm = create_ceramic_membrane(ceramic_type="Al2O3", thickness_um=300.0, area_cm2=2.0)
        self.assertEqual(cm.ceramic_params.ceramic_type, "Al2O3")
        self.assertAlmostEqual(cm.thickness, 300e-6)
        self.assertAlmostEqual(cm.area, 2e-4)


if __name__ == "__main__":
    unittest.main()
