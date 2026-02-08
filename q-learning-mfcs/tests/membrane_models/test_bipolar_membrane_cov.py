"""Coverage tests for bipolar_membrane.py targeting 98%+ coverage."""
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

from membrane_models.bipolar_membrane import (
    BipolarMembrane,
    BipolarParameters,
    create_bipolar_membrane,
)
from membrane_models.base_membrane import IonType


class TestBipolarParameters(unittest.TestCase):
    def test_defaults(self):
        p = BipolarParameters()
        self.assertAlmostEqual(p.pem_thickness_fraction, 0.4)
        self.assertAlmostEqual(p.aem_thickness_fraction, 0.4)
        self.assertAlmostEqual(p.junction_thickness_fraction, 0.2)
        self.assertAlmostEqual(p.water_splitting_efficiency, 0.9)
        self.assertAlmostEqual(p.water_splitting_voltage, 1.2)
        self.assertAlmostEqual(p.interface_resistance, 1e-3)
        self.assertAlmostEqual(p.catalytic_activity, 1.0)


class TestBipolarMembrane(unittest.TestCase):
    def test_init(self):
        params = BipolarParameters()
        bm = BipolarMembrane(params)
        self.assertIn(IonType.PROTON, bm.ion_transport)
        self.assertIn(IonType.HYDROXIDE, bm.ion_transport)

    def test_ionic_conductivity(self):
        bm = create_bipolar_membrane()
        cond = bm.calculate_ionic_conductivity()
        self.assertAlmostEqual(cond, (10.0 + 4.0) / 2)

    def test_ionic_conductivity_with_args(self):
        bm = create_bipolar_membrane()
        cond = bm.calculate_ionic_conductivity(temperature=350.0, water_content=12.0)
        self.assertAlmostEqual(cond, 7.0)


class TestCreateBipolarMembrane(unittest.TestCase):
    def test_default(self):
        bm = create_bipolar_membrane()
        self.assertAlmostEqual(bm.thickness, 200e-6)
        self.assertAlmostEqual(bm.area, 1e-4)

    def test_custom(self):
        bm = create_bipolar_membrane(thickness_um=300.0, area_cm2=5.0)
        self.assertAlmostEqual(bm.thickness, 300e-6)
        self.assertAlmostEqual(bm.area, 5e-4)


if __name__ == "__main__":
    unittest.main()
