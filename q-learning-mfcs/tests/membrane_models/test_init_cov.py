"""Coverage tests for membrane_models/__init__.py targeting 98%+ coverage."""
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

import membrane_models


class TestInitImports(unittest.TestCase):
    def test_base_classes(self):
        self.assertTrue(hasattr(membrane_models, "BaseMembraneModel"))
        self.assertTrue(hasattr(membrane_models, "IonTransportMechanisms"))
        self.assertTrue(hasattr(membrane_models, "MembraneParameters"))

    def test_pem_classes(self):
        self.assertTrue(hasattr(membrane_models, "ProtonExchangeMembrane"))
        self.assertTrue(hasattr(membrane_models, "PEMParameters"))
        self.assertTrue(hasattr(membrane_models, "create_nafion_membrane"))
        self.assertTrue(hasattr(membrane_models, "create_speek_membrane"))

    def test_aem_classes(self):
        self.assertTrue(hasattr(membrane_models, "AnionExchangeMembrane"))
        self.assertTrue(hasattr(membrane_models, "AEMParameters"))
        self.assertTrue(hasattr(membrane_models, "create_aem_membrane"))

    def test_bipolar_classes(self):
        self.assertTrue(hasattr(membrane_models, "BipolarMembrane"))
        self.assertTrue(hasattr(membrane_models, "BipolarParameters"))
        self.assertTrue(hasattr(membrane_models, "create_bipolar_membrane"))

    def test_ceramic_classes(self):
        self.assertTrue(hasattr(membrane_models, "CeramicMembrane"))
        self.assertTrue(hasattr(membrane_models, "CeramicParameters"))
        self.assertTrue(hasattr(membrane_models, "create_ceramic_membrane"))

    def test_fouling_classes(self):
        self.assertTrue(hasattr(membrane_models, "FoulingModel"))
        self.assertTrue(hasattr(membrane_models, "FoulingParameters"))
        self.assertTrue(hasattr(membrane_models, "calculate_fouling_resistance"))

    def test_all_list(self):
        self.assertIsInstance(membrane_models.__all__, list)
        self.assertGreater(len(membrane_models.__all__), 10)

    def test_version(self):
        self.assertEqual(membrane_models.__version__, "1.0.0")


if __name__ == "__main__":
    unittest.main()
