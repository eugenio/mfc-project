"""Gap coverage tests for pathway_database.py -- covers missing lines."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from metabolic_model.pathway_database import PathwayDatabase, Species, Substrate


class TestGetPathwayKeyError(unittest.TestCase):
    """Cover lines 494-499: KeyError for unknown species/substrate combination."""

    def test_invalid_combination(self):
        db = PathwayDatabase()
        # Patch _pathways to remove a key
        key = (Species.GEOBACTER, Substrate.ACETATE)
        saved = db._pathways.pop(key)
        with self.assertRaises(KeyError) as ctx:
            db.get_pathway(Species.GEOBACTER, Substrate.ACETATE)
        self.assertIn("not available", str(ctx.exception))
        db._pathways[key] = saved


class TestGetMetaboliteNotFound(unittest.TestCase):
    """Cover lines 508-509: KeyError for unknown metabolite."""

    def test_unknown_metabolite(self):
        db = PathwayDatabase()
        with self.assertRaises(KeyError) as ctx:
            db.get_metabolite_properties("unknown_metabolite_xyz")
        self.assertIn("not found", str(ctx.exception))


class TestGetElectronCarrierNotFound(unittest.TestCase):
    """Cover lines 514-517: KeyError for unknown electron carrier."""

    def test_unknown_carrier(self):
        db = PathwayDatabase()
        with self.assertRaises(KeyError) as ctx:
            db.get_electron_carrier_properties("unknown_carrier_xyz")
        self.assertIn("not found", str(ctx.exception))


class TestGetAvailableCombinations(unittest.TestCase):
    """Cover line 548."""

    def test_available(self):
        db = PathwayDatabase()
        combos = db.get_available_combinations()
        self.assertIsInstance(combos, list)
        self.assertGreater(len(combos), 0)
        self.assertIn((Species.GEOBACTER, Substrate.ACETATE), combos)
        self.assertIn((Species.SHEWANELLA, Substrate.LACTATE), combos)


class TestGetElectronCarrierValid(unittest.TestCase):
    def test_cytochrome_c(self):
        db = PathwayDatabase()
        props = db.get_electron_carrier_properties("cytochrome_c")
        self.assertIn("redox_potential", props)
        self.assertIn("molecular_weight", props)

    def test_flavin_mononucleotide(self):
        db = PathwayDatabase()
        props = db.get_electron_carrier_properties("flavin_mononucleotide")
        self.assertIn("location", props)

    def test_riboflavin(self):
        db = PathwayDatabase()
        props = db.get_electron_carrier_properties("riboflavin")
        self.assertEqual(props["location"], "extracellular")


if __name__ == "__main__":
    unittest.main()
