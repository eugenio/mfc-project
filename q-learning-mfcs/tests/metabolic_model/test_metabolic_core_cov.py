"""Gap coverage tests for metabolic_core.py -- covers missing lines."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from metabolic_model.metabolic_core import MetabolicModel
from metabolic_model.pathway_database import Species, Substrate


class TestMetabolicFluxesKiInhibition(unittest.TestCase):
    """Cover lines 302-309: ki as float, ki as dict with matching metabolites."""

    def test_ki_float_inhibition(self):
        """Cover ki isinstance float branch (line 305-309)."""
        model = MetabolicModel(species="geobacter", substrate="acetate", use_gpu=False)
        # The default pathway reactions may have ki_values as dict (empty).
        # We need to trigger the ki float path.
        # Modify a reaction's ki_values to be a float to trigger that code path.
        for reaction in model.current_pathway.reactions:
            reaction.ki_values = 0.5  # Set as float
        model.metabolites["o2"] = 0.01
        fluxes = model.calculate_metabolic_fluxes(
            biomass=10.0, growth_rate=0.1, anode_potential=-0.2, substrate_supply=1.0
        )
        self.assertGreater(len(fluxes), 0)

    def test_ki_dict_inhibition(self):
        """Cover ki dict branch with matching metabolites (line 302-304)."""
        model = MetabolicModel(species="geobacter", substrate="acetate", use_gpu=False)
        for reaction in model.current_pathway.reactions:
            reaction.ki_values = {"o2": 0.01}
        model.metabolites["o2"] = 0.005
        fluxes = model.calculate_metabolic_fluxes(
            biomass=10.0, growth_rate=0.1, anode_potential=-0.2, substrate_supply=1.0
        )
        self.assertGreater(len(fluxes), 0)

    def test_km_float_with_substrate(self):
        """Cover km isinstance float branch (lines 290-297)."""
        model = MetabolicModel(species="geobacter", substrate="acetate", use_gpu=False)
        for reaction in model.current_pathway.reactions:
            reaction.km_values = 0.5  # Set km as float
        model.metabolites["acetate"] = 5.0
        fluxes = model.calculate_metabolic_fluxes(
            biomass=10.0, growth_rate=0.1, anode_potential=-0.2, substrate_supply=1.0
        )
        self.assertGreater(len(fluxes), 0)


class TestCoulombicEfficiencyGap(unittest.TestCase):
    """Cover line 562: substrate_config degradation_pathways branch."""

    def test_efficiency_with_zero_substrate(self):
        model = MetabolicModel(species="geobacter", substrate="acetate", use_gpu=False)
        eff = model.calculate_coulombic_efficiency(
            current_output=0.0, substrate_consumed=0.0, dt=1.0
        )
        self.assertEqual(eff, 0.0)

    def test_efficiency_unrealistic_high(self):
        """Cover line 584-594: out-of-range efficiency forcing proxy calculation."""
        model = MetabolicModel(species="geobacter", substrate="acetate", use_gpu=False)
        # Very high current vs small substrate -> efficiency > 1.5
        eff = model.calculate_coulombic_efficiency(
            current_output=100.0, substrate_consumed=0.001, dt=1.0
        )
        self.assertGreater(eff, 0.0)
        self.assertLessEqual(eff, 1.0)

    def test_efficiency_unrealistic_low(self):
        """Cover the efficiency < 0.001 branch."""
        model = MetabolicModel(species="geobacter", substrate="acetate", use_gpu=False)
        eff = model.calculate_coulombic_efficiency(
            current_output=1e-10, substrate_consumed=100.0, dt=1.0
        )
        self.assertGreater(eff, 0.0)


class TestOptimizeForCurrentProduction(unittest.TestCase):
    """Cover lines 785-796."""

    def test_optimize(self):
        model = MetabolicModel(species="geobacter", substrate="acetate", use_gpu=False)
        optimal = model.optimize_for_current_production(
            target_current=10.0, constraints={}
        )
        self.assertIsInstance(optimal, dict)
        self.assertGreater(len(optimal), 0)
        # Reactions with electron_anode should have ub
        for reaction in model.current_pathway.reactions:
            if reaction.stoichiometry.get("electron_anode", 0) > 0:
                self.assertAlmostEqual(optimal[reaction.id], reaction.ub)
            else:
                self.assertAlmostEqual(
                    optimal[reaction.id], (reaction.lb + reaction.ub) / 2
                )


class TestCoulombicEfficiencyNoConfig(unittest.TestCase):
    """Ensure line 562 is covered: else branch with no substrate_config."""

    def test_no_substrate_config(self):
        model = MetabolicModel(species="geobacter", substrate="acetate", use_gpu=False)
        model.substrate_config = None
        eff = model.calculate_coulombic_efficiency(
            current_output=0.05, substrate_consumed=1.0, dt=1.0
        )
        self.assertGreater(eff, 0.0)
        self.assertLessEqual(eff, 1.0)


class TestResetState(unittest.TestCase):
    def test_reset(self):
        model = MetabolicModel(species="shewanella", substrate="lactate", use_gpu=False)
        model.metabolites["lactate"] = 0.0
        model.reset_state()
        self.assertGreater(model.metabolites["lactate"], 0.0)


if __name__ == "__main__":
    unittest.main()
