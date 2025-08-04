"""
Comprehensive tests for metabolic model components.

Tests cover:
- Metabolic pathway database
- Membrane transport calculations
- Electron shuttle dynamics
- Integrated metabolic modeling
- Species and substrate combinations
"""

import unittest
import sys
import os
import warnings

# Suppress matplotlib backend warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add source path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from metabolic_model import MetabolicModel, PathwayDatabase, MembraneTransport, ElectronShuttleModel
from metabolic_model.pathway_database import Species, Substrate
from metabolic_model.electron_shuttles import ShuttleType


class TestPathwayDatabase(unittest.TestCase):
    """Test metabolic pathway database functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.pathway_db = PathwayDatabase()

    def test_pathway_loading(self):
        """Test loading of species-substrate pathways."""
        # Test all available combinations
        combinations = [
            (Species.GEOBACTER, Substrate.ACETATE),
            (Species.GEOBACTER, Substrate.LACTATE),
            (Species.SHEWANELLA, Substrate.LACTATE),
            (Species.SHEWANELLA, Substrate.ACETATE)
        ]

        for species, substrate in combinations:
            pathway = self.pathway_db.get_pathway(species, substrate)

            # Verify pathway properties
            self.assertIsNotNone(pathway)
            self.assertGreater(len(pathway.reactions), 0)
            self.assertGreater(pathway.electron_yield, 0)
            self.assertGreaterEqual(pathway.energy_yield, 0)

    def test_reaction_properties(self):
        """Test metabolic reaction properties."""
        pathway = self.pathway_db.get_pathway(Species.GEOBACTER, Substrate.ACETATE)

        for reaction in pathway.reactions:
            # Verify required properties
            self.assertIsInstance(reaction.id, str)
            self.assertIsInstance(reaction.name, str)
            self.assertIsInstance(reaction.stoichiometry, dict)
            self.assertGreater(reaction.vmax, 0)
            self.assertIsInstance(reaction.km_values, dict)
            self.assertIsInstance(reaction.reversible, bool)

            # Verify bounds
            self.assertLessEqual(reaction.lb, reaction.ub)

    def test_stoichiometry_calculation(self):
        """Test overall pathway stoichiometry calculation."""
        net_stoich = self.pathway_db.calculate_pathway_stoichiometry(
            Species.GEOBACTER, Substrate.ACETATE
        )

        # Should have net electron production
        self.assertIn("electron_anode", net_stoich)
        self.assertGreater(net_stoich.get("electron_anode", 0), 0)

    def test_metabolite_properties(self):
        """Test metabolite property retrieval."""
        metabolites = ["acetate", "lactate", "pyruvate", "nadh"]

        for metabolite in metabolites:
            props = self.pathway_db.get_metabolite_properties(metabolite)

            self.assertIn("formula", props)
            self.assertIn("mw", props)
            self.assertIn("charge", props)
            self.assertIn("kegg_id", props)

    def test_kegg_pathway_ids(self):
        """Test KEGG pathway ID retrieval."""
        kegg_ids = self.pathway_db.get_kegg_pathway_ids(Species.GEOBACTER)

        self.assertIsInstance(kegg_ids, dict)
        self.assertIn("central_metabolism", kegg_ids)
        self.assertIn("electron_transport", kegg_ids)


class TestMembraneTransport(unittest.TestCase):
    """Test membrane transport model functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.membrane_117 = MembraneTransport("Nafion-117", use_gpu=False)
        self.membrane_212 = MembraneTransport("Nafion-212", use_gpu=False)

    def test_membrane_properties(self):
        """Test membrane property loading."""
        props_117 = self.membrane_117.get_membrane_properties()
        props_212 = self.membrane_212.get_membrane_properties()

        # Verify thickness difference
        self.assertGreater(props_117["thickness_um"], props_212["thickness_um"])

        # Verify common properties
        for props in [props_117, props_212]:
            self.assertIn("density", props)
            self.assertIn("base_conductivity", props)
            self.assertIn("oxygen_permeability", props)

    def test_oxygen_crossover(self):
        """Test oxygen crossover calculations."""
        anode_o2 = 0.001  # mol/m³
        cathode_o2 = 0.25  # mol/m³ (air-saturated)

        flux = self.membrane_117.calculate_oxygen_crossover(
            anode_o2, cathode_o2, temperature=303.0
        )

        # Flux should be positive (cathode to anode)
        self.assertGreater(flux, 0)

        # Thinner membrane should have higher flux
        flux_212 = self.membrane_212.calculate_oxygen_crossover(
            anode_o2, cathode_o2, temperature=303.0
        )
        self.assertGreater(flux_212, flux)

    def test_proton_conductivity(self):
        """Test proton conductivity with environmental effects."""
        # Base conductivity
        conductivity_base = self.membrane_117.calculate_proton_conductivity(
            temperature=303.0, relative_humidity=100.0
        )

        # Temperature effect
        conductivity_hot = self.membrane_117.calculate_proton_conductivity(
            temperature=323.0, relative_humidity=100.0
        )
        self.assertGreater(conductivity_hot, conductivity_base)

        # Humidity effect
        conductivity_dry = self.membrane_117.calculate_proton_conductivity(
            temperature=303.0, relative_humidity=50.0
        )
        self.assertLess(conductivity_dry, conductivity_base)

    def test_membrane_resistance(self):
        """Test membrane resistance calculations."""
        area = 0.01  # m²

        resistance = self.membrane_117.calculate_membrane_resistance(
            area, temperature=303.0, relative_humidity=100.0
        )

        self.assertGreater(resistance, 0)

        # Larger area should have lower resistance
        resistance_large = self.membrane_117.calculate_membrane_resistance(
            area * 10, temperature=303.0, relative_humidity=100.0
        )
        self.assertLess(resistance_large, resistance)

    def test_water_transport(self):
        """Test water transport calculations."""
        current_density = 1000  # A/m²

        water_flux = self.membrane_117.calculate_water_transport(
            current_density, temperature=303.0
        )

        self.assertGreater(water_flux, 0)

        # Higher current should transport more water
        water_flux_high = self.membrane_117.calculate_water_transport(
            current_density * 2, temperature=303.0
        )
        self.assertGreater(water_flux_high, water_flux)

    def test_efficiency_loss(self):
        """Test oxygen crossover efficiency loss."""
        oxygen_flux = 1e-6  # mol/m²/s
        area = 0.01  # m²
        substrate_flux = 1e-5  # mol/m²/s

        loss = self.membrane_117.calculate_oxygen_consumption_loss(
            oxygen_flux, area, substrate_flux
        )

        self.assertGreaterEqual(loss, 0)
        self.assertLessEqual(loss, 1)


class TestElectronShuttles(unittest.TestCase):
    """Test electron shuttle model functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.shuttle_model = ElectronShuttleModel()

    def test_shuttle_properties(self):
        """Test electron shuttle property loading."""
        # Test key shuttles
        shuttles = [ShuttleType.FLAVIN_MONONUCLEOTIDE, ShuttleType.RIBOFLAVIN]

        for shuttle_type in shuttles:
            props = self.shuttle_model.get_shuttle_properties(shuttle_type)

            self.assertIn("molecular_weight", props)
            self.assertIn("redox_potential", props)
            self.assertIn("electrons_transferred", props)
            self.assertGreater(props["electrons_transferred"], 0)

    def test_shuttle_production(self):
        """Test shuttle production calculations."""
        # S. oneidensis produces flavins
        production = self.shuttle_model.calculate_shuttle_production(
            species="shewanella_oneidensis",
            biomass=10.0,  # g/L
            growth_rate=0.1,  # 1/h
            dt=1.0  # h
        )

        # Should produce flavins
        self.assertGreater(production[ShuttleType.RIBOFLAVIN], 0)
        self.assertGreater(production[ShuttleType.FLAVIN_MONONUCLEOTIDE], 0)

        # G. sulfurreducens doesn't produce flavins significantly
        production_geo = self.shuttle_model.calculate_shuttle_production(
            species="geobacter_sulfurreducens",
            biomass=10.0,
            growth_rate=0.1,
            dt=1.0
        )

        self.assertEqual(production_geo[ShuttleType.RIBOFLAVIN], 0)

    def test_shuttle_degradation(self):
        """Test shuttle degradation kinetics."""
        # Set initial concentrations
        self.shuttle_model.shuttle_concentrations[ShuttleType.RIBOFLAVIN] = 1.0

        # Calculate degradation
        degradation = self.shuttle_model.calculate_shuttle_degradation(dt=1.0)

        self.assertGreater(degradation[ShuttleType.RIBOFLAVIN], 0)
        self.assertLess(degradation[ShuttleType.RIBOFLAVIN], 1.0)

    def test_electron_transfer_rate(self):
        """Test electron transfer rate calculations."""
        concentration = 0.1  # mmol/L
        electrode_potential = 0.0  # V vs SHE

        # Riboflavin should transfer electrons at positive potentials
        rate = self.shuttle_model.calculate_electron_transfer_rate(
            ShuttleType.RIBOFLAVIN,
            concentration,
            electrode_potential
        )

        self.assertGreater(rate, 0)

        # No transfer at very negative potentials
        rate_negative = self.shuttle_model.calculate_electron_transfer_rate(
            ShuttleType.RIBOFLAVIN,
            concentration,
            -0.5  # V vs SHE
        )

        self.assertEqual(rate_negative, 0)

    def test_shuttle_dynamics_update(self):
        """Test complete shuttle dynamics update."""
        # Initial state
        self.shuttle_model.reset_concentrations()

        # Production
        production = {ShuttleType.RIBOFLAVIN: 0.5}
        degradation = {ShuttleType.RIBOFLAVIN: 0.1}
        consumption = {ShuttleType.RIBOFLAVIN: 0.2}

        # Update
        self.shuttle_model.update_shuttle_concentrations(
            production, degradation, consumption
        )

        # Net change should be 0.5 - 0.1 - 0.2 = 0.2
        expected = 0.2
        actual = self.shuttle_model.shuttle_concentrations[ShuttleType.RIBOFLAVIN]
        self.assertAlmostEqual(actual, expected, places=5)

    def test_species_shuttle_efficiency_coverage(self):
        """Test electron shuttle efficiency methods for coverage."""
        # Test get_species_shuttle_efficiency for unknown species
        unknown_efficiencies = self.shuttle_model.get_species_shuttle_efficiency("unknown_species")
        
        # Should return dict with efficiency 0.0 for all shuttles the model knows about
        self.assertIsInstance(unknown_efficiencies, dict)
        # All returned efficiencies should be 0.0 for unknown species
        for shuttle_type, efficiency in unknown_efficiencies.items():
            self.assertEqual(efficiency, 0.0)
        
        # Test for known species
        known_efficiencies = self.shuttle_model.get_species_shuttle_efficiency("shewanella_oneidensis")
        self.assertIsInstance(known_efficiencies, dict)
        # At least some efficiency should be non-zero for known species
        has_nonzero = any(eff > 0.0 for eff in known_efficiencies.values())
        self.assertTrue(has_nonzero)
    
    def test_estimate_optimal_shuttle_concentration(self):
        """Test optimal shuttle concentration estimation."""
        # Test the estimation method with required area parameter
        optimal_conc = self.shuttle_model.estimate_optimal_shuttle_concentration(
            target_current=0.1,  # A
            electrode_potential=0.2,  # V
            volume=0.001,  # m³
            area=0.01  # m²
        )
        
        # Should return dict with concentrations
        self.assertIsInstance(optimal_conc, dict)
        
        # All concentrations should be non-negative
        for shuttle_type, concentration in optimal_conc.items():
            self.assertGreaterEqual(concentration, 0.0)


class TestMetabolicModel(unittest.TestCase):
    """Test integrated metabolic model functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.model_geo_acetate = MetabolicModel(
            species="geobacter", substrate="acetate", use_gpu=False
        )
        self.model_she_lactate = MetabolicModel(
            species="shewanella", substrate="lactate", use_gpu=False
        )
        self.model_mixed = MetabolicModel(
            species="mixed", substrate="lactate", use_gpu=False
        )

    def test_model_initialization(self):
        """Test model initialization with different configurations."""
        models = [self.model_geo_acetate, self.model_she_lactate, self.model_mixed]

        for model in models:
            # Verify components initialized
            self.assertIsNotNone(model.pathway_db)
            self.assertIsNotNone(model.membrane_model)
            self.assertIsNotNone(model.shuttle_model)

            # Verify metabolites initialized
            self.assertIn("nadh", model.metabolites)
            self.assertIn("atp", model.metabolites)

    def test_metabolic_flux_calculation(self):
        """Test metabolic flux calculations."""
        fluxes = self.model_geo_acetate.calculate_metabolic_fluxes(
            biomass=10.0,
            growth_rate=0.1,
            anode_potential=-0.2,
            substrate_supply=1.0
        )

        # Should have flux values for all reactions
        self.assertGreater(len(fluxes), 0)

        # Fluxes should be within bounds
        for flux in fluxes.values():
            self.assertIsInstance(flux, (int, float))
            self.assertGreaterEqual(flux, -50)  # Reasonable lower bound
            self.assertLessEqual(flux, 50)      # Reasonable upper bound

    def test_metabolite_update(self):
        """Test metabolite concentration updates."""
        # Initial acetate
        initial_acetate = self.model_geo_acetate.metabolites["acetate"]

        # Create test fluxes
        fluxes = {"GSU_R001": 1.0}  # Acetate consumption

        # Update
        self.model_geo_acetate.update_metabolite_concentrations(
            fluxes, dt=1.0, volume=1.0
        )

        # Acetate should change
        final_acetate = self.model_geo_acetate.metabolites["acetate"]
        self.assertNotEqual(initial_acetate, final_acetate)

    def test_oxygen_crossover_effects(self):
        """Test oxygen crossover calculations."""
        o2_rate = self.model_geo_acetate.calculate_oxygen_crossover_effects(
            cathode_o2_conc=0.25,  # mol/m³
            membrane_area=0.01,    # m²
            temperature=303.0
        )

        self.assertGreater(o2_rate, 0)

    def test_current_output_calculation(self):
        """Test current output calculations."""
        # Set some fluxes
        self.model_geo_acetate.fluxes = {"GSU_R004": 10.0}  # Electron transport

        direct, mediated = self.model_geo_acetate.calculate_current_output(
            biomass=10.0,
            volume=1.0,
            electrode_area=0.01
        )

        # Should have some direct current
        self.assertGreaterEqual(direct, 0)
        self.assertGreaterEqual(mediated, 0)

    def test_coulombic_efficiency(self):
        """Test coulombic efficiency calculation."""
        efficiency = self.model_geo_acetate.calculate_coulombic_efficiency(
            current_output=0.1,  # A
            substrate_consumed=1.0,  # mmol
            dt=1.0  # h
        )

        self.assertGreaterEqual(efficiency, 0)
        self.assertLessEqual(efficiency, 1)

    def test_full_metabolic_step(self):
        """Test complete metabolic time step."""
        state = self.model_she_lactate.step_metabolism(
            dt=0.1,
            biomass=10.0,
            growth_rate=0.1,
            anode_potential=-0.2,
            substrate_supply=1.0,
            cathode_o2_conc=0.25,
            membrane_area=0.01,
            volume=1.0,
            electrode_area=0.01
        )

        # Verify state structure
        self.assertIsInstance(state.metabolites, dict)
        self.assertIsInstance(state.fluxes, dict)
        self.assertGreaterEqual(state.atp_production, 0)
        self.assertGreaterEqual(state.coulombic_efficiency, 0)
        self.assertLessEqual(state.coulombic_efficiency, 1)

    def test_metabolic_summary(self):
        """Test metabolic summary generation."""
        summary = self.model_mixed.get_metabolic_summary()

        expected_keys = [
            "species", "substrate", "is_mixed_culture",
            "substrate_concentration", "nadh_ratio", "atp_level",
            "coulombic_efficiency"
        ]

        for key in expected_keys:
            self.assertIn(key, summary)

    def test_species_substrate_combinations(self):
        """Test different species-substrate combinations."""
        combinations = [
            ("geobacter", "acetate"),
            ("geobacter", "lactate"),
            ("shewanella", "lactate"),
            ("shewanella", "acetate"),
            ("mixed", "lactate")
        ]

        for species, substrate in combinations:
            model = MetabolicModel(species=species, substrate=substrate, use_gpu=False)

            # Should initialize without errors
            self.assertEqual(model.species_str, species)
            self.assertEqual(model.substrate_str, substrate)

            # Should have appropriate substrate concentration
            if substrate == "acetate":
                self.assertGreater(model.metabolites["acetate"], 0)
            else:
                self.assertGreater(model.metabolites["lactate"], 0)


class TestIntegration(unittest.TestCase):
    """Test integration between model components."""

    def test_shuttle_membrane_interaction(self):
        """Test interaction between shuttles and membrane."""
        model = MetabolicModel(species="shewanella", substrate="lactate", use_gpu=False)

        # Run a few steps to produce shuttles
        for _ in range(5):
            model.step_metabolism(
                dt=0.1,
                biomass=20.0,
                growth_rate=0.15,
                anode_potential=-0.1,
                substrate_supply=2.0,
                cathode_o2_conc=0.25,
                membrane_area=0.01,
                volume=1.0,
                electrode_area=0.01
            )

        # Should have produced some shuttles
        dominant = model.shuttle_model.get_dominant_shuttle()
        if dominant:
            conc = model.shuttle_model.shuttle_concentrations[dominant]
            self.assertGreater(conc, 0)

    def test_mixed_culture_metabolism(self):
        """Test mixed culture metabolic dynamics."""
        model = MetabolicModel(species="mixed", substrate="lactate", use_gpu=False)

        # Mixed culture should handle lactate well
        state = model.step_metabolism(
            dt=1.0,
            biomass=15.0,
            growth_rate=0.12,
            anode_potential=-0.15,
            substrate_supply=1.5,
            cathode_o2_conc=0.2,
            membrane_area=0.01,
            volume=1.0,
            electrode_area=0.01
        )

        # Should have reasonable efficiency
        self.assertGreater(state.coulombic_efficiency, 0)

        # Should consume substrate
        self.assertLess(model.metabolites["lactate"], 10.0)

    def test_long_term_stability(self):
        """Test model stability over extended simulation."""
        model = MetabolicModel(species="geobacter", substrate="acetate", use_gpu=False)

        # Run for 24 hours
        states = []
        for hour in range(24):
            state = model.step_metabolism(
                dt=1.0,
                biomass=10.0,
                growth_rate=0.05,
                anode_potential=-0.2,
                substrate_supply=0.5,
                cathode_o2_conc=0.25,
                membrane_area=0.01,
                volume=1.0,
                electrode_area=0.01
            )
            states.append(state)

        # Should maintain reasonable values
        final_state = states[-1]

        # Metabolites should be non-negative
        for metabolite, conc in final_state.metabolites.items():
            self.assertGreaterEqual(conc, 0, f"{metabolite} went negative")

        # Should have consumed substrate
        self.assertLess(final_state.substrate_utilization, 1.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
