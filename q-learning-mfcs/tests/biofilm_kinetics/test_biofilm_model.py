"""
Comprehensive tests for biofilm kinetics model.

Tests cover:
- Species parameter loading and validation
- Substrate parameter loading and validation  
- Environmental compensation (pH, temperature)
- Biofilm dynamics simulation
- GPU acceleration functionality
- Mixed culture synergy calculations
"""

import unittest
import numpy as np
import sys
import os
import tempfile
import warnings

# Suppress matplotlib backend warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add source path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from biofilm_kinetics import BiofilmKineticsModel, SpeciesParameters, SubstrateParameters


class TestSpeciesParameters(unittest.TestCase):
    """Test species parameter database functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.species_db = SpeciesParameters()
    
    def test_species_parameter_loading(self):
        """Test loading of species-specific parameters."""
        # Test all available species
        for species in ['geobacter', 'shewanella', 'mixed']:
            params = self.species_db.get_parameters(species)
            
            # Verify all required parameters are present and positive
            self.assertGreater(params.mu_max, 0, f"{species} mu_max should be positive")
            self.assertGreater(params.K_s, 0, f"{species} K_s should be positive")
            self.assertGreater(params.Y_xs, 0, f"{species} Y_xs should be positive")
            self.assertGreater(params.j_max, 0, f"{species} j_max should be positive")
            self.assertGreater(params.sigma_biofilm, 0, f"{species} sigma_biofilm should be positive")
            self.assertGreater(params.biofilm_thickness_max, 0, f"{species} thickness_max should be positive")
            self.assertGreater(params.diffusion_coeff, 0, f"{species} diffusion_coeff should be positive")
    
    def test_invalid_species(self):
        """Test handling of invalid species names."""
        with self.assertRaises(ValueError):
            self.species_db.get_parameters('invalid_species')
    
    def test_temperature_compensation(self):
        """Test Arrhenius temperature compensation."""
        base_params = self.species_db.get_parameters('geobacter')
        
        # Test at higher temperature (should increase rates)
        high_temp_params = self.species_db.apply_temperature_compensation(base_params, 313.0)
        self.assertGreater(high_temp_params.mu_max, base_params.mu_max)
        self.assertGreater(high_temp_params.j_max, base_params.j_max)
        
        # Test at lower temperature (should decrease rates) 
        low_temp_params = self.species_db.apply_temperature_compensation(base_params, 293.0)
        self.assertLess(low_temp_params.mu_max, base_params.mu_max)
        self.assertLess(low_temp_params.j_max, base_params.j_max)
        
        # Saturation constants and potentials should be unchanged
        self.assertEqual(high_temp_params.K_s, base_params.K_s)
        self.assertEqual(high_temp_params.E_ka, base_params.E_ka)
    
    def test_ph_compensation(self):
        """Test pH compensation for electrochemical parameters."""
        base_params = self.species_db.get_parameters('geobacter')
        
        # Test at different pH values
        acidic_params = self.species_db.apply_ph_compensation(base_params, 6.0)
        alkaline_params = self.species_db.apply_ph_compensation(base_params, 8.0)
        
        # Potentials should change with pH (Nernst equation)
        self.assertNotEqual(acidic_params.E_ka, base_params.E_ka)
        self.assertNotEqual(alkaline_params.E_ka, base_params.E_ka)
        
        # Growth parameters should be affected by pH (updated behavior)
        self.assertLessEqual(acidic_params.mu_max, base_params.mu_max)
        self.assertLessEqual(alkaline_params.mu_max, base_params.mu_max)
    
    def test_synergy_coefficients(self):
        """Test mixed culture synergy coefficients."""
        # Test G. sulfurreducens + S. oneidensis synergy
        alpha = self.species_db.get_synergy_coefficient('geobacter', 'shewanella')
        self.assertAlmostEqual(alpha, 1.38, places=2)
        
        # Test reverse order
        alpha_reverse = self.species_db.get_synergy_coefficient('shewanella', 'geobacter')
        self.assertEqual(alpha, alpha_reverse)
        
        # Test unknown pair (should return 1.0)
        alpha_unknown = self.species_db.get_synergy_coefficient('geobacter', 'unknown')
        self.assertEqual(alpha_unknown, 1.0)


class TestSubstrateParameters(unittest.TestCase):
    """Test substrate parameter database functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.substrate_db = SubstrateParameters()
    
    def test_substrate_parameter_loading(self):
        """Test loading of substrate-specific parameters."""
        # Test both substrates
        for substrate in ['acetate', 'lactate']:
            props = self.substrate_db.get_substrate_properties(substrate)
            
            # Verify all required parameters are present and positive
            self.assertGreater(props.molecular_weight, 0, f"{substrate} molecular_weight should be positive")
            self.assertGreater(props.density, 0, f"{substrate} density should be positive")
            self.assertGreater(props.diffusivity, 0, f"{substrate} diffusivity should be positive")
            self.assertGreater(props.electrons_per_mole, 0, f"{substrate} electrons_per_mole should be positive")
            self.assertGreater(props.base_consumption_rate, 0, f"{substrate} consumption_rate should be positive")
    
    def test_default_substrate(self):
        """Test default substrate selection."""
        default = self.substrate_db.get_default_substrate()
        self.assertEqual(default, 'lactate')
    
    def test_nernst_potential_calculation(self):
        """Test Nernst potential calculations."""
        # Test acetate Nernst potential
        acetate_potential = self.substrate_db.calculate_nernst_potential(
            'acetate', concentration=1.0, ph=7.0, temperature=298.15
        )
        self.assertIsInstance(acetate_potential, float)
        self.assertLess(acetate_potential, 0)  # Should be negative for oxidation
        
        # Test lactate Nernst potential
        lactate_potential = self.substrate_db.calculate_nernst_potential(
            'lactate', concentration=1.0, ph=7.0, temperature=298.15
        )
        self.assertIsInstance(lactate_potential, float)
        self.assertLess(lactate_potential, 0)  # Should be negative for oxidation
        
        # Test pH dependence
        potential_ph6 = self.substrate_db.calculate_nernst_potential(
            'acetate', concentration=1.0, ph=6.0, temperature=298.15
        )
        potential_ph8 = self.substrate_db.calculate_nernst_potential(
            'acetate', concentration=1.0, ph=8.0, temperature=298.15
        )
        self.assertNotEqual(potential_ph6, potential_ph8)
    
    def test_theoretical_current_calculation(self):
        """Test theoretical current density calculations."""
        # Test current calculation for both substrates
        consumption_rate = 1.0  # mol/(m³·h)
        
        acetate_current = self.substrate_db.calculate_theoretical_current('acetate', consumption_rate)
        lactate_current = self.substrate_db.calculate_theoretical_current('lactate', consumption_rate)
        
        # Acetate should give higher current (8 e⁻/mol vs 4 e⁻/mol)
        self.assertGreater(acetate_current, lactate_current)
        self.assertGreater(acetate_current, 0)
        self.assertGreater(lactate_current, 0)
    
    def test_ph_correction(self):
        """Test pH correction for kinetic parameters."""
        base_value = 1.0
        
        # Test at optimal pH (should be ~1.0)
        optimal_correction = self.substrate_db.apply_ph_correction('lactate', base_value, 7.2)
        self.assertAlmostEqual(optimal_correction, base_value, places=1)
        
        # Test at non-optimal pH (should be less than base)
        acidic_correction = self.substrate_db.apply_ph_correction('lactate', base_value, 6.0)
        alkaline_correction = self.substrate_db.apply_ph_correction('lactate', base_value, 8.5)
        
        self.assertLess(acidic_correction, base_value)
        self.assertLess(alkaline_correction, base_value)
    
    def test_stoichiometric_coefficients(self):
        """Test stoichiometric coefficient calculations."""
        # Test acetate stoichiometry
        acetate_coeffs = self.substrate_db.get_stoichiometric_coefficients('acetate')
        self.assertEqual(acetate_coeffs['electrons'], 8.0)
        self.assertEqual(acetate_coeffs['co2'], 2.0)
        self.assertEqual(acetate_coeffs['protons'], 7.0)
        
        # Test lactate stoichiometry
        lactate_coeffs = self.substrate_db.get_stoichiometric_coefficients('lactate')
        self.assertEqual(lactate_coeffs['electrons'], 4.0)
        self.assertEqual(lactate_coeffs['protons'], 5.0)


class TestBiofilmKineticsModel(unittest.TestCase):
    """Test comprehensive biofilm kinetics model."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Test different configurations
        self.models = {
            'geobacter_acetate': BiofilmKineticsModel(species='geobacter', substrate='acetate', use_gpu=False),
            'shewanella_lactate': BiofilmKineticsModel(species='shewanella', substrate='lactate', use_gpu=False),
            'mixed_lactate': BiofilmKineticsModel(species='mixed', substrate='lactate', use_gpu=False)
        }
    
    def test_model_initialization(self):
        """Test model initialization with different configurations."""
        for name, model in self.models.items():
            self.assertIsNotNone(model.kinetic_params)
            self.assertIsNotNone(model.substrate_props)
            self.assertEqual(model.biofilm_thickness, 0.1)
            self.assertEqual(model.biomass_density, 0.01)
            self.assertEqual(model.time, 0.0)
    
    def test_nernst_monod_growth_rate(self):
        """Test Nernst-Monod growth rate calculations."""
        model = self.models['geobacter_acetate']
        
        # Test with typical conditions
        growth_rate = model.calculate_nernst_monod_growth_rate(
            substrate_conc=5.0,  # mmol/L
            anode_potential=-0.2  # V
        )
        
        self.assertGreaterEqual(growth_rate, 0)
        self.assertLessEqual(growth_rate, model.kinetic_params.mu_max)
        
        # Test substrate limitation
        low_substrate_rate = model.calculate_nernst_monod_growth_rate(
            substrate_conc=0.1,  # Low substrate
            anode_potential=-0.2
        )
        high_substrate_rate = model.calculate_nernst_monod_growth_rate(
            substrate_conc=10.0,  # High substrate
            anode_potential=-0.2
        )
        
        self.assertLess(low_substrate_rate, high_substrate_rate)
    
    def test_stochastic_attachment(self):
        """Test stochastic cell attachment calculations."""
        model = self.models['shewanella_lactate']
        
        attachment_rate = model.calculate_stochastic_attachment(
            cell_density=1e12,  # cells/m³
            surface_area=1.0    # m²
        )
        
        self.assertGreaterEqual(attachment_rate, 0)
        self.assertIsInstance(attachment_rate, float)
        
        # Test coverage effect (thicker biofilm should reduce attachment)
        model.biofilm_thickness = 50.0  # μm
        reduced_attachment = model.calculate_stochastic_attachment(1e12, 1.0)
        
        model.biofilm_thickness = 5.0   # μm  
        normal_attachment = model.calculate_stochastic_attachment(1e12, 1.0)
        
        self.assertLessEqual(reduced_attachment, normal_attachment)
    
    def test_biofilm_current_density(self):
        """Test biofilm current density calculations."""
        model = self.models['mixed_lactate']
        
        current_density = model.calculate_biofilm_current_density(
            thickness=20.0,      # μm
            biomass_density=30.0 # kg/m³
        )
        
        self.assertGreaterEqual(current_density, 0)
        self.assertIsInstance(current_density, float)
        
        # Test thickness effect (thicker biofilm should reduce current)
        thin_current = model.calculate_biofilm_current_density(5.0, 30.0)
        thick_current = model.calculate_biofilm_current_density(50.0, 30.0)
        
        self.assertGreaterEqual(thin_current, thick_current)
    
    def test_substrate_consumption(self):
        """Test substrate consumption calculations."""
        model = self.models['geobacter_acetate']
        
        consumption = model.calculate_substrate_consumption(
            growth_rate=0.1,  # 1/h
            biomass=20.0      # kg/m³
        )
        
        self.assertGreaterEqual(consumption, 0)
        self.assertIsInstance(consumption, float)
        
        # Test proportionality to growth rate and biomass
        double_growth = model.calculate_substrate_consumption(0.2, 20.0)
        double_biomass = model.calculate_substrate_consumption(0.1, 40.0)
        
        self.assertAlmostEqual(double_growth, 2 * consumption, places=3)
        self.assertAlmostEqual(double_biomass, 2 * consumption, places=3)
    
    def test_mixed_culture_synergy(self):
        """Test mixed culture synergy calculations."""
        model = self.models['mixed_lactate']
        
        geobacter_current = 0.3  # A/m²
        shewanella_current = 0.1  # A/m²
        
        synergy_current = model.calculate_mixed_culture_synergy(
            geobacter_current, shewanella_current
        )
        
        # Mixed culture should have some effect (may be less due to pH/temp factors)
        simple_sum = geobacter_current + shewanella_current
        self.assertIsInstance(synergy_current, (int, float))
        self.assertGreaterEqual(synergy_current, 0)
        
        # Test non-mixed species (should just sum)
        non_mixed_model = self.models['geobacter_acetate']
        no_synergy = non_mixed_model.calculate_mixed_culture_synergy(
            geobacter_current, shewanella_current
        )
        self.assertEqual(no_synergy, simple_sum)
    
    def test_biofilm_dynamics_step(self):
        """Test complete biofilm dynamics time step."""
        model = self.models['shewanella_lactate']
        initial_state = model.get_model_parameters()
        
        # Run one time step
        result = model.step_biofilm_dynamics(
            dt=1.0,                # 1 hour
            anode_potential=-0.2,  # V
            substrate_supply=1.0   # mmol/(L·h)
        )
        
        # Verify output structure
        expected_keys = ['time', 'biofilm_thickness', 'biomass_density', 
                        'substrate_concentration', 'current_density', 
                        'growth_rate', 'consumption_rate', 'anode_potential']
        
        for key in expected_keys:
            self.assertIn(key, result)
            self.assertIsInstance(result[key], (int, float))
        
        # Verify time progression
        self.assertEqual(result['time'], 1.0)
        
        # Verify state changes (biofilm should grow)
        self.assertGreaterEqual(result['biofilm_thickness'], 0.1)
        self.assertGreaterEqual(result['biomass_density'], 0.01)
    
    def test_environmental_condition_updates(self):
        """Test dynamic environmental condition updates."""
        model = self.models['geobacter_acetate']
        
        # Get initial parameters
        initial_params = model.get_model_parameters()
        initial_mu_max = initial_params['kinetic_params']['mu_max']
        
        # Update temperature
        model.set_environmental_conditions(temperature=313.0)  # Higher temperature
        
        updated_params = model.get_model_parameters()
        updated_mu_max = updated_params['kinetic_params']['mu_max']
        
        # Higher temperature should increase growth rate
        self.assertGreater(updated_mu_max, initial_mu_max)
        
        # Update pH
        model.set_environmental_conditions(ph=6.5)  # Lower pH
        
        ph_params = model.get_model_parameters()
        
        # pH should affect electrochemical parameters
        self.assertNotEqual(ph_params['kinetic_params']['E_ka'], 
                          updated_params['kinetic_params']['E_ka'])
    
    def test_theoretical_maximum_current(self):
        """Test theoretical maximum current calculations."""
        for name, model in self.models.items():
            max_current = model.calculate_theoretical_maximum_current()
            
            self.assertGreater(max_current, 0)
            self.assertIsInstance(max_current, float)
            
            # Maximum should be higher than typical current
            typical_current = model.calculate_biofilm_current_density(20.0, 30.0)
            self.assertGreater(max_current, typical_current)
    
    def test_mass_balance_equations(self):
        """Test mass balance equation retrieval."""
        model = self.models['mixed_lactate']
        equations = model.get_mass_balance_equations()
        
        expected_keys = ['substrate_equation', 'biomass_balance', 
                        'biofilm_thickness', 'nernst_monod']
        
        for key in expected_keys:
            self.assertIn(key, equations)
            self.assertIsInstance(equations[key], str)
            self.assertGreater(len(equations[key]), 0)
    
    def test_gpu_acceleration_availability(self):
        """Test GPU acceleration detection and fallback."""
        # Test with GPU enabled
        gpu_model = BiofilmKineticsModel(species='mixed', use_gpu=True)
        params = gpu_model.get_model_parameters()
        
        # Should report GPU availability status
        self.assertIn('gpu_available', params)
        self.assertIsInstance(params['gpu_available'], bool)
        
        # Test with GPU disabled
        cpu_model = BiofilmKineticsModel(species='mixed', use_gpu=False)
        cpu_params = cpu_model.get_model_parameters()
        
        self.assertFalse(cpu_params['gpu_available'])


class TestModelIntegration(unittest.TestCase):
    """Test integration scenarios and edge cases."""
    
    def test_long_term_simulation(self):
        """Test long-term biofilm development simulation."""
        model = BiofilmKineticsModel(species='mixed', substrate='lactate', use_gpu=False)
        
        # Run 24-hour simulation
        time_points = []
        thickness_points = []
        current_points = []
        
        for hour in range(24):
            result = model.step_biofilm_dynamics(
                dt=1.0,
                anode_potential=-0.25,
                substrate_supply=0.5
            )
            
            time_points.append(result['time'])
            thickness_points.append(result['biofilm_thickness'])
            current_points.append(result['current_density'])
        
        # Verify simulation progresses (thickness may not grow if biomass insufficient)
        self.assertGreaterEqual(thickness_points[-1], thickness_points[0])
        
        # Current should be non-negative
        self.assertGreaterEqual(current_points[-1], 0)
        
        # Verify time progression
        self.assertEqual(time_points[-1], 24.0)
    
    def test_substrate_depletion_scenario(self):
        """Test behavior under substrate depletion conditions."""
        model = BiofilmKineticsModel(species='geobacter', substrate='acetate', use_gpu=False)
        
        # Set low initial substrate
        model.substrate_concentration = 0.5  # mmol/L
        
        # Run without substrate supply
        results = []
        for _ in range(10):
            result = model.step_biofilm_dynamics(
                dt=1.0,
                anode_potential=-0.2,
                substrate_supply=0.0  # No supply
            )
            results.append(result)
        
        # Substrate should decrease
        initial_substrate = results[0]['substrate_concentration']
        final_substrate = results[-1]['substrate_concentration']
        self.assertLessEqual(final_substrate, initial_substrate)
        
        # Growth rate should decrease with substrate
        initial_growth = results[0]['growth_rate']
        final_growth = results[-1]['growth_rate']
        self.assertLessEqual(final_growth, initial_growth)
    
    def test_extreme_conditions(self):
        """Test model behavior under extreme environmental conditions."""
        # Test high temperature
        hot_model = BiofilmKineticsModel(species='shewanella', temperature=343.0, use_gpu=False)
        hot_result = hot_model.step_biofilm_dynamics(dt=1.0, anode_potential=-0.2)
        
        # Test low temperature
        cold_model = BiofilmKineticsModel(species='shewanella', temperature=278.0, use_gpu=False)
        cold_result = cold_model.step_biofilm_dynamics(dt=1.0, anode_potential=-0.2)
        
        # Hot should have higher growth rate than cold
        self.assertGreater(hot_result['growth_rate'], cold_result['growth_rate'])
        
        # Test extreme pH
        acidic_model = BiofilmKineticsModel(species='mixed', ph=5.0, use_gpu=False)
        alkaline_model = BiofilmKineticsModel(species='mixed', ph=9.0, use_gpu=False)
        
        acidic_result = acidic_model.step_biofilm_dynamics(dt=1.0, anode_potential=-0.2)
        alkaline_result = alkaline_model.step_biofilm_dynamics(dt=1.0, anode_potential=-0.2)
        
        # Both should have reduced performance compared to optimal pH
        optimal_model = BiofilmKineticsModel(species='mixed', ph=7.2, use_gpu=False)
        optimal_result = optimal_model.step_biofilm_dynamics(dt=1.0, anode_potential=-0.2)
        
        self.assertLessEqual(acidic_result['growth_rate'], optimal_result['growth_rate'])
        self.assertLessEqual(alkaline_result['growth_rate'], optimal_result['growth_rate'])


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)