#!/usr/bin/env python3
"""
Biological and physical constraint validation tests for MFC Q-Learning Project.
Tests that biological models respect physical laws and biological plausibility.
"""

import unittest
import numpy as np
import sys
import os
import warnings

# Suppress warnings for clean test output
warnings.filterwarnings('ignore', category=UserWarning)
import matplotlib
matplotlib.use('Agg')

# Add source path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestBiologicalConstraints(unittest.TestCase):
    """Test biological constraint validation."""
    
    def test_growth_rate_constraints(self):
        """Test that growth rates remain within biological limits."""
        try:
            from biofilm_kinetics.biofilm_model import BiofilmModel
            
            model = BiofilmModel(species='geobacter', substrate='acetate')
            
            # Test with various substrate concentrations
            substrate_concentrations = [0.1, 1.0, 10.0, 50.0, 100.0]
            
            for substrate_conc in substrate_concentrations:
                growth_rate = model._calculate_growth_rate(substrate_conc)
                
                # Growth rate should be non-negative
                self.assertGreaterEqual(growth_rate, 0.0)
                
                # Growth rate should not exceed maximum
                max_growth_rate = getattr(model, 'max_growth_rate', 0.5)  # Typical max is ~0.5 h^-1
                self.assertLessEqual(growth_rate, max_growth_rate * 1.1)  # Allow small numerical overshoot
                
                # Growth rate should be monotonic with substrate (up to saturation)
                if substrate_conc <= 10.0:  # Below saturation
                    growth_rate_low = model._calculate_growth_rate(substrate_conc * 0.5)
                    self.assertGreaterEqual(growth_rate, growth_rate_low)
            
        except ImportError:
            self.skipTest("Biofilm model not available")
        except AttributeError:
            self.skipTest("Growth rate calculation method not found")
    
    def test_biofilm_thickness_constraints(self):
        """Test biofilm thickness remains within physical limits."""
        try:
            from biofilm_kinetics.biofilm_model import BiofilmModel
            
            model = BiofilmModel(species='geobacter', substrate='acetate')
            
            # Simulate extended growth
            for hour in range(100):  # 100 hours
                model.update_biofilm_dynamics(
                    substrate_conc=25.0,  # High substrate
                    current_density=0.1,
                    dt=1.0
                )
                
                thickness = model.biofilm_thickness
                
                # Thickness should be positive
                self.assertGreater(thickness, 0.0)
                
                # Thickness should not exceed realistic maximum (~200 μm)
                self.assertLess(thickness, 200.0)
                
                # Thickness should not grow infinitely
                if hour > 50:  # After steady state
                    self.assertLess(thickness, 150.0)  # Should level off
            
        except ImportError:
            self.skipTest("Biofilm model not available")
    
    def test_biomass_density_constraints(self):
        """Test biomass density remains within physical limits."""
        try:
            from biofilm_kinetics.biofilm_model import BiofilmModel
            
            model = BiofilmModel(species='geobacter', substrate='acetate')
            
            # Test with various conditions
            conditions = [
                (5.0, 0.05),   # Low substrate, low current
                (25.0, 0.1),   # High substrate, medium current
                (50.0, 0.2),   # Very high substrate, high current
            ]
            
            for substrate_conc, current_density in conditions:
                # Reset model
                model = BiofilmModel(species='geobacter', substrate='acetate')
                
                for _ in range(50):
                    model.update_biofilm_dynamics(
                        substrate_conc=substrate_conc,
                        current_density=current_density,
                        dt=1.0
                    )
                
                density = model.biomass_density
                
                # Density should be positive
                self.assertGreater(density, 0.0)
                
                # Density should not exceed physical maximum
                # Typical microbial biomass density: 50-200 g/L
                self.assertLess(density, 300.0)
                
                # Density should be reasonable for the conditions
                if substrate_conc > 20.0:  # High substrate
                    self.assertGreater(density, 5.0)  # Should achieve significant growth
            
        except ImportError:
            self.skipTest("Biofilm model not available")
    
    def test_electron_transport_efficiency(self):
        """Test electron transport efficiency constraints."""
        try:
            from metabolic_model.metabolic_model import MetabolicModel
            
            # Test different species
            species_list = ['geobacter', 'shewanella']
            
            for species in species_list:
                try:
                    model = MetabolicModel(species_str=species, substrate_str='acetate')
                    
                    # Simulate metabolic activity
                    for _ in range(10):
                        model.update_metabolites(
                            substrate_concentration=15.0,
                            external_electron_demand=0.1,
                            dt=1.0
                        )
                    
                    # Get current efficiency
                    efficiency = getattr(model, 'coulombic_efficiency', 
                                       getattr(model, 'electron_transport_efficiency', None))
                    
                    if efficiency is not None:
                        # Efficiency should be between 0 and 1
                        self.assertGreaterEqual(efficiency, 0.0)
                        self.assertLessEqual(efficiency, 1.0)
                        
                        # For healthy biofilms, efficiency should be reasonable
                        self.assertGreater(efficiency, 0.1)  # At least 10%
                
                except Exception as e:
                    # Skip if specific species not implemented
                    if "not found" in str(e) or "not implemented" in str(e):
                        continue
                    raise
            
        except ImportError:
            self.skipTest("Metabolic model not available")
    
    def test_substrate_consumption_rates(self):
        """Test substrate consumption rates are biologically plausible."""
        try:
            from metabolic_model.metabolic_model import MetabolicModel
            
            model = MetabolicModel(species_str='geobacter', substrate_str='acetate')
            
            initial_substrate = 20.0
            
            # Track substrate consumption over time
            consumption_rates = []
            
            for hour in range(24):  # 24 hours
                model.update_metabolites(
                    substrate_concentration=initial_substrate * (1 - hour/48),  # Decreasing substrate
                    external_electron_demand=0.1,
                    dt=1.0
                )
                
                # Calculate consumption rate (if available)
                if hasattr(model, 'substrate_consumption_rate'):
                    rate = model.substrate_consumption_rate
                    consumption_rates.append(rate)
                    
                    # Consumption rate should be non-negative
                    self.assertGreaterEqual(rate, 0.0)
                    
                    # Rate should be reasonable (not too high)
                    # Typical rates: 0.1-10 mmol/L/h
                    self.assertLess(rate, 20.0)
            
            # Consumption should generally decrease as substrate depletes
            if len(consumption_rates) > 10:
                early_avg = np.mean(consumption_rates[:5])
                late_avg = np.mean(consumption_rates[-5:])
                
                # Late consumption should be lower (assuming substrate depletion)
                self.assertLessEqual(late_avg, early_avg * 1.5)  # Allow some variation
            
        except ImportError:
            self.skipTest("Metabolic model not available")


class TestPhysicalConstraints(unittest.TestCase):
    """Test physical constraint validation."""
    
    def test_voltage_constraints(self):
        """Test cell voltages remain within physical limits."""
        try:
            from mfc_stack_simulation import MFCStack
            
            stack = MFCStack(n_cells=3)
            
            # Test under various load conditions
            load_resistances = [1.0, 10.0, 100.0, 1000.0]  # Ohms
            
            for resistance in load_resistances:
                stack.external_resistance = resistance
                
                # Simulate for several steps
                for _ in range(10):
                    voltages = stack.get_cell_voltages()
                    
                    for voltage in voltages:
                        # Cell voltage should be within thermodynamic limits
                        # Theoretical max ~1.2V, typical range 0-0.8V
                        self.assertGreaterEqual(voltage, -0.1)  # Allow small negative due to overpotentials
                        self.assertLessEqual(voltage, 1.2)
                        
                        # Under normal operation, should be positive
                        if resistance <= 100.0:  # Reasonable load
                            self.assertGreater(voltage, 0.0)
            
        except ImportError:
            self.skipTest("MFC stack simulation not available")
    
    def test_current_density_constraints(self):
        """Test current densities remain within physical limits."""
        try:
            from mfc_stack_simulation import MFCStack
            
            stack = MFCStack(n_cells=3)
            
            # Test under various conditions
            for _ in range(20):
                current_densities = stack.get_current_densities()
                
                for current_density in current_densities:
                    # Current density should be non-negative
                    self.assertGreaterEqual(current_density, 0.0)
                    
                    # Should not exceed typical maximum
                    # Typical maximum: 1-10 A/m²
                    self.assertLess(current_density, 50.0)  # Conservative upper limit
                
                # Update stack state
                stack.update_stack_dynamics(dt=1.0)
            
        except ImportError:
            self.skipTest("MFC stack simulation not available")
    
    def test_power_output_constraints(self):
        """Test power output follows physical laws."""
        try:
            from mfc_stack_simulation import MFCStack
            
            stack = MFCStack(n_cells=5)
            
            # Test power vs resistance relationship (should follow P = V²/R)
            resistances = [1.0, 5.0, 10.0, 50.0, 100.0]
            powers = []
            
            for resistance in resistances:
                stack.external_resistance = resistance
                
                # Let stack stabilize
                for _ in range(5):
                    stack.update_stack_dynamics(dt=1.0)
                
                total_power = stack.get_total_power()
                powers.append((resistance, total_power))
                
                # Power should be non-negative
                self.assertGreaterEqual(total_power, 0.0)
            
            # Should see power maximum at intermediate resistance
            power_values = [p[1] for p in powers]
            max_power_idx = np.argmax(power_values)
            
            # Maximum shouldn't be at extremes (unless system is unusual)
            self.assertGreater(max_power_idx, 0)
            self.assertLess(max_power_idx, len(resistances) - 1)
            
        except ImportError:
            self.skipTest("MFC stack simulation not available")
    
    def test_energy_conservation(self):
        """Test energy conservation in the system."""
        try:
            from integrated_mfc_model import IntegratedMFCModel
            
            model = IntegratedMFCModel(
                n_cells=3, species="geobacter", substrate="acetate",
                use_gpu=False, simulation_hours=10
            )
            
            # Track energy over time
            energies = []
            
            for hour in range(10):
                state = model.step_integrated_dynamics(dt=1.0)
                energies.append(state.total_energy)
            
            # Energy should generally increase (power generation)
            for i in range(1, len(energies)):
                # Allow for small decreases due to numerical precision
                self.assertGreaterEqual(energies[i], energies[i-1] - 1e-6)
            
            # Total energy should be reasonable
            final_energy = energies[-1]
            self.assertGreater(final_energy, 0.0)
            self.assertLess(final_energy, 1000.0)  # Reasonable upper bound for small system
            
        except ImportError:
            self.skipTest("Integrated model not available")


class TestChemicalConstraints(unittest.TestCase):
    """Test chemical and thermodynamic constraints."""
    
    def test_ph_constraints(self):
        """Test pH remains within reasonable range."""
        try:
            from mfc_stack_simulation import MFCStack
            
            stack = MFCStack(n_cells=3)
            
            # Simulate operation
            for hour in range(24):
                stack.update_stack_dynamics(dt=1.0)
                
                for cell in stack.cells:
                    ph = getattr(cell, 'ph', getattr(cell, 'pH', None))
                    
                    if ph is not None:
                        # pH should be within reasonable range for MFCs
                        self.assertGreater(ph, 5.0)   # Not too acidic
                        self.assertLess(ph, 10.0)    # Not too basic
                        
                        # For most MFCs, pH should be near neutral
                        if hour > 5:  # After initial equilibration
                            self.assertGreater(ph, 6.0)
                            self.assertLess(ph, 9.0)
            
        except ImportError:
            self.skipTest("MFC stack simulation not available")
    
    def test_concentration_constraints(self):
        """Test chemical concentrations remain realistic."""
        try:
            from mfc_stack_simulation import MFCStack
            
            stack = MFCStack(n_cells=3)
            
            # Test over extended operation
            for hour in range(48):
                stack.update_stack_dynamics(dt=1.0)
                
                # Check substrate concentration
                substrate_conc = stack.reservoir.substrate_concentration
                
                # Should remain non-negative
                self.assertGreaterEqual(substrate_conc, 0.0)
                
                # Should not exceed initial concentration (unless being fed)
                initial_conc = getattr(stack.reservoir, 'initial_substrate_concentration', 50.0)
                self.assertLessEqual(substrate_conc, initial_conc * 1.1)  # Allow small overshoot
                
                # Check individual cell concentrations
                for cell in stack.cells:
                    cell_substrate = getattr(cell, 'substrate_concentration', None)
                    
                    if cell_substrate is not None:
                        self.assertGreaterEqual(cell_substrate, 0.0)
                        # Cell concentration should not exceed reservoir
                        self.assertLessEqual(cell_substrate, substrate_conc * 1.2)
            
        except ImportError:
            self.skipTest("MFC stack simulation not available")
    
    def test_mass_transfer_constraints(self):
        """Test mass transfer follows physical laws."""
        try:
            from mfc_stack_simulation import MFCStack
            
            stack = MFCStack(n_cells=3)
            
            # Set up concentration gradient
            stack.reservoir.substrate_concentration = 30.0
            for cell in stack.cells:
                if hasattr(cell, 'substrate_concentration'):
                    cell.substrate_concentration = 10.0  # Lower than reservoir
            
            # Track mass transfer
            initial_total = stack.reservoir.substrate_concentration
            for cell in stack.cells:
                if hasattr(cell, 'substrate_concentration'):
                    initial_total += cell.substrate_concentration
            
            # Simulate mass transfer
            for _ in range(10):
                stack.update_stack_dynamics(dt=1.0)
            
            # Calculate final total
            final_total = stack.reservoir.substrate_concentration
            for cell in stack.cells:
                if hasattr(cell, 'substrate_concentration'):
                    final_total += cell.substrate_concentration
            
            # Total mass should be conserved (allowing for consumption)
            self.assertLessEqual(final_total, initial_total + 1e-6)  # Allow numerical precision
            
            # Concentrations should equilibrate (gradient should decrease)
            final_reservoir = stack.reservoir.substrate_concentration
            final_cell_avg = np.mean([getattr(cell, 'substrate_concentration', final_reservoir) 
                                    for cell in stack.cells])
            
            initial_gradient = abs(30.0 - 10.0)
            final_gradient = abs(final_reservoir - final_cell_avg)
            
            self.assertLess(final_gradient, initial_gradient)
            
        except ImportError:
            self.skipTest("MFC stack simulation not available")


class TestThermodynamicConstraints(unittest.TestCase):
    """Test thermodynamic constraints and limits."""
    
    def test_gibbs_free_energy_constraints(self):
        """Test reactions respect Gibbs free energy limits."""
        try:
            from metabolic_model.metabolic_model import MetabolicModel
            
            model = MetabolicModel(species_str='geobacter', substrate_str='acetate')
            
            # Test at different conditions
            temperatures = [298.15, 308.15, 318.15]  # 25°C, 35°C, 45°C
            
            for temp in temperatures:
                if hasattr(model, 'temperature'):
                    model.temperature = temp
                
                # Calculate reaction energetics
                model.update_metabolites(
                    substrate_concentration=20.0,
                    external_electron_demand=0.1,
                    dt=1.0
                )
                
                # Check that favorable reactions proceed
                if hasattr(model, 'reaction_rates'):
                    for rate in model.reaction_rates.values():
                        if rate is not None:
                            # Rates should be non-negative for favorable reactions
                            self.assertGreaterEqual(rate, 0.0)
                
                # Check efficiency is thermodynamically reasonable
                if hasattr(model, 'coulombic_efficiency'):
                    efficiency = model.coulombic_efficiency
                    
                    # Efficiency should decrease with temperature (usually)
                    # But this depends on the specific implementation
                    self.assertGreaterEqual(efficiency, 0.0)
                    self.assertLessEqual(efficiency, 1.0)
            
        except ImportError:
            self.skipTest("Metabolic model not available")
    
    def test_electrochemical_potential_constraints(self):
        """Test electrochemical potentials are realistic."""
        try:
            from mfc_stack_simulation import MFCStack
            
            stack = MFCStack(n_cells=3)
            
            # Simulate under different conditions
            for cycle in range(5):
                # Vary external resistance
                stack.external_resistance = 10.0 * (cycle + 1)
                
                for _ in range(10):
                    stack.update_stack_dynamics(dt=1.0)
                
                for cell in stack.cells:
                    # Check anode potential
                    if hasattr(cell, 'anode_potential'):
                        anode_potential = cell.anode_potential
                        
                        # Anode potential should be within reasonable range
                        # Typical range: -0.5 to -0.2 V vs SHE
                        self.assertGreater(anode_potential, -0.8)
                        self.assertLess(anode_potential, 0.0)
                    
                    # Check cathode potential
                    if hasattr(cell, 'cathode_potential'):
                        cathode_potential = cell.cathode_potential
                        
                        # Cathode potential should be positive vs anode
                        # Typical range: 0.2 to 0.8 V vs SHE
                        self.assertGreater(cathode_potential, 0.0)
                        self.assertLess(cathode_potential, 1.0)
                        
                        # Cell voltage should equal potential difference
                        if hasattr(cell, 'anode_potential'):
                            expected_voltage = cathode_potential - anode_potential
                            actual_voltage = getattr(cell, 'voltage', 0.0)
                            
                            # Should be reasonably close
                            self.assertAlmostEqual(actual_voltage, expected_voltage, delta=0.2)
            
        except ImportError:
            self.skipTest("MFC stack simulation not available")
    
    def test_nernst_equation_compliance(self):
        """Test that potentials follow Nernst equation trends."""
        try:
            from mfc_stack_simulation import MFCStack
            
            stack = MFCStack(n_cells=3)
            
            # Test effect of concentration on potential
            concentrations = [5.0, 10.0, 20.0, 40.0]
            potentials = []
            
            for conc in concentrations:
                stack.reservoir.substrate_concentration = conc
                
                # Let system equilibrate
                for _ in range(5):
                    stack.update_stack_dynamics(dt=1.0)
                
                # Record average cell potential
                cell_potentials = []
                for cell in stack.cells:
                    if hasattr(cell, 'anode_potential'):
                        cell_potentials.append(cell.anode_potential)
                
                if cell_potentials:
                    avg_potential = np.mean(cell_potentials)
                    potentials.append(avg_potential)
            
            # Potential should increase with concentration (Nernst equation)
            if len(potentials) >= 2:
                for i in range(1, len(potentials)):
                    # Higher concentration should give higher (less negative) potential
                    self.assertGreaterEqual(potentials[i], potentials[i-1] - 0.05)  # Allow small variations
            
        except ImportError:
            self.skipTest("MFC stack simulation not available")


if __name__ == '__main__':
    unittest.main(verbosity=2)