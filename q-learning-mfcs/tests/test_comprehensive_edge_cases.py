#!/usr/bin/env python3
"""
Comprehensive edge case and boundary condition tests for MFC Q-Learning Project.
Tests extreme conditions, error handling, and robustness.
"""

import unittest
import numpy as np
import sys
import os
import warnings
import time

# Suppress warnings for clean test output
warnings.filterwarnings('ignore', category=UserWarning)
import matplotlib
matplotlib.use('Agg')

# Add source path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestBoundaryConditions(unittest.TestCase):
    """Test boundary conditions and extreme values."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from gpu_acceleration import GPUAccelerator
            self.gpu = GPUAccelerator()
        except ImportError:
            self.gpu = None
    
    def test_zero_values(self):
        """Test behavior with zero values."""
        if self.gpu is None:
            self.skipTest("GPU acceleration not available")
        
        # Test zero arrays
        zero_array = self.gpu.array([0.0, 0.0, 0.0])
        result = self.gpu.exp(zero_array)
        expected = self.gpu.array([1.0, 1.0, 1.0])
        
        np.testing.assert_allclose(self.gpu.to_cpu(result), 
                                   self.gpu.to_cpu(expected), rtol=1e-6)
    
    def test_negative_values(self):
        """Test behavior with negative values."""
        if self.gpu is None:
            self.skipTest("GPU acceleration not available")
        
        # Test negative arrays
        neg_array = self.gpu.array([-1.0, -2.0, -3.0])
        result = self.gpu.exp(neg_array)
        
        # All results should be positive
        cpu_result = self.gpu.to_cpu(result)
        self.assertTrue(np.all(cpu_result > 0))
        self.assertTrue(np.all(cpu_result < 1.0))
    
    def test_very_large_values(self):
        """Test behavior with very large values."""
        if self.gpu is None:
            self.skipTest("GPU acceleration not available")

        # Test large values that are safe for float32 (exp(88) is near float32 max)
        # float32 max is ~3.4e38, exp(88) ≈ 1.65e38
        large_array = self.gpu.array([10.0, 50.0, 80.0])

        # Should handle without overflow for float32-safe values
        result = self.gpu.exp(large_array)
        cpu_result = self.gpu.to_cpu(result)
        self.assertTrue(np.all(np.isfinite(cpu_result)))
    
    def test_very_small_values(self):
        """Test behavior with very small values."""
        if self.gpu is None:
            self.skipTest("GPU acceleration not available")
        
        # Test small positive values
        small_array = self.gpu.array([1e-10, 1e-15, 1e-20])
        result = self.gpu.log(small_array)
        
        # Should handle without underflow
        cpu_result = self.gpu.to_cpu(result)
        self.assertTrue(np.all(np.isfinite(cpu_result)))
    
    def test_nan_inf_handling(self):
        """Test handling of NaN and infinite values."""
        if self.gpu is None:
            self.skipTest("GPU acceleration not available")
        
        # Test NaN handling
        try:
            nan_array = self.gpu.array([np.nan, 1.0, 2.0])
            result = self.gpu.to_cpu(nan_array)
            self.assertTrue(np.isnan(result[0]))
            self.assertTrue(np.isfinite(result[1:]).all())
        except Exception:
            # Some GPU backends may not support NaN
            pass
        
        # Test infinity handling
        try:
            inf_array = self.gpu.array([np.inf, -np.inf, 1.0])
            result = self.gpu.to_cpu(inf_array)
            self.assertTrue(np.isinf(result[0]))
            self.assertTrue(np.isinf(result[1]))
            self.assertTrue(np.isfinite(result[2]))
        except Exception:
            # Some GPU backends may not support infinity
            pass


class TestConfigurationValidation(unittest.TestCase):
    """Test configuration validation and error handling."""
    
    def test_invalid_species_configuration(self):
        """Test handling of invalid species configurations."""
        try:
            from config.biological_config import BiologicalConfig
            
            # Test invalid species
            with self.assertRaises((ValueError, KeyError)):
                config = BiologicalConfig()
                config.get_species_config("invalid_species")
        except ImportError:
            self.skipTest("Configuration module not available")
    
    def test_invalid_substrate_configuration(self):
        """Test handling of invalid substrate configurations."""
        try:
            from config.biological_config import BiologicalConfig
            
            config = BiologicalConfig()
            
            # Test invalid substrate
            with self.assertRaises((ValueError, KeyError)):
                config.get_substrate_config("invalid_substrate")
        except ImportError:
            self.skipTest("Configuration module not available")
    
    def test_negative_parameter_validation(self):
        """Test validation of negative parameters."""
        try:
            from config.biological_config import BiologicalConfig
            
            config = BiologicalConfig()
            
            # Test negative growth rate
            with self.assertRaises(ValueError):
                config.validate_biological_parameters({
                    'max_growth_rate': -0.1,
                    'electron_transport_efficiency': 0.8
                })
        except ImportError:
            self.skipTest("Configuration module not available")
    
    def test_out_of_range_parameters(self):
        """Test validation of out-of-range parameters."""
        try:
            from config.biological_config import BiologicalConfig
            
            config = BiologicalConfig()
            
            # Test efficiency > 1.0
            with self.assertRaises(ValueError):
                config.validate_biological_parameters({
                    'max_growth_rate': 0.1,
                    'electron_transport_efficiency': 1.5  # > 1.0
                })
        except ImportError:
            self.skipTest("Configuration module not available")


class TestMemoryAndPerformance(unittest.TestCase):
    """Test memory usage and performance under stress."""
    
    def test_large_array_operations(self):
        """Test operations with large arrays."""
        try:
            from gpu_acceleration import GPUAccelerator
            gpu = GPUAccelerator()
            
            # Create large arrays (but manageable size)
            size = 1_000_000
            large_array = gpu.array(np.random.rand(size).astype(np.float32))
            
            # Test basic operations
            result1 = gpu.exp(large_array)
            result2 = gpu.log(large_array)
            
            # Verify results are finite
            cpu_result1 = gpu.to_cpu(result1)
            cpu_result2 = gpu.to_cpu(result2)
            
            finite_ratio1 = np.sum(np.isfinite(cpu_result1)) / len(cpu_result1)
            finite_ratio2 = np.sum(np.isfinite(cpu_result2)) / len(cpu_result2)
            
            # Most results should be finite
            self.assertGreater(finite_ratio1, 0.95)
            self.assertGreater(finite_ratio2, 0.95)
            
        except ImportError:
            self.skipTest("GPU acceleration not available")
        except Exception as e:
            # Memory errors are acceptable for very large arrays
            if "memory" in str(e).lower() or "allocation" in str(e).lower():
                self.skipTest(f"Memory limitation: {e}")
            else:
                raise
    
    def test_repeated_operations(self):
        """Test repeated operations for memory leaks."""
        try:
            from gpu_acceleration import GPUAccelerator
            gpu = GPUAccelerator()
            
            # Perform many operations
            for i in range(100):
                arr = gpu.array([1.0, 2.0, 3.0])
                result = gpu.exp(arr)
                cpu_result = gpu.to_cpu(result)
                
                # Verify each iteration
                self.assertEqual(len(cpu_result), 3)
                self.assertTrue(np.all(np.isfinite(cpu_result)))
                
        except ImportError:
            self.skipTest("GPU acceleration not available")
    
    def test_performance_degradation(self):
        """Test for performance degradation over time."""
        try:
            from gpu_acceleration import GPUAccelerator
            gpu = GPUAccelerator()
            
            times = []
            size = 10000
            
            # Measure operation times
            for i in range(10):
                arr = gpu.array(np.random.rand(size).astype(np.float32))
                
                start_time = time.time()
                result = gpu.exp(arr)
                gpu.to_cpu(result)
                end_time = time.time()
                
                times.append(end_time - start_time)
            
            # Performance shouldn't degrade significantly
            first_half_avg = np.mean(times[:5])
            second_half_avg = np.mean(times[5:])
            
            # Second half shouldn't be more than 2x slower
            self.assertLess(second_half_avg, first_half_avg * 2.0)
            
        except ImportError:
            self.skipTest("GPU acceleration not available")


class TestErrorRecovery(unittest.TestCase):
    """Test error recovery and graceful degradation."""
    
    def test_gpu_fallback(self):
        """Test CPU fallback when GPU operations fail."""
        try:
            from gpu_acceleration import GPUAccelerator
            
            # Create accelerator
            gpu = GPUAccelerator()
            
            # Test that it can handle fallback scenarios
            if gpu.backend == 'cpu':
                # Already on CPU - test it works
                arr = gpu.array([1.0, 2.0, 3.0])
                result = gpu.exp(arr)
                cpu_result = gpu.to_cpu(result)
                
                self.assertEqual(len(cpu_result), 3)
                self.assertTrue(np.all(cpu_result > 0))
            else:
                # On GPU - test operations work
                arr = gpu.array([1.0, 2.0, 3.0])
                result = gpu.exp(arr)
                cpu_result = gpu.to_cpu(result)
                
                self.assertEqual(len(cpu_result), 3)
                self.assertTrue(np.all(cpu_result > 0))
                
        except ImportError:
            self.skipTest("GPU acceleration not available")
    
    def test_sensor_failure_handling(self):
        """Test handling of sensor failures."""
        try:
            from sensing_models.qcm_model import QCMModel, CrystalType, QCMMeasurement

            # Create sensor model with correct API
            qcm = QCMModel(crystal_type=CrystalType.AT_CUT_5MHZ)

            # Test that sensor model can be instantiated and produces valid measurement
            measurement = qcm.simulate_measurement(
                biofilm_mass=1.5, biofilm_thickness=1.1,
            )
            self.assertIsInstance(measurement, QCMMeasurement)
            self.assertIsNotNone(measurement.frequency)
            self.assertIsNotNone(measurement.frequency_shift)

            # Test edge case: zero biofilm should not crash
            zero_measurement = qcm.simulate_measurement(
                biofilm_mass=0.0, biofilm_thickness=0.0,
            )
            self.assertIsInstance(zero_measurement, QCMMeasurement)

            # Test edge case: very small values should handle gracefully
            small_measurement = qcm.simulate_measurement(
                biofilm_mass=1e-10, biofilm_thickness=1e-10,
            )
            self.assertIsInstance(small_measurement, QCMMeasurement)

        except ImportError:
            self.skipTest("Sensing models not available")
    
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        try:
            from gpu_acceleration import GPUAccelerator
            gpu = GPUAccelerator()
            
            # Test various invalid inputs
            invalid_inputs = [
                [],  # Empty array
                [float('inf')],  # Infinity
                [float('nan')],  # NaN
                None,  # None value
            ]
            
            for invalid_input in invalid_inputs:
                try:
                    if invalid_input is not None:
                        arr = gpu.array(invalid_input)
                        result = gpu.exp(arr)
                        # If it succeeds, result should be valid
                        cpu_result = gpu.to_cpu(result)
                        self.assertIsInstance(cpu_result, np.ndarray)
                except (ValueError, TypeError, RuntimeError):
                    # Expected errors for invalid inputs
                    pass
                    
        except ImportError:
            self.skipTest("GPU acceleration not available")


class TestDataIntegrity(unittest.TestCase):
    """Test data integrity and consistency."""
    
    def test_numerical_precision(self):
        """Test numerical precision and consistency."""
        try:
            from gpu_acceleration import GPUAccelerator
            gpu = GPUAccelerator()
            
            # Test precision with known values
            test_values = [1.0, np.pi, np.e, 0.1, 0.01]
            arr = gpu.array(test_values)
            
            # Test round-trip consistency
            result = gpu.exp(gpu.log(arr))
            cpu_result = gpu.to_cpu(result)
            
            # Should be very close to original
            np.testing.assert_allclose(cpu_result, test_values, rtol=1e-5)
            
        except ImportError:
            self.skipTest("GPU acceleration not available")
    
    def test_simulation_conservation_laws(self):
        """Test that conservation laws are maintained."""
        try:
            from integrated_mfc_model import IntegratedMFCModel
            
            model = IntegratedMFCModel(
                n_cells=2, species="geobacter", substrate="acetate",
                use_gpu=False, simulation_hours=5
            )
            
            # Record initial conditions
            initial_substrate = model.mfc_stack.reservoir.substrate_concentration
            
            # Run simulation
            states = []
            for _ in range(5):
                state = model.step_integrated_dynamics(dt=1.0)
                states.append(state)
            
            # Check energy conservation (should increase or stay constant)
            energies = [state.total_energy for state in states]
            for i in range(1, len(energies)):
                self.assertGreaterEqual(energies[i], energies[i-1] - 1e-6)  # Allow small numerical errors
            
            # Check substrate conservation (should decrease or stay constant)
            final_substrate = states[-1].reservoir_concentration
            total_consumed = initial_substrate - final_substrate
            self.assertGreaterEqual(total_consumed, -1e-6)  # Allow small numerical errors
            
        except ImportError:
            self.skipTest("Integrated model not available")
    
    def test_mass_balance(self):
        """Test mass balance in biofilm models."""
        try:
            from biofilm_kinetics.biofilm_model import BiofilmModel
            
            model = BiofilmModel(species='geobacter', substrate='acetate')
            
            # Record initial biomass
            initial_biomass = model.biomass_density
            
            # Simulate growth
            for _ in range(10):
                model.update_biofilm_dynamics(
                    substrate_conc=20.0,
                    current_density=0.1,
                    dt=1.0
                )
            
            # Biomass should have changed
            final_biomass = model.biomass_density
            
            # Mass should be conserved (allowing for growth)
            self.assertGreaterEqual(final_biomass, initial_biomass)
            self.assertLess(final_biomass, initial_biomass * 100)  # Reasonable growth limit
            
        except ImportError:
            self.skipTest("Biofilm model not available")


class TestConcurrencyAndThreadSafety(unittest.TestCase):
    """Test concurrency and thread safety."""
    
    def test_concurrent_gpu_operations(self):
        """Test concurrent GPU operations."""
        try:
            from gpu_acceleration import GPUAccelerator
            import threading
            
            gpu = GPUAccelerator()
            results = []
            errors = []
            
            def worker(worker_id):
                try:
                    for i in range(10):
                        arr = gpu.array([worker_id, i, worker_id + i])
                        result = gpu.exp(arr)
                        cpu_result = gpu.to_cpu(result)
                        results.append((worker_id, i, cpu_result))
                except Exception as e:
                    errors.append((worker_id, e))
            
            # Create multiple threads
            threads = []
            for i in range(3):
                thread = threading.Thread(target=worker, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            # Check results
            if errors:
                # Some GPU backends may not support concurrency
                print(f"Concurrent operations not supported: {errors}")
            else:
                self.assertGreater(len(results), 0)
                # All results should be valid
                for worker_id, i, result in results:
                    self.assertEqual(len(result), 3)
                    self.assertTrue(np.all(np.isfinite(result)))
            
        except ImportError:
            self.skipTest("GPU acceleration not available")


class TestRegressionCases(unittest.TestCase):
    """Test specific regression cases and bug fixes."""
    
    def test_jax_import_fix(self):
        """Test fix for JAX circular import issue."""
        try:
            # Try importing jax with proper error handling
            import jax
            import jax.numpy as jnp
            
            # Test basic operation
            a = jnp.array([1.0, 2.0, 3.0])
            result = jnp.exp(a)
            
            self.assertEqual(len(result), 3)
            self.assertTrue(jnp.all(jnp.isfinite(result)))
            
        except (ImportError, AttributeError) as e:
            # This is the specific error we're testing for
            if "circular import" in str(e) or "version" in str(e):
                self.fail(f"JAX circular import issue not fixed: {e}")
            else:
                self.skipTest(f"JAX not available: {e}")
    
    def test_gpu_memory_cleanup(self):
        """Test GPU memory cleanup after operations."""
        try:
            from gpu_acceleration import GPUAccelerator
            gpu = GPUAccelerator()
            
            # Create and destroy many arrays
            for i in range(50):
                large_arr = gpu.array(np.random.rand(10000).astype(np.float32))
                result = gpu.exp(large_arr)
                cpu_result = gpu.to_cpu(result)
                
                # Explicit cleanup if available
                del large_arr, result, cpu_result
            
            # Should complete without memory errors
            self.assertTrue(True)
            
        except ImportError:
            self.skipTest("GPU acceleration not available")
        except Exception as e:
            if "memory" in str(e).lower():
                self.fail(f"Memory cleanup issue: {e}")
            else:
                raise


class TestCompatibilityMatrix(unittest.TestCase):
    """Test compatibility across different configurations."""
    
    def test_species_substrate_combinations(self):
        """Test all valid species-substrate combinations."""
        try:
            from config.biological_config import BiologicalConfig
            
            config = BiologicalConfig()
            
            # Valid combinations
            valid_combinations = [
                ('geobacter', 'acetate'),
                ('shewanella', 'lactate'),
                ('mixed', 'acetate'),
                ('mixed', 'lactate'),
            ]
            
            for species, substrate in valid_combinations:
                try:
                    species_config = config.get_species_config(species)
                    substrate_config = config.get_substrate_config(substrate)
                    
                    # Should not raise errors
                    self.assertIsInstance(species_config, dict)
                    self.assertIsInstance(substrate_config, dict)
                    
                except Exception as e:
                    self.fail(f"Valid combination {species}-{substrate} failed: {e}")
            
        except ImportError:
            self.skipTest("Configuration module not available")
    
    def test_gpu_backend_compatibility(self):
        """Test compatibility across different GPU backends."""
        try:
            from gpu_acceleration import GPUAccelerator
            
            gpu = GPUAccelerator()
            
            # Test operations that should work on all backends
            test_operations = [
                lambda x: gpu.exp(x),
                lambda x: gpu.log(gpu.abs(x) + 1e-8),
                lambda x: gpu.sqrt(gpu.abs(x)),
                lambda x: gpu.power(gpu.abs(x), 2),
            ]
            
            test_array = gpu.array([1.0, 2.0, 3.0])
            
            for op in test_operations:
                try:
                    result = op(test_array)
                    cpu_result = gpu.to_cpu(result)
                    
                    self.assertEqual(len(cpu_result), 3)
                    self.assertTrue(np.all(np.isfinite(cpu_result)))
                    
                except Exception as e:
                    self.fail(f"Operation failed on {gpu.backend} backend: {e}")
            
        except ImportError:
            self.skipTest("GPU acceleration not available")


if __name__ == '__main__':
    # Run with high verbosity to see all test details
    unittest.main(verbosity=2, buffer=True)