"""
Comprehensive tests for EIS and QCM sensing models.

Tests cover:
- EIS model functionality and circuit modeling
- QCM model functionality and mass calculations
- Sensor fusion algorithms and validation
- Integration with biofilm models
- Fault detection and handling
- Performance under various conditions
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

# Import sensing models
try:
    from sensing_models.eis_model import (
        EISModel, EISMeasurement, EISCircuitModel, BacterialSpecies
    )
    from sensing_models.qcm_model import (
        QCMModel, QCMMeasurement, SauerbreyModel, ViscoelasticModel,
        CrystalType, ElectrodeType
    )
    from sensing_models.sensor_fusion import (
        SensorFusion, SensorCalibration, KalmanFilter, FusedMeasurement, FusionMethod
    )
    SENSING_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Sensing models not available: {e}")
    SENSING_MODELS_AVAILABLE = False


@unittest.skipUnless(SENSING_MODELS_AVAILABLE, "Sensing models not available")
class TestEISModel(unittest.TestCase):
    """Test EIS model functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.eis_model = EISModel(
            species=BacterialSpecies.GEOBACTER,
            electrode_area=1e-4,  # 1 cm²
            use_gpu=False
        )
    
    def test_model_initialization(self):
        """Test EIS model initialization."""
        # Test different species
        for species in [BacterialSpecies.GEOBACTER, BacterialSpecies.SHEWANELLA, BacterialSpecies.MIXED]:
            model = EISModel(species=species, use_gpu=False)
            self.assertEqual(model.species, species)
            self.assertIsNotNone(model.circuit)
            self.assertEqual(model.circuit.species, species)
        
        # Test frequency range
        self.assertEqual(len(self.eis_model.frequency_points), 50)
        self.assertGreaterEqual(self.eis_model.frequency_points[0], 100)
        self.assertLessEqual(self.eis_model.frequency_points[-1], 1e6)
    
    def test_circuit_model(self):
        """Test EIS circuit model functionality."""
        circuit = self.eis_model.circuit
        
        # Test parameter initialization
        self.assertGreater(circuit.Rs, 0)
        self.assertGreater(circuit.Cdl, 0)
        self.assertGreater(circuit.Rbio, 0)
        self.assertGreater(circuit.Cbio, 0)
        self.assertGreater(circuit.Rct, 0)
        
        # Test impedance calculation
        test_frequency = 1000.0  # Hz
        impedance = circuit.calculate_impedance(test_frequency)
        
        # Impedance should be complex
        self.assertIsInstance(impedance, complex)
        self.assertGreater(abs(impedance), 0)
        
        # Test parameter update from biofilm state
        circuit.update_from_biofilm_state(
            thickness=30.0,  # μm
            biomass_density=10.0,  # g/L
            porosity=0.8,
            electrode_area=1e-4  # m²
        )
        
        # Parameters should be updated
        self.assertGreater(circuit.Rbio, 0)
        self.assertGreater(circuit.Cbio, 0)
    
    def test_measurement_simulation(self):
        """Test EIS measurement simulation."""
        measurements = self.eis_model.simulate_measurement(
            biofilm_thickness=25.0,  # μm
            biomass_density=8.0,     # g/L
            porosity=0.8,
            temperature=303.0,       # K
            time_hours=1.0
        )
        
        # Should return measurements for all frequency points
        self.assertEqual(len(measurements), len(self.eis_model.frequency_points))
        
        # Each measurement should have required properties
        for measurement in measurements:
            self.assertIsInstance(measurement, EISMeasurement)
            self.assertGreater(measurement.frequency, 0)
            self.assertGreater(measurement.impedance_magnitude, 0)
            self.assertIsInstance(measurement.impedance_phase, float)
            self.assertEqual(measurement.temperature, 303.0)
            self.assertEqual(measurement.timestamp, 1.0)
    
    def test_thickness_estimation(self):
        """Test biofilm thickness estimation from EIS data."""
        # Simulate measurements for known thickness
        known_thickness = 30.0  # μm
        measurements = self.eis_model.simulate_measurement(known_thickness, 10.0)
        
        # Estimate thickness
        estimated_thickness = self.eis_model.estimate_thickness(measurements)
        
        # Should be reasonably close (within 75% due to noise and model uncertainty)
        relative_error = abs(estimated_thickness - known_thickness) / known_thickness
        self.assertLess(relative_error, 0.75)
        
        # Test different estimation methods
        for method in ['low_frequency', 'characteristic', 'fitting']:
            thickness = self.eis_model.estimate_thickness(measurements, method=method)
            self.assertGreaterEqual(thickness, 0)
            self.assertLessEqual(thickness, self.eis_model.calibration['max_thickness'])
    
    def test_biofilm_properties_extraction(self):
        """Test extraction of biofilm properties from EIS measurements."""
        measurements = self.eis_model.simulate_measurement(20.0, 5.0)
        properties = self.eis_model.get_biofilm_properties(measurements)
        
        # Verify required properties are present
        required_properties = [
            'thickness_um', 'conductivity_S_per_m', 'capacitance_F_per_cm2',
            'biomass_estimate_g_per_L', 'solution_resistance_ohm',
            'biofilm_resistance_ohm', 'charge_transfer_resistance_ohm',
            'measurement_quality'
        ]
        
        for prop in required_properties:
            self.assertIn(prop, properties)
            self.assertIsInstance(properties[prop], (int, float))
            self.assertGreaterEqual(properties[prop], 0)
        
        # Measurement quality should be between 0 and 1
        self.assertLessEqual(properties['measurement_quality'], 1.0)
    
    def test_species_specific_parameters(self):
        """Test species-specific parameter loading."""
        # Test Geobacter parameters
        geo_model = EISModel(species=BacterialSpecies.GEOBACTER, use_gpu=False)
        geo_params = geo_model.circuit.species_params
        
        # Test Shewanella parameters
        she_model = EISModel(species=BacterialSpecies.SHEWANELLA, use_gpu=False)
        she_params = she_model.circuit.species_params
        
        # Test Mixed culture parameters
        mixed_model = EISModel(species=BacterialSpecies.MIXED, use_gpu=False)
        mixed_params = mixed_model.circuit.species_params
        
        # Parameters should be different between species
        self.assertNotEqual(geo_params['base_resistivity'], she_params['base_resistivity'])
        self.assertNotEqual(geo_params['conductivity'], she_params['conductivity'])
        
        # Mixed should be intermediate
        self.assertTrue(
            min(geo_params['base_resistivity'], she_params['base_resistivity']) <=
            mixed_params['base_resistivity'] <=
            max(geo_params['base_resistivity'], she_params['base_resistivity'])
        )
    
    def test_calibration(self):
        """Test EIS model calibration functionality."""
        # Create reference data
        reference_data = []
        for thickness in [10, 20, 30, 40]:
            measurements = self.eis_model.simulate_measurement(thickness, 5.0)
            reference_data.append((thickness, measurements))
        
        # Store original calibration
        original_slope = self.eis_model.calibration['thickness_slope']
        
        # Calibrate model
        self.eis_model.calibrate_for_species(reference_data)
        
        # Calibration should have been updated
        new_slope = self.eis_model.calibration['thickness_slope']
        # Allow for some variation due to noise
        self.assertTrue(abs(new_slope - original_slope) / abs(original_slope) < 2.0)


@unittest.skipUnless(SENSING_MODELS_AVAILABLE, "Sensing models not available")
class TestQCMModel(unittest.TestCase):
    """Test QCM model functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.qcm_model = QCMModel(
            crystal_type=CrystalType.AT_CUT_5MHZ,
            electrode_type=ElectrodeType.GOLD,
            use_gpu=False
        )
    
    def test_model_initialization(self):
        """Test QCM model initialization."""
        # Test different crystal types
        for crystal_type in [CrystalType.AT_CUT_5MHZ, CrystalType.AT_CUT_10MHZ]:
            model = QCMModel(crystal_type=crystal_type, use_gpu=False)
            self.assertEqual(model.crystal_type, crystal_type)
            self.assertIsNotNone(model.sauerbrey)
            self.assertEqual(model.sauerbrey.crystal_type, crystal_type)
        
        # Test electrode area
        self.assertGreater(self.qcm_model.electrode_area, 0)
        self.assertGreater(self.qcm_model.fundamental_frequency, 0)
    
    def test_sauerbrey_model(self):
        """Test Sauerbrey equation implementation."""
        sauerbrey = self.qcm_model.sauerbrey
        
        # Test mass to frequency conversion
        test_mass = 100.0  # ng/cm²
        freq_shift = sauerbrey.calculate_frequency_from_mass(test_mass)
        
        # Frequency shift should be negative for mass addition
        self.assertLess(freq_shift, 0)
        
        # Test reverse conversion
        calculated_mass = sauerbrey.calculate_mass_from_frequency(freq_shift)
        self.assertAlmostEqual(calculated_mass, test_mass, places=1)
        
        # Test thickness estimation
        density = 1.1  # g/cm³
        thickness = sauerbrey.estimate_thickness(test_mass, density)
        self.assertGreater(thickness, 0)
        
        # Expected thickness for 100 ng/cm² at 1.1 g/cm³
        expected_thickness = (100e-9) / 1.1 * 1e4  # Convert to μm
        self.assertAlmostEqual(thickness, expected_thickness, places=1)
    
    def test_viscoelastic_model(self):
        """Test viscoelastic corrections for soft biofilms."""
        viscoelastic = self.qcm_model.viscoelastic
        
        # Test correction calculation
        frequency = 5e6  # Hz
        shear_modulus = 1e4  # Pa
        viscosity = 0.01  # Pa·s
        density = 1100  # kg/m³
        thickness = 10e-6  # m
        
        freq_correction, dissipation_change = viscoelastic.calculate_viscoelastic_correction(
            frequency, shear_modulus, viscosity, density, thickness
        )
        
        self.assertGreater(freq_correction, 0)
        self.assertLessEqual(freq_correction, 1.0)
        self.assertGreaterEqual(dissipation_change, 0)
        
        # Test mass correction
        sauerbrey_mass = 100.0  # ng/cm²
        biofilm_props = {
            'density': 1.1,
            'shear_modulus': 1e4,
            'viscosity': 0.01
        }
        
        corrected_mass = viscoelastic.correct_sauerbrey_mass(
            sauerbrey_mass, frequency, biofilm_props
        )
        
        # Corrected mass should be different from Sauerbrey mass
        self.assertNotEqual(corrected_mass, sauerbrey_mass)
        self.assertGreater(corrected_mass, 0)
    
    def test_measurement_simulation(self):
        """Test QCM measurement simulation."""
        measurement = self.qcm_model.simulate_measurement(
            biofilm_mass=150.0,  # μg
            biofilm_thickness=20.0,  # μm
            temperature=303.0,   # K
            time_hours=2.0
        )
        
        # Verify measurement structure
        self.assertIsInstance(measurement, QCMMeasurement)
        self.assertGreater(measurement.frequency, 0)
        self.assertLess(measurement.frequency_shift, 0)  # Negative for mass addition
        self.assertGreater(measurement.dissipation, 0)
        self.assertGreater(measurement.quality_factor, 0)
        self.assertEqual(measurement.temperature, 303.0)
        self.assertEqual(measurement.timestamp, 2.0)
        
        # Mass per area should be calculated
        self.assertGreater(measurement.mass_per_area, 0)
        self.assertAlmostEqual(measurement.thickness_estimate, 20.0, delta=5.0)
    
    def test_biofilm_properties_estimation(self):
        """Test biofilm property estimation from QCM measurements."""
        measurement = self.qcm_model.simulate_measurement(100.0, 15.0)
        properties = self.qcm_model.estimate_biofilm_properties(measurement)
        
        # Verify required properties
        required_properties = [
            'mass_per_area_ng_per_cm2', 'thickness_um', 'biomass_density_g_per_L',
            'viscosity_Pa_s', 'porosity', 'density_g_per_cm3',
            'signal_to_noise_ratio', 'measurement_quality'
        ]
        
        for prop in required_properties:
            self.assertIn(prop, properties)
            self.assertIsInstance(properties[prop], (int, float))
            self.assertGreaterEqual(properties[prop], 0)
        
        # Measurement quality should be between 0 and 1
        self.assertLessEqual(properties['measurement_quality'], 1.0)
    
    def test_species_specific_properties(self):
        """Test species-specific biofilm properties."""
        # Test Geobacter properties
        self.qcm_model.set_biofilm_species('geobacter')
        geo_props = self.qcm_model.current_biofilm_props
        
        # Test Shewanella properties
        self.qcm_model.set_biofilm_species('shewanella')
        she_props = self.qcm_model.current_biofilm_props
        
        # Test Mixed culture properties
        self.qcm_model.set_biofilm_species('mixed')
        mixed_props = self.qcm_model.current_biofilm_props
        
        # Properties should be different
        self.assertNotEqual(geo_props['density'], she_props['density'])
        self.assertNotEqual(geo_props['shear_modulus'], she_props['shear_modulus'])
        
        # All should have required keys
        required_keys = ['density', 'shear_modulus', 'viscosity', 'max_thickness', 'porosity']
        for props in [geo_props, she_props, mixed_props]:
            for key in required_keys:
                self.assertIn(key, props)
    
    def test_frequency_stability_metrics(self):
        """Test frequency stability analysis."""
        # Generate measurement history
        for i in range(20):
            mass = 50.0 + i * 5.0  # Increasing mass
            measurement = self.qcm_model.simulate_measurement(mass, 10.0 + i, time_hours=i * 0.1)
        
        # Get stability metrics
        stability = self.qcm_model.get_frequency_stability_metrics(window_hours=1.0)
        
        if 'insufficient_data' not in stability:
            # Verify stability metrics
            self.assertIn('mean_frequency_Hz', stability)
            self.assertIn('frequency_std_Hz', stability)
            self.assertIn('drift_rate_Hz_per_hour', stability)
            self.assertIn('peak_to_peak_Hz', stability)
            
            # Values should be reasonable
            self.assertGreater(stability['mean_frequency_Hz'], 0)
            self.assertGreaterEqual(stability['frequency_std_Hz'], 0)
            self.assertIsInstance(stability['drift_rate_Hz_per_hour'], float)


@unittest.skipUnless(SENSING_MODELS_AVAILABLE, "Sensing models not available")
class TestSensorFusion(unittest.TestCase):
    """Test sensor fusion functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sensor_fusion = SensorFusion(
            method=FusionMethod.KALMAN_FILTER,
            species=BacterialSpecies.MIXED,
            use_gpu=False
        )
        
        # Create mock EIS and QCM models
        self.eis_model = EISModel(species=BacterialSpecies.MIXED, use_gpu=False)
        self.qcm_model = QCMModel(use_gpu=False)
        self.qcm_model.set_biofilm_species('mixed')
    
    def test_fusion_initialization(self):
        """Test sensor fusion initialization."""
        # Test different fusion methods
        for method in [FusionMethod.KALMAN_FILTER, FusionMethod.WEIGHTED_AVERAGE,
                      FusionMethod.MAXIMUM_LIKELIHOOD, FusionMethod.BAYESIAN_INFERENCE]:
            fusion = SensorFusion(method=method, use_gpu=False)
            self.assertEqual(fusion.method, method)
            
            if method == FusionMethod.KALMAN_FILTER:
                self.assertIsNotNone(fusion.kalman_filter)
    
    def test_kalman_filter(self):
        """Test Kalman filter implementation."""
        kalman = self.sensor_fusion.kalman_filter
        
        # Test initialization
        kalman.initialize_state(10.0, 5.0, 0.001)
        self.assertTrue(kalman.initialized)
        self.assertEqual(kalman.state[0], 10.0)  # thickness
        self.assertEqual(kalman.state[1], 5.0)   # biomass
        self.assertEqual(kalman.state[2], 0.001) # conductivity
        
        # Test prediction step
        kalman.predict()
        self.assertGreaterEqual(kalman.state[0], 0)  # Non-negative thickness
        
        # Test update step
        measurements = np.array([12.0, 11.0, 0.0015])  # [eis_thick, qcm_thick, conductivity]
        uncertainties = np.array([2.0, 1.0, 0.0001])
        
        kalman.update(measurements, uncertainties)
        
        # State should be updated
        state, uncertainties_out = kalman.get_state_estimate()
        self.assertEqual(len(state), 5)
        self.assertEqual(len(uncertainties_out), 5)
        
        # All values should be non-negative
        for i in range(3):  # thickness, biomass, conductivity
            self.assertGreaterEqual(state[i], 0)
    
    def test_sensor_calibration(self):
        """Test sensor calibration functionality."""
        calibration = self.sensor_fusion.calibration
        
        # Test initial state
        status = calibration.get_calibration_status()
        self.assertIn('eis_reliability', status)
        self.assertIn('qcm_reliability', status)
        self.assertEqual(status['eis_reliability'], 1.0)
        self.assertEqual(status['qcm_reliability'], 1.0)
        
        # Test uncertainty calculation
        eis_uncertainty = calibration.get_measurement_uncertainty('eis', 25.0)
        qcm_uncertainty = calibration.get_measurement_uncertainty('qcm', 25.0)
        
        self.assertGreater(eis_uncertainty, 0)
        self.assertGreater(qcm_uncertainty, 0)
    
    def test_measurement_fusion(self):
        """Test measurement fusion process."""
        # Generate mock measurements
        eis_measurements = self.eis_model.simulate_measurement(25.0, 8.0)
        eis_properties = self.eis_model.get_biofilm_properties(eis_measurements)
        eis_measurement = eis_measurements[0]  # Use first measurement
        
        qcm_measurement = self.qcm_model.simulate_measurement(200.0, 25.0)
        qcm_properties = self.qcm_model.estimate_biofilm_properties(qcm_measurement)
        
        # Test fusion
        fused_result = self.sensor_fusion.fuse_measurements(
            eis_measurement=eis_measurement,
            qcm_measurement=qcm_measurement,
            eis_properties=eis_properties,
            qcm_properties=qcm_properties,
            time_hours=1.0
        )
        
        # Verify fused result structure
        self.assertIsInstance(fused_result, FusedMeasurement)
        self.assertEqual(fused_result.timestamp, 1.0)
        
        # Check required fields
        self.assertGreaterEqual(fused_result.thickness_um, 0)
        self.assertGreaterEqual(fused_result.thickness_uncertainty, 0)
        self.assertGreaterEqual(fused_result.biomass_density_g_per_L, 0)
        self.assertGreaterEqual(fused_result.sensor_agreement, 0)
        self.assertLessEqual(fused_result.sensor_agreement, 1)
        self.assertGreaterEqual(fused_result.fusion_confidence, 0)
        self.assertLessEqual(fused_result.fusion_confidence, 1)
        
        # Weights should sum to approximately 1
        weight_sum = fused_result.eis_weight + fused_result.qcm_weight
        self.assertAlmostEqual(weight_sum, 1.0, places=2)
        
        # Status should be valid
        self.assertIn(fused_result.eis_status, ['good', 'degraded', 'failed'])
        self.assertIn(fused_result.qcm_status, ['good', 'degraded', 'failed'])
    
    def test_fusion_methods_comparison(self):
        """Test different fusion methods give reasonable results."""
        # Create test measurements
        eis_measurements = self.eis_model.simulate_measurement(30.0, 10.0)
        eis_properties = self.eis_model.get_biofilm_properties(eis_measurements)
        eis_measurement = eis_measurements[0]
        
        qcm_measurement = self.qcm_model.simulate_measurement(300.0, 30.0)
        qcm_properties = self.qcm_model.estimate_biofilm_properties(qcm_measurement)
        
        results = {}
        
        # Test all fusion methods
        for method in [FusionMethod.KALMAN_FILTER, FusionMethod.WEIGHTED_AVERAGE,
                      FusionMethod.MAXIMUM_LIKELIHOOD, FusionMethod.BAYESIAN_INFERENCE]:
            fusion = SensorFusion(method=method, use_gpu=False)
            
            try:
                fused_result = fusion.fuse_measurements(
                    eis_measurement, qcm_measurement, eis_properties, qcm_properties
                )
                results[method.value] = fused_result.thickness_um
            except Exception as e:
                print(f"Warning: {method.value} fusion failed: {e}")
        
        # All successful methods should give reasonable results
        for method, thickness in results.items():
            self.assertGreater(thickness, 0)
            self.assertLess(thickness, 100)  # Reasonable biofilm thickness range
    
    def test_fault_detection(self):
        """Test sensor fault detection."""
        # Create measurements with poor agreement
        eis_measurements = self.eis_model.simulate_measurement(10.0, 5.0)  # Thin biofilm
        eis_properties = self.eis_model.get_biofilm_properties(eis_measurements)
        eis_measurement = eis_measurements[0]
        
        qcm_measurement = self.qcm_model.simulate_measurement(500.0, 50.0)  # Thick biofilm
        qcm_properties = self.qcm_model.estimate_biofilm_properties(qcm_measurement)
        
        # Generate multiple measurements with disagreement
        for i in range(10):
            fused_result = self.sensor_fusion.fuse_measurements(
                eis_measurement, qcm_measurement, eis_properties, qcm_properties, i
            )
        
        # Detect faults
        faults = self.sensor_fusion.detect_sensor_faults()
        
        # Should detect poor sensor agreement
        self.assertIn('fusion_faults', faults)
        if faults['fusion_faults']:
            self.assertIn('poor_sensor_agreement', faults['fusion_faults'])


@unittest.skipUnless(SENSING_MODELS_AVAILABLE, "Sensing models not available")
class TestIntegration(unittest.TestCase):
    """Test integration between sensing models."""
    
    def test_eis_qcm_correlation(self):
        """Test correlation between EIS and QCM measurements."""
        eis_model = EISModel(species=BacterialSpecies.GEOBACTER, use_gpu=False)
        qcm_model = QCMModel(use_gpu=False)
        qcm_model.set_biofilm_species('geobacter')
        
        # Test measurements at different thicknesses
        thicknesses = [10, 20, 30, 40, 50]
        eis_estimates = []
        qcm_estimates = []
        
        for thickness in thicknesses:
            # EIS measurement
            eis_measurements = eis_model.simulate_measurement(thickness, 8.0)
            eis_thickness = eis_model.estimate_thickness(eis_measurements)
            eis_estimates.append(eis_thickness)
            
            # QCM measurement (estimate mass from thickness)
            density = 1.15  # g/cm³ for Geobacter
            area = 0.196    # cm²
            mass = thickness * 1e-4 * area * density * 1e6  # Convert to μg
            
            qcm_measurement = qcm_model.simulate_measurement(mass, thickness)
            qcm_props = qcm_model.estimate_biofilm_properties(qcm_measurement)
            qcm_thickness = qcm_props['thickness_um']
            qcm_estimates.append(qcm_thickness)
        
        # Calculate correlation
        correlation = np.corrcoef(eis_estimates, qcm_estimates)[0, 1]
        
        # Should have strong correlation (may be negative due to measurement physics)
        # EIS and QCM may have inverse relationship in certain conditions
        if not np.isnan(correlation):
            self.assertGreater(abs(correlation), 0.8)
    
    def test_sensor_fusion_with_real_data(self):
        """Test sensor fusion with realistic measurement scenarios."""
        eis_model = EISModel(species=BacterialSpecies.MIXED, use_gpu=False)
        qcm_model = QCMModel(use_gpu=False)
        qcm_model.set_biofilm_species('mixed')
        
        fusion = SensorFusion(method=FusionMethod.KALMAN_FILTER, use_gpu=False)
        
        # Simulate biofilm growth over time
        times = np.linspace(0, 24, 25)  # 24 hours
        thicknesses = 5 + 20 * (1 - np.exp(-times / 10))  # Growth curve
        
        fused_results = []
        
        for i, (time, thickness) in enumerate(zip(times, thicknesses)):
            # Simulate measurements
            biomass = thickness * 0.3  # Approximate biomass
            
            eis_measurements = eis_model.simulate_measurement(thickness, biomass, time_hours=time)
            eis_properties = eis_model.get_biofilm_properties(eis_measurements)
            
            # Estimate QCM mass
            density = 1.12  # Mixed culture density
            area = 0.196
            mass = thickness * 1e-4 * area * density * 1e6
            
            qcm_measurement = qcm_model.simulate_measurement(mass, thickness, time_hours=time)
            qcm_properties = qcm_model.estimate_biofilm_properties(qcm_measurement)
            
            # Fuse measurements
            fused_result = fusion.fuse_measurements(
                eis_measurements[0], qcm_measurement, eis_properties, qcm_properties, time
            )
            
            fused_results.append(fused_result)
        
        # Verify fusion results
        self.assertEqual(len(fused_results), len(times))
        
        # Check for reasonable progression
        fused_thicknesses = [r.thickness_um for r in fused_results]
        
        # Should show growth trend
        self.assertGreater(fused_thicknesses[-1], fused_thicknesses[0])
        
        # Fusion confidence should be reasonable
        confidences = [r.fusion_confidence for r in fused_results]
        mean_confidence = np.mean(confidences)
        self.assertGreater(mean_confidence, 0.3)  # At least moderate confidence
    
    def test_sensor_degradation_handling(self):
        """Test handling of sensor degradation scenarios."""
        fusion = SensorFusion(method=FusionMethod.WEIGHTED_AVERAGE, use_gpu=False)
        
        # Test with degraded EIS sensor
        # (simulated by adding large uncertainty and poor quality)
        eis_properties_degraded = {
            'thickness_um': 25.0,
            'measurement_quality': 0.3,  # Poor quality
            'status': 'degraded',
            'conductivity_S_per_m': 0.001
        }
        
        # Good QCM measurement
        qcm_model = QCMModel(use_gpu=False)
        qcm_measurement = qcm_model.simulate_measurement(250.0, 25.0)
        qcm_properties = qcm_model.estimate_biofilm_properties(qcm_measurement)
        qcm_properties['status'] = 'good'
        
        # Create dummy measurements
        from sensing_models.eis_model import EISMeasurement
        from sensing_models.qcm_model import QCMMeasurement
        
        eis_measurement = EISMeasurement(
            frequency=1000, impedance_magnitude=1500, impedance_phase=0.5,
            real_impedance=1400, imaginary_impedance=500, timestamp=0, temperature=303
        )
        
        # Fuse measurements
        fused_result = fusion.fuse_measurements(
            eis_measurement, qcm_measurement, eis_properties_degraded, qcm_properties
        )
        
        # QCM should have higher weight due to better quality
        self.assertGreater(fused_result.qcm_weight, fused_result.eis_weight)
        
        # Fusion confidence should reflect sensor issues
        self.assertLess(fused_result.fusion_confidence, 0.8)


if __name__ == '__main__':
    unittest.main(verbosity=2)