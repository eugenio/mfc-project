"""
Electrochemical Impedance Spectroscopy (EIS) Model for Biofilm Characterization

This module implements EIS-based biofilm thickness and conductivity measurement
based on literature-validated parameters from multiple studies (2022-2024).

Key References:
- PMC10485796: Real-time biofilm monitoring with impedance biosensors
- PMC10452506: Comprehensive EIS biofilm sensing review
- Frontiers Microbiology 2022: EIS applied to MFCs

Implementation includes:
- Equivalent circuit modeling (Randles circuit with biofilm component)
- Frequency-dependent impedance calculations
- Biofilm thickness estimation from impedance data
- Species-specific calibration (G. sulfurreducens, S. oneidensis)
- Real-time measurement simulation
"""

import cmath
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

# Add GPU acceleration
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from gpu_acceleration import get_gpu_accelerator
except ImportError:
    get_gpu_accelerator = None


class BacterialSpecies(Enum):
    """Bacterial species for EIS measurements."""
    GEOBACTER = "geobacter_sulfurreducens"
    SHEWANELLA = "shewanella_oneidensis"
    MIXED = "mixed_culture"


@dataclass
class EISMeasurement:
    """Container for EIS measurement data."""

    frequency: float  # Hz
    impedance_magnitude: float  # Ohm
    impedance_phase: float  # radians
    real_impedance: float  # Ohm
    imaginary_impedance: float  # Ohm
    timestamp: float  # hours
    temperature: float  # K

    @property
    def complex_impedance(self) -> complex:
        """Get complex impedance."""
        return complex(self.real_impedance, self.imaginary_impedance)

    @classmethod
    def from_complex(cls, frequency: float, impedance: complex,
                    timestamp: float = 0.0, temperature: float = 303.0) -> 'EISMeasurement':
        """Create measurement from complex impedance."""
        magnitude = abs(impedance)
        phase = cmath.phase(impedance)
        return cls(
            frequency=frequency,
            impedance_magnitude=magnitude,
            impedance_phase=phase,
            real_impedance=impedance.real,
            imaginary_impedance=impedance.imag,
            timestamp=timestamp,
            temperature=temperature
        )


class EISCircuitModel:
    """
    Equivalent circuit model for EIS biofilm characterization.

    Implements modified Randles circuit with biofilm component:
    Rs - (Cdl || (Rbio - Cbio)) - Rct

    Where:
    - Rs: Solution resistance
    - Cdl: Double layer capacitance
    - Rbio: Biofilm resistance
    - Cbio: Biofilm capacitance
    - Rct: Charge transfer resistance
    """

    def __init__(self, species: BacterialSpecies = BacterialSpecies.MIXED):
        """Initialize circuit model with species-specific parameters."""
        self.species = species
        self._load_species_parameters()

        # Literature-validated frequency range (PMC10452506)
        self.frequency_range = (100, 1e6)  # Hz

        # Default circuit parameters (will be updated by biofilm state)
        self.reset_parameters()

    def _load_species_parameters(self):
        """Load species-specific EIS parameters from literature."""
        if self.species == BacterialSpecies.GEOBACTER:
            # G. sulfurreducens - conductive biofilms
            self.species_params = {
                'base_resistivity': 100.0,  # Ohm*cm (conductive biofilm)
                'capacitance_factor': 1e-6,  # F/cm² per μm
                'thickness_sensitivity': 0.8,  # Impedance change per μm
                'max_thickness': 80.0,  # μm (literature maximum)
                'conductivity': 1e-3,  # S/cm (moderate conductivity)
            }
        elif self.species == BacterialSpecies.SHEWANELLA:
            # S. oneidensis - shuttle-mediated transfer
            self.species_params = {
                'base_resistivity': 500.0,  # Ohm*cm (less conductive)
                'capacitance_factor': 2e-6,  # F/cm² per μm
                'thickness_sensitivity': 0.6,  # Lower impedance sensitivity
                'max_thickness': 60.0,  # μm
                'conductivity': 5e-4,  # S/cm (lower conductivity)
            }
        else:  # Mixed culture
            # Average of both species with synergy factor
            self.species_params = {
                'base_resistivity': 250.0,  # Ohm*cm (intermediate)
                'capacitance_factor': 1.5e-6,  # F/cm² per μm
                'thickness_sensitivity': 0.7,  # Intermediate sensitivity
                'max_thickness': 70.0,  # μm
                'conductivity': 7.5e-4,  # S/cm (intermediate)
            }

    def reset_parameters(self):
        """Reset circuit parameters to baseline values."""
        # Solution resistance (electrode geometry dependent)
        self.Rs = 50.0  # Ohm (typical for small electrodes)

        # Double layer capacitance (literature: ~10-100 μF/cm²)
        self.Cdl = 50e-6  # F (for 1 cm² electrode)

        # Biofilm parameters (updated by biofilm state)
        self.Rbio = 1000.0  # Ohm (initial high resistance)
        self.Cbio = 1e-6   # F (initial low capacitance)

        # Charge transfer resistance
        self.Rct = 500.0   # Ohm

    def update_from_biofilm_state(self, thickness: float, biomass_density: float,
                                  porosity: float = 0.8, electrode_area: float = 1e-4):
        """
        Update circuit parameters from biofilm physical state.

        Args:
            thickness: Biofilm thickness (μm)
            biomass_density: Biomass density (g/L)
            porosity: Biofilm porosity (0-1)
            electrode_area: Electrode area (m²)
        """
        # Convert area to cm²
        area_cm2 = electrode_area * 1e4

        # Biofilm resistance calculation
        # Based on: R = ρ * L / A
        resistivity = self.species_params['base_resistivity']
        resistivity *= (1 + (1 - porosity) * biomass_density / 10.0)  # Density effect

        thickness_cm = thickness * 1e-4  # μm to cm
        self.Rbio = resistivity * thickness_cm / area_cm2

        # Ensure minimum resistance
        self.Rbio = max(self.Rbio, 10.0)

        # Biofilm capacitance calculation
        # Based on: C = ε * A / d
        capacitance_factor = self.species_params['capacitance_factor']
        self.Cbio = capacitance_factor * area_cm2 * thickness / 10.0

        # Ensure minimum capacitance
        self.Cbio = max(self.Cbio, 1e-9)

        # Update charge transfer resistance (biofilm thickness effect)
        # For electroactive biofilms, thicker biofilms have LOWER charge transfer resistance
        # due to more electron transfer pathways
        thickness_factor = max(0.1, 1.0 - 0.8 * thickness / self.species_params['max_thickness'])
        self.Rct = 500.0 * thickness_factor

    def calculate_impedance(self, frequency: float) -> complex:
        """
        Calculate complex impedance at given frequency.

        Args:
            frequency: Frequency (Hz)

        Returns:
            Complex impedance (Ohm)
        """
        omega = 2 * np.pi * frequency

        # Capacitive impedances
        Zcdl = 1 / (1j * omega * self.Cdl)
        Zcbio = 1 / (1j * omega * self.Cbio)

        # Biofilm branch: Rbio in series with Cbio
        Zbio_branch = self.Rbio + Zcbio

        # Parallel combination of Cdl and biofilm branch
        Zparallel = 1 / (1/Zcdl + 1/Zbio_branch)

        # Total impedance: Rs + parallel combination + Rct
        Z_total = self.Rs + Zparallel + self.Rct

        return Z_total

    def fit_parameters(self, measurements: list[EISMeasurement]) -> dict[str, float]:
        """
        Fit circuit parameters to experimental data.

        Args:
            measurements: List of EIS measurements

        Returns:
            Fitted parameters dictionary
        """
        # Extract frequency and impedance data
        frequencies = np.array([m.frequency for m in measurements])
        impedances = np.array([m.complex_impedance for m in measurements])

        # Simple parameter estimation using high and low frequency limits
        # High frequency: Rs ≈ Re(Z) at f → ∞
        high_freq_idx = np.argmax(frequencies)
        self.Rs = impedances[high_freq_idx].real

        # Low frequency: Total resistance ≈ Re(Z) at f → 0
        low_freq_idx = np.argmin(frequencies)
        total_resistance = impedances[low_freq_idx].real
        self.Rct = max(total_resistance - self.Rs - self.Rbio, 10.0)

        # Estimate capacitances from phase behavior
        # Find frequency where phase is maximum (characteristic frequency)
        phases = np.array([abs(m.impedance_phase) for m in measurements])
        max_phase_idx = np.argmax(phases)
        char_freq = frequencies[max_phase_idx]

        # Estimate Cdl from characteristic frequency
        self.Cdl = 1 / (2 * np.pi * char_freq * self.Rs)
        self.Cdl = np.clip(self.Cdl, 1e-9, 1e-3)  # Reasonable bounds

        return {
            'Rs': self.Rs,
            'Cdl': self.Cdl,
            'Rbio': self.Rbio,
            'Cbio': self.Cbio,
            'Rct': self.Rct
        }


class EISModel:
    """
    Complete EIS model for biofilm characterization in MFCs.

    Features:
    - Real-time impedance spectrum simulation
    - Biofilm thickness estimation
    - Species-specific calibration
    - Noise and drift modeling
    - Integration with biofilm kinetics
    """

    def __init__(self, species: BacterialSpecies = BacterialSpecies.MIXED,
                 electrode_area: float = 1e-4, use_gpu: bool = True):
        """
        Initialize EIS model.

        Args:
            species: Bacterial species
            electrode_area: Electrode area (m²)
            use_gpu: Enable GPU acceleration
        """
        self.species = species
        self.electrode_area = electrode_area
        self.use_gpu = use_gpu

        # Initialize GPU if available
        self.gpu_acc = None
        if use_gpu and get_gpu_accelerator:
            self.gpu_acc = get_gpu_accelerator()
            self.gpu_available = self.gpu_acc.is_gpu_available()
        else:
            self.gpu_available = False

        # Initialize circuit model
        self.circuit = EISCircuitModel(species)

        # Measurement parameters
        self.frequency_points = np.logspace(2, 6, 50)  # 100 Hz to 1 MHz, 50 points
        self.measurement_amplitude = 0.010  # 10 mV AC amplitude

        # Calibration parameters (species-specific)
        self._initialize_calibration()

        # Noise and drift parameters
        self.noise_level = 0.02  # 2% relative noise
        self.drift_rate = 0.001  # 0.1% per hour

        # Historical data
        self.measurement_history = []
        self.thickness_history = []
        self.time_history = []

        # Current state
        self.current_thickness = 0.0
        self.current_biomass = 0.0
        self.measurement_time = 0.0

    def _initialize_calibration(self):
        """Initialize species-specific calibration parameters."""
        if self.species == BacterialSpecies.GEOBACTER:
            # G. sulfurreducens calibration (literature-based)
            # For electroactive biofilms, impedance decreases with thickness due to enhanced electron transfer
            self.calibration = {
                'thickness_slope': 150.0,  # Ohm per μm (positive - impedance decreases with thickness)
                'thickness_intercept': 2000.0,  # Ohm at zero thickness
                'max_thickness': 80.0,  # μm
                'sensitivity_range': (5.0, 60.0),  # μm (linear range)
            }
        elif self.species == BacterialSpecies.SHEWANELLA:
            # S. oneidensis calibration
            self.calibration = {
                'thickness_slope': 100.0,  # Ohm per μm (positive - impedance decreases with thickness)
                'thickness_intercept': 1500.0,  # Ohm
                'max_thickness': 60.0,  # μm
                'sensitivity_range': (3.0, 45.0),  # μm
            }
        else:  # Mixed culture
            # Average calibration
            self.calibration = {
                'thickness_slope': 125.0,  # Ohm per μm (positive - impedance decreases with thickness)
                'thickness_intercept': 1750.0,  # Ohm
                'max_thickness': 70.0,  # μm
                'sensitivity_range': (4.0, 50.0),  # μm
            }

    def simulate_measurement(self, biofilm_thickness: float, biomass_density: float,
                           porosity: float = 0.8, temperature: float = 303.0,
                           time_hours: float = 0.0) -> list[EISMeasurement]:
        """
        Simulate EIS measurement for given biofilm state.

        Args:
            biofilm_thickness: Biofilm thickness (μm)
            biomass_density: Biomass density (g/L)
            porosity: Biofilm porosity (0-1)
            temperature: Temperature (K)
            time_hours: Measurement time (hours)

        Returns:
            List of EIS measurements across frequency range
        """
        # Update circuit parameters from biofilm state
        self.circuit.update_from_biofilm_state(
            biofilm_thickness, biomass_density, porosity, self.electrode_area
        )

        measurements = []

        # Calculate impedance across frequency range
        if self.gpu_available:
            # GPU-accelerated calculation
            frequencies_gpu = self.gpu_acc.array(self.frequency_points)
            impedances = self._calculate_impedance_spectrum_gpu(frequencies_gpu)
        else:
            # CPU calculation
            impedances = []
            for freq in self.frequency_points:
                Z = self.circuit.calculate_impedance(freq)
                impedances.append(Z)

        # Create measurements with noise
        for i, freq in enumerate(self.frequency_points):
            Z = impedances[i]

            # Add measurement noise
            noise_factor = 1.0 + np.random.normal(0, self.noise_level)
            Z_noisy = Z * noise_factor

            # Add drift effect
            drift_factor = 1.0 + self.drift_rate * time_hours
            Z_drift = Z_noisy * drift_factor

            measurement = EISMeasurement.from_complex(
                frequency=freq,
                impedance=Z_drift,
                timestamp=time_hours,
                temperature=temperature
            )
            measurements.append(measurement)

        # Store current state
        self.current_thickness = biofilm_thickness
        self.current_biomass = biomass_density
        self.measurement_time = time_hours

        # Update history
        self.measurement_history.append(measurements)
        self.thickness_history.append(biofilm_thickness)
        self.time_history.append(time_hours)

        return measurements

    def _calculate_impedance_spectrum_gpu(self, frequencies_gpu):
        """Calculate impedance spectrum using GPU acceleration."""
        # GPU implementation for parallel frequency calculation
        self.gpu_acc.multiply(frequencies_gpu, 2 * np.pi)

        # Capacitive impedances (simplified for GPU)
        # This is a simplified version - full complex arithmetic requires more GPU ops
        impedances = []
        frequencies_cpu = self.gpu_acc.to_cpu(frequencies_gpu)

        for freq in frequencies_cpu:
            Z = self.circuit.calculate_impedance(freq)
            impedances.append(Z)

        return impedances

    def estimate_thickness(self, measurements: list[EISMeasurement],
                          method: str = 'low_frequency') -> float:
        """
        Estimate biofilm thickness from EIS measurements.

        Args:
            measurements: EIS measurement data
            method: Estimation method ('low_frequency', 'characteristic', 'fitting')

        Returns:
            Estimated thickness (μm)
        """
        if not measurements:
            return 0.0

        if method == 'low_frequency':
            # Use low-frequency impedance for thickness estimation
            low_freq_measurements = [m for m in measurements if m.frequency <= 1000]
            if not low_freq_measurements:
                low_freq_measurements = [min(measurements, key=lambda x: x.frequency)]

            avg_impedance = np.mean([m.impedance_magnitude for m in low_freq_measurements])

            # Linear calibration model
            thickness = (self.calibration['thickness_intercept'] - avg_impedance) / self.calibration['thickness_slope']

        elif method == 'characteristic':
            # Use characteristic frequency for thickness estimation
            phases = [abs(m.impedance_phase) for m in measurements]
            max_phase_idx = np.argmax(phases)
            char_impedance = measurements[max_phase_idx].impedance_magnitude

            # Characteristic impedance correlation
            thickness = (self.calibration['thickness_intercept'] - char_impedance) / self.calibration['thickness_slope']

        elif method == 'fitting':
            # Use circuit fitting for thickness estimation
            fitted_params = self.circuit.fit_parameters(measurements)

            # Relate biofilm resistance to thickness
            Rbio = fitted_params['Rbio']
            resistivity = self.circuit.species_params['base_resistivity']
            thickness_cm = Rbio * self.electrode_area * 1e4 / resistivity
            thickness = thickness_cm * 1e4  # cm to μm

        else:
            raise ValueError(f"Unknown estimation method: {method}")

        # Apply bounds
        thickness = np.clip(thickness, 0.0, self.calibration['max_thickness'])

        return thickness

    def get_biofilm_properties(self, measurements: list[EISMeasurement]) -> dict[str, float]:
        """
        Extract biofilm properties from EIS measurements.

        Args:
            measurements: EIS measurement data

        Returns:
            Dictionary of biofilm properties
        """
        if not measurements:
            return {}

        # Estimate thickness
        thickness = self.estimate_thickness(measurements)

        # Fit circuit parameters
        fitted_params = self.circuit.fit_parameters(measurements)

        # Calculate derived properties
        conductivity = 1.0 / fitted_params['Rbio'] * (thickness * 1e-6) / self.electrode_area
        capacitance_per_area = fitted_params['Cbio'] / (self.electrode_area * 1e4)  # F/cm²

        # Estimate biomass from impedance (empirical correlation)
        low_freq_impedance = min([m.impedance_magnitude for m in measurements])
        biomass_estimate = max(0, (2000 - low_freq_impedance) / 100.0)  # g/L (empirical)

        return {
            'thickness_um': thickness,
            'conductivity_S_per_m': conductivity,
            'capacitance_F_per_cm2': capacitance_per_area,
            'biomass_estimate_g_per_L': biomass_estimate,
            'solution_resistance_ohm': fitted_params['Rs'],
            'biofilm_resistance_ohm': fitted_params['Rbio'],
            'charge_transfer_resistance_ohm': fitted_params['Rct'],
            'measurement_quality': self._assess_measurement_quality(measurements)
        }

    def _assess_measurement_quality(self, measurements: list[EISMeasurement]) -> float:
        """Assess measurement quality (0-1 score)."""
        if len(measurements) < 10:
            return 0.5  # Insufficient data

        # Check frequency coverage
        frequencies = [m.frequency for m in measurements]
        freq_range = max(frequencies) / min(frequencies)
        coverage_score = min(1.0, freq_range / 1e4)  # Good if 4+ decades

        # Check phase behavior (should have characteristic shape)
        phases = [abs(m.impedance_phase) for m in measurements]
        phase_range = max(phases) - min(phases)
        phase_score = min(1.0, phase_range / (np.pi/2))  # Good if >45° range

        # Check impedance magnitude consistency
        magnitudes = [m.impedance_magnitude for m in measurements]
        magnitude_cv = np.std(magnitudes) / np.mean(magnitudes)
        consistency_score = max(0.0, 1.0 - magnitude_cv)

        return (coverage_score + phase_score + consistency_score) / 3.0

    def calibrate_for_species(self, reference_measurements: list[tuple[float, list[EISMeasurement]]]):
        """
        Calibrate EIS model for specific species using reference data.

        Args:
            reference_measurements: List of (known_thickness, measurements) pairs
        """
        if len(reference_measurements) < 3:
            print("Warning: Insufficient calibration data (need ≥3 points)")
            return

        thicknesses = []
        impedances = []

        for known_thickness, measurements in reference_measurements:
            # Use low-frequency impedance as calibration metric
            low_freq_impedance = min([m.impedance_magnitude for m in measurements])
            thicknesses.append(known_thickness)
            impedances.append(low_freq_impedance)

        # Linear regression for calibration
        thicknesses = np.array(thicknesses)
        impedances = np.array(impedances)

        # Calculate slope and intercept
        slope = np.sum((thicknesses - np.mean(thicknesses)) * (impedances - np.mean(impedances))) / \
                np.sum((thicknesses - np.mean(thicknesses))**2)
        intercept = np.mean(impedances) - slope * np.mean(thicknesses)

        # Update calibration parameters
        self.calibration['thickness_slope'] = slope
        self.calibration['thickness_intercept'] = intercept

        print(f"EIS calibration updated: slope={slope:.1f} Ohm/μm, intercept={intercept:.1f} Ohm")

    def get_measurement_summary(self) -> dict[str, Any]:
        """Get summary of current EIS measurement state."""
        return {
            'species': self.species.value,
            'electrode_area_m2': self.electrode_area,
            'current_thickness_um': self.current_thickness,
            'current_biomass_g_per_L': self.current_biomass,
            'measurement_time_hours': self.measurement_time,
            'frequency_range_Hz': (self.frequency_points[0], self.frequency_points[-1]),
            'number_frequency_points': len(self.frequency_points),
            'measurement_history_length': len(self.measurement_history),
            'calibration_parameters': self.calibration,
            'circuit_parameters': {
                'Rs': self.circuit.Rs,
                'Cdl': self.circuit.Cdl,
                'Rbio': self.circuit.Rbio,
                'Cbio': self.circuit.Cbio,
                'Rct': self.circuit.Rct
            },
            'gpu_available': self.gpu_available
        }
