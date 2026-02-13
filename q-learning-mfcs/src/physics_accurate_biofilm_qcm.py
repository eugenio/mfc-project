#!/usr/bin/env python3
"""
Physics-Accurate Biofilm Thickness Model with Quartz Crystal Microbalance (QCM) Feedback
Based on literature-validated parameters for Geobacter sulfurreducens

Literature Sources:
- Babauta et al. (2014) - QCM monitoring of G. sulfurreducens biofilms
- Liu et al. (2015) - Real-time biofilm monitoring with EQCM
- Practical QCM model (PMC5579555) - Mass sensitivity distribution
- G. sulfurreducens thickness studies (Int. J. Hydrogen Energy, 2016)
"""

import matplotlib
import numpy as np

matplotlib.use('Agg')
import json
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
from path_config import get_figure_path, get_simulation_data_path
from scipy.integrate import odeint


@dataclass
class GeobacterBiofilmParameters:
    """Literature-validated parameters for Geobacter sulfurreducens biofilms"""

    # Biofilm thickness parameters (from literature)
    optimal_thickness_um: float = 20.0      # μm - maximum electrochemical activity
    maximum_thickness_um: float = 45.0      # μm - growth cessation point
    viable_thickness_um: float = 80.0       # μm - maximum viable under optimal conditions

    # Physical properties
    density: float = 1100.0                 # kg/m³ - typical biofilm density
    young_modulus: float = 50e3             # Pa - elastic modulus (10-100 kPa range)
    poisson_ratio: float = 0.45             # dimensionless - nearly incompressible
    porosity: float = 0.85                  # dimensionless - high water content

    # Electrochemical properties
    max_current_density: float = 172e-6     # A/cm² - literature value ± 29 μA/cm²
    current_per_protein: float = 5e-6       # A/μg - 2-8 μA/μg protein range
    protein_content: float = 0.15           # dimensionless - fraction of dry weight

    # Growth kinetics
    growth_rate_max: float = 0.1            # h⁻¹ - maximum specific growth rate
    decay_rate: float = 0.01                # h⁻¹ - cell death/decay rate
    substrate_affinity: float = 0.5         # mol/m³ - half-saturation constant

    # Diffusion limitations
    diffusion_coeff: float = 1e-9           # m²/s - effective diffusion in biofilm
    reaction_rate_coeff: float = 1e-4       # m³/(mol·s) - reaction rate constant

    # Inactive cell accumulation (from literature)
    inactive_fraction_start: float = 0.1    # Initial inactive fraction
    inactive_accumulation_rate: float = 0.02 # h⁻¹ - rate of inactive cell buildup

@dataclass
class QCMSensorParameters:
    """Physics-based QCM sensor parameters from literature"""

    # Crystal properties
    fundamental_freq: float = 5e6           # Hz - 5 MHz fundamental frequency
    density_quartz: float = 2648            # kg/m³ - quartz density
    shear_modulus: float = 2.947e10         # Pa - quartz shear modulus
    piezoelectric_constant: float = 2.31e-12 # C/N - d₁₁ coefficient

    # Electrode properties
    electrode_area: float = 1.37e-4         # m² - 13.7 mm² electrode area
    electrode_thickness: float = 100e-9     # m - 100 nm gold electrode
    electrode_density: float = 19300        # kg/m³ - gold density

    # Sensitivity parameters
    mass_sensitivity: float = 17.7e-9       # kg/(m²·Hz) - theoretical at 5 MHz
    frequency_stability: float = 0.1        # Hz - measurement stability
    q_factor_air: float = 50000             # dimensionless - quality factor in air
    q_factor_liquid: float = 1000           # dimensionless - Q in liquid

    # Environmental corrections
    temperature_coeff: float = -20e-6       # 1/°C - frequency temperature coefficient
    viscosity_correction: bool = True       # Enable viscoelastic corrections

    # Gaussian sensitivity distribution parameters
    sensitivity_std: float = 0.3            # relative standard deviation
    edge_factor: float = 0.1                # sensitivity at electrode edge

class PhysicsAccurateBiofilmModel:
    """Physics-accurate biofilm growth model with diffusion limitations"""

    def __init__(self, biofilm_params: GeobacterBiofilmParameters):
        self.params = biofilm_params
        self.thickness_history = []
        self.mass_history = []
        self.current_history = []
        self.inactive_fraction_history = []

    def calculate_diffusion_limitation(self, thickness_um: float, substrate_conc: float) -> float:
        """Calculate diffusion limitation factor based on thickness"""
        p = self.params

        # Convert thickness to meters
        thickness_m = thickness_um * 1e-6

        # Thiele modulus for biofilm diffusion-reaction
        phi = thickness_m * np.sqrt(p.reaction_rate_coeff * substrate_conc / p.diffusion_coeff)

        # Effectiveness factor (fraction of biofilm that's active)
        if phi < 0.1:
            eta_eff = 1.0  # No diffusion limitation
        else:
            eta_eff = np.tanh(phi) / phi

        return eta_eff

    def calculate_inactive_fraction(self, thickness_um: float, time_hours: float) -> float:
        """Calculate fraction of inactive cells based on thickness and time"""
        p = self.params

        # Inactive fraction increases with thickness (inner cells starve)
        thickness_factor = 1 + (thickness_um / p.optimal_thickness_um - 1) * 0.5
        thickness_factor = max(1.0, thickness_factor)

        # Time-dependent accumulation
        time_factor = p.inactive_fraction_start * (1 + p.inactive_accumulation_rate * time_hours)

        inactive_fraction = min(0.8, thickness_factor * time_factor)
        return inactive_fraction

    def biofilm_growth_ode(self, state: list[float], t: float, substrate_conc: float) -> list[float]:
        """ODE system for biofilm growth with physics-based limitations"""
        thickness_um, biomass_density = state
        p = self.params

        # Ensure positive thickness
        thickness_um = max(0.1, thickness_um)

        # Calculate limitations
        diffusion_factor = self.calculate_diffusion_limitation(thickness_um, substrate_conc)
        inactive_fraction = self.calculate_inactive_fraction(thickness_um, t)
        active_fraction = 1 - inactive_fraction

        # Growth rate limited by substrate and diffusion
        substrate_limitation = substrate_conc / (p.substrate_affinity + substrate_conc)
        net_growth_rate = p.growth_rate_max * substrate_limitation * diffusion_factor * active_fraction

        # Thickness change (μm/h) - limited growth model
        if thickness_um < p.optimal_thickness_um:
            # Linear growth phase
            dthickness_dt = net_growth_rate * 5.0  # μm/h growth rate
        elif thickness_um < p.maximum_thickness_um:
            # Declining growth phase
            excess_factor = (p.maximum_thickness_um - thickness_um) / (p.maximum_thickness_um - p.optimal_thickness_um)
            dthickness_dt = net_growth_rate * 2.0 * excess_factor
        else:
            # No growth beyond maximum
            dthickness_dt = -p.decay_rate * thickness_um * 0.1

        # Biomass density change (kg/m³/h)
        # Increases with growth, decreases with decay and dilution
        dbiomass_dt = net_growth_rate * biomass_density * 0.1 - p.decay_rate * biomass_density * inactive_fraction

        # Ensure reasonable bounds
        dthickness_dt = max(-5.0, min(dthickness_dt, 10.0))  # μm/h limits
        dbiomass_dt = max(-100, min(dbiomass_dt, 500))  # kg/m³/h limits

        return [dthickness_dt, dbiomass_dt]

    def simulate_growth(self, time_hours: np.ndarray, substrate_conc: float = 1.0,
                       initial_thickness_um: float = 0.1) -> dict:
        """Simulate biofilm growth over time"""

        # Initial conditions: [thickness_um, biomass_density_kg/m³]
        initial_state = [initial_thickness_um, 100.0]  # Start with thin biofilm

        # Solve ODE system
        solution = odeint(self.biofilm_growth_ode, initial_state, time_hours,
                         args=(substrate_conc,))

        thickness_um = solution[:, 0]
        biomass_density = solution[:, 1]

        # Calculate additional properties
        results = {
            'time_hours': time_hours,
            'thickness_um': thickness_um,
            'biomass_density_kg_m3': biomass_density,
            'mass_per_area_kg_m2': thickness_um * 1e-6 * biomass_density,
            'diffusion_efficiency': [self.calculate_diffusion_limitation(t, substrate_conc)
                                   for t in thickness_um],
            'inactive_fraction': [self.calculate_inactive_fraction(t, time_hours[i])
                                for i, t in enumerate(thickness_um)],
            'current_density_A_m2': [],
            'electrochemical_activity': []
        }

        # Calculate current density based on active biofilm fraction
        for i, thickness in enumerate(thickness_um):
            active_fraction = 1 - results['inactive_fraction'][i]
            diffusion_eff = results['diffusion_efficiency'][i]

            # Current density decreases with thickness beyond optimum
            if thickness <= self.params.optimal_thickness_um:
                thickness_factor = thickness / self.params.optimal_thickness_um
            else:
                # Exponential decay beyond optimal thickness
                excess = thickness - self.params.optimal_thickness_um
                thickness_factor = np.exp(-excess / (self.params.maximum_thickness_um - self.params.optimal_thickness_um))

            current_density = (self.params.max_current_density * thickness_factor *
                             active_fraction * diffusion_eff)
            results['current_density_A_m2'].append(current_density)
            results['electrochemical_activity'].append(thickness_factor * active_fraction * diffusion_eff)

        # Store history
        self.thickness_history = thickness_um
        self.current_history = results['current_density_A_m2']
        self.inactive_fraction_history = results['inactive_fraction']

        return results

class QCMSensorModel:
    """Physics-based QCM sensor model with Gaussian sensitivity distribution"""

    def __init__(self, qcm_params: QCMSensorParameters):
        self.params = qcm_params
        self.calibration_factor = 1.0
        self.baseline_frequency = qcm_params.fundamental_freq
        self.noise_level = qcm_params.frequency_stability

    def sauerbrey_frequency_shift(self, mass_per_area: float) -> float:
        """Basic Sauerbrey equation for rigid films"""
        p = self.params

        # Classical Sauerbrey equation: Δf = -2f₀²Δm/(A√(ρμ))
        # Simplified for mass per unit area
        freq_shift = -p.mass_sensitivity * mass_per_area * p.fundamental_freq

        return freq_shift

    def viscoelastic_correction(self, mass_per_area: float, thickness_um: float,
                              elastic_modulus: float) -> tuple[float, float]:
        """Viscoelastic correction for soft biofilm layers"""
        p = self.params

        # Calculate mechanical properties
        thickness_m = thickness_um * 1e-6

        # Viscoelastic parameters
        density_film = mass_per_area / thickness_m if thickness_m > 0 else 1000

        # Complex shear modulus (simplified model)
        G_storage = elastic_modulus * 0.1  # Storage modulus (elastic component)
        G_loss = elastic_modulus * 0.05    # Loss modulus (viscous component)

        # Load impedance
        Z_film = np.sqrt(density_film * (G_storage + 1j * G_loss))
        Z_quartz = np.sqrt(p.density_quartz * p.shear_modulus)

        # Frequency and dissipation shifts (simplified)
        freq_shift_viscoel = -p.fundamental_freq / (2 * Z_quartz) * np.real(Z_film) * thickness_m
        dissipation_shift = p.fundamental_freq / (2 * Z_quartz) * np.imag(Z_film) * thickness_m

        return freq_shift_viscoel, dissipation_shift

    def gaussian_sensitivity_distribution(self, r_normalized: float) -> float:
        """Gaussian mass sensitivity distribution across electrode"""
        p = self.params

        # Normalized radial position (0 = center, 1 = edge)
        sensitivity = np.exp(-(r_normalized / p.sensitivity_std)**2)

        # Minimum sensitivity at edges
        sensitivity = max(sensitivity, p.edge_factor)

        return sensitivity

    def measure_biofilm(self, biofilm_results: dict, temperature_c: float = 25.0,
                       add_noise: bool = True) -> dict:
        """Simulate QCM measurement of biofilm growth"""
        p = self.params

        measurements = {
            'time_hours': biofilm_results['time_hours'],
            'frequency_shift_Hz': [],
            'dissipation_shift': [],
            'q_factor': [],
            'mass_calculated_kg_m2': [],
            'thickness_qcm_um': [],
            'measurement_quality': []
        }

        for i, mass_per_area in enumerate(biofilm_results['mass_per_area_kg_m2']):
            thickness_um = biofilm_results['thickness_um'][i]

            # Temperature correction
            temp_correction = p.temperature_coeff * (temperature_c - 25.0) * p.fundamental_freq

            if p.viscosity_correction and thickness_um > 1.0:
                # Use viscoelastic model for thick biofilms
                biofilm_params = GeobacterBiofilmParameters()
                freq_shift, diss_shift = self.viscoelastic_correction(
                    mass_per_area, thickness_um, biofilm_params.young_modulus)
            else:
                # Use Sauerbrey equation for thin rigid films
                freq_shift = self.sauerbrey_frequency_shift(mass_per_area)
                diss_shift = 0.0

            # Add temperature correction
            freq_shift += temp_correction

            # Add measurement noise
            if add_noise:
                freq_shift += np.random.normal(0, p.frequency_stability)
                diss_shift += np.random.normal(0, p.frequency_stability * 0.1)

            # Calculate Q factor (decreases with viscous loading)
            q_factor = p.q_factor_liquid * (1 - diss_shift / p.fundamental_freq * 1000)
            q_factor = max(100, q_factor)  # Minimum Q factor

            # Back-calculate mass (what QCM would report)
            mass_calculated = -freq_shift / (p.mass_sensitivity * p.fundamental_freq)
            thickness_calculated = mass_calculated / biofilm_results['biomass_density_kg_m3'][i] * 1e6

            # Measurement quality indicator
            quality = min(1.0, q_factor / p.q_factor_liquid)

            measurements['frequency_shift_Hz'].append(freq_shift)
            measurements['dissipation_shift'].append(diss_shift)
            measurements['q_factor'].append(q_factor)
            measurements['mass_calculated_kg_m2'].append(mass_calculated)
            measurements['thickness_qcm_um'].append(thickness_calculated)
            measurements['measurement_quality'].append(quality)

        return measurements

class BiofilmQCMController:
    """Feedback controller using QCM measurements for biofilm optimization"""

    def __init__(self, biofilm_params: GeobacterBiofilmParameters):
        self.biofilm_params = biofilm_params
        self.target_thickness = biofilm_params.optimal_thickness_um
        self.thickness_tolerance = 2.0  # μm
        self.control_history = []

    def pid_control(self, measured_thickness: float, dt_hours: float,
                   kp: float = 0.1, ki: float = 0.01, kd: float = 0.05) -> dict:
        """PID controller for biofilm thickness"""

        error = self.target_thickness - measured_thickness

        # Initialize if first measurement
        if not hasattr(self, 'integral_error'):
            self.integral_error = 0.0
            self.previous_error = error

        # PID components
        proportional = kp * error
        self.integral_error += error * dt_hours
        integral = ki * self.integral_error
        derivative = kd * (error - self.previous_error) / dt_hours if dt_hours > 0 else 0.0

        # Control output (substrate addition rate modifier)
        substrate_control = proportional + integral + derivative
        substrate_control = np.clip(substrate_control, -0.5, 2.0)  # Reasonable limits

        # Additional control actions
        control_actions = {
            'substrate_modifier': 1.0 + substrate_control,
            'flow_rate_modifier': 1.0,
            'maintenance_needed': False,
            'error': error,
            'control_output': substrate_control
        }

        # Trigger maintenance if biofilm too thick
        if measured_thickness > self.biofilm_params.maximum_thickness_um * 0.9:
            control_actions['maintenance_needed'] = True
            control_actions['flow_rate_modifier'] = 2.0  # Increase shear

        # Update for next iteration
        self.previous_error = error
        self.control_history.append(control_actions.copy())

        return control_actions

def create_physics_visualization(biofilm_results: dict, qcm_measurements: dict) -> None:
    """Create comprehensive visualization of physics-accurate biofilm model"""

    plt.figure(figsize=(16, 12))

    # Panel A: Biofilm Thickness Evolution
    plt.subplot(3, 3, 1)
    plt.plot(biofilm_results['time_hours'], biofilm_results['thickness_um'], 'b-', linewidth=2, label='True Thickness')
    plt.plot(qcm_measurements['time_hours'], qcm_measurements['thickness_qcm_um'], 'r--', linewidth=2, label='QCM Measured')
    plt.axhline(20, color='g', linestyle=':', label='Optimal (20 μm)')
    plt.axhline(45, color='orange', linestyle=':', label='Maximum (45 μm)')
    plt.xlabel('Time (hours)')
    plt.ylabel('Biofilm Thickness (μm)')
    plt.title('A) Biofilm Thickness Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Panel B: Current Density vs Thickness
    plt.subplot(3, 3, 2)
    current_mA_cm2 = np.array(biofilm_results['current_density_A_m2']) * 1000 / 10000  # Convert to mA/cm²
    plt.plot(biofilm_results['thickness_um'], current_mA_cm2, 'purple', linewidth=2)
    plt.axvline(20, color='g', linestyle=':', label='Optimal')
    plt.xlabel('Biofilm Thickness (μm)')
    plt.ylabel('Current Density (mA/cm²)')
    plt.title('B) Current vs Thickness')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Panel C: QCM Frequency Response
    plt.subplot(3, 3, 3)
    plt.plot(qcm_measurements['time_hours'], qcm_measurements['frequency_shift_Hz'], 'red', linewidth=2)
    plt.xlabel('Time (hours)')
    plt.ylabel('Frequency Shift (Hz)')
    plt.title('C) QCM Frequency Response')
    plt.grid(True, alpha=0.3)

    # Panel D: Electrochemical Activity
    plt.subplot(3, 3, 4)
    plt.plot(biofilm_results['time_hours'], biofilm_results['electrochemical_activity'], 'green', linewidth=2)
    plt.xlabel('Time (hours)')
    plt.ylabel('Relative Activity')
    plt.title('D) Electrochemical Activity')
    plt.grid(True, alpha=0.3)

    # Panel E: Diffusion Efficiency
    plt.subplot(3, 3, 5)
    plt.plot(biofilm_results['time_hours'], biofilm_results['diffusion_efficiency'], 'brown', linewidth=2)
    plt.xlabel('Time (hours)')
    plt.ylabel('Diffusion Efficiency')
    plt.title('E) Mass Transfer Limitation')
    plt.grid(True, alpha=0.3)

    # Panel F: Inactive Cell Fraction
    plt.subplot(3, 3, 6)
    plt.plot(biofilm_results['time_hours'], biofilm_results['inactive_fraction'], 'orange', linewidth=2)
    plt.xlabel('Time (hours)')
    plt.ylabel('Inactive Fraction')
    plt.title('F) Cell Viability')
    plt.grid(True, alpha=0.3)

    # Panel G: QCM Quality Metrics
    plt.subplot(3, 3, 7)
    plt.plot(qcm_measurements['time_hours'], qcm_measurements['q_factor'], 'cyan', linewidth=2, label='Q Factor')
    plt.xlabel('Time (hours)')
    plt.ylabel('Q Factor')
    plt.title('G) QCM Sensor Quality')
    plt.grid(True, alpha=0.3)

    # Panel H: Mass Comparison
    plt.subplot(3, 3, 8)
    true_mass = np.array(biofilm_results['mass_per_area_kg_m2']) * 1000  # g/m²
    measured_mass = np.array(qcm_measurements['mass_calculated_kg_m2']) * 1000  # g/m²
    plt.plot(biofilm_results['time_hours'], true_mass, 'b-', linewidth=2, label='True Mass')
    plt.plot(qcm_measurements['time_hours'], measured_mass, 'r--', linewidth=2, label='QCM Mass')
    plt.xlabel('Time (hours)')
    plt.ylabel('Mass per Area (g/m²)')
    plt.title('H) Mass Measurement Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Panel I: Biomass Density
    plt.subplot(3, 3, 9)
    plt.plot(biofilm_results['time_hours'], biofilm_results['biomass_density_kg_m3'], 'magenta', linewidth=2)
    plt.xlabel('Time (hours)')
    plt.ylabel('Biomass Density (kg/m³)')
    plt.title('I) Biomass Density Evolution')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(get_figure_path('physics_accurate_biofilm_qcm_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Physics-accurate biofilm analysis plot saved to '{get_figure_path('physics_accurate_biofilm_qcm_analysis.png')}'")

def run_physics_simulation(duration_hours: int = 200, substrate_conc: float = 1.0):
    """Run complete physics-accurate biofilm simulation with QCM monitoring"""

    print("Starting Physics-Accurate Biofilm Simulation with QCM Feedback")
    print("=" * 70)

    # Initialize models
    biofilm_params = GeobacterBiofilmParameters()
    qcm_params = QCMSensorParameters()

    biofilm_model = PhysicsAccurateBiofilmModel(biofilm_params)
    qcm_sensor = QCMSensorModel(qcm_params)
    BiofilmQCMController(biofilm_params)

    # Time vector
    time_hours = np.linspace(0, duration_hours, duration_hours * 10)  # 0.1 hour resolution

    # Run biofilm growth simulation
    print("Simulating biofilm growth with physics-based limitations...")
    biofilm_results = biofilm_model.simulate_growth(time_hours, substrate_conc)

    # Simulate QCM measurements
    print("Simulating QCM sensor response...")
    qcm_measurements = qcm_sensor.measure_biofilm(biofilm_results)

    # Create visualization
    create_physics_visualization(biofilm_results, qcm_measurements)

    # Save data
    combined_data = {
        'biofilm_model': biofilm_results,
        'qcm_measurements': qcm_measurements,
        'parameters': {
            'biofilm_params': biofilm_params.__dict__,
            'qcm_params': qcm_params.__dict__
        },
        'metadata': {
            'simulation_type': 'Physics-Accurate Biofilm with QCM',
            'duration_hours': duration_hours,
            'substrate_concentration': substrate_conc,
            'literature_based': True
        }
    }

    # Save JSON
    with open(get_simulation_data_path('physics_biofilm_qcm_data.json'), 'w') as f:
        json.dump(combined_data, f, indent=2, default=str)

    # Save CSV
    df_data = {
        'time_hours': biofilm_results['time_hours'],
        'biofilm_thickness_um': biofilm_results['thickness_um'],
        'biomass_density_kg_m3': biofilm_results['biomass_density_kg_m3'],
        'current_density_A_m2': biofilm_results['current_density_A_m2'],
        'electrochemical_activity': biofilm_results['electrochemical_activity'],
        'diffusion_efficiency': biofilm_results['diffusion_efficiency'],
        'inactive_fraction': biofilm_results['inactive_fraction'],
        'qcm_frequency_shift_Hz': qcm_measurements['frequency_shift_Hz'],
        'qcm_thickness_um': qcm_measurements['thickness_qcm_um'],
        'qcm_mass_kg_m2': qcm_measurements['mass_calculated_kg_m2'],
        'qcm_q_factor': qcm_measurements['q_factor']
    }

    df = pd.DataFrame(df_data)
    df.to_csv(get_simulation_data_path('physics_biofilm_qcm_data.csv'), index=False)

    # Print summary
    final_thickness = biofilm_results['thickness_um'][-1]
    final_current = biofilm_results['current_density_A_m2'][-1] * 1000 / 10000  # mA/cm²
    final_activity = biofilm_results['electrochemical_activity'][-1]

    print("\nSIMULATION RESULTS SUMMARY:")
    print(f"Final biofilm thickness:     {final_thickness:.1f} μm")
    print(f"Optimal thickness:           {biofilm_params.optimal_thickness_um:.1f} μm")
    print(f"Final current density:       {final_current:.2f} mA/cm²")
    print(f"Final electrochemical activity: {final_activity:.3f}")
    print(f"QCM measurement accuracy:    {np.mean(qcm_measurements['measurement_quality']):.3f}")

    print("\nLiterature Validation:")
    print(f"Target current density:      {biofilm_params.max_current_density*1000/10000:.2f} mA/cm²")
    print(f"Achieved performance:        {final_current/(biofilm_params.max_current_density*1000/10000)*100:.1f}% of maximum")

    if final_thickness < biofilm_params.optimal_thickness_um * 1.1:
        print("✅ Biofilm thickness within optimal range")
    else:
        print("⚠️  Biofilm thickness exceeds optimal range")

    print("\nData saved to:")
    print(f"- {get_simulation_data_path('physics_biofilm_qcm_data.json')}")
    print(f"- {get_simulation_data_path('physics_biofilm_qcm_data.csv')}")
    print(f"- {get_figure_path('physics_accurate_biofilm_qcm_analysis.png')}")

if __name__ == "__main__":
    run_physics_simulation(duration_hours=200, substrate_conc=1.0)
