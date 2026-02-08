"""Tests for physics_accurate_biofilm_qcm.py - coverage target 98%+."""
import sys
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


@pytest.fixture(autouse=True)
def mock_deps():
    with patch("physics_accurate_biofilm_qcm.get_figure_path", return_value="/tmp/f.png"):
        with patch("physics_accurate_biofilm_qcm.get_simulation_data_path", return_value="/tmp/d.csv"):
            yield


class TestGeobacterBiofilmParameters:
    def test_defaults(self):
        from physics_accurate_biofilm_qcm import GeobacterBiofilmParameters
        p = GeobacterBiofilmParameters()
        assert p.optimal_thickness_um == 20.0
        assert p.density == 1100.0


class TestQCMSensorParameters:
    def test_defaults(self):
        from physics_accurate_biofilm_qcm import QCMSensorParameters
        p = QCMSensorParameters()
        assert p.fundamental_freq == 5e6


class TestPhysicsAccurateBiofilmModel:
    def test_diffusion_no_limitation(self):
        from physics_accurate_biofilm_qcm import GeobacterBiofilmParameters, PhysicsAccurateBiofilmModel
        m = PhysicsAccurateBiofilmModel(GeobacterBiofilmParameters())
        eta = m.calculate_diffusion_limitation(0.01, 1.0)
        assert eta == pytest.approx(1.0, abs=0.01)

    def test_diffusion_with_limitation(self):
        from physics_accurate_biofilm_qcm import GeobacterBiofilmParameters, PhysicsAccurateBiofilmModel
        m = PhysicsAccurateBiofilmModel(GeobacterBiofilmParameters())
        eta = m.calculate_diffusion_limitation(50.0, 1.0)
        assert 0 < eta <= 1.0

    def test_inactive_fraction_thin(self):
        from physics_accurate_biofilm_qcm import GeobacterBiofilmParameters, PhysicsAccurateBiofilmModel
        m = PhysicsAccurateBiofilmModel(GeobacterBiofilmParameters())
        f = m.calculate_inactive_fraction(5.0, 0.0)
        assert 0 <= f <= 0.8

    def test_inactive_fraction_thick(self):
        from physics_accurate_biofilm_qcm import GeobacterBiofilmParameters, PhysicsAccurateBiofilmModel
        m = PhysicsAccurateBiofilmModel(GeobacterBiofilmParameters())
        f = m.calculate_inactive_fraction(50.0, 100.0)
        assert 0 <= f <= 0.8

    def test_growth_ode_linear_phase(self):
        from physics_accurate_biofilm_qcm import GeobacterBiofilmParameters, PhysicsAccurateBiofilmModel
        m = PhysicsAccurateBiofilmModel(GeobacterBiofilmParameters())
        deriv = m.biofilm_growth_ode([5.0, 500.0], 0, 1.0)
        assert len(deriv) == 2

    def test_growth_ode_declining_phase(self):
        from physics_accurate_biofilm_qcm import GeobacterBiofilmParameters, PhysicsAccurateBiofilmModel
        m = PhysicsAccurateBiofilmModel(GeobacterBiofilmParameters())
        deriv = m.biofilm_growth_ode([30.0, 800.0], 50, 1.0)
        assert len(deriv) == 2

    def test_growth_ode_beyond_max(self):
        from physics_accurate_biofilm_qcm import GeobacterBiofilmParameters, PhysicsAccurateBiofilmModel
        m = PhysicsAccurateBiofilmModel(GeobacterBiofilmParameters())
        deriv = m.biofilm_growth_ode([50.0, 800.0], 100, 1.0)
        assert deriv[0] <= 0

    def test_simulate_growth(self):
        from physics_accurate_biofilm_qcm import GeobacterBiofilmParameters, PhysicsAccurateBiofilmModel
        m = PhysicsAccurateBiofilmModel(GeobacterBiofilmParameters())
        time_h = np.linspace(0, 10, 50)
        results = m.simulate_growth(time_h, substrate_conc=1.0)
        assert "thickness_um" in results
        assert "current_density_A_m2" in results
        assert len(results["thickness_um"]) == 50


class TestQCMSensorModel:
    def test_sauerbrey(self):
        from physics_accurate_biofilm_qcm import QCMSensorParameters, QCMSensorModel
        s = QCMSensorModel(QCMSensorParameters())
        df = s.sauerbrey_frequency_shift(1e-6)
        assert df < 0

    def test_viscoelastic_correction(self):
        from physics_accurate_biofilm_qcm import QCMSensorParameters, QCMSensorModel
        s = QCMSensorModel(QCMSensorParameters())
        df, dd = s.viscoelastic_correction(1e-6, 10.0, 50e3)
        assert isinstance(df, (float, np.floating))
        assert isinstance(dd, (float, np.floating))

    def test_gaussian_sensitivity(self):
        from physics_accurate_biofilm_qcm import QCMSensorParameters, QCMSensorModel
        s = QCMSensorModel(QCMSensorParameters())
        center = s.gaussian_sensitivity_distribution(0.0)
        edge = s.gaussian_sensitivity_distribution(1.0)
        assert center > edge

    def test_measure_biofilm(self):
        from physics_accurate_biofilm_qcm import (
            GeobacterBiofilmParameters, PhysicsAccurateBiofilmModel,
            QCMSensorParameters, QCMSensorModel,
        )
        np.random.seed(42)
        bm = PhysicsAccurateBiofilmModel(GeobacterBiofilmParameters())
        time_h = np.linspace(0, 5, 20)
        bio_results = bm.simulate_growth(time_h)
        qcm = QCMSensorModel(QCMSensorParameters())
        meas = qcm.measure_biofilm(bio_results)
        assert "frequency_shift_Hz" in meas
        assert len(meas["frequency_shift_Hz"]) == 20

    def test_measure_biofilm_no_noise(self):
        from physics_accurate_biofilm_qcm import (
            GeobacterBiofilmParameters, PhysicsAccurateBiofilmModel,
            QCMSensorParameters, QCMSensorModel,
        )
        bm = PhysicsAccurateBiofilmModel(GeobacterBiofilmParameters())
        time_h = np.linspace(0, 5, 20)
        bio_results = bm.simulate_growth(time_h)
        qcm = QCMSensorModel(QCMSensorParameters())
        meas = qcm.measure_biofilm(bio_results, add_noise=False)
        assert len(meas["thickness_qcm_um"]) == 20

    def test_measure_thin_film(self):
        from physics_accurate_biofilm_qcm import (
            GeobacterBiofilmParameters, PhysicsAccurateBiofilmModel,
            QCMSensorParameters, QCMSensorModel,
        )
        bm = PhysicsAccurateBiofilmModel(GeobacterBiofilmParameters())
        time_h = np.array([0.0, 0.1])
        bio_results = bm.simulate_growth(time_h, initial_thickness_um=0.1)
        qcm_params = QCMSensorParameters()
        qcm_params.viscosity_correction = False
        qcm = QCMSensorModel(qcm_params)
        meas = qcm.measure_biofilm(bio_results, add_noise=False)
        assert len(meas["frequency_shift_Hz"]) == 2


class TestBiofilmQCMController:
    def test_pid_first_step(self):
        from physics_accurate_biofilm_qcm import GeobacterBiofilmParameters, BiofilmQCMController
        c = BiofilmQCMController(GeobacterBiofilmParameters())
        result = c.pid_control(15.0, 1.0)
        assert "substrate_modifier" in result
        assert "error" in result

    def test_pid_consecutive_steps(self):
        from physics_accurate_biofilm_qcm import GeobacterBiofilmParameters, BiofilmQCMController
        c = BiofilmQCMController(GeobacterBiofilmParameters())
        c.pid_control(15.0, 1.0)
        result = c.pid_control(18.0, 1.0)
        assert len(c.control_history) == 2

    def test_pid_maintenance_needed(self):
        from physics_accurate_biofilm_qcm import GeobacterBiofilmParameters, BiofilmQCMController
        c = BiofilmQCMController(GeobacterBiofilmParameters())
        result = c.pid_control(42.0, 1.0)
        assert result["maintenance_needed"] is True
        assert result["flow_rate_modifier"] == 2.0

    def test_pid_zero_dt(self):
        from physics_accurate_biofilm_qcm import GeobacterBiofilmParameters, BiofilmQCMController
        c = BiofilmQCMController(GeobacterBiofilmParameters())
        c.pid_control(15.0, 1.0)
        result = c.pid_control(16.0, 0.0)
        assert "control_output" in result


class TestVisualization:
    def test_create_physics_visualization(self):
        with patch("physics_accurate_biofilm_qcm.plt") as mock_plt:
            mock_plt.figure.return_value = MagicMock()
            mock_plt.subplot.return_value = MagicMock()
            from physics_accurate_biofilm_qcm import create_physics_visualization
            n = 5
            bio = {
                "time_hours": np.linspace(0, 10, n),
                "thickness_um": np.ones(n) * 15,
                "biomass_density_kg_m3": np.ones(n) * 1000,
                "mass_per_area_kg_m2": np.ones(n) * 1e-3,
                "diffusion_efficiency": [0.9] * n,
                "inactive_fraction": [0.1] * n,
                "current_density_A_m2": [1e-4] * n,
                "electrochemical_activity": [0.8] * n,
            }
            qcm = {
                "time_hours": np.linspace(0, 10, n),
                "thickness_qcm_um": np.ones(n) * 14,
                "frequency_shift_Hz": np.ones(n) * -50,
                "q_factor": np.ones(n) * 900,
                "mass_calculated_kg_m2": np.ones(n) * 1e-3,
            }
            create_physics_visualization(bio, qcm)
            mock_plt.savefig.assert_called_once()


class TestRunPhysicsSimulation:
    def test_run_short(self):
        with patch("physics_accurate_biofilm_qcm.plt") as mock_plt:
            mock_plt.figure.return_value = MagicMock()
            mock_plt.subplot.return_value = MagicMock()
            with patch("builtins.open", MagicMock()):
                with patch("physics_accurate_biofilm_qcm.pd.DataFrame") as mock_df:
                    mock_df.return_value = MagicMock()
                    np.random.seed(42)
                    from physics_accurate_biofilm_qcm import run_physics_simulation
                    run_physics_simulation(duration_hours=5, substrate_conc=1.0)
