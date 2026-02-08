"""Tests for sensor_simulation_plotter.py."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


@pytest.fixture
def plotter():
    from sensor_simulation_plotter import SensorMFCPlotter
    return SensorMFCPlotter(timestamp="20250101_000000")


@pytest.fixture
def sample_data():
    n = 100
    return {
        "time_hours": np.linspace(0, 100, n).tolist(),
        "stack_power": np.random.uniform(0.001, 0.005, n).tolist(),
        "biofilm_thickness": np.linspace(0.1, 0.6, n).tolist(),
        "substrate_concentrations": np.linspace(20, 5, n).tolist(),
        "acetate_concentrations": np.linspace(0, 8, n).tolist(),
        "total_energy": 0.35,
        "average_power": 0.0035,
        "peak_power": 0.005,
        "coulombic_efficiency": 85.2,
        "fusion_accuracy": 92.5,
        "simulation_time": 0.2,
    }


@pytest.fixture
def empty_data():
    return {}


@pytest.fixture
def _mock_paths(tmp_path):
    with patch(
        "sensor_simulation_plotter.get_figure_path",
        side_effect=lambda f: str(tmp_path / f),
    ), patch(
        "sensor_simulation_plotter.get_simulation_data_path",
        side_effect=lambda f: str(tmp_path / f),
    ):
        yield tmp_path


class TestSensorMFCPlotterInit:
    def test_default_timestamp(self):
        from sensor_simulation_plotter import SensorMFCPlotter
        p = SensorMFCPlotter()
        assert p.timestamp is not None

    def test_custom_timestamp(self):
        from sensor_simulation_plotter import SensorMFCPlotter
        p = SensorMFCPlotter(timestamp="test123")
        assert p.timestamp == "test123"

    def test_colors(self, plotter):
        assert "primary" in plotter.colors
        assert len(plotter.cell_colors) == 5


class TestSetupPlotStyle:
    def test_sets_rcparams(self, plotter):
        assert plt.rcParams["font.size"] == 10


class TestDashboards:
    def test_comprehensive_dashboard(self, plotter, sample_data, _mock_paths):
        path = plotter.create_comprehensive_dashboard(sample_data)
        assert path.endswith(".png")

    def test_comprehensive_dashboard_empty(self, plotter, empty_data, _mock_paths):
        path = plotter.create_comprehensive_dashboard(empty_data)
        assert path.endswith(".png")

    def test_sensor_analysis_dashboard(self, plotter, sample_data, _mock_paths):
        path = plotter.create_sensor_analysis_dashboard(sample_data)
        assert path.endswith(".png")

    def test_sensor_analysis_empty(self, plotter, empty_data, _mock_paths):
        path = plotter.create_sensor_analysis_dashboard(empty_data)
        assert path.endswith(".png")

    def test_performance_summary(self, plotter, sample_data, _mock_paths):
        path = plotter.create_performance_summary(sample_data)
        assert path.endswith(".png")

    def test_performance_summary_empty(self, plotter, empty_data, _mock_paths):
        path = plotter.create_performance_summary(empty_data)
        assert path.endswith(".png")


class TestInternalPlotMethods:
    def test_power_evolution_with_data(self, plotter, sample_data):
        fig, ax = plt.subplots()
        plotter._plot_power_evolution(ax, sample_data)

    def test_power_evolution_empty(self, plotter, empty_data):
        fig, ax = plt.subplots()
        plotter._plot_power_evolution(ax, empty_data)

    def test_sensor_status_dashboard(self, plotter, sample_data):
        fig, ax = plt.subplots()
        plotter._plot_sensor_status_dashboard(ax, sample_data)

    def test_biofilm_sensor_validation(self, plotter, sample_data):
        fig, ax = plt.subplots()
        plotter._plot_biofilm_sensor_validation(ax, sample_data)

    def test_biofilm_sensor_validation_empty(self, plotter, empty_data):
        fig, ax = plt.subplots()
        plotter._plot_biofilm_sensor_validation(ax, empty_data)

    def test_current_voltage_eis(self, plotter, sample_data):
        fig, ax = plt.subplots()
        plotter._plot_current_voltage_eis(ax, sample_data)

    def test_qcm_mass_monitoring(self, plotter, sample_data):
        fig, ax = plt.subplots()
        plotter._plot_qcm_mass_monitoring(ax, sample_data)

    def test_sensor_fusion_performance(self, plotter, sample_data):
        fig, ax = plt.subplots()
        plotter._plot_sensor_fusion_performance(ax, sample_data)

    def test_system_performance_heatmap(self, plotter, sample_data):
        fig, ax = plt.subplots()
        plotter._plot_system_performance_heatmap(ax, sample_data)

    def test_eis_impedance_evolution(self, plotter, sample_data):
        fig, ax = plt.subplots()
        plotter._plot_eis_impedance_evolution(ax, sample_data)

    def test_eis_impedance_empty(self, plotter, empty_data):
        fig, ax = plt.subplots()
        plotter._plot_eis_impedance_evolution(ax, empty_data)

    def test_qcm_frequency_response(self, plotter, sample_data):
        fig, ax = plt.subplots()
        plotter._plot_qcm_frequency_response(ax, sample_data)

    def test_qcm_frequency_empty(self, plotter, empty_data):
        fig, ax = plt.subplots()
        plotter._plot_qcm_frequency_response(ax, empty_data)

    def test_sensor_fusion_confidence(self, plotter, sample_data):
        fig, ax = plt.subplots()
        plotter._plot_sensor_fusion_confidence(ax, sample_data)

    def test_biofilm_growth_validation(self, plotter, sample_data):
        fig, ax = plt.subplots()
        plotter._plot_biofilm_growth_validation(ax, sample_data)

    def test_biofilm_growth_validation_empty(self, plotter, empty_data):
        fig, ax = plt.subplots()
        plotter._plot_biofilm_growth_validation(ax, empty_data)

    def test_substrate_monitoring(self, plotter, sample_data):
        fig, ax = plt.subplots()
        plotter._plot_substrate_concentration_monitoring(ax, sample_data)

    def test_substrate_monitoring_empty(self, plotter, empty_data):
        fig, ax = plt.subplots()
        plotter._plot_substrate_concentration_monitoring(ax, empty_data)

    def test_substrate_only_substrate(self, plotter):
        data = {
            "time_hours": np.linspace(0, 100, 50).tolist(),
            "substrate_concentrations": np.linspace(20, 5, 50).tolist(),
        }
        fig, ax = plt.subplots()
        plotter._plot_substrate_concentration_monitoring(ax, data)

    def test_mass_transfer_correlation(self, plotter, sample_data):
        fig, ax = plt.subplots()
        plotter._plot_mass_transfer_correlation(ax, sample_data)

    def test_mass_transfer_empty(self, plotter, empty_data):
        fig, ax = plt.subplots()
        plotter._plot_mass_transfer_correlation(ax, empty_data)

    def test_sensor_quality_metrics(self, plotter, sample_data):
        fig, ax = plt.subplots()
        plotter._plot_sensor_quality_metrics(ax, sample_data)

    def test_sensor_quality_empty(self, plotter, empty_data):
        fig, ax = plt.subplots()
        plotter._plot_sensor_quality_metrics(ax, empty_data)

    def test_multicell_sensor_comparison(self, plotter, sample_data):
        fig, ax = plt.subplots()
        plotter._plot_multicell_sensor_comparison(ax, sample_data)

    def test_multicell_empty(self, plotter, empty_data):
        fig, ax = plt.subplots()
        plotter._plot_multicell_sensor_comparison(ax, empty_data)

    def test_key_performance_indicators(self, plotter, sample_data):
        fig, ax = plt.subplots()
        plotter._plot_key_performance_indicators(ax, sample_data)

    def test_kpi_empty(self, plotter, empty_data):
        fig, ax = plt.subplots()
        plotter._plot_key_performance_indicators(ax, empty_data)

    def test_efficiency_metrics(self, plotter, sample_data):
        fig, ax = plt.subplots()
        plotter._plot_efficiency_metrics(ax, sample_data)

    def test_efficiency_empty(self, plotter, empty_data):
        fig, ax = plt.subplots()
        plotter._plot_efficiency_metrics(ax, empty_data)

    def test_power_distribution(self, plotter, sample_data):
        fig, ax = plt.subplots()
        plotter._plot_power_distribution(ax, sample_data)

    def test_power_distribution_empty(self, plotter, empty_data):
        fig, ax = plt.subplots()
        plotter._plot_power_distribution(ax, empty_data)

    def test_cell_performance_comparison(self, plotter, sample_data):
        fig, ax = plt.subplots()
        plotter._plot_cell_performance_comparison(ax, sample_data)

    def test_cell_perf_empty(self, plotter, empty_data):
        fig, ax = plt.subplots()
        plotter._plot_cell_performance_comparison(ax, empty_data)

    def test_final_system_state(self, plotter, sample_data):
        fig, ax = plt.subplots()
        plotter._plot_final_system_state(ax, sample_data)

    def test_final_system_empty(self, plotter, empty_data):
        fig, ax = plt.subplots()
        plotter._plot_final_system_state(ax, empty_data)

    def test_simulation_info_box(self, plotter, sample_data):
        fig = plt.figure()
        plotter._add_simulation_info_box(fig, sample_data)

    def test_simulation_info_empty(self, plotter, empty_data):
        fig = plt.figure()
        plotter._add_simulation_info_box(fig, empty_data)


class TestHelperMethods:
    def test_consistent_biofilm_thickness(self, plotter):
        time = np.linspace(0, 100, 100)
        result = plotter._calculate_consistent_biofilm_thickness(time)
        assert len(result) == 100
        assert result[0] < result[-1]

    def test_estimate_thickness_from_eis(self, plotter):
        time = np.linspace(0, 100, 100)
        true_thickness = np.linspace(0.1, 0.6, 100)
        result = plotter._estimate_thickness_from_eis(time, true_thickness)
        assert len(result) == 100
        assert np.all(result >= 0.05)

    def test_estimate_thickness_from_qcm(self, plotter):
        time = np.linspace(0, 100, 100)
        true_thickness = np.linspace(0.1, 0.6, 100)
        result = plotter._estimate_thickness_from_qcm(time, true_thickness)
        assert len(result) == 100
        assert np.all(result >= 0.05)

    def test_generate_substrate_profile(self, plotter):
        time = np.linspace(0, 100, 200)
        result = plotter._generate_substrate_profile_with_additions(time)
        assert len(result) == 200

    def test_identify_substrate_additions(self, plotter):
        time = np.linspace(0, 100, 200)
        conc = np.linspace(20, 5, 200)
        # Create a spike
        conc[100] = 20.0
        result = plotter._identify_substrate_additions(time, conc)
        assert isinstance(result, list)

    def test_generate_acetate_profile(self, plotter):
        time = np.linspace(0, 100, 200)
        lactate = np.linspace(20, 5, 200)
        result = plotter._generate_acetate_accumulation_profile(time, lactate)
        assert len(result) == 200


class TestSaveSimulationData:
    def test_save_data(self, plotter, sample_data, _mock_paths):
        csv_path, json_path = plotter.save_simulation_data(sample_data)
        assert csv_path.endswith(".csv")
        assert json_path.endswith(".json")


class TestCreateAllSensorPlots:
    def test_creates_all_plots(self, sample_data, _mock_paths):
        from sensor_simulation_plotter import create_all_sensor_plots
        plots = create_all_sensor_plots(sample_data, timestamp="test_ts")
        assert "comprehensive_dashboard" in plots
        assert "sensor_analysis" in plots
        assert "performance_summary" in plots
        assert "csv_data" in plots
        assert "json_data" in plots
