"""Tests for plotting_system.py - standardized MFC plotting utilities."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from unittest.mock import MagicMock, patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


class TestSubplotLabeler:
    def test_initial_labels(self):
        from plotting_system import SubplotLabeler
        labeler = SubplotLabeler()
        assert labeler.next_label() == "a"
        assert labeler.next_label() == "b"

    def test_reset(self):
        from plotting_system import SubplotLabeler
        labeler = SubplotLabeler()
        labeler.next_label()
        labeler.next_label()
        labeler.reset()
        assert labeler.next_label() == "a"

    def test_labels_beyond_z(self):
        from plotting_system import SubplotLabeler
        labeler = SubplotLabeler()
        for _ in range(26):
            labeler.next_label()
        assert labeler.next_label() == "aa"
        assert labeler.next_label() == "ab"

    def test_labels_beyond_zz(self):
        from plotting_system import SubplotLabeler
        labeler = SubplotLabeler()
        # Skip 26 + 26*26 = 702 labels to get past 'zz'
        for _ in range(26 + 26 * 26):
            labeler.next_label()
        label = labeler.next_label()
        assert label.startswith("subplot_")


class TestCreateLabeledSubplots:
    def test_single_subplot(self):
        from plotting_system import create_labeled_subplots
        fig, axes, labeler = create_labeled_subplots(1, 1)
        assert len(axes) == 1
        assert labeler.current_index == 1

    def test_multi_subplot(self):
        from plotting_system import create_labeled_subplots
        fig, axes, labeler = create_labeled_subplots(2, 3)
        assert len(axes) == 6
        assert labeler.current_index == 6

    def test_with_title(self):
        from plotting_system import create_labeled_subplots
        fig, axes, labeler = create_labeled_subplots(
            1, 2, figsize=(10, 5), title="Test"
        )
        assert fig._suptitle is not None


class TestAddSubplotLabel:
    def test_adds_label(self):
        from plotting_system import add_subplot_label
        fig, ax = plt.subplots()
        add_subplot_label(ax, "a")
        texts = ax.texts
        assert any("(a)" in t.get_text() for t in texts)

    def test_custom_fontsize(self):
        from plotting_system import add_subplot_label
        fig, ax = plt.subplots()
        add_subplot_label(ax, "b", fontsize=20, fontweight="normal")
        texts = ax.texts
        assert any("(b)" in t.get_text() for t in texts)


class TestSetupAxis:
    def test_basic_setup(self):
        from plotting_system import setup_axis
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4], label="test")
        setup_axis(ax, "X", "Y", "Title")
        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Y"
        assert ax.get_title() == "Title"

    def test_no_grid_no_legend(self):
        from plotting_system import setup_axis
        fig, ax = plt.subplots()
        setup_axis(ax, "X", "Y", "Title", grid=False, legend=False)

    def test_legend_with_no_handles(self):
        from plotting_system import setup_axis
        fig, ax = plt.subplots()
        setup_axis(ax, "X", "Y", "Title", legend=True)

    def test_legend_location(self):
        from plotting_system import setup_axis
        fig, ax = plt.subplots()
        ax.plot([1], [1], label="test")
        setup_axis(ax, "X", "Y", "T", legend_loc="upper left")


class TestPlotTimeSeries:
    def test_basic_plot(self):
        from plotting_system import plot_time_series
        fig, ax = plt.subplots()
        df = pd.DataFrame({"t": [0, 1, 2], "a": [1, 2, 3], "b": [3, 2, 1]})
        plot_time_series(ax, df, "t", ["a", "b"])

    def test_with_all_options(self):
        from plotting_system import plot_time_series
        fig, ax = plt.subplots()
        df = pd.DataFrame({"t": [0, 1], "a": [1, 2], "b": [3, 4]})
        plot_time_series(
            ax, df, "t", ["a", "b"],
            labels=["Series A", "Series B"],
            colors=["red", "blue"],
            linestyles=["-", "--"],
            linewidths=[2.0, 3.0],
        )

    def test_with_partial_options(self):
        from plotting_system import plot_time_series
        fig, ax = plt.subplots()
        df = pd.DataFrame({"t": [0, 1], "a": [1, 2], "b": [3, 4]})
        plot_time_series(
            ax, df, "t", ["a", "b"],
            colors=["red"],
            linestyles=["--"],
            linewidths=[1.0],
        )


class TestAddHorizontalLine:
    def test_adds_line(self):
        from plotting_system import add_horizontal_line
        fig, ax = plt.subplots()
        add_horizontal_line(ax, 5.0, "threshold")

    def test_custom_style(self):
        from plotting_system import add_horizontal_line
        fig, ax = plt.subplots()
        add_horizontal_line(ax, 3.0, "ref", color="green", linestyle=":", alpha=0.8)


class TestAddTextAnnotation:
    def test_default_annotation(self):
        from plotting_system import add_text_annotation
        fig, ax = plt.subplots()
        add_text_annotation(ax, "test annotation")
        assert len(ax.texts) == 1

    def test_custom_position(self):
        from plotting_system import add_text_annotation
        fig, ax = plt.subplots()
        add_text_annotation(
            ax, "custom", x=0.5, y=0.5, ha="center", va="center",
            boxstyle="square", facecolor="lightblue", alpha=0.8,
        )


class TestSaveFigure:
    def test_save_figure(self, tmp_path):
        from plotting_system import save_figure
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        path = str(tmp_path / "test.png")
        save_figure(fig, path, dpi=72)
        assert os.path.exists(path)


class TestPlotMfcSimulationResults:
    @pytest.fixture
    def csv_data(self, tmp_path):
        n = 100
        df = pd.DataFrame({
            "time_hours": np.linspace(0, 100, n),
            "reservoir_concentration": np.random.uniform(20, 30, n),
            "outlet_concentration": np.random.uniform(15, 25, n),
            "substrate_addition_rate": np.random.uniform(0, 5, n),
            "total_power": np.random.uniform(0.1, 0.5, n),
        })
        path = tmp_path / "sim.csv"
        df.to_csv(path, index=False)
        return str(path)

    @pytest.fixture
    def csv_data_with_cells(self, tmp_path):
        n = 100
        data = {
            "time_hours": np.linspace(0, 100, n),
            "reservoir_concentration": np.random.uniform(20, 30, n),
            "outlet_concentration": np.random.uniform(15, 25, n),
            "substrate_addition_rate": np.random.uniform(0, 5, n),
            "total_power": np.random.uniform(0.1, 0.5, n),
            "q_value": np.random.uniform(-1, 1, n),
            "epsilon": np.linspace(0.3, 0.01, n),
        }
        for i in range(5):
            data[f"power_cell_{i}"] = np.random.uniform(0.02, 0.1, n)
            data[f"biofilm_thickness_cell_{i}"] = np.linspace(1.0, 1.5, n)
            data[f"voltage_cell_{i}"] = np.random.uniform(0.5, 0.8, n)
            data[f"current_cell_{i}"] = np.random.uniform(0.1, 0.5, n)
        df = pd.DataFrame(data)
        path = tmp_path / "sim_cells.csv"
        df.to_csv(path, index=False)
        return str(path)

    def test_basic_results(self, csv_data, tmp_path):
        from plotting_system import plot_mfc_simulation_results
        with patch("plotting_system.get_figure_path", side_effect=lambda f: str(tmp_path / f)):
            ts = plot_mfc_simulation_results(csv_data, output_prefix="test")
        assert isinstance(ts, str)

    def test_with_cell_data(self, csv_data_with_cells, tmp_path):
        from plotting_system import plot_mfc_simulation_results
        with patch("plotting_system.get_figure_path", side_effect=lambda f: str(tmp_path / f)):
            ts = plot_mfc_simulation_results(csv_data_with_cells, output_prefix="test")
        assert isinstance(ts, str)

    def test_with_json_metadata(self, csv_data, tmp_path):
        import json
        json_path = csv_data.replace(".csv", ".json")
        with open(json_path, "w") as f:
            json.dump({"substrate_target_reservoir": 25.0, "substrate_max_threshold": 30.0}, f)
        from plotting_system import plot_mfc_simulation_results
        with patch("plotting_system.get_figure_path", side_effect=lambda f: str(tmp_path / f)):
            ts = plot_mfc_simulation_results(csv_data, output_prefix="test")
        assert isinstance(ts, str)

    def test_with_substrate_conc_and_qaction(self, tmp_path):
        """Test branches for substrate_conc_cell_* columns and q_action."""
        n = 100
        data = {
            "time_hours": np.linspace(0, 50, n),
            "reservoir_concentration": np.random.uniform(20, 30, n),
            "outlet_concentration": np.random.uniform(15, 25, n),
            "substrate_addition_rate": np.random.uniform(0, 5, n),
            "total_power": np.random.uniform(0.1, 0.5, n),
            "q_action": np.random.randint(0, 5, n),
        }
        for i in range(5):
            data[f"power_cell_{i}"] = np.random.uniform(0.02, 0.1, n)
            data[f"biofilm_thickness_cell_{i}"] = np.linspace(1.0, 1.5, n)
            data[f"substrate_conc_cell_{i}"] = np.random.uniform(10, 25, n)
        df = pd.DataFrame(data)
        path = tmp_path / "sim_full.csv"
        df.to_csv(path, index=False)

        from plotting_system import plot_mfc_simulation_results
        with patch("plotting_system.get_figure_path", side_effect=lambda f: str(tmp_path / f)):
            ts = plot_mfc_simulation_results(str(path), output_prefix="test")
        assert isinstance(ts, str)

    def test_long_duration_dynamics(self, tmp_path):
        """Test the long-term dynamics branch (duration > 100h)."""
        n = 200
        data = {
            "time_hours": np.linspace(0, 500, n),
            "reservoir_concentration": np.random.uniform(20, 30, n),
            "outlet_concentration": np.random.uniform(15, 25, n),
            "substrate_addition_rate": np.random.uniform(0, 5, n),
            "total_power": np.random.uniform(0.1, 0.5, n),
        }
        df = pd.DataFrame(data)
        path = tmp_path / "sim_long.csv"
        df.to_csv(path, index=False)

        from plotting_system import plot_mfc_simulation_results
        with patch("plotting_system.get_figure_path", side_effect=lambda f: str(tmp_path / f)):
            ts = plot_mfc_simulation_results(str(path), output_prefix="test")
        assert isinstance(ts, str)

    def test_long_duration_with_cells_and_qaction(self, tmp_path):
        """Test long duration with cell data, substrate_conc and q_action."""
        n = 200
        data = {
            "time_hours": np.linspace(0, 500, n),
            "reservoir_concentration": np.random.uniform(20, 30, n),
            "outlet_concentration": np.random.uniform(15, 25, n),
            "substrate_addition_rate": np.random.uniform(0, 5, n),
            "total_power": np.random.uniform(0.1, 0.5, n),
            "q_value": np.random.uniform(-1, 1, n),
            "epsilon": np.linspace(0.3, 0.01, n),
            "q_action": np.random.randint(0, 5, n),
        }
        for i in range(5):
            data[f"power_cell_{i}"] = np.random.uniform(0.02, 0.1, n)
            data[f"biofilm_thickness_cell_{i}"] = np.linspace(1.0, 1.5, n)
            data[f"substrate_conc_cell_{i}"] = np.random.uniform(10, 25, n)
        df = pd.DataFrame(data)
        path = tmp_path / "sim_long_full.csv"
        df.to_csv(path, index=False)

        from plotting_system import plot_mfc_simulation_results
        with patch("plotting_system.get_figure_path", side_effect=lambda f: str(tmp_path / f)):
            ts = plot_mfc_simulation_results(str(path), output_prefix="test")
        assert isinstance(ts, str)


class TestPlotLatestSimulation:
    def test_no_files_found(self, tmp_path):
        from plotting_system import plot_latest_simulation
        result = plot_latest_simulation(
            pattern="nonexistent_*.csv",
            data_dir=str(tmp_path),
        )
        assert result is None

    def test_finds_and_plots(self, tmp_path):
        n = 50
        df = pd.DataFrame({
            "time_hours": np.linspace(0, 50, n),
            "reservoir_concentration": np.random.uniform(20, 30, n),
            "outlet_concentration": np.random.uniform(15, 25, n),
            "substrate_addition_rate": np.random.uniform(0, 5, n),
            "total_power": np.random.uniform(0.1, 0.5, n),
        })
        csv_path = tmp_path / "mfc_recirculation_control_001.csv"
        df.to_csv(csv_path, index=False)

        from plotting_system import plot_latest_simulation
        with patch("plotting_system.get_figure_path", side_effect=lambda f: str(tmp_path / f)):
            ts = plot_latest_simulation(
                pattern="mfc_recirculation_control_*.csv",
                data_dir=str(tmp_path),
            )
        assert isinstance(ts, str)


class TestPlotGpuSimulationResults:
    def test_no_csv_files(self, tmp_path):
        from plotting_system import plot_gpu_simulation_results
        result = plot_gpu_simulation_results(str(tmp_path))
        assert result is None

    def test_with_data(self, tmp_path):
        import gzip
        import json as json_mod

        n = 500
        data = {
            "time_hours": np.linspace(0, 2400, n),
            "reservoir_concentration": np.random.uniform(20, 100, n),
            "outlet_concentration": np.random.uniform(15, 50, n),
            "substrate_addition_rate": np.random.uniform(0, 5, n),
            "total_power": np.random.uniform(0.001, 0.01, n),
            "q_action": np.random.randint(0, 5, n),
            "epsilon": np.linspace(0.3, 0.01, n),
            "reward": np.random.uniform(-10, 10, n),
        }
        df = pd.DataFrame(data)

        csv_gz_path = tmp_path / "gpu_sim.csv.gz"
        with gzip.open(csv_gz_path, "wt") as f:
            df.to_csv(f, index=False)

        metadata = {
            "performance_metrics": {
                "final_reservoir_concentration": 95.0,
                "mean_power": 0.005,
                "control_effectiveness_2mM": 0.02,
                "control_effectiveness_5mM": 0.05,
            },
            "simulation_info": {
                "total_runtime_hours": 2.5,
                "acceleration_backend": "JAX/GPU",
            },
        }
        json_path = tmp_path / "gpu_sim.json"
        with open(json_path, "w") as f:
            json_mod.dump(metadata, f)

        from plotting_system import plot_gpu_simulation_results
        with patch("plotting_system.get_figure_path", side_effect=lambda f: str(tmp_path / f)):
            ts = plot_gpu_simulation_results(str(tmp_path))
        assert isinstance(ts, str)

    def test_without_json(self, tmp_path):
        import gzip

        n = 500
        data = {
            "time_hours": np.linspace(0, 2400, n),
            "reservoir_concentration": np.random.uniform(20, 100, n),
            "outlet_concentration": np.random.uniform(15, 50, n),
            "substrate_addition_rate": np.random.uniform(0, 5, n),
            "total_power": np.random.uniform(0.001, 0.01, n),
            "q_action": np.random.randint(0, 5, n),
            "epsilon": np.linspace(0.3, 0.01, n),
            "reward": np.random.uniform(-10, 10, n),
        }
        df = pd.DataFrame(data)

        csv_gz_path = tmp_path / "gpu_sim.csv.gz"
        with gzip.open(csv_gz_path, "wt") as f:
            df.to_csv(f, index=False)

        from plotting_system import plot_gpu_simulation_results
        with patch("plotting_system.get_figure_path", side_effect=lambda f: str(tmp_path / f)):
            ts = plot_gpu_simulation_results(str(tmp_path))
        assert isinstance(ts, str)

    def test_with_substrate_consumption_metadata(self, tmp_path):
        import gzip
        import json as json_mod

        n = 500
        data = {
            "time_hours": np.linspace(0, 2400, n),
            "reservoir_concentration": np.random.uniform(20, 100, n),
            "outlet_concentration": np.random.uniform(15, 50, n),
            "substrate_addition_rate": np.random.uniform(0, 5, n),
            "total_power": np.random.uniform(0.001, 0.01, n),
            "q_action": np.random.randint(0, 5, n),
            "epsilon": np.linspace(0.3, 0.01, n),
            "reward": np.random.uniform(-10, 10, n),
        }
        df = pd.DataFrame(data)

        csv_gz_path = tmp_path / "gpu_sim.csv.gz"
        with gzip.open(csv_gz_path, "wt") as f:
            df.to_csv(f, index=False)

        metadata = {
            "performance_metrics": {},
            "simulation_info": {
                "total_runtime_hours": 1.0,
                "acceleration_backend": "CPU",
            },
            "substrate_consumption": {
                "daily_rate_mmol": 10.0,
            },
        }
        json_path = tmp_path / "gpu_sim.json"
        with open(json_path, "w") as f:
            json_mod.dump(metadata, f)

        from plotting_system import plot_gpu_simulation_results
        with patch("plotting_system.get_figure_path", side_effect=lambda f: str(tmp_path / f)):
            ts = plot_gpu_simulation_results(str(tmp_path))
        assert isinstance(ts, str)
