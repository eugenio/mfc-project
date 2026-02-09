"""Tests for integrated_mfc_model.py - coverage target 98%+.

Covers IntegratedMFCState, IntegratedMFCModel (init, dynamics,
reward, simulation, results, save, plot, compatibility).
"""

import os
import sys
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


@pytest.fixture
def model():
    m = IntegratedMFCModel(
        n_cells=2,
        species="mixed",
        substrate="lactate",
        use_gpu=False,
        simulation_hours=5,
    )
    if not hasattr(m.flow_controller, "actions"):
        m.flow_controller.actions = np.array([-10, -5, -2, -1, 0, 1, 2, 5, 10])
    return m


from integrated_mfc_model import IntegratedMFCModel, IntegratedMFCState  # noqa: E402


class TestIntegratedMFCState:
    def test_dataclass(self):
        s = IntegratedMFCState(
            time=1.0,
            total_energy=0.5,
            average_power=0.05,
            coulombic_efficiency=0.7,
            biofilm_thickness=[10.0, 12.0],
            biomass_density=[0.5, 0.6],
            attachment_fraction=[0.5, 0.5],
            substrate_concentration=[1.0, 0.8],
            nadh_ratio=[0.3, 0.3],
            atp_level=[2.0, 2.0],
            electron_flux=[0.01, 0.01],
            cell_voltages=[0.35, 0.35],
            current_densities=[0.1, 0.1],
            anode_potentials=[-0.3, -0.3],
            reservoir_concentration=20.0,
            flow_rate=10.0,
            pump_power=0.001,
            epsilon=0.3,
            q_table_size=100,
            learning_progress=0.5,
        )
        assert s.time == 1.0
        assert len(s.biofilm_thickness) == 2
        assert s.epsilon == 0.3


class TestIntegratedMFCModelInit:
    def test_default(self, model):
        assert model.n_cells == 2
        assert model.species == "mixed"
        assert model.substrate == "lactate"
        assert len(model.biofilm_models) == 2
        assert len(model.metabolic_models) == 2

    def test_gpu_false(self, model):
        assert model.gpu_available is False

    def test_mfc_stack_compat(self, model):
        assert hasattr(model, "mfc_stack")
        assert model.mfc_stack.n_cells == 2
        assert model.mfc_stack.reservoir is model.reservoir
        assert model.mfc_stack.mfc_cells is model.mfc_cells

    def test_agent_compat(self, model):
        assert hasattr(model, "agent")
        assert model.agent.n_cells == 2

    def test_tracking_init(self, model):
        assert model.time == 0.0
        assert model.history == []
        assert model.total_energy_generated == 0.0


class TestStepDynamics:
    def test_single_step(self, model):
        state = model.step_integrated_dynamics(dt=1.0)
        assert isinstance(state, IntegratedMFCState)
        assert state.time > 0
        assert len(state.biofilm_thickness) == 2
        assert len(state.cell_voltages) == 2
        assert len(model.history) == 1

    def test_multiple_steps(self, model):
        for _ in range(3):
            model.step_integrated_dynamics(dt=1.0)
        assert len(model.history) == 3
        assert model.time == 3.0

    def test_energy_accumulates(self, model):
        model.step_integrated_dynamics(dt=1.0)
        assert model.total_energy_generated >= 0.0

    def test_flow_rate_changes(self, model):
        model.step_integrated_dynamics(dt=1.0)
        assert model.flow_rate_ml_h > 0


class TestCalculateReward:
    def test_basic(self, model):
        mfc_state = {"cell_voltages": [0.3, 0.3]}
        biofilm_states = [
            {"biofilm_thickness": 35.0, "biomass_density": 1.0},
            {"biofilm_thickness": 35.0, "biomass_density": 1.0},
        ]

        class FakeMS:
            coulombic_efficiency = 0.7
            metabolites = {"nadh": 0.3, "nad_plus": 0.7, "lactate": 5.0}

        metabolic_states = [FakeMS(), FakeMS()]
        enhanced_currents = [0.001, 0.001]
        r = model._calculate_integrated_reward(
            mfc_state,
            biofilm_states,
            metabolic_states,
            enhanced_currents,
        )
        assert isinstance(r, float)


class TestCompileResults:
    def test_empty_history(self, model):
        assert model._compile_results() == {}

    def test_with_history(self, model):
        model.step_integrated_dynamics(dt=1.0)
        r = model._compile_results()
        assert "total_energy" in r
        assert "average_power" in r
        assert "time_series" in r
        assert "configuration" in r


class TestRunSimulation:
    def test_short_run(self, model):
        model.simulation_hours = 2
        results = model.run_simulation(dt=1.0, save_interval=100)
        assert "computation_time" in results
        assert "total_energy" in results

    def test_checkpoint(self, model):
        model.simulation_hours = 2
        with patch.object(model, "_save_checkpoint") as mock_save:
            model.run_simulation(dt=1.0, save_interval=1)
            assert mock_save.called


class TestSaveCheckpoint:
    def test_save(self, model):
        model.step_integrated_dynamics(dt=1.0)
        with patch("builtins.open", mock_open()), patch("pickle.dump") as mock_dump:
            model._save_checkpoint(1)
            mock_dump.assert_called_once()


class TestSaveResults:
    def test_save(self, model):
        model.step_integrated_dynamics(dt=1.0)
        results = model._compile_results()
        with (
            patch("builtins.open", mock_open()),
            patch("json.dump") as mock_json,
            patch("pickle.dump") as mock_pkl,
        ):
            mock_df = MagicMock()
            with patch("pandas.DataFrame", return_value=mock_df):
                model.save_results(results)
            mock_json.assert_called_once()
            mock_pkl.assert_called_once()


class _FakeAxes:
    """2D indexable axes container mimicking matplotlib subplots."""

    def __init__(self, rows, cols):
        self._data = [[MagicMock() for _ in range(cols)] for _ in range(rows)]

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self._data[key[0]][key[1]]
        return self._data[key]


class TestPlotResults:
    def _make_mock_plt(self):
        """Create a mock plt with subplots returning (fig, axes) tuple."""
        mock_plt = MagicMock()
        mock_fig = MagicMock()
        mock_axes = _FakeAxes(2, 2)
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        return mock_plt

    def test_plot(self, model):
        for _ in range(3):
            model.step_integrated_dynamics(dt=1.0)
        results = model._compile_results()
        mock_plt = self._make_mock_plt()
        mock_mpl = MagicMock()
        mock_mpl.pyplot = mock_plt
        with patch.dict(
            "sys.modules",
            {"matplotlib": mock_mpl, "matplotlib.pyplot": mock_plt},
        ):
            model.plot_results(results, save_plots=True)

    def test_plot_no_save(self, model):
        for _ in range(3):
            model.step_integrated_dynamics(dt=1.0)
        results = model._compile_results()
        mock_plt = self._make_mock_plt()
        mock_mpl = MagicMock()
        mock_mpl.pyplot = mock_plt
        with patch.dict(
            "sys.modules",
            {"matplotlib": mock_mpl, "matplotlib.pyplot": mock_plt},
        ):
            model.plot_results(results, save_plots=False)


class TestMain:
    def test_main_default(self):
        """Cover main() with default args (no --plot)."""
        from integrated_mfc_model import main

        with (
            patch("sys.argv", ["prog", "--cells", "2", "--hours", "2"]),
            patch("integrated_mfc_model.IntegratedMFCModel") as mock_model_cls,
        ):
            mock_instance = MagicMock()
            mock_instance.run_simulation.return_value = {"total_energy": 1.0}
            mock_model_cls.return_value = mock_instance

            main()

            mock_model_cls.assert_called_once_with(
                n_cells=2,
                species="mixed",
                substrate="lactate",
                use_gpu=False,
                simulation_hours=2,
            )
            mock_instance.run_simulation.assert_called_once()
            mock_instance.save_results.assert_called_once()
            mock_instance.plot_results.assert_not_called()

    def test_main_with_plot(self):
        """Cover main() with --plot flag."""
        from integrated_mfc_model import main

        with (
            patch(
                "sys.argv",
                [
                    "prog",
                    "--cells",
                    "1",
                    "--hours",
                    "1",
                    "--species",
                    "geobacter",
                    "--substrate",
                    "acetate",
                    "--gpu",
                    "--plot",
                ],
            ),
            patch("integrated_mfc_model.IntegratedMFCModel") as mock_model_cls,
        ):
            mock_instance = MagicMock()
            mock_instance.run_simulation.return_value = {"total_energy": 0.5}
            mock_model_cls.return_value = mock_instance

            main()

            mock_model_cls.assert_called_once_with(
                n_cells=1,
                species="geobacter",
                substrate="acetate",
                use_gpu=True,
                simulation_hours=1,
            )
            mock_instance.plot_results.assert_called_once()
