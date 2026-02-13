"""Extended coverage tests for mfc_gpu_accelerated module.

Covers remaining edge cases and branches not hit by existing test suite.
"""
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

import mfc_gpu_accelerated as mga

class TestDetectGPUBackendEdgeCases:
    """Additional tests for detect_gpu_backend."""

    def test_nvidia_smi_success(self):
        """Cover nvidia-smi success path (lines 24-32)."""
        mock_result = MagicMock(returncode=0)
        with patch(
            "mfc_gpu_accelerated.subprocess.run", return_value=mock_result
        ):
            backend, gpu_type = mga.detect_gpu_backend()
            assert backend == "cuda"
            assert gpu_type == "nvidia"

    def test_nvidia_fails_rocm_success(self):
        """Cover rocm-smi detection (lines 37-45)."""

        def side_effect(cmd, **kwargs):
            if cmd[0] == "nvidia-smi":
                return MagicMock(returncode=1)
            if cmd[0] == "rocm-smi":
                return MagicMock(returncode=0)
            return MagicMock(returncode=1)

        with patch(
            "mfc_gpu_accelerated.subprocess.run", side_effect=side_effect
        ):
            backend, gpu_type = mga.detect_gpu_backend()
            assert backend == "rocm"
            assert gpu_type == "amd"

    def test_lspci_amd_rdna(self):
        """Cover lspci RDNA detection (lines 50-56)."""

        def side_effect(cmd, **kwargs):
            if cmd[0] in ("nvidia-smi", "rocm-smi"):
                raise FileNotFoundError()
            if cmd[0] == "lspci":
                return MagicMock(
                    returncode=0, stdout="AMD RDNA 3 Graphics"
                )
            return MagicMock(returncode=1)

        with patch(
            "mfc_gpu_accelerated.subprocess.run", side_effect=side_effect
        ):
            backend, gpu_type = mga.detect_gpu_backend()
            assert backend == "rocm"
            assert gpu_type == "amd"

    def test_lspci_no_amd(self):
        """Cover lspci without AMD (falls through to cpu)."""

        def side_effect(cmd, **kwargs):
            if cmd[0] in ("nvidia-smi", "rocm-smi"):
                raise FileNotFoundError()
            if cmd[0] == "lspci":
                return MagicMock(
                    returncode=0, stdout="Intel HD Graphics"
                )
            return MagicMock(returncode=1)

        with patch(
            "mfc_gpu_accelerated.subprocess.run", side_effect=side_effect
        ):
            backend, gpu_type = mga.detect_gpu_backend()
            assert backend == "cpu"

    def test_all_commands_not_found(self):
        """Cover all commands raising FileNotFoundError."""
        with patch(
            "mfc_gpu_accelerated.subprocess.run",
            side_effect=FileNotFoundError(),
        ):
            backend, gpu_type = mga.detect_gpu_backend()
            assert backend == "cpu"
            assert gpu_type == "cpu"

    def test_rocm_smi_nonzero_return(self):
        """Cover rocm-smi returning nonzero."""

        def side_effect(cmd, **kwargs):
            if cmd[0] == "nvidia-smi":
                raise FileNotFoundError()
            if cmd[0] == "rocm-smi":
                return MagicMock(returncode=1)
            if cmd[0] == "lspci":
                raise FileNotFoundError()
            return MagicMock(returncode=1)

        with patch(
            "mfc_gpu_accelerated.subprocess.run", side_effect=side_effect
        ):
            backend, gpu_type = mga.detect_gpu_backend()
            assert backend == "cpu"

class TestSetupGPUBackendEdgeCases:
    """Additional tests for setup_gpu_backend."""

    def test_rocm_backend_fails(self):
        """Cover ROCm backend setup failure (lines 93-94)."""
        import builtins

        real_import = builtins.__import__

        def failing_import(name, *args, **kwargs):
            if name == "jax":
                raise ImportError("no jax")
            return real_import(name, *args, **kwargs)

        with patch.object(
            mga, "detect_gpu_backend", return_value=("rocm", "amd")
        ):
            with patch(
                "builtins.__import__", side_effect=failing_import
            ):
                result = mga.setup_gpu_backend()
                assert result[5] == "CPU (NumPy)"

    def test_cpu_jax_fails(self):
        """Cover CPU JAX fallback failure (lines 107-110)."""
        import builtins

        real_import = builtins.__import__

        def failing_import(name, *args, **kwargs):
            if name == "jax":
                raise ImportError("no jax")
            return real_import(name, *args, **kwargs)

        with patch.object(
            mga, "detect_gpu_backend", return_value=("cpu", "cpu")
        ):
            with patch(
                "builtins.__import__", side_effect=failing_import
            ):
                result = mga.setup_gpu_backend()
                assert result[5] == "CPU (NumPy)"
                assert result[6] is False

class TestGPUAcceleratedMFCEdgeCases:
    """Additional tests for GPUAcceleratedMFC."""

    def _make_mfc(self):
        config = MagicMock()
        config.n_cells = 3
        config.reservoir_volume_liters = 1.0
        config.anode_area_per_cell = 0.001
        config.cathode_area_per_cell = 0.001
        config.eis_sensor_area = 0.0001
        config.qcm_sensor_area = 0.0001
        config.enhanced_epsilon = 0.1
        config.initial_cell_concentration = 25.0
        config.initial_substrate_concentration = 25.0
        config.substrate_target_outlet = 20.0
        config.substrate_target_concentration = 25.0
        config.enhanced_learning_rate = 0.1
        config.enhanced_discount_factor = 0.95
        config.advanced_epsilon_decay = 0.999
        config.advanced_epsilon_min = 0.01
        config.reward_weights = MagicMock()
        config.reward_weights.substrate_penalty_multiplier = 1.0
        config.reward_weights.power_weight = 1.0
        with patch.object(
            mga.GPUAcceleratedMFC,
            "load_pretrained_qtable",
            return_value=np.zeros((100, 10)),
        ):
            with patch.object(
                mga.GPUAcceleratedMFC,
                "load_trained_epsilon",
                return_value=0.1,
            ):
                mfc = mga.GPUAcceleratedMFC(config)
        return mfc

    def test_update_biofilm_growth_tiny_thickness(self):
        """Cover minimum thickness clamping (line 370)."""
        mfc = self._make_mfc()
        t = np.array([0.05, 0.05, 0.05])  # Very thin
        s = np.array([1.0, 1.0, 1.0])  # Low substrate
        result = mfc.update_biofilm_growth(t, s, 0.1)
        assert all(r >= 0.1 for r in result)

    def test_update_biofilm_growth_large_thickness(self):
        """Cover max thickness clamping (line 369)."""
        mfc = self._make_mfc()
        t = np.array([199.0, 199.0, 199.0])  # Near max
        s = np.array([50.0, 50.0, 50.0])  # High substrate
        result = mfc.update_biofilm_growth(t, s, 10.0)
        assert all(r <= 200.0 for r in result)

    def test_update_biofilm_growth_zero_thiele(self):
        """Cover effectiveness = 1.0 when thiele_modulus < 1e-6 (line 350)."""
        mfc = self._make_mfc()
        # Very thin biofilm -> tiny thiele modulus
        t = np.array([1e-8, 1e-8, 1e-8])
        s = np.array([25.0, 25.0, 25.0])
        result = mfc.update_biofilm_growth(t, s, 0.1)
        assert result.shape == (3,)

    def test_calculate_power_output_thick_biofilm(self):
        """Cover thickness_factor capping at 1.0 (line 382)."""
        mfc = self._make_mfc()
        # Biofilm thicker than max_effective_thickness
        p = mfc.calculate_power_output(np.float64(150.0), np.float64(25.0))
        assert float(p) > 0

    def test_calculate_power_output_thin_biofilm(self):
        """Cover thickness_factor < 1.0 path."""
        mfc = self._make_mfc()
        p = mfc.calculate_power_output(np.float64(10.0), np.float64(25.0))
        assert float(p) > 0

    def test_calculate_power_output_low_substrate(self):
        """Cover low substrate limiting power."""
        mfc = self._make_mfc()
        p = mfc.calculate_power_output(np.float64(50.0), np.float64(0.1))
        assert float(p) >= 0

    def test_calculate_reward_near_target(self):
        """Cover stability_bonus = 50.0 when deviation < 2.0 (line 438)."""
        mfc = self._make_mfc()
        r = mfc.calculate_reward(25.0, 20.0, 0.5, target_conc=25.0)
        # deviation = 0 < 2.0, so stability_bonus = 50
        assert float(r) > 0

    def test_calculate_reward_far_from_target(self):
        """Cover stability_bonus = 0 when deviation >= 2.0 (line 438)."""
        mfc = self._make_mfc()
        r = mfc.calculate_reward(30.0, 20.0, 0.5, target_conc=25.0)
        # deviation = 5 >= 2.0, so stability_bonus = 0
        assert isinstance(float(r), float)

    def test_simulate_timestep_concentration_clamp_high(self):
        """Cover concentration clamping at 100.0 (line 523-526)."""
        mfc = self._make_mfc()
        mfc.reservoir_concentration = 99.0
        # Force substrate addition to be high
        mfc.q_learning_action_selection = lambda s, q, e, k=None: (9, None)
        result = mfc.simulate_timestep(0.1)
        assert mfc.reservoir_concentration <= 100.0

    def test_simulate_timestep_concentration_clamp_low(self):
        """Cover concentration clamping at 0.1 (line 523-526)."""
        mfc = self._make_mfc()
        mfc.reservoir_concentration = 0.5
        # Force negative substrate addition
        mfc.q_learning_action_selection = lambda s, q, e, k=None: (0, None)
        result = mfc.simulate_timestep(10.0)  # Large dt to force low
        assert mfc.reservoir_concentration >= 0.1

    def test_simulate_multiple_timesteps(self):
        """Cover epsilon decay and stability buffer updates."""
        mfc = self._make_mfc()
        initial_epsilon = mfc.epsilon
        for _ in range(10):
            result = mfc.simulate_timestep(0.1)
        assert mfc.epsilon < initial_epsilon
        assert mfc.stability_index == 10

    def test_run_simulation_progress_reporting(self):
        """Cover progress reporting in run_simulation (lines 644-647)."""
        mfc = self._make_mfc()
        results, metrics = mfc.run_simulation(1.0, "/tmp/test_out")
        assert len(results["time_hours"]) > 0
        assert "final_reservoir_concentration" in metrics

    def test_calculate_final_metrics_short_series(self):
        """Cover final metrics with minimal data."""
        mfc = self._make_mfc()
        results = {
            "reservoir_concentration": [25.0, 24.0, 26.0],
            "total_power": [0.5, 0.6, 0.4],
        }
        m = mfc.calculate_final_metrics(results)
        assert "final_reservoir_concentration" in m
        assert m["final_reservoir_concentration"] == 26.0

    def test_calculate_final_metrics_large_deviation(self):
        """Cover control effectiveness calculations."""
        mfc = self._make_mfc()
        # Create data with varying deviations from target=25
        concs = list(np.linspace(10, 40, 2000))
        powers = list(np.random.uniform(0, 1, 2000))
        results = {
            "reservoir_concentration": concs,
            "total_power": powers,
        }
        m = mfc.calculate_final_metrics(results)
        assert 0 <= m["control_effectiveness_2mM"] <= 100
        assert 0 <= m["control_effectiveness_5mM"] <= 100

    def test_cleanup_gpu_resources_no_jax(self):
        """Cover cleanup when JAX not available."""
        mfc = self._make_mfc()
        old_jax_available = mga.JAX_AVAILABLE
        mga.JAX_AVAILABLE = False
        try:
            mfc.cleanup_gpu_resources()  # Should not raise
        finally:
            mga.JAX_AVAILABLE = old_jax_available

    def test_load_pretrained_qtable_action_ge_70(self):
        """Cover action_id >= 70 skip (line 273)."""
        ckpt = {"q_table": {"(0, 0, 2, 0)": {"75": 1.5, "80": 2.0}}}
        with patch("mfc_gpu_accelerated.Path") as mp:
            mf = MagicMock()
            mf.stat.return_value.st_mtime = time.time()
            mp.return_value.glob.return_value = [mf]
            with patch(
                "builtins.open",
                mock_open(read_data=json.dumps(ckpt)),
            ):
                obj = mga.GPUAcceleratedMFC.__new__(mga.GPUAcceleratedMFC)
                result = obj.load_pretrained_qtable()
                assert result.shape == (100, 10)
                # Actions >= 70 should be skipped, so q_table stays zero
                assert np.max(np.abs(result)) == 0.0

    def test_load_trained_epsilon_missing_hyperparameters(self):
        """Cover missing hyperparameters key in checkpoint."""
        ckpt = {"some_other_key": {}}
        cfg = MagicMock()
        cfg.enhanced_epsilon = 0.2
        with patch("mfc_gpu_accelerated.Path") as mp:
            mf = MagicMock()
            mf.stat.return_value.st_mtime = time.time()
            mp.return_value.glob.return_value = [mf]
            with patch(
                "builtins.open",
                mock_open(read_data=json.dumps(ckpt)),
            ):
                obj = mga.GPUAcceleratedMFC.__new__(mga.GPUAcceleratedMFC)
                obj.config = cfg
                eps = obj.load_trained_epsilon()
                assert eps == 0.2  # Fallback to config default

class TestCalculateMaintenanceRequirementsEdgeCases:
    """Additional edge cases for calculate_maintenance_requirements."""

    def test_small_consumption(self):
        """Cover small consumption values."""
        r = mga.calculate_maintenance_requirements(10.0, 24.0)
        assert r["substrate"]["daily_consumption_mmol"] == 10.0
        assert r["buffer"]["total_consumed_mmol"] == 3.0

    def test_large_consumption(self):
        """Cover large consumption values."""
        r = mga.calculate_maintenance_requirements(50000.0, 8784.0)
        assert r["substrate"]["annual_refills_needed"] > 0
        assert r["buffer"]["annual_refills_needed"] > 0

class TestSignalHandlerEdgeCases:
    """Additional tests for signal_handler."""

    def test_signal_handler_raises_system_exit(self):
        """Cover signal_handler (line 831)."""
        with pytest.raises(SystemExit) as exc_info:
            mga.signal_handler(2, None)
        assert exc_info.value.code == 0

class TestRunGPUAcceleratedSimulationEdgeCases:
    """Additional tests for run_gpu_accelerated_simulation."""

    def test_device_attribute_missing(self, tmp_path):
        """Cover hasattr device check (line 792)."""
        mock_mfc = MagicMock(spec=[])  # No device attribute
        mock_mfc.total_substrate_added = 100.0
        mock_results = {
            "time_hours": [0.0],
            "reservoir_concentration": [25.0],
            "outlet_concentration": [24.0],
            "total_power": [0.5],
            "biofilm_thicknesses": [[1.0]],
            "substrate_addition_rate": [0.1],
            "q_action": [4],
            "epsilon": [0.1],
            "reward": [10.0],
        }
        mock_metrics = {
            "final_reservoir_concentration": 24.5,
            "mean_concentration": 24.75,
            "std_concentration": 0.25,
            "max_deviation": 0.5,
            "mean_deviation": 0.25,
            "final_power": 0.6,
            "mean_power": 0.55,
            "total_substrate_added": 100.0,
            "control_effectiveness_2mM": 100.0,
            "control_effectiveness_5mM": 100.0,
            "stability_coefficient_variation": 1.0,
        }
        mock_mfc.run_simulation = MagicMock(
            return_value=(mock_results, mock_metrics)
        )

        mock_output_dir = MagicMock()
        mock_output_dir.__truediv__ = (
            lambda self, other: tmp_path / other
        )

        with patch.object(
            mga, "GPUAcceleratedMFC", return_value=mock_mfc
        ):
            with patch.object(mga, "Path", return_value=mock_output_dir):
                with patch("builtins.open", mock_open()):
                    with patch.object(mga, "json") as mock_json:
                        with patch.dict(
                            "sys.modules",
                            {"pandas": MagicMock()},
                        ):
                            summary, out = (
                                mga.run_gpu_accelerated_simulation(1.0)
                            )
                            assert "simulation_info" in summary