"""End-to-end tests for the unified simulation runner.

Tests the complete simulation workflow from configuration through
execution to results validation. Uses demo mode for speed.
"""

import os

import pytest


@pytest.fixture
def output_dir(tmp_path):
    """Provide a temporary output directory."""
    d = tmp_path / "sim_output"
    d.mkdir()
    yield str(d)


class TestDemoSimulationE2E:
    """E2E test: full demo simulation workflow."""

    def test_demo_completes_successfully(self, output_dir):
        from run_simulation import SimulationConfig, UnifiedSimulationRunner

        config = SimulationConfig.from_mode("demo", output_dir=output_dir)
        runner = UnifiedSimulationRunner(config)
        results = runner.run()

        assert results["success"] is True
        assert "metadata" in results
        assert results["metadata"]["mode"] == "demo"
        assert results["metadata"]["execution_time"] > 0
        assert results["metadata"]["interrupted"] is False

    def test_demo_produces_time_series(self, output_dir):
        from run_simulation import SimulationConfig, UnifiedSimulationRunner

        config = SimulationConfig.from_mode("demo", output_dir=output_dir)
        runner = UnifiedSimulationRunner(config)
        results = runner.run()

        assert "time_series" in results
        ts = results["time_series"]
        assert "time" in ts
        assert "voltage" in ts
        assert "power" in ts
        assert len(ts["time"]) > 0
        assert len(ts["voltage"]) == len(ts["time"])
        assert len(ts["power"]) == len(ts["time"])

    def test_demo_produces_energy_metrics(self, output_dir):
        from run_simulation import SimulationConfig, UnifiedSimulationRunner

        config = SimulationConfig.from_mode("demo", output_dir=output_dir)
        runner = UnifiedSimulationRunner(config)
        results = runner.run()

        assert "total_energy" in results
        assert "average_power" in results
        assert "final_power" in results
        assert results["total_energy"] >= 0
        assert results["average_power"] > 0

    def test_demo_saves_results_file(self, output_dir):
        from run_simulation import SimulationConfig, UnifiedSimulationRunner

        config = SimulationConfig.from_mode("demo", output_dir=output_dir)
        runner = UnifiedSimulationRunner(config)
        runner.run()

        # Check that output directory has files
        assert os.path.isdir(output_dir)

    def test_demo_voltage_in_physical_range(self, output_dir):
        from run_simulation import SimulationConfig, UnifiedSimulationRunner

        config = SimulationConfig.from_mode("demo", output_dir=output_dir)
        runner = UnifiedSimulationRunner(config)
        results = runner.run()

        voltages = results["time_series"]["voltage"]
        for v in voltages:
            assert 0.0 < v <= 1.5, f"Voltage {v} outside physical range"


class TestConfigOverridesE2E:
    """E2E test: configuration overrides work correctly."""

    def test_custom_cell_count(self, output_dir):
        from run_simulation import SimulationConfig, UnifiedSimulationRunner

        config = SimulationConfig.from_mode(
            "demo", n_cells=2, output_dir=output_dir,
        )
        assert config.n_cells == 2
        runner = UnifiedSimulationRunner(config)
        results = runner.run()
        assert results["success"] is True

    def test_quiet_mode(self, output_dir):
        from run_simulation import SimulationConfig, UnifiedSimulationRunner

        config = SimulationConfig.from_mode(
            "demo", verbose=False, output_dir=output_dir,
        )
        runner = UnifiedSimulationRunner(config)
        results = runner.run()
        assert results["success"] is True

    def test_invalid_mode_falls_back_to_demo(self, output_dir):
        from run_simulation import SimulationConfig, UnifiedSimulationRunner

        config = SimulationConfig.from_mode(
            "nonexistent", output_dir=output_dir,
        )
        runner = UnifiedSimulationRunner(config)
        results = runner.run()
        assert results["success"] is True
