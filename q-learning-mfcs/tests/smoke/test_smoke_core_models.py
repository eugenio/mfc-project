"""Smoke tests: verify core model classes can be instantiated.

Each test should complete in < 2 seconds. Tests minimal object creation
and basic method availability — no full simulation runs.
"""

import pytest


class TestBiofilmKineticsModel:
    """Smoke tests for BiofilmKineticsModel."""

    def test_instantiate_default(self):
        from biofilm_kinetics import BiofilmKineticsModel

        model = BiofilmKineticsModel()
        assert model is not None

    def test_instantiate_with_species(self):
        from biofilm_kinetics import BiofilmKineticsModel

        model = BiofilmKineticsModel(species="geobacter")
        assert model is not None

    def test_step_biofilm_dynamics(self):
        from biofilm_kinetics import BiofilmKineticsModel

        model = BiofilmKineticsModel()
        result = model.step_biofilm_dynamics(dt=1.0, anode_potential=-0.3)
        assert result is not None


class TestSpeciesParameters:
    """Smoke tests for SpeciesParameters."""

    def test_instantiate(self):
        from biofilm_kinetics import SpeciesParameters

        params = SpeciesParameters()
        assert params is not None


class TestMetabolicModel:
    """Smoke tests for MetabolicModel."""

    def test_instantiate_default(self):
        from metabolic_model import MetabolicModel

        model = MetabolicModel()
        assert model is not None

    def test_step_metabolism(self):
        from metabolic_model import MetabolicModel

        model = MetabolicModel()
        result = model.step_metabolism(
            dt=1.0,
            biomass=0.1,
            growth_rate=0.05,
            anode_potential=-0.3,
            substrate_supply=1.0,
            cathode_o2_conc=0.2,
            membrane_area=0.01,
            volume=0.001,
            electrode_area=0.005,
        )
        assert result is not None


class TestSimulationConfig:
    """Smoke tests for SimulationConfig."""

    def test_default_config(self):
        from run_simulation import SimulationConfig

        config = SimulationConfig()
        assert config.mode == "demo"
        assert config.n_cells == 5

    @pytest.mark.parametrize(
        "mode", ["demo", "100h", "1year", "gpu", "stack", "comprehensive"],
    )
    def test_from_mode(self, mode):
        from run_simulation import SimulationConfig

        config = SimulationConfig.from_mode(mode)
        assert config.mode == mode
        assert config.duration_hours > 0
        assert config.n_cells > 0
        assert config.time_step > 0


class TestUnifiedSimulationRunner:
    """Smoke tests for UnifiedSimulationRunner (no actual run)."""

    def test_instantiate(self):
        from run_simulation import SimulationConfig, UnifiedSimulationRunner

        config = SimulationConfig.from_mode("demo")
        runner = UnifiedSimulationRunner(config)
        assert runner is not None
        assert callable(runner.run)


class TestRecirculationControl:
    """Smoke tests for recirculation control classes."""

    def test_anolyte_reservoir(self):
        from mfc_recirculation_control import AnolytereservoirSystem

        reservoir = AnolytereservoirSystem()
        assert reservoir is not None

    def test_substrate_controller(self):
        from mfc_recirculation_control import SubstrateConcentrationController

        controller = SubstrateConcentrationController()
        assert controller is not None

    def test_qlearning_flow_controller(self):
        from mfc_recirculation_control import AdvancedQLearningFlowController

        controller = AdvancedQLearningFlowController()
        assert controller is not None
        assert callable(controller.choose_action)

    def test_mfc_cell_with_monitoring(self):
        from mfc_recirculation_control import MFCCellWithMonitoring

        cell = MFCCellWithMonitoring(cell_id=0)
        assert cell is not None


class TestGPUAcceleration:
    """Smoke tests for GPU acceleration module."""

    def test_get_accelerator(self):
        from gpu_acceleration import get_gpu_accelerator

        accel = get_gpu_accelerator()
        assert accel is not None


class TestPathConfig:
    """Smoke tests for path configuration."""

    def test_get_figure_path(self):
        from path_config import get_figure_path

        path = get_figure_path("test.png")
        assert isinstance(path, str)
        assert path.endswith("test.png")

    def test_get_simulation_data_path(self):
        from path_config import get_simulation_data_path

        path = get_simulation_data_path("test.json")
        assert isinstance(path, str)
        assert path.endswith("test.json")
