"""Smoke tests: verify configuration modules load and validate.

Each test should complete in < 1 second. Tests that default configs
load correctly and validation passes.
"""



class TestElectrodeConfig:
    """Smoke tests for electrode configuration."""

    def test_import_and_defaults(self):
        from config.electrode_config import (
            DEFAULT_CARBON_FELT_CONFIG,
            ElectrodeConfiguration,
        )

        assert ElectrodeConfiguration is not None
        assert DEFAULT_CARBON_FELT_CONFIG is not None


class TestMembraneConfig:
    """Smoke tests for membrane configuration."""

    def test_import_and_defaults(self):
        from config.membrane_config import (
            MEMBRANE_PROPERTIES_DATABASE,
            MembraneConfiguration,
        )

        assert MembraneConfiguration is not None
        assert len(MEMBRANE_PROPERTIES_DATABASE) > 0


class TestSensorConfig:
    """Smoke tests for sensor configuration."""

    def test_import_and_defaults(self):
        from config.sensor_config import SensorConfig

        cfg = SensorConfig()
        assert cfg is not None


class TestQLearningConfig:
    """Smoke tests for Q-learning configuration."""

    def test_import_and_defaults(self):
        from config.qlearning_config import QLearningConfig

        cfg = QLearningConfig()
        assert cfg is not None
        assert hasattr(cfg, "learning_rate") or hasattr(cfg, "alpha")


class TestSubstrateConfig:
    """Smoke tests for substrate configuration."""

    def test_import_and_defaults(self):
        from config.substrate_config import DEFAULT_SUBSTRATE_CONFIGS

        assert DEFAULT_SUBSTRATE_CONFIGS is not None
        assert len(DEFAULT_SUBSTRATE_CONFIGS) > 0


class TestBiologicalConfig:
    """Smoke tests for biological configuration."""

    def test_import_and_defaults(self):
        from config.biological_config import BiofilmKineticsConfig

        cfg = BiofilmKineticsConfig()
        assert cfg is not None


class TestControlConfig:
    """Smoke tests for control configuration."""

    def test_import_and_defaults(self):
        from config.control_config import ControlSystemConfig

        cfg = ControlSystemConfig()
        assert cfg is not None


class TestConfigManager:
    """Smoke tests for config manager."""

    def test_import(self):
        from config.config_manager import ConfigManager

        assert ConfigManager is not None

    def test_instantiate(self):
        from config.config_manager import ConfigManager

        mgr = ConfigManager()
        assert mgr is not None


class TestUnitConverter:
    """Smoke tests for unit converter."""

    def test_import(self):
        from config.unit_converter import UnitConverter

        assert UnitConverter is not None
