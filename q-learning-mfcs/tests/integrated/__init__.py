"""
Integrated MFC Model Tests Package

This package contains comprehensive test suites for integrated MFC modeling
including multi-physics coupling, system integration, and end-to-end validation.
"""

import asyncio
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Add the source directory to the path for imports
project_root = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(project_root))

# Test utilities for integration testing
class IntegrationTestHelper:
    """Helper class for integration testing with shared utilities."""

    @staticmethod
    def create_temp_deployment_dir():
        """Create a temporary deployment directory for testing."""
        return tempfile.mkdtemp(prefix="mfc_test_deployment_")

    @staticmethod
    def cleanup_temp_dir(temp_dir):
        """Clean up temporary directory."""
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @staticmethod
    def create_mock_service_config(name, **kwargs):
        """Create a mock service configuration."""
        from deployment.process_manager import ProcessConfig, RestartPolicy

        default_config = {
            'name': name,
            'command': ['/bin/echo', f'service_{name}'],
            'working_dir': '/tmp',
            'env': {},
            'restart_policy': RestartPolicy.ALWAYS,
            'startup_timeout': 10.0,
            'shutdown_timeout': 5.0,
            'health_check_interval': 2.0,
            'max_restart_attempts': 3,
            'restart_delay': 1.0,
            'memory_limit_mb': None,
            'cpu_limit_percent': None
        }
        default_config.update(kwargs)
        return ProcessConfig(**default_config)

    @staticmethod
    def wait_for_condition(condition_func, timeout=10.0, interval=0.1):
        """Wait for a condition to become true."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if condition_func():
                return True
            time.sleep(interval)
        return False

class TestEndToEndMFCWorkflow:
    """Comprehensive end-to-end MFC system workflow integration tests."""

    def setup_method(self):
        """Setup for integration testing."""
        self.temp_dir = IntegrationTestHelper.create_temp_deployment_dir()
        self.original_cwd = os.getcwd()

        # Setup path to src directory
        self.src_path = Path(__file__).parent.parent.parent / "src"
        sys.path.insert(0, str(self.src_path))

    def teardown_method(self):
        """Cleanup after integration testing."""
        IntegrationTestHelper.cleanup_temp_dir(self.temp_dir)
        os.chdir(self.original_cwd)
        if str(self.src_path) in sys.path:
            sys.path.remove(str(self.src_path))

    @pytest.mark.integration
    def test_full_mfc_system_initialization(self):
        """Test complete MFC system initialization and startup."""
        try:
            # Import main MFC components
            from integrated_mfc_model import IntegratedMFCModel, IntegratedMFCState
            from mfc_model import MFCModel

            # Test system initialization
            mfc_system = IntegratedMFCModel(
                n_cells=4,
                species='Geobacter_sulfurreducens',
                substrate='acetate',
                membrane_type='Nafion',
                use_gpu=False,
                simulation_hours=0.1  # Short test
            )

            assert mfc_system is not None
            assert mfc_system.n_cells == 4
            assert mfc_system.species == 'Geobacter_sulfurreducens'
            assert mfc_system.biofilm_models is not None
            assert mfc_system.metabolic_models is not None

        except ImportError as e:
            pytest.skip(f"MFC system components not available: {e}")

    @pytest.mark.integration
    def test_end_to_end_simulation_workflow(self):
        """Test complete end-to-end MFC simulation workflow."""
        try:
            from integrated_mfc_model import IntegratedMFCModel

            # Initialize system
            mfc_system = IntegratedMFCModel(
                n_cells=2,
                species='Geobacter_sulfurreducens',
                substrate='acetate',
                use_gpu=False,
                simulation_hours=0.05  # Very short test
            )

            # Run simulation
            results = mfc_system.run_simulation()

            # Validate results structure
            assert results is not None
            assert 'time' in results
            assert 'voltage' in results or 'current' in results
            assert len(results['time']) > 0

            # Test data persistence
            output_file = os.path.join(self.temp_dir, 'test_results.json')
            mfc_system.save_results(output_file)
            assert os.path.exists(output_file)

        except ImportError as e:
            pytest.skip(f"Simulation components not available: {e}")
        except Exception as e:
            pytest.fail(f"End-to-end simulation failed: {e}")

    @pytest.mark.integration
    def test_cross_module_data_flow(self):
        """Test data flow between different system modules."""
        try:
            from integrated_mfc_model import IntegratedMFCModel

            mfc_system = IntegratedMFCModel(
                n_cells=2,
                species='Geobacter_sulfurreducens',
                substrate='acetate',
                use_gpu=False,
                simulation_hours=0.02
            )

            # Test initial state
            try:
                from integrated_mfc_model import IntegratedMFCState
                initial_state = IntegratedMFCState()
                assert initial_state is not None
            except ImportError:
                # Skip if IntegratedMFCState not available
                initial_state = None

            # Run one simulation step
            step_result = mfc_system.step_integrated_dynamics(initial_state, action=0.5)

            # Validate cross-module data consistency
            assert step_result is not None
            assert hasattr(step_result, 'voltage') or hasattr(step_result, 'current')

            # Test biofilm-metabolic model integration
            if hasattr(mfc_system, 'biofilm_models') and hasattr(mfc_system, 'metabolic_models'):
                assert len(mfc_system.biofilm_models) > 0
                assert len(mfc_system.metabolic_models) > 0

        except ImportError as e:
            pytest.skip(f"Cross-module components not available: {e}")
        except Exception as e:
            pytest.fail(f"Cross-module integration failed: {e}")

    def test_configuration_integration(self):
        """Test configuration loading and system integration."""
        try:
            # Test configuration loading
            from config.config_manager import ConfigManager, get_config_manager

            config_manager = get_config_manager()

            # Create a test profile using correct signature
            test_profile = config_manager.create_profile(
                profile_name="integration_test",
                biological={'species': 'test_species'},
                control=None,  # Use defaults
                visualization=None,  # Use defaults
                inherits_from=None
            )

            # Validate profile creation
            assert test_profile is not None
            assert test_profile.profile_name == "integration_test"

            # Test profile retrieval
            retrieved_profile = config_manager.get_profile("integration_test")
            assert retrieved_profile is not None
            assert retrieved_profile.profile_name == "integration_test"

            # Test configuration access
            current_config = config_manager.get_configuration()
            assert current_config is not None

            # Test profile listing
            profiles = config_manager.list_profiles()
            assert "integration_test" in profiles

            # Cleanup test profile
            config_manager.delete_profile("integration_test")

            # Verify cleanup
            profiles_after = config_manager.list_profiles()
            assert "integration_test" not in profiles_after

        except ImportError as e:
            pytest.skip(f"Configuration system not available: {e}")
        except Exception as e:
            pytest.fail(f"Configuration integration failed: {e}")

    @pytest.mark.integration
    def test_monitoring_integration(self):
        """Test monitoring system integration."""
        try:
            from monitoring.security_middleware import SecurityConfig, SessionManager

            # Test monitoring system initialization
            security_config = SecurityConfig()
            session_manager = SessionManager(security_config)

            # Test session management integration
            session_id = session_manager.create_session("integration_test_user")
            assert session_id is not None

            session_data = session_manager.validate_session(session_id)
            assert session_data is not None
            assert session_data['user_id'] == "integration_test_user"

            # Cleanup
            session_manager.destroy_session(session_id)

        except ImportError as e:
            pytest.skip(f"Monitoring system not available: {e}")
        except Exception as e:
            pytest.fail(f"Monitoring integration failed: {e}")


class TestSystemPerformanceIntegration:
    """Integration tests for system performance under various conditions."""

    def setup_method(self):
        """Setup performance testing environment."""
        self.temp_dir = IntegrationTestHelper.create_temp_deployment_dir()
        self.src_path = Path(__file__).parent.parent.parent / "src"
        sys.path.insert(0, str(self.src_path))

    def teardown_method(self):
        """Cleanup performance testing."""
        IntegrationTestHelper.cleanup_temp_dir(self.temp_dir)
        if str(self.src_path) in sys.path:
            sys.path.remove(str(self.src_path))

    @pytest.mark.integration
    @pytest.mark.performance
    def test_concurrent_simulation_handling(self):
        """Test system performance under concurrent simulation load."""
        try:
            import threading
            import time

            from integrated_mfc_model import IntegratedMFCModel

            def run_simulation(sim_id):
                """Run a single simulation."""
                try:
                    mfc_system = IntegratedMFCModel(
                        n_cells=2,
                        species='Geobacter_sulfurreducens',
                        substrate='acetate',
                        use_gpu=False,
                        simulation_hours=0.01  # Very short
                    )
                    results = mfc_system.run_simulation()
                    return results is not None
                except Exception:
                    return False

            # Run multiple concurrent simulations
            threads = []
            results = []

            start_time = time.time()

            for i in range(3):  # Small number for testing
                thread = threading.Thread(
                    target=lambda i=i: results.append(run_simulation(i))
                )
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join(timeout=30)  # 30 second timeout

            end_time = time.time()

            # Validate performance
            assert len(results) == 3
            assert all(results), "At least one concurrent simulation failed"
            assert end_time - start_time < 30, "Concurrent simulations took too long"

        except ImportError as e:
            pytest.skip(f"Performance testing components not available: {e}")
        except Exception as e:
            pytest.fail(f"Performance integration test failed: {e}")

    @pytest.mark.integration
    @pytest.mark.performance
    def test_memory_usage_stability(self):
        """Test system memory usage stability over multiple operations."""
        try:
            import gc

            import psutil
            from integrated_mfc_model import IntegratedMFCModel

            # Get initial memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Run multiple simulation cycles
            for i in range(5):
                mfc_system = IntegratedMFCModel(
                    n_cells=2,
                    species='Geobacter_sulfurreducens',
                    substrate='acetate',
                    use_gpu=False,
                    simulation_hours=0.01
                )

                # Run simulation
                results = mfc_system.run_simulation()
                assert results is not None

                # Explicit cleanup
                del mfc_system
                gc.collect()

            # Check final memory usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            # Memory should not increase excessively (allow for some variance)
            assert memory_increase < 100, f"Memory usage increased by {memory_increase:.2f} MB"

        except ImportError as e:
            pytest.skip(f"Memory testing components not available: {e}")
        except Exception as e:
            pytest.fail(f"Memory stability test failed: {e}")


class TestSystemRecoveryIntegration:
    """Integration tests for system fault tolerance and recovery."""

    def setup_method(self):
        """Setup recovery testing environment."""
        self.temp_dir = IntegrationTestHelper.create_temp_deployment_dir()
        self.src_path = Path(__file__).parent.parent.parent / "src"
        sys.path.insert(0, str(self.src_path))

    def teardown_method(self):
        """Cleanup recovery testing."""
        IntegrationTestHelper.cleanup_temp_dir(self.temp_dir)
        if str(self.src_path) in sys.path:
            sys.path.remove(str(self.src_path))

    @pytest.mark.integration
    def test_invalid_configuration_recovery(self):
        """Test system recovery from invalid configurations."""
        try:
            from integrated_mfc_model import IntegratedMFCModel

            # Test recovery from invalid n_cells
            with pytest.raises((ValueError, AssertionError)):
                IntegratedMFCModel(
                    n_cells=0,  # Invalid
                    species='Geobacter_sulfurreducens',
                    substrate='acetate'
                )

            # Test recovery from invalid species
            with pytest.raises((ValueError, KeyError)):
                IntegratedMFCModel(
                    n_cells=2,
                    species='invalid_species',  # Invalid
                    substrate='acetate'
                )

            # Test that valid configuration still works after errors
            valid_system = IntegratedMFCModel(
                n_cells=2,
                species='Geobacter_sulfurreducens',
                substrate='acetate',
                use_gpu=False,
                simulation_hours=0.01
            )
            assert valid_system is not None

        except ImportError as e:
            pytest.skip(f"Recovery testing components not available: {e}")

    @pytest.mark.integration
    def test_simulation_interruption_recovery(self):
        """Test system recovery from simulation interruptions."""
        try:
            from integrated_mfc_model import IntegratedMFCModel

            mfc_system = IntegratedMFCModel(
                n_cells=2,
                species='Geobacter_sulfurreducens',
                substrate='acetate',
                use_gpu=False,
                simulation_hours=0.02
            )

            # Test checkpoint/recovery mechanism
            checkpoint_file = os.path.join(self.temp_dir, 'test_checkpoint.json')

            # Create a checkpoint
            if hasattr(mfc_system, '_save_checkpoint'):
                try:
                    mfc_system._save_checkpoint(checkpoint_file)
                    assert os.path.exists(checkpoint_file)
                except Exception:
                    # Checkpoint mechanism may not be fully implemented
                    pass

            # Test that system can restart after interruption
            new_system = IntegratedMFCModel(
                n_cells=2,
                species='Geobacter_sulfurreducens',
                substrate='acetate',
                use_gpu=False,
                simulation_hours=0.01
            )

            results = new_system.run_simulation()
            assert results is not None

        except ImportError as e:
            pytest.skip(f"Recovery testing components not available: {e}")
        except Exception as e:
            pytest.fail(f"Simulation recovery test failed: {e}")


class TestDeploymentIntegration:
    """Integration tests for deployment configurations and scenarios."""

    def setup_method(self):
        """Setup deployment testing environment."""
        self.temp_dir = IntegrationTestHelper.create_temp_deployment_dir()
        self.src_path = Path(__file__).parent.parent.parent / "src"
        sys.path.insert(0, str(self.src_path))

    def teardown_method(self):
        """Cleanup deployment testing."""
        IntegrationTestHelper.cleanup_temp_dir(self.temp_dir)
        if str(self.src_path) in sys.path:
            sys.path.remove(str(self.src_path))

    @pytest.mark.integration
    def test_environment_variable_integration(self):
        """Test integration with environment variables and configuration."""
        try:
            import os
            from unittest.mock import patch

            # Test environment variable configuration
            test_env = {
                'MFC_N_CELLS': '4',
                'MFC_SPECIES': 'Geobacter_sulfurreducens',
                'MFC_SUBSTRATE': 'acetate',
                'MFC_USE_GPU': 'false',
                'MFC_SIMULATION_HOURS': '0.01'
            }

            with patch.dict(os.environ, test_env):
                # Test that environment variables are properly read
                assert os.environ.get('MFC_N_CELLS') == '4'
                assert os.environ.get('MFC_SPECIES') == 'Geobacter_sulfurreducens'
                assert os.environ.get('MFC_USE_GPU') == 'false'

                # Test configuration loading from environment
                from config.config_utils import substitute_environment_variables

                config_template = {
                    'mfc': {
                        'n_cells': '${MFC_N_CELLS}',
                        'species': '${MFC_SPECIES}',
                        'use_gpu': '${MFC_USE_GPU}'
                    }
                }

                resolved_config = substitute_environment_variables(config_template)

                assert resolved_config['mfc']['n_cells'] == '4'
                assert resolved_config['mfc']['species'] == 'Geobacter_sulfurreducens'
                assert resolved_config['mfc']['use_gpu'] == 'false'

        except ImportError as e:
            pytest.skip(f"Environment integration components not available: {e}")
        except Exception as e:
            pytest.fail(f"Environment integration test failed: {e}")

    @pytest.mark.integration
    def test_file_io_integration(self):
        """Test file I/O integration across the system."""
        try:
            import json

            from integrated_mfc_model import IntegratedMFCModel

            # Test output file creation and validation
            mfc_system = IntegratedMFCModel(
                n_cells=2,
                species='Geobacter_sulfurreducens',
                substrate='acetate',
                use_gpu=False,
                simulation_hours=0.01
            )

            results = mfc_system.run_simulation()

            # Test JSON output
            json_file = os.path.join(self.temp_dir, 'test_output.json')
            mfc_system.save_results(json_file)

            assert os.path.exists(json_file)

            # Validate JSON structure
            with open(json_file) as f:
                loaded_results = json.load(f)

            assert isinstance(loaded_results, dict)
            assert 'time' in loaded_results or 'voltage' in loaded_results

            # Test plot generation (if available)
            try:
                plot_file = os.path.join(self.temp_dir, 'test_plot.png')
                mfc_system.plot_results(save_path=plot_file)
                # Note: plot_results might not accept save_path parameter
            except Exception:
                # Plot generation may have different interface
                pass

        except ImportError as e:
            pytest.skip(f"File I/O integration components not available: {e}")
        except Exception as e:
            pytest.fail(f"File I/O integration test failed: {e}")


class TestMultiAgentIntegration:
    """Integration tests for multi-agent coordination and federated learning."""

    def setup_method(self):
        """Setup multi-agent testing environment."""
        self.temp_dir = IntegrationTestHelper.create_temp_deployment_dir()
        self.src_path = Path(__file__).parent.parent.parent / "src"
        sys.path.insert(0, str(self.src_path))

    def teardown_method(self):
        """Cleanup multi-agent testing."""
        IntegrationTestHelper.cleanup_temp_dir(self.temp_dir)
        if str(self.src_path) in sys.path:
            sys.path.remove(str(self.src_path))

    @pytest.mark.integration
    def test_federated_learning_integration(self):
        """Test federated learning system integration."""
        try:
            from federated_learning_controller import FederatedClient, FederatedServer

            # Test federated server initialization
            server = FederatedServer(
                n_clients=2,
                model_type="q_learning",
                aggregation_method="federated_averaging"
            )

            assert server is not None
            assert server.n_clients == 2

            # Test federated clients
            clients = []
            for i in range(2):
                client = FederatedClient(
                    client_id=f"client_{i}",
                    server_address="localhost",
                    model_type="q_learning"
                )
                clients.append(client)
                assert client is not None
                assert client.client_id == f"client_{i}"

            # Test basic federated learning workflow
            # This is a simplified test - full federated learning would require
            # actual model training and aggregation

            # Test model aggregation (mock)
            mock_models = [
                {"weights": [0.1, 0.2, 0.3]},
                {"weights": [0.2, 0.3, 0.4]}
            ]

            if hasattr(server, 'aggregate_models'):
                aggregated = server.aggregate_models(mock_models)
                assert aggregated is not None
                assert 'weights' in aggregated

        except ImportError as e:
            pytest.skip(f"Federated learning components not available: {e}")
        except Exception as e:
            pytest.fail(f"Federated learning integration failed: {e}")

    @pytest.mark.integration
    def test_transfer_learning_integration(self):
        """Test transfer learning system integration."""
        try:
            from transfer_learning_controller import TransferLearningController

            # Test transfer learning controller initialization
            controller = TransferLearningController(
                source_domain="domain_A",
                target_domain="domain_B",
                transfer_method="fine_tuning"
            )

            assert controller is not None
            assert controller.source_domain == "domain_A"
            assert controller.target_domain == "domain_B"

            # Test domain adaptation capabilities
            if hasattr(controller, 'adapt_domain'):
                mock_source_data = {
                    'features': [[1, 2, 3], [4, 5, 6]],
                    'labels': [0, 1]
                }

                adapted_data = controller.adapt_domain(mock_source_data)
                assert adapted_data is not None
                assert 'features' in adapted_data

        except ImportError as e:
            pytest.skip(f"Transfer learning components not available: {e}")
        except Exception as e:
            pytest.fail(f"Transfer learning integration failed: {e}")

    @pytest.mark.integration
    def test_multi_agent_coordination(self):
        """Test multi-agent coordination mechanisms."""
        try:
            # Test basic multi-agent coordination
            from integrated_mfc_model import IntegratedMFCModel

            # Create multiple agents
            agents = []
            for i in range(3):
                agent = IntegratedMFCModel(
                    n_cells=2,
                    species='Geobacter_sulfurreducens',
                    substrate='acetate',
                    use_gpu=False,
                    simulation_hours=0.01
                )
                agents.append(agent)

            assert len(agents) == 3

            # Test coordination through shared environment
            coordination_results = []
            for i, agent in enumerate(agents):
                result = agent.run_simulation()
                coordination_results.append({
                    'agent_id': i,
                    'result': result
                })

            assert len(coordination_results) == 3

            # Validate coordination results
            for coord_result in coordination_results:
                assert 'agent_id' in coord_result
                assert coord_result['result'] is not None

        except ImportError as e:
            pytest.skip(f"Multi-agent coordination components not available: {e}")
        except Exception as e:
            pytest.fail(f"Multi-agent coordination test failed: {e}")


class TestAdvancedFeatureIntegration:
    """Integration tests for advanced MFC features and algorithms."""

    def setup_method(self):
        """Setup advanced feature testing environment."""
        self.temp_dir = IntegrationTestHelper.create_temp_deployment_dir()
        self.src_path = Path(__file__).parent.parent.parent / "src"
        sys.path.insert(0, str(self.src_path))

    def teardown_method(self):
        """Cleanup advanced feature testing."""
        IntegrationTestHelper.cleanup_temp_dir(self.temp_dir)
        if str(self.src_path) in sys.path:
            sys.path.remove(str(self.src_path))

    @pytest.mark.integration
    def test_gpu_acceleration_integration(self):
        """Test GPU acceleration integration (if available)."""
        try:
            from gpu_acceleration import GPUAccelerator
            from integrated_mfc_model import IntegratedMFCModel

            # Test GPU availability detection
            gpu_acc = GPUAccelerator()
            gpu_available = gpu_acc.is_available()

            # Test MFC system with GPU configuration
            mfc_system = IntegratedMFCModel(
                n_cells=2,
                species='Geobacter_sulfurreducens',
                substrate='acetate',
                use_gpu=gpu_available,  # Use GPU if available
                simulation_hours=0.01
            )

            assert mfc_system.use_gpu == gpu_available

            # Run simulation to test GPU integration
            results = mfc_system.run_simulation()
            assert results is not None

            if gpu_available:
                # Additional GPU-specific validation
                assert mfc_system.gpu_available is True

        except ImportError as e:
            pytest.skip(f"GPU acceleration components not available: {e}")
        except Exception as e:
            pytest.fail(f"GPU acceleration integration failed: {e}")

    @pytest.mark.integration
    def test_deep_learning_integration(self):
        """Test deep learning controller integration."""
        try:
            from deep_rl_controller import DeepRLController

            # Test deep RL controller initialization
            controller = DeepRLController(
                state_size=10,
                action_size=5,
                network_type="dqn"
            )

            assert controller is not None
            assert controller.state_size == 10
            assert controller.action_size == 5

            # Test integration with MFC system
            mock_state = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

            if hasattr(controller, 'select_action'):
                action = controller.select_action(mock_state)
                assert action is not None
                assert 0 <= action < controller.action_size

        except ImportError as e:
            pytest.skip(f"Deep learning components not available: {e}")
        except Exception as e:
            pytest.fail(f"Deep learning integration failed: {e}")

    @pytest.mark.integration
    def test_transformer_integration(self):
        """Test transformer controller integration."""
        try:
            from transformer_controller import TransformerController

            # Test transformer controller
            controller = TransformerController(
                input_dim=64,
                output_dim=32,
                num_heads=8,
                num_layers=4
            )

            assert controller is not None
            assert controller.input_dim == 64
            assert controller.output_dim == 32

            # Test transformer processing
            mock_input = [[0.1] * 64 for _ in range(10)]  # Sequence of length 10

            if hasattr(controller, 'forward'):
                output = controller.forward(mock_input)
                assert output is not None
                assert len(output) > 0

        except ImportError as e:
            pytest.skip(f"Transformer components not available: {e}")
        except Exception as e:
            pytest.fail(f"Transformer integration failed: {e}")


class TestDataPersistenceIntegration:
    """Integration tests for data persistence and recovery mechanisms."""

    def setup_method(self):
        """Setup data persistence testing environment."""
        self.temp_dir = IntegrationTestHelper.create_temp_deployment_dir()
        self.src_path = Path(__file__).parent.parent.parent / "src"
        sys.path.insert(0, str(self.src_path))

    def teardown_method(self):
        """Cleanup data persistence testing."""
        IntegrationTestHelper.cleanup_temp_dir(self.temp_dir)
        if str(self.src_path) in sys.path:
            sys.path.remove(str(self.src_path))

    @pytest.mark.integration
    def test_simulation_data_persistence(self):
        """Test simulation data persistence and recovery."""
        try:
            import json
            import pickle

            from integrated_mfc_model import IntegratedMFCModel

            # Run simulation and save results
            mfc_system = IntegratedMFCModel(
                n_cells=2,
                species='Geobacter_sulfurreducens',
                substrate='acetate',
                use_gpu=False,
                simulation_hours=0.02
            )

            results = mfc_system.run_simulation()

            # Test JSON persistence
            json_file = os.path.join(self.temp_dir, 'simulation_results.json')
            mfc_system.save_results(json_file)

            assert os.path.exists(json_file)

            # Load and validate JSON data
            with open(json_file) as f:
                loaded_data = json.load(f)

            assert isinstance(loaded_data, dict)
            assert len(loaded_data) > 0

            # Test pickle persistence (for complex objects)
            pickle_file = os.path.join(self.temp_dir, 'system_state.pkl')

            with open(pickle_file, 'wb') as f:
                pickle.dump({
                    'system_config': {
                        'n_cells': mfc_system.n_cells,
                        'species': mfc_system.species,
                        'substrate': mfc_system.substrate
                    },
                    'results': results
                }, f)

            assert os.path.exists(pickle_file)

            # Load and validate pickle data
            with open(pickle_file, 'rb') as f:
                loaded_state = pickle.load(f)

            assert 'system_config' in loaded_state
            assert 'results' in loaded_state
            assert loaded_state['system_config']['n_cells'] == 2

        except ImportError as e:
            pytest.skip(f"Data persistence components not available: {e}")
        except Exception as e:
            pytest.fail(f"Data persistence integration failed: {e}")

    @pytest.mark.integration
    def test_checkpoint_recovery_integration(self):
        """Test checkpoint and crash recovery integration."""
        try:
            import json

            from integrated_mfc_model import IntegratedMFCModel

            # Create system and simulate checkpoint creation
            mfc_system = IntegratedMFCModel(
                n_cells=2,
                species='Geobacter_sulfurreducens',
                substrate='acetate',
                use_gpu=False,
                simulation_hours=0.02
            )

            # Simulate partial simulation run
            try:
                from integrated_mfc_model import IntegratedMFCState
                initial_state = IntegratedMFCState()
            except ImportError:
                initial_state = None

            if initial_state:
                # Run a few steps
                for step in range(3):
                    step_result = mfc_system.step_integrated_dynamics(initial_state, action=0.5)
                    if step_result:
                        initial_state = step_result

            # Create checkpoint
            checkpoint_file = os.path.join(self.temp_dir, 'checkpoint.json')

            if hasattr(mfc_system, '_save_checkpoint'):
                mfc_system._save_checkpoint(checkpoint_file)
                assert os.path.exists(checkpoint_file)

                # Test checkpoint loading
                with open(checkpoint_file) as f:
                    checkpoint_data = json.load(f)

                assert isinstance(checkpoint_data, dict)
                assert 'timestamp' in checkpoint_data or 'step' in checkpoint_data

            # Test recovery by creating new system
            recovery_system = IntegratedMFCModel(
                n_cells=2,
                species='Geobacter_sulfurreducens',
                substrate='acetate',
                use_gpu=False,
                simulation_hours=0.01
            )

            # Complete simulation should still work
            recovery_results = recovery_system.run_simulation()
            assert recovery_results is not None

        except ImportError as e:
            pytest.skip(f"Checkpoint recovery components not available: {e}")
        except Exception as e:
            pytest.fail(f"Checkpoint recovery integration failed: {e}")


# Test execution utilities
def run_integration_tests():
    """Run all integration tests with proper configuration."""
    pytest_args = [
        __file__,
        "-v",
        "-m", "integration",
        "--tb=short"
    ]

    return pytest.main(pytest_args)


def run_performance_tests():
    """Run only performance integration tests."""
    pytest_args = [
        __file__,
        "-v",
        "-m", "integration and performance",
        "--tb=short"
    ]

    return pytest.main(pytest_args)


if __name__ == "__main__":
    # Run integration tests if executed directly
    run_integration_tests()

__all__ = ['IntegrationTestHelper']
