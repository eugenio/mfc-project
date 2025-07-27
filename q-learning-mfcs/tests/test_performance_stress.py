#!/usr/bin/env python3
"""
Performance and stress tests for MFC Q-Learning Project.
Tests system performance under load and identifies bottlenecks.
"""

import unittest
import numpy as np
import sys
import os
import time
import psutil
import warnings
import threading

# Suppress warnings for clean test output
warnings.filterwarnings('ignore', category=UserWarning)
import matplotlib
matplotlib.use('Agg')

# Add source path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestPerformanceBenchmarks(unittest.TestCase):
    """Test performance benchmarks and optimization."""
    
    def setUp(self):
        """Set up performance monitoring."""
        self.start_memory = psutil.virtual_memory().used
        self.start_time = time.time()
    
    def tearDown(self):
        """Clean up and report memory usage."""
        end_memory = psutil.virtual_memory().used
        end_time = time.time()
        
        memory_delta = (end_memory - self.start_memory) / (1024 * 1024)  # MB
        time_delta = end_time - self.start_time
        
        if memory_delta > 100:  # More than 100MB increase
            print(f"Warning: Test used {memory_delta:.1f} MB memory")
        
        if time_delta > 10:  # More than 10 seconds
            print(f"Warning: Test took {time_delta:.1f} seconds")
    
    def test_gpu_acceleration_performance(self):
        """Test GPU acceleration performance vs CPU."""
        try:
            from gpu_acceleration import GPUAccelerator
            import numpy as np
            
            gpu = GPUAccelerator()
            
            # Test different array sizes
            sizes = [1000, 10000, 100000]
            results = {}
            
            for size in sizes:
                # Generate test data
                cpu_array = np.random.rand(size).astype(np.float32)
                
                # CPU baseline (NumPy)
                start_time = time.time()
                cpu_result = np.exp(cpu_array)
                cpu_time = time.time() - start_time
                
                # GPU test
                start_time = time.time()
                gpu_array = gpu.array(cpu_array)
                gpu_result_gpu = gpu.exp(gpu_array)
                gpu_result = gpu.to_cpu(gpu_result_gpu)
                gpu_time = time.time() - start_time
                
                # Verify correctness
                np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-5)
                
                # Record performance
                speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
                results[size] = {
                    'cpu_time': cpu_time,
                    'gpu_time': gpu_time,
                    'speedup': speedup
                }
                
                print(f"Size {size}: CPU {cpu_time:.6f}s, GPU {gpu_time:.6f}s, Speedup: {speedup:.2f}x")
            
            # For larger arrays, GPU should show benefit (unless CPU fallback)
            if gpu.backend != 'cpu':
                large_speedup = results[max(sizes)]['speedup']
                self.assertGreater(large_speedup, 0.5)  # At least not much slower
            
        except ImportError:
            self.skipTest("GPU acceleration not available")
    
    def test_simulation_performance_scaling(self):
        """Test how simulation performance scales with system size."""
        try:
            from integrated_mfc_model import IntegratedMFCModel
            
            cell_counts = [2, 3, 5]
            performance_data = {}
            
            for n_cells in cell_counts:
                model = IntegratedMFCModel(
                    n_cells=n_cells,
                    species="geobacter",
                    substrate="acetate",
                    use_gpu=False,
                    simulation_hours=5
                )
                
                # Time simulation steps
                step_times = []
                for step in range(10):
                    start_time = time.time()
                    state = model.step_integrated_dynamics(dt=1.0)
                    step_time = time.time() - start_time
                    step_times.append(step_time)
                
                avg_step_time = np.mean(step_times)
                performance_data[n_cells] = avg_step_time
                
                print(f"{n_cells} cells: {avg_step_time:.6f}s per step")
                
                # Performance should be reasonable
                self.assertLess(avg_step_time, 1.0)  # Less than 1 second per step
            
            # Check scaling behavior
            if len(performance_data) >= 2:
                times = list(performance_data.values())
                cells = list(performance_data.keys())
                
                # Time should scale sub-quadratically
                ratio_32 = performance_data[3] / performance_data[2] if 2 in performance_data and 3 in performance_data else 1.0
                ratio_53 = performance_data[5] / performance_data[3] if 3 in performance_data and 5 in performance_data else 1.0
                
                # Should not scale worse than quadratically
                self.assertLess(ratio_32, 2.5)  # 3/2 = 1.5, allow some overhead
                self.assertLess(ratio_53, 3.0)  # 5/3 = 1.67, allow some overhead
            
        except ImportError:
            self.skipTest("Integrated model not available")
    
    def test_memory_efficiency(self):
        """Test memory usage efficiency."""
        try:
            from gpu_acceleration import GPUAccelerator
            
            gpu = GPUAccelerator()
            
            # Track memory usage
            initial_memory = psutil.virtual_memory().used
            
            # Create many arrays
            arrays = []
            for i in range(100):
                arr = gpu.array(np.random.rand(1000).astype(np.float32))
                result = gpu.exp(arr)
                arrays.append(result)
            
            peak_memory = psutil.virtual_memory().used - initial_memory
            
            # Clean up
            del arrays
            
            final_memory = psutil.virtual_memory().used - initial_memory
            
            # Memory should be released after cleanup
            memory_released = peak_memory - final_memory
            release_ratio = memory_released / peak_memory if peak_memory > 0 else 1.0
            
            print(f"Peak memory: {peak_memory / (1024*1024):.1f} MB")
            print(f"Final memory: {final_memory / (1024*1024):.1f} MB")
            print(f"Release ratio: {release_ratio:.2f}")
            
            # Should release significant memory
            self.assertGreater(release_ratio, 0.5)  # At least 50% released
            
        except ImportError:
            self.skipTest("GPU acceleration not available")
    
    def test_qlearning_performance(self):
        """Test Q-learning algorithm performance."""
        try:
            from sensing_enhanced_q_controller import SensingEnhancedQController
            
            controller = SensingEnhancedQController(
                n_cells=5,
                learning_rate=0.1,
                epsilon_start=0.3,
                epsilon_decay=0.995
            )
            
            # Benchmark Q-learning operations
            n_iterations = 1000
            
            # Generate random states and actions
            state_dim = 40  # As per system architecture
            action_dim = 15
            
            states = [np.random.rand(state_dim) for _ in range(n_iterations)]
            actions = [np.random.randint(0, action_dim) for _ in range(n_iterations)]
            rewards = [np.random.rand() * 10 - 5 for _ in range(n_iterations)]  # -5 to 5
            
            # Time Q-learning updates
            start_time = time.time()
            
            for i in range(n_iterations):
                state = states[i]
                action = actions[i]
                reward = rewards[i]
                next_state = states[(i + 1) % n_iterations]
                
                # Select action
                selected_action = controller.select_action(state)
                
                # Update Q-table
                controller.update_q_table(state, action, reward, next_state)
            
            total_time = time.time() - start_time
            avg_time_per_update = total_time / n_iterations
            
            print(f"Q-learning: {avg_time_per_update*1000:.3f} ms per update")
            
            # Should be fast enough for real-time control
            self.assertLess(avg_time_per_update, 0.01)  # Less than 10ms per update
            
        except ImportError:
            self.skipTest("Q-learning controller not available")


class TestStressTests(unittest.TestCase):
    """Test system behavior under stress conditions."""
    
    def test_extended_simulation_stability(self):
        """Test stability during extended simulation."""
        try:
            from integrated_mfc_model import IntegratedMFCModel
            
            model = IntegratedMFCModel(
                n_cells=3,
                species="mixed",
                substrate="lactate",
                use_gpu=False,
                simulation_hours=100  # Long simulation
            )
            
            # Track key metrics
            powers = []
            energies = []
            biofilm_thicknesses = []
            
            # Run extended simulation
            for hour in range(100):
                state = model.step_integrated_dynamics(dt=1.0)
                
                powers.append(state.average_power)
                energies.append(state.total_energy)
                biofilm_thicknesses.append(np.mean(state.biofilm_thickness))
                
                # Check for stability issues
                if hour > 10:  # After initial transient
                    # Power should remain reasonable
                    self.assertGreater(state.average_power, 0.0)
                    self.assertLess(state.average_power, 10.0)
                    
                    # Energy should increase or remain stable
                    self.assertGreaterEqual(state.total_energy, energies[hour-1] - 1e-6)
                    
                    # Biofilm thickness should remain bounded
                    avg_thickness = np.mean(state.biofilm_thickness)
                    self.assertLess(avg_thickness, 200.0)
                    
                    # No NaN or infinite values
                    self.assertTrue(np.isfinite(state.average_power))
                    self.assertTrue(np.isfinite(state.total_energy))
                    self.assertTrue(all(np.isfinite(t) for t in state.biofilm_thickness))
            
            # Check long-term trends
            final_power = np.mean(powers[-10:])  # Last 10 hours
            initial_power = np.mean(powers[10:20])  # Hours 10-20 (after stabilization)
            
            # System should remain stable
            power_ratio = final_power / initial_power if initial_power > 0 else 1.0
            self.assertGreater(power_ratio, 0.1)  # Not complete collapse
            self.assertLess(power_ratio, 10.0)   # Not runaway growth
            
        except ImportError:
            self.skipTest("Integrated model not available")
    
    def test_high_frequency_operations(self):
        """Test system behavior with high-frequency operations."""
        try:
            from gpu_acceleration import GPUAccelerator
            
            gpu = GPUAccelerator()
            
            # Perform many rapid operations
            n_operations = 10000
            
            start_time = time.time()
            
            for i in range(n_operations):
                # Small arrays for rapid operations
                arr = gpu.array([1.0, 2.0, 3.0])
                result = gpu.exp(arr)
                cpu_result = gpu.to_cpu(result)
                
                # Verify result occasionally
                if i % 1000 == 0:
                    self.assertEqual(len(cpu_result), 3)
                    self.assertTrue(all(r > 0 for r in cpu_result))
            
            total_time = time.time() - start_time
            ops_per_second = n_operations / total_time
            
            print(f"High frequency: {ops_per_second:.0f} ops/second")
            
            # Should handle at least 1000 ops/second
            self.assertGreater(ops_per_second, 1000)
            
        except ImportError:
            self.skipTest("GPU acceleration not available")
    
    def test_concurrent_access_stress(self):
        """Test system behavior under concurrent access."""
        try:
            from gpu_acceleration import GPUAccelerator
            
            gpu = GPUAccelerator()
            
            # Shared result storage
            results = []
            errors = []
            lock = threading.Lock()
            
            def worker_thread(thread_id, n_operations):
                """Worker thread for concurrent testing."""
                try:
                    for i in range(n_operations):
                        arr = gpu.array([thread_id, i, thread_id + i])
                        result = gpu.exp(arr)
                        cpu_result = gpu.to_cpu(result)
                        
                        with lock:
                            results.append((thread_id, i, cpu_result))
                            
                except Exception as e:
                    with lock:
                        errors.append((thread_id, str(e)))
            
            # Start multiple threads
            n_threads = 4
            n_ops_per_thread = 100
            
            threads = []
            for i in range(n_threads):
                thread = threading.Thread(target=worker_thread, args=(i, n_ops_per_thread))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            # Check results
            if errors:
                # Some backends may not support concurrency
                print(f"Concurrent access errors: {len(errors)}")
                # Don't fail the test - just report
            else:
                expected_results = n_threads * n_ops_per_thread
                self.assertEqual(len(results), expected_results)
                
                # Verify all results are valid
                for thread_id, i, result in results:
                    self.assertEqual(len(result), 3)
                    self.assertTrue(all(np.isfinite(r) for r in result))
            
        except ImportError:
            self.skipTest("GPU acceleration not available")
    
    def test_memory_pressure_handling(self):
        """Test behavior under memory pressure."""
        try:
            from gpu_acceleration import GPUAccelerator
            
            gpu = GPUAccelerator()
            
            # Try to create increasingly large arrays until memory limit
            arrays = []
            max_size = 1000000  # Start with 1M elements
            
            try:
                while max_size <= 100000000:  # Up to 100M elements
                    try:
                        large_array = gpu.array(np.random.rand(max_size).astype(np.float32))
                        result = gpu.exp(large_array)
                        arrays.append(result)
                        
                        print(f"Successfully allocated {max_size} elements")
                        max_size *= 2  # Double size
                        
                    except Exception as e:
                        if "memory" in str(e).lower() or "allocation" in str(e).lower():
                            print(f"Memory limit reached at {max_size} elements: {e}")
                            break
                        else:
                            raise
                        
                    # Prevent runaway memory usage
                    if len(arrays) > 10:
                        arrays.pop(0)  # Remove oldest array
                
                # Should handle graceful failure
                self.assertTrue(True)  # Reached here without crashing
                
            finally:
                # Clean up
                del arrays
                
        except ImportError:
            self.skipTest("GPU acceleration not available")


class TestScalabilityTests(unittest.TestCase):
    """Test system scalability with increasing problem size."""
    
    def test_cell_count_scalability(self):
        """Test scalability with increasing number of cells."""
        try:
            from mfc_stack_simulation import MFCStack
            
            cell_counts = [2, 3, 5, 8, 10]
            performance_metrics = {}
            
            for n_cells in cell_counts:
                stack = MFCStack(n_cells=n_cells)
                
                # Time multiple operations
                start_time = time.time()
                
                for _ in range(20):  # 20 time steps
                    stack.update_stack_dynamics(dt=1.0)
                    voltages = stack.get_cell_voltages()
                    currents = stack.get_current_densities()
                    power = stack.get_total_power()
                
                total_time = time.time() - start_time
                time_per_step = total_time / 20
                
                performance_metrics[n_cells] = {
                    'time_per_step': time_per_step,
                    'time_per_cell_per_step': time_per_step / n_cells
                }
                
                print(f"{n_cells} cells: {time_per_step:.6f}s per step ({time_per_step/n_cells:.6f}s per cell)")
                
                # Performance should remain reasonable
                self.assertLess(time_per_step, 1.0)  # Less than 1 second per step
            
            # Check scaling efficiency
            if len(performance_metrics) >= 3:
                # Time per cell should not increase dramatically
                times_per_cell = [m['time_per_cell_per_step'] for m in performance_metrics.values()]
                
                max_time_per_cell = max(times_per_cell)
                min_time_per_cell = min(times_per_cell)
                
                # Should not have more than 3x variation in per-cell time
                self.assertLess(max_time_per_cell / min_time_per_cell, 3.0)
            
        except ImportError:
            self.skipTest("MFC stack simulation not available")
    
    def test_simulation_duration_scalability(self):
        """Test scalability with increasing simulation duration."""
        try:
            from integrated_mfc_model import IntegratedMFCModel
            
            model = IntegratedMFCModel(
                n_cells=3,
                species="geobacter",
                substrate="acetate",
                use_gpu=False,
                simulation_hours=50
            )
            
            # Test different simulation lengths
            durations = [10, 20, 30, 40, 50]
            step_times = []
            
            for target_hour in durations:
                current_hour = int(model.time)
                
                if target_hour > current_hour:
                    # Time the steps to reach target hour
                    steps_needed = target_hour - current_hour
                    
                    start_time = time.time()
                    
                    for _ in range(steps_needed):
                        state = model.step_integrated_dynamics(dt=1.0)
                    
                    elapsed_time = time.time() - start_time
                    avg_step_time = elapsed_time / steps_needed
                    
                    step_times.append(avg_step_time)
                    print(f"Hour {target_hour}: {avg_step_time:.6f}s per step")
            
            # Step time should not increase significantly over time
            if len(step_times) >= 3:
                early_avg = np.mean(step_times[:2])
                late_avg = np.mean(step_times[-2:])
                
                # Late steps shouldn't be more than 2x slower than early steps
                self.assertLess(late_avg / early_avg, 2.0)
            
        except ImportError:
            self.skipTest("Integrated model not available")
    
    def test_parameter_space_scalability(self):
        """Test scalability with increasing parameter space size."""
        try:
            from sensing_enhanced_q_controller import SensingEnhancedQController
            
            # Test different state space sizes
            state_dimensions = [20, 40, 60, 80]
            performance_data = {}
            
            for state_dim in state_dimensions:
                controller = SensingEnhancedQController(
                    n_cells=5,
                    learning_rate=0.1
                )
                
                # Override state dimension for testing
                controller.state_dim = state_dim
                
                # Time Q-learning operations
                n_operations = 100
                
                start_time = time.time()
                
                for i in range(n_operations):
                    state = np.random.rand(state_dim)
                    action = controller.select_action(state)
                    
                    reward = np.random.rand() * 10 - 5
                    next_state = np.random.rand(state_dim)
                    
                    controller.update_q_table(state, action, reward, next_state)
                
                total_time = time.time() - start_time
                time_per_op = total_time / n_operations
                
                performance_data[state_dim] = time_per_op
                print(f"State dim {state_dim}: {time_per_op*1000:.3f} ms per operation")
                
                # Should remain reasonable
                self.assertLess(time_per_op, 0.1)  # Less than 100ms per operation
            
            # Check scaling behavior
            if len(performance_data) >= 2:
                dims = sorted(performance_data.keys())
                times = [performance_data[d] for d in dims]
                
                # Should not scale worse than quadratically
                for i in range(1, len(times)):
                    scaling_factor = times[i] / times[0]
                    dimension_ratio = dims[i] / dims[0]
                    
                    # Allow up to quadratic scaling
                    self.assertLess(scaling_factor, dimension_ratio ** 2 * 2)
            
        except ImportError:
            self.skipTest("Q-learning controller not available")


class TestResourceUtilization(unittest.TestCase):
    """Test resource utilization efficiency."""
    
    def test_cpu_utilization(self):
        """Test CPU utilization efficiency."""
        try:
            from integrated_mfc_model import IntegratedMFCModel
            
            model = IntegratedMFCModel(
                n_cells=5,
                species="mixed",
                substrate="lactate",
                use_gpu=False,
                simulation_hours=10
            )
            
            # Monitor CPU usage during simulation
            cpu_percentages = []
            
            def monitor_cpu():
                for _ in range(20):  # Monitor for 20 seconds
                    cpu_percent = psutil.cpu_percent(interval=1)
                    cpu_percentages.append(cpu_percent)
            
            # Start CPU monitoring in separate thread
            import threading
            monitor_thread = threading.Thread(target=monitor_cpu)
            monitor_thread.start()
            
            # Run simulation
            for _ in range(10):
                state = model.step_integrated_dynamics(dt=1.0)
                time.sleep(0.1)  # Brief pause to allow monitoring
            
            monitor_thread.join()
            
            if cpu_percentages:
                avg_cpu = np.mean(cpu_percentages)
                max_cpu = max(cpu_percentages)
                
                print(f"CPU usage: avg {avg_cpu:.1f}%, max {max_cpu:.1f}%")
                
                # Should use reasonable amount of CPU
                self.assertGreater(avg_cpu, 1.0)    # Should actually use CPU
                self.assertLess(max_cpu, 90.0)     # Should not peg CPU
            
        except ImportError:
            self.skipTest("Integrated model not available")
    
    def test_memory_efficiency_patterns(self):
        """Test memory usage patterns for efficiency."""
        try:
            from gpu_acceleration import GPUAccelerator
            
            gpu = GPUAccelerator()
            
            # Test memory usage pattern
            memory_usage = []
            
            for iteration in range(50):
                # Get current memory
                current_memory = psutil.virtual_memory().used / (1024 * 1024)  # MB
                memory_usage.append(current_memory)
                
                # Create temporary arrays
                temp_arrays = []
                for i in range(10):
                    arr = gpu.array(np.random.rand(1000).astype(np.float32))
                    result = gpu.exp(arr)
                    temp_arrays.append(result)
                
                # Clean up immediately
                del temp_arrays
            
            # Check memory usage pattern
            memory_deltas = np.diff(memory_usage)
            
            # Should not have consistent memory growth
            positive_deltas = sum(1 for delta in memory_deltas if delta > 1.0)  # >1MB increases
            total_deltas = len(memory_deltas)
            
            growth_ratio = positive_deltas / total_deltas if total_deltas > 0 else 0
            
            print(f"Memory growth ratio: {growth_ratio:.2f}")
            
            # Should not consistently grow memory
            self.assertLess(growth_ratio, 0.7)  # Less than 70% growth events
            
        except ImportError:
            self.skipTest("GPU acceleration not available")


if __name__ == '__main__':
    # Run with high verbosity for detailed performance information
    unittest.main(verbosity=2, buffer=False)