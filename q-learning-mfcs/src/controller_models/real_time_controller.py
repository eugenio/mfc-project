"""
Real-Time Controller with Timing Constraints

This module implements a real-time control system for MFC Q-learning execution
with strict timing constraints, interrupt handling, and deterministic behavior.
"""

import numpy as np
import time
import threading
import queue
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Callable
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class ControllerMode(Enum):
    """Controller operating modes"""
    MANUAL = "manual"
    AUTOMATIC = "automatic"
    LEARNING = "learning"
    SAFETY = "safety"
    MAINTENANCE = "maintenance"


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 0  # Safety-critical tasks
    HIGH = 1      # Control loops
    MEDIUM = 2    # Data logging
    LOW = 3       # Diagnostics


@dataclass
class TimingConstraints:
    """Timing constraints for real-time control"""
    control_loop_period_ms: float  # Main control loop period
    max_jitter_ms: float          # Maximum allowed timing jitter
    deadline_violation_limit: int  # Max allowed deadline violations per hour
    interrupt_response_time_us: float  # Maximum interrupt response time
    context_switch_time_us: float  # Time for task context switching
    worst_case_execution_time_ms: float  # WCET for critical path
    
    # Safety timeouts
    watchdog_timeout_ms: float     # Hardware watchdog timeout
    safety_stop_timeout_ms: float  # Emergency stop response time
    sensor_timeout_ms: float       # Sensor data timeout


@dataclass
class ControlTask:
    """Real-time control task definition"""
    task_id: str
    priority: TaskPriority
    period_ms: float
    deadline_ms: float
    wcet_ms: float              # Worst Case Execution Time
    callback: Callable[[], Any]
    last_execution: float = 0.0
    next_execution: float = 0.0
    deadline_violations: int = 0
    execution_history: List[float] = field(default_factory=list)
    enabled: bool = True


@dataclass
class ControlLoop:
    """Control loop configuration"""
    loop_id: str
    input_channels: List[int]   # ADC channels for inputs
    output_channels: List[int]  # DAC channels for outputs
    control_algorithm: str      # PID, Q-learning, etc.
    setpoint: float
    gains: Dict[str, float]     # Control gains (Kp, Ki, Kd, etc.)
    limits: Dict[str, Tuple[float, float]]  # Output limits
    enabled: bool = True
    
    # State variables
    error_integral: float = 0.0
    previous_error: float = 0.0
    output_value: float = 0.0


@dataclass
class ControllerMeasurement:
    """Real-time controller measurement"""
    timestamp: float
    mode: ControllerMode
    cpu_utilization_pct: float
    memory_usage_mb: float
    control_loop_period_actual_ms: float
    jitter_ms: float
    deadline_violations_recent: int
    interrupt_count: int
    task_execution_times: Dict[str, float]
    active_control_loops: List[str]
    safety_state: str
    fault_flags: List[str]


class RealTimeController:
    """
    Real-time controller for MFC Q-learning execution
    
    Provides deterministic control loops, interrupt handling, and timing constraints
    for executing trained Q-learning models in real-time MFC control applications.
    """
    
    def __init__(self, timing_constraints: TimingConstraints):
        self.timing_constraints = timing_constraints
        self.mode = ControllerMode.MANUAL
        
        # Task scheduling
        self.tasks = {}
        self.task_queue = queue.PriorityQueue()
        self.scheduler_thread = None
        self.running = False
        
        # Control loops
        self.control_loops = {}
        self.sensor_data = {}
        self.actuator_outputs = {}
        
        # Timing and performance
        self.loop_start_time = 0.0
        self.loop_execution_times = deque(maxlen=1000)
        self.jitter_history = deque(maxlen=1000)
        self.deadline_violations = 0
        self.interrupt_count = 0
        
        # Safety system
        self.safety_state = "NORMAL"
        self.fault_flags = []
        self.emergency_stop = False
        self.watchdog_last_pet = time.time()
        
        # Real-time synchronization
        self.control_lock = threading.RLock()
        self.data_lock = threading.RLock()
        
        # Performance monitoring
        self.cpu_utilization = 0.0
        self.memory_usage = 0.0
        self.context_switches = 0
        
        # Initialize timing analysis
        self.timing_analyzer = TimingAnalyzer()
        
    def add_task(self, task: ControlTask):
        """Add a real-time task to the scheduler"""
        with self.control_lock:
            self.tasks[task.task_id] = task
            task.next_execution = time.time() + (task.period_ms / 1000.0)
            logger.info(f"Added task {task.task_id} with period {task.period_ms}ms")
    
    def remove_task(self, task_id: str):
        """Remove a task from the scheduler"""
        with self.control_lock:
            if task_id in self.tasks:
                del self.tasks[task_id]
                logger.info(f"Removed task {task_id}")
    
    def add_control_loop(self, loop: ControlLoop):
        """Add a control loop"""
        with self.control_lock:
            self.control_loops[loop.loop_id] = loop
            logger.info(f"Added control loop {loop.loop_id}")
    
    def set_mode(self, mode: ControllerMode):
        """Set controller operating mode"""
        with self.control_lock:
            old_mode = self.mode
            self.mode = mode
            logger.info(f"Controller mode changed from {old_mode.value} to {mode.value}")
            
            # Mode-specific initialization
            if mode == ControllerMode.SAFETY:
                self._enter_safety_mode()
            elif mode == ControllerMode.LEARNING:
                self._enter_learning_mode()
    
    def start(self):
        """Start the real-time controller"""
        if self.running:
            logger.warning("Controller already running")
            return
        
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        # Start watchdog
        self.watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self.watchdog_thread.start()
        
        logger.info("Real-time controller started")
    
    def stop(self):
        """Stop the real-time controller"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=1.0)
        if self.watchdog_thread:
            self.watchdog_thread.join(timeout=1.0)
        
        logger.info("Real-time controller stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop with real-time constraints"""
        last_loop_time = time.time()
        target_period = self.timing_constraints.control_loop_period_ms / 1000.0
        
        while self.running:
            loop_start = time.time()
            
            try:
                # Calculate timing metrics
                actual_period = loop_start - last_loop_time
                jitter = abs(actual_period - target_period) * 1000  # Convert to ms
                
                self.jitter_history.append(jitter)
                
                # Check for deadline violation
                if jitter > self.timing_constraints.max_jitter_ms:
                    self.deadline_violations += 1
                    if jitter > self.timing_constraints.max_jitter_ms * 2:
                        logger.error(f"Severe timing violation: {jitter:.2f}ms jitter")
                        self._handle_timing_violation()
                
                # Execute scheduled tasks
                self._execute_scheduled_tasks(loop_start)
                
                # Execute control loops
                if self.mode in [ControllerMode.AUTOMATIC, ControllerMode.LEARNING]:
                    self._execute_control_loops()
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Pet the watchdog
                self.watchdog_last_pet = loop_start
                
                # Calculate execution time
                execution_time = time.time() - loop_start
                self.loop_execution_times.append(execution_time * 1000)  # Convert to ms
                
                # Check worst-case execution time
                if execution_time * 1000 > self.timing_constraints.worst_case_execution_time_ms:
                    logger.warning(f"WCET violation: {execution_time*1000:.2f}ms")
                
                last_loop_time = loop_start
                
                # Sleep until next period (with real-time precision)
                sleep_time = target_period - execution_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    logger.warning(f"Control loop overrun: {-sleep_time*1000:.2f}ms")
                    
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                self._handle_controller_fault("SCHEDULER_ERROR")
    
    def _execute_scheduled_tasks(self, current_time: float):
        """Execute tasks based on their scheduling requirements"""
        tasks_to_execute = []
        
        # Find tasks ready for execution
        with self.control_lock:
            for task in self.tasks.values():
                if task.enabled and current_time >= task.next_execution:
                    tasks_to_execute.append(task)
        
        # Sort by priority
        tasks_to_execute.sort(key=lambda t: t.priority.value)
        
        # Execute tasks
        for task in tasks_to_execute:
            task_start = time.time()
            
            try:
                # Check if we have time to execute this task
                time_since_loop_start = task_start - current_time
                remaining_time = (self.timing_constraints.control_loop_period_ms / 1000.0) - time_since_loop_start
                
                if remaining_time < task.wcet_ms / 1000.0:
                    logger.warning(f"Skipping task {task.task_id} due to insufficient time")
                    continue
                
                # Execute task
                task.callback()
                
                # Update task timing
                execution_time = (time.time() - task_start) * 1000  # ms
                task.execution_history.append(execution_time)
                if len(task.execution_history) > 100:
                    task.execution_history.pop(0)
                
                # Check deadline
                if execution_time > task.deadline_ms:
                    task.deadline_violations += 1
                    logger.warning(f"Task {task.task_id} deadline violation: {execution_time:.2f}ms > {task.deadline_ms}ms")
                
                # Schedule next execution
                task.last_execution = task_start
                task.next_execution = task_start + (task.period_ms / 1000.0)
                
                # Update statistics
                self.timing_analyzer.record_task_execution(task.task_id, execution_time)
                
            except Exception as e:
                logger.error(f"Task {task.task_id} execution error: {e}")
                task.deadline_violations += 1
    
    def _execute_control_loops(self):
        """Execute active control loops"""
        with self.control_lock:
            for loop_id, loop in self.control_loops.items():
                if not loop.enabled:
                    continue
                
                try:
                    # Read sensor inputs
                    inputs = {}
                    for channel in loop.input_channels:
                        if channel in self.sensor_data:
                            inputs[channel] = self.sensor_data[channel]
                        else:
                            logger.warning(f"Missing sensor data for channel {channel}")
                            inputs[channel] = 0.0
                    
                    # Calculate control output
                    if loop.control_algorithm == "PID":
                        output = self._execute_pid_control(loop, inputs)
                    elif loop.control_algorithm == "Q-learning":
                        output = self._execute_qlearning_control(loop, inputs)
                    else:
                        logger.error(f"Unknown control algorithm: {loop.control_algorithm}")
                        continue
                    
                    # Apply output limits
                    if 'output' in loop.limits:
                        min_out, max_out = loop.limits['output']
                        output = np.clip(output, min_out, max_out)
                    
                    loop.output_value = output
                    
                    # Send outputs to actuators
                    for channel in loop.output_channels:
                        self.actuator_outputs[channel] = output
                        
                except Exception as e:
                    logger.error(f"Control loop {loop_id} execution error: {e}")
                    self._handle_controller_fault(f"CONTROL_LOOP_ERROR_{loop_id}")
    
    def _execute_pid_control(self, loop: ControlLoop, inputs: Dict[int, float]) -> float:
        """Execute PID control algorithm"""
        # Get process variable (assume first input channel)
        if not inputs:
            return 0.0
        
        pv = list(inputs.values())[0]
        error = loop.setpoint - pv
        
        # PID calculation
        kp = loop.gains.get('kp', 1.0)
        ki = loop.gains.get('ki', 0.0)
        kd = loop.gains.get('kd', 0.0)
        
        # Proportional term
        p_term = kp * error
        
        # Integral term
        dt = self.timing_constraints.control_loop_period_ms / 1000.0
        loop.error_integral += error * dt
        
        # Integral windup protection
        if 'integral' in loop.limits:
            min_int, max_int = loop.limits['integral']
            loop.error_integral = np.clip(loop.error_integral, min_int, max_int)
        
        i_term = ki * loop.error_integral
        
        # Derivative term
        d_term = kd * (error - loop.previous_error) / dt
        loop.previous_error = error
        
        # Derivative filtering (optional)
        alpha = loop.gains.get('derivative_filter', 1.0)
        d_term = alpha * d_term + (1 - alpha) * getattr(loop, '_filtered_derivative', 0.0)
        loop._filtered_derivative = d_term
        
        output = p_term + i_term + d_term
        
        return output
    
    def _execute_qlearning_control(self, loop: ControlLoop, inputs: Dict[int, float]) -> float:
        """Execute Q-learning control algorithm"""
        # This would interface with the ModelInferenceEngine
        # For now, implement a simple placeholder
        
        # Convert inputs to state vector (for future Q-learning integration)
        # state = np.array(list(inputs.values()) + [loop.setpoint])
        
        # Placeholder: simple bang-bang control based on error
        if not inputs:
            return 0.0
        
        pv = list(inputs.values())[0]
        error = loop.setpoint - pv
        
        # Simple decision logic (would be replaced by Q-learning inference)
        if abs(error) < 0.1:
            return 0.0  # No action needed
        elif error > 0:
            return 1.0  # Increase output
        else:
            return -1.0  # Decrease output
    
    def _handle_timing_violation(self):
        """Handle timing constraint violations"""
        if self.deadline_violations > self.timing_constraints.deadline_violation_limit:
            logger.critical("Too many deadline violations - entering safety mode")
            self.set_mode(ControllerMode.SAFETY)
    
    def _handle_controller_fault(self, fault_code: str):
        """Handle controller faults"""
        self.fault_flags.append(fault_code)
        logger.error(f"Controller fault: {fault_code}")
        
        # Critical faults trigger safety mode
        critical_faults = ["SCHEDULER_ERROR", "WATCHDOG_TIMEOUT", "SAFETY_VIOLATION"]
        if any(cf in fault_code for cf in critical_faults):
            self.set_mode(ControllerMode.SAFETY)
    
    def _enter_safety_mode(self):
        """Enter safety mode - disable all outputs"""
        self.safety_state = "EMERGENCY_STOP"
        
        # Clear all actuator outputs
        with self.control_lock:
            for channel in self.actuator_outputs:
                self.actuator_outputs[channel] = 0.0
            
            # Disable all control loops
            for loop in self.control_loops.values():
                loop.enabled = False
        
        logger.critical("Entered safety mode - all outputs disabled")
    
    def _enter_learning_mode(self):
        """Enter learning mode for Q-learning"""
        self.safety_state = "LEARNING"
        logger.info("Entered learning mode")
    
    def _watchdog_loop(self):
        """Hardware watchdog monitoring"""
        while self.running:
            current_time = time.time()
            time_since_pet = (current_time - self.watchdog_last_pet) * 1000  # ms
            
            if time_since_pet > self.timing_constraints.watchdog_timeout_ms:
                logger.critical("Watchdog timeout - system failure detected")
                self._handle_controller_fault("WATCHDOG_TIMEOUT")
                break
            
            time.sleep(0.1)  # Check every 100ms
    
    def _update_performance_metrics(self):
        """Update CPU and memory usage metrics"""
        try:
            import psutil
            self.cpu_utilization = psutil.cpu_percent()
            process = psutil.Process()
            self.memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            # Fallback if psutil not available
            self.cpu_utilization = len(self.loop_execution_times) * 2.0  # Rough estimate
            self.memory_usage = 50.0  # Fixed estimate
    
    def update_sensor_data(self, channel: int, value: float):
        """Update sensor data (thread-safe)"""
        with self.data_lock:
            self.sensor_data[channel] = value
    
    def get_actuator_output(self, channel: int) -> float:
        """Get actuator output value (thread-safe)"""
        with self.data_lock:
            return self.actuator_outputs.get(channel, 0.0)
    
    def set_control_loop_setpoint(self, loop_id: str, setpoint: float):
        """Set control loop setpoint"""
        with self.control_lock:
            if loop_id in self.control_loops:
                self.control_loops[loop_id].setpoint = setpoint
    
    def enable_control_loop(self, loop_id: str, enabled: bool = True):
        """Enable/disable control loop"""
        with self.control_lock:
            if loop_id in self.control_loops:
                self.control_loops[loop_id].enabled = enabled
    
    def get_measurement(self) -> ControllerMeasurement:
        """Get comprehensive controller measurement"""
        with self.control_lock:
            # Calculate timing metrics
            recent_jitter = list(self.jitter_history)[-100:] if self.jitter_history else [0]
            avg_jitter = np.mean(recent_jitter)
            
            recent_periods = list(self.loop_execution_times)[-100:] if self.loop_execution_times else [0]
            avg_period = np.mean(recent_periods)
            
            # Count recent deadline violations
            recent_violations = sum(1 for task in self.tasks.values() 
                                  if task.deadline_violations > 0)
            
            # Get task execution times
            task_times = {}
            for task_id, task in self.tasks.items():
                if task.execution_history:
                    task_times[task_id] = np.mean(task.execution_history[-10:])
                else:
                    task_times[task_id] = 0.0
            
            # Get active control loops
            active_loops = [loop_id for loop_id, loop in self.control_loops.items() 
                          if loop.enabled]
            
            return ControllerMeasurement(
                timestamp=time.time(),
                mode=self.mode,
                cpu_utilization_pct=self.cpu_utilization,
                memory_usage_mb=self.memory_usage,
                control_loop_period_actual_ms=avg_period,
                jitter_ms=avg_jitter,
                deadline_violations_recent=recent_violations,
                interrupt_count=self.interrupt_count,
                task_execution_times=task_times,
                active_control_loops=active_loops,
                safety_state=self.safety_state,
                fault_flags=self.fault_flags.copy()
            )
    
    def get_timing_analysis(self) -> Dict[str, Any]:
        """Get detailed timing analysis"""
        return self.timing_analyzer.get_analysis()


class TimingAnalyzer:
    """Analyze timing performance and detect anomalies"""
    
    def __init__(self):
        self.task_executions = {}
        self.timing_violations = []
        self.performance_history = deque(maxlen=10000)
    
    def record_task_execution(self, task_id: str, execution_time_ms: float):
        """Record task execution time"""
        if task_id not in self.task_executions:
            self.task_executions[task_id] = deque(maxlen=1000)
        
        self.task_executions[task_id].append(execution_time_ms)
        self.performance_history.append({
            'timestamp': time.time(),
            'task_id': task_id,
            'execution_time_ms': execution_time_ms
        })
    
    def get_analysis(self) -> Dict[str, Any]:
        """Get comprehensive timing analysis"""
        analysis = {
            'tasks': {},
            'overall': {}
        }
        
        # Per-task analysis
        for task_id, executions in self.task_executions.items():
            if executions:
                exec_times = list(executions)
                analysis['tasks'][task_id] = {
                    'count': len(exec_times),
                    'mean_ms': np.mean(exec_times),
                    'std_ms': np.std(exec_times),
                    'min_ms': np.min(exec_times),
                    'max_ms': np.max(exec_times),
                    'p95_ms': np.percentile(exec_times, 95),
                    'p99_ms': np.percentile(exec_times, 99)
                }
        
        # Overall system analysis
        if self.performance_history:
            all_times = [entry['execution_time_ms'] for entry in self.performance_history]
            analysis['overall'] = {
                'total_executions': len(all_times),
                'mean_execution_time_ms': np.mean(all_times),
                'system_utilization_pct': min(100.0, np.sum(all_times) / 
                                             (len(all_times) * 10.0))  # Assume 10ms period
            }
        
        return analysis


def create_standard_real_time_controllers() -> Dict[str, RealTimeController]:
    """Create standard real-time controller configurations"""
    
    # High-performance real-time controller
    hp_timing = TimingConstraints(
        control_loop_period_ms=1.0,       # 1ms control loop
        max_jitter_ms=0.1,                # 100μs jitter tolerance
        deadline_violation_limit=10,       # Max 10 violations per hour
        interrupt_response_time_us=10.0,   # 10μs interrupt response
        context_switch_time_us=5.0,        # 5μs context switch
        worst_case_execution_time_ms=0.5,  # 500μs WCET
        watchdog_timeout_ms=100.0,         # 100ms watchdog
        safety_stop_timeout_ms=10.0,       # 10ms emergency stop
        sensor_timeout_ms=50.0             # 50ms sensor timeout
    )
    
    # Low-power embedded controller
    lp_timing = TimingConstraints(
        control_loop_period_ms=10.0,       # 10ms control loop
        max_jitter_ms=2.0,                 # 2ms jitter tolerance
        deadline_violation_limit=50,       # Max 50 violations per hour
        interrupt_response_time_us=100.0,  # 100μs interrupt response
        context_switch_time_us=50.0,       # 50μs context switch
        worst_case_execution_time_ms=5.0,  # 5ms WCET
        watchdog_timeout_ms=1000.0,        # 1s watchdog
        safety_stop_timeout_ms=100.0,      # 100ms emergency stop
        sensor_timeout_ms=500.0            # 500ms sensor timeout
    )
    
    controllers = {
        'high_performance': RealTimeController(hp_timing),
        'low_power': RealTimeController(lp_timing)
    }
    
    return controllers


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create real-time controllers
    controllers = create_standard_real_time_controllers()
    
    # Test high-performance controller
    hp_controller = controllers['high_performance']
    
    print("Testing real-time controller")
    print(f"Control period: {hp_controller.timing_constraints.control_loop_period_ms}ms")
    print(f"Max jitter: {hp_controller.timing_constraints.max_jitter_ms}ms")
    
    # Add some test tasks
    def sensor_task():
        """Simulate sensor reading"""
        time.sleep(0.001)  # 1ms simulated work
        return "sensor_data"
    
    def control_task():
        """Simulate control calculation"""
        time.sleep(0.0005)  # 0.5ms simulated work
        return "control_output"
    
    sensor_task_def = ControlTask(
        task_id="sensor_read",
        priority=TaskPriority.HIGH,
        period_ms=2.0,
        deadline_ms=1.5,
        wcet_ms=1.2,
        callback=sensor_task
    )
    
    control_task_def = ControlTask(
        task_id="control_calc",
        priority=TaskPriority.CRITICAL,
        period_ms=1.0,
        deadline_ms=0.8,
        wcet_ms=0.6,
        callback=control_task
    )
    
    hp_controller.add_task(sensor_task_def)
    hp_controller.add_task(control_task_def)
    
    # Add a control loop
    pid_loop = ControlLoop(
        loop_id="temperature_control",
        input_channels=[0],
        output_channels=[0],
        control_algorithm="PID",
        setpoint=25.0,
        gains={'kp': 1.0, 'ki': 0.1, 'kd': 0.05},
        limits={'output': (-10.0, 10.0), 'integral': (-5.0, 5.0)}
    )
    
    hp_controller.add_control_loop(pid_loop)
    
    # Start controller
    hp_controller.set_mode(ControllerMode.AUTOMATIC)
    hp_controller.start()
    
    # Simulate operation
    print("\nSimulating real-time operation...")
    start_time = time.time()
    
    try:
        for i in range(50):  # Run for 50 cycles
            # Update sensor data
            hp_controller.update_sensor_data(0, 23.0 + np.random.normal(0, 0.5))
            
            # Get measurement
            if i % 10 == 0:  # Every 10 cycles
                measurement = hp_controller.get_measurement()
                print(f"Cycle {i}: Mode={measurement.mode.value}, "
                      f"Jitter={measurement.jitter_ms:.3f}ms, "
                      f"CPU={measurement.cpu_utilization_pct:.1f}%")
            
            time.sleep(0.01)  # 10ms between updates
            
    finally:
        hp_controller.stop()
    
    # Get timing analysis
    timing_analysis = hp_controller.get_timing_analysis()
    print("\nTiming Analysis:")
    for task_id, stats in timing_analysis.get('tasks', {}).items():
        print(f"  {task_id}: mean={stats['mean_ms']:.3f}ms, "
              f"max={stats['max_ms']:.3f}ms, "
              f"p99={stats['p99_ms']:.3f}ms")
    
    print(f"\nTotal runtime: {time.time() - start_time:.2f}s")