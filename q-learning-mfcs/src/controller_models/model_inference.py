"""
Model Inference Engine for Q-Learning Execution

This module implements a high-performance inference engine for executing trained
Q-learning models in real-time MFC control applications. Supports multiple model
formats and provides optimized execution paths for embedded deployment.
"""

import json
import logging
import pickle
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class ModelFormat(Enum):
    """Supported model formats for inference"""
    PICKLE = "pickle"  # Python pickle format
    JSON = "json"  # JSON Q-table format
    NUMPY = "numpy"  # NumPy array format
    ONNX = "onnx"  # ONNX format (for neural networks)
    TFLITE = "tflite"  # TensorFlow Lite format


@dataclass
class InferenceSpecs:
    """Specifications for model inference engine"""
    model_format: ModelFormat
    max_inference_time_ms: float  # Maximum allowed inference time
    memory_limit_mb: float  # Memory limit for model
    cache_size: int  # Number of cached inferences
    batch_processing: bool  # Enable batch processing
    quantization: bool  # Use quantized models
    optimization_level: int  # 0=none, 1=basic, 2=aggressive
    power_consumption: float  # W (for inference hardware)
    cost: float  # USD

    # Hardware constraints
    cpu_cores: int  # Available CPU cores
    ram_mb: float  # Available RAM
    storage_mb: float  # Storage for models
    temperature_range: Tuple[float, float]  # Operating temperature range


@dataclass
class InferenceMeasurement:
    """Single inference measurement"""
    timestamp: float
    input_state: np.ndarray
    output_action: Union[int, np.ndarray]
    confidence_score: float  # 0-1 confidence in output
    inference_time_ms: float
    memory_usage_mb: float
    cpu_usage_pct: float
    cache_hit: bool
    model_version: str


class ModelInferenceEngine:
    """High-performance inference engine for Q-learning models"""

    def __init__(self, specs: InferenceSpecs):
        self.specs = specs
        self.model = None
        self.model_metadata = {}
        self.inference_cache = {}
        self.performance_history = []

        # Performance monitoring
        self.total_inferences = 0
        self.cache_hits = 0
        self.average_inference_time = 0.0
        self.peak_memory_usage = 0.0

        # Model optimization settings
        self.state_discretization_bins = 10
        self.epsilon_greedy = False
        self.epsilon = 0.01

        # Real-time constraints
        self.deadline_violations = 0
        self.last_inference_time = 0.0

        # Initialize optimization based on level
        self._initialize_optimization()

    def _initialize_optimization(self):
        """Initialize optimization features based on level"""
        if self.specs.optimization_level >= 1:
            # Basic optimizations
            self.enable_vectorization = True
            self.precompute_common_operations = True

        if self.specs.optimization_level >= 2:
            # Aggressive optimizations
            self.enable_jit_compilation = True
            self.use_lookup_tables = True
            self.parallel_processing = True

    def load_model(self, model_path: str, metadata_path: Optional[str] = None) -> bool:
        """
        Load trained Q-learning model from file
        
        Args:
            model_path: Path to model file
            metadata_path: Optional path to metadata file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            model_file = Path(model_path)
            if not model_file.exists():
                logger.error(f"Model file not found: {model_path}")
                return False

            # Load model based on format
            if self.specs.model_format == ModelFormat.PICKLE:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)

            elif self.specs.model_format == ModelFormat.JSON:
                with open(model_path) as f:
                    model_data = json.load(f)
                    self.model = self._convert_json_to_qtable(model_data)

            elif self.specs.model_format == ModelFormat.NUMPY:
                self.model = np.load(model_path, allow_pickle=True)

            else:
                logger.error(f"Unsupported model format: {self.specs.model_format}")
                return False

            # Load metadata if provided
            if metadata_path and Path(metadata_path).exists():
                with open(metadata_path) as f:
                    self.model_metadata = json.load(f)

            # Optimize model for inference
            self._optimize_model()

            logger.info(f"Model loaded successfully from {model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _convert_json_to_qtable(self, model_data: Dict) -> np.ndarray:
        """Convert JSON Q-table to NumPy array"""
        if 'q_table' in model_data:
            # Handle nested dictionary format
            q_table_dict = model_data['q_table']
            if isinstance(q_table_dict, dict):
                # Convert string keys back to tuples/integers
                q_table = {}
                for state_key, action_values in q_table_dict.items():
                    try:
                        # Try to evaluate as tuple
                        state = eval(state_key) if isinstance(state_key, str) else state_key
                        q_table[state] = action_values
                    except (ValueError, SyntaxError, NameError):
                        # Fallback to string key
                        q_table[state_key] = action_values
                return q_table
            else:
                return np.array(q_table_dict)
        else:
            return model_data

    def _optimize_model(self):
        """Optimize loaded model for inference"""
        if self.model is None:
            return

        # Apply quantization if enabled
        if self.specs.quantization:
            self._quantize_model()

        # Pre-compute lookup tables for common operations
        if hasattr(self, 'use_lookup_tables') and self.use_lookup_tables:
            self._create_lookup_tables()

        # Vectorize operations if possible
        if hasattr(self, 'enable_vectorization') and self.enable_vectorization:
            self._vectorize_operations()

    def _quantize_model(self):
        """Apply quantization to reduce model size and improve speed"""
        if isinstance(self.model, np.ndarray):
            # Quantize floating point values to int16
            if self.model.dtype == np.float64 or self.model.dtype == np.float32:
                model_min, model_max = self.model.min(), self.model.max()
                self.quantization_scale = (model_max - model_min) / 65535
                self.quantization_offset = model_min
                self.model = ((self.model - model_min) / self.quantization_scale).astype(np.int16)
                logger.info("Model quantized to int16")

    def _create_lookup_tables(self):
        """Create lookup tables for fast state-action value retrieval"""
        if isinstance(self.model, dict):
            # Convert dictionary to array-based lookup for faster access
            self.state_lookup = {}
            self.action_lookup = {}
            for i, (state, actions) in enumerate(self.model.items()):
                self.state_lookup[state] = i
                self.action_lookup[i] = actions

    def _vectorize_operations(self):
        """Vectorize common operations for batch processing"""
        # Pre-compile common mathematical operations
        if self.specs.batch_processing:
            self.batch_argmax = np.vectorize(np.argmax, signature='(n)->()')

    def infer(self, state: np.ndarray, use_cache: bool = True) -> InferenceMeasurement:
        """
        Perform inference on input state
        
        Args:
            state: Input state vector
            use_cache: Whether to use inference cache
            
        Returns:
            InferenceMeasurement with results and performance metrics
        """
        start_time = time.time()
        memory_before = self._get_memory_usage()

        # Check cache first
        cache_hit = False
        if use_cache and self.specs.cache_size > 0:
            state_key = tuple(state) if isinstance(state, np.ndarray) else state
            if state_key in self.inference_cache:
                cached_result = self.inference_cache[state_key]
                cache_hit = True
                self.cache_hits += 1
                action = cached_result['action']
                confidence = cached_result['confidence']

        if not cache_hit:
            # Perform actual inference
            action, confidence = self._execute_inference(state)

            # Cache result if enabled
            if use_cache and self.specs.cache_size > 0:
                state_key = tuple(state) if isinstance(state, np.ndarray) else state
                if len(self.inference_cache) >= self.specs.cache_size:
                    # Remove oldest entry (simple FIFO)
                    oldest_key = next(iter(self.inference_cache))
                    del self.inference_cache[oldest_key]

                self.inference_cache[state_key] = {
                    'action': action,
                    'confidence': confidence
                }

        # Calculate performance metrics
        end_time = time.time()
        inference_time_ms = (end_time - start_time) * 1000
        memory_after = self._get_memory_usage()
        memory_usage_mb = memory_after - memory_before
        cpu_usage_pct = self._get_cpu_usage()

        # Update statistics
        self.total_inferences += 1
        self.average_inference_time = (
            (self.average_inference_time * (self.total_inferences - 1) + inference_time_ms) /
            self.total_inferences
        )
        self.peak_memory_usage = max(self.peak_memory_usage, memory_usage_mb)
        self.last_inference_time = inference_time_ms

        # Check deadline violation
        if inference_time_ms > self.specs.max_inference_time_ms:
            self.deadline_violations += 1
            logger.warning(f"Inference deadline violated: {inference_time_ms:.2f}ms > {self.specs.max_inference_time_ms:.2f}ms")

        measurement = InferenceMeasurement(
            timestamp=end_time,
            input_state=state.copy() if isinstance(state, np.ndarray) else state,
            output_action=action,
            confidence_score=confidence,
            inference_time_ms=inference_time_ms,
            memory_usage_mb=memory_usage_mb,
            cpu_usage_pct=cpu_usage_pct,
            cache_hit=cache_hit,
            model_version=self.model_metadata.get('version', 'unknown')
        )

        # Store performance history (limited size)
        self.performance_history.append(measurement)
        if len(self.performance_history) > 1000:
            self.performance_history.pop(0)

        return measurement

    def _execute_inference(self, state: np.ndarray) -> Tuple[Union[int, np.ndarray], float]:
        """Execute the actual inference based on model type"""
        if self.model is None:
            raise RuntimeError("No model loaded")

        try:
            if isinstance(self.model, dict):
                # Dictionary-based Q-table
                return self._infer_from_qtable_dict(state)
            elif isinstance(self.model, np.ndarray):
                # Array-based Q-table
                return self._infer_from_qtable_array(state)
            else:
                # Try to use model as callable (neural network, etc.)
                return self._infer_from_callable(state)

        except Exception as e:
            logger.error(f"Inference execution failed: {e}")
            return 0, 0.0  # Default action with zero confidence

    def _infer_from_qtable_dict(self, state: np.ndarray) -> Tuple[int, float]:
        """Infer action from dictionary-based Q-table"""
        # Discretize continuous state if necessary
        discrete_state = self._discretize_state(state)
        state_key = tuple(discrete_state) if isinstance(discrete_state, np.ndarray) else discrete_state

        if state_key in self.model:
            q_values = self.model[state_key]
            if isinstance(q_values, dict):
                # Convert to array
                q_values = np.array(list(q_values.values()))

            # Apply epsilon-greedy if enabled
            if self.epsilon_greedy and np.random.random() < self.epsilon:
                action = np.random.randint(len(q_values))
                confidence = 0.1  # Low confidence for random action
            else:
                action = np.argmax(q_values)
                # Calculate confidence based on Q-value spread
                q_max = np.max(q_values)
                q_mean = np.mean(q_values)
                confidence = min(1.0, max(0.0, (q_max - q_mean) / (np.std(q_values) + 1e-8)))

            return int(action), float(confidence)
        else:
            # State not found, return random action with low confidence
            logger.warning(f"State {state_key} not found in Q-table")
            return 0, 0.0

    def _infer_from_qtable_array(self, state: np.ndarray) -> Tuple[int, float]:
        """Infer action from array-based Q-table"""
        # For 2D Q-table, assume first dimension is state, second is action
        if len(self.model.shape) == 2:
            state_idx = self._state_to_index(state)
            if 0 <= state_idx < self.model.shape[0]:
                q_values = self.model[state_idx, :]

                # Dequantize if model was quantized
                if hasattr(self, 'quantization_scale'):
                    q_values = q_values.astype(np.float32) * self.quantization_scale + self.quantization_offset

                action = np.argmax(q_values)
                confidence = self._calculate_confidence(q_values)
                return int(action), float(confidence)

        return 0, 0.0

    def _infer_from_callable(self, state: np.ndarray) -> Tuple[Union[int, np.ndarray], float]:
        """Infer from callable model (neural network, etc.)"""
        try:
            output = self.model(state)
            if isinstance(output, (list, tuple)):
                action = output[0]
                confidence = output[1] if len(output) > 1 else 1.0
            else:
                action = output
                confidence = 1.0

            return action, confidence
        except Exception as e:
            logger.error(f"Callable inference failed: {e}")
            return 0, 0.0

    def _discretize_state(self, state: np.ndarray) -> np.ndarray:
        """Discretize continuous state space"""
        if not isinstance(state, np.ndarray):
            return state

        # Simple uniform binning
        state_min = np.min(state)
        state_max = np.max(state)

        if state_max == state_min:
            return np.zeros_like(state, dtype=int)

        # Normalize to [0, bins-1]
        normalized = (state - state_min) / (state_max - state_min)
        discretized = np.floor(normalized * (self.state_discretization_bins - 1)).astype(int)

        return np.clip(discretized, 0, self.state_discretization_bins - 1)

    def _state_to_index(self, state: np.ndarray) -> int:
        """Convert state vector to single index"""
        if len(state) == 1:
            return int(state[0])

        # Multi-dimensional state: convert to single index
        # Simple approach: hash the state vector
        state_tuple = tuple(self._discretize_state(state))
        return hash(state_tuple) % self.model.shape[0]

    def _calculate_confidence(self, q_values: np.ndarray) -> float:
        """Calculate confidence score from Q-values"""
        if len(q_values) <= 1:
            return 1.0

        # Confidence based on how much the best action dominates
        q_max = np.max(q_values)
        q_second = np.partition(q_values, -2)[-2]

        if q_max == q_second:
            return 0.5  # Ambiguous

        # Normalize difference to [0, 1]
        q_range = np.max(q_values) - np.min(q_values)
        if q_range == 0:
            return 0.5

        confidence = min(1.0, (q_max - q_second) / q_range)
        return confidence

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            import psutil
            return psutil.cpu_percent()
        except ImportError:
            return 0.0

    def batch_infer(self, states: List[np.ndarray]) -> List[InferenceMeasurement]:
        """Perform batch inference on multiple states"""
        if not self.specs.batch_processing:
            # Fall back to individual inferences
            return [self.infer(state) for state in states]

        start_time = time.time()
        results = []

        # Process states in batches for efficiency
        batch_size = min(len(states), 32)  # Configurable batch size

        for i in range(0, len(states), batch_size):
            batch_states = states[i:i+batch_size]
            batch_results = self._process_batch(batch_states)
            results.extend(batch_results)

        total_time = (time.time() - start_time) * 1000
        logger.info(f"Batch inference completed: {len(states)} states in {total_time:.2f}ms")

        return results

    def _process_batch(self, states: List[np.ndarray]) -> List[InferenceMeasurement]:
        """Process a batch of states efficiently"""
        results = []

        if hasattr(self, 'batch_argmax') and isinstance(self.model, np.ndarray):
            # Vectorized processing for array-based models
            state_indices = [self._state_to_index(state) for state in states]
            valid_indices = [idx for idx in state_indices if 0 <= idx < self.model.shape[0]]

            if valid_indices:
                q_values_batch = self.model[valid_indices, :]
                actions_batch = np.argmax(q_values_batch, axis=1)
                confidences_batch = [self._calculate_confidence(q_values) for q_values in q_values_batch]

                for i, (state, action, confidence) in enumerate(zip(states, actions_batch, confidences_batch)):
                    measurement = InferenceMeasurement(
                        timestamp=time.time(),
                        input_state=state.copy(),
                        output_action=int(action),
                        confidence_score=float(confidence),
                        inference_time_ms=0.0,  # Batch timing handled separately
                        memory_usage_mb=0.0,
                        cpu_usage_pct=0.0,
                        cache_hit=False,
                        model_version=self.model_metadata.get('version', 'unknown')
                    )
                    results.append(measurement)
        else:
            # Fall back to individual processing
            results = [self.infer(state, use_cache=False) for state in states]

        return results

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        cache_hit_rate = self.cache_hits / max(1, self.total_inferences)
        deadline_violation_rate = self.deadline_violations / max(1, self.total_inferences)

        recent_inference_times = [m.inference_time_ms for m in self.performance_history[-100:]]

        return {
            'total_inferences': self.total_inferences,
            'average_inference_time_ms': self.average_inference_time,
            'peak_memory_usage_mb': self.peak_memory_usage,
            'cache_hit_rate': cache_hit_rate,
            'deadline_violation_rate': deadline_violation_rate,
            'recent_min_time_ms': min(recent_inference_times) if recent_inference_times else 0,
            'recent_max_time_ms': max(recent_inference_times) if recent_inference_times else 0,
            'recent_std_time_ms': np.std(recent_inference_times) if recent_inference_times else 0,
            'model_format': self.specs.model_format.value,
            'optimization_level': self.specs.optimization_level,
            'quantization_enabled': self.specs.quantization,
            'batch_processing_enabled': self.specs.batch_processing
        }

    def get_power_consumption(self) -> float:
        """Get estimated power consumption during inference"""
        base_power = self.specs.power_consumption

        # Scale power based on current CPU usage and optimization level
        cpu_factor = max(0.1, self._get_cpu_usage() / 100.0)
        optimization_factor = 1.0 - (self.specs.optimization_level * 0.1)  # Optimizations reduce power

        return base_power * cpu_factor * optimization_factor

    def get_cost_analysis(self) -> Dict[str, float]:
        """Get comprehensive cost analysis for inference engine"""
        initial_cost = self.specs.cost

        # Operating cost based on power consumption
        power_cost_per_hour = self.get_power_consumption() * 0.15 / 1000  # $0.15/kWh

        # Compute cost based on inference load
        compute_cost_per_inference = 0.0001  # $0.0001 per inference (cloud computing analogy)
        compute_cost_per_hour = compute_cost_per_inference * (3600 / max(0.001, self.average_inference_time / 1000))

        # Storage cost for models and cache
        storage_cost_per_hour = (self.specs.storage_mb / 1024) * 0.001  # $0.001/GB/hour

        total_cost_per_hour = power_cost_per_hour + compute_cost_per_hour + storage_cost_per_hour

        return {
            'initial_cost': initial_cost,
            'power_cost_per_hour': power_cost_per_hour,
            'compute_cost_per_hour': compute_cost_per_hour,
            'storage_cost_per_hour': storage_cost_per_hour,
            'total_cost_per_hour': total_cost_per_hour,
            'cost_per_inference': total_cost_per_hour * (self.average_inference_time / 1000) / 3600
        }


def create_standard_inference_engines() -> Dict[str, ModelInferenceEngine]:
    """Create standard inference engine configurations"""

    # High-performance engine for real-time control
    high_performance_specs = InferenceSpecs(
        model_format=ModelFormat.NUMPY,
        max_inference_time_ms=1.0,  # 1ms deadline
        memory_limit_mb=512.0,
        cache_size=1000,
        batch_processing=True,
        quantization=True,
        optimization_level=2,
        power_consumption=5.0,  # W
        cost=500.0,  # USD
        cpu_cores=4,
        ram_mb=1024.0,
        storage_mb=128.0,
        temperature_range=(-10, 70)
    )

    # Low-power engine for embedded systems
    low_power_specs = InferenceSpecs(
        model_format=ModelFormat.JSON,
        max_inference_time_ms=10.0,  # 10ms deadline
        memory_limit_mb=64.0,
        cache_size=100,
        batch_processing=False,
        quantization=True,
        optimization_level=1,
        power_consumption=0.5,  # W
        cost=100.0,  # USD
        cpu_cores=1,
        ram_mb=128.0,
        storage_mb=32.0,
        temperature_range=(-40, 85)
    )

    # Balanced engine for general use
    balanced_specs = InferenceSpecs(
        model_format=ModelFormat.PICKLE,
        max_inference_time_ms=5.0,  # 5ms deadline
        memory_limit_mb=256.0,
        cache_size=500,
        batch_processing=True,
        quantization=False,
        optimization_level=1,
        power_consumption=2.0,  # W
        cost=200.0,  # USD
        cpu_cores=2,
        ram_mb=512.0,
        storage_mb=64.0,
        temperature_range=(-20, 60)
    )

    engines = {
        'high_performance': ModelInferenceEngine(high_performance_specs),
        'low_power': ModelInferenceEngine(low_power_specs),
        'balanced': ModelInferenceEngine(balanced_specs)
    }

    return engines


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create inference engines
    engines = create_standard_inference_engines()

    # Test high-performance engine
    hp_engine = engines['high_performance']

    print(f"Testing {hp_engine.specs.model_format.value} inference engine")
    print(f"Max inference time: {hp_engine.specs.max_inference_time_ms}ms")
    print(f"Power consumption: {hp_engine.specs.power_consumption}W")

    # Create dummy Q-table for testing
    dummy_qtable = np.random.rand(100, 5)  # 100 states, 5 actions
    hp_engine.model = dummy_qtable

    # Test inference
    test_state = np.array([0.5, 0.3, 0.8, 0.1])

    print("\nRunning inference tests:")
    for i in range(10):
        measurement = hp_engine.infer(test_state)
        print(f"Inference {i+1}: Action={measurement.output_action}, "
              f"Confidence={measurement.confidence_score:.3f}, "
              f"Time={measurement.inference_time_ms:.3f}ms, "
              f"Cache={measurement.cache_hit}")

    # Test batch inference
    test_states = [np.random.rand(4) for _ in range(20)]
    batch_results = hp_engine.batch_infer(test_states)
    print(f"\nBatch inference: {len(batch_results)} states processed")

    # Performance statistics
    stats = hp_engine.get_performance_stats()
    print("\nPerformance Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Cost analysis
    cost_analysis = hp_engine.get_cost_analysis()
    print("\nCost Analysis:")
    for key, value in cost_analysis.items():
        print(f"  {key}: ${value:.6f}")
