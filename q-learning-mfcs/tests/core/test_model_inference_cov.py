"""Tests for model_inference.py - 98%+ coverage target.

Covers ModelInferenceEngine, InferenceSpecs, ModelFormat,
InferenceMeasurement, and helper functions.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from controller_models.model_inference import (
    InferenceMeasurement,
    InferenceSpecs,
    ModelFormat,
    ModelInferenceEngine,
    create_standard_inference_engines,
)


def _make_specs(fmt=ModelFormat.NUMPY, opt=1, quant=False, batch=False, cache=100):
    return InferenceSpecs(
        model_format=fmt, max_inference_time_ms=10.0,
        memory_limit_mb=128.0, cache_size=cache,
        batch_processing=batch, quantization=quant,
        optimization_level=opt, power_consumption=2.0,
        cost=200.0, cpu_cores=2, ram_mb=256.0,
        storage_mb=64.0, temperature_range=(-20, 60),
    )


class TestEnums:
    def test_model_format_values(self):
        assert ModelFormat.PICKLE.value == "pickle"
        assert ModelFormat.JSON.value == "json"
        assert ModelFormat.NUMPY.value == "numpy"
        assert ModelFormat.ONNX.value == "onnx"
        assert ModelFormat.TFLITE.value == "tflite"


class TestInferenceSpecs:
    def test_create_specs(self):
        specs = _make_specs()
        assert specs.model_format == ModelFormat.NUMPY
        assert specs.cpu_cores == 2


class TestInferenceMeasurement:
    def test_create_measurement(self):
        m = InferenceMeasurement(
            timestamp=1000.0, input_state=np.array([1, 2]),
            output_action=0, confidence_score=0.8,
            inference_time_ms=0.5, memory_usage_mb=10.0,
            cpu_usage_pct=20.0, cache_hit=False,
            model_version="v1",
        )
        assert m.output_action == 0
        assert m.confidence_score == 0.8


class TestModelInferenceEngineInit:
    def test_init_opt_0(self):
        specs = _make_specs(opt=0)
        engine = ModelInferenceEngine(specs)
        assert not hasattr(engine, "enable_vectorization")

    def test_init_opt_1(self):
        specs = _make_specs(opt=1)
        engine = ModelInferenceEngine(specs)
        assert engine.enable_vectorization is True
        assert engine.precompute_common_operations is True

    def test_init_opt_2(self):
        specs = _make_specs(opt=2)
        engine = ModelInferenceEngine(specs)
        assert engine.enable_jit_compilation is True
        assert engine.use_lookup_tables is True
        assert engine.parallel_processing is True


class TestLoadModel:
    def test_load_numpy(self):
        specs = _make_specs(fmt=ModelFormat.NUMPY)
        engine = ModelInferenceEngine(specs)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            np.save(f.name, np.random.rand(10, 5))
            result = engine.load_model(f.name)
        assert result is True

    def test_load_json(self):
        specs = _make_specs(fmt=ModelFormat.JSON, opt=2)
        engine = ModelInferenceEngine(specs)
        data = {"q_table": {"(0, 1)": [1.0, 2.0, 3.0], "(1, 0)": [4.0, 5.0, 6.0]}}
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False,
        ) as f:
            json.dump(data, f)
            f.flush()
            result = engine.load_model(f.name)
        assert result is True

    def test_load_json_array_qtable(self):
        specs = _make_specs(fmt=ModelFormat.JSON)
        engine = ModelInferenceEngine(specs)
        data = {"q_table": [[1, 2], [3, 4]]}
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False,
        ) as f:
            json.dump(data, f)
            f.flush()
            result = engine.load_model(f.name)
        assert result is True

    def test_load_json_no_qtable(self):
        specs = _make_specs(fmt=ModelFormat.JSON)
        engine = ModelInferenceEngine(specs)
        data = {"other": [1, 2, 3]}
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False,
        ) as f:
            json.dump(data, f)
            f.flush()
            result = engine.load_model(f.name)
        assert result is True

    def test_load_json_bad_key(self):
        specs = _make_specs(fmt=ModelFormat.JSON, opt=2)
        engine = ModelInferenceEngine(specs)
        data = {"q_table": {"not_valid_python!!!": [1.0, 2.0]}}
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False,
        ) as f:
            json.dump(data, f)
            f.flush()
            result = engine.load_model(f.name)
        assert result is True

    def test_load_pickle(self):
        import pickle
        specs = _make_specs(fmt=ModelFormat.PICKLE)
        engine = ModelInferenceEngine(specs)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(np.random.rand(10, 5), f)
            f.flush()
            result = engine.load_model(f.name)
        assert result is True

    def test_load_with_metadata(self):
        specs = _make_specs(fmt=ModelFormat.NUMPY)
        engine = ModelInferenceEngine(specs)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            np.save(f.name, np.random.rand(10, 5))
            model_path = f.name
        meta = {"version": "v2"}
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False,
        ) as f:
            json.dump(meta, f)
            f.flush()
            meta_path = f.name
        result = engine.load_model(model_path, meta_path)
        assert result is True
        assert engine.model_metadata["version"] == "v2"

    def test_load_nonexistent(self):
        specs = _make_specs()
        engine = ModelInferenceEngine(specs)
        assert engine.load_model("/nonexistent/path.npy") is False

    def test_load_unsupported_format(self):
        specs = _make_specs(fmt=ModelFormat.ONNX)
        engine = ModelInferenceEngine(specs)
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(b"dummy")
            f.flush()
            result = engine.load_model(f.name)
        assert result is False

    def test_load_corrupt_file(self):
        specs = _make_specs(fmt=ModelFormat.NUMPY)
        engine = ModelInferenceEngine(specs)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            f.write(b"not_a_numpy_file")
            f.flush()
            result = engine.load_model(f.name)
        assert result is False


class TestQuantization:
    def test_quantize_float64(self):
        specs = _make_specs(quant=True)
        engine = ModelInferenceEngine(specs)
        engine.model = np.random.rand(10, 5).astype(np.float64)
        engine._quantize_model()
        assert engine.model.dtype == np.int16
        assert hasattr(engine, "quantization_scale")

    def test_quantize_non_array(self):
        specs = _make_specs(quant=True)
        engine = ModelInferenceEngine(specs)
        engine.model = {"key": "value"}
        engine._quantize_model()
        # Should not change non-array models
        assert isinstance(engine.model, dict)


class TestOptimizeModel:
    def test_optimize_with_lookup_tables(self):
        specs = _make_specs(opt=2)
        engine = ModelInferenceEngine(specs)
        engine.model = {(0, 1): [1.0, 2.0], (1, 0): [3.0, 4.0]}
        engine._optimize_model()
        assert hasattr(engine, "state_lookup")

    def test_optimize_vectorize_with_batch(self):
        specs = _make_specs(opt=1, batch=True)
        engine = ModelInferenceEngine(specs)
        engine.model = np.random.rand(10, 5)
        engine._optimize_model()
        assert hasattr(engine, "batch_argmax")

    def test_optimize_none_model(self):
        specs = _make_specs()
        engine = ModelInferenceEngine(specs)
        engine.model = None
        engine._optimize_model()  # Should not raise


class TestInfer:
    def test_infer_array_model(self):
        specs = _make_specs(cache=10)
        engine = ModelInferenceEngine(specs)
        engine.model = np.random.rand(100, 5)
        state = np.array([0.5, 0.3])
        result = engine.infer(state)
        assert isinstance(result, InferenceMeasurement)
        assert 0 <= result.output_action < 5

    def test_infer_dict_model(self):
        specs = _make_specs(cache=10)
        engine = ModelInferenceEngine(specs)
        engine.model = {(0, 0): [1.0, 2.0, 3.0]}
        state = np.array([0.0, 0.0])
        result = engine.infer(state)
        assert isinstance(result, InferenceMeasurement)

    def test_infer_cache_hit(self):
        specs = _make_specs(cache=10)
        engine = ModelInferenceEngine(specs)
        engine.model = np.random.rand(100, 5)
        state = np.array([0.5])
        engine.infer(state)
        result = engine.infer(state)
        assert result.cache_hit is True
        assert engine.cache_hits >= 1

    def test_infer_cache_eviction(self):
        specs = _make_specs(cache=2)
        engine = ModelInferenceEngine(specs)
        engine.model = np.random.rand(100, 5)
        engine.infer(np.array([0.1]))
        engine.infer(np.array([0.2]))
        engine.infer(np.array([0.3]))
        assert len(engine.inference_cache) <= 2

    def test_infer_no_cache(self):
        specs = _make_specs(cache=0)
        engine = ModelInferenceEngine(specs)
        engine.model = np.random.rand(100, 5)
        result = engine.infer(np.array([0.5]), use_cache=False)
        assert result.cache_hit is False

    def test_infer_no_model(self):
        specs = _make_specs()
        engine = ModelInferenceEngine(specs)
        with pytest.raises(RuntimeError, match="No model loaded"):
            engine.infer(np.array([0.5]))

    def test_infer_performance_history_limit(self):
        specs = _make_specs(cache=0)
        engine = ModelInferenceEngine(specs)
        engine.model = np.random.rand(100, 5)
        for i in range(1005):
            engine.infer(np.array([float(i)]), use_cache=False)
        assert len(engine.performance_history) <= 1000

    def test_infer_epsilon_greedy(self):
        specs = _make_specs(cache=0)
        engine = ModelInferenceEngine(specs)
        engine.model = {(0,): [1.0, 2.0, 3.0]}
        engine.epsilon_greedy = True
        engine.epsilon = 1.0
        with patch("numpy.random.random", return_value=0.0):
            result = engine.infer(np.array([0.0]), use_cache=False)
            assert result.confidence_score == 0.1

    def test_infer_dict_state_not_found(self):
        specs = _make_specs(cache=0)
        engine = ModelInferenceEngine(specs)
        # Use a single unique key that the discretization won't match
        engine.model = {(99, 99): [1.0, 2.0, 3.0]}
        # State [0.5] discretizes to (0,) which won't be in the dict
        result = engine.infer(np.array([0.5]), use_cache=False)
        assert result.confidence_score == 0.0

    def test_infer_dict_with_dict_values(self):
        specs = _make_specs(cache=0)
        engine = ModelInferenceEngine(specs)
        engine.model = {(0,): {0: 1.0, 1: 5.0, 2: 3.0}}
        result = engine.infer(np.array([0.0]), use_cache=False)
        assert result.output_action == 1

    def test_infer_callable_model(self):
        specs = _make_specs(cache=0)
        engine = ModelInferenceEngine(specs)
        engine.model = lambda x: (2, 0.9)
        result = engine.infer(np.array([0.5]), use_cache=False)
        assert result.output_action == 2
        assert result.confidence_score == 0.9

    def test_infer_callable_single_output(self):
        specs = _make_specs(cache=0)
        engine = ModelInferenceEngine(specs)
        engine.model = lambda x: 3
        result = engine.infer(np.array([0.5]), use_cache=False)
        assert result.output_action == 3

    def test_infer_callable_raises(self):
        specs = _make_specs(cache=0)
        engine = ModelInferenceEngine(specs)
        engine.model = lambda x: (_ for _ in ()).throw(RuntimeError("fail"))
        result = engine.infer(np.array([0.5]), use_cache=False)
        assert result.output_action == 0
        assert result.confidence_score == 0.0

    def test_infer_quantized_array(self):
        specs = _make_specs(quant=True, cache=0)
        engine = ModelInferenceEngine(specs)
        model = np.random.rand(100, 5)
        engine.model = model
        engine._quantize_model()
        state = np.array([0.5])
        result = engine.infer(state, use_cache=False)
        assert isinstance(result, InferenceMeasurement)


class TestDiscreteAndIndex:
    def test_discretize_non_array(self):
        specs = _make_specs()
        engine = ModelInferenceEngine(specs)
        result = engine._discretize_state("string_state")
        assert result == "string_state"

    def test_discretize_uniform(self):
        specs = _make_specs()
        engine = ModelInferenceEngine(specs)
        result = engine._discretize_state(np.array([5.0, 5.0, 5.0]))
        assert np.all(result == 0)

    def test_state_to_index_1d(self):
        specs = _make_specs()
        engine = ModelInferenceEngine(specs)
        engine.model = np.random.rand(100, 5)
        idx = engine._state_to_index(np.array([3]))
        assert idx == 3

    def test_state_to_index_multidim(self):
        specs = _make_specs()
        engine = ModelInferenceEngine(specs)
        engine.model = np.random.rand(100, 5)
        idx = engine._state_to_index(np.array([0.5, 0.3, 0.8]))
        assert 0 <= idx < 100


class TestCalculateConfidence:
    def test_single_value(self):
        specs = _make_specs()
        engine = ModelInferenceEngine(specs)
        assert engine._calculate_confidence(np.array([5.0])) == 1.0

    def test_tied_values(self):
        specs = _make_specs()
        engine = ModelInferenceEngine(specs)
        assert engine._calculate_confidence(np.array([5.0, 5.0])) == 0.5

    def test_clear_winner(self):
        specs = _make_specs()
        engine = ModelInferenceEngine(specs)
        c = engine._calculate_confidence(np.array([1.0, 10.0, 1.0]))
        assert c > 0.5

    def test_all_zero(self):
        specs = _make_specs()
        engine = ModelInferenceEngine(specs)
        assert engine._calculate_confidence(np.array([0.0, 0.0])) == 0.5


class TestBatchInfer:
    def test_batch_no_batch_flag(self):
        specs = _make_specs(batch=False, cache=0)
        engine = ModelInferenceEngine(specs)
        engine.model = np.random.rand(100, 5)
        states = [np.random.rand(2) for _ in range(5)]
        results = engine.batch_infer(states)
        assert len(results) == 5

    def test_batch_with_array_model(self):
        specs = _make_specs(batch=True, opt=1, cache=0)
        engine = ModelInferenceEngine(specs)
        engine.model = np.random.rand(100, 5)
        engine._vectorize_operations()
        states = [np.random.rand(2) for _ in range(10)]
        results = engine.batch_infer(states)
        assert len(results) == 10

    def test_batch_with_dict_model(self):
        specs = _make_specs(batch=True, cache=0)
        engine = ModelInferenceEngine(specs)
        engine.model = {(0,): [1.0, 2.0, 3.0]}
        states = [np.array([0.0]) for _ in range(3)]
        results = engine.batch_infer(states)
        assert len(results) == 3


class TestPerformanceStats:
    def test_get_stats_empty(self):
        specs = _make_specs()
        engine = ModelInferenceEngine(specs)
        stats = engine.get_performance_stats()
        assert stats["total_inferences"] == 0

    def test_get_stats_with_history(self):
        specs = _make_specs(cache=0)
        engine = ModelInferenceEngine(specs)
        engine.model = np.random.rand(100, 5)
        for i in range(5):
            engine.infer(np.array([float(i)]), use_cache=False)
        stats = engine.get_performance_stats()
        assert stats["total_inferences"] == 5

    def test_get_power_consumption(self):
        specs = _make_specs()
        engine = ModelInferenceEngine(specs)
        power = engine.get_power_consumption()
        assert power >= 0.0

    def test_get_cost_analysis(self):
        specs = _make_specs()
        engine = ModelInferenceEngine(specs)
        cost = engine.get_cost_analysis()
        assert "initial_cost" in cost
        assert "total_cost_per_hour" in cost


class TestMemoryAndCPU:
    def test_get_memory_no_psutil(self):
        specs = _make_specs()
        engine = ModelInferenceEngine(specs)
        with patch.dict("sys.modules", {"psutil": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                mem = engine._get_memory_usage()
        assert mem >= 0.0

    def test_get_cpu_no_psutil(self):
        specs = _make_specs()
        engine = ModelInferenceEngine(specs)
        with patch.dict("sys.modules", {"psutil": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                cpu = engine._get_cpu_usage()
        assert cpu >= 0.0


class TestEdgeCases:
    def test_infer_array_out_of_bounds(self):
        specs = _make_specs(cache=0)
        engine = ModelInferenceEngine(specs)
        engine.model = np.random.rand(5, 3)
        # 1D state with large value triggers state_to_index returning >5
        result = engine.infer(np.array([1000.0]), use_cache=False)
        # Should return (0, 0.0) for out-of-bounds
        assert isinstance(result, InferenceMeasurement)

    def test_infer_callable_list_output(self):
        specs = _make_specs(cache=0)
        engine = ModelInferenceEngine(specs)
        engine.model = lambda x: [5]
        result = engine.infer(np.array([0.5]), use_cache=False)
        assert result.output_action == 5
        assert result.confidence_score == 1.0

    def test_optimize_lookup_tables_only(self):
        specs = _make_specs(opt=2, quant=False)
        engine = ModelInferenceEngine(specs)
        engine.model = {(0,): [1.0], (1,): [2.0]}
        engine._create_lookup_tables()
        assert hasattr(engine, "state_lookup")
        assert len(engine.state_lookup) == 2


class TestCreateStandardEngines:
    def test_creates_three_configs(self):
        engines = create_standard_inference_engines()
        assert "high_performance" in engines
        assert "low_power" in engines
        assert "balanced" in engines

    def test_all_are_engines(self):
        engines = create_standard_inference_engines()
        for engine in engines.values():
            assert isinstance(engine, ModelInferenceEngine)
