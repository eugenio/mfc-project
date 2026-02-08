"""Tests for controller_models/model_inference.py - coverage target 98%+."""
import sys
import os
import json
import pickle
import tempfile
import time

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from controller_models.model_inference import (
    ModelInferenceEngine, InferenceSpecs, ModelFormat, InferenceMeasurement,
    create_standard_inference_engines,
)


def make_specs(**overrides):
    defaults = dict(model_format=ModelFormat.NUMPY, max_inference_time_ms=10.0,
        memory_limit_mb=256.0, cache_size=100, batch_processing=False,
        quantization=False, optimization_level=0, power_consumption=2.0,
        cost=100.0, cpu_cores=2, ram_mb=512.0, storage_mb=64.0, temperature_range=(-20,60))
    defaults.update(overrides)
    return InferenceSpecs(**defaults)


@pytest.fixture
def engine():
    return ModelInferenceEngine(make_specs())


class TestLoadModelFormats:
    def test_load_pickle(self, tmp_path):
        specs = make_specs(model_format=ModelFormat.PICKLE)
        eng = ModelInferenceEngine(specs)
        mf = tmp_path / "m.pkl"
        with open(mf, "wb") as f: pickle.dump(np.random.rand(10, 3), f)
        assert eng.load_model(str(mf)) is True

    def test_load_json_qtable(self, tmp_path):
        specs = make_specs(model_format=ModelFormat.JSON)
        eng = ModelInferenceEngine(specs)
        mf = tmp_path / "m.json"
        data = {"q_table": {"(0, 1)": [1.0, 2.0, 3.0], "(1, 0)": [4.0, 5.0, 6.0]}}
        with open(mf, "w") as f: json.dump(data, f)
        assert eng.load_model(str(mf)) is True

    def test_load_numpy(self, tmp_path):
        specs = make_specs(model_format=ModelFormat.NUMPY)
        eng = ModelInferenceEngine(specs)
        mf = tmp_path / "m.npy"
        np.save(mf, np.random.rand(10, 3))
        assert eng.load_model(str(mf)) is True

    def test_load_unsupported_format(self, tmp_path):
        specs = make_specs(model_format=ModelFormat.ONNX)
        eng = ModelInferenceEngine(specs)
        mf = tmp_path / "m.onnx"
        mf.write_text("fake")
        assert eng.load_model(str(mf)) is False

    def test_load_nonexistent(self, engine):
        assert engine.load_model("/nonexistent/path.npy") is False

    def test_load_exception(self, tmp_path):
        specs = make_specs(model_format=ModelFormat.PICKLE)
        eng = ModelInferenceEngine(specs)
        mf = tmp_path / "bad.pkl"
        mf.write_text("not pickle")
        assert eng.load_model(str(mf)) is False

    def test_load_with_metadata(self, tmp_path):
        specs = make_specs(model_format=ModelFormat.NUMPY)
        eng = ModelInferenceEngine(specs)
        mf = tmp_path / "m.npy"; np.save(mf, np.random.rand(5, 3))
        meta = tmp_path / "meta.json"
        with open(meta, "w") as f: json.dump({"version": "1.0"}, f)
        assert eng.load_model(str(mf), str(meta)) is True
        assert eng.model_metadata["version"] == "1.0"


class TestConvertJsonQtable:
    def test_json_dict_format(self, engine):
        data = {"q_table": {"(0,)": [1, 2], "bad_key": [3, 4]}}
        result = engine._convert_json_to_qtable(data)
        assert isinstance(result, dict)

    def test_json_array_format(self, engine):
        data = {"q_table": [[1, 2], [3, 4]]}
        result = engine._convert_json_to_qtable(data)
        assert isinstance(result, np.ndarray)

    def test_json_no_qtable_key(self, engine):
        data = {"other": "data"}
        result = engine._convert_json_to_qtable(data)
        assert result == data


class TestQuantizeModel:
    def test_quantize_float_model(self):
        specs = make_specs(quantization=True)
        eng = ModelInferenceEngine(specs)
        eng.model = np.random.rand(10, 5).astype(np.float64)
        eng._quantize_model()
        assert eng.model.dtype == np.int16
        assert hasattr(eng, "quantization_scale")


class TestCreateLookupTables:
    def test_create_lookup_tables(self):
        specs = make_specs(optimization_level=2)
        eng = ModelInferenceEngine(specs)
        eng.model = {(0,): [1, 2], (1,): [3, 4]}
        eng._create_lookup_tables()
        assert hasattr(eng, "state_lookup")


class TestVectorizeOperations:
    def test_vectorize_with_batch(self):
        specs = make_specs(optimization_level=2, batch_processing=True)
        eng = ModelInferenceEngine(specs)
        eng._vectorize_operations()
        assert hasattr(eng, "batch_argmax")


class TestInference:
    def test_infer_with_cache_hit(self, engine):
        engine.model = np.random.rand(100, 5)
        state = np.array([0.5, 0.3])
        m1 = engine.infer(state, use_cache=True)
        m2 = engine.infer(state, use_cache=True)
        assert m2.cache_hit is True

    def test_infer_cache_eviction(self):
        specs = make_specs(cache_size=2)
        eng = ModelInferenceEngine(specs)
        eng.model = np.random.rand(100, 5)
        eng.infer(np.array([0.1]), use_cache=True)
        eng.infer(np.array([0.2]), use_cache=True)
        eng.infer(np.array([0.3]), use_cache=True)
        assert len(eng.inference_cache) <= 2

    def test_infer_deadline_violation(self):
        specs = make_specs(max_inference_time_ms=0.0001)
        eng = ModelInferenceEngine(specs)
        eng.model = np.random.rand(100, 5)
        eng.infer(np.array([0.5]))
        assert eng.deadline_violations >= 1


class TestExecuteInference:
    def test_no_model_raises(self, engine):
        with pytest.raises(RuntimeError, match="No model"):
            engine._execute_inference(np.array([0.5]))

    def test_dict_model(self, engine):
        engine.model = {(0, 0): [1.0, 2.0, 3.0]}
        a, c = engine._execute_inference(np.array([0.0, 0.0]))
        assert isinstance(a, int)

    def test_array_model(self, engine):
        engine.model = np.random.rand(100, 5)
        a, c = engine._execute_inference(np.array([50]))
        assert isinstance(a, int)

    def test_callable_model(self, engine):
        engine.model = lambda x: (2, 0.9)
        a, c = engine._execute_inference(np.array([0.5]))
        assert a == 2

    def test_callable_model_single_output(self, engine):
        engine.model = lambda x: 3
        a, c = engine._execute_inference(np.array([0.5]))
        assert a == 3

    def test_callable_model_exception(self, engine):
        engine.model = lambda x: (_ for _ in ()).throw(RuntimeError("fail"))
        a, c = engine._execute_inference(np.array([0.5]))
        assert a == 0 and c == 0.0

    def test_inference_exception_fallback(self, engine):
        engine.model = np.random.rand(100, 5)
        with patch.object(engine, "_infer_from_qtable_array", side_effect=ValueError("x")):
            a, c = engine._execute_inference(np.array([50]))
            assert a == 0


class TestInferFromQtableDict:
    def test_state_not_found(self, engine):
        engine.model = {}
        a, c = engine._infer_from_qtable_dict(np.array([99.0, 99.0]))
        assert a == 0 and c == 0.0

    def test_epsilon_greedy(self, engine):
        engine.model = {(0, 0): [1.0, 2.0, 3.0]}
        engine.epsilon_greedy = True; engine.epsilon = 1.0
        np.random.seed(42)
        a, c = engine._infer_from_qtable_dict(np.array([0.0, 0.0]))
        assert c == 0.1

    def test_dict_values_in_qtable(self, engine):
        engine.model = {(0, 0): {"a": 1.0, "b": 5.0, "c": 2.0}}
        a, c = engine._infer_from_qtable_dict(np.array([0.0, 0.0]))
        assert a == 1


class TestInferFromQtableArray:
    def test_quantized_model(self):
        specs = make_specs(quantization=True)
        eng = ModelInferenceEngine(specs)
        eng.model = np.random.rand(100, 5).astype(np.float64)
        eng._quantize_model()
        a, c = eng._infer_from_qtable_array(np.array([50]))
        assert isinstance(a, int)

    def test_out_of_range_index(self, engine):
        engine.model = np.random.rand(10, 5)
        a, c = engine._infer_from_qtable_array(np.array([-999]))
        # hash may wrap around, so just check it returns


class TestCalculateConfidence:
    def test_single_value(self, engine):
        assert engine._calculate_confidence(np.array([5.0])) == 1.0

    def test_equal_values(self, engine):
        assert engine._calculate_confidence(np.array([5.0, 5.0])) == 0.5

    def test_zero_range(self, engine):
        assert engine._calculate_confidence(np.array([3.0, 3.0, 3.0])) == 0.5


class TestBatchInfer:
    def test_batch_no_batch_processing(self, engine):
        engine.model = np.random.rand(100, 5)
        states = [np.array([float(i)]) for i in range(3)]
        results = engine.batch_infer(states)
        assert len(results) == 3

    def test_batch_with_batch_processing(self):
        specs = make_specs(batch_processing=True, optimization_level=2)
        eng = ModelInferenceEngine(specs)
        eng.model = np.random.rand(100, 5)
        eng._vectorize_operations()
        states = [np.array([float(i)]) for i in range(5)]
        results = eng.batch_infer(states)
        assert len(results) == 5


class TestProcessBatch:
    def test_process_batch_vectorized(self):
        specs = make_specs(batch_processing=True, optimization_level=2)
        eng = ModelInferenceEngine(specs)
        eng.model = np.random.rand(100, 5)
        eng._vectorize_operations()
        states = [np.array([float(i)]) for i in range(3)]
        results = eng._process_batch(states)
        assert len(results) >= 1

    def test_process_batch_fallback(self, engine):
        engine.model = {(0,): [1, 2]}
        states = [np.array([0.0])]
        results = engine._process_batch(states)
        assert len(results) == 1


class TestCostAndPower:
    def test_get_power_consumption(self, engine):
        p = engine.get_power_consumption()
        assert p >= 0

    def test_get_cost_analysis(self, engine):
        ca = engine.get_cost_analysis()
        assert "initial_cost" in ca
        assert "total_cost_per_hour" in ca


class TestPerformanceStats:
    def test_get_performance_stats(self, engine):
        engine.model = np.random.rand(100, 5)
        engine.infer(np.array([0.5]))
        stats = engine.get_performance_stats()
        assert stats["total_inferences"] >= 1


class TestCreateStandard:
    def test_create_standard_engines(self):
        engines = create_standard_inference_engines()
        assert "high_performance" in engines
        assert "low_power" in engines
        assert "balanced" in engines
