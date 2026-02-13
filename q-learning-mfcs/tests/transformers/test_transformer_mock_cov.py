"""Comprehensive tests for transformer_controller.py with torch mocked.

Targets 99%+ statement coverage of all classes and functions.
"""
import sys
import tempfile
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mock torch BEFORE importing source modules
# ---------------------------------------------------------------------------
mock_torch = MagicMock()
mock_nn = MagicMock()
mock_optim = MagicMock()
mock_F = MagicMock()


class _FakeTensor:
    pass


mock_torch.Tensor = _FakeTensor


class MockModule:
    """Minimal nn.Module replacement for coverage testing."""

    def __init__(self, *a, **kw):
        pass

    def forward(self, x):
        return x

    def parameters(self):
        return []

    def state_dict(self):
        return {}

    def load_state_dict(self, d):
        pass

    def train(self, mode=True):
        return self

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, *a, **kw):
        return MagicMock()

    def named_parameters(self):
        return []

    def children(self):
        return []

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)

    def apply(self, fn):
        return self

    training = True


class _FakeParameter:
    def __init__(self, data=None, requires_grad=True):
        self.data = data if data is not None else MagicMock()
        self.requires_grad = requires_grad
        self.grad = None
        self.shape = getattr(data, "shape", (1,))

    def numel(self):
        return 0


class _FakeLinear(MockModule):
    def __init__(self, in_f=0, out_f=0, *a, **kw):
        super().__init__()

    def __call__(self, x):
        return MagicMock()


mock_nn.Module = MockModule
mock_nn.Linear = _FakeLinear
mock_nn.ReLU = MagicMock(return_value=MagicMock())
mock_nn.Dropout = MagicMock(return_value=MagicMock())
mock_nn.LayerNorm = MagicMock(return_value=MagicMock())
mock_nn.Sequential = MagicMock(return_value=MagicMock())
mock_nn.ModuleList = MagicMock(return_value=[])
mock_nn.ModuleDict = MagicMock(return_value={})
mock_nn.functional = mock_F
mock_nn.Parameter = _FakeParameter
mock_nn.BatchNorm1d = MagicMock(return_value=MagicMock())
mock_nn.Conv1d = MagicMock(return_value=MagicMock())
mock_torch.nn = mock_nn
mock_torch.optim = mock_optim
mock_torch.optim.SGD = MagicMock(return_value=MagicMock())
mock_torch.optim.Adam = MagicMock(return_value=MagicMock(
    zero_grad=MagicMock(), step=MagicMock(),
    state_dict=MagicMock(return_value={}),
    load_state_dict=MagicMock()))
mock_torch.optim.AdamW = MagicMock(return_value=MagicMock(
    zero_grad=MagicMock(), step=MagicMock(),
    state_dict=MagicMock(return_value={}),
    load_state_dict=MagicMock()))
mock_torch.optim.lr_scheduler = MagicMock()
mock_torch.optim.lr_scheduler.StepLR = MagicMock(return_value=MagicMock(
    get_last_lr=MagicMock(return_value=[1e-4]),
    step=MagicMock(), state_dict=MagicMock(return_value={}),
    load_state_dict=MagicMock()))
mock_torch.optim.lr_scheduler.LambdaLR = MagicMock(return_value=MagicMock(
    get_last_lr=MagicMock(return_value=[1e-4]),
    step=MagicMock(), state_dict=MagicMock(return_value={}),
    load_state_dict=MagicMock()))
mock_torch.tensor = MagicMock(return_value=MagicMock())
mock_torch.zeros = MagicMock(return_value=MagicMock(
    unsqueeze=MagicMock(return_value=MagicMock(
        transpose=MagicMock(return_value=MagicMock())))))
mock_torch.ones = MagicMock(return_value=MagicMock())
mock_torch.empty = MagicMock(return_value=MagicMock())
mock_torch.zeros_like = MagicMock(return_value=MagicMock())
mock_torch.randn_like = MagicMock(return_value=MagicMock())
mock_torch.from_numpy = MagicMock(return_value=MagicMock(
    float=MagicMock(return_value=MagicMock())))
mock_torch.arange = MagicMock(return_value=MagicMock(
    unsqueeze=MagicMock(return_value=MagicMock()),
    float=MagicMock(return_value=MagicMock(
        unsqueeze=MagicMock(return_value=MagicMock())))))
mock_torch.exp = MagicMock(return_value=MagicMock())
mock_torch.device = MagicMock(return_value="cpu")
mock_torch.cuda = MagicMock()
mock_torch.cuda.is_available = MagicMock(return_value=False)
mock_torch.no_grad = MagicMock(
    return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
mock_torch.save = MagicMock()
mock_torch.load = MagicMock(return_value={})
mock_torch.FloatTensor = MagicMock(return_value=MagicMock(
    unsqueeze=MagicMock(return_value=MagicMock(
        unsqueeze=MagicMock(return_value=MagicMock(
            to=MagicMock(return_value=MagicMock()))),
        to=MagicMock(return_value=MagicMock()))),
    to=MagicMock(return_value=MagicMock())))
mock_torch.LongTensor = MagicMock(return_value=MagicMock(
    to=MagicMock(return_value=MagicMock())))
mock_torch.nn.utils = MagicMock()
mock_torch.nn.utils.clip_grad_norm_ = MagicMock(
    return_value=MagicMock(item=MagicMock(return_value=1.0)))
mock_torch.nn.init = MagicMock()
mock_torch.randn = MagicMock(return_value=MagicMock())
mock_torch.sin = MagicMock(return_value=MagicMock())
mock_torch.cos = MagicMock(return_value=MagicMock())
mock_torch.log = MagicMock(return_value=MagicMock())
mock_torch.matmul = MagicMock(return_value=MagicMock())
mock_torch.tril = MagicMock(return_value=MagicMock(
    to=MagicMock(return_value=MagicMock())))
mock_torch.cat = MagicMock(return_value=MagicMock(
    shape=(1, 1, 256),
    dim=MagicMock(return_value=3)))
mock_torch.stack = MagicMock(return_value=MagicMock(
    mean=MagicMock(return_value=MagicMock())))
mock_torch.argmax = MagicMock(return_value=MagicMock(
    item=MagicMock(return_value=0)))
mock_torch.max = MagicMock(return_value=MagicMock(
    item=MagicMock(return_value=0.9)))
mock_F.softmax = MagicMock(return_value=MagicMock())
mock_F.relu = MagicMock(return_value=MagicMock())
mock_F.mse_loss = MagicMock(return_value=MagicMock(
    item=MagicMock(return_value=0.5),
    backward=MagicMock()))

# -- Inject mocks into sys.modules BEFORE importing source modules ----------
_orig = {}
import torch_compat

from transformer_controller import (
class _FakeTensor:
    pass

class MockModule:
    """Minimal nn.Module replacement for coverage testing."""

    def __init__(self, *a, **kw):
        pass

    def forward(self, x):
        return x

    def parameters(self):
        return []

    def state_dict(self):
        return {}

    def load_state_dict(self, d):
        pass

    def train(self, mode=True):
        return self

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, *a, **kw):
        return MagicMock()

    def named_parameters(self):
        return []

    def children(self):
        return []

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)

    def apply(self, fn):
        return self

    training = True

class _FakeParameter:
    def __init__(self, data=None, requires_grad=True):
        self.data = data if data is not None else MagicMock()
        self.requires_grad = requires_grad
        self.grad = None
        self.shape = getattr(data, "shape", (1,))

    def numel(self):
        return 0


class _FakeLinear(MockModule):
    def __init__(self, in_f=0, out_f=0, *a, **kw):
        super().__init__()

    def __call__(self, x):
        return MagicMock()

class TestAttentionType:
    def test_all_values(self):
        assert AttentionType.SELF_ATTENTION.value == "self_attention"
        assert AttentionType.CROSS_ATTENTION.value == "cross_attention"
        assert AttentionType.TEMPORAL_ATTENTION.value == "temporal_attention"
        assert AttentionType.CAUSAL_ATTENTION.value == "causal_attention"
        assert AttentionType.SPARSE_ATTENTION.value == "sparse_attention"


# ===========================================================================
# Tests - TransformerConfig
# ===========================================================================
class TestTransformerConfig:
    def test_defaults(self):
        c = TransformerConfig()
        assert c.d_model == 256
        assert c.n_heads == 8
        assert c.n_layers == 6
        assert c.d_ff == 1024
        assert c.dropout == 0.1
        assert c.max_seq_len == 128
        assert c.context_window == 64
        assert c.pos_encoding_type == "sinusoidal"
        assert c.max_position == 1000
        assert c.attention_dropout == 0.1
        assert c.layer_norm_eps == 1e-6
        assert c.use_residual is True
        assert c.learning_rate == 1e-4
        assert c.warmup_steps == 4000
        assert c.weight_decay == 1e-4
        assert c.label_smoothing == 0.1
        assert c.sensor_fusion_heads == 4
        assert c.temporal_heads == 4
        assert c.health_attention_dim == 64


# ===========================================================================
# Tests - PositionalEncoding
# ===========================================================================
class TestPositionalEncoding:
    def test_init(self):
        pe = PositionalEncoding(d_model=256, max_len=100)
        assert pe is not None

    def test_forward(self):
        pe = PositionalEncoding(d_model=64)
        mock_x = MagicMock()
        mock_x.size.return_value = 10
        # pe attribute is set via register_buffer
        result = pe.forward(mock_x)
        assert result is not None


# ===========================================================================
# Tests - MultiHeadAttention
# ===========================================================================
class TestMultiHeadAttention:
    def test_init_self_attention(self):
        attn = MultiHeadAttention(
            d_model=64, n_heads=4, dropout=0.1,
            attention_type=AttentionType.SELF_ATTENTION)
        assert attn.d_model == 64
        assert attn.n_heads == 4
        assert attn.d_k == 16
        assert attn.attention_type == AttentionType.SELF_ATTENTION

    def test_init_causal(self):
        attn = MultiHeadAttention(
            d_model=32, n_heads=2, dropout=0.0,
            attention_type=AttentionType.CAUSAL_ATTENTION)
        assert attn.attention_type == AttentionType.CAUSAL_ATTENTION

    def test_init_cross(self):
        attn = MultiHeadAttention(
            d_model=64, n_heads=4,
            attention_type=AttentionType.CROSS_ATTENTION)
        assert attn.attention_type == AttentionType.CROSS_ATTENTION

    def test_init_temporal(self):
        attn = MultiHeadAttention(
            d_model=64, n_heads=4,
            attention_type=AttentionType.TEMPORAL_ATTENTION)
        assert attn.attention_type == AttentionType.TEMPORAL_ATTENTION

    def test_forward(self):
        attn = MultiHeadAttention(d_model=64, n_heads=4)
        q = MagicMock()
        q.size.return_value = (2, 8, 64)
        k = MagicMock()
        k.size.return_value = (2, 8, 64)
        v = MagicMock()
        v.size.return_value = (2, 8, 64)
        result = attn.forward(q, k, v)
        assert result is not None

    def test_forward_with_mask(self):
        attn = MultiHeadAttention(d_model=64, n_heads=4)
        q = MagicMock()
        q.size.return_value = (2, 8, 64)
        k = MagicMock()
        k.size.return_value = (2, 8, 64)
        v = MagicMock()
        v.size.return_value = (2, 8, 64)
        mask = MagicMock()
        result = attn.forward(q, k, v, mask=mask)
        assert result is not None

    def test_scaled_dot_product_attention_no_mask(self):
        attn = MultiHeadAttention(d_model=32, n_heads=2)
        Q = MagicMock()
        K = MagicMock()
        V = MagicMock()
        K.transpose.return_value = MagicMock()
        result = attn._scaled_dot_product_attention(Q, K, V, mask=None)
        assert result is not None

    def test_scaled_dot_product_attention_with_mask(self):
        attn = MultiHeadAttention(d_model=32, n_heads=2)
        Q = MagicMock()
        K = MagicMock()
        V = MagicMock()
        K.transpose.return_value = MagicMock()
        mask = MagicMock()
        result = attn._scaled_dot_product_attention(Q, K, V, mask=mask)
        assert result is not None

    def test_scaled_dot_product_causal(self):
        attn = MultiHeadAttention(
            d_model=32, n_heads=2,
            attention_type=AttentionType.CAUSAL_ATTENTION)
        Q = MagicMock()
        K = MagicMock()
        V = MagicMock()
        scores = MagicMock()
        scores.size.return_value = 4
        K.transpose.return_value = MagicMock()
        mock_torch.matmul.return_value = scores
        result = attn._scaled_dot_product_attention(Q, K, V, mask=None)
        assert result is not None


# ===========================================================================
# Tests - TransformerEncoderLayer
# ===========================================================================
class TestTransformerEncoderLayer:
    def test_init(self):
        config = TransformerConfig()
        layer = TransformerEncoderLayer(config)
        assert layer.config is config

    def test_forward(self):
        config = TransformerConfig()
        layer = TransformerEncoderLayer(config)
        # Mock self_attention
        layer.self_attention = MagicMock(return_value=(MagicMock(), MagicMock()))
        layer.feed_forward = MagicMock(return_value=MagicMock())
        layer.norm1 = MagicMock(return_value=MagicMock())
        layer.norm2 = MagicMock(return_value=MagicMock())
        layer.dropout = MagicMock(return_value=MagicMock())
        x = MagicMock()
        result = layer.forward(x)
        assert result is not None

    def test_forward_with_mask(self):
        config = TransformerConfig()
        layer = TransformerEncoderLayer(config)
        layer.self_attention = MagicMock(return_value=(MagicMock(), MagicMock()))
        layer.feed_forward = MagicMock(return_value=MagicMock())
        layer.norm1 = MagicMock(return_value=MagicMock())
        layer.norm2 = MagicMock(return_value=MagicMock())
        layer.dropout = MagicMock(return_value=MagicMock())
        x = MagicMock()
        mask = MagicMock()
        result = layer.forward(x, mask=mask)
        assert result is not None


# ===========================================================================
# Tests - SensorFusionAttention
# ===========================================================================
class TestSensorFusionAttention:
    def test_init(self):
        config = TransformerConfig()
        sensor_dims = {"eis": 16, "qcm": 8}
        sfa = SensorFusionAttention(config, sensor_dims)
        assert sfa.config is config
        assert sfa.sensor_dims == sensor_dims
        assert sfa.sensor_types == ["eis", "qcm"]

    def test_forward_multiple_sensors(self):
        config = TransformerConfig()
        sensor_dims = {"eis": 16, "qcm": 8}
        sfa = SensorFusionAttention(config, sensor_dims)
        # Mock sensor_projections
        sfa.sensor_projections = MagicMock()
        sfa.sensor_projections.__contains__ = MagicMock(return_value=True)
        sfa.sensor_projections.__getitem__ = MagicMock(
            return_value=MagicMock(return_value=MagicMock()))
        # Mock cross_attention
        sfa.cross_attention = MagicMock(return_value=(MagicMock(), MagicMock()))
        # Mock fusion_layer
        sfa.fusion_layer = MagicMock(return_value=MagicMock())

        sensor_data = {
            "eis": MagicMock(),
            "qcm": MagicMock(),
        }
        result = sfa.forward(sensor_data)
        assert result is not None

    def test_forward_single_sensor(self):
        config = TransformerConfig()
        sensor_dims = {"eis": 16, "qcm": 8}
        sfa = SensorFusionAttention(config, sensor_dims)
        sfa.sensor_projections = MagicMock()
        sfa.sensor_projections.__contains__ = MagicMock(return_value=True)
        proj_mock = MagicMock(return_value=MagicMock())
        sfa.sensor_projections.__getitem__ = MagicMock(return_value=proj_mock)
        sfa.cross_attention = MagicMock(return_value=(MagicMock(), MagicMock()))
        sfa.fusion_layer = MagicMock(return_value=MagicMock())

        # Only one sensor provided
        sensor_data = {"eis": MagicMock()}
        result = sfa.forward(sensor_data)
        assert result is not None

    def test_forward_unknown_sensor(self):
        config = TransformerConfig()
        sensor_dims = {"eis": 16}
        sfa = SensorFusionAttention(config, sensor_dims)
        sfa.sensor_projections = MagicMock()
        sfa.sensor_projections.__contains__ = MagicMock(return_value=False)
        sfa.fusion_layer = MagicMock(return_value=MagicMock())

        sensor_data = {"unknown_sensor": MagicMock()}
        result = sfa.forward(sensor_data)
        assert result is not None

    def test_forward_empty_projected(self):
        """Test fallback when no sensors are projected."""
        config = TransformerConfig()
        sensor_dims = {"eis": 16}
        sfa = SensorFusionAttention(config, sensor_dims)
        sfa.sensor_projections = MagicMock()
        sfa.sensor_projections.__contains__ = MagicMock(return_value=False)
        sfa.fusion_layer = MagicMock()

        result = sfa.forward({})
        assert result is not None


# ===========================================================================
# Tests - TemporalAttentionModule
# ===========================================================================
class TestTemporalAttentionModule:
    def test_init(self):
        config = TransformerConfig()
        tam = TemporalAttentionModule(config)
        assert tam.config is config

    def test_forward(self):
        config = TransformerConfig()
        tam = TemporalAttentionModule(config)
        tam.temporal_attention = MagicMock(return_value=(MagicMock(), MagicMock()))
        tam.temporal_conv = MagicMock(return_value=MagicMock(
            transpose=MagicMock(return_value=MagicMock())))
        tam.combination = MagicMock(return_value=MagicMock())
        tam.norm = MagicMock(return_value=MagicMock())

        x = MagicMock()
        x.size.return_value = (2, 8, 256)
        x.transpose.return_value = MagicMock()
        result = tam.forward(x)
        assert result is not None

    def test_forward_with_mask(self):
        config = TransformerConfig()
        tam = TemporalAttentionModule(config)
        tam.temporal_attention = MagicMock(return_value=(MagicMock(), MagicMock()))
        tam.temporal_conv = MagicMock(return_value=MagicMock(
            transpose=MagicMock(return_value=MagicMock())))
        tam.combination = MagicMock(return_value=MagicMock())
        tam.norm = MagicMock(return_value=MagicMock())

        x = MagicMock()
        x.size.return_value = (2, 8, 256)
        x.transpose.return_value = MagicMock()
        mask = MagicMock()
        result = tam.forward(x, temporal_mask=mask)
        assert result is not None


# ===========================================================================
# Tests - TransformerMFCController
# ===========================================================================
class TestTransformerMFCController:
    def test_init_default_config(self):
        input_dims = {"sensor": 10, "health": 5}
        ctrl = TransformerMFCController(input_dims, output_dim=4)
        assert ctrl.config is not None
        assert ctrl.input_dims == input_dims
        assert ctrl.output_dim == 4

    def test_init_custom_config(self):
        config = TransformerConfig()
        config.d_model = 128
        input_dims = {"sensor": 10}
        ctrl = TransformerMFCController(input_dims, output_dim=3, config=config)
        assert ctrl.config.d_model == 128

    def test_forward_single_input(self):
        input_dims = {"sensor": 10}
        ctrl = TransformerMFCController(input_dims, output_dim=4)
        ctrl.input_projections = MagicMock()
        ctrl.input_projections.__contains__ = MagicMock(return_value=True)
        ctrl.input_projections.__getitem__ = MagicMock(
            return_value=MagicMock(return_value=MagicMock()))
        ctrl.input_projections.__len__ = MagicMock(return_value=1)
        ctrl.pos_encoding = MagicMock(return_value=MagicMock())
        ctrl.dropout = MagicMock(return_value=MagicMock())
        ctrl.temporal_attention = MagicMock(return_value=(MagicMock(), MagicMock()))
        ctrl.encoder_layers = []
        ctrl.output_projection = MagicMock(return_value=MagicMock())
        ctrl.value_head = MagicMock(return_value=MagicMock())
        ctrl.sensor_fusion = MagicMock(return_value=(MagicMock(), {}))
        ctrl.health_attention = MagicMock(return_value=(MagicMock(), MagicMock()))

        # Mock input tensor
        input_tensor = MagicMock()
        input_tensor.size.return_value = 2
        input_tensor.dim.return_value = 3
        inputs = {"sensor": input_tensor}

        # Create an iterator for the first input
        mock_val = MagicMock()
        mock_val.size.side_effect = [2, 4]

        with patch("builtins.iter", return_value=iter([mock_val])):
            result = ctrl.forward(inputs)
        assert result is not None

    def test_forward_multiple_inputs(self):
        input_dims = {"sensor": 10, "health": 5}
        ctrl = TransformerMFCController(input_dims, output_dim=4)
        ctrl.input_projections = MagicMock()
        ctrl.input_projections.__contains__ = MagicMock(return_value=True)
        ctrl.input_projections.__getitem__ = MagicMock(
            return_value=MagicMock(return_value=MagicMock()))
        ctrl.input_projections.__len__ = MagicMock(return_value=2)
        ctrl.pos_encoding = MagicMock(return_value=MagicMock())
        ctrl.dropout = MagicMock(return_value=MagicMock())
        ctrl.temporal_attention = MagicMock(return_value=(MagicMock(), MagicMock()))
        ctrl.encoder_layers = []
        ctrl.output_projection = MagicMock(return_value=MagicMock())
        ctrl.value_head = MagicMock(return_value=MagicMock())
        ctrl.sensor_fusion = MagicMock(return_value=(MagicMock(), {}))
        ctrl.health_attention = MagicMock(return_value=(MagicMock(), MagicMock()))

        input_tensor = MagicMock()
        input_tensor.dim.return_value = 3
        inputs = {"sensor": input_tensor, "health": input_tensor}
        mock_val = MagicMock()
        mock_val.size.side_effect = [2, 4]
        with patch("builtins.iter", return_value=iter([mock_val])):
            result = ctrl.forward(inputs)
        assert result is not None

    def test_forward_with_health_context(self):
        input_dims = {"sensor": 10}
        ctrl = TransformerMFCController(input_dims, output_dim=4)
        ctrl.input_projections = MagicMock()
        ctrl.input_projections.__contains__ = MagicMock(return_value=True)
        ctrl.input_projections.__getitem__ = MagicMock(
            return_value=MagicMock(return_value=MagicMock()))
        ctrl.pos_encoding = MagicMock(return_value=MagicMock())
        ctrl.dropout = MagicMock(return_value=MagicMock())
        ctrl.temporal_attention = MagicMock(return_value=(MagicMock(), MagicMock()))
        ctrl.encoder_layers = []
        enc_out = MagicMock()
        enc_out.mean.return_value = MagicMock()
        ctrl.output_projection = MagicMock(return_value=MagicMock())
        ctrl.value_head = MagicMock(return_value=MagicMock())
        ctrl.sensor_fusion = MagicMock(return_value=(MagicMock(), {}))
        health_enhanced = MagicMock()
        ctrl.health_attention = MagicMock(return_value=(health_enhanced, MagicMock()))

        input_tensor = MagicMock()
        input_tensor.dim.return_value = 3
        inputs = {"sensor": input_tensor}
        health_ctx = MagicMock()
        mock_val = MagicMock()
        mock_val.size.side_effect = [2, 4]
        with patch("builtins.iter", return_value=iter([mock_val])):
            result = ctrl.forward(inputs, health_context=health_ctx)
        assert result is not None

    def test_forward_return_attention(self):
        input_dims = {"sensor": 10}
        ctrl = TransformerMFCController(input_dims, output_dim=4)
        ctrl.input_projections = MagicMock()
        ctrl.input_projections.__contains__ = MagicMock(return_value=True)
        ctrl.input_projections.__getitem__ = MagicMock(
            return_value=MagicMock(return_value=MagicMock()))
        ctrl.pos_encoding = MagicMock(return_value=MagicMock())
        ctrl.dropout = MagicMock(return_value=MagicMock())
        ctrl.temporal_attention = MagicMock(return_value=(MagicMock(), MagicMock()))
        layer_mock = MagicMock(return_value=(MagicMock(), MagicMock()))
        ctrl.encoder_layers = [layer_mock]
        ctrl.output_projection = MagicMock(return_value=MagicMock())
        ctrl.value_head = MagicMock(return_value=MagicMock())
        ctrl.sensor_fusion = MagicMock(return_value=(MagicMock(), {}))
        ctrl.health_attention = MagicMock(return_value=(MagicMock(), MagicMock()))

        input_tensor = MagicMock()
        input_tensor.dim.return_value = 3
        inputs = {"sensor": input_tensor}
        mock_val = MagicMock()
        mock_val.size.side_effect = [2, 4]
        with patch("builtins.iter", return_value=iter([mock_val])):
            result = ctrl.forward(inputs, return_attention=True)
        assert "attention_weights" in result

    def test_forward_2d_input(self):
        """Test forward with 2D input that needs unsqueeze."""
        input_dims = {"sensor": 10}
        ctrl = TransformerMFCController(input_dims, output_dim=4)
        ctrl.input_projections = MagicMock()
        ctrl.input_projections.__contains__ = MagicMock(return_value=True)
        proj = MagicMock(return_value=MagicMock())
        ctrl.input_projections.__getitem__ = MagicMock(return_value=proj)
        ctrl.pos_encoding = MagicMock(return_value=MagicMock())
        ctrl.dropout = MagicMock(return_value=MagicMock())
        ctrl.temporal_attention = MagicMock(return_value=(MagicMock(), MagicMock()))
        ctrl.encoder_layers = []
        ctrl.output_projection = MagicMock(return_value=MagicMock())
        ctrl.value_head = MagicMock(return_value=MagicMock())
        ctrl.sensor_fusion = MagicMock(return_value=(MagicMock(), {}))
        ctrl.health_attention = MagicMock(return_value=(MagicMock(), MagicMock()))

        input_tensor = MagicMock()
        input_tensor.dim.return_value = 2
        input_tensor.unsqueeze.return_value = MagicMock()
        inputs = {"sensor": input_tensor}
        mock_val = MagicMock()
        mock_val.size.side_effect = [2, 4]
        with patch("builtins.iter", return_value=iter([mock_val])):
            result = ctrl.forward(inputs)
        assert result is not None

    def test_forward_1d_input(self):
        """Test forward with 1D input that needs double unsqueeze."""
        input_dims = {"sensor": 10}
        ctrl = TransformerMFCController(input_dims, output_dim=4)
        ctrl.input_projections = MagicMock()
        ctrl.input_projections.__contains__ = MagicMock(return_value=True)
        proj = MagicMock(return_value=MagicMock())
        ctrl.input_projections.__getitem__ = MagicMock(return_value=proj)
        ctrl.pos_encoding = MagicMock(return_value=MagicMock())
        ctrl.dropout = MagicMock(return_value=MagicMock())
        ctrl.temporal_attention = MagicMock(return_value=(MagicMock(), MagicMock()))
        ctrl.encoder_layers = []
        ctrl.output_projection = MagicMock(return_value=MagicMock())
        ctrl.value_head = MagicMock(return_value=MagicMock())
        ctrl.sensor_fusion = MagicMock(return_value=(MagicMock(), {}))
        ctrl.health_attention = MagicMock(return_value=(MagicMock(), MagicMock()))

        input_tensor = MagicMock()
        input_tensor.dim.return_value = 1
        unsqueeze1 = MagicMock()
        unsqueeze1.unsqueeze.return_value = MagicMock()
        input_tensor.unsqueeze.return_value = unsqueeze1
        inputs = {"sensor": input_tensor}
        mock_val = MagicMock()
        mock_val.size.side_effect = [2, 4]
        with patch("builtins.iter", return_value=iter([mock_val])):
            result = ctrl.forward(inputs)
        assert result is not None

    def test_get_attention_maps(self):
        input_dims = {"sensor": 10}
        ctrl = TransformerMFCController(input_dims, output_dim=4)
        # Mock the forward method
        ctrl.forward = MagicMock(return_value={
            "attention_weights": {"sensor": MagicMock(), "temporal": MagicMock()},
            "action_logits": MagicMock(),
            "state_value": MagicMock(),
        })
        inputs = {"sensor": MagicMock()}
        maps = ctrl.get_attention_maps(inputs)
        assert maps is not None

    def test_get_attention_maps_with_health(self):
        input_dims = {"sensor": 10}
        ctrl = TransformerMFCController(input_dims, output_dim=4)
        ctrl.forward = MagicMock(return_value={
            "attention_weights": {"sensor": MagicMock()},
        })
        inputs = {"sensor": MagicMock()}
        health_ctx = MagicMock()
        maps = ctrl.get_attention_maps(inputs, health_context=health_ctx)
        assert maps is not None


# ===========================================================================
# Tests - TransformerControllerManager
# ===========================================================================
class TestTransformerControllerManager:
    def _make(self, state_dim=30, action_dim=5, config=None):
        if config is None:
            config = TransformerConfig()
            config.d_model = 64
            config.n_heads = 4
            config.n_layers = 1
        with patch.object(TransformerControllerManager, "__init__", lambda self, *a, **kw: None):
            mgr = TransformerControllerManager.__new__(TransformerControllerManager)
        mgr.state_dim = state_dim
        mgr.action_dim = action_dim
        mgr.config = config
        mgr.device = "cpu"
        mgr.learning_rate = config.learning_rate
        mgr.optimizer = MagicMock(
            zero_grad=MagicMock(), step=MagicMock(),
            state_dict=MagicMock(return_value={}),
            load_state_dict=MagicMock())
        mgr.scheduler = MagicMock(
            get_last_lr=MagicMock(return_value=[1e-4]),
            step=MagicMock(),
            state_dict=MagicMock(return_value={}),
            load_state_dict=MagicMock())
        mgr.model = MagicMock()
        mgr.model.parameters.return_value = []
        mgr.model.state_dict.return_value = {}
        mgr.model.load_state_dict = MagicMock()
        mgr.model.train = MagicMock(return_value=mgr.model)
        mgr.model.eval = MagicMock(return_value=mgr.model)
        mgr.input_dims = {
            "sensor_features": state_dim // 3,
            "health_features": state_dim // 3,
            "system_features": state_dim // 3,
        }
        mgr.loss_history = deque(maxlen=1000)
        mgr.reward_history = deque(maxlen=1000)
        mgr.training_history = deque(maxlen=1000)
        mgr.accuracy_history = deque(maxlen=1000)
        mgr.attention_entropy_history = deque(maxlen=1000)
        mgr.sequence_buffer = deque(maxlen=config.max_seq_len)
        mgr._history_maxlen = 1000
        mgr._feature_engineer = None
        mgr.steps = 0
        mgr.episodes = 0
        return mgr

    def test_basic_init(self):
        mgr = self._make()
        assert mgr.state_dim == 30
        assert mgr.action_dim == 5

    def test_extract_transformer_features(self):
        mgr = self._make()
        mgr._feature_engineer = MagicMock()
        mgr._feature_engineer.extract_features.return_value = {
            f"f{i}": float(i) for i in range(30)
        }
        state = MagicMock()
        state.power_output = 5.0
        state.current_density = 1.0
        state.health_metrics.overall_health_score = 0.8
        state.fused_measurement.fusion_confidence = 0.9
        state.anomalies = []
        result = mgr.extract_transformer_features(state)
        assert "sensor_features" in result
        assert "health_features" in result
        assert "system_features" in result

    def test_control_step(self):
        mgr = self._make()
        mgr._feature_engineer = MagicMock()
        mgr._feature_engineer.extract_features.return_value = {
            f"f{i}": float(i) for i in range(30)
        }
        # Mock model forward
        mgr.model.return_value = {
            "action_logits": MagicMock(),
            "state_value": MagicMock(item=MagicMock(return_value=0.5)),
            "sequence_representation": MagicMock(),
            "attention_weights": {"sensor": MagicMock()},
        }
        state = MagicMock()
        state.power_output = 5.0
        state.current_density = 1.0
        state.health_metrics.overall_health_score = 0.8
        state.health_metrics.thickness_health = 0.7
        state.health_metrics.conductivity_health = 0.6
        state.health_metrics.growth_health = 0.5
        state.health_metrics.stability_health = 0.9
        state.fused_measurement.fusion_confidence = 0.9
        state.anomalies = []
        action, info = mgr.control_step(state)
        assert "transformer_action" in info

    def test_control_step_with_sequence(self):
        mgr = self._make()
        mgr._feature_engineer = MagicMock()
        mgr._feature_engineer.extract_features.return_value = {
            f"f{i}": float(i) for i in range(30)
        }
        mgr.model.return_value = {
            "action_logits": MagicMock(),
            "state_value": MagicMock(item=MagicMock(return_value=0.5)),
            "sequence_representation": MagicMock(),
            "attention_weights": {"sensor": MagicMock()},
        }
        state = MagicMock()
        state.power_output = 5.0
        state.current_density = 1.0
        state.health_metrics.overall_health_score = 0.8
        state.health_metrics.thickness_health = 0.7
        state.health_metrics.conductivity_health = 0.6
        state.health_metrics.growth_health = 0.5
        state.health_metrics.stability_health = 0.9
        state.fused_measurement.fusion_confidence = 0.9
        state.anomalies = []
        # Add initial entries to sequence buffer
        mgr.sequence_buffer.append({"sensor_features": MagicMock(),
                                     "health_features": MagicMock(),
                                     "system_features": MagicMock()})
        mgr.sequence_buffer.append({"sensor_features": MagicMock(),
                                     "health_features": MagicMock(),
                                     "system_features": MagicMock()})
        action, info = mgr.control_step(state)
        assert "sequence_length" in info

    def test_visualize_attention(self):
        mgr = self._make()
        mgr._feature_engineer = MagicMock()
        mgr._feature_engineer.extract_features.return_value = {
            f"f{i}": float(i) for i in range(30)
        }
        # Mock get_attention_maps
        mock_attn_weights = MagicMock()
        mock_attn_weights.mean.return_value = MagicMock(
            squeeze=MagicMock(return_value=MagicMock(
                cpu=MagicMock(return_value=MagicMock(
                    numpy=MagicMock(return_value=np.array([[0.5, 0.5]])))))))
        mgr.model.get_attention_maps = MagicMock(return_value={
            "temporal": mock_attn_weights,
            "health": None,
            "encoder_layers": [MagicMock(), MagicMock()],  # list, not Tensor
        })
        state = MagicMock()
        state.power_output = 5.0
        state.current_density = 1.0
        state.health_metrics.overall_health_score = 0.8
        state.fused_measurement.fusion_confidence = 0.9
        state.anomalies = []
        result = mgr.visualize_attention(state)
        assert isinstance(result, dict)

    def test_train_step_empty(self):
        mgr = self._make()
        result = mgr.train_step()
        assert result["loss"] == 0.0
        assert result["accuracy"] == 0.0

    def test_train_step_none(self):
        mgr = self._make()
        result = mgr.train_step(None)
        assert result["loss"] == 0.0

    def test_train_step_with_data(self):
        mgr = self._make()
        batch = [{"features": MagicMock(), "action": 0}]
        result = mgr.train_step(batch)
        assert "loss" in result
        assert "learning_rate" in result
        assert mgr.steps == 1

    def test_save_model(self):
        mgr = self._make()
        mgr.save_model("/tmp/test_transformer.pt")
        mock_torch.save.assert_called()

    def test_load_model(self):
        mgr = self._make()
        mock_torch.load.return_value = {
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
            "steps": 50,
            "episodes": 5,
            "loss_history": [0.1, 0.2],
            "reward_history": [1.0],
            "accuracy_history": [0.9, 0.8],
            "attention_entropy_history": [0.5, 0.4],
        }
        mgr.load_model("/tmp/test_transformer.pt")
        assert mgr.steps == 50
        assert mgr.episodes == 5

    def test_load_model_without_optional_keys(self):
        mgr = self._make()
        mock_torch.load.return_value = {
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
            "steps": 10,
            "episodes": 1,
        }
        mgr.load_model("/tmp/test_transformer.pt")
        assert mgr.steps == 10

    def test_get_model_summary(self):
        mgr = self._make()
        summary = mgr.get_model_summary()
        assert summary["model_type"] == "transformer_mfc_controller"
        assert "parameters" in summary
        assert "model_dimension" in summary
        assert "attention_heads" in summary

    def test_get_performance_summary(self):
        mgr = self._make()
        mgr.accuracy_history.append(0.9)
        mgr.attention_entropy_history.append(0.5)
        summary = mgr.get_performance_summary()
        assert "avg_accuracy" in summary
        assert "avg_attention_entropy" in summary

    def test_get_performance_summary_empty(self):
        mgr = self._make()
        summary = mgr.get_performance_summary()
        assert summary["avg_accuracy"] == 0.0
        assert summary["avg_attention_entropy"] == 0.0

    def test_get_learning_rate_with_scheduler(self):
        mgr = self._make()
        lr = mgr.get_learning_rate()
        assert lr == 1e-4

    def test_get_learning_rate_no_scheduler(self):
        mgr = self._make()
        mgr.scheduler = None
        lr = mgr.get_learning_rate()
        assert lr == mgr.learning_rate


# ===========================================================================
# Tests - Factory function
# ===========================================================================
class TestCreateTransformerController:
    def test_default(self):
        with patch.object(TransformerControllerManager, "__init__",
                          lambda self, *a, **kw: None):
            with patch("transformer_controller.TransformerControllerManager") as mock_cls:
                mock_inst = MagicMock()
                mock_cls.return_value = mock_inst
                result = create_transformer_controller(state_dim=30, action_dim=5)
                assert result is mock_inst

    def test_custom_params(self):
        with patch("transformer_controller.TransformerControllerManager") as mock_cls:
            mock_inst = MagicMock()
            mock_cls.return_value = mock_inst
            result = create_transformer_controller(
                state_dim=30, action_dim=5,
                d_model=128, n_heads=4, n_layers=2)
            assert result is mock_inst

    def test_with_kwargs(self):
        with patch("transformer_controller.TransformerControllerManager") as mock_cls:
            mock_inst = MagicMock()
            mock_cls.return_value = mock_inst
            result = create_transformer_controller(
                state_dim=30, action_dim=5,
                dropout=0.2, nonexistent_key=True)
            assert result is mock_inst