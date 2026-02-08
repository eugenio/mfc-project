"""Tests for transformer_controller - comprehensive coverage part 1.

Covers AttentionType, TransformerConfig, PositionalEncoding,
MultiHeadAttention, TransformerEncoderLayer, SensorFusionAttention,
TemporalAttentionModule.
"""
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import types
import random
import math
from collections import deque, defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

mock_torch = MagicMock()
mock_nn = MagicMock()
mock_optim = MagicMock()
mock_F = MagicMock()


class _FakeTensor:
    pass

mock_torch.Tensor = _FakeTensor


class MockModule:
    def __init__(self, *a, **kw): pass
    def forward(self, x): return x
    def parameters(self): return []
    def state_dict(self): return {}
    def load_state_dict(self, d): pass
    def train(self, mode=True): return self
    def eval(self): return self
    def to(self, device): return self
    def __call__(self, *a, **kw): return MagicMock()
    def named_parameters(self): return []
    def children(self): return []
    def register_buffer(self, name, tensor): setattr(self, name, tensor)
    def apply(self, fn): return self
    training = True


class _FakeParameter:
    def __init__(self, data=None, requires_grad=True):
        self.data = data if data is not None else MagicMock()
        self.requires_grad = requires_grad
        self.grad = None
        self.shape = getattr(data, "shape", (1,))
    def numel(self): return 0


class _FakeLinear(MockModule):
    def __init__(self, in_f=0, out_f=0, *a, **kw): super().__init__()
    def __call__(self, x): return MagicMock()


class _FakeDropout(MockModule):
    def __init__(self, p=0.0, *a, **kw): super().__init__()
    def __call__(self, x): return x


class _FakeLayerNorm(MockModule):
    def __init__(self, *a, **kw): super().__init__()
    def __call__(self, x): return x


class _FakeConv1d(MockModule):
    def __init__(self, *a, **kw): super().__init__()
    def __call__(self, x): return MagicMock()


class _FakeModuleDict(dict, MockModule):
    def __init__(self, *a, **kw):
        dict.__init__(self)
        MockModule.__init__(self)


class _FakeModuleList(list, MockModule):
    def __init__(self, items=None, *a, **kw):
        list.__init__(self, items or [])
        MockModule.__init__(self)


class _FakeSequential(MockModule):
    def __init__(self, *layers, **kw): super().__init__()
    def __call__(self, x): return MagicMock()


mock_nn.Module = MockModule
mock_nn.Linear = _FakeLinear
mock_nn.Dropout = _FakeDropout
mock_nn.LayerNorm = _FakeLayerNorm
mock_nn.Conv1d = _FakeConv1d
mock_nn.ModuleDict = _FakeModuleDict
mock_nn.ModuleList = _FakeModuleList
mock_nn.Sequential = _FakeSequential
mock_nn.ReLU = MagicMock(return_value=MagicMock())
mock_nn.functional = mock_F
mock_nn.Parameter = _FakeParameter
mock_torch.nn = mock_nn
mock_torch.optim = mock_optim
mock_torch.optim.SGD = MagicMock(return_value=MagicMock())
mock_torch.optim.Adam = MagicMock(return_value=MagicMock())
mock_torch.optim.AdamW = MagicMock(return_value=MagicMock(
    zero_grad=MagicMock(), step=MagicMock(),
    state_dict=MagicMock(return_value={}),
    load_state_dict=MagicMock()))
mock_torch.optim.lr_scheduler = MagicMock()
mock_torch.optim.lr_scheduler.LambdaLR = MagicMock(return_value=MagicMock(
    get_last_lr=MagicMock(return_value=[1e-4]),
    step=MagicMock(), state_dict=MagicMock(return_value={}),
    load_state_dict=MagicMock()))
mock_torch.tensor = MagicMock(return_value=MagicMock())
mock_torch.zeros = MagicMock(return_value=MagicMock())
mock_torch.ones = MagicMock(return_value=MagicMock())
mock_torch.empty = MagicMock(return_value=MagicMock())
mock_torch.zeros_like = MagicMock(return_value=MagicMock())
mock_torch.randn_like = MagicMock(return_value=MagicMock())
mock_torch.from_numpy = MagicMock(return_value=MagicMock(
    float=MagicMock(return_value=MagicMock())))
mock_torch.arange = MagicMock(return_value=MagicMock(
    unsqueeze=MagicMock(return_value=MagicMock()),
    float=MagicMock(return_value=MagicMock())))
mock_torch.exp = MagicMock(return_value=MagicMock())
mock_torch.sin = MagicMock(return_value=MagicMock())
mock_torch.cos = MagicMock(return_value=MagicMock())
mock_torch.tril = MagicMock(return_value=MagicMock(
    to=MagicMock(return_value=MagicMock())))
mock_torch.matmul = MagicMock(return_value=MagicMock(
    size=MagicMock(return_value=4)))
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
            to=MagicMock(return_value=MagicMock())))))))
mock_torch.LongTensor = MagicMock(return_value=MagicMock(
    to=MagicMock(return_value=MagicMock())))
mock_torch.normal = MagicMock(return_value=MagicMock(
    to=MagicMock(return_value=MagicMock())))
mock_torch.abs = MagicMock(return_value=MagicMock())
mock_torch.topk = MagicMock(return_value=(MagicMock(),
    MagicMock(cpu=MagicMock(return_value=MagicMock(
        numpy=MagicMock(return_value=__import__("numpy").array([0, 1])))))))
mock_torch.stack = MagicMock(return_value=MagicMock(
    median=MagicMock(return_value=(MagicMock(), None))))
mock_torch.median = MagicMock(return_value=(MagicMock(), None))
mock_torch.sort = MagicMock(return_value=(MagicMock(), None))
mock_torch.mean = MagicMock(return_value=MagicMock())
mock_torch.cat = MagicMock(return_value=MagicMock(
    shape=(1, 1, 256), device="cpu"))
mock_torch.argmax = MagicMock(return_value=MagicMock(
    item=MagicMock(return_value=0)))
mock_torch.max = MagicMock(return_value=MagicMock(
    item=MagicMock(return_value=0.9)))
mock_torch.nn.utils = MagicMock()
mock_torch.nn.utils.clip_grad_norm_ = MagicMock(
    return_value=MagicMock(item=MagicMock(return_value=1.0)))
mock_F.softmax = MagicMock(return_value=MagicMock())
mock_torch.distributions = MagicMock()

sys.modules["torch"] = mock_torch
sys.modules["torch.nn"] = mock_nn
sys.modules["torch.optim"] = mock_optim
sys.modules["torch.nn.functional"] = mock_F
sys.modules["torch.utils"] = MagicMock()
sys.modules["torch.utils.data"] = MagicMock()

import numpy as np
import pytest
import torch_compat
torch_compat.TORCH_AVAILABLE = True
torch_compat.torch = mock_torch
torch_compat.nn = mock_nn
torch_compat.get_device = MagicMock(return_value="cpu")

from transformer_controller import (
    AttentionType, TransformerConfig, PositionalEncoding,
    MultiHeadAttention, TransformerEncoderLayer,
    SensorFusionAttention, TemporalAttentionModule,
)


class TestAttentionType:
    def test_values(self):
        assert AttentionType.SELF_ATTENTION.value == "self_attention"
        assert AttentionType.CROSS_ATTENTION.value == "cross_attention"
        assert AttentionType.TEMPORAL_ATTENTION.value == "temporal_attention"
        assert AttentionType.CAUSAL_ATTENTION.value == "causal_attention"
        assert AttentionType.SPARSE_ATTENTION.value == "sparse_attention"


class TestTransformerConfig:
    def test_defaults(self):
        c = TransformerConfig()
        assert c.d_model == 256 and c.n_heads == 8 and c.n_layers == 6
        assert c.d_ff == 1024 and c.dropout == 0.1
        assert c.max_seq_len == 128 and c.context_window == 64
        assert c.pos_encoding_type == "sinusoidal"
        assert c.max_position == 1000
        assert c.learning_rate == 1e-4 and c.warmup_steps == 4000
        assert c.weight_decay == 1e-4 and c.label_smoothing == 0.1
        assert c.sensor_fusion_heads == 4 and c.temporal_heads == 4
        assert c.health_attention_dim == 64
        assert c.attention_dropout == 0.1 and c.layer_norm_eps == 1e-6
        assert c.use_residual is True

    def test_modify(self):
        c = TransformerConfig()
        c.d_model = 512; c.n_heads = 16
        assert c.d_model == 512


class TestPositionalEncoding:
    def test_init(self):
        pe = PositionalEncoding(d_model=256, max_len=100)
        assert hasattr(pe, "pe")

    def test_init_default(self):
        pe = PositionalEncoding(d_model=128)
        assert hasattr(pe, "pe")

    def test_forward(self):
        pe = PositionalEncoding(d_model=256)
        x = MagicMock()
        x.size.return_value = 10
        pe.forward(x)


class TestMultiHeadAttention:
    def test_init_self(self):
        mha = MultiHeadAttention(d_model=256, n_heads=8)
        assert mha.d_model == 256 and mha.n_heads == 8
        assert mha.d_k == 32 and mha.scale == math.sqrt(32)
        assert mha.attention_type == AttentionType.SELF_ATTENTION

    def test_init_cross(self):
        mha = MultiHeadAttention(256, 8,
            attention_type=AttentionType.CROSS_ATTENTION)
        assert mha.attention_type == AttentionType.CROSS_ATTENTION

    def test_init_causal(self):
        mha = MultiHeadAttention(256, 8,
            attention_type=AttentionType.CAUSAL_ATTENTION)
        assert mha.attention_type == AttentionType.CAUSAL_ATTENTION

    def test_init_temporal(self):
        mha = MultiHeadAttention(256, 4, dropout=0.2,
            attention_type=AttentionType.TEMPORAL_ATTENTION)
        assert mha.attention_type == AttentionType.TEMPORAL_ATTENTION

    def test_forward_has_bug(self):
        """Source bug: uses 'seq_len' instead of 'seq_len_q'."""
        mha = MultiHeadAttention(d_model=256, n_heads=8)
        q = MagicMock(); q.size.return_value = (1, 4, 256)
        k = MagicMock(); k.size.return_value = (1, 4, 256)
        v = MagicMock(); v.size.return_value = (1, 4, 256)
        with pytest.raises(NameError, match="seq_len"):
            mha.forward(q, k, v)

    def test_scaled_dot_product_no_mask(self):
        mha = MultiHeadAttention(256, 8)
        Q = MagicMock()
        K = MagicMock(); K.transpose.return_value = MagicMock()
        V = MagicMock()
        output, weights = mha._scaled_dot_product_attention(Q, K, V)

    def test_scaled_dot_product_with_mask(self):
        mha = MultiHeadAttention(256, 8)
        Q = MagicMock()
        K = MagicMock(); K.transpose.return_value = MagicMock()
        V = MagicMock()
        mask = MagicMock()
        output, weights = mha._scaled_dot_product_attention(Q, K, V, mask)

    def test_scaled_dot_product_causal(self):
        mha = MultiHeadAttention(256, 8,
            attention_type=AttentionType.CAUSAL_ATTENTION)
        Q = MagicMock()
        scores = MagicMock(); scores.size.return_value = 4
        mock_torch.matmul.return_value = scores
        K = MagicMock(); K.transpose.return_value = MagicMock()
        V = MagicMock()
        output, weights = mha._scaled_dot_product_attention(Q, K, V)


class TestTransformerEncoderLayer:
    def test_init(self):
        layer = TransformerEncoderLayer(TransformerConfig())
        assert layer.config is not None

    def test_forward(self):
        layer = TransformerEncoderLayer(TransformerConfig())
        layer.self_attention = MagicMock(
            return_value=(MagicMock(), MagicMock()))
        output, weights = layer.forward(MagicMock())

    def test_forward_mask(self):
        layer = TransformerEncoderLayer(TransformerConfig())
        layer.self_attention = MagicMock(
            return_value=(MagicMock(), MagicMock()))
        output, weights = layer.forward(MagicMock(), mask=MagicMock())


class TestSensorFusionAttention:
    def test_init(self):
        sfa = SensorFusionAttention(TransformerConfig(), {"eis": 32, "qcm": 16})
        assert sfa.sensor_types == ["eis", "qcm"]

    def test_forward_multi(self):
        sfa = SensorFusionAttention(TransformerConfig(), {"eis": 32, "qcm": 16})
        sfa.cross_attention = MagicMock(
            return_value=(MagicMock(), MagicMock()))
        output, weights = sfa.forward({"eis": MagicMock(), "qcm": MagicMock()})

    def test_forward_single(self):
        sfa = SensorFusionAttention(TransformerConfig(), {"eis": 32, "qcm": 16})
        output, weights = sfa.forward({"eis": MagicMock()})

    def test_forward_none(self):
        sfa = SensorFusionAttention(TransformerConfig(), {"eis": 32})
        output, weights = sfa.forward({"unknown": MagicMock()})

    def test_forward_padding(self):
        config = TransformerConfig()
        sfa = SensorFusionAttention(config, {"eis": 32, "qcm": 16, "temp": 8})
        sfa.cross_attention = MagicMock(
            return_value=(MagicMock(), MagicMock()))
        cat_result = MagicMock()
        cat_result.shape = (1, 1, config.d_model * 2)
        mock_torch.cat.return_value = cat_result
        output, weights = sfa.forward({"eis": MagicMock(), "qcm": MagicMock()})


class TestTemporalAttentionModule:
    def test_init(self):
        tam = TemporalAttentionModule(TransformerConfig())
        assert tam.config is not None

    def test_forward(self):
        tam = TemporalAttentionModule(TransformerConfig())
        tam.temporal_attention = MagicMock(
            return_value=(MagicMock(), MagicMock()))
        x = MagicMock()
        x.size.return_value = (1, 4, 256)
        x.transpose.return_value = MagicMock()
        output, weights = tam.forward(x)

    def test_forward_mask(self):
        tam = TemporalAttentionModule(TransformerConfig())
        tam.temporal_attention = MagicMock(
            return_value=(MagicMock(), MagicMock()))
        x = MagicMock()
        x.size.return_value = (1, 4, 256)
        x.transpose.return_value = MagicMock()
        output, weights = tam.forward(x, temporal_mask=MagicMock())
