"""Tests for config/sensitivity_analysis.py - targeting 98%+ coverage.

This test file mocks ALL heavy dependencies (numpy, matplotlib, pandas,
scipy) to avoid the numpy reimport conflict that occurs when multiple
numpy installations exist.
"""
import importlib.util
import math
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

import random as _stdlib_random

# ============ Mock ALL heavy deps BEFORE any import =============


class FakeNdarray:
    """Minimal ndarray for test purposes with full 2D support."""

    def __init__(self, data, dtype=None):
        if isinstance(data, FakeNdarray):
            data = data._to_nested_list()
        if isinstance(data, (list, tuple)):
            self._data = [list(r) if isinstance(r, (list, tuple)) else r for r in data]
        else:
            self._data = [data]
        self._dtype = dtype

    def _is_2d(self):
        return bool(self._data) and isinstance(self._data[0], list)

    def _to_nested_list(self):
        if self._is_2d():
            return [list(row) for row in self._data]
        return list(self._data)

    @property
    def ndim(self):
        return 2 if self._is_2d() else 1

    @property
    def shape(self):
        if not self._data:
            return (0,)
        if self._is_2d():
            return (len(self._data), len(self._data[0]))
        return (len(self._data),)

    @property
    def size(self):
        s = self.shape
        r = 1
        for d in s:
            r *= d
        return r

    def _get_col(self, col_idx):
        """Extract column col_idx from 2D array, returns FakeNdarray."""
        return FakeNdarray([row[col_idx] for row in self._data])

    def _set_col(self, col_idx, values):
        """Set column col_idx from a FakeNdarray or list."""
        if isinstance(values, FakeNdarray):
            vals = values._data
        elif isinstance(values, (list, tuple)):
            vals = list(values)
        else:
            vals = [values] * len(self._data)
        for r, v in enumerate(vals):
            self._data[r][col_idx] = v

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            row_idx, col_idx = idx
            # Handle slice rows
            if isinstance(row_idx, slice):
                rows = self._data[row_idx]
                if isinstance(col_idx, (int,)):
                    # arr[:, i] -> column extraction
                    return FakeNdarray([r[col_idx] for r in rows])
                # arr[:, :] or similar
                return FakeNdarray([list(r) for r in rows])
            # Single row, single col
            if isinstance(col_idx, (int,)):
                if self._is_2d():
                    return self._data[row_idx][col_idx]
                return self._data[row_idx]
            return self._data[row_idx]
        if isinstance(idx, slice):
            sliced = self._data[idx]
            return FakeNdarray(sliced)
        return self._data[idx]

    def __setitem__(self, idx, val):
        if isinstance(idx, tuple):
            row_idx, col_idx = idx
            if isinstance(row_idx, slice):
                # arr[:, i] = values  (column assignment)
                rows = range(*row_idx.indices(len(self._data)))
                if isinstance(val, FakeNdarray):
                    vals = val._data
                elif isinstance(val, (list, tuple)):
                    vals = list(val)
                else:
                    vals = [val] * len(rows)
                for ri, vi in zip(rows, vals):
                    if self._is_2d():
                        self._data[ri][col_idx] = vi
                    else:
                        self._data[ri] = vi
                return
            # Single element
            if self._is_2d():
                self._data[row_idx][col_idx] = val
            else:
                self._data[row_idx] = val
        else:
            if isinstance(idx, slice):
                if isinstance(val, FakeNdarray):
                    self._data[idx] = val._data
                else:
                    self._data[idx] = val
            else:
                # When assigning to a row in a 2D array, ensure lists stay
                if self._is_2d() and isinstance(val, FakeNdarray):
                    self._data[idx] = list(val._data)
                elif self._is_2d() and isinstance(val, (list, tuple)):
                    self._data[idx] = list(val)
                else:
                    self._data[idx] = val

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        # For 2D, yield each row as a FakeNdarray
        if self._is_2d():
            for row in self._data:
                yield FakeNdarray(list(row))
        else:
            yield from self._data

    def _flat_values(self):
        """Return flat list of all scalar values."""
        if self._is_2d():
            result = []
            for row in self._data:
                result.extend(row)
            return result
        return list(self._data)

    def __mul__(self, other):
        if isinstance(other, FakeNdarray):
            if self._is_2d() and other._is_2d():
                return FakeNdarray(
                    [[a * b for a, b in zip(r1, r2)]
                     for r1, r2 in zip(self._data, other._data)]
                )
            if not self._is_2d() and not other._is_2d():
                return FakeNdarray(
                    [a * b for a, b in zip(self._data, other._data)]
                )
            # 1D * 1D element-wise
            return FakeNdarray(
                [a * b for a, b in zip(self._flat_values(), other._flat_values())]
            )
        # scalar
        if self._is_2d():
            return FakeNdarray(
                [[v * other for v in row] for row in self._data]
            )
        return FakeNdarray([x * other for x in self._data])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        if isinstance(other, FakeNdarray):
            if self._is_2d() and other._is_2d():
                return FakeNdarray(
                    [[a - b for a, b in zip(r1, r2)]
                     for r1, r2 in zip(self._data, other._data)]
                )
            if not self._is_2d() and not other._is_2d():
                return FakeNdarray(
                    [a - b for a, b in zip(self._data, other._data)]
                )
            return FakeNdarray(
                [a - b for a, b in zip(self._flat_values(), other._flat_values())]
            )
        if self._is_2d():
            return FakeNdarray(
                [[v - other for v in row] for row in self._data]
            )
        return FakeNdarray([x - other for x in self._data])

    def __rsub__(self, other):
        if self._is_2d():
            return FakeNdarray(
                [[other - v for v in row] for row in self._data]
            )
        return FakeNdarray([other - x for x in self._data])

    def __add__(self, other):
        if isinstance(other, FakeNdarray):
            if self._is_2d() and other._is_2d():
                return FakeNdarray(
                    [[a + b for a, b in zip(r1, r2)]
                     for r1, r2 in zip(self._data, other._data)]
                )
            if not self._is_2d() and not other._is_2d():
                return FakeNdarray(
                    [a + b for a, b in zip(self._data, other._data)]
                )
            return FakeNdarray(
                [a + b for a, b in zip(self._flat_values(), other._flat_values())]
            )
        if self._is_2d():
            return FakeNdarray(
                [[v + other for v in row] for row in self._data]
            )
        return FakeNdarray([x + other for x in self._data])

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        if self._is_2d():
            return FakeNdarray([[-v for v in row] for row in self._data])
        return FakeNdarray([-x for x in self._data])

    def __abs__(self):
        if self._is_2d():
            return FakeNdarray([[abs(v) for v in row] for row in self._data])
        return FakeNdarray([abs(x) for x in self._data])

    def __truediv__(self, other):
        if isinstance(other, FakeNdarray):
            sf = self._flat_values()
            of = other._flat_values()
            return FakeNdarray([a / b if b != 0 else 0.0 for a, b in zip(sf, of)])
        if other == 0:
            return FakeNdarray([0.0] * len(self._flat_values()))
        if self._is_2d():
            return FakeNdarray(
                [[v / other for v in row] for row in self._data]
            )
        return FakeNdarray([x / other for x in self._data])

    def __pow__(self, other):
        if self._is_2d():
            return FakeNdarray(
                [[v ** other for v in row] for row in self._data]
            )
        return FakeNdarray([x ** other for x in self._data])

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            flat = self._flat_values()
            return all(v == other for v in flat)
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, (int, float)):
            flat = self._flat_values()
            return any(v != other for v in flat)
        return NotImplemented

    def __float__(self):
        flat = self._flat_values()
        if len(flat) == 1:
            return float(flat[0])
        raise TypeError("only single-element arrays can be converted to float")

    def __bool__(self):
        flat = self._flat_values()
        if len(flat) == 1:
            return bool(flat[0])
        raise ValueError("truth value of array with more than one element is ambiguous")

    def __hash__(self):
        return id(self)

    def copy(self):
        return FakeNdarray(self._to_nested_list())

    def reshape(self, *args):
        if len(args) == 1 and isinstance(args[0], tuple):
            new_shape = args[0]
        else:
            new_shape = args
        flat = self._flat_values()
        if len(new_shape) == 2:
            rows, cols = new_shape
            if cols == -1:
                cols = len(flat) // rows
            if rows == -1:
                rows = len(flat) // cols
            if rows == 1:
                return FakeNdarray([flat[:cols]])
            data = []
            for i in range(rows):
                data.append(flat[i * cols: (i + 1) * cols])
            return FakeNdarray(data)
        if len(new_shape) == 1:
            return FakeNdarray(flat)
        return FakeNdarray(flat)

    def ravel(self):
        return FakeNdarray(self._flat_values())

    def flatten(self):
        return tuple(self._flat_values())

    def tolist(self):
        if self._is_2d():
            return [list(row) for row in self._data]
        return list(self._data)


# ---- numpy mock functions ----


def _np_array(data, dtype=None):
    if isinstance(data, FakeNdarray):
        return data
    return FakeNdarray(data, dtype=dtype)


def _np_zeros(shape, dtype=None):
    if isinstance(shape, (tuple, list)):
        if len(shape) == 1:
            return FakeNdarray([0.0] * shape[0])
        if len(shape) == 2:
            return FakeNdarray([[0.0] * shape[1] for _ in range(shape[0])])
    return FakeNdarray([0.0] * int(shape))


def _np_zeros_like(arr, dtype=None):
    s = arr.shape
    return _np_zeros(s, dtype=dtype)


def _np_ones(shape):
    if isinstance(shape, (tuple, list)):
        if len(shape) == 1:
            return FakeNdarray([1.0] * shape[0])
        return FakeNdarray([[1.0] * shape[1] for _ in range(shape[0])])
    return FakeNdarray([1.0] * int(shape))


def _np_arange(n):
    return FakeNdarray(list(range(int(n))))


def _np_linspace(start, stop, num):
    num = int(num)
    if num <= 1:
        return FakeNdarray([start])
    step = (stop - start) / (num - 1)
    return FakeNdarray([start + i * step for i in range(num)])


def _np_mean(arr, axis=None):
    if isinstance(arr, FakeNdarray):
        if axis is not None:
            if isinstance(axis, tuple):
                flat = arr._flat_values()
                if not flat:
                    return 0.0
                return sum(flat) / len(flat)
            if axis == 0 and arr._is_2d():
                n_rows = len(arr._data)
                n_cols = len(arr._data[0])
                result = []
                for j in range(n_cols):
                    s = sum(arr._data[i][j] for i in range(n_rows))
                    result.append(s / n_rows)
                return FakeNdarray(result)
            # axis=1
            if axis == 1 and arr._is_2d():
                return FakeNdarray(
                    [sum(row) / len(row) for row in arr._data]
                )
            return _np_mean(arr)
        flat = arr._flat_values()
        if not flat:
            return 0.0
        return sum(flat) / len(flat)
    if isinstance(arr, (list, tuple)):
        return sum(arr) / len(arr) if arr else 0.0
    return arr


def _np_var(arr, axis=None):
    if isinstance(arr, FakeNdarray):
        flat = arr._flat_values()
    elif isinstance(arr, (list, tuple)):
        flat = list(arr)
    else:
        return 0.0
    if not flat:
        return 0.0
    m = sum(flat) / len(flat)
    return sum((x - m) ** 2 for x in flat) / len(flat)


def _np_std(arr, axis=None):
    return math.sqrt(max(0, _np_var(arr, axis)))


def _np_sqrt(x):
    if isinstance(x, FakeNdarray):
        return FakeNdarray([math.sqrt(max(0, v)) for v in x._flat_values()])
    return math.sqrt(max(0, x))


def _np_abs(x):
    if isinstance(x, FakeNdarray):
        if x._is_2d():
            return FakeNdarray([[abs(v) for v in row] for row in x._data])
        return FakeNdarray([abs(v) for v in x._data])
    return abs(x)


def _np_clip(arr, a_min, a_max):
    if isinstance(arr, FakeNdarray):
        flat = arr._flat_values()
        clipped = [max(a_min, min(a_max, v)) for v in flat]
        if arr._is_2d():
            rows, cols = arr.shape
            return FakeNdarray(
                [clipped[i * cols: (i + 1) * cols] for i in range(rows)]
            )
        return FakeNdarray(clipped)
    return max(a_min, min(a_max, arr))


def _np_all(arr):
    if isinstance(arr, FakeNdarray):
        return all(arr._flat_values())
    return all(arr)


def _np_ceil(x):
    return math.ceil(x)


def _np_log(x):
    if isinstance(x, FakeNdarray):
        return FakeNdarray(
            [math.log(max(1e-300, v)) for v in x._flat_values()]
        )
    return math.log(max(1e-300, x))


def _np_exp(x):
    if isinstance(x, FakeNdarray):
        return FakeNdarray(
            [math.exp(min(700, v)) for v in x._flat_values()]
        )
    return math.exp(min(700, x))


def _np_concatenate(arrays, axis=0):
    result = []
    for arr in arrays:
        if isinstance(arr, FakeNdarray):
            result.extend(arr._data)
        elif isinstance(arr, (list, tuple)):
            result.extend(arr)
        else:
            result.append(arr)
    return FakeNdarray(result)


def _np_vstack(arrays):
    result = []
    for arr in arrays:
        if isinstance(arr, FakeNdarray):
            if arr._is_2d():
                result.extend(arr._data)
            else:
                result.append(list(arr._data))
        elif isinstance(arr, (list, tuple)):
            result.extend(arr)
        else:
            result.append(arr)
    return FakeNdarray(result)


def _np_column_stack(arrays):
    # Each array should be 1D of same length or ravel-able
    flat_arrays = []
    for arr in arrays:
        if isinstance(arr, FakeNdarray):
            flat_arrays.append(arr.ravel()._data)
        elif isinstance(arr, (list, tuple)):
            flat_arrays.append(list(arr))
        else:
            flat_arrays.append([arr])
    n = len(flat_arrays[0])
    result = []
    for i in range(n):
        row = [fa[i] for fa in flat_arrays]
        result.append(row)
    return FakeNdarray(result)


def _np_meshgrid(*arrays):
    if len(arrays) == 2:
        a, b = arrays
        a_data = a._data if isinstance(a, FakeNdarray) else list(a)
        b_data = b._data if isinstance(b, FakeNdarray) else list(b)
        r1 = FakeNdarray([[x for _ in b_data] for x in a_data])
        r2 = FakeNdarray([[y for y in b_data] for _ in a_data])
        return [r1, r2]
    return [arrays[0]]


# ---- numpy.random mock ----


class _FakeRandomState:
    def seed(self, s):
        _stdlib_random.seed(s)

    def uniform(self, low, high, size=None):
        if size is None:
            return _stdlib_random.uniform(low, high)
        if isinstance(size, (tuple, list)):
            if len(size) == 2:
                return FakeNdarray(
                    [[_stdlib_random.uniform(low, high) for _ in range(size[1])]
                     for _ in range(size[0])]
                )
            return FakeNdarray(
                [_stdlib_random.uniform(low, high) for _ in range(size[0])]
            )
        return FakeNdarray(
            [_stdlib_random.uniform(low, high) for _ in range(size)]
        )

    def normal(self, loc, scale, size=None):
        if size is None:
            return _stdlib_random.gauss(loc, scale)
        return FakeNdarray(
            [_stdlib_random.gauss(loc, scale) for _ in range(size)]
        )

    def random(self, size=None):
        if size is None:
            return _stdlib_random.random()
        return FakeNdarray(
            [_stdlib_random.random() for _ in range(size)]
        )

    def randint(self, low, high, size=None):
        if size is None:
            return _stdlib_random.randint(low, high - 1)
        if isinstance(size, int):
            return FakeNdarray(
                [_stdlib_random.randint(low, high - 1) for _ in range(size)]
            )
        return FakeNdarray(
            [_stdlib_random.randint(low, high - 1) for _ in range(size)]
        )

    def permutation(self, n):
        lst = list(range(n))
        _stdlib_random.shuffle(lst)
        return FakeNdarray(lst)


# ---- Build mock numpy module ----

_mock_np = MagicMock()
_mock_np.ndarray = FakeNdarray
_fake_random = _FakeRandomState()

_mock_np.array = _np_array
_mock_np.zeros = _np_zeros
_mock_np.zeros_like = _np_zeros_like
_mock_np.ones = _np_ones
_mock_np.arange = _np_arange
_mock_np.linspace = _np_linspace
_mock_np.mean = _np_mean
_mock_np.var = _np_var
_mock_np.std = _np_std
_mock_np.sqrt = _np_sqrt
_mock_np.abs = _np_abs
_mock_np.clip = _np_clip
_mock_np.all = _np_all
_mock_np.ceil = _np_ceil
_mock_np.log = _np_log
_mock_np.exp = _np_exp
_mock_np.concatenate = _np_concatenate
_mock_np.vstack = _np_vstack
_mock_np.column_stack = _np_column_stack
_mock_np.meshgrid = _np_meshgrid
_mock_np.random = _fake_random
_mock_np.isclose = lambda a, b, rtol=1e-5: abs(a - b) <= rtol * abs(b)
_mock_np.isnan = lambda x: False
_mock_np.pi = math.pi
_mock_np.inf = float("inf")

sys.modules.setdefault("numpy", _mock_np)

# ---- Mock matplotlib ----
_mock_fig = MagicMock()
_mock_ax = MagicMock()
_mock_plt = MagicMock()
_mock_plt.subplots.return_value = (_mock_fig, _mock_ax)
_mock_mpl = MagicMock()
sys.modules.setdefault("matplotlib", _mock_mpl)
sys.modules.setdefault("matplotlib.pyplot", _mock_plt)
_mock_mpl.pyplot = _mock_plt

# ---- Mock pandas ----
_mock_pd = MagicMock()
_mock_pd.Timestamp.now.return_value.isoformat.return_value = "2025-01-01T00:00:00"
sys.modules.setdefault("pandas", _mock_pd)

# ---- Mock scipy ----
_mock_scipy = MagicMock()
_mock_scipy_stats = MagicMock()
_mock_qmc = MagicMock()

# Make qmc samplers dimension-aware via a factory


def _make_sampler_factory(cls_name):
    """Create a sampler mock factory that captures d and returns properly shaped samples."""
    def factory(d=2, seed=None, scramble=True):
        sampler = MagicMock()
        sampler.random.side_effect = lambda n: FakeNdarray(
            [[_stdlib_random.random() for _ in range(d)] for _ in range(n)]
        )
        return sampler
    return factory


_mock_qmc.LatinHypercube = _make_sampler_factory("LatinHypercube")
_mock_qmc.Sobol = _make_sampler_factory("Sobol")
_mock_qmc.Halton = _make_sampler_factory("Halton")

sys.modules.setdefault("scipy", _mock_scipy)
sys.modules.setdefault("scipy.stats", _mock_scipy_stats)
sys.modules["scipy.stats"].qmc = _mock_qmc
sys.modules.setdefault("scipy.stats.qmc", _mock_qmc)

# scipy.stats.norm / lognorm for distribution scaling
_mock_norm = MagicMock()


def _norm_ppf(x, loc=0, scale=1):
    if isinstance(x, FakeNdarray):
        vals = x._flat_values()
    elif isinstance(x, (list, tuple)):
        vals = list(x)
    else:
        vals = [x]
    return FakeNdarray([loc + scale * float(v) for v in vals])


_mock_norm.ppf = _norm_ppf
_mock_scipy_stats.norm = _mock_norm

_mock_lognorm = MagicMock()


def _lognorm_ppf(x, s=1, scale=1):
    if isinstance(x, FakeNdarray):
        vals = x._flat_values()
    elif isinstance(x, (list, tuple)):
        vals = list(x)
    else:
        vals = [x]
    return FakeNdarray([float(scale) * max(0.01, float(v)) for v in vals])


_mock_lognorm.ppf = _lognorm_ppf
_mock_scipy_stats.lognorm = _mock_lognorm

# ---- Load source modules ----

_src = os.path.join(os.path.dirname(__file__), "..", "..", "src")

# Load visualization_config first (dependency)
_spec_vc = importlib.util.spec_from_file_location(
    "config.visualization_config",
    os.path.join(_src, "config", "visualization_config.py"),
)
_mod_vc = importlib.util.module_from_spec(_spec_vc)
sys.modules.setdefault("config.visualization_config", _mod_vc)
_spec_vc.loader.exec_module(_mod_vc)

# Load sensitivity_analysis
_spec = importlib.util.spec_from_file_location(
    "config.sensitivity_analysis",
    os.path.join(_src, "config", "sensitivity_analysis.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["config.sensitivity_analysis"] = _mod
_spec.loader.exec_module(_mod)

SensitivityMethod = _mod.SensitivityMethod
SamplingMethod = _mod.SamplingMethod
ParameterBounds = _mod.ParameterBounds
ParameterDefinition = _mod.ParameterDefinition
SensitivityResult = _mod.SensitivityResult
ParameterSpace = _mod.ParameterSpace
SensitivityAnalyzer = _mod.SensitivityAnalyzer
SensitivityVisualizer = _mod.SensitivityVisualizer


# ---- Test helpers ----


def _make_params(n=2):
    params = []
    for i in range(n):
        params.append(
            ParameterDefinition(
                name=f"param_{i}",
                bounds=ParameterBounds(
                    min_value=0.0, max_value=10.0, nominal_value=5.0
                ),
                config_path=["cfg", f"p{i}"],
            )
        )
    return params


def _make_model_fn(output_names=None):
    if output_names is None:
        output_names = ["power"]

    def model_fn(params):
        n = params.shape[0]
        result = {}
        for name in output_names:
            result[name] = FakeNdarray(
                [_stdlib_random.uniform(0.5, 1.5) for _ in range(n)]
            )
        return result

    return model_fn


# ---- Tests ----


class TestEnums:
    def test_sensitivity_method_values(self):
        assert SensitivityMethod.ONE_AT_A_TIME.value == "one_at_a_time"
        assert SensitivityMethod.GRADIENT_BASED.value == "gradient_based"
        assert SensitivityMethod.MORRIS.value == "morris"
        assert SensitivityMethod.SOBOL.value == "sobol"
        assert SensitivityMethod.FAST.value == "fast"
        assert SensitivityMethod.DELTA.value == "delta"

    def test_sampling_method_values(self):
        assert SamplingMethod.RANDOM.value == "random"
        assert SamplingMethod.LATIN_HYPERCUBE.value == "latin_hypercube"
        assert SamplingMethod.SOBOL_SEQUENCE.value == "sobol_sequence"
        assert SamplingMethod.HALTON.value == "halton"
        assert SamplingMethod.GRID.value == "grid"


class TestParameterBounds:
    def test_valid_bounds(self):
        b = ParameterBounds(min_value=0.0, max_value=10.0)
        assert b.distribution == "uniform"
        assert b.nominal_value is None

    def test_invalid_bounds_min_ge_max(self):
        with pytest.raises(ValueError, match="Minimum value must be less"):
            ParameterBounds(min_value=10.0, max_value=5.0)

    def test_equal_bounds_raises(self):
        with pytest.raises(ValueError, match="Minimum value must be less"):
            ParameterBounds(min_value=5.0, max_value=5.0)

    def test_nominal_within_bounds(self):
        b = ParameterBounds(min_value=0.0, max_value=10.0, nominal_value=5.0)
        assert b.nominal_value == 5.0

    def test_nominal_outside_bounds(self):
        with pytest.raises(ValueError, match="Nominal value must be within"):
            ParameterBounds(min_value=0.0, max_value=10.0, nominal_value=15.0)


class TestParameterDefinition:
    def test_defaults(self):
        p = ParameterDefinition(
            name="test",
            bounds=ParameterBounds(min_value=0.0, max_value=1.0),
            config_path=["a", "b"],
        )
        assert p.description == ""
        assert p.units == ""
        assert p.category == "general"

    def test_custom(self):
        p = ParameterDefinition(
            name="flow",
            bounds=ParameterBounds(min_value=0.0, max_value=100.0),
            config_path=["control", "flow"],
            description="Flow rate",
            units="mL/h",
            category="control",
        )
        assert p.units == "mL/h"
        assert p.category == "control"


class TestSensitivityResult:
    def test_defaults(self):
        r = SensitivityResult(
            method=SensitivityMethod.SOBOL,
            parameter_names=["a", "b"],
            output_names=["power"],
        )
        assert r.n_samples == 0
        assert r.computation_time == 0.0
        assert r.first_order_indices is None
        assert r.morris_means is None
        assert r.local_sensitivities is None
        assert r.created_at is not None


class TestParameterSpace:
    def test_init_empty_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            ParameterSpace([])

    def test_init_basic(self):
        params = _make_params(3)
        ps = ParameterSpace(params)
        assert ps.n_parameters == 3
        assert len(ps.parameter_names) == 3

    def test_nominal_values_default(self):
        p = ParameterDefinition(
            name="test",
            bounds=ParameterBounds(min_value=0.0, max_value=10.0),
            config_path=[],
        )
        ps = ParameterSpace([p])
        assert ps.nominal_values[0] == 5.0

    def test_nominal_values_explicit(self):
        p = ParameterDefinition(
            name="test",
            bounds=ParameterBounds(min_value=0.0, max_value=10.0, nominal_value=3.0),
            config_path=[],
        )
        ps = ParameterSpace([p])
        assert ps.nominal_values[0] == 3.0

    def test_sample_random(self):
        ps = ParameterSpace(_make_params(2))
        samples = ps.sample(10, SamplingMethod.RANDOM, seed=42)
        assert samples.shape[0] == 10

    def test_sample_latin_hypercube(self):
        ps = ParameterSpace(_make_params(2))
        samples = ps.sample(10, SamplingMethod.LATIN_HYPERCUBE, seed=42)
        assert samples.shape[0] == 10

    def test_sample_sobol_sequence(self):
        ps = ParameterSpace(_make_params(2))
        samples = ps.sample(8, SamplingMethod.SOBOL_SEQUENCE, seed=42)
        assert samples.shape[0] == 8

    def test_sample_halton(self):
        ps = ParameterSpace(_make_params(2))
        samples = ps.sample(10, SamplingMethod.HALTON, seed=42)
        assert samples.shape[0] == 10

    def test_sample_grid(self):
        ps = ParameterSpace(_make_params(2))
        samples = ps.sample(10, SamplingMethod.GRID)
        assert len(samples) >= 1

    def test_sample_no_seed(self):
        ps = ParameterSpace(_make_params(2))
        samples = ps.sample(10, SamplingMethod.RANDOM)
        assert samples.shape[0] == 10

    def test_scale_samples_normal(self):
        p = ParameterDefinition(
            name="test",
            bounds=ParameterBounds(
                min_value=1.0, max_value=10.0, distribution="normal"
            ),
            config_path=[],
        )
        ps = ParameterSpace([p])
        unit = FakeNdarray([[0.5]])
        scaled = ps._scale_samples(unit)
        assert scaled.shape == (1, 1)

    def test_scale_samples_lognormal(self):
        p = ParameterDefinition(
            name="test",
            bounds=ParameterBounds(
                min_value=1.0, max_value=10.0, distribution="lognormal"
            ),
            config_path=[],
        )
        ps = ParameterSpace([p])
        unit = FakeNdarray([[0.5]])
        scaled = ps._scale_samples(unit)
        assert scaled.shape == (1, 1)

    def test_get_parameter_by_name(self):
        params = _make_params(3)
        ps = ParameterSpace(params)
        p = ps.get_parameter_by_name("param_1")
        assert p.name == "param_1"

    def test_get_parameter_by_name_not_found(self):
        ps = ParameterSpace(_make_params(2))
        with pytest.raises(ValueError, match="Parameter not found"):
            ps.get_parameter_by_name("nonexistent")


class TestSensitivityAnalyzer:
    def setup_method(self):
        self.params = _make_params(2)
        self.ps = ParameterSpace(self.params)
        self.model_fn = _make_model_fn(["power"])
        self.analyzer = SensitivityAnalyzer(self.ps, self.model_fn, ["power"])

    def test_init(self):
        assert self.analyzer.cache_enabled is True
        assert self.analyzer.parameter_space is self.ps

    def test_evaluate_model_cached(self):
        samples = FakeNdarray([[5.0, 5.0]])
        out1 = self.analyzer._evaluate_model(samples)
        out2 = self.analyzer._evaluate_model(samples)
        assert "power" in out1
        assert "power" in out2

    def test_evaluate_model_no_cache(self):
        self.analyzer.cache_enabled = False
        samples = FakeNdarray([[5.0, 5.0]])
        out = self.analyzer._evaluate_model(samples)
        assert "power" in out

    def test_analyze_oat(self):
        result = self.analyzer.analyze_sensitivity(
            SensitivityMethod.ONE_AT_A_TIME, n_samples=5
        )
        assert result.method == SensitivityMethod.ONE_AT_A_TIME
        assert result.local_sensitivities is not None
        assert "power" in result.local_sensitivities
        assert result.n_samples == 5

    def test_analyze_gradient(self):
        result = self.analyzer.analyze_sensitivity(
            SensitivityMethod.GRADIENT_BASED
        )
        assert result.method == SensitivityMethod.GRADIENT_BASED
        assert result.local_sensitivities is not None

    def test_analyze_morris(self):
        result = self.analyzer.analyze_sensitivity(
            SensitivityMethod.MORRIS, n_samples=3
        )
        assert result.method == SensitivityMethod.MORRIS
        assert result.morris_means is not None
        assert result.morris_stds is not None
        assert result.morris_means_star is not None

    def test_analyze_sobol(self):
        result = self.analyzer.analyze_sensitivity(
            SensitivityMethod.SOBOL, n_samples=8
        )
        assert result.method == SensitivityMethod.SOBOL
        assert result.first_order_indices is not None
        assert result.total_order_indices is not None

    def test_analyze_unsupported_method(self):
        with pytest.raises(ValueError, match="Method not implemented"):
            self.analyzer.analyze_sensitivity(SensitivityMethod.FAST)

    def test_oat_zero_baseline(self):
        def zero_model(params):
            n = params.shape[0]
            return {"power": _np_zeros(n)}

        analyzer = SensitivityAnalyzer(self.ps, zero_model, ["power"])
        analyzer.cache_enabled = False
        result = analyzer.analyze_sensitivity(
            SensitivityMethod.ONE_AT_A_TIME, n_samples=5
        )
        assert result.local_sensitivities is not None

    def test_oat_empty_output(self):
        def empty_model(params):
            return {"power": FakeNdarray([])}

        analyzer = SensitivityAnalyzer(self.ps, empty_model, ["power"])
        analyzer.cache_enabled = False
        result = analyzer.analyze_sensitivity(
            SensitivityMethod.ONE_AT_A_TIME, n_samples=5
        )
        assert result.local_sensitivities is not None

    def test_gradient_empty_output(self):
        def empty_model(params):
            return {"power": FakeNdarray([])}

        analyzer = SensitivityAnalyzer(self.ps, empty_model, ["power"])
        analyzer.cache_enabled = False
        result = analyzer.analyze_sensitivity(SensitivityMethod.GRADIENT_BASED)
        assert result.local_sensitivities is not None

    def test_sobol_zero_variance(self):
        def constant_model(params):
            n = params.shape[0]
            return {"power": FakeNdarray([5.0] * n)}

        analyzer = SensitivityAnalyzer(self.ps, constant_model, ["power"])
        analyzer.cache_enabled = False
        result = analyzer.analyze_sensitivity(
            SensitivityMethod.SOBOL, n_samples=8
        )
        assert result.first_order_indices is not None

    def test_sobol_non_sobol_sampling(self):
        result = self.analyzer.analyze_sensitivity(
            SensitivityMethod.SOBOL,
            n_samples=8,
            sampling_method=SamplingMethod.RANDOM,
        )
        assert result.first_order_indices is not None

    def test_sobol_missing_output(self):
        def partial_model(params):
            n = params.shape[0]
            return {"other": FakeNdarray([1.0] * n)}

        analyzer = SensitivityAnalyzer(self.ps, partial_model, ["power", "other"])
        analyzer.cache_enabled = False
        result = analyzer.analyze_sensitivity(
            SensitivityMethod.SOBOL, n_samples=8
        )
        assert "other" in result.first_order_indices

    def test_sobol_multidim_output(self):
        def multidim_model(params):
            n = params.shape[0]
            arr = FakeNdarray(
                [[_stdlib_random.random() for _ in range(3)] for _ in range(n)]
            )
            return {"power": arr}

        analyzer = SensitivityAnalyzer(self.ps, multidim_model, ["power"])
        analyzer.cache_enabled = False
        result = analyzer.analyze_sensitivity(
            SensitivityMethod.SOBOL, n_samples=8
        )
        assert result.first_order_indices is not None

    def test_rank_parameters_total_order(self):
        result = self.analyzer.analyze_sensitivity(
            SensitivityMethod.SOBOL, n_samples=8
        )
        ranking = self.analyzer.rank_parameters(result, "power", "total_order")
        assert len(ranking) == 2

    def test_rank_parameters_first_order(self):
        result = self.analyzer.analyze_sensitivity(
            SensitivityMethod.SOBOL, n_samples=8
        )
        ranking = self.analyzer.rank_parameters(result, "power", "first_order")
        assert len(ranking) == 2

    def test_rank_parameters_morris_mean_star(self):
        result = self.analyzer.analyze_sensitivity(
            SensitivityMethod.MORRIS, n_samples=3
        )
        ranking = self.analyzer.rank_parameters(
            result, "power", "morris_mean_star"
        )
        assert len(ranking) == 2

    def test_rank_parameters_local(self):
        result = self.analyzer.analyze_sensitivity(
            SensitivityMethod.ONE_AT_A_TIME, n_samples=5
        )
        ranking = self.analyzer.rank_parameters(result, "power", "local")
        assert len(ranking) == 2

    def test_analyze_sobol_with_sobol_sequence(self):
        """Cover lines 661-663: Sobol analysis using SOBOL_SEQUENCE sampling."""
        result = self.analyzer.analyze_sensitivity(
            SensitivityMethod.SOBOL,
            n_samples=8,
            sampling_method=SamplingMethod.SOBOL_SEQUENCE,
        )
        assert result.method == SensitivityMethod.SOBOL
        assert result.first_order_indices is not None
        assert result.total_order_indices is not None

    def test_rank_parameters_unavailable_metric(self):
        result = SensitivityResult(
            method=SensitivityMethod.SOBOL,
            parameter_names=["a"],
            output_names=["power"],
        )
        with pytest.raises(ValueError, match="Metric .* not available"):
            self.analyzer.rank_parameters(result, "power", "total_order")


class TestSensitivityVisualizer:
    def test_init_default(self):
        viz = SensitivityVisualizer()
        assert viz.config is not None

    def test_init_with_config(self):
        vc = _mod_vc.VisualizationConfig()
        viz = SensitivityVisualizer(vc)
        assert viz.config is vc

    def test_plot_sensitivity_indices(self):
        viz = SensitivityVisualizer()
        result = SensitivityResult(
            method=SensitivityMethod.SOBOL,
            parameter_names=["a", "b"],
            output_names=["power"],
            first_order_indices={"power": FakeNdarray([0.3, 0.5])},
            total_order_indices={"power": FakeNdarray([0.4, 0.6])},
        )
        fig = viz.plot_sensitivity_indices(result, "power")
        assert fig is not None

    def test_plot_sensitivity_indices_with_save(self, tmp_path):
        viz = SensitivityVisualizer()
        result = SensitivityResult(
            method=SensitivityMethod.SOBOL,
            parameter_names=["a"],
            output_names=["power"],
            first_order_indices={"power": FakeNdarray([0.3])},
            total_order_indices={"power": FakeNdarray([0.4])},
        )
        save_path = str(tmp_path / "test.png")
        fig = viz.plot_sensitivity_indices(result, "power", save_path=save_path)
        assert fig is not None

    def test_plot_sensitivity_indices_no_indices(self):
        viz = SensitivityVisualizer()
        result = SensitivityResult(
            method=SensitivityMethod.SOBOL,
            parameter_names=["a"],
            output_names=["power"],
        )
        fig = viz.plot_sensitivity_indices(result, "power")
        assert fig is not None

    def test_plot_morris_results(self):
        viz = SensitivityVisualizer()
        result = SensitivityResult(
            method=SensitivityMethod.MORRIS,
            parameter_names=["a", "b"],
            output_names=["power"],
            morris_means_star={"power": FakeNdarray([0.1, 0.5])},
            morris_stds={"power": FakeNdarray([0.05, 0.1])},
        )
        fig = viz.plot_morris_results(result, "power")
        assert fig is not None

    def test_plot_morris_results_with_save(self, tmp_path):
        viz = SensitivityVisualizer()
        result = SensitivityResult(
            method=SensitivityMethod.MORRIS,
            parameter_names=["a"],
            output_names=["power"],
            morris_means_star={"power": FakeNdarray([0.1])},
            morris_stds={"power": FakeNdarray([0.05])},
        )
        save_path = str(tmp_path / "morris.png")
        fig = viz.plot_morris_results(result, "power", save_path=save_path)
        assert fig is not None

    def test_plot_morris_results_no_data(self):
        viz = SensitivityVisualizer()
        result = SensitivityResult(
            method=SensitivityMethod.MORRIS,
            parameter_names=["a"],
            output_names=["power"],
        )
        with pytest.raises(ValueError, match="Morris results not available"):
            viz.plot_morris_results(result, "power")

    def test_plot_parameter_ranking(self):
        viz = SensitivityVisualizer()
        result = SensitivityResult(
            method=SensitivityMethod.SOBOL,
            parameter_names=["a", "b", "c"],
            output_names=["power"],
            total_order_indices={"power": FakeNdarray([0.1, 0.5, 0.3])},
        )
        fig = viz.plot_parameter_ranking(result, "power")
        assert fig is not None

    def test_plot_parameter_ranking_with_bar_labels(self):
        """Cover line 962: bar label iteration in plot_parameter_ranking."""
        viz = SensitivityVisualizer()
        result = SensitivityResult(
            method=SensitivityMethod.SOBOL,
            parameter_names=["a", "b"],
            output_names=["power"],
            total_order_indices={"power": FakeNdarray([0.3, 0.7])},
        )
        # Make barh return iterable mock bars
        bar1 = MagicMock()
        bar1.get_width.return_value = 0.3
        bar1.get_y.return_value = 0.0
        bar1.get_height.return_value = 0.8
        bar2 = MagicMock()
        bar2.get_width.return_value = 0.7
        bar2.get_y.return_value = 1.0
        bar2.get_height.return_value = 0.8
        _mock_ax.barh.return_value = [bar1, bar2]
        fig = viz.plot_parameter_ranking(result, "power")
        assert fig is not None
        # Reset barh to default
        _mock_ax.barh.return_value = MagicMock()

    def test_plot_parameter_ranking_top_n(self):
        viz = SensitivityVisualizer()
        result = SensitivityResult(
            method=SensitivityMethod.SOBOL,
            parameter_names=["a", "b", "c"],
            output_names=["power"],
            total_order_indices={"power": FakeNdarray([0.1, 0.5, 0.3])},
        )
        fig = viz.plot_parameter_ranking(result, "power", top_n=2)
        assert fig is not None

    def test_plot_parameter_ranking_with_save(self, tmp_path):
        viz = SensitivityVisualizer()
        result = SensitivityResult(
            method=SensitivityMethod.SOBOL,
            parameter_names=["a"],
            output_names=["power"],
            total_order_indices={"power": FakeNdarray([0.5])},
        )
        save_path = str(tmp_path / "rank.png")
        fig = viz.plot_parameter_ranking(result, "power", save_path=save_path)
        assert fig is not None
