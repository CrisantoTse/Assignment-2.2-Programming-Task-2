
import numpy as np

from numcompute.utils import _as_float_array, _validate_integer


def integer(value, name, minimum=1):
    return _validate_integer(value, name=name, minimum=minimum)


def table(X, width=None, *, allow_empty=True):
    array = np.asarray(X)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError("X must be a 2D array with at least one feature column.")
    if not allow_empty and not len(array):
        raise ValueError("X must contain at least one sample.")
    if width is not None and width != array.shape[1]:
        raise ValueError(f"X has {array.shape[1]} features; expected {width}.")
    return array


def numeric(X, width=None, *, allow_empty=True):
    array = _as_float_array(table(X, width, allow_empty=allow_empty), name="X")
    if np.isinf(array).any():
        raise ValueError("X must not contain infinity.")
    return array


def labels(y, *, name="y", size=None):
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if size is not None and y.size != size:
        raise ValueError(f"{name} has {y.size} labels; expected {size}.")
    if y.dtype.kind not in "biufUS":
        raise TypeError(f"{name} must contain numeric, boolean or string labels.")
    if y.dtype.kind == "f" and not np.isfinite(y).all():
        raise ValueError(f"{name} must not contain NaN or infinity.")
    return y


def label_kind(y):
    return y.dtype.kind if y.dtype.kind in "US" else "numeric"
