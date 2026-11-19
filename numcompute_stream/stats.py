
from builtins import min as builtins_min
from collections import deque

import numpy as np

from numcompute.stats import histogram, max, mean, median, min, quantile, std, variance

from ._validation import integer, numeric

__all__ = ["StreamingStats", "update_stats", "mean", "variance", "std",
           "min", "max", "median", "quantile", "histogram"]


class _Moments:

    def __init__(self):
        self.count = self.mean = self.m2 = None

    def update(self, X):
        count = np.sum(~np.isnan(X), axis=0)
        if self.count is not None and count.shape != self.count.shape:
            raise ValueError("The number of reduced features changed.")
        if not len(X):
            return
        valid = ~np.isnan(X)
        observed = np.where(valid, X, 0.0)
        scale = np.max(np.abs(observed), axis=0)
        scale = np.where(scale == 0, 1.0, scale)
        try:
            with np.errstate(over="raise", invalid="raise"):
                scaled_mean = np.sum(observed / scale, axis=0) / np.maximum(count, 1)
                batch_mean = np.clip(scaled_mean, -1, 1) * scale
                reference = X[np.argmax(valid, axis=0), np.arange(X.shape[1])]
                reference = np.where(count > 0, reference, 0.0)
                with np.errstate(over="ignore", invalid="ignore"):
                    offset = np.where(valid, observed - reference, 0.0)
                    corrected = reference + np.sum(offset / np.maximum(count, 1), axis=0)
                batch_mean = np.where(np.isfinite(corrected), corrected, batch_mean)
                deviations = np.where(valid, observed - batch_mean, 0.0)
                batch_m2 = np.sum(deviations * deviations, axis=0)
                if self.count is None:
                    new_count, new_mean, new_m2 = count, batch_mean, batch_m2
                else:
                    new_count = self.count + count
                    overlap = (self.count > 0) & (count > 0)
                    delta = np.where(overlap, batch_mean, 0.0) - np.where(overlap, self.mean, 0.0)
                    weight = count / np.maximum(new_count, 1)
                    new_mean = np.where(self.count == 0, batch_mean, self.mean + delta * weight)
                    correction = delta * np.sqrt(self.count * weight)
                    new_m2 = self.m2 + batch_m2 + correction * correction
        except FloatingPointError as exc:
            raise ValueError("Running moments exceed float64 range.") from exc
        self.count, self.mean, self.m2 = new_count, new_mean, new_m2

    def variance(self, ddof=0):
        denom = self.count - ddof
        out = np.full_like(self.mean, np.nan)
        return np.divide(self.m2, denom, out=out, where=denom > 0)


class StreamingStats:

    def __init__(self, axis=0, *, window_size=None, retain=True, nan_policy="omit"):
        if axis is not None:
            if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)) or axis != 0:
                raise ValueError("StreamingStats axis must be 0 or None.")
        if nan_policy not in ("omit", "raise", "propagate"):
            raise ValueError("nan_policy must be 'omit', 'raise' or 'propagate'.")
        self.axis = axis
        self.window_size = None if window_size is None else integer(window_size, "window_size")
        self.retain = retain
        self.nan_policy = nan_policy
        self.reset()

    def reset(self):
        self._width = None
        self._moments = _Moments()
        self._has_nan = None
        self._history = deque()
        self._history_count = 0
        return self

    def update_stats(self, X_chunk):
        array = np.asarray(X_chunk)
        if array.ndim == 1:
            array = array[:, None]
        X = numeric(array, self._width)
        if not len(X):
            return self
        data = X.reshape(-1, 1) if self.axis is None else X
        any_nan = np.any(np.isnan(data), axis=0)
        if self.nan_policy == "raise" and np.any(any_nan):
            raise ValueError("X contains NaN under nan_policy='raise'.")
        self._moments.update(data)
        self._has_nan = any_nan if self._has_nan is None else self._has_nan | any_nan
        self._width = X.shape[1]
        if self.retain:
            self._history.append(X)
            self._history_count += len(X)
            if self.window_size is not None:
                while self._history_count > self.window_size:
                    first = self._history.popleft()
                    remove = builtins_min(len(first), self._history_count - self.window_size)
                    if remove < len(first):
                        self._history.appendleft(first[remove:].copy())
                    self._history_count -= remove
        return self

    update = partial_fit = update_stats

    def _shape(self, array):
        return array[0] if self.axis is None else array

    def _apply_nan(self, values):
        if self.nan_policy == "propagate":
            values = np.where(self._has_nan, np.nan, values)
        return self._shape(values)

    @property
    def count(self):
        count = self._moments.count
        return 0 if count is None else self._shape(count.copy())

    @property
    def mean(self):
        if self._moments.count is None:
            return np.nan
        return self._apply_nan(np.where(self._moments.count > 0, self._moments.mean, np.nan))

    def variance(self, ddof=0):
        ddof = integer(ddof, "ddof", 0)
        if self._moments.count is None:
            return np.nan
        return self._apply_nan(self._moments.variance(ddof))

    def std(self, ddof=0):
        return np.sqrt(self.variance(ddof))

    def _retained_rows(self):
        if not self.retain:
            raise RuntimeError("Distribution queries require retain=True.")
        if not self._history:
            raise RuntimeError("Update with at least one row before querying distributions.")
        return np.concatenate(list(self._history), axis=0)

    def quantile(self, q, method="linear"):
        return quantile(self._retained_rows(), q, axis=self.axis,
                        method=method, nan_policy=self.nan_policy)

    def histogram(self, bins=10, range=None):
        X = self._retained_rows()
        if self.axis is None:
            return histogram(X.ravel(), bins=bins, range=range, nan_policy=self.nan_policy)
        return [histogram(column, bins=bins, range=range, nan_policy=self.nan_policy)
                for column in X.T]

    def result(self, ddof=0):
        return {"count": self.count, "mean": self.mean, "variance": self.variance(ddof)}


def update_stats(X_chunk, accumulator=None):
    state = StreamingStats() if accumulator is None else accumulator
    return state.update_stats(X_chunk)
