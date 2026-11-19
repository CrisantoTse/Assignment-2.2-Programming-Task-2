
from copy import copy
from numbers import Real

import numpy as np

from numcompute.preprocessing import ColumnSelector as _ColumnSelector
from numcompute.preprocessing import OneHotEncoder as _BatchEncoder

from ._validation import numeric, table
from .stats import _Moments


class _Incremental:
    n_features_in_ = None

    def fit(self, X, y=None):
        X = table(X, allow_empty=False)
        fresh = copy(self)
        fresh.n_features_in_ = None
        fresh.partial_fit(X, y)
        self.__dict__.update(fresh.__dict__)
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def _input(self, X):
        if self.n_features_in_ is None:
            raise RuntimeError(f"{type(self).__name__} must be fitted before transform.")
        return numeric(X, self.n_features_in_)


class StandardScaler(_Incremental):

    def partial_fit(self, X, y=None):
        X = numeric(X, self.n_features_in_)
        if len(X):
            moments = _Moments() if self.n_features_in_ is None else self._moments
            moments.update(X)
            self._moments = moments
            self.n_features_in_ = X.shape[1]
            self.n_samples_seen_ = moments.count.copy()
            self.mean_, self.var_ = moments.mean.copy(), moments.variance()
            self.scale_ = np.sqrt(np.where(self.var_ > 0, self.var_, 1.0))
        return self

    def transform(self, X):
        X = self._input(X)
        try:
            with np.errstate(over="raise", invalid="raise"):
                return (X - self.mean_) / self.scale_
        except FloatingPointError as exc:
            raise ValueError("Scaled values exceed float64 range.") from exc


class MinMaxScaler(_Incremental):

    def __init__(self, feature_range=(0, 1)):
        low, high = feature_range
        if not all(isinstance(v, Real) and not isinstance(v, (bool, np.bool_)) for v in (low, high)):
            raise TypeError("feature_range must contain two real numbers.")
        if not np.isfinite([low, high]).all() or not high > low:
            raise ValueError("feature_range must be finite and increasing.")
        self.feature_range = float(low), float(high)

    def partial_fit(self, X, y=None):
        X = numeric(X, self.n_features_in_)
        if not len(X):
            return self
        low = np.min(np.where(np.isnan(X), np.inf, X), axis=0)
        high = np.max(np.where(np.isnan(X), -np.inf, X), axis=0)
        if self.n_features_in_ is not None:
            low, high = np.minimum(self.data_min_, low), np.maximum(self.data_max_, high)
        self.data_min_, self.data_max_, self.n_features_in_ = low, high, X.shape[1]
        return self

    def transform(self, X):
        X = self._input(X)
        lo = np.where(np.isfinite(self.data_min_), self.data_min_, 0.0)
        hi = np.where(np.isfinite(self.data_max_), self.data_max_, 0.0)
        low, high = self.feature_range
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            span = hi - lo
            direct = (X - lo) / np.where(span == 0, 1.0, span)
            half_span = hi / 2 - lo / 2
            half = (X / 2 - lo / 2) / np.where(half_span == 0, 1.0, half_span)
            fraction = np.where(np.isinf(span) | np.isinf(direct), half, direct)
            result = (1 - fraction) * low + fraction * high
        if np.any(~np.isfinite(result) & ~np.isnan(X)):
            raise ValueError("Scaled values exceed float64 range.")
        return result


class Imputer(_Incremental):

    def __init__(self, strategy="mean", fill_value=0.0):
        if strategy not in ("mean", "constant"):
            raise ValueError("strategy must be 'mean' or 'constant'.")
        if not isinstance(fill_value, Real) or isinstance(fill_value, (bool, np.bool_)):
            raise TypeError("fill_value must be a real number.")
        if not np.isfinite(fill_value):
            raise ValueError("fill_value must be finite.")
        self.strategy, self.fill_value = strategy, float(fill_value)

    def partial_fit(self, X, y=None):
        X = numeric(X, self.n_features_in_)
        if not len(X):
            return self
        if self.strategy == "mean":
            count = np.sum(~np.isnan(X), axis=0)
            observed = np.where(np.isnan(X), 0.0, X)
            scale = np.max(np.abs(observed), axis=0)
            base = np.where(scale == 0, 1.0, scale)
            batch_mean = np.clip(np.sum(observed / base, axis=0) / np.maximum(count, 1), -1, 1) * base
            if self.n_features_in_ is None:
                old_count, old_mean = np.zeros_like(count), np.zeros_like(batch_mean)
            else:
                old_count, old_mean = self.n_samples_seen_, self.statistics_
            total = old_count + count
            weight = count / np.maximum(total, 1)
            with np.errstate(over="ignore", invalid="ignore"):
                combined = old_mean + (batch_mean - old_mean) * weight
            combined = np.where(np.isfinite(combined), combined,
                                (1 - weight) * old_mean + weight * batch_mean)
            combined = np.where(old_count == 0, batch_mean, combined)
            self.statistics_ = np.where(total > 0, combined, self.fill_value)
            self.n_samples_seen_ = total
        else:
            self.statistics_ = np.full(X.shape[1], self.fill_value)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = self._input(X)
        return np.where(np.isnan(X), self.statistics_, X)


class SimpleImputer(Imputer):

    def __init__(self, strategy="constant", fill_value=0.0):
        super().__init__(strategy, fill_value)


class OneHotEncoder(_Incremental):

    def __init__(self, handle_unknown="ignore"):
        self._validator = _BatchEncoder(handle_unknown)
        self.handle_unknown = handle_unknown

    def _codes(self, column):
        if column.dtype.kind in "iufU":
            if column.dtype.kind == "f" and np.isinf(column).any():
                raise ValueError("Categories must not contain infinity.")
            values, first, inverse = np.unique(column, return_index=True, return_inverse=True)
            keys = [None if isinstance(v, Real) and np.isnan(v) else v.item() for v in values]
            return keys, first, inverse
        keyed = [self._validator._category_key(value) for value in column]
        mapping = dict.fromkeys(keyed)
        for index, key in enumerate(mapping):
            mapping[key] = index
        inverse = np.fromiter((mapping[key] for key in keyed), dtype=np.intp, count=len(keyed))
        return list(mapping), np.arange(len(mapping)), inverse

    @staticmethod
    def _family(keys):
        kinds = {"string" if isinstance(key, str) else "numeric" for key in keys if key is not None}
        if len(kinds) > 1:
            raise TypeError("Each feature must contain either strings or numbers.")
        return next(iter(kinds), None)

    def partial_fit(self, X, y=None):
        X = table(X, self.n_features_in_)
        if not len(X):
            return self
        maps = []
        for i, column in enumerate(X.T):
            keys, first, _ = self._codes(column)
            mapping = {} if self.n_features_in_ is None else self._mappings[i].copy()
            self._family(list(mapping) + keys)
            for ix in np.argsort(first, kind="stable"):
                mapping.setdefault(keys[ix], len(mapping))
            maps.append(mapping)
        self._mappings = maps
        self.categories_ = [np.array(list(mapping), dtype=object) for mapping in maps]
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        if self.n_features_in_ is None:
            raise RuntimeError("OneHotEncoder must be fitted before transform.")
        X = table(X, self.n_features_in_)
        out = np.zeros((len(X), sum(map(len, self._mappings))))
        offset = 0
        for column, mapping in zip(X.T, self._mappings):
            keys, _, inverse = self._codes(column)
            self._family(list(mapping) + keys)
            local = np.array([mapping.get(key, -1) for key in keys], dtype=np.intp)[inverse]
            known = local >= 0
            if self.handle_unknown == "error" and not known.all():
                raise ValueError("X contains an unknown category.")
            out[np.flatnonzero(known), offset + local[known]] = 1
            offset += len(mapping)
        return out


class ColumnSelector(_ColumnSelector):

    def partial_fit(self, X, y=None):
        return self.fit(X, y)
