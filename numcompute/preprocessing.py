
from numbers import Real
import numpy as np
from .utils import _as_float_array


class SimpleImputer:

    def __init__(self, strategy='constant', fill_value=0.0):
        self.strategy = strategy
        self.fill_value = fill_value
        self._validated_fill_value()

    def _validated_fill_value(self):
        if not isinstance(self.strategy, str):
            raise TypeError('strategy must be a string.')
        if self.strategy != 'constant':
            raise ValueError("Only strategy='constant' is supported.")
        if isinstance(self.fill_value, (bool, np.bool_)) or not isinstance(
                self.fill_value, Real):
            raise TypeError('fill_value must be a real numeric scalar.')
        try:
            with np.errstate(over='ignore'):
                value = float(self.fill_value)
        except (OverflowError, ValueError) as exc:
            raise ValueError('fill_value must be representable as finite float64.') from exc
        if not np.isfinite(value):
            raise ValueError('fill_value must be finite.')
        return value

    def fit(self, X, y=None):
        fill = self._validated_fill_value()
        array = _as_float_array(X, name='X', ndim=2)
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError('fit requires at least one sample and one feature.')
        self.n_features_in_ = array.shape[1]
        self.fill_value_ = fill
        return self

    def transform(self, X):
        if not hasattr(self, 'n_features_in_'):
            raise RuntimeError('SimpleImputer must be fitted before transform.')
        array = _as_float_array(X, name='X', ndim=2)
        if array.shape[1] != self.n_features_in_:
            raise ValueError(
                f'X must have {self.n_features_in_} features; got {array.shape[1]}.')
        array[np.isnan(array)] = self.fill_value_
        return array

    def fit_transform(self, X, y=None):
        return self.fit(X, y=y).transform(X)

class StandardScaler:

    def _validate_input(self, X, *, fitting=False):
        array = _as_float_array(X, name="X", ndim=2)

        if np.any(np.isinf(array)):
            raise ValueError("X must not contain infinity.")

        if fitting:
            if array.shape[0] == 0 or array.shape[1] == 0:
                raise ValueError("fit requires at least one sample and one feature.")

            if np.any(np.all(np.isnan(array), axis=0)):
                raise ValueError("Each training column must contain a non-NaN value.")

        return array

    def fit(self, X, y=None):
        array = self._validate_input(X, fitting=True)

        with np.errstate(over="ignore", invalid="ignore"):
            mean = np.nanmean(array, axis=0)
            scale = np.nanstd(array, axis=0, ddof=0)

        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
            raise ValueError("Training statistics exceed the supported numeric range.")

        scale = np.where(scale == 0.0, 1.0, scale)

        self.mean_ = mean
        self.scale_ = scale
        self.n_features_in_ = array.shape[1]

        return self

    def transform(self, X):
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("StandardScaler must be fitted before transform.")

        array = self._validate_input(X)

        if array.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have {self.n_features_in_} features; " 
                             f"got {array.shape[1]}."
            )

        with np.errstate(over="ignore", invalid="ignore"):
            result = (array - self.mean_) / self.scale_

        if np.any(np.isfinite(array) & ~np.isfinite(result)):
            raise ValueError("Transformed values exceed the supported numeric range.")

        return result

    def fit_transform(self, X, y=None):
        return self.fit(X, y=y).transform(X)

class MinMaxScaler:

    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self._validate_range()

    def _validate_range(self):
        if not isinstance(self.feature_range, (tuple, list)):
            raise TypeError("feature_range must be a tuple or list.")

        if len(self.feature_range) != 2:
            raise ValueError("feature_range must contain two bounds.")

        bounds = []

        for value in self.feature_range:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise TypeError("Each bound must be a real numeric scalar.")

            try:
                value = float(value)
            except (OverflowError, ValueError) as exc:
                raise ValueError("Bounds must be representable as finite float64.") from exc

            if not np.isfinite(value):
                raise ValueError("Bounds must be finite.")

            bounds.append(value)

        lower, upper = bounds

        if lower >= upper:
            raise ValueError("The lower bound must be less than the upper.")

        span = upper - lower

        if not np.isfinite(span):
            raise ValueError("The target range exceeds the numeric limit.")

        return lower, upper, span

    def _validate_input(self, X, *, fitting=False):
        array = _as_float_array(X, name="X", ndim=2)

        if np.any(np.isinf(array)):
            raise ValueError("X must not contain infinity.")

        if fitting:
            if array.shape[0] == 0 or array.shape[1] == 0:
                raise ValueError("fit requires at least one sample and one feature.")

            if np.any(np.all(np.isnan(array), axis=0)):
                raise ValueError("Each training column must contain a non-NaN value.")

        return array

    def fit(self, X, y=None):
        lower, upper, span = self._validate_range()
        array = self._validate_input(X, fitting=True)

        data_min = np.nanmin(array, axis=0)
        data_max = np.nanmax(array, axis=0)

        with np.errstate(over="ignore", invalid="ignore"):
            data_range = data_max - data_min

        if not np.all(np.isfinite(data_range)):
            raise ValueError("Training ranges exceed the supported numeric range.")

        effective_range = np.where(data_range == 0.0, 1.0, data_range)

        self.data_min_ = data_min
        self.data_max_ = data_max
        self.n_features_in_ = array.shape[1]

        self._effective_range = effective_range
        self._lower = lower
        self._span = span

        return self

    def transform(self, X):
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("MinMaxScaler must be fitted before transform.")

        array = self._validate_input(X)

        if array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X must have {self.n_features_in_} features; "
                f"got {array.shape[1]}."
            )

        with np.errstate(over="ignore", invalid="ignore"):
            result = (array - self.data_min_) / self._effective_range
            result = result * self._span + self._lower

        if np.any(np.isfinite(array) & ~np.isfinite(result)):
            raise ValueError("Transformed values exceed the supported numeric range.")

        return result

    def fit_transform(self, X, y=None):
        return self.fit(X, y=y).transform(X)

class OneHotEncoder:

    def __init__(self, handle_unknown="error"):
        self.handle_unknown = handle_unknown
        self._validate_option()

    def _validate_option(self):
        if not isinstance(self.handle_unknown, str):
            raise TypeError("handle_unknown must be a string.")

        if self.handle_unknown not in ("error", "ignore"):
            raise ValueError("handle_unknown must be 'error' or 'ignore'.")

        return self.handle_unknown

    def _category_key(self, value):
        if value is None:
            return None

        if isinstance(value, str):
            return value

        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(
                "Categories must be strings or real numbers, "
                "not booleans or other objects."
            )

        if value != value:
            return None

        if value == np.inf or value == -np.inf:
            raise ValueError("Categories must not contain infinity.")

        return value

    def _validate_input(self, X, *, fitting=False):
        try:
            array = np.array(X, dtype=object, copy=True)
        except ValueError as exc:
            raise ValueError("X must have a regular rectangular shape.") from exc

        if array.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if fitting:
            if array.shape[0] == 0 or array.shape[1] == 0:
                raise ValueError("fit requires at least one sample and one feature.")

        for column in range(array.shape[1]):
            column_type = None

            for row in range(array.shape[0]):
                key = self._category_key(array[row, column])

                if key is not None:
                    current_type = ("string" if isinstance(key, str) else "numeric")

                    if column_type is None:
                        column_type = current_type
                    elif current_type != column_type:
                        raise TypeError(f"Column {column} must not mix, strings and numbers.")

                array[row, column] = key

        return array

    def fit(self, X, y=None):
        option = self._validate_option()
        array = self._validate_input(X, fitting=True)

        categories = []
        mappings = []

        for column in range(array.shape[1]):
            mapping = {}
            values = []

            for key in array[:, column]:
                if key not in mapping:
                    mapping[key] = len(values)
                    values.append(key)

            categories.append(np.array(values, dtype=object))
            mappings.append(mapping)

        widths = [len(values) for values in categories]
        offsets = np.cumsum([0] + widths[:-1], dtype=np.intp)

        self.categories_ = categories
        self.n_features_in_ = array.shape[1]

        self._mappings = mappings
        self._offsets = offsets
        self._n_output_features = sum(widths)
        self._handle_unknown = option

        return self

    def transform(self, X):
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("OneHotEncoder must be fitted before transform.")

        array = self._validate_input(X)

        if array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X must have {self.n_features_in_} features; "
                f"got {array.shape[1]}."
            )

        result = np.zeros(
            (array.shape[0], self._n_output_features),
            dtype=np.float64,
        )

        for column, mapping in enumerate(self._mappings):
            offset = self._offsets[column]

            for row, key in enumerate(array[:, column]):
                index = mapping.get(key)

                if index is None:
                    if self._handle_unknown == "error":
                        raise ValueError(
                            f"Unknown category {key!r} "
                            f"in column {column}."
                        )
                    continue

                result[row, offset + index] = 1.0

        return result

    def fit_transform(self, X, y=None):
        return self.fit(X, y=y).transform(X)

class ColumnSelector:

    def __init__(self, columns, dtype=None):
        self.columns = columns
        self.dtype = dtype
        self._validated_columns()
        self._validated_dtype()

    def _validated_columns(self):
        if not isinstance(self.columns, (list, tuple, np.ndarray)):
            raise TypeError("columns must be a list, tuple or 1D array.")

        columns = np.asarray(self.columns, dtype=object)

        if columns.ndim != 1 or columns.size == 0:
            raise ValueError("columns must be a non-empty 1D sequence.")

        indices = []
        for column in columns:
            if isinstance(column, (bool, np.bool_)) or not isinstance(
                column, (int, np.integer)
            ):
                raise TypeError("columns must contain integers, not booleans.")

            column = int(column)
            if column < 0:
                raise ValueError("columns must contain non-negative indices.")

            indices.append(column)

        if len(set(indices)) != len(indices):
            raise ValueError("columns must not contain repeated indices.")

        return tuple(indices)

    def _validated_dtype(self):
        return None if self.dtype is None else np.dtype(self.dtype)

    @staticmethod
    def _as_table(X):
        array = np.asarray(X)
        if array.ndim != 2:
            raise ValueError("X must be a 2D table.")
        return array

    @staticmethod
    def _select(array, columns, dtype):
        result = array[:, list(columns)]
        if dtype is not None:
            result = result.astype(dtype, copy=True)
        return result

    def fit(self, X, y=None):
        array = self._as_table(X)
        columns = self._validated_columns()
        dtype = self._validated_dtype()

        if array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("fit requires at least one sample and feature.")

        if any(column >= array.shape[1] for column in columns):
            raise ValueError("columns contains an out-of-range index.")

        self._select(array, columns, dtype)

        self.columns_ = columns
        self.dtype_ = dtype
        self.n_features_in_ = array.shape[1]
        return self

    def transform(self, X):
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("ColumnSelector must be fitted before transform.")

        columns = self._validated_columns()
        dtype = self._validated_dtype()

        if dtype is None or self.dtype_ is None:
            dtype_changed = dtype is not self.dtype_
        else:
            dtype_changed = dtype != self.dtype_

        if columns != self.columns_ or dtype_changed:
            raise RuntimeError("ColumnSelector parameters changed; fit again.")

        array = self._as_table(X)

        if array.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have {self.n_features_in_} features; "
                             f"got {array.shape[1]}.")

        return self._select(array, self.columns_, self.dtype_)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)