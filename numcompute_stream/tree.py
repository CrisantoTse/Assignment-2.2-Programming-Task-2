
from dataclasses import dataclass
from numbers import Real
from time import perf_counter

import numpy as np

from numcompute.stats import _safe_midpoint

from ._validation import integer, label_kind, labels, numeric, table


@dataclass(slots=True)
class _Node:
    probabilities: np.ndarray
    feature: int = -1
    threshold: float = 0.0
    missing_left: bool = False
    left: "_Node | None" = None
    right: "_Node | None" = None


def _impurity(counts, criterion):
    totals = counts.sum(axis=-1, keepdims=True)
    probabilities = counts / np.maximum(totals, 1)
    if criterion == "gini":
        return 1 - np.sum(probabilities * probabilities, axis=-1)
    log_probabilities = np.zeros_like(probabilities)
    np.log2(probabilities, out=log_probabilities, where=probabilities > 0)
    return -np.sum(probabilities * log_probabilities, axis=-1)


def _split_scores(column, y, total, criterion="gini", min_leaf=1):
    present = np.flatnonzero(~np.isnan(column))
    if present.size == 0:
        return np.empty(0), np.empty((0, 2))
    order = present[np.argsort(column[present], kind="stable")]
    ordered = column[order]
    boundaries = np.flatnonzero(ordered[:-1] < ordered[1:]) + 1
    mid = _safe_midpoint(ordered[boundaries - 1], ordered[boundaries])
    mid = np.where(mid < ordered[boundaries], mid, ordered[boundaries - 1])
    flags = np.zeros((len(order) + 1, total.size), dtype=np.int64)
    flags[np.arange(1, len(order) + 1), y[order]] = 1
    prefix = np.cumsum(flags, axis=0)
    if len(order) != len(column):
        boundaries = np.r_[0, boundaries, len(order)]
        mid = np.r_[-np.inf, mid, ordered[-1]]
    left = np.stack((prefix[boundaries], prefix[boundaries] + total - prefix[-1]), axis=1)
    right = total - left
    n_left, n_right = left.sum(axis=-1), right.sum(axis=-1)
    scores = (n_left * _impurity(left, criterion) +
              n_right * _impurity(right, criterion)) / len(column)
    scores[(n_left < min_leaf) | (n_right < min_leaf)] = np.inf
    return mid, scores


class DecisionTreeClassifier:

    def __init__(self, max_depth=5, min_samples_split=2, max_features=None,
                 buffer_size=500, *, criterion="gini", min_samples_leaf=1,
                 random_state=0):
        self.max_depth = None if max_depth is None else integer(max_depth, "max_depth", 0)
        self.min_samples_split = integer(min_samples_split, "min_samples_split", 2)
        self.min_samples_leaf = integer(min_samples_leaf, "min_samples_leaf")
        self.buffer_size = integer(buffer_size, "buffer_size")
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'.")
        self.criterion = criterion
        self.max_features = max_features
        self._feature_count(None)
        self.random_state = None if random_state is None else integer(random_state, "random_state", 0)
        self.reset()

    def reset(self):
        self._root = None
        self.classes_ = None
        self.n_features_in_ = None
        self._raw_width = None
        self._label_kind = None
        self._buffer = []
        self.pending_samples_ = 0
        self.retrain_count_ = 0
        self.last_train_size_ = 0
        self.last_train_seconds_ = 0.0
        return self

    @property
    def is_fitted(self):
        return self._root is not None

    def _feature_count(self, width):
        option = self.max_features
        if option is None:
            return width
        if isinstance(option, str):
            if option not in ("sqrt", "log2"):
                raise ValueError("String max_features must be 'sqrt' or 'log2'.")
            if width is None:
                return None
            return max(1, int(np.sqrt(width) if option == "sqrt" else np.log2(width)))
        if isinstance(option, (int, np.integer)) and not isinstance(option, (bool, np.bool_)):
            result = integer(option, "max_features")
            if width is not None and result > width:
                raise ValueError(f"max_features cannot exceed the feature count ({width}).")
            return result
        if isinstance(option, Real) and not isinstance(option, (bool, np.bool_)) and 0 < option <= 1:
            return None if width is None else max(1, int(option * width))
        raise ValueError("max_features must be None, a positive integer, a fraction in (0, 1], 'sqrt' or 'log2'.")

    def _build(self, X, y):
        classes, codes = np.unique(y, return_inverse=True)
        count = self._feature_count(X.shape[1])
        rng = np.random.default_rng(self.random_state)
        root = None
        work = [(np.arange(len(X)), 0, None, None)]
        while work:
            rows, depth, parent, side = work.pop()
            node_y = codes[rows]
            total = np.bincount(node_y, minlength=classes.size)
            node = _Node(total / len(rows))
            if parent is None:
                root = node
            else:
                setattr(parent, side, node)
            if (len(rows) < max(self.min_samples_split, 2 * self.min_samples_leaf)
                    or np.count_nonzero(total) == 1
                    or (self.max_depth is not None and depth >= self.max_depth)):
                continue
            features = (range(X.shape[1]) if count == X.shape[1] else
                        np.sort(rng.choice(X.shape[1], size=count, replace=False)))
            candidates = []
            for feature in features:
                thresholds, scores = _split_scores(X[rows, feature], node_y, total,
                                                   self.criterion, self.min_samples_leaf)
                if scores.size == 0:
                    continue
                feature_score = float(scores.min())
                if np.isfinite(feature_score):
                    candidates.append((int(feature), thresholds, scores, feature_score))
            if not candidates:
                continue
            best_score = min(candidate[3] for candidate in candidates)
            current_score = float(_impurity(total, self.criterion))
            if current_score - best_score <= 1e-12:
                continue
            for feature, thresholds, scores, feature_score in candidates:
                if feature_score <= best_score + 1e-12:
                    break
            threshold_index, missing_left = np.argwhere(scores <= best_score + 1e-12)[0]
            node.feature = feature
            node.threshold = float(thresholds[threshold_index])
            node.missing_left = bool(missing_left)
            column = X[rows, feature]
            goes_left = np.where(np.isnan(column), node.missing_left, column <= node.threshold)
            work.append((rows[~goes_left], depth + 1, node, "right"))
            work.append((rows[goes_left], depth + 1, node, "left"))
        return root, classes

    def _store(self, X, y, *, raw_width, new_run=False):
        started = perf_counter()
        root, classes = self._build(X, y)
        seconds = perf_counter() - started
        if new_run:
            self.reset()
        self._root = root
        self.classes_ = classes
        self.n_features_in_ = X.shape[1]
        self._raw_width = raw_width
        self._label_kind = label_kind(y)
        self.last_train_seconds_ = seconds
        self.last_train_size_ = len(X)
        self.retrain_count_ += 1
        self._buffer.clear()
        self.pending_samples_ = 0

    def fit(self, X, y):
        X = numeric(X, allow_empty=False)
        y = labels(y, size=len(X))
        self._store(X, y, raw_width=X.shape[1], new_run=True)
        return self

    def partial_fit(self, X_chunk, y_chunk):
        return self._partial_fit(X_chunk, y_chunk)

    def _partial_fit(self, X, y, prepare=None):
        if prepare is None:
            X = numeric(X, self._raw_width)
        else:
            X = table(X, self._raw_width)
        y = labels(y, size=len(X))
        if not len(X):
            return self
        family = label_kind(y)
        if self._label_kind is not None and family != self._label_kind:
            raise TypeError("Label dtype family changed between chunks.")
        size = self.pending_samples_ + len(X)
        if size < self.buffer_size:
            self._buffer.append((X.copy(), y.copy()))
            self.pending_samples_ = size
            self._raw_width = X.shape[1]
            self._label_kind = family
        else:
            if self._buffer:
                all_x = np.concatenate([chunk_x for chunk_x, _ in self._buffer] + [X])
                all_y = np.concatenate([chunk_y for _, chunk_y in self._buffer] + [y])
            else:
                all_x, all_y = X, y
            train_x = all_x
            if prepare is not None:
                train_x = numeric(prepare(all_x))
            if len(train_x) != len(all_y):
                raise ValueError("Transformers must preserve the sample count.")
            self._store(train_x, all_y, raw_width=X.shape[1])
        return self

    def predict_proba(self, X):
        if not self.is_fitted:
            raise RuntimeError("No fitted tree. Call fit or buffer enough samples with partial_fit.")
        X = numeric(X, self.n_features_in_)
        result = np.empty((len(X), self.classes_.size))
        work = [(self._root, np.arange(len(X)))]
        while work:
            node, rows = work.pop()
            if rows.size == 0:
                continue
            if node.feature == -1:
                result[rows] = node.probabilities
                continue
            column = X[rows, node.feature]
            left = np.where(np.isnan(column), node.missing_left, column <= node.threshold)
            work.append((node.right, rows[~left]))
            work.append((node.left, rows[left]))
        return result

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


DecisionTree = DecisionTreeClassifier
