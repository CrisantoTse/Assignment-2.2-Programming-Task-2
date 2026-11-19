
from collections import deque

import numpy as np

from numcompute.metrics import accuracy, confusion_matrix, f1, mse, precision, recall

from ._validation import integer, label_kind, labels

__all__ = ["StreamingClassificationMetrics", "StreamingMetrics", "accuracy", "precision",
           "recall", "f1", "confusion_matrix", "mse", "roc_auc"]


def _matrix(actual, predicted, classes):
    k = classes.size
    ids = k * np.searchsorted(classes, actual) + np.searchsorted(classes, predicted)
    return np.bincount(ids, minlength=k * k).reshape(k, k)


def _ratios(numerator, denominator):
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float),
                     where=denominator != 0)


def _auc_counts(positives, negatives):
    p, n = positives.sum(), negatives.sum()
    if p == 0 or n == 0:
        return np.nan
    weaker_negatives = np.cumsum(negatives) - negatives
    return float(np.dot(positives / p, (weaker_negatives + 0.5 * negatives) / n))


def _auc(y_true, y_score, positive_label):
    scores, codes = np.unique(y_score, return_inverse=True)
    positive = y_true == positive_label
    p = np.bincount(codes[positive], minlength=len(scores))
    n = np.bincount(codes[~positive], minlength=len(scores))
    return _auc_counts(p, n)


def _scores(y_score, size):
    scores = np.asarray(y_score)
    if scores.shape != (size,) or scores.dtype.kind not in "biuf" or not np.isfinite(scores).all():
        raise ValueError("y_score must be a finite real array with one score per row.")
    return scores.astype(float)


def roc_auc(y_true, y_score, positive_label=1):
    y_true = labels(y_true, name="y_true")
    if np.unique(y_true).size > 2:
        raise ValueError("ROC AUC requires at most two classes.")
    return _auc(y_true, _scores(y_score, y_true.size), positive_label)


class StreamingClassificationMetrics:

    def __init__(self, window_size=None, *, positive_label=1, track_auc=True):
        self.window_size = None if window_size is None else integer(window_size, "window_size")
        self.positive_label = positive_label
        self.track_auc = track_auc
        self.reset()

    def reset(self):
        self.classes_ = np.empty(0, dtype=int)
        self.confusion_matrix_ = np.zeros((0, 0), dtype=np.int64)
        self._kind = None
        self._score_counts = {}
        self._recent = deque()
        self._recent_count = 0
        return self

    def update(self, y_true_chunk, y_pred_chunk, y_score=None):
        actual = labels(y_true_chunk, name="y_true_chunk")
        predicted = labels(y_pred_chunk, name="y_pred_chunk", size=actual.size)
        if not actual.size:
            if y_score is not None:
                _scores(y_score, 0)
            return self
        kind = label_kind(actual)
        if kind != label_kind(predicted) or (self._kind is not None and kind != self._kind):
            raise TypeError("True and predicted labels must keep a compatible dtype family.")
        score = ((predicted == self.positive_label).astype(float) if y_score is None else
                 _scores(y_score, actual.size))
        classes = np.unique(np.concatenate([actual, predicted] + ([self.classes_] if self._kind else [])))
        counts = _matrix(actual, predicted, classes)
        if self._kind:
            old_positions = np.searchsorted(classes, self.classes_)
            counts[np.ix_(old_positions, old_positions)] += self.confusion_matrix_
        if self.track_auc and len(classes) <= 2:
            for is_positive in (False, True):
                keys, freq = np.unique(score[(actual == self.positive_label) == is_positive], return_counts=True)
                for value, number in zip(keys, freq):
                    p, n = self._score_counts.setdefault(float(value), [0, 0])
                    self._score_counts[float(value)] = [p + int(number), n] if is_positive else [p, n + int(number)]
        else:
            self._score_counts.clear()
        if self.window_size is not None:
            length = min(actual.size, self.window_size)
            self._recent.append((actual[-length:].copy(), predicted[-length:].copy(), score[-length:].copy()))
            self._recent_count += length
            while self._recent_count > self.window_size:
                first = self._recent.popleft()
                remove = min(len(first[0]), self._recent_count - self.window_size)
                if remove < len(first[0]):
                    self._recent.appendleft(tuple(v[remove:].copy() for v in first))
                self._recent_count -= remove
        self.classes_, self.confusion_matrix_, self._kind = classes, counts, kind
        return self

    partial_fit = update

    def _summary(self, counts, auc, average):
        if average not in (None, "macro", "micro", "weighted", "binary"):
            raise ValueError("Unsupported average; use None, macro, micro, weighted or binary.")
        diagonal = counts.diagonal()
        support = counts.sum(axis=1)
        predicted = counts.sum(axis=0)
        total = int(support.sum())
        acc = float(diagonal.sum() / total) if total else np.nan

        def reduce_metric(values):
            if average is None:
                return values
            if not total:
                return np.nan
            if average == "micro":
                return acc
            if average == "weighted":
                return float(np.average(values, weights=support))
            if average == "binary":
                if self.classes_.size > 2:
                    raise ValueError("Binary averaging requires at most two classes.")
                which = np.flatnonzero(self.classes_ == self.positive_label)
                return float(values[which[0]]) if which.size else 0.0
            return float(values[(support + predicted) > 0].mean())

        return {"count": total, "accuracy": acc, "error_rate": 1.0 - acc,
                "precision": reduce_metric(_ratios(diagonal, predicted)),
                "recall": reduce_metric(_ratios(diagonal, support)),
                "f1": reduce_metric(_ratios(2 * diagonal, predicted + support)),
                "confusion_matrix": counts.copy(), "labels": self.classes_.copy(), "auc": auc}

    def result(self, average="macro"):
        values = [self._score_counts[key] for key in sorted(self._score_counts)]
        auc = _auc_counts(*np.asarray(values, dtype=np.int64).T) if values else np.nan
        return self._summary(self.confusion_matrix_, auc, average)

    def rolling_result(self, average="macro"):
        if self.window_size is None:
            raise RuntimeError("Set window_size to request rolling results.")
        if not self._recent:
            return self._summary(np.zeros_like(self.confusion_matrix_), np.nan, average)
        actual, predicted, scores = (np.concatenate(parts) for parts in zip(*self._recent))
        counts = _matrix(actual, predicted, self.classes_)
        auc = (_auc(actual, scores, self.positive_label) if self.track_auc and
               np.unique(np.r_[actual, predicted]).size <= 2 else np.nan)
        return self._summary(counts, auc, average)


StreamingMetrics = StreamingClassificationMetrics
