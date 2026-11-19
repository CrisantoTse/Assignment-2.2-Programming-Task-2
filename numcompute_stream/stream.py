
from collections import deque
from dataclasses import fields, is_dataclass
from sys import getsizeof
from time import perf_counter

import numpy as np

from ._validation import labels, table
from .metrics import StreamingClassificationMetrics
from .pipeline import Pipeline


def _owned_bytes(root):
    seen, pending, size = set(), [root], 0
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        size += getsizeof(item)
        if isinstance(item, np.ndarray):
            if item.base is not None:
                pending.append(item.base)
            if item.dtype.kind == "O":
                pending.extend(item.flat)
        elif isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, (tuple, list, deque, set)):
            pending.extend(item)
        elif is_dataclass(item):
            pending.extend(getattr(item, field.name) for field in fields(item))
        elif hasattr(item, "__dict__") and not isinstance(item, type):
            pending.append(vars(item))
    return size


class StreamTrainer:

    def __init__(self, model=None, *, pipeline=None, metrics=None, window_size=256, positive_label=1):
        if (model is None) == (pipeline is None):
            raise ValueError("Provide exactly one model or pipeline.")
        self.pipeline = pipeline if pipeline is not None else model
        self.model = self.pipeline.model if isinstance(self.pipeline, Pipeline) else self.pipeline
        if (self.model is None or not all(callable(getattr(self.pipeline, method, None))
                for method in ("partial_fit", "predict")) or not hasattr(self.pipeline, "is_fitted")):
            raise TypeError("Trainer needs a streaming estimator with is_fitted, partial_fit and predict.")
        self.metrics = (StreamingClassificationMetrics(window_size, positive_label=positive_label)
                        if metrics is None else metrics)
        self.logs = []
        self.last_predictions_ = None
        self.last_truth_ = None

    def score_chunk(self, X, y):
        X = table(X, allow_empty=False)
        y = labels(y, size=len(X))
        if not self.pipeline.is_fitted:
            return {"scored": False, "accuracy": np.nan, "predictions": None, "scores": None}
        if callable(getattr(self.model, "predict_proba", None)) and getattr(self.model, "classes_", None) is not None:
            proba = self.pipeline.predict_proba(X)
            classes = self.model.classes_
            preds = classes[np.argmax(proba, axis=1)]
            positive = np.flatnonzero(classes == self.metrics.positive_label)
            scores = proba[:, positive[0]] if positive.size else np.zeros(len(X))
        else:
            preds = self.pipeline.predict(X)
            scores = None
        return {"scored": True, "accuracy": float(np.mean(y == preds)),
                "predictions": preds, "scores": scores}

    def fit_chunk(self, X, y):
        X = table(X, allow_empty=False)
        y = labels(y, size=len(X))
        check = self.score_chunk(X, y)
        previous = getattr(self.model, "retrain_count_", 0)
        started = perf_counter()
        self.pipeline.partial_fit(X, y)
        update_time = perf_counter() - started
        retrains = getattr(self.model, "retrain_count_", 0)
        retrained = retrains > previous
        if check["scored"]:
            self.metrics.update(y, check["predictions"], check["scores"])
            self.last_truth_ = y.copy()
            self.last_predictions_ = check["predictions"].copy()
        cumulative = self.metrics.result()
        rolling = self.metrics.rolling_result() if self.metrics.window_size else cumulative
        row = {"chunk": len(self.logs) + 1, "samples": len(X), "scored": check["scored"],
               "accuracy": check["accuracy"], "error_rate": 1.0 - check["accuracy"],
               "cumulative_accuracy": cumulative["accuracy"], "rolling_accuracy": rolling["accuracy"],
               "cumulative_auc": cumulative["auc"], "scored_samples": cumulative["count"],
               "pending_samples": getattr(self.model, "pending_samples_", 0),
               "retrained": retrained, "retrains": retrains,
               "train_seconds": self.model.last_train_seconds_ if retrained else 0.0,
               "update_seconds": update_time, "memory_bytes": _owned_bytes(self.pipeline)}
        self.logs.append(row)
        return row.copy()
