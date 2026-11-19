
from copy import deepcopy

from ._validation import labels, table
from .tree import DecisionTreeClassifier


class Pipeline:

    def __init__(self, steps):
        self.steps = list(steps)
        if not self.steps:
            raise ValueError("Pipeline needs at least one named step.")
        names, objects = zip(*self.steps)
        if not all(isinstance(name, str) and name for name in names) or len(set(names)) != len(names):
            raise ValueError("Pipeline step names must be unique nonempty strings.")
        if len({id(obj) for obj in objects}) != len(objects):
            raise ValueError("Each pipeline step must have an independent object.")
        self._has_model = callable(getattr(objects[-1], "predict", None))
        count = len(objects) - int(self._has_model)
        for name, step in self.steps[:count]:
            if not all(callable(getattr(step, method, None)) for method in ("fit", "partial_fit", "transform")):
                raise TypeError(f"Transformer {name!r} needs fit, partial_fit and transform.")
        if self._has_model and not all(callable(getattr(self.model, m, None)) for m in ("fit", "partial_fit")):
            raise TypeError("The final estimator needs fit and partial_fit.")
        self._active = None
        self._width = None

    @property
    def named_steps(self):
        return dict(self.steps)

    @property
    def model(self):
        return self.steps[-1][1] if self._has_model else None

    @property
    def _transformers(self):
        return self.steps[:-1] if self._has_model else self.steps

    @property
    def is_fitted(self):
        return self._active is not None

    @staticmethod
    def _apply(steps, X):
        for name, transformer in steps:
            X = transformer.transform(X)
        return X

    def _update_steps(self, X, y, method):
        draft = deepcopy(self._transformers)
        result = X
        for name, transformer in draft:
            getattr(transformer, method)(result, y)
            result = table(transformer.transform(result))
            if len(result) != len(X):
                raise ValueError(f"Transformer {name!r} must preserve the sample count.")
        return draft, result

    def _set_steps(self, draft):
        self.steps = draft + ([self.steps[-1]] if self._has_model else [])

    def fit(self, X, y=None):
        X = table(X, allow_empty=False)
        if self._has_model:
            y = labels(y, size=len(X))
        draft, result = self._update_steps(X, y, "fit")
        if self._has_model:
            self.model.fit(result, y)
        self._set_steps(draft)
        self._active, self._width = deepcopy(draft), X.shape[1]
        return self

    def partial_fit(self, X, y=None):
        X = table(X, self._width)
        if self._has_model:
            y = labels(y, size=len(X))
        if len(X) == 0:
            return self
        draft, result = self._update_steps(X, y, "partial_fit")
        if isinstance(self.model, DecisionTreeClassifier):
            retrains = self.model.retrain_count_
            self.model._partial_fit(X, y, prepare=lambda batch: self._apply(draft, batch))
            if self.model.retrain_count_ > retrains:
                self._active = deepcopy(draft)
        else:
            if self._has_model:
                self.model.partial_fit(result, y)
            self._active = deepcopy(draft)
        self._set_steps(draft)
        self._width = X.shape[1]
        return self

    def transform(self, X):
        if not self.is_fitted:
            raise RuntimeError("Pipeline is not fitted; fit or provide enough training samples.")
        return self._apply(self._active, table(X, self._width))

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def predict(self, X):
        if not self._has_model:
            raise RuntimeError("Pipeline does not have a final estimator.")
        return self.model.predict(self.transform(X))

    def predict_proba(self, X):
        method = getattr(self.model, "predict_proba", None)
        if method is None:
            raise RuntimeError("The final estimator does not support predict_proba.")
        return method(self.transform(X))
