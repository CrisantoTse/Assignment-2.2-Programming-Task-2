
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ._validation import label_kind, labels


def _metric(values):
    metric = np.asarray(values, dtype=float)
    if metric.ndim != 1 or np.isinf(metric).any():
        raise ValueError("Metric values must be a 1D sequence without infinity.")
    return metric


def _axes(ax, figsize):
    return plt.subplots(figsize=figsize, layout="constrained") if ax is None else (ax.figure, ax)


def _finish(fig, ax, save_path, show):
    ax.grid(axis="y", alpha=0.24)
    ax.spines[["top", "right"]].set_visible(False)
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_metric_over_time(metric_values, title, ylabel, *, save_path=None, show=False, ax=None):
    values = _metric(metric_values)
    fig, ax = _axes(ax, (8, 3.4))
    ax.plot(np.arange(1, values.size + 1), values, color="#007F86", marker="o", ms=3)
    ax.set(xlabel="Chunk", ylabel=ylabel, title=title)
    return _finish(fig, ax, save_path, show)


def compare_models(metric1, metric2, labels, *, save_path=None, show=False, ax=None,
                   title="Streaming model comparison", ylabel="Metric"):
    m1, m2 = _metric(metric1), _metric(metric2)
    if len(m1) != len(m2) or isinstance(labels, str) or len(labels) != 2:
        raise ValueError("Provide two equally long histories and two legend labels.")
    fig, ax = _axes(ax, (8, 3.4))
    chunks = np.arange(1, m1.size + 1)
    ax.plot(chunks, m1, label=labels[0], color="#007F86", lw=2)
    ax.plot(chunks, m2, label=labels[1], color="#C16B22", lw=2, ls="--")
    ax.set(xlabel="Chunk", ylabel=ylabel, title=title)
    ax.legend(frameon=False)
    return _finish(fig, ax, save_path, show)


def plot_predictions_vs_ground_truth(y_true, y_pred, *, save_path=None, show=False, ax=None):
    actual = labels(y_true, name="y_true")
    predicted = labels(y_pred, name="y_pred", size=actual.size)
    if actual.size and label_kind(actual) != label_kind(predicted):
        raise TypeError("True and predicted labels need compatible dtype families.")
    classes = np.unique(np.r_[actual, predicted])
    indexes = np.arange(1, actual.size + 1)
    fig, ax = _axes(ax, (8, 3.2))
    ax.scatter(indexes, np.searchsorted(classes, actual), label="Actual", color="#007F86", s=24)
    ax.scatter(indexes, np.searchsorted(classes, predicted), label="Predicted", color="#C16B22", s=26, marker="x")
    ax.set(yticks=np.arange(len(classes)), yticklabels=classes.astype(str), xlabel="Sample in chunk",
           ylabel="Class", title="Latest scored chunk: predictions vs actual labels")
    ax.legend(frameon=False, loc="best")
    return _finish(fig, ax, save_path, show)
