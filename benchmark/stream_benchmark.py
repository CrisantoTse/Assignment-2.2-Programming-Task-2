
import argparse
import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from demo.stream_demo import run_stream
from numcompute_stream.tree import _split_scores


def loop_split_scores(column, y, class_count):
    rows = list(zip(column.tolist(), y.tolist()))
    unique = sorted(set(column.tolist()))
    thresholds = [(a + b) / 2 for a, b in zip(unique, unique[1:])]
    scores = []
    for threshold in thresholds:
        left = [0] * class_count
        right = [0] * class_count
        for value, label in rows:
            if value <= threshold:
                left[label] += 1
            else:
                right[label] += 1
        n_left, n_right = sum(left), sum(right)
        g_left = 1 - sum((n / n_left) ** 2 for n in left)
        g_right = 1 - sum((n / n_right) ** 2 for n in right)
        scores.append((n_left * g_left + n_right * g_right) / len(rows))
    return np.asarray(thresholds), np.asarray(scores)


def vector_split_scores(column, y, class_count):
    thresholds, costs = _split_scores(column, y, np.bincount(y, minlength=class_count))
    return thresholds, costs[:, 0]


def median_seconds(task, repeats):
    values = []
    for _ in range(repeats):
        started = perf_counter()
        task()
        values.append(perf_counter() - started)
    return median(values)


def split_benchmark(repeats):
    rng = np.random.default_rng(2026)
    rows = []
    for size in (128, 512, 1536):
        column = rng.normal(size=size)
        codes = rng.integers(0, 3, size=size)
        loop = loop_split_scores(column, codes, 3)
        vector = vector_split_scores(column, codes, 3)
        np.testing.assert_allclose(loop[0], vector[0], atol=1e-14)
        np.testing.assert_allclose(loop[1], vector[1], atol=1e-14)
        loop_seconds = median_seconds(lambda: loop_split_scores(column, codes, 3), repeats)
        numpy_seconds = median_seconds(lambda: vector_split_scores(column, codes, 3), repeats)
        rows.append({"rows": size, "thresholds": len(loop[0]), "loop_ms": loop_seconds * 1e3,
                     "numpy_ms": numpy_seconds * 1e3, "speedup": loop_seconds / numpy_seconds})
    return rows


def buffer_benchmark(repeats):
    sizes = (64, 256, 512)
    rows = []
    for size in sizes:
        trainer, _ = run_stream(size)
        logs = trainer.logs
        common = [row for row in logs if row["chunk"] >= 9]
        assert all(row["scored"] for row in common)
        runtimes, costs = [], []
        for _ in range(repeats):
            started = perf_counter()
            sample, _ = run_stream(size)
            runtimes.append(perf_counter() - started)
            costs.append(sum(row["train_seconds"] for row in sample.logs))
        accuracy = np.average([row["accuracy"] for row in common], weights=[row["samples"] for row in common])
        drift = np.mean([row["accuracy"] for row in common if 17 <= row["chunk"] <= 20])
        retrains = trainer.model.retrain_count_
        rows.append({"buffer_size": size, "first_scored_chunk": next(r["chunk"] for r in logs if r["scored"]),
                     "matched_accuracy": float(accuracy), "drift_17_to_20_accuracy": float(drift),
                     "retrain_count": retrains, "total_tree_ms": median(costs) * 1e3,
                     "tree_ms_per_retrain": median(costs) / retrains * 1e3,
                     "stream_ms": median(runtimes) * 1e3,
                     "peak_retained_kib": max(r["memory_bytes"] for r in logs) / 1024})
    return rows


def main():
    parser = argparse.ArgumentParser(description='Measure split scoring and the retraining/accuracy tradeoff of buffer size.')
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("benchmark/stream_results.json"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    record = {"python": platform.python_version(), "numpy": np.__version__,
              "platform": f"{platform.system()} {platform.machine()}", "median_of": args.repeats,
              "split_scores": split_benchmark(args.repeats),
              "buffer_sizes": buffer_benchmark(args.repeats),
              "matched_chunk_interval": [9, 32], "chunk_size": 64}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
