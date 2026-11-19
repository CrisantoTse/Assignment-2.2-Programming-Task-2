import csv
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from numcompute.benchmarking import benchmark, environment_info
from numcompute.utils import _validate_integer
from benchmark.baselines import (
    loop_mean, loop_standardize, loop_mse, full_sort_topk,
)
from numcompute.stats import mean
from numcompute.preprocessing import StandardScaler
from numcompute.metrics import mse
from numcompute.sort_search import topk
import subprocess
import json
from datetime import datetime, timezone

def check_equivalent(expected, actual, *, exact=False):
    if not isinstance(exact, (bool, np.bool_)):
        raise TypeError("exact must be a boolean.")

    if isinstance(expected, tuple) != isinstance(actual, tuple):
        raise AssertionError("Output structures differ.")

    expected_parts = expected if isinstance(expected, tuple) else (expected,)
    actual_parts = actual if isinstance(actual, tuple) else (actual,)

    if len(expected_parts) != len(actual_parts):
        raise AssertionError("Output counts differ.")

    for wanted, received in zip(expected_parts, actual_parts):
        if np.shape(wanted) != np.shape(received):
            raise AssertionError("Output shapes differ.")

        if exact:
            assert_array_equal(received, wanted)
        else:
            assert_allclose(
                received, wanted,
                rtol=1e-10, atol=1e-12, equal_nan=False,
            )

def run_case(case_id, baseline, candidate, args=(),
             exact=False, warmup=3, repeats=10):
    warmup = _validate_integer(warmup, name="warmup", minimum=0)
    repeats = _validate_integer(repeats, name="repeats", minimum=1)

    expected = baseline(*args)
    actual = candidate(*args)
    check_equivalent(expected, actual, exact=exact)

    baseline_timing = benchmark(
        baseline, *args, warmup=warmup, repeats=repeats
    )
    candidate_timing = benchmark(
        candidate, *args, warmup=warmup, repeats=repeats
    )

    baseline_seconds = baseline_timing["median_seconds"]
    candidate_seconds = candidate_timing["median_seconds"]

    if (baseline_seconds > 0 and candidate_seconds > 0
            and np.isfinite(baseline_seconds)
            and np.isfinite(candidate_seconds)):
        speedup = baseline_seconds / candidate_seconds
    else:
        speedup = None

    return {
        "case_id": case_id,
        "equivalent": True,
        "baseline": baseline_timing,
        "candidate": candidate_timing,
        "speedup": speedup,
    }

def prepare_inputs(n=1000, seed=42):
    n = _validate_integer(n, name="n", minimum=10)
    seed = _validate_integer(seed, name="seed", minimum=0)

    rng = np.random.default_rng(seed)

    values = rng.normal(size=n)
    features = rng.normal(size=(n, 10))
    y_true = rng.normal(size=n)
    y_pred = rng.normal(size=n)

    features[:, 0] = 7.0

    return {
        "values": values,
        "features": features,
        "y_true": y_true,
        "y_pred": y_pred,
    }

def project_standardize(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def run_experiments(n=1000, seed=42, warmup=3, repeats=10):
    warmup = _validate_integer(warmup, name="warmup", minimum=0)
    repeats = _validate_integer(repeats, name="repeats", minimum=1)

    data = prepare_inputs(n, seed)
    values = data["values"]
    features = data["features"]
    y_true = data["y_true"]
    y_pred = data["y_pred"]
    n = values.size

    cases = [
        ("mean", loop_mean, mean, (values,), False),
        ("standardization", loop_standardize, project_standardize,
         (features,), False),
        ("mse", loop_mse, mse, (y_true, y_pred), False),
        ("topk_fixed", full_sort_topk, topk, (values, 10), True),
        ("topk_fraction", full_sort_topk, topk,
         (values, n // 10), True),
    ]

    results = []

    for case in cases:
        name, baseline, candidate, args, exact = case

        result = run_case(
            name + "_" + str(n),
            baseline, candidate,
            args=args, exact=exact,
            warmup=warmup, repeats=repeats,
        )

        result["task"] = name
        result["shape"] = list(args[0].shape)
        result["dtype"] = str(args[0].dtype)
        result["baseline_name"] = baseline.__name__
        result["candidate_name"] = candidate.__name__

        if exact:
            result["task"] = "topk"
            result["k"] = args[1]
            result["comparison"] = "full_sort_vs_argpartition"
        else:
            result["k"] = None
            result["comparison"] = "loop_vs_vectorized"

        results.append(result)

    return results

def save_raw_timings(results, path, run_id):
    if not isinstance(run_id, str):
        raise TypeError("run_id must be a string.")

    if not run_id.strip():
        raise ValueError("run_id must not be empty.")

    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "run_id", "case_id", "implementation",
            "repeat_index", "seconds_per_call",
        ])

        for result in results:
            for implementation in ("baseline", "candidate"):
                times = result[implementation]["times_seconds"]

                for repeat_index, seconds in enumerate(times, start=1):
                    writer.writerow([
                        run_id,
                        result["case_id"],
                        implementation,
                        repeat_index,
                        seconds,
                    ])

    return csv_path

def save_summary(results, path, run_id):
   
    if not isinstance(run_id, str):
        raise TypeError("run_id must be a string.")

    if not run_id.strip():
        raise ValueError("run_id must not be empty.")

    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "run_id", "case_id", "task", "shape", "dtype", "k",
            "comparison", "baseline_name", "candidate_name",
            "baseline_median_seconds", "candidate_median_seconds",
            "baseline_min_seconds", "candidate_min_seconds",
            "speedup", "equivalent",
        ])

        for result in results:
            baseline = result["baseline"]
            candidate = result["candidate"]

            writer.writerow([
                run_id,
                result["case_id"],
                result["task"],
                result["shape"],
                result["dtype"],
                result["k"],
                result["comparison"],
                result["baseline_name"],
                result["candidate_name"],
                baseline["median_seconds"],
                candidate["median_seconds"],
                baseline["min_seconds"],
                candidate["min_seconds"],
                result["speedup"],
                result["equivalent"],
            ])

    return csv_path

def git_info():
    root = Path(__file__).resolve().parent.parent

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )

        status = subprocess.run(
            [
                "git", "--no-optional-locks", "status",
                "--porcelain", "--untracked-files=all",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )

    except (OSError, subprocess.SubprocessError) as error:
        return {
            "commit": "unknown",
            "dirty": None,
            "status": "unknown",
            "error": str(error),
        }

    if status.stdout.strip():
        dirty = True
    else:
        dirty = False

    return {
        "commit": commit.stdout.strip(),
        "dirty": dirty,
        "status": status.stdout,
        "error": None,
    }

def save_environment(info, path):
    if not isinstance(info, dict):
        raise TypeError("info must be a dictionary.")

    content = json.dumps(
        info,
        indent=4,
        ensure_ascii=False,
        allow_nan=False,
    )

    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("x", encoding="utf-8") as stream:
        stream.write(content)

    return json_path

def build_run_info(run_id, sizes, seed=42, warmup=3, repeats=10):
    if not isinstance(run_id, str):
        raise TypeError("run_id must be a string.")

    if not run_id.strip():
        raise ValueError("run_id must not be empty.")

    if not isinstance(sizes, (list, tuple)):
        raise TypeError("sizes must be a list or tuple.")

    if not sizes:
        raise ValueError("sizes must not be empty.")

    checked_sizes = []
    for n in sizes:
        n = _validate_integer(n, name="n", minimum=10)
        checked_sizes.append(n)

    seed = _validate_integer(seed, name="seed", minimum=0)
    warmup = _validate_integer(warmup, name="warmup", minimum=0)
    repeats = _validate_integer(repeats, name="repeats", minimum=1)

    info = {
        "run_id": run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment_info(),
        "git": git_info(),
    }

    info["config"] = {
        "sizes": checked_sizes,
        "seed": seed,
        "warmup": warmup,
        "repeats": repeats,
        "calls_per_sample": 1,
        "dtype": "float64",
        "feature_count": 10,
        "constant_column": 0,
        "constant_value": 7.0,
        "topk_fixed": 10,
        "topk_fraction": "n // 10",
        "distribution": "standard normal",
        "rng": "numpy.random.default_rng, reset for each size",
        "generation_order": ["values", "features", "y_true", "y_pred"],
        "rtol": 1e-10,
        "atol": 1e-12,
        "topk_comparison": "exact values and indices",
    }

    info["timing"] = {
        "clock": "time.perf_counter",
        "included": "function validation, copying and allocation",
        "excluded": "input generation, correctness checks and file I/O",
        "standardization": "new scaler, fit and transform",
        "order": "baseline warmup and samples, then candidate",
    }

    return info

def main(sizes=(1000, 10000, 100000), seed=42,
         warmup=3, repeats=10, output_dir=None):
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "results"
    else:
        output_dir = Path(output_dir)

    run_id = datetime.now(timezone.utc).strftime(
        "run_%Y%m%dT%H%M%S_%fZ"
    )

    info = build_run_info(run_id, sizes, seed, warmup, repeats)
    config = info["config"]
    results = []

    for n in config["sizes"]:
        print("Running n =", n)

        size_results = run_experiments(
            n=n,
            seed=config["seed"],
            warmup=config["warmup"],
            repeats=config["repeats"],
        )
        results.extend(size_results)

    info["cases"] = []
    for result in results:
        info["cases"].append({
            "case_id": result["case_id"],
            "shape": result["shape"],
            "dtype": result["dtype"],
            "k": result["k"],
            "comparison": result["comparison"],
        })

    info["measurements_finished_at_utc"] = (
        datetime.now(timezone.utc).isoformat()
    )

    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    save_raw_timings(results, run_dir / "raw_timings.csv", run_id)
    save_summary(results, run_dir / "summary.csv", run_id)
    save_environment(info, run_dir / "environment.json")

    print("Results saved to:", run_dir)
    return run_dir


if __name__ == "__main__":
    main()