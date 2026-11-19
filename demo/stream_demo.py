
import argparse
import csv
from pathlib import Path

import numpy as np

from numcompute_stream import DecisionTreeClassifier, Pipeline, StandardScaler, StreamingStats, StreamTrainer
from numcompute_stream.io import iter_supervised_csv
from numcompute_stream.visualise import compare_models, plot_metric_over_time, plot_predictions_vs_ground_truth

DATA = Path(__file__).with_name("stream_data.csv")


def run_stream(buffer_size, *, path=DATA, chunk_size=64):
    tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=4, buffer_size=buffer_size)
    pipe = Pipeline([("scale", StandardScaler()), ("tree", tree)])
    trainer = StreamTrainer(pipe, window_size=256)
    stats = StreamingStats(window_size=256)
    for X, y in iter_supervised_csv(path, chunk_size, skip_header=1):
        stats.update_stats(X)
        trainer.fit_chunk(X, y)
    return trainer, stats


def log_preview(trainer, *, chunks=(1, 2, 4, 5, 16, 17, 18, 20, 24, 32)):
    print(f"Buffer size {trainer.model.buffer_size}")
    print(f"{'Chunk':>5} {'Accuracy':>10} {'Overall':>10} {'Retrain':>8} {'Pending':>8} {'Memory KiB':>11}")
    for row in trainer.logs:
        if row["chunk"] not in chunks:
            continue
        if row["scored"]:
            accuracy = f"{row['accuracy']:.3f}"
            overall = f"{row['cumulative_accuracy']:.3f}"
        else:
            accuracy = "--"
            overall = "--"
        memory = row["memory_bytes"] / 1024
        print(f"{row['chunk']:>5} {accuracy:>10} {overall:>10} "
              f"{str(row['retrained']):>8} {row['pending_samples']:>8} {memory:>11.2f}")


def figures(fast, slow, output=None):
    def dest(name):
        return None if output is None else Path(output) / name

    def values(trainer, key):
        return [row[key] for row in trainer.logs]

    fig_accuracy, ax = plot_metric_over_time(values(fast, "accuracy"), "Accuracy by chunk", "Accuracy")
    ax.axvline(16.5, color="#737B85", ls=":", label="Rule changes at chunk 17")
    ax.set_ylim(-0.02, 1.04)
    ax.legend(frameon=False)
    if dest("accuracy.png") is not None:
        Path(output).mkdir(parents=True, exist_ok=True)
        fig_accuracy.savefig(dest("accuracy.png"), dpi=160, bbox_inches="tight")
    fig_error, _ = plot_metric_over_time(values(fast, "error_rate"), "Errors by chunk", "Error rate",
                                       save_path=dest("error.png"))
    fig_compare, comparison = compare_models(values(fast, "rolling_accuracy"), values(slow, "rolling_accuracy"),
                                           [f"Buffer {fast.model.buffer_size}", f"Buffer {slow.model.buffer_size}"],
                                           title="Small and large buffers", ylabel="Accuracy (last 256 scored rows)")
    comparison.axvline(16.5, color="#737B85", ls=":")
    comparison.set_ylim(-0.02, 1.04)
    if dest("buffer_comparison.png") is not None:
        fig_compare.savefig(dest("buffer_comparison.png"), dpi=160, bbox_inches="tight")
    fig_prediction, _ = plot_predictions_vs_ground_truth(fast.last_truth_, fast.last_predictions_,
                                                         save_path=dest("predictions.png"))
    return fig_accuracy, fig_error, fig_compare, fig_prediction


def main():
    parser = argparse.ArgumentParser(description='Run the Notebook experiment from the terminal, using only project dependencies.')
    parser.add_argument("--output", type=Path, default=Path("demo/artifacts"))
    parser.add_argument("--show", action="store_true", help="Open the plot window after saving files.")
    options = parser.parse_args()
    from matplotlib import pyplot as plt

    with np.printoptions(precision=3, suppress=True):
        fast, stats = run_stream(64)
        slow, _ = run_stream(256)
        print(f"CSV {DATA.name}. 2,048 generated rows. 64 rows per chunk.")
        log_preview(fast)
        log_preview(slow)
        result = fast.metrics.result()
        print(f"Scored rows {result['count']}. Overall accuracy {result['accuracy']:.3f}. AUC {result['auc']:.3f}.")
        print("Mean of each feature", stats.mean)
        print("Median of the last 256 rows", stats.quantile(0.5))
        figures(fast, slow, options.output)
        for trainer in (fast, slow):
            path = options.output / f"buffer_{trainer.model.buffer_size}_logs.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=trainer.logs[0])
                writer.writeheader()
                writer.writerows(trainer.logs)
        print(f"Saved four PNG figures and two CSV logs to {options.output}.")
    if options.show:
        plt.show()


if __name__ == "__main__":
    main()
