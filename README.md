# NumCompute-Stream

COMP5004 Assignment 2.2. This project adds streaming learning to the group NumCompute package.

## Run the project

Open a terminal in this folder and run these commands.

```sh
python -m pip install -e .
python -m demo.stream_demo
python -m unittest discover -q
python -m benchmark.stream_benchmark
```

Use Python 3.10 or later. The package needs NumPy and Matplotlib.
For the notebook, open `demo/stream_demo.ipynb` and click **Run All**.
Use `python -m demo.stream_demo --show` to open the saved plots in a window.

## How it works

1. Read 64 CSV rows at a time. Each group of rows is a chunk.
2. Use the current tree to predict the labels.
3. Save the scores and add the chunk to a buffer.
4. When the buffer is full, train a new tree on all buffered rows.
5. Clear the buffer and continue.

The first chunks fill the buffer before a tree is ready. Their accuracy is blank.
An unfinished final buffer stays pending.

## Small example

```python
from numcompute_stream import DecisionTreeClassifier, Pipeline, StandardScaler, StreamTrainer
from numcompute_stream.io import iter_supervised_csv

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("tree", DecisionTreeClassifier(max_depth=5, buffer_size=256)),
])
trainer = StreamTrainer(pipe)

for X, y in iter_supervised_csv("demo/stream_data.csv", 64, skip_header=1):
    log = trainer.fit_chunk(X, y)
    print(log["chunk"], log["accuracy"])
```

`X` has shape `(rows, features)`. `y` has one label per row.
`predict(X)` returns one label per row. `predict_proba(X)` returns one probability per class.

## Main files

| File in `numcompute_stream/` | What it does |
| --- | --- |
| `io.py` | Reads CSV chunks. Blank numeric cells become NaN. |
| `tree.py` | Fits one decision tree and supports buffered `partial_fit`. |
| `preprocessing.py` | Updates scalers, missing-value estimates and categories. |
| `stats.py` | Updates means and variances. Finds quantiles and histogram counts. |
| `metrics.py` | Updates accuracy, precision, recall, F1, confusion counts and binary AUC. |
| `pipeline.py` | Runs the data steps and the model in order. |
| `stream.py` | Scores and trains each chunk. Logs time, memory and accuracy. |
| `visualise.py` | Draws score, comparison and prediction plots. |

## Data, results and tests

The demo CSV contains 2,048 generated rows. The label rule changes at chunk 17.
Run `python -m demo.make_stream_data` to create the same data again.
The demo saves four plots and two log files in `demo/artifacts/`.

There are 58 streaming tests and 497 group tests.
The benchmark compares Python loops with NumPy and checks buffer sizes 64, 256 and 512.
All buffer sizes are scored on chunks 9-32. Results are in `benchmark/stream_results.json`.

## Details

- `fit` trains now. `partial_fit` adds a chunk and trains when enough rows have arrived.
- The tree checks Gini or entropy scores. Equal scores use the first feature and threshold.
- Missing values follow the branch that gives the better split. Missing labels are rejected.
- Each tree uses the same saved preprocessing settings for training and prediction.
- Metrics support `update`, `result` and `reset`. `rolling_result` uses the last N scored rows.
- Statistics use `axis=0` for each feature and `axis=None` for the whole array.
- Mean and variance use all rows. `window_size` limits rows stored for quantiles and histograms.
- Exact quantiles and AUC need extra stored data. `retain=False` keeps only moments and `track_auc=False` turns off AUC.

The original group package is available under `numcompute/`.
