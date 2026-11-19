
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from numcompute_stream import DecisionTreeClassifier, OneHotEncoder, Pipeline, StandardScaler, StreamTrainer
from numcompute_stream.io import iter_supervised_csv
from numcompute_stream.visualise import compare_models, plot_metric_over_time, plot_predictions_vs_ground_truth


class PipelineWorkflowTests(unittest.TestCase):
    def test_live_scaler_changes_do_not_move_active_split(self):
        model = DecisionTreeClassifier(buffer_size=4)
        pipe = Pipeline([("scale", StandardScaler()), ("model", model)])
        X, y = [[0], [1], [2], [3]], [0, 0, 1, 1]
        pipe.partial_fit(X, y)
        expected = pipe.transform([[2]])
        pipe.partial_fit([[99]], [1])
        self.assertGreater(pipe.named_steps["scale"].mean_[0], 10)
        assert_array_equal(pipe.transform([[2]]), expected)
        assert_array_equal(pipe.predict(X), y)

    def test_buffer_is_transformed_together_after_scaler_moves(self):
        model = DecisionTreeClassifier(buffer_size=4)
        pipe = Pipeline([("scale", StandardScaler()), ("model", model)])
        pipe.partial_fit([[0], [1]], [0, 0]).partial_fit([[20], [21]], [1, 1])
        assert_array_equal(pipe.predict([[0], [1], [20], [21]]), [0, 0, 1, 1])

    def test_encoder_adds_categories_without_changing_current_model_shape(self):
        model = DecisionTreeClassifier(buffer_size=2)
        pipe = Pipeline([("encode", OneHotEncoder()), ("model", model)])
        pipe.partial_fit([["a"], ["b"]], [0, 1])
        pipe.partial_fit([["c"]], [1])
        self.assertEqual(pipe.transform([["c"]]).shape, (1, 2))
        pipe.partial_fit([["a"]], [0])
        self.assertEqual(pipe.transform([["c"]]).shape, (1, 3))
        assert_array_equal(pipe.predict([["a"], ["c"]]), [0, 1])

    def test_failed_chunk_does_not_commit_transformer_update(self):
        model = DecisionTreeClassifier(buffer_size=4)
        pipe = Pipeline([("scale", StandardScaler()), ("model", model)])
        pipe.partial_fit([[0], [1]], [0, 1])
        with self.assertRaises(TypeError):
            pipe.partial_fit([[8], [9]], ["x", "y"])
        assert_allclose(pipe.named_steps["scale"].mean_, [0.5])
        self.assertEqual(model.pending_samples_, 2)

    def test_batch_fit_and_transform_only_mode(self):
        pipe = Pipeline([("scale", StandardScaler())])
        assert_allclose(pipe.fit_transform([[0], [2]]), [[-1], [1]])
        with self.assertRaises(RuntimeError):
            pipe.predict([[3]])

    def test_batch_fit_then_stream(self):
        pipe = Pipeline([("scale", StandardScaler()), ("model", DecisionTreeClassifier(buffer_size=2))])
        pipe.fit([[0], [2]], [0, 1])
        assert_array_equal(pipe.predict([[0], [2]]), [0, 1])
        pipe.partial_fit([[0], [2]], [1, 0])
        assert_array_equal(pipe.predict([[0], [2]]), [1, 0])

    def test_score_precedes_training_on_drift_boundary(self):
        trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=4), window_size=4)
        X = [[0], [1], [2], [3]]
        base, shift = [0, 0, 1, 1], [1, 1, 0, 0]
        warm = trainer.fit_chunk(X, base)
        self.assertFalse(warm["scored"])
        self.assertTrue(warm["retrained"])
        pre = trainer.score_chunk(X, shift)
        self.assertEqual(pre["accuracy"], 0)
        self.assertEqual(trainer.metrics.result()["count"], 0)
        first = trainer.fit_chunk(X, shift)
        self.assertEqual(first["accuracy"], 0)
        self.assertGreater(first["memory_bytes"], 0)
        second = trainer.fit_chunk(X, shift)
        self.assertEqual(second["accuracy"], 1)
        self.assertEqual(second["cumulative_accuracy"], 0.5)
        self.assertEqual(second["rolling_accuracy"], 1)

    def test_chunk_csv_keeps_last_short_chunk_and_nan(self):
        with TemporaryDirectory() as temp:
            path = Path(temp, "mini.csv")
            path.write_text("target\ta\tb\n0\t1\t2\n1\t\t3\n1\t8\t9\n", encoding="utf-8")
            chunks = list(iter_supervised_csv(path, 2, target_column=0, delimiter="\t", skip_header=1))
            self.assertEqual([x.shape for x, _ in chunks], [(2, 2), (1, 2)])
            assert_allclose(chunks[0][0], [[1, 2], [np.nan, 3]], equal_nan=True)
            assert_array_equal(chunks[1][1], [1])

    def test_plots_return_axes_and_export_png(self):
        with TemporaryDirectory() as temp:
            path = Path(temp, "plots", "acc.png")
            f1, a1 = plot_metric_over_time([np.nan, 0.3, 0.7], "Accuracy", "Fraction", save_path=path)
            f2, a2 = compare_models([np.nan, 0.8], [0.6, 0.7], ["a", "b"])
            f3, a3 = plot_predictions_vs_ground_truth(["no", "yes"], ["no", "no"])
            try:
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                assert_allclose(a1.lines[0].get_ydata(), [np.nan, 0.3, 0.7], equal_nan=True)
                self.assertEqual(len(a2.lines), 2)
                self.assertEqual([tick.get_text() for tick in a3.get_yticklabels()], ["no", "yes"])
            finally:
                for fig in (f1, f2, f3):
                    plt.close(fig)
