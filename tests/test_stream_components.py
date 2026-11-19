
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from numcompute_stream.metrics import StreamingMetrics, roc_auc
from numcompute_stream.preprocessing import Imputer, MinMaxScaler, OneHotEncoder, StandardScaler
from numcompute_stream.stats import StreamingStats, update_stats


class StatisticsTests(unittest.TestCase):
    def test_uneven_chunks_with_missing_cells_match_batch_moments(self):
        X = np.array([[1, np.nan, 0], [9, 6, 4], [3, 8, np.nan], [5, 1, 7]], dtype=float)
        value = update_stats(X[:1])
        update_stats(X[1:3], value).partial_fit(X[3:])
        assert_array_equal(value.count, [4, 3, 3])
        assert_allclose(value.mean, np.nanmean(X, axis=0))
        assert_allclose(value.variance(ddof=1), np.nanvar(X, axis=0, ddof=1))

    def test_flat_statistics_agree_on_all_elements(self):
        stats = StreamingStats(axis=None).update([[1, 2], [np.nan, 4]]).update([[5, 0]])
        self.assertEqual(stats.count, 5)
        self.assertAlmostEqual(stats.mean, 2.4)
        self.assertAlmostEqual(stats.variance(), np.var([1, 2, 4, 5, 0]))

    def test_moments_with_large_offset(self):
        X = 1e12 + np.arange(9)
        stats = StreamingStats(axis=None).update(X[:2]).update(X[2:7]).update(X[7:])
        self.assertEqual(stats.mean, np.mean(X))
        self.assertAlmostEqual(stats.variance(), np.var(X))

    def test_missing_policy_propagates_per_feature(self):
        stats = StreamingStats(nan_policy="propagate").update([[2, np.nan]]).update([[4, 2]])
        assert_allclose(stats.mean, [3, np.nan], equal_nan=True)
        assert_allclose(stats.variance(), [1, np.nan], equal_nan=True)

    def test_nan_raise_is_atomic(self):
        stats = StreamingStats(nan_policy="raise").update([7])
        with self.assertRaises(ValueError):
            stats.update([8, np.nan])
        assert_array_equal(stats.count, [1])
        assert_array_equal(stats.mean, [7])

    def test_all_nan_column_and_ddof(self):
        stats = StreamingStats().update([[np.nan, 1]])
        assert_array_equal(stats.count, [0, 1])
        assert_allclose(stats.variance(ddof=1), [np.nan, np.nan], equal_nan=True)

    def test_exact_quantile_across_chunks(self):
        stats = StreamingStats().update([[0, 2], [2, 4]]).update([[4, np.nan], [6, 6]])
        assert_allclose(stats.quantile([0.25, 0.5]), [[1.5, 3], [3, 4]])

    def test_rolling_histogram_and_quantile_trim_partial_chunk(self):
        stats = StreamingStats(axis=None, window_size=3).update([0, 1, 2]).update([3, 4])
        counts, edges = stats.histogram(bins=[0, 2, 3, 5])
        assert_array_equal(counts, [0, 1, 2])
        assert_array_equal(edges, [0, 2, 3, 5])
        self.assertEqual(stats.quantile(0.5), 3)
        self.assertEqual(stats.mean, 2)

    def test_no_history_mode_and_reset(self):
        stats = StreamingStats(retain=False).update([1, 2])
        with self.assertRaises(RuntimeError):
            stats.quantile(0.5)
        stats.reset().update([[9, 8]])
        assert_array_equal(stats.mean, [9, 8])


class PreprocessingTests(unittest.TestCase):
    def test_running_scaler_agrees_with_batch(self):
        X = np.array([[1, np.nan, 5], [3, 12, 5], [5, 3, 5], [7, 9, 5]])
        model = StandardScaler().partial_fit(X[:1]).partial_fit(X[1:])
        assert_allclose(model.mean_, np.nanmean(X, axis=0))
        assert_allclose(model.var_, np.nanvar(X, axis=0))
        assert_allclose(model.transform(X)[:, 2], [0, 0, 0, 0])
        self.assertTrue(np.isnan(model.transform(X)[0, 1]))

    def test_scaler_fit_resets_previous_state(self):
        scaler = StandardScaler().partial_fit([[100]]).fit([[30], [40]])
        assert_allclose(scaler.transform([[30], [40]]), [[-1], [1]])

    def test_scaler_rejects_wrong_shape_without_losing_state(self):
        scaler = StandardScaler().partial_fit([[3]])
        with self.assertRaisesRegex(ValueError, "features"):
            scaler.partial_fit([[1, 2]])
        assert_allclose(scaler.mean_, [3])

    def test_imputer_updates_mean_and_all_missing_fallback(self):
        imputer = Imputer(fill_value=-1).partial_fit([[np.nan, 2], [np.nan, np.nan]])
        assert_allclose(imputer.transform([[np.nan, np.nan]]), [[-1, 2]])
        imputer.partial_fit([[5, 4], [9, 6]])
        assert_allclose(imputer.transform([[np.nan, np.nan]]), [[7, 4]])

    def test_imputer_constant_and_independent_output(self):
        X = np.array([[np.nan, 2]])
        result = Imputer(strategy="constant", fill_value=-7).fit_transform(X)
        assert_allclose(result, [[-7, 2]])
        self.assertTrue(np.isnan(X[0, 0]))

    def test_minmax_accumulates_extrema_and_ignores_missing(self):
        scaler = MinMaxScaler(feature_range=(-1, 1)).partial_fit([[2], [np.nan]]).partial_fit([[6], [4]])
        assert_allclose(scaler.transform([[2], [4], [6], [np.nan]]), [[-1], [0], [1], [np.nan]], equal_nan=True)

    def test_minmax_extreme_finite_endpoints(self):
        data = [[-1e308], [0], [1e308]]
        assert_allclose(MinMaxScaler().fit_transform(data), [[0], [0.5], [1]])

    def test_encoder_append_order_and_missing_category(self):
        encoder = OneHotEncoder().partial_fit([[2.0], [np.nan], [1.0]]).partial_fit([[3], [1]])
        assert_array_equal(encoder.categories_[0], [2, None, 1, 3])
        assert_allclose(encoder.transform([[3], [np.nan], [4]]), [[0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 0, 0]])

    def test_encoder_separate_feature_blocks(self):
        X = np.array([[1, "b"], [2, "a"], [None, "a"]], dtype=object)
        assert_array_equal(OneHotEncoder().fit_transform(X), [[1, 0, 0, 1, 0], [0, 1, 0, 0, 1], [0, 0, 1, 0, 1]])

    def test_encoder_rejects_new_family_and_keeps_previous_state(self):
        encoder = OneHotEncoder(handle_unknown="error").fit([[1], [2]])
        with self.assertRaisesRegex(TypeError, "strings or numbers"):
            encoder.partial_fit([["one"]])
        with self.assertRaisesRegex(ValueError, "unknown"):
            encoder.transform([[3]])
        assert_allclose(encoder.transform([[1]]), [[1, 0]])


class MetricsTests(unittest.TestCase):
    def test_new_class_remaps_older_matrix(self):
        metric = StreamingMetrics().update([1, 2, 2], [2, 2, 1]).update([0, 1], [0, 0])
        result = metric.result()
        assert_array_equal(result["labels"], [0, 1, 2])
        assert_array_equal(result["confusion_matrix"], [[1, 0, 0], [1, 0, 1], [0, 1, 1]])
        self.assertEqual(result["accuracy"], 2 / 5)
        self.assertTrue(np.isnan(result["auc"]))

    def test_per_class_precision_recall_and_f1(self):
        r = StreamingMetrics().update([0, 0, 1, 1], [0, 1, 1, 1]).result(average=None)
        assert_allclose(r["precision"], [1, 2 / 3])
        assert_allclose(r["recall"], [0.5, 1])
        assert_allclose(r["f1"], [2 / 3, 0.8])

    def test_rolling_samples_drop_old_rows_and_partial_chunks(self):
        metric = StreamingMetrics(window_size=3).update([0, 0, 1], [0, 1, 0]).update([1, 0], [1, 0])
        self.assertEqual(metric.result()["count"], 5)
        self.assertAlmostEqual(metric.rolling_result()["accuracy"], 2 / 3)
        assert_array_equal(metric.rolling_result()["confusion_matrix"], [[1, 0], [1, 1]])

    def test_rolling_window_smaller_than_a_chunk(self):
        metric = StreamingMetrics(window_size=2).update([0, 0, 1, 1], [1, 1, 1, 1])
        self.assertEqual(metric.rolling_result()["count"], 2)
        self.assertEqual(metric.rolling_result()["accuracy"], 1)

    def test_auc_ties_and_cross_chunk_pairs(self):
        y, p = [0, 1, 0, 1], [0.2, 0.5, 0.5, 0.9]
        metric = StreamingMetrics(window_size=2)
        metric.update(y[:2], [0, 1], p[:2]).update(y[2:], [1, 1], p[2:])
        self.assertAlmostEqual(metric.result()["auc"], 0.875)
        self.assertAlmostEqual(metric.rolling_result()["auc"], 1.0)
        self.assertAlmostEqual(roc_auc(y, p), 0.875)

    def test_auc_undefined_without_positives(self):
        self.assertTrue(np.isnan(StreamingMetrics().update([0, 0], [1, 0]).result()["auc"]))

    def test_binary_average_with_string_positive_label(self):
        metric = StreamingMetrics(positive_label="yes").update(["no", "yes"], ["yes", "yes"])
        self.assertEqual(metric.result(average="binary")["precision"], 0.5)
        self.assertEqual(metric.result()["auc"], 0.5)

    def test_invalid_scores_leave_metric_unchanged(self):
        metric = StreamingMetrics().update([0], [0])
        with self.assertRaises(ValueError):
            metric.update([1], [1], [np.nan])
        self.assertEqual(metric.result()["count"], 1)

    def test_reset_and_zero_division(self):
        metric = StreamingMetrics(window_size=4).update([0, 1], [0, 0])
        self.assertEqual(metric.result(average="binary")["precision"], 0)
        metric.reset()
        self.assertEqual(metric.rolling_result()["count"], 0)
        self.assertTrue(np.isnan(metric.result()["accuracy"]))
