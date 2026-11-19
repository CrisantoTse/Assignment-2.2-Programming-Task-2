
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from numcompute_stream.tree import DecisionTreeClassifier


class DecisionTreeTests(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(6, dtype=float).reshape(-1, 1)
        self.y = np.array([0, 0, 0, 1, 1, 1])

    def test_separable_binary_data(self):
        tree = DecisionTreeClassifier(max_depth=1).fit(self.X, self.y)
        assert_array_equal(tree.predict(self.X), self.y)
        assert_allclose(tree.predict_proba([[0], [5]]), [[1, 0], [0, 1]])

    def test_entropy_three_classes(self):
        y = ["red", "red", "blue", "blue", "green", "green"]
        tree = DecisionTreeClassifier(criterion="entropy", max_depth=2).fit(self.X, y)
        assert_array_equal(tree.predict(self.X), y)
        assert_array_equal(tree.classes_, ["blue", "green", "red"])

    def test_depth_zero_and_majority_tie(self):
        tree = DecisionTreeClassifier(max_depth=0).fit(self.X, self.y)
        assert_array_equal(tree.predict([[0], [5]]), [0, 0])
        assert_allclose(tree.predict_proba([[0], [5]]), [[0.5, 0.5]] * 2)

    def test_equal_feature_gain_prefers_first(self):
        X = np.c_[self.X, self.X]
        tree = DecisionTreeClassifier(max_depth=1).fit(X, self.y)
        assert_array_equal(tree.predict([[0, 5], [5, 0]]), [0, 1])

    def test_equal_threshold_gain_prefers_lower_split(self):
        tree = DecisionTreeClassifier(max_depth=1).fit([[0], [1], [2]], [0, 1, 0])
        assert_allclose(tree.predict_proba([[0], [2]]), [[1, 0], [0.5, 0.5]])

    def test_missingness_as_a_predictive_feature(self):
        tree = DecisionTreeClassifier().fit([[1.0], [1.0], [np.nan], [np.nan]], [0, 0, 1, 1])
        assert_array_equal(tree.predict([[1.0], [np.nan]]), [0, 1])

    def test_missing_routes_with_the_more_pure_child(self):
        tree = DecisionTreeClassifier(max_depth=1).fit([[0], [1], [np.nan], [np.nan]], [1, 0, 1, 1])
        assert_array_equal(tree.predict([[0], [1], [np.nan]]), [1, 0, 1])

    def test_all_missing_feature_is_ignored(self):
        tree = DecisionTreeClassifier().fit(np.c_[np.full(6, np.nan), self.X], self.y)
        assert_array_equal(tree.predict(np.c_[np.full(6, np.nan), self.X]), self.y)

    def test_adjacent_float_threshold(self):
        lo = np.nextafter(1.0, 2.0)
        hi = np.nextafter(lo, 2.0)
        data = [[lo], [hi]]
        assert_array_equal(DecisionTreeClassifier().fit(data, [1, 2]).predict(data), [1, 2])

    def test_no_overflow_midpoint(self):
        data = [[-np.finfo(float).max], [np.finfo(float).max]]
        with np.errstate(over="raise", invalid="raise"):
            assert_array_equal(DecisionTreeClassifier().fit(data, [1, 2]).predict(data), [1, 2])

    def test_min_samples_split_can_prevent_a_split(self):
        tree = DecisionTreeClassifier(min_samples_split=7).fit(self.X, self.y)
        assert_array_equal(tree.predict(self.X), np.zeros(6))

    def test_min_samples_leaf_rules_out_singleton_split(self):
        tree = DecisionTreeClassifier(min_samples_leaf=2).fit(self.X, [0, 0, 0, 0, 0, 1])
        assert_array_equal(tree.predict(self.X), np.zeros(6))

    def test_seeded_feature_subsampling_is_reproducible(self):
        X = np.c_[self.X, self.X[::-1], np.ones(6)]
        left = DecisionTreeClassifier(max_features=1, random_state=14).fit(X, self.y)
        right = DecisionTreeClassifier(max_features=1, random_state=14).fit(X, self.y)
        assert_array_equal(left.predict(X), right.predict(X))

    def test_max_features_cannot_exceed_column_count(self):
        with self.assertRaisesRegex(ValueError, "max_features"):
            DecisionTreeClassifier(max_features=2).fit(self.X, self.y)

    def test_unfitted_and_empty_prediction(self):
        tree = DecisionTreeClassifier()
        with self.assertRaisesRegex(RuntimeError, "No fitted tree"):
            tree.predict(self.X)
        tree.fit(self.X, self.y)
        self.assertEqual(tree.predict(np.empty((0, 1))).shape, (0,))

    def test_sizes_over_threshold_use_the_whole_chunk(self):
        tree = DecisionTreeClassifier(buffer_size=4).partial_fit(self.X[:2], self.y[:2])
        self.assertEqual(tree.pending_samples_, 2)
        self.assertFalse(tree.is_fitted)
        tree.partial_fit(self.X[2:], self.y[2:])
        self.assertEqual((tree.last_train_size_, tree.retrain_count_, tree.pending_samples_), (6, 1, 0))
        assert_array_equal(tree.predict(self.X), self.y)

    def test_buffer_is_copied_and_active_model_is_retained(self):
        tree = DecisionTreeClassifier(buffer_size=6).fit(self.X, self.y)
        X, y = self.X[:3].copy(), 1 - self.y[:3]
        tree.partial_fit(X, y)
        X[:] = 100
        y[:] = 0
        assert_array_equal(tree.predict(self.X), self.y)
        tree.partial_fit(self.X[3:], 1 - self.y[3:])
        assert_array_equal(tree.predict(self.X), 1 - self.y)
        self.assertEqual(tree.retrain_count_, 2)

    def test_failed_chunk_keeps_pending_rows(self):
        tree = DecisionTreeClassifier(buffer_size=6).partial_fit(self.X[:3], self.y[:3])
        with self.assertRaisesRegex(ValueError, "features"):
            tree.partial_fit(np.ones((3, 2)), self.y[3:])
        with self.assertRaisesRegex(ValueError, "labels"):
            tree.partial_fit(self.X[3:], [0, 1])
        self.assertEqual(tree.pending_samples_, 3)
        tree.partial_fit(self.X[3:], self.y[3:])
        assert_array_equal(tree.predict(self.X), self.y)

    def test_explicit_fit_discards_pending_and_resets_width(self):
        tree = DecisionTreeClassifier(buffer_size=6).partial_fit(self.X[:3], self.y[:3])
        tree.fit([[0, 1], [1, 2]], ["cold", "warm"])
        self.assertEqual(tree.pending_samples_, 0)
        assert_array_equal(tree.predict([[0, 1], [1, 2]]), ["cold", "warm"])

    def test_each_retrain_replaces_old_classes(self):
        tree = DecisionTreeClassifier(buffer_size=2).partial_fit([[0], [1]], ["a", "b"])
        tree.partial_fit([[0], [1]], ["c", "c"])
        assert_array_equal(tree.classes_, ["c"])
        assert_array_equal(tree.predict([[0], [1]]), ["c", "c"])

    def test_invalid_values_do_not_fit(self):
        tree = DecisionTreeClassifier()
        for X, y in (([[np.inf]], [1]), ([[0]], [np.nan]), ([[0]], [[1]])):
            with self.subTest(X=X, y=y), self.assertRaises(ValueError):
                tree.fit(X, y)
        self.assertFalse(tree.is_fitted)
