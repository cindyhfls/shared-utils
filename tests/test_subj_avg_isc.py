"""Regression tests for the correlation-valued per-subject ISC contract."""
import unittest
import numpy as np
from isc_utils import get_subj_avg_isc


class SubjectAverageISCTest(unittest.TestCase):
    def test_constant_correlations_return_r_not_z(self):
        for r in (0.8, -0.4, 0.0):
            v = np.full((4, 6), r)  # four subjects, six condensed pairs
            np.testing.assert_allclose(get_subj_avg_isc(v), np.full(4, r), atol=1e-15)

    def test_fisher_mean_across_searchlights_and_other_subjects(self):
        v = np.array([[0.1, 0.7, -0.3], [0.5, 0.9, 0.2]])
        # Condensed pairs for three subjects: (0,1), (0,2), (1,2).
        z01, z02, z12 = np.arctanh(v).mean(axis=0)
        expected = np.tanh([(z01 + z02) / 2, (z01 + z12) / 2, (z02 + z12) / 2])
        actual = get_subj_avg_isc(v)
        np.testing.assert_allclose(actual, expected, atol=1e-15)
        self.assertEqual(actual.shape, (3,))

    def test_missing_values_still_propagate_by_default(self):
        v = np.array([[np.nan, 0.4, 0.6], [0.2, 0.3, 0.5]])
        actual = get_subj_avg_isc(v)
        self.assertTrue(np.isnan(actual[:2]).all())
        self.assertTrue(np.isfinite(actual[2]))

    def test_input_is_not_mutated(self):
        v = np.array([[0.2, -0.1, 0.8], [0.3, 0.5, -0.2]])
        before = v.copy()
        get_subj_avg_isc(v)
        np.testing.assert_array_equal(v, before)


if __name__ == '__main__':
    unittest.main()
