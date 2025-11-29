import unittest

import numpy as np

from si.data.dataset import Dataset


class TestDataset(unittest.TestCase):

    def test_dataset_construction(self):

        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2])

        features = np.array(['a', 'b', 'c'])
        label = 'y'
        dataset = Dataset(X, y, features, label)

        self.assertEqual(2.5, dataset.get_mean()[0])
        self.assertEqual((2, 3), dataset.shape())
        self.assertTrue(dataset.has_label())
        self.assertEqual(1, dataset.get_classes()[0])
        self.assertEqual(2.25, dataset.get_variance()[0])
        self.assertEqual(1, dataset.get_min()[0])
        self.assertEqual(4, dataset.get_max()[0])
        self.assertEqual(2.5, dataset.summary().iloc[0, 0])

    def test_dataset_from_random(self):
        dataset = Dataset.from_random(10, 5, 3, features=['a', 'b', 'c', 'd', 'e'], label='y')
        self.assertEqual((10, 5), dataset.shape())
        self.assertTrue(dataset.has_label())
    
    # =====================================================
    # Testes Exercício 2 — dropna, fillna, remove_by_index
    # =====================================================

    def test_dropna_basic(self):
        X = np.array([
            [1.0, 2.0],
            [np.nan, 3.0],
            [4.0, np.nan],
            [5.0, 6.0]
        ])
        y = np.array([0, 1, 0, 1])
        ds = Dataset(X, y, features=["f1", "f2"], label="lab")

        ds.dropna()
        # apenas rows 0 e 3 devem ficar
        self.assertEqual(ds.X.shape, (2, 2))
        self.assertEqual(ds.y.shape, (2,))
        self.assertFalse(np.isnan(ds.X).any())


    def test_fillna_numeric(self):
        X = np.array([
            [1.0, 2.0],
            [np.nan, 3.0],
            [4.0, np.nan],
            [5.0, 6.0]
        ])
        y = np.array([0, 1, 0, 1])
        ds = Dataset(X, y, features=["f1","f2"], label="lab")

        ds.fillna(0.0)

        self.assertFalse(np.isnan(ds.X).any())
        self.assertEqual(ds.X[1,0], 0.0)
        self.assertEqual(ds.X[2,1], 0.0)


    def test_fillna_mean_and_median(self):
        X = np.array([
            [1.0, 2.0],
            [np.nan, 3.0],
            [4.0, np.nan],
            [5.0, 6.0]
        ])
        y = np.array([0, 1, 0, 1])
        ds_mean = Dataset(X.copy(), y.copy(), features=["f1","f2"], label="lab")
        ds_med  = Dataset(X.copy(), y.copy(), features=["f1","f2"], label="lab")

        # mean
        ds_mean.fillna("mean")
        expected_mean_col0 = np.nanmean([1.0, 4.0, 5.0])
        self.assertAlmostEqual(ds_mean.X[1,0], expected_mean_col0, places=6)

        # median
        ds_med.fillna("median")
        expected_median_col1 = np.nanmedian([2.0, 3.0, 6.0])
        self.assertAlmostEqual(ds_med.X[2,1], expected_median_col1, places=6)


    def test_remove_by_index(self):
        X = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ])
        y = np.array([0,1,0])

        ds = Dataset(X.copy(), y.copy(), features=["f1","f2"], label="lab")
        ds.remove_by_index(1)
        self.assertEqual(ds.X.shape, (2,2))
        self.assertEqual(ds.y.shape, (2,))
        self.assertTrue((ds.X == np.array([[1.0,2.0],[5.0,6.0]])).all())


    def test_remove_by_index_negative(self):
        X = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ])
        y = np.array([0,1,0])

        ds = Dataset(X.copy(), y.copy(), features=["f1","f2"], label="lab")
        ds.remove_by_index(-1)
        self.assertEqual(ds.X.shape, (2,2))
        self.assertTrue((ds.X == np.array([[1.0,2.0],[3.0,4.0]])).all())


    def test_remove_by_index_out_of_bounds(self):
        X = np.array([[1,2],[3,4]])
        y = np.array([0,1])
        ds = Dataset(X,y,features=["f1","f2"],label="lab")

        with self.assertRaises(IndexError):
            ds.remove_by_index(5)

