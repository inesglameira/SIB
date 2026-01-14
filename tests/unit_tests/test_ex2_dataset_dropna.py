import numpy as np
from si.data.dataset import Dataset

def test_dropna_removes_nan_rows():
    X = np.array([
        [1.0, 2.0],
        [np.nan, 3.0],
        [4.0, 5.0]
    ])
    y = np.array([0, 1, 0])

    ds = Dataset(X, y)
    ds.dropna()

    assert ds.X.shape[0] == 2
    assert ds.y.shape[0] == 2

def test_dropna_no_nans_after():
    X = np.array([
        [1.0, np.nan],
        [2.0, 3.0],
        [4.0, 5.0]
    ])
    y = np.array([1, 0, 1])

    ds = Dataset(X, y)
    ds.dropna()

    assert not np.isnan(ds.X).any()

def test_dropna_multiple_nans():
    X = np.array([
        [np.nan, np.nan],
        [1.0, 2.0],
        [3.0, 4.0]
    ])
    y = np.array([0, 1, 1])

    ds = Dataset(X, y)
    ds.dropna()

    assert ds.X.shape == (2, 2)
    assert ds.y.tolist() == [1, 1]

def test_dropna_no_nan_keeps_data():
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])
    y = np.array([0, 1])

    ds = Dataset(X.copy(), y.copy())
    ds.dropna()

    assert np.array_equal(ds.X, X)
    assert np.array_equal(ds.y, y)

def test_dropna_returns_self():
    X = np.array([[1.0], [np.nan]])
    y = np.array([0, 1])

    ds = Dataset(X, y)
    result = ds.dropna()

    assert result is ds

