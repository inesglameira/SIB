import numpy as np
from si.feature_selection.select_percentile import SelectPercentile
from si.data.dataset import Dataset

def test_select_percentile_number_of_features():
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)

    ds = Dataset(X, y)

    selector = SelectPercentile(percentile=50)
    selector.fit(ds)
    new_ds = selector.transform(ds)

    # 50% de 10 features = 5
    assert new_ds.X.shape[1] == 5

def test_select_percentile_100_percent():
    X = np.random.rand(20, 6)
    y = np.random.randint(0, 3, 20)

    ds = Dataset(X, y)

    selector = SelectPercentile(percentile=100)
    new_ds = selector.fit_transform(ds)

    assert new_ds.X.shape[1] == X.shape[1]

def test_select_percentile_small_percent():
    X = np.random.rand(50, 10)
    y = np.random.randint(0, 2, 50)

    ds = Dataset(X, y)

    selector = SelectPercentile(percentile=20)
    new_ds = selector.fit_transform(ds)

    # 20% de 10 = 2
    assert new_ds.X.shape[1] == 2

def test_select_percentile_preserves_labels():
    X = np.random.rand(30, 8)
    y = np.random.randint(0, 2, 30)

    ds = Dataset(X, y)

    selector = SelectPercentile(percentile=50)
    new_ds = selector.fit_transform(ds)

    assert np.array_equal(new_ds.y, y)
