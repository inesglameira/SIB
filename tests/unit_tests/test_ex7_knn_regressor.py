import numpy as np
from si.metrics.rmse import rmse
from si.models.knn_regressor import KNNRegressor
from si.data.dataset import Dataset
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split



def test_rmse_zero_error():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])

    error = rmse(y_true, y_pred)

    assert error == 0.0

def test_rmse_positive():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 4.0])

    error = rmse(y_true, y_pred)

    assert error > 0

def test_knn_regressor_fit():
    X = np.random.rand(20, 2)
    y = np.random.rand(20)

    ds = Dataset(X, y)

    model = KNNRegressor(k=3)
    model.fit(ds)

    assert model.dataset is not None

def test_knn_regressor_predict_shape():
    X = np.random.rand(30, 2)
    y = np.random.rand(30)

    ds = Dataset(X, y)

    model = KNNRegressor(k=5)
    model.fit(ds)

    y_pred = model.predict(ds)

    assert y_pred.shape == y.shape

def test_knn_regressor_score():
    X = np.random.rand(40, 3)
    y = np.random.rand(40)

    ds = Dataset(X, y)

    model = KNNRegressor(k=3)
    model.fit(ds)

    score = model.score(ds)

    assert isinstance(score, float)
    assert score >= 0

def test_knn_regressor_cpu_dataset():
    dataset = read_csv(
        "datasets/cpu/cpu.csv",
        sep=",",
        label=True
    )

    train_ds, test_ds = train_test_split(dataset, test_size=0.2)

    model = KNNRegressor(k=5)
    model.fit(train_ds)

    score = model.score(test_ds)

    assert isinstance(score, float)
    assert score >= 0




