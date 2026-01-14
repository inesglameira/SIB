import numpy as np
from si.data.dataset import Dataset
from si.models.ridge_regression import RidgeRegression

def test_ridge_fit_runs():
    X = np.random.rand(20, 3)
    y = np.random.rand(20)

    ds = Dataset(X, y)

    model = RidgeRegression(l2_penalty=1.0)
    model.fit(ds)

    assert model.theta is not None
    assert model.theta_zero is not None

def test_ridge_theta_shape():
    X = np.random.rand(30, 4)
    y = np.random.rand(30)

    ds = Dataset(X, y)

    model = RidgeRegression(l2_penalty=0.5)
    model.fit(ds)

    assert model.theta.shape == (X.shape[1],)

def test_ridge_predict_shape():
    X = np.random.rand(25, 2)
    y = np.random.rand(25)

    ds = Dataset(X, y)

    model = RidgeRegression(l2_penalty=1.0)
    model.fit(ds)

    y_pred = model.predict(ds)

    assert y_pred.shape == y.shape

def test_ridge_score_scalar():
    X = np.random.rand(40, 3)
    y = np.random.rand(40)

    ds = Dataset(X, y)

    model = RidgeRegression(l2_penalty=1.0)
    model.fit(ds)

    score = model.score(ds)

    assert isinstance(score, float)

def test_ridge_with_scaling():
    X = np.random.rand(50, 3) * 100
    y = np.random.rand(50)

    ds = Dataset(X, y)

    model = RidgeRegression(l2_penalty=1.0, scale=True)
    model.fit(ds)

    assert model.mean is not None
    assert model.std is not None


