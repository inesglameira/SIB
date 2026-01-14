import numpy as np
from si.data.dataset import Dataset
from si.decomposition.pca import PCA

def test_pca_reduces_dimensions():
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)

    ds = Dataset(X, y)

    pca = PCA(n_components=2)
    pca.fit(ds)
    transformed = pca.transform(ds)

    assert transformed.X.shape == (100, 2)

def test_pca_components_shape():
    X = np.random.rand(50, 4)
    ds = Dataset(X)

    pca = PCA(n_components=3)
    pca.fit(ds)

    assert pca.components_.shape == (3, 4)

def test_pca_mean_shape():
    X = np.random.rand(30, 6)
    ds = Dataset(X)

    pca = PCA(n_components=2)
    pca.fit(ds)

    assert pca.mean_.shape == (6,)
