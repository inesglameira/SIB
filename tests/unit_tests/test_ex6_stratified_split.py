import numpy as np
from si.io.csv_file import read_csv
from si.data.dataset import Dataset
from si.model_selection.split import stratified_train_test_split


def test_stratified_split_basic():
    X = np.random.rand(150, 4)
    y = np.array([0]*50 + [1]*50 + [2]*50)

    ds = Dataset(X, y)

    train_ds, test_ds = stratified_train_test_split(ds, test_size=0.2)

    assert train_ds.X.shape[0] > 0
    assert test_ds.X.shape[0] > 0
    assert train_ds.X.shape[0] + test_ds.X.shape[0] == X.shape[0]

def test_stratified_split_preserves_class_distribution():
    X = np.random.rand(150, 2)
    y = np.array([0]*50 + [1]*50 + [2]*50)

    ds = Dataset(X, y)

    train_ds, test_ds = stratified_train_test_split(ds, test_size=0.2)

    # contagens por classe
    _, train_counts = np.unique(train_ds.y, return_counts=True)
    _, test_counts = np.unique(test_ds.y, return_counts=True)

    # treino deve ter ~80% de cada classe
    assert np.all(train_counts == np.array([40, 40, 40]))
    # teste deve ter ~20% de cada classe
    assert np.all(test_counts == np.array([10, 10, 10]))

def test_stratified_split_iris_dataset():
    dataset = read_csv(
        "datasets/iris/iris.csv",
        sep=",",
        label=True
    )

    train_ds, test_ds = stratified_train_test_split(dataset, test_size=0.2)

    # verificar tamanhos
    assert train_ds.X.shape[0] > test_ds.X.shape[0]

    # verificar classes preservadas
    train_classes = set(train_ds.y)
    test_classes = set(test_ds.y)

    assert train_classes == test_classes

