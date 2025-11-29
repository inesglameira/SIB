# tests/unit_tests/test_select_percentile.py
import numpy as np
from si.feature_selection.select_percentile import SelectPercentile
from si.data.dataset import Dataset

def make_classification_dataset():
    """
    Pequeno dataset sintético com 4 features e 6 amostras, duas classes.
    As features 2 e 3 (índices 2 e 3) serão as mais discriminantes.
    """
    X = np.array([
        [1.0, 10.0, 100.0, 0.1],
        [2.0, 11.0,  98.0, 0.2],
        [1.5, 10.5,102.0, 0.15],
        [5.0, 5.0,  50.0, 0.5],
        [6.0, 6.0,  48.0, 0.6],
        [5.5, 5.4,  52.0, 0.55],
    ], dtype=float)
    y = np.array([0,0,0,1,1,1])
    return Dataset(X=X, y=y, features=["f1","f2","f3","f4"], label="lab")

def test_select_percentile_top50():
    ds = make_classification_dataset()
    sp = SelectPercentile(percentile=50.0)  # 50% de 4 features => 2 features
    sp.fit(ds)
    ds_new = sp.transform(ds)
    assert ds_new.X.shape[1] == 2
    assert ds_new.X.shape[0] == ds.X.shape[0]

def test_select_percentile_ties():
    # caso com empates no limiar
    F_vals = np.array([1.2, 3.4, 2.1, 5.6, 4.3, 5.6, 7.8, 6.5, 5.6, 3.2])
    n = F_vals.shape[0]
    # construir dataset fictício com estes F via monkeypatch da função de scoring:
    X = np.random.rand(10, n)
    y = np.array([0]*5 + [1]*5)
    ds = Dataset(X=X, y=y, features=[f"f{i}" for i in range(n)], label="lab")

    # criar um score_func que devolve F_vals e p-values dummy
    def fake_score(dataset):
        return F_vals, np.ones_like(F_vals)

    sp = SelectPercentile(score_func=fake_score, percentile=40.0)  # 40% de 10 => 4 features
    sp.fit(ds)
    ds_new = sp.transform(ds)
    assert ds_new.X.shape[1] == 4

