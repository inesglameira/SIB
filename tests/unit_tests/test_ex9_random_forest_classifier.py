import numpy as np

from si.data.dataset import Dataset
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.metrics.accuracy import accuracy
from si.models.random_forest_classifier import RandomForestClassifier


def test_random_forest_classifier_iris():
    """
    Exercise 9 test:
    - Load iris dataset
    - Split into train and test
    - Train RandomForestClassifier
    - Evaluate accuracy on test set
    """

    # 1) Load dataset
    dataset = read_csv(
        "datasets/iris/iris.csv",
        sep=",",
        label=True
    )

    assert dataset.X is not None
    assert dataset.y is not None

    # 2) Train / test split
    train_ds, test_ds = train_test_split(
        dataset,
        test_size=0.2,
        random_state=42
    )

    assert train_ds.X.shape[0] > 0
    assert test_ds.X.shape[0] > 0

    # 3) Create RandomForestClassifier
    rf = RandomForestClassifier(
        n_estimators=10,
        max_depth=5,
        random_state=42
    )

    # 4) Train model
    rf.fit(train_ds)

    # 5) Predict on test set
    y_pred = rf.predict(test_ds)

    # 6) Evaluate accuracy
    acc = accuracy(test_ds.y, y_pred)

    # 7) Assertions
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0

    # Sanity check: model should perform better than random guessing
    assert acc > 0.5
