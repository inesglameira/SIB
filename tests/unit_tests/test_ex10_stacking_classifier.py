import numpy as np

from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.metrics.accuracy import accuracy

from si.models.knn_classifier import KNNClassifier
from si.models.logistic_regression import LogisticRegression
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.ensemble.stacking_classifier import StackingClassifier


def test_stacking_classifier_breast_bin():
    """
    Exercise 10 test:
    - Load breast-bin.csv dataset
    - Split into train and test sets
    - Create KNN, LogisticRegression and DecisionTree models
    - Create a second KNN as final model
    - Train and evaluate the StackingClassifier
    """

    # 1) Load dataset
    dataset = read_csv(
        "datasets/breast_bin/breast-bin.csv",
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

    # 3) Base models
    knn = KNNClassifier(k=5, distance_function=np.linalg.norm)
    logreg = LogisticRegression()
    dt = DecisionTreeClassifier(max_depth=5)

    base_models = [knn, logreg, dt]

    # 4) Final model (second KNN)
    final_knn = KNNClassifier(k=3, distance_function=np.linalg.norm)

    # 5) StackingClassifier
    stacking = StackingClassifier(
        models=base_models,
        final_model=final_knn
    )

    # 6) Train
    stacking.fit(train_ds)

    # 7) Predict
    y_pred = stacking.predict(test_ds)

    # 8) Evaluate
    acc = accuracy(test_ds.y, y_pred)

    # Assertions
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0
    assert acc > 0.7  # sanity check
