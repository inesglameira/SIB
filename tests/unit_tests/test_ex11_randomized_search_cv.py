import numpy as np

from si.io.csv_file import read_csv
from si.model_selection.randomized_search import randomized_search_cv
from si.metrics.accuracy import accuracy
from si.models.logistic_regression import LogisticRegression


def test_randomized_search_cv_logistic_regression():
    """
    Exercise 11 test:
    - Load breast-bin.csv dataset
    - Create a LogisticRegression model
    - Perform randomized search with CV
    - Validate best score and hyperparameters
    """

    # 1) Load dataset
    dataset = read_csv(
        "datasets/breast_bin/breast-bin.csv",
        sep=",",
        label=True
    )

    assert dataset.X is not None
    assert dataset.y is not None

    # 2) Create model
    model = LogisticRegression()

    # 3) Hyperparameter distributions
    param_grid = {
        "l2_penalty": np.linspace(1, 10, 10),
        "alpha": np.linspace(0.001, 0.0001, 100),
        "max_iter": np.linspace(1000, 2000, 200, dtype=int),
    }

    # 4) Randomized Search CV
    results = randomized_search_cv(
        model=model,
        dataset=dataset,
        hyperparameter_grid=param_grid,
        scoring=accuracy,
        cv=3,
        n_iter=10,
        random_state=42
    )

    # 5) Assertions on output structure
    assert "best_score" in results
    assert "best_hyperparameters" in results
    assert "cv_results" in results

    # 6) Assertions on score
    best_score = results["best_score"]
    assert isinstance(best_score, float)
    assert 0.0 <= best_score <= 1.0
    assert best_score > 0.7  # sanity check

    # 7) Assertions on hyperparameters
    best_params = results["best_hyperparameters"]
    assert "l2_penalty" in best_params
    assert "alpha" in best_params
    assert "max_iter" in best_params
