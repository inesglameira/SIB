import numpy as np
from si.neural_networks.losses import CategoricalCrossEntropy

def test_categorical_crossentropy_forward():
    loss = CategoricalCrossEntropy()

    # one-hot labels
    y_true = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ])

    # probabilidades previstas
    y_pred = np.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1]
    ])

    value = loss.forward(y_true, y_pred)

    assert value > 0
    assert isinstance(value, float)

def test_categorical_crossentropy_perfect_prediction():
    loss = CategoricalCrossEntropy()

    y_true = np.array([
        [0, 1, 0],
        [1, 0, 0]
    ])

    y_pred = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0]
    ])

    value = loss.forward(y_true, y_pred)

    # Muito próxima de zero
    assert value < 1e-6

def test_categorical_crossentropy_numerical_stability():
    loss = CategoricalCrossEntropy()

    y_true = np.array([[1, 0, 0]])
    y_pred = np.array([[1.0, 0.0, 0.0]])  # contém zeros

    value = loss.forward(y_true, y_pred)

    assert not np.isnan(value)
    assert not np.isinf(value)

def test_categorical_crossentropy_backward_shape():
    loss = CategoricalCrossEntropy()

    y_true = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ])

    y_pred = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1]
    ])

    grad = loss.backward(y_true, y_pred)

    assert grad.shape == y_pred.shape

def test_categorical_crossentropy_backward_values():
    loss = CategoricalCrossEntropy()

    y_true = np.array([[0, 1, 0]])
    y_pred = np.array([[0.2, 0.7, 0.1]])

    grad = loss.backward(y_true, y_pred)

    assert not np.any(np.isnan(grad))
    assert not np.any(np.isinf(grad))
