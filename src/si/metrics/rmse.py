from __future__ import annotations
from typing import Sequence
import numpy as np


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """
    Calcula o RMSE entre y_true e y_pred.

    Args:
        y_true: vetor de valores verdadeiros (1D).
        y_pred: vetor de valores preditos (1D).

    Returns:
        float: valor do RMSE.

    Raises:
        ValueError: se shapes incompatíveis.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("rmse: y_true e y_pred devem ter o mesmo shape.")

    mse = np.mean((y_true_arr - y_pred_arr) ** 2)
    return float(np.sqrt(mse))
