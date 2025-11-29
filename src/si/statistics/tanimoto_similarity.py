from __future__ import annotations
from typing import Union, Sequence
import numpy as np


def tanimoto_similarity(x: Union[Sequence[int], np.ndarray],
                        y: Union[Sequence[Sequence[int]], np.ndarray]) -> np.ndarray:
    """
    Calcula a DISTÂNCIA de Tanimoto entre um vector binário `x` (1D)
    e cada sample em `y` (1D ou 2D).

    A distância devolvida é: distance = 1 - similarity, onde
        similarity = intersection / (|x| + |y| - intersection)

    Regras:
    - Aceita valores binários (0/1) ou booleanos. Qualquer valor != 0 é tratado como 1 (True).
    - Se `y` for 1D, é tratado como um único sample e devolve-se um array com um elemento.
    - Se o denominador for 0 (i.e., x e y são ambos vetores nulos) define-se similarity = 1.0,
      logo distance = 0.0 (vectores nulos considerados idênticos).

    Args:
        x: vector 1D (shape (n_features,)) contendo 0/1 ou booleanos.
        y: array 2D (n_samples, n_features) contendo 0/1, booleanos, ou array 1D para um único sample.

    Returns:
        np.ndarray: array 1D com as distâncias de Tanimoto entre x e cada linha de y.
    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    if x_arr.ndim != 1:
        raise ValueError("tanimoto_similarity: 'x' deve ser um vector 1D (shape=(n_features,)).")

    # normalizar para boolean (qualquer valor != 0 é True)
    x_bool = x_arr != 0

    # permitir y 1D -> transformar para (1, n_features)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(1, -1)
    if y_arr.ndim != 2:
        raise ValueError("tanimoto_similarity: 'y' deve ser 1D ou 2D array (n_samples, n_features).")

    if y_arr.shape[1] != x_bool.shape[0]:
        raise ValueError("tanimoto_similarity: 'x' e as linhas de 'y' devem ter o mesmo número de features.")

    y_bool = y_arr != 0

    # Interseção (x & y) por linha
    intersection = np.logical_and(x_bool, y_bool).sum(axis=1).astype(float)

    # soma de elementos (|x| e |y|)
    sum_x = float(x_bool.sum())
    sum_y = y_bool.sum(axis=1).astype(float)

    denom = sum_x + sum_y - intersection

    # calcular similarity com cuidado para denom == 0
    similarity = np.empty_like(denom, dtype=float)
    zero_mask = denom == 0.0
    nonzero_mask = ~zero_mask

    similarity[zero_mask] = 1.0
    if nonzero_mask.any():
        similarity[nonzero_mask] = intersection[nonzero_mask] / denom[nonzero_mask]

    # distância = 1 - similarity
    distance = 1.0 - similarity
    return distance
