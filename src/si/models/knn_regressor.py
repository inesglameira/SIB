from __future__ import annotations
from typing import Optional
import numpy as np

from si.data.dataset import Dataset
from si.metrics.rmse import rmse


class KNNRegressor:
    """
    K-Nearest Neighbors regressor.

    Parâmetros
    ----------
    n_neighbors : int
        Número de vizinhos a usar (k).
    weights : str
        'uniform' (média simples) ou 'distance' (ponderado por 1/d).
    """

    def __init__(self, n_neighbors: int = 3, weights: str = "uniform") -> None:
        if n_neighbors <= 0:
            raise ValueError("n_neighbors deve ser um inteiro positivo.")
        if weights not in ("uniform", "distance"):
            raise ValueError("weights deve ser 'uniform' ou 'distance'.")

        self.n_neighbors = int(n_neighbors)
        self.weights = weights
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

    def fit(self, dataset: Dataset) -> "KNNRegressor":
        """
        Guarda os dados de treino.

        Args:
            dataset: Dataset com X e y (y não pode ser None).

        Returns:
            self
        """
        if dataset.y is None:
            raise ValueError("KNNRegressor.fit requer dataset com y.")

        self.X_train = np.asarray(dataset.X, dtype=float)
        self.y_train = np.asarray(dataset.y, dtype=float)
        return self

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Computa a matriz de distâncias Euclidianas entre X (m x d) e X_train (n x d).
        Retorna uma matriz (m, n).
        """
        if self.X_train is None:
            raise ValueError("Modelo não está treinado. Chame fit() primeiro.")
        # efficient squared distances: (x - y)^2 = x^2 + y^2 - 2xy
        X = np.asarray(X, dtype=float)
        a = np.sum(X ** 2, axis=1)[:, None]  # (m,1)
        b = np.sum(self.X_train ** 2, axis=1)[None, :]  # (1,n)
        ab = X.dot(self.X_train.T)  # (m,n)
        d2 = a + b - 2 * ab
        # numerical floor
        d2 = np.maximum(d2, 0.0)
        return np.sqrt(d2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prediz os valores para X.

        Args:
            X: matriz (m_samples, n_features)

        Returns:
            np.ndarray: vetor predito shape (m_samples,)
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Modelo não está treinado. Chame fit() primeiro.")

        X_arr = np.asarray(X, dtype=float)
        distances = self._compute_distances(X_arr)  # (m, n_train)
        n_train = self.X_train.shape[0]
        k = min(self.n_neighbors, n_train)

        # argsort por distância crescente e pegar primeiros k
        neigh_idx = np.argsort(distances, axis=1)[:, :k]  # (m, k)
        neigh_dist = np.take_along_axis(distances, neigh_idx, axis=1)  # (m, k)
        neigh_y = self.y_train[neigh_idx]  # (m, k)

        if self.weights == "uniform":
            preds = np.mean(neigh_y, axis=1)
        else:
            # distance weighting: w = 1 / (d + eps)  (eps para evitar div por zero)
            eps = 1e-8
            weights = 1.0 / (neigh_dist + eps)
            # normalizar pesos
            weights_sum = np.sum(weights, axis=1, keepdims=True)
            preds = np.sum(weights * neigh_y, axis=1) / weights_sum.flatten()

        return preds

    def score(self, dataset: Dataset) -> float:
        """
        Calcula o RMSE entre os valores verdadeiros e os preditos no dataset.
        """
        if dataset.y is None:
            raise ValueError("KNNRegressor.score requer dataset com y.")
        y_pred = self.predict(dataset.X)
        return rmse(dataset.y, y_pred)