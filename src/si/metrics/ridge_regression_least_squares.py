from __future__ import annotations
from typing import Optional
import numpy as np

from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegressionLeastSquares:
    """
    Regressão Ridge (Least Squares com regularização L2).

    Parâmetros
    ----------
    l2_penalty : float
        Parâmetro de regularização L2 (lambda). Não negativo.
    scale : bool
        Se True, centraliza e escala features (z-score) durante fit/predict.

    Atributos estimados
    -------------------
    theta : np.ndarray
        Coeficientes do modelo para cada feature (shape (n_features,)).
    theta_zero : float
        Intercepto (bias).
    mean_ : np.ndarray
        Média por feature usada no scaling.
    std_ : np.ndarray
        Desvio padrão por feature usado no scaling.
    """

    def __init__(self, l2_penalty: float = 1.0, scale: bool = True) -> None:
        if l2_penalty < 0:
            raise ValueError("l2_penalty deve ser não-negativo.")
        self.l2_penalty = float(l2_penalty)
        self.scale = bool(scale)

        self.theta: Optional[np.ndarray] = None
        self.theta_zero: Optional[float] = None
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, dataset: Dataset) -> "RidgeRegressionLeastSquares":
        """
        Ajusta o modelo aos dados usando a equação normal regularizada.

        Passos:
        1) opcionalmente escala X (guardar mean_ e std_)
        2) construir X_design com coluna de 1s na primeira posição
        3) construir matriz de penalização P = lambda * I (posição [0,0] = 0)
        4) resolver (X.T X + P) theta = X.T y
        5) separar theta_zero e theta
        """
        X = np.asarray(dataset.X, dtype=float)
        if X.ndim != 2:
            raise ValueError("fit: X deve ser matriz 2D (n_samples, n_features)")
        if dataset.y is None:
            raise ValueError("fit: dataset deve conter y")
        y = np.asarray(dataset.y, dtype=float).ravel()

        n_samples, n_features = X.shape

        # 1) escala (se for o caso)
        if self.scale:
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0, ddof=0)
            std_safe = np.where(std == 0.0, 1.0, std)
            Xs = (X - mean) / std_safe
            self.mean_ = mean
            self.std_ = std_safe
        else:
            Xs = X.copy()
            self.mean_ = np.zeros(n_features, dtype=float)
            self.std_ = np.ones(n_features, dtype=float)

        # 2) adicionar intercepto (coluna de 1s)
        X_design = np.c_[np.ones((n_samples, 1)), Xs]  # shape (n_samples, n_features+1)

        # 3) matriz de penalização
        p = X_design.shape[1]
        P = self.l2_penalty * np.eye(p, dtype=float)
        P[0, 0] = 0.0  # não penalizar o intercepto

        # 4) equação normal regularizada: (X^T X + P) theta = X^T y
        XtX = X_design.T.dot(X_design)
        A = XtX + P
        b = X_design.T.dot(y)

        # usar solve (mais estável que inv)
        theta_full = np.linalg.solve(A, b)  # shape (p,)

        # 5) separar intercepto e coeficientes
        self.theta_zero = float(theta_full[0])
        self.theta = theta_full[1:].astype(float)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prediz y para as amostras X.

        Passos:
        1) verificar que o modelo foi ajustado
        2) aplicar scale usando mean_ e std_ se necessário
        3) adicionar coluna de 1s e multiplicar por thetas
        """
        if self.theta is None or self.theta_zero is None:
            raise ValueError("predict: o modelo não foi ajustado. Chame fit() primeiro.")

        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("predict: X deve ser matriz 2D (n_samples, n_features)")
        if X_arr.shape[1] != self.theta.shape[0]:
            raise ValueError("predict: número de features incompatível com o modelo.")

        if self.scale:
            Xs = (X_arr - self.mean_) / self.std_
        else:
            Xs = X_arr

        theta_full = np.r_[self.theta_zero, self.theta]
        X_design = np.c_[np.ones((Xs.shape[0], 1)), Xs]
        y_pred = X_design.dot(theta_full)
        return y_pred

    def score(self, dataset: Dataset) -> float:
        """
        Calcula o MSE entre y real e y predito no dataset.
        """
        if dataset.y is None:
            raise ValueError("score: dataset deve conter y.")
        y_pred = self.predict(dataset.X)
        return mse(dataset.y, y_pred)
