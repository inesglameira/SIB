from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod


class LossFunction(ABC):
    """
    Classe abstrata para funções de perda.
    """

    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass


class CategoricalCrossEntropy(LossFunction):
    """
    Função de perda Categorical Cross-Entropy.

    Usada em problemas de classificação multi-classe com:
      - y_true em one-hot encoding
      - y_pred como probabilidades (ex.: softmax)

    Inclui proteção contra log(0) através de clipping.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcula o valor da perda (loss).

        Args:
            y_true (np.ndarray): labels verdadeiros em one-hot (shape: n_samples, n_classes)
            y_pred (np.ndarray): probabilidades previstas (shape: n_samples, n_classes)

        Returns:
            float: valor médio da Categorical Cross-Entropy
        """
        # evitar log(0)
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1.0 - eps)

        loss = -np.sum(y_true * np.log(y_pred), axis=1)
        return float(np.mean(loss))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calcula a derivada da loss em relação às previsões.

        Args:
            y_true (np.ndarray): labels verdadeiros (one-hot)
            y_pred (np.ndarray): probabilidades previstas

        Returns:
            np.ndarray: gradiente da loss relativamente a y_pred
        """
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1.0 - eps)

        return -y_true / y_pred
