from __future__ import annotations
from typing import Optional, Sequence
import numpy as np

from si.data.dataset import Dataset
from si.base.transformer import Transformer


class PCA(Transformer):
    """
    PCA por decomposição da matriz de covariância.

    Parâmetros
    ----------
    n_components : int
        Número de componentes principais a calcular (k). Deve satisfazer 1 <= k <= n_features.

    Atributos estimados (após fit)
    ------------------------------
    mean_ : np.ndarray, shape (n_features,)
        Média das features (usada para centrar os dados).
    components_ : np.ndarray, shape (n_components, n_features)
        Vetores próprios (componentes principais). Cada linha é um componente.
    explained_variance_ : np.ndarray, shape (n_components,)
        Variância explicada por cada componente (fração do total de variância).
    eigenvalues_ : np.ndarray, shape (n_features,)
        Autovalores completos (ordenados desc).
    """

    def __init__(self, n_components: int) -> None:
        super().__init__()
        if not isinstance(n_components, int) or n_components <= 0:
            raise ValueError("n_components deve ser um inteiro positivo.")
        self.n_components = n_components

        # atributos estimados (iniciais)
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.eigenvalues_: Optional[np.ndarray] = None

    def _fit(self, dataset: Dataset) -> "PCA":
        """
        Estima os autovalores e autovetores da matriz de covariância dos dados.

        Passos:
        1. Inferir a média por coluna e centrar os dados (X - mean).
        2. Calcular a matriz de covariância (np.cov com rowvar=False).
        3. Fazer eigen-decomposition: np.linalg.eig.
        4. Ordenar autovalores/auto-vetores por autovalor descendente.
        5. Guardar os n_components primeiros vetores próprios como components_.
        6. Calcular explained_variance_ como autovalor / soma(autovalores).

        Retorna:
            self
        """
        # garantir numpy array
        X = np.asarray(dataset.X, dtype=float)
        if X.ndim != 2:
            raise ValueError("_fit: X deve ser uma matriz 2D (n_samples, n_features).")

        n_samples, n_features = X.shape
        if self.n_components > n_features:
            raise ValueError("n_components não pode ser maior que o número de features.")

        # 1) centrar
        mean = np.mean(X, axis=0)
        X_centered = X - mean

        # 2) covariância (por colunas)
        cov = np.cov(X_centered, rowvar=False, bias=False)  # shape (n_features, n_features)

        # 3) eigen-decomposition
        eigenvals, eigenvecs = np.linalg.eig(cov)  # eigenvecs columns correspond to eigenvals

        # 4) ordenar por eigenval descendente
        order = np.argsort(-eigenvals)
        eigenvals_sorted = eigenvals[order].astype(float)
        eigenvecs_sorted = eigenvecs[:, order]  # columns rearranged

        # 5) selecionar os n_components primeiros
        selected_vecs = eigenvecs_sorted[:, : self.n_components]  # shape (n_features, n_components)
        # transpor para ter components_ como (n_components, n_features)
        components = selected_vecs.T

        # 6) explained variance (fração)
        total_var = eigenvals_sorted.sum()
        if total_var == 0:
            explained = np.zeros(self.n_components, dtype=float)
        else:
            explained = eigenvals_sorted[: self.n_components] / total_var

        # guardar atributos
        self.mean_ = mean
        self.components_ = components
        self.explained_variance_ = explained
        self.eigenvalues_ = eigenvals_sorted

        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Transforma o dataset projetando os dados centrados sobre as componentes principais.

        Passos:
        1. Centrar X usando self.mean_ calculada no _fit.
        2. Calcular X_reduced = X_centered.dot(components_.T)  (resulta em (n_samples, n_components))
        3. Retornar um novo Dataset com X reduzido, mantendo y e definindo features como ['PC1', ...].

        Retorna:
            Dataset (com X reduzido)
        """
        if self.components_ is None or self.mean_ is None:
            raise ValueError("PCA: deve chamar fit() antes de transform().")

        X = np.asarray(dataset.X, dtype=float)
        if X.ndim != 2:
            raise ValueError("_transform: X deve ser 2D (n_samples, n_features).")

        # verificar compatibilidade de dimensão
        n_features = X.shape[1]
        if self.components_.shape[1] != n_features:
            raise ValueError(
                f"PCA: número de features do dataset ({n_features}) incompatível com components_ ({self.components_.shape[1]})."
            )

        # 1) centrar com a média armazenada
        X_centered = X - self.mean_

        # 2) projetar: components_ shape (n_components, n_features)
        # X_centered (n_samples, n_features) dot components_.T (n_features, n_components)
        X_reduced = np.dot(X_centered, self.components_.T)

        # 3) construir nomes das novas features
        features_new = [f"PC{i+1}" for i in range(self.components_.shape[0])]

        return Dataset(X=X_reduced, y=dataset.y, features=features_new, label=dataset.label)

    def fit(self, dataset: Dataset) -> "PCA":
        """Wrapper para _fit."""
        return self._fit(dataset)

    def transform(self, dataset: Dataset) -> Dataset:
        """Wrapper para _transform."""
        return self._transform(dataset)
