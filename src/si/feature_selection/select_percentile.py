"""
SelectPercentile transformer: seleciona as features com base num percentil dos scores F.

Implementação compatível com a arquitetura Transformer (possui _fit e _transform e wrappers fit/transform).
"""

from __future__ import annotations
from typing import Callable, Optional, Tuple, List
import numpy as np

from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification
from si.base.transformer import Transformer  

ScoreFuncType = Callable[[Dataset], Tuple[np.ndarray, np.ndarray]]


class SelectPercentile(Transformer):
    """
    Seleciona um percentil das melhores features com base nos F-values.

    Parâmetros
    ----------
    score_func : callable, opcional
        Função que, dado um Dataset, devolve (F_values, p_values) onde cada um é um array com
        um valor por feature. Por defeito usa-se f_classification.
    percentile : float
        Percentil (0..100) de features a seleccionar (ex.: 40.0 significa seleccionar as melhores
        40% das features).

    Atributos estimados
    -------------------
    F_ : np.ndarray
        Valores F estimados por feature.
    p_ : np.ndarray
        Valores p estimados por feature.
    mask_ : np.ndarray
        Máscara booleana das features seleccionadas (True = seleccionada).
    """

    def __init__(self, score_func: ScoreFuncType = f_classification, percentile: float = 50.0) -> None:
        super().__init__()
        self.score_func = score_func
        self.percentile = float(percentile)
        self.F_: Optional[np.ndarray] = None
        self.p_: Optional[np.ndarray] = None
        self.mask_: Optional[np.ndarray] = None

    def _fit(self, dataset: Dataset) -> "SelectPercentile":
        """
        Estima os valores F e p para cada feature usando a score_func.

        Returns:
            self
        """
        F, p = self.score_func(dataset)
        self.F_ = np.asarray(F)
        self.p_ = np.asarray(p)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Seleciona as features correspondentes ao percentil definido de acordo com os F-values.
        Garante que o número de features seleccionadas corresponde ao percentil (trata empates na
        fronteira selecionando as primeiras ocorrências dos empates conforme necessário).
        """
        if self.F_ is None:
            raise ValueError("SelectPercentile: deve chamar fit() antes de transform().")

        n_features = int(self.F_.shape[0])
        if n_features == 0:
            X_new = np.empty((dataset.X.shape[0], 0))
            return Dataset(X=X_new, y=dataset.y, features=[], label=dataset.label)

        k = int(np.ceil((self.percentile / 100.0) * n_features))
        if k <= 0:
            X_new = np.empty((dataset.X.shape[0], 0))
            return Dataset(X=X_new, y=dataset.y, features=[], label=dataset.label)

        order_desc = np.argsort(-self.F_)
        kth_value = self.F_[order_desc[k-1]]

        mask = self.F_ > kth_value
        selected_so_far = int(mask.sum())
        remaining = k - selected_so_far
        if remaining > 0:
            tied_indices = np.where(self.F_ == kth_value)[0]
            tied_in_order = [idx for idx in order_desc if idx in tied_indices]
            for idx in tied_in_order[:remaining]:
                mask[idx] = True

        self.mask_ = mask.astype(bool)

        selected_cols = np.where(self.mask_)[0]
        X_new = dataset.X[:, selected_cols]

        features_new: Optional[List[str]] = None
        if dataset.features is not None:
            features_new = [dataset.features[i] for i in selected_cols]

        return Dataset(X=X_new, y=dataset.y, features=features_new, label=dataset.label)

    def fit(self, dataset: Dataset) -> "SelectPercentile":
        """Wrapper que chama _fit."""
        return self._fit(dataset)

    def transform(self, dataset: Dataset) -> Dataset:
        """Wrapper que chama _transform."""
        return self._transform(dataset)
