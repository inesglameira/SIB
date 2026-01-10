from __future__ import annotations
from typing import List
import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


class StackingClassifier(Model):
    """
    Implementação do StackingClassifier.

    O StackingClassifier combina vários modelos base (nível 0),
    usando as suas previsões para treinar um modelo final (nível 1).

    Parâmetros
    ----------
    models : list[Model]
        Lista de modelos base.
    final_model : Model
        Modelo final (meta-modelo), treinado sobre as previsões
        dos modelos base.
    """

    def __init__(self, models: List[Model], final_model: Model, **kwargs):
        super().__init__(**kwargs)
        self.models = models
        self.final_model = final_model

    # ------------------------------------------------------------------
    # FIT
    # ------------------------------------------------------------------
    def _fit(self, dataset: Dataset) -> "StackingClassifier":
        """
        Treina o StackingClassifier.

        Passos:
        1. Treinar todos os modelos base com o dataset original
        2. Obter as previsões de cada modelo base
        3. Construir um novo dataset com essas previsões
        4. Treinar o modelo final com esse novo dataset
        """
        if dataset.y is None:
            raise ValueError("_fit: dataset deve conter y.")

        # 1) Treinar modelos base
        for model in self.models:
            model.fit(dataset)

        # 2) Obter previsões dos modelos base
        base_predictions = []
        for model in self.models:
            preds = model.predict(dataset)
            base_predictions.append(preds)

        # shape: (n_samples, n_models)
        X_meta = np.column_stack(base_predictions)

        # 3) Criar dataset para o modelo final
        meta_dataset = Dataset(
            X=X_meta,
            y=dataset.y,
            features=[f"model_{i}" for i in range(len(self.models))],
            label=dataset.label
        )

        # 4) Treinar modelo final
        self.final_model.fit(meta_dataset)

        return self

    # ------------------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------------------
    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Produz previsões usando o Stacking.

        Passos:
        1. Obter previsões dos modelos base
        2. Usar essas previsões como input do modelo final
        """
        base_predictions = []
        for model in self.models:
            preds = model.predict(dataset)
            base_predictions.append(preds)

        X_meta = np.column_stack(base_predictions)

        meta_dataset = Dataset(X=X_meta, y=None)

        return self.final_model.predict(meta_dataset)

    # ------------------------------------------------------------------
    # SCORE
    # ------------------------------------------------------------------
    def _score(self, dataset: Dataset) -> float:
        """
        Calcula a accuracy do StackingClassifier.
        """
        if dataset.y is None:
            raise ValueError("_score: dataset deve conter y.")

        preds = self._predict(dataset)
        return accuracy(dataset.y, preds)
