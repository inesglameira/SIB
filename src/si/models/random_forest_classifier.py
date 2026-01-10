from __future__ import annotations
from typing import List, Tuple, Optional, Union
import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.metrics.accuracy import accuracy


class RandomForestClassifier(Model):
    """
    Implementação de um classificador Random Forest.

    O Random Forest é um modelo de ensemble baseado em múltiplas Decision Trees,
    onde cada árvore é treinada sobre:
      • um *bootstrap sample* (amostragem aleatória com reposição)
      • um subconjunto aleatório de features

    A predição final é feita por *voto maioritário* entre as árvores.

    Parâmetros
    ----------
    n_estimators : int, default=10
        Número de árvores na floresta.
    max_features : int, float, str ou None, default=None
        Número de features a selecionar para cada árvore.
        • None ou "sqrt": usa int(sqrt(n_features))
        • "log2": usa int(log2(n_features))
        • float em (0,1]: representa fração das features
        • int: número absoluto de features
    min_sample_split : int, default=2
        Número mínimo de amostras necessárias para dividir um nó.
    max_depth : int ou None, default=10
        Profundidade máxima das árvores.
        Se None → profundidade ilimitada (internamente convertido em infinito).
    mode : {'gini', 'entropy'}, default='gini'
        Critério de impureza para a DecisionTree.
    seed : int ou None, default=None
        Seed para controlo da aleatoriedade.
    """

    def __init__(
        self,
        n_estimators: int = 10,
        max_features: Optional[Union[int, float, str]] = None,
        min_sample_split: int = 2,
        max_depth: Optional[int] = 10,
        mode: str = 'gini',
        seed: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_estimators = int(n_estimators)
        self.max_features = max_features
        self.min_sample_split = int(min_sample_split)
        self.max_depth = None if max_depth is None else int(max_depth)
        self.mode = mode
        self.seed = seed

        # Lista de tuplos (feature_indices, árvore)
        self.trees: List[Tuple[np.ndarray, DecisionTreeClassifier]] = []
        self.classes_: Optional[np.ndarray] = None

    # -------------------------------------------------------------------------
    #                  FUNÇÕES AUXILIARES PARA BOOTSTRAP E FEATURES
    # -------------------------------------------------------------------------

    def _determine_m(self, n_features: int) -> int:
        """
        Determina o número de features a selecionar (m) com base no valor de
        self.max_features. Aceita:
            - None ou "sqrt" → sqrt(n_features)
            - "log2"        → log2(n_features)
            - float (0,1]   → fração das features
            - int           → número fixo de features
        """
        mf = self.max_features

        if mf is None:
            m = int(np.sqrt(n_features))
        elif isinstance(mf, str):
            s = mf.lower()
            if s == "sqrt" or s == "auto":
                m = max(1, int(np.sqrt(n_features)))
            elif s == "log2":
                m = max(1, int(np.log2(n_features)))
            else:
                raise ValueError(f"max_features string inválida: {mf}")
        elif isinstance(mf, float):
            if not (0 < mf <= 1):
                raise ValueError("max_features float deve estar em (0,1].")
            m = max(1, int(np.ceil(mf * n_features)))
        else:
            m = int(mf)

        return max(1, min(m, n_features))

    def _bootstrap_indices(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """
        Devolve índices para bootstrap (amostragem com reposição).
        """
        return rng.choice(n_samples, size=n_samples, replace=True)

    def _sample_features(self, n_features: int, rng: np.random.Generator) -> np.ndarray:
        """
        Seleciona m features de forma aleatória e sem reposição,
        onde m é determinado por _determine_m().
        """
        m = self._determine_m(n_features)
        return rng.choice(n_features, size=m, replace=False)

    # -------------------------------------------------------------------------
    #                                TREINO
    # -------------------------------------------------------------------------

    def _fit(self, dataset: Dataset) -> "RandomForestClassifier":
        """
        Treina o Random Forest criando n árvores:
          1) Geração de bootstrap samples (linhas)
          2) Sub-amostragem de features (colunas)
          3) Treino de DecisionTreeClassifier
          4) Armazenamento da árvore e respetivas features
        """
        if dataset.y is None:
            raise ValueError("_fit: dataset deve conter y.")

        X = np.asarray(dataset.X, dtype=float)
        y = np.asarray(dataset.y)
        n_samples, n_features = X.shape

        rng = np.random.default_rng(self.seed)
        self.trees = []
        self.classes_ = np.unique(y)

        for _ in range(self.n_estimators):

            # 1) bootstrap das amostras
            sample_idx = self._bootstrap_indices(n_samples, rng)
            X_sample = X[sample_idx]
            y_sample = y[sample_idx]

            # 2) amostragem aleatória de features
            feat_idx = self._sample_features(n_features, rng)
            X_boot = X_sample[:, feat_idx]

            # manter nomes das features se existirem
            if dataset.features is not None:
                boot_features = [dataset.features[i] for i in feat_idx]
            else:
                boot_features = None

            ds_boot = Dataset(X=X_boot, y=y_sample,
                              features=boot_features, label=dataset.label)

            # 3) criação da árvore
            # Se max_depth é None → profundidade ilimitada (float('inf'))
            tree_max_depth = float("inf") if self.max_depth is None else self.max_depth

            tree = DecisionTreeClassifier(
                min_sample_split=self.min_sample_split,
                max_depth=tree_max_depth,
                mode=self.mode
            )

            tree.fit(ds_boot)

            # 4) guardar (features usadas, árvore treinada)
            self.trees.append((feat_idx.copy(), tree))

        return self

    # -------------------------------------------------------------------------
    #                                PREDIÇÃO
    # -------------------------------------------------------------------------

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Produz previsões através de voto maioritário entre todas as árvores.

        Para cada árvore:
          • selecionam-se apenas as features usadas no treino,
          • obtém-se a previsão individual,
          • realiza-se voto maioritário por amostra.
        """
        if not self.trees:
            raise ValueError("_predict: nenhuma árvore treinada.")

        X = np.asarray(dataset.X, dtype=float)
        n_samples = X.shape[0]
        n_trees = len(self.trees)

        all_preds = np.empty((n_trees, n_samples), dtype=object)

        for i, (feat_idx, tree) in enumerate(self.trees):
            X_sub = X[:, feat_idx]

            try:
                preds = tree.predict(X_sub)
            except Exception:
                # fallback se a árvore exigir Dataset; retorna um numpy array mas o input é sempre um dataset 
                tmp_ds = Dataset(X_sub, y=None, features=None, label=None)
                preds = tree.predict(tmp_ds)

            preds = np.asarray(preds, dtype=object)

            if preds.ndim != 1 or preds.shape[0] != n_samples:
                raise ValueError(f"_predict: árvore {i} devolveu shape inválido {preds.shape}")

            all_preds[i, :] = preds

        # voto maioritário por amostra
        final_preds = np.empty(n_samples, dtype=object)
        for j in range(n_samples):
            valores = all_preds[:, j]
            uniq, counts = np.unique(valores, return_counts=True)
            final_preds[j] = uniq[np.argmax(counts)]

        # tentar devolver com o mesmo tipo de y
        try:
            return final_preds.astype(dataset.y.dtype)
        except:
            return final_preds

    # -------------------------------------------------------------------------
    #                                SCORE
    # -------------------------------------------------------------------------

    def _score(self, dataset: Dataset) -> float:
        """
        Calcula a accuracy do modelo no dataset fornecido.
        """
        if dataset.y is None:
            raise ValueError("_score: dataset deve conter y.")
        preds = self._predict(dataset)
        return float(accuracy(dataset.y, preds))

    # -------------------------------------------------------------------------
    #                         MÉTODOS PÚBLICOS
    # -------------------------------------------------------------------------

    def fit(self, dataset: Dataset) -> "RandomForestClassifier":
        """Treina o modelo."""
        return self._fit(dataset)

    def predict(self, dataset_or_X) -> np.ndarray:
        """
        Produz previsões a partir de:
          • um Dataset
          • ou uma matriz numpy 2D (X)
        """
        if isinstance(dataset_or_X, Dataset):
            return self._predict(dataset_or_X)

        X = np.asarray(dataset_or_X, dtype=float)
        if X.ndim != 2:
            raise ValueError("predict: X deve ser matriz 2D.")

        tmp = Dataset(X=X, y=None)
        return self._predict(tmp)

    def score(self, dataset: Dataset) -> float:
        """Calcula a accuracy no dataset fornecido."""
        return self._score(dataset)
