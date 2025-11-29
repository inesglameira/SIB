from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np

from si.data.dataset import Dataset
from si.models.decision_tree_classifier import DecisionTreeClassifier 
from si.metrics.accuracy import accuracy  

class RandomForestClassifier:
    """
    Random Forest Classifier.

    Parâmetros
    ----------
    n_estimators : int
        Número de árvores na floresta.
    max_features : Optional[int]
        Número máximo de features por árvore (se None -> int(sqrt(n_features))).
    min_sample_split : int
        Minimum samples to allow a split (pass-through para DecisionTree).
    max_depth : Optional[int]
        Maximum depth das árvores (pass-through para DecisionTree).
    mode : str
        'gini' ou 'entropy' (impurity mode, pass-through).
    seed : Optional[int]
        Semente aleatória para reprodutibilidade.

    Atributos estimados
    -------------------
    trees : List[Tuple[np.ndarray, DecisionTreeClassifier]]
        Lista de tuplos (features_index_array, trained_tree).
    """

    def __init__(
        self,
        n_estimators: int = 10,
        max_features: Optional[int] = None,
        min_sample_split: int = 2,
        max_depth: Optional[int] = None,
        mode: str = "gini",
        seed: Optional[int] = None,
    ) -> None:
        if n_estimators <= 0:
            raise ValueError("n_estimators deve ser inteiro positivo.")
        if mode not in ("gini", "entropy"):
            raise ValueError("mode deve ser 'gini' ou 'entropy'.")

        self.n_estimators = int(n_estimators)
        self.max_features = max_features
        self.min_sample_split = int(min_sample_split)
        self.max_depth = None if max_depth is None else int(max_depth)
        self.mode = mode
        self.seed = seed

        # lista de (features_idx, tree)
        self.trees: List[Tuple[np.ndarray, DecisionTreeClassifier]] = []

    def _bootstrap_indices(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """
        Gera indices bootstrap (com reposição) de tamanho n_samples.
        """
        return rng.choice(n_samples, size=n_samples, replace=True)

    def _sample_features(self, n_features: int, rng: np.random.Generator) -> np.ndarray:
        """
        Escolhe sem reposição os features para esta árvore.
        """
        if self.max_features is None:
            m = int(np.sqrt(n_features))
            m = max(1, m)
        else:
            m = min(n_features, int(self.max_features))
        return rng.choice(n_features, size=m, replace=False)

    def _fit(self, dataset: Dataset) -> "RandomForestClassifier":
        """
        Treina a floresta: para cada estimador cria um bootstrap dataset
        e uma sub-amostra de features, treina uma DecisionTreeClassifier e guarda.
        """
        if dataset.y is None:
            raise ValueError("_fit: dataset deve conter y")

        X = np.asarray(dataset.X, dtype=float)
        y = np.asarray(dataset.y)
        n_samples, n_features = X.shape

        rng = np.random.default_rng(self.seed)

        self.trees = []

        for i in range(self.n_estimators):
            # 1) bootstrap indices (amostras com reposição)
            boot_idx = self._bootstrap_indices(n_samples, rng)

            # 2) feature subspace (sem reposição)
            feat_idx = self._sample_features(n_features, rng)

            # 3) criar dataset bootstrap apenas com as features selecionadas
            X_boot = X[boot_idx][:, feat_idx]
            y_boot = y[boot_idx]

            ds_boot = Dataset(X=X_boot, y=y_boot, features=[f"f{j}" for j in feat_idx], label=dataset.label)

            # 4) criar e treinar árvore (pass-through de parâmetros relevantes)
            tree = DecisionTreeClassifier(
                min_samples_split=self.min_sample_split,
                max_depth=self.max_depth,
                mode=self.mode
            )
            tree.fit(ds_boot)

            # 5) guardar (features indices, tree)
            self.trees.append((feat_idx.copy(), tree))

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Prediz labels para dataset.X usando voto maioritário das árvores treinadas.
        """
        if len(self.trees) == 0:
            raise ValueError("_predict: modelo não treinado. Chame _fit() primeiro.")

        X = np.asarray(dataset.X, dtype=float)
        if X.ndim != 2:
            raise ValueError("_predict: X deve ser matriz 2D (n_samples, n_features).")
        n_samples = X.shape[0]

        # número real de árvores treinadas
        n_trees = len(self.trees)

        # Matriz onde cada linha são as previsões das árvores
        all_preds = np.empty((n_trees, n_samples), dtype=object)

        for i, (feat_idx, tree) in enumerate(self.trees):
            X_sub = X[:, feat_idx]

            # tentar prever diretamente
            try:
                preds = tree.predict(X_sub)
            except Exception:
                from si.data.dataset import Dataset as _DS
                tmp_ds = _DS(X=X_sub, y=None, features=None, label=None)
                preds = tree.predict(tmp_ds)

            # garantir formato correto
            preds = np.asarray(preds, dtype=object)
            if preds.ndim != 1 or preds.shape[0] != n_samples:
                raise ValueError(
                    f"_predict: árvore {i} retornou shape inválido {preds.shape}"
                )

            all_preds[i, :] = preds

        # Voto maioritário
        final_preds = np.empty(n_samples, dtype=object)

        for j in range(n_samples):
            col = all_preds[:, j]

            # remover None, se existirem
            col_valid = [v for v in col if v is not None]
            if len(col_valid) == 0:
                raise ValueError(f"_predict: nenhuma previsão válida para amostra {j}")

            uniq, counts = np.unique(col_valid, return_counts=True)
            final_preds[j] = uniq[np.argmax(counts)]

        return np.asarray(final_preds, dtype=final_preds.dtype)

