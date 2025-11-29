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
        Prediz labels para dataset.X usando voto majoritário das árvores.
        Retorna um array shape (n_samples,) com as labels preditas.
        """
        X = np.asarray(dataset.X, dtype=float)
        n_samples = X.shape[0]

        if len(self.trees) == 0:
            raise ValueError("_predict: modelo não treinado. Chame _fit().")

        # recolher preds de cada árvore: shape (n_estimators, n_samples)
        all_preds = np.empty((self.n_estimators, n_samples), dtype=object)

        for i, (feat_idx, tree) in enumerate(self.trees):
            # selecionar colunas usadas na árvore
            X_sub = X[:, feat_idx]
            preds = tree.predict(X_sub)  # assume retorno shape (n_samples,)
            all_preds[i, :] = preds

        # para cada sample, escolher a classe mais frequente (majority vote)
        # converter all_preds para int se necessário
        # vamos operar coluna a coluna
        final_preds = np.empty(n_samples, dtype=all_preds.dtype)

        for j in range(n_samples):
            col = all_preds[:, j]
            # bincount necessita de ints não negativos; labels podem não ser 0..K-1
            # por segurança, usamos unique com counts
            uniq, counts = np.unique(col, return_counts=True)
            winner = uniq[np.argmax(counts)]
            final_preds[j] = winner

        return np.asarray(final_preds, dtype=final_preds.dtype)

    def _score(self, dataset: Dataset) -> float:
        """
        Calcula a accuracy do modelo no dataset fornecido.
        """
        if dataset.y is None:
            raise ValueError("_score: dataset deve conter y")
        y_pred = self._predict(dataset)
        return accuracy(dataset.y, y_pred)

    # Syntactic sugar: fit/predict/score que chamam os métodos internos
    def fit(self, dataset: Dataset) -> "RandomForestClassifier":
        return self._fit(dataset)

    def predict(self, dataset_or_X) -> np.ndarray:
        """
        Conveniência: aceita Dataset ou array X.
        """
        if isinstance(dataset_or_X, Dataset):
            data = dataset_or_X
        else:
            # assumir matrix X fornecida - criar dataset temporário (sem y)
            X = np.asarray(dataset_or_X, dtype=float)
            data = Dataset(X=X, y=None, features=None, label=None)
        return self._predict(data)

    def score(self, dataset: Dataset) -> float:
        return self._score(dataset)
    
    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Prediz labels para dataset.X usando voto majoritário das árvores treinadas.
        """
        if len(self.trees) == 0:
            raise ValueError("_predict: modelo não treinado. Chame _fit() primeiro.")

        X = np.asarray(dataset.X, dtype=float)
        n_samples = X.shape[0]

        # Matriz onde cada linha são as previsões de uma árvore
        all_preds = np.empty((self.n_estimators, n_samples), dtype=object)

        for i, (feat_idx, tree) in enumerate(self.trees):
            X_sub = X[:, feat_idx]

            # tentar prever diretamente com X_sub
            try:
                preds = tree.predict(X_sub)
            except Exception:
                # API alternativa: árvore requer Dataset
                from si.data.dataset import Dataset as _DS
                tmp_ds = _DS(X=X_sub, y=None, features=None, label=None)
                preds = tree.predict(tmp_ds)

            all_preds[i, :] = preds

        # Voto maioritário
        final_preds = np.empty(n_samples, dtype=object)
        for j in range(n_samples):
            col = all_preds[:, j]
            uniq, counts = np.unique(col, return_counts=True)
            winner = uniq[np.argmax(counts)]
            final_preds[j] = winner

        return np.asarray(final_preds, dtype=final_preds.dtype)
