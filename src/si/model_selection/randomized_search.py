from __future__ import annotations
from typing import Dict, Callable, Any, List
import numpy as np

from si.data.dataset import Dataset
from si.model_selection.cross_validate import k_fold_cross_validation


def randomized_search_cv(
    model,
    dataset: Dataset,
    hyperparameter_grid: Dict[str, np.ndarray],
    scoring: Callable,
    cv: int = 3,
    n_iter: int = 10,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executa Randomized Search com validação cruzada para otimização de hiperparâmetros.

    Em cada iteração, é escolhida aleatoriamente uma combinação de hiperparâmetros
    a partir das distribuições fornecidas, o modelo é avaliado com validação cruzada
    e o score médio é registado.

    Parâmetros
    ----------
    model : object
        Modelo a otimizar (deve suportar fit e score).
    dataset : Dataset
        Dataset completo (X e y).
    hyperparameter_grid : dict
        Dicionário onde cada chave é o nome de um hiperparâmetro e o valor é
        um array de valores possíveis.
    scoring : callable
        Função de scoring (ex.: accuracy).
    cv : int
        Número de folds da validação cruzada.
    n_iter : int
        Número de combinações aleatórias a testar.
    random_state : int
        Seed para reprodutibilidade.

    Returns
    -------
    dict
        Dicionário com:
        - hyperparameters: lista de dicionários testados
        - scores: lista de scores médios
        - best_hyperparameters: melhor conjunto encontrado
        - best_score: melhor score obtido
    """

    rng = np.random.default_rng(random_state)

    # Verificar se os hiperparâmetros existem no modelo
    for param in hyperparameter_grid.keys():
        if not hasattr(model, param):
            raise ValueError(f"O modelo não possui o hiperparâmetro '{param}'.")

    tested_params: List[Dict[str, Any]] = []
    scores: List[float] = []

    for _ in range(n_iter):

        # 1) Escolher uma combinação aleatória de hiperparâmetros
        current_params = {
            param: rng.choice(values)
            for param, values in hyperparameter_grid.items()
        }

        # 2) Definir os hiperparâmetros no modelo
        for param, value in current_params.items():
            setattr(model, param, value)

        # 3) Avaliar com validação cruzada
        cv_scores = k_fold_cross_validation(
            model=model,
            dataset=dataset,
            scoring=scoring,
            cv=cv
        )

        mean_score = float(np.mean(cv_scores))

        # 4) Guardar resultados
        tested_params.append(current_params)
        scores.append(mean_score)

    # 5) Melhor resultado
    best_idx = int(np.argmax(scores))
    best_score = scores[best_idx]
    best_params = tested_params[best_idx]

    return {
        "hyperparameters": tested_params,
        "scores": scores,
        "best_hyperparameters": best_params,
        "best_score": best_score
    }
