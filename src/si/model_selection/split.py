import numpy as np
from si.data.dataset import Dataset

def train_test_split(dataset: Dataset, test_size: float, random_state: int = 42):
    np.random.seed(random_state)
    n_samples = dataset.shape()[0]
    test_samples = int(n_samples * test_size)
    permutations = np.random.permutation(n_samples)
    test_indexes = permutations[:test_samples]
    train_indexes = permutations[test_samples:]

    train_dataset = Dataset( X = dataset.X[train_indexes], y = dataset.y[train_indexes], 
                            features = dataset.features, label = dataset.label)
    
    test_dataset = Dataset( X = dataset.X[train_indexes], y = dataset.y[train_indexes], 
                            features = dataset.features, label = dataset.label)
    
    return train_dataset, test_dataset

import numpy as np
from si.data.dataset import Dataset

def stratified_train_test_split(
    dataset: Dataset, 
    test_size: float = 0.2, 
    random_state: int = 42
):
    """
    Realiza uma divisão estratificada do Dataset em treino e teste,
    garantindo que as proporções das classes são mantidas.

    Parâmetros
    ----------
    dataset : Dataset
        O objeto Dataset a dividir.
    test_size : float
        Proporção do conjunto de teste (ex.: 0.2 = 20%).
    random_state : int
        Para baralhar os índices de cada classe.

    Retorna
    -------
    (Dataset, Dataset)
        Um tuplo contendo (train_dataset, test_dataset), ambos estratificados.
    """

    if dataset.y is None:
        raise ValueError("stratified_train_test_split requer que o Dataset tenha um vetor y.")

    # Fixar aleatoriedade
    rng = np.random.default_rng(random_state)

    # Obter classes únicas
    labels = np.unique(dataset.y)

    train_indices = []
    test_indices = []

    # Loop pelas classes (como descrito no enunciado)
    for label in labels:
        # índices das amostras desta classe
        class_indices = np.where(dataset.y == label)[0]

        # Baralhar
        rng.shuffle(class_indices)

        # calcular nº de elementos desta classe que irão para teste
        n_test = int(len(class_indices) * test_size)

        # Separar
        test_cls = class_indices[:n_test]
        train_cls = class_indices[n_test:]

        test_indices.extend(test_cls)
        train_indices.extend(train_cls)

    # Converter listas em arrays
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    # Criar datasets usando os índices estratificados
    X_train = dataset.X[train_indices]
    y_train = dataset.y[train_indices]

    X_test = dataset.X[test_indices]
    y_test = dataset.y[test_indices]

    train_dataset = Dataset(
        X=X_train, 
        y=y_train, 
        features=dataset.features, 
        label=dataset.label
    )

    test_dataset = Dataset(
        X=X_test, 
        y=y_test, 
        features=dataset.features, 
        label=dataset.label
    )

    return train_dataset, test_dataset
