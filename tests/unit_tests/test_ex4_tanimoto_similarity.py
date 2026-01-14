import numpy as np
from si.statistics.tanimoto_similarity import tanimoto_similarity

def test_tanimoto_simple_case():
    x = np.array([1, 0, 1, 1])
    y = np.array([
        [1, 0, 1, 1],  # igual a x
        [1, 1, 0, 0]   # parcialmente diferente
    ])

    result = tanimoto_similarity(x, y)

    # Similaridade com ele próprio deve ser 1
    assert np.isclose(result[0], 1.0)

    # Segundo valor deve estar entre 0 e 1
    assert 0 <= result[1] <= 1

def test_tanimoto_output_shape():
    x = np.array([1, 0, 0, 1])
    y = np.random.randint(0, 2, size=(5, 4))

    result = tanimoto_similarity(x, y)

    assert isinstance(result, np.ndarray)
    assert result.shape == (5,)

def test_tanimoto_value_range():
    x = np.random.randint(0, 2, size=6)
    y = np.random.randint(0, 2, size=(10, 6))

    result = tanimoto_similarity(x, y)

    assert np.all(result >= 0)
    assert np.all(result <= 1)

def test_tanimoto_accepts_lists():
    x = [1, 0, 1]
    y = [
        [1, 0, 1],
        [0, 1, 0]
    ]

    result = tanimoto_similarity(x, y)

    assert len(result) == 2
