import numpy as np
from si.data.dataset import Dataset

def train_test_split(dataset: Dataset, test_size: float, random_state: int = 42):
    np.random.seed(random_state)
    n_samples = dataset.shape()[0]
    test_samples = int(n_samples * test_size)
    permutations = np.random.permutation(n_samples)
    