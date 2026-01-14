import numpy as np
from si.base.model import Model
from si.metrics.accuracy import accuracy


class KNNClassifier(Model):
    def __init__(self, k: int, distance_function: callable, **kwargs):
        self.k = k
        self.distance_function = distance_function

        self.dataset = None
   
    def _fit(self, dataset) -> "KNNClassifier":
        self.dataset = dataset
    
    def _get_closest_neighbors(self, sample: np.ndarray):
        distances = np.linalg.norm(self.dataset.X - sample, axis=1)
        idx = np.argsort(distances)[:self.k]
        labels = self.dataset.y[idx]
        values, counts = np.unique(labels, return_counts=True)
        return values[np.argmax(counts)]

        
    def _predict(self, dataset) -> np.ndarray:
        return np.apply_along_axis(self._get_closest_neighbors, axis = 1, arr = dataset.X)
    
    def score(self, dataset):
        return accuracy(dataset.y, self.predict(dataset))
    
    def _score(self, dataset, predictions):
        return accuracy(dataset.y, predictions)