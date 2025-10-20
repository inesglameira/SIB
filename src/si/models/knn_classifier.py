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
        distance_to_all_points = self.distance_function(sample, self.dataset.X)
        indexes_of_nn = np.argsort(distance_to_all_points)[:self.k]
        nn_labels = self.dataset.y[indexes_of_nn]
        unique_labels, counts = np.unique(nn_labels, return_counts = True)
        label = unique_labels[np.argmax(counts)]
        return label
        
    def _predict(self, dataset) -> np.ndarray:
        return np.apply_along_axis(self._get_closest_neighbors, axis = 1, arr = dataset.X)
    
    def score(self, dataset):
        return accuracy(dataset.y, self.predict(dataset))