from si.base.transformer import Transformer
from si.data.dataset import Dataset

class VarianceThreshold(Transformer):

    def __init__(self, threshold: float, **kwargs):
        self.threshold = threshold
        self.variances = None
    
    def _fit(self, dataset: Dataset) -> 'VarianceThreshold':
        self.variances = np.var(dataset.X, axis=0)

    def _transform(self, dataset: Dataset) -> Dataset:
        mask = self.variance >= self.threshold
        X = dataset.X[: , mask]
        features = np.array(dataset.features)[mask]
        return Dataset(X=X, features = features, y = dataset.y, label = dataset.label)


        