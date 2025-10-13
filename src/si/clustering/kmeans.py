import numpy as np
from si.data.dataset import Dataset
from si.base.transformer import Transformer
from si.statistics import euclidean_distance

class KMeans(Transformer):
    def __init__(self, k = 3, max_iter = 100, distance = euclidean_distance, **kwargs):

        self.k = k
        self.max_iter = max_iter
        self.distance = distance

        self.centroids = None
        self.labels = None
    
    def _init_centroids(self, dataset: Dataset):
        random_indices = np.random.permutation(dataset.shape()[0])[:self.k] #shape -> método da classe dataset
        self.centroids = dataset.X[random_indices, :] # centroides são as linhas

    def _calculate_distances(self, sample): #_ a classe é privado, se não tiver o _ no ínicio é público, os métodos privados são utilizados internamente e não pelo utilizador
        return self.distance(sample, self.centroids)  
        
    def _get_closest_centroids(self, sample):
        centroids_distance = self.calculate_distances(sample)
        np.argmin(centroids_distance, axis = 0) # axis =1 por defeito 
        return centroids_distance
    
    def _fit(self, dataset):
        i = 0
        convergence = False

        labels = np.zeros(dataset.shape()[0])
        while not convergence and i < self.max_iter:

            new_labels = np.apply_along_axis(self._get_closest_centroids, axis = 1, arr = dataset.X) #datapoint é um amostra, uma feature. Aplica vetorização, é muito mais eficiente (é o mesmo que aplicar um ciclo for). Retorna um vetor com o indce dos centroides, um vetor igual ao tamanho do numero de linhas. Para cada linha vamos ter um centróide associado
            
            self.labels = new_labels
            
            centroids = []
            for j in range(self.k):
                mask = new_labels == j #mask é um vetor booleano
                new_centroid = np.mean(dataset.X[mask])
                centroids.append(new_centroid)
            
            centroids = np.array(centroids)
            convergence = not np.any(new_labels != labels) #verifica se houve alterações
    
            labels = new_labels
            i += 1

        self.labels = labels
        return self 
            
    def _transform(self, dataset):
        euclidean_distance = np.apply_along_axis(self._calculate_distancesds, axis = 1, arr = dataset.X) #datapoint é um amostra, uma feature. Aplica vetorização, é muito mais eficiente (é o mesmo que aplicar um ciclo for). Retorna um vetor com o indce dos centroides, um vetor igual ao tamanho do numero de linhas. Para cada linha vamos ter um centróide associado
        return euclidean_distance
    
    def _predict(self, dataset):
        pass

    