import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegression(Model):
    """
    Regressão linear com regularização L2 (Ridge) usando Gradient Descent.

    Parâmetros
    ----------
    l2_penalty : float
    Parâmetro de regularização L2 (lambda).

    alpha : float
    Learning rate (taxa de aprendizagem).

    max_iter : int
    Número máximo de iterações do gradient descent.

    patience : int
    Número de iterações sem melhoria antes de fazer early stopping.

    scale : bool
    Se True, escala (normaliza) as features pelo z-score (mean/std).

    Atributos estimados
    -------------------
    theta : np.ndarray, shape (n_features,)
        Coeficientes (weights) do modelo.
    theta_zero : float
        Intercepto (bias).
    mean : np.ndarray, shape (n_features,)
        Média das features usada na normalização (se scale=True).
    std : np.ndarray, shape (n_features,)
        Desvio padrão das features usado na normalização (se scale=True).
    cost_history : dict
        Histórico do custo por iteração (útil para debugging/plot).
    """

    def __init__(self, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 1000, patience: int = 5,
                 scale: bool = True, **kwargs):
        """

        Parâmetros
        ----------
        l2_penalty : float  
            Parâmetro de regularização L2.  
        alpha : float  
            Taxa de aprendizagem (learning rate).  
        max_iter : int  
            Número máximo de iterações do algoritmo.  
        patience : int  
            Número de iterações consecutivas sem melhoria permitido antes de parar o treino (early stopping).  
        scale : bool  
            Indica se os dados devem ser escalados (normalizados) antes do treino.
        """

        # parametros
        super().__init__(**kwargs)
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale

        # atributos
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = {}

    def _fit(self, dataset: Dataset) -> 'RidgeRegression':
        """
        Ajusta (treina) o modelo utilizando o dataset fornecido.

        Parâmetros
        ----------
        dataset : Dataset
            O dataset sobre o qual o modelo será treinado.

        Retorna
        -------
        self : RidgeRegression
            O modelo treinado.
        """
        if self.scale:
            # computa mean and std
            self.mean = np.nanmean(dataset.X, axis=0)
            self.std = np.nanstd(dataset.X, axis=0)
            # scale do dataset
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        m, n = dataset.shape()

        # inicia os model parameters
        self.theta = np.zeros(n)
        self.theta_zero = 0

        i = 0
        early_stopping = 0
        # gradient descent
        while i < self.max_iter and early_stopping < self.patience:
            # predicted y
            y_pred = np.dot(X, self.theta) + self.theta_zero

            # computa and atualiza o gradient com o  learning rate
            gradient = (self.alpha / m) * np.dot(y_pred - dataset.y, X)

            # computa a penalidade
            penalization_term = self.theta * (1 - self.alpha * (self.l2_penalty / m))

            # atualiza os model parameters
            self.theta = penalization_term - gradient
            self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)

            # computa o custo
            self.cost_history[i] = self.cost(dataset)
            if i > 0 and self.cost_history[i] > self.cost_history[i - 1]:
                early_stopping += 1
            else:
                early_stopping = 0
            i += 1

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Prediz o output (variável dependente) para o dataset fornecido.

        Parâmetros
        ----------
        dataset : Dataset
            O dataset para o qual se pretende obter previsões.

        Retorna
        -------
        predictions : np.ndarray
            As previsões geradas pelo modelo para o dataset.
        """
        X = (dataset.X - self.mean) / self.std if self.scale else dataset.X
        return np.dot(X, self.theta) + self.theta_zero

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Calcula o erro quadrático médio (MSE) do modelo no dataset fornecido.

        Parâmetros
        ----------
        dataset : Dataset
            O dataset no qual se pretende calcular o MSE.
        predictions : np.ndarray
            Os valores preditos pelo modelo.

        Retorna
        -------
        mse : float
        O erro quadrático médio do modelo.
        """
        return mse(dataset.y, predictions)

    def cost(self, dataset: Dataset) -> float:
        """
            Calcula a função de custo (função J) do modelo no dataset, utilizando regularização L2.

        Parâmetros
        ----------
        dataset : Dataset
            O dataset no qual se pretende calcular a função de custo.

        Retorna
        -------
        cost : float
            O valor da função de custo do modelo.
        """
        y_pred = self.predict(dataset)
        return (np.sum((y_pred - dataset.y) ** 2) + (self.l2_penalty * np.sum(self.theta ** 2))) / (2 * len(dataset.y))


if __name__ == '__main__':
    # importa dataset
    from si.data.dataset import Dataset

    # faz um dataset linear
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    dataset_ = Dataset(X=X, y=y)

    # fit do model
    model = RidgeRegression()
    model.fit(dataset_)

    # obter coefs
    print(f"Parameters: {model.theta}")

    # computa o score
    score = model.score(dataset_)
    print(f"Score: {score}")

    # computa o custo
    cost = model.cost(dataset_)
    print(f"Cost: {cost}")

    # predict
    y_pred_ = model.predict(Dataset(X=np.array([[3, 5]])))
    print(f"Predictions: {y_pred_}")