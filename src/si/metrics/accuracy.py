import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    accuracy_score = correct_predictions / total_predictions
    return accuracy_score
