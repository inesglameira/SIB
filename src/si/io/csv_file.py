from si.data.dataset import Dataset

def read_csv(filename: str, sep: str, features: bool, label: bool) -> Dataset:
    import pandas as pd
    import numpy as np


    dataframe = pd.read_csv(filepath_or_buffer=filename, sep=sep)

    if features and label:
        X = dataframe.iloc[:, :-1].to_numpy()
        y = dataframe.iloc[:, -1].to_numpy()
        feature_names = dataframe.columns[:-1]
        label_name = dataframe.columns[-1] 
        return Dataset(X=X, y=y, features=feature_names, label=label_name)
    
    elif features:
        X = dataframe.to_numpy()
        feature_names = dataframe.columns
        return Dataset(X=X, features=feature_names)
    
    elif label:
        X = np.array()
        y = dataframe.iloc[:, -1].to_numpy()
        label_name = dataframe.columns[-1]
        return Dataset(X=X, y=y, label=label_name)  #X=X é um dataset vazio
    
    else:
        return None  #não faz sentido ter um dataset sem features nem label
    
def write_csv(filename: str, dataset: Dataset, sep: str, features: bool = False, label: bool = False) -> None:
    """
    Writes a Dataset object to a CSV file
    Parameters
    ----------
    dataset : Dataset
        The Dataset object to be written to a CSV file
    filename : str
        The path to the CSV file
    sep : str, optional
        The separator to be used in the CSV file, by default ","
    """
    import pandas as pd

    dataframe = pd.DataFrame(dataset.X)
    if features:
        dataframe.columns = dataset.features

    if label:
        y = dataset.y
        label_name = dataset.label
        dataframe[label_name] = y

    else:
        y = None
        label_name = None
    
    dataframe.to_csv(filename, sep=sep, index=False)
