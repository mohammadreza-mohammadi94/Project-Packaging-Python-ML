from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import pandas as pd


class DataProcessor:
    """Handles loading and preprocessing of the Iris dataset."""

    def __init__(self):
        self.scaler = StandardScaler()

    def load_data(self):
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = iris.target
        return X, y
    
    def preprocess(self, X):
        X = X.fillna(X.mean())
        X_scaled = self.scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns)
    
    def transform(self, X):
        X = X.fillna(X.mean())
        return pd.DataFrame(self.scaler.transform(X), columns=X.columns)
    
    