# iris_classifier/model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

class IrisClassifier:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state)
        self.is_trained = False

    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_trained = True
        return self
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call fit() first.")
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call fit() first.")
        y_pred = self.predict(X)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "report": classification_report(y, y_pred, target_names=["setosa", "versicolor", "virginica"])
        }
    
    def save(self, filepath):
        """Save the trained model to a file."""
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Cannot save.")
        joblib.dump(self.model, filepath)
    
    @staticmethod
    def load(filepath):
        """Load a trained model from a file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found.")
        model = IrisClassifier()
        model.model = joblib.load(filepath)
        model.is_trained = True
        return model


