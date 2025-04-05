# tests/test_model.py
from iris_classifier import DataProcessor, IrisClassifier, split_data
import pandas as pd
import pytest

def test_iris_classifier():
    processor = DataProcessor()
    X, y = processor.load_data()
    X_processed = processor.preprocess(X)
    
    X_train, X_test, y_train, y_test = split_data(X_processed, y)
    
    model = IrisClassifier(n_estimators=10)
    model.fit(X_train, y_train)
    
    metrics = model.evaluate(X_test, y_test)
    assert metrics["accuracy"] > 0.9, "Accuracy should be above 90%"
    
    model.save("test_model.pkl")
    loaded_model = IrisClassifier.load("test_model.pkl")
    loaded_metrics = loaded_model.evaluate(X_test, y_test)
    assert loaded_metrics["accuracy"] == metrics["accuracy"], "Loaded model performance differs"
    
    import os
    os.remove("test_model.pkl")