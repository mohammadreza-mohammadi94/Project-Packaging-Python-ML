from iris_classifier import DataProcessor, IrisClassifier, split_data

# Load and preprocess data
processor = DataProcessor()
X, y = processor.load_data()
X_processed = processor.preprocess(X)

# Split data
X_train, X_test, y_train, y_test = split_data(X_processed, y)

# Train model
model = IrisClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
metrics = model.evaluate(X_test, y_test)
print("Accuracy:", metrics["accuracy"])
print("Report:\n", metrics["report"])

# Save model
model.save("iris_model.pkl")