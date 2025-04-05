
# Iris Packaging Demo

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)

A Python package demonstrating how to structure, build, and package a machine learning project using modern tools like `pyproject.toml` and `build`. This project implements a Random Forest classifier for the Iris dataset, with features like data preprocessing, model training, evaluation, and model persistence.

---

## Table of Contents
- [Purpose](#purpose)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Packaging Workflow](#packaging-workflow)
- [Running Tests](#running-tests)
- [Contributing](#contributing)
- [License](#license)

---

## Purpose
This repository serves as a practical example of packaging a machine learning project in Python. It showcases:
- How to structure a Python package with multiple modules.
- Using `pyproject.toml` for modern package configuration.
- Building and distributing the package with `build` and `wheel`.
- Best practices for testing and documentation.

The project uses the famous Iris dataset and a Random Forest classifier as a real-world ML example.

---

## Features
- **Data Preprocessing**: Handles missing values and scales features using `StandardScaler`.
- **Model Training**: Implements a Random Forest classifier with configurable hyperparameters.
- **Evaluation**: Provides accuracy and detailed classification metrics.
- **Model Persistence**: Save and load trained models using `joblib`.
- **Tests**: Includes unit tests with `pytest`.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- `pip` or `conda` for package management

### Install from Source
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/iris-packaging-demo.git
   cd iris-packaging-demo
   ```
2. Install the package:
   ```bash
   pip install .
   ```

### Install from Wheel (Pre-built)
If you’ve built the package locally:
```bash
pip install dist/iris_classifier-0.1.0-py3-none-any.whl
```

---

## Usage

Here’s a quick example of how to use the package:

```python
from iris_classifier import DataProcessor, IrisClassifier, split_data

# Load and preprocess data
processor = DataProcessor()
X, y = processor.load_data()
X_processed = processor.preprocess(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = split_data(X_processed, y)

# Train the model
model = IrisClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate the model
metrics = model.evaluate(X_test, y_test)
print("Accuracy:", metrics["accuracy"])
print("Report:\n", metrics["report"])

# Save the model
露

# Load and use a saved model
loaded_model = IrisClassifier.load("iris_model.pkl")
```

**Expected Output:**
```
Accuracy: 0.9666666666666667
Report:
               precision    recall  f1-score   support
      setosa       1.00      1.00      1.00        10
  versicolor       0.90      1.00      0.95         9
   virginica       1.00      0.91      0.95        11
```

---

## Project Structure

```
iris-packaging-demo/
├── iris_classifier/        # Main package directory
│   ├── __init__.py
│   ├── data.py           # Data loading and preprocessing
│   ├── model.py          # Model definition and training
│   └── utils.py          # Utility functions
├── tests/                 # Test directory
│   ├── __init__.py
│   └── test_model.py     # Unit tests
├── pyproject.toml         # Package configuration
├── README.md              # This file
└── LICENSE                # License file (MIT)
```

---

## Packaging Workflow

To build and package this project locally:

1. **Install build tools**:
   ```bash
   pip install build setuptools wheel
   ```

2. **Build the package**:
   ```bash
   python -m build
   ```
   This generates `.tar.gz` and `.whl` files in the `dist/` directory.

3. **Install the built package**:
   ```bash
   pip install dist/iris_classifier-0.1.0-py3-none-any.whl
   ```

---

## Running Tests

To run the unit tests:

1. **Install test dependencies**:
   ```bash
   pip install -e ".[test]"
   ```

2. **Run tests**:
   ```bash
   pytest tests/
   ```

---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to your branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

Please ensure your code follows PEP 8 style guidelines and includes tests.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
