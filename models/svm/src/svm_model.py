
import joblib
import pandas as pd

def load_svm_model(model_path="svm_outputs/svm_model.joblib"):
    """Load the trained SVM model."""
    return joblib.load(model_path)


def predict_crash_segments(model, data):
    """Predict crash presence for road segments."""
    return model.predict(data)


def evaluate_svm(model, X_test, y_test):
    """Evaluate SVM performance."""
    from sklearn.metrics import (
        accuracy_score, precision_score,
        recall_score, f1_score,
        roc_auc_score, average_precision_score
    )

    predictions = model.predict(X_test)
    scores = model.decision_function(X_test)

    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, scores),
        "PR-AUC": average_precision_score(y_test, scores)
    }
