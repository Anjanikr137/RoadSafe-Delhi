"""
models/logistic_regression/src/logistic_model.py
Defines the Logistic Regression model configuration and builder for RoadSafe Delhi.
Supports baseline unweighted and balanced class-weight configurations.
"""

from sklearn.linear_model import LogisticRegression


def build_logistic_model(
    C=1.0,
    class_weight="balanced",
    solver="lbfgs",
    max_iter=1000,
    random_state=42
):
    """
    Constructs an L2-regularized Logistic Regression classifier.
    """
    model = LogisticRegression(
        C=C,
        class_weight=class_weight,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state
    )
    return model
