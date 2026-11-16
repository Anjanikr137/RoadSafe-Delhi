"""
models/random_forest/src/rf_model.py
Random Forest model definition and hyperparameter configuration for RoadSafe Delhi.

Assigned Member: Nakshtr
Model: Random Forest Classifier (sklearn.ensemble.RandomForestClassifier)
Target: is_crash_segment
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, f1_score


def build_random_forest_model(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced_subsample",
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
    **kwargs
):
    """
    Constructs configured RandomForestClassifier.
    Default parameters correspond to the validated optimal candidate M6.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        bootstrap=bootstrap,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs
    )


def find_optimal_threshold(y_true, y_probs, metric="f1"):
    """
    Finds optimal classification threshold on validation predictions.
    Defaults to maximizing minority-class F1-score.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    # Compute F1 for each threshold
    f1s = np.zeros_like(thresholds)
    for i, t in enumerate(thresholds):
        preds = (y_probs >= t).astype(int)
        f1s[i] = f1_score(y_true, preds, zero_division=0)
    best_idx = int(np.argmax(f1s))
    optimal_t = float(thresholds[best_idx])
    best_f1 = float(f1s[best_idx])
    return optimal_t, best_f1, precisions, recalls, thresholds
