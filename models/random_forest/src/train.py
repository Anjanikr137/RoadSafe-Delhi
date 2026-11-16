"""
models/random_forest/src/train.py
Training, validation, and artifact serialization script for RoadSafe Delhi Random Forest Classifier.

Assigned Member: Nakshtr
Model: Random Forest Classifier (sklearn.ensemble.RandomForestClassifier)
Target: is_crash_segment
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# Ensure local imports work cleanly
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import RoadSafeRFPreprocessor
from rf_model import build_random_forest_model, find_optimal_threshold


def train_random_forest(
    data_dir=None,
    artifacts_dir=None,
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=42
):
    start_time = time.time()

    # Project-relative directories
    if data_dir is None:
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data_final"))
    if artifacts_dir is None:
        artifacts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../artifacts"))

    os.makedirs(artifacts_dir, exist_ok=True)

    print("=================================================================")
    print("ROADSAFE DELHI — RANDOM FOREST CLASSIFIER TRAINING")
    print("=================================================================")
    print(f"Data directory:      {data_dir}")
    print(f"Artifacts directory: {artifacts_dir}")

    # Load canonical partitions
    train_path = os.path.join(data_dir, "train.parquet")
    val_path = os.path.join(data_dir, "validation.parquet")

    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)

    print(f"\nLoaded Train partition: {len(df_train):,} segments (Crash: {df_train['is_crash_segment'].sum():,})")
    print(f"Loaded Val partition:   {len(df_val):,} segments (Crash: {df_val['is_crash_segment'].sum():,})")

    # Spatial block integrity verification
    train_blocks = set(df_train["spatial_block"].unique())
    val_blocks = set(df_val["spatial_block"].unique())
    assert train_blocks.isdisjoint(val_blocks), "FATAL: Spatial overlap between Train and Val blocks!"
    print(f"Spatial Block Isolation Confirmed: {len(train_blocks)} train blocks, {len(val_blocks)} val blocks (0 overlap).")

    # 1. Zero-leakage Preprocessing
    preprocessor = RoadSafeRFPreprocessor()
    X_train = preprocessor.fit_transform(df_train)
    X_val = preprocessor.transform(df_val)

    y_train = df_train["is_crash_segment"].values.astype(int)
    y_val = df_val["is_crash_segment"].values.astype(int)

    print(f"Feature matrix dimension: {X_train.shape[1]} features (One-hot encoded, scale-invariant).")

    # 2. Train Selected Model
    print(f"\nFitting RandomForestClassifier with parameters:")
    print(f"  n_estimators:      {n_estimators}")
    print(f"  max_depth:         {max_depth}")
    print(f"  min_samples_split: {min_samples_split}")
    print(f"  min_samples_leaf:  {min_samples_leaf}")
    print(f"  max_features:      {max_features}")
    print(f"  class_weight:      {class_weight}")
    print(f"  random_state:      {random_state}")

    rf_model = build_random_forest_model(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state
    )

    t0 = time.time()
    rf_model.fit(X_train, y_train)
    fit_duration = time.time() - t0
    print(f"Model fitted in {fit_duration:.2f} seconds.")

    # 3. Validation Assessment & Threshold Tuning
    val_probs = rf_model.predict_proba(X_val)[:, 1]
    val_roc_auc = float(roc_auc_score(y_val, val_probs))
    val_pr_auc = float(average_precision_score(y_val, val_probs))

    optimal_t, best_val_f1, _, _, _ = find_optimal_threshold(y_val, val_probs)
    val_preds = (val_probs >= optimal_t).astype(int)

    val_acc = float(accuracy_score(y_val, val_preds))
    val_prec = float(precision_score(y_val, val_preds, zero_division=0))
    val_rec = float(recall_score(y_val, val_preds, zero_division=0))
    val_cm = confusion_matrix(y_val, val_preds).tolist()

    print("\n--- Official Validation Partition Performance ---")
    print(f"  Validation ROC-AUC:   {val_roc_auc:.4f}")
    print(f"  Validation PR-AUC:    {val_pr_auc:.4f}")
    print(f"  Optimal Threshold:    {optimal_t:.6f}")
    print(f"  Validation Accuracy:  {val_acc*100:.2f}%")
    print(f"  Validation Precision: {val_prec:.4f}")
    print(f"  Validation Recall:    {val_rec:.4f}")
    print(f"  Validation F1-Score:  {best_val_f1:.4f}")
    print(f"  Confusion Matrix:     {val_cm}")

    # 4. Serialize Production Artifacts
    print(f"\nSerializing artifacts to {artifacts_dir}...")
    joblib.dump(rf_model, os.path.join(artifacts_dir, "random_forest_model.joblib"))
    preprocessor.save(artifacts_dir)

    # Threshold artifact
    threshold_meta = {
        "optimal_threshold": optimal_t,
        "tuning_metric": "f1_score",
        "tuning_partition": "validation",
        "validation_f1": best_val_f1,
        "validation_precision": val_prec,
        "validation_recall": val_rec,
        "validation_pr_auc": val_pr_auc,
        "validation_roc_auc": val_roc_auc
    }
    with open(os.path.join(artifacts_dir, "threshold.json"), "w") as f:
        json.dump(threshold_meta, f, indent=2)

    # Model configuration artifact
    model_config = {
        "model_name": "Random Forest Classifier",
        "algorithm": "sklearn.ensemble.RandomForestClassifier",
        "assigned_member": "Nakshtr",
        "target": "is_crash_segment",
        "target_type": "binary_classification",
        "hyperparameters": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "class_weight": class_weight,
            "bootstrap": True,
            "random_state": random_state
        },
        "feature_count": X_train.shape[1],
        "training_samples": len(df_train),
        "validation_samples": len(df_val),
        "scaler_applied": False,
        "validation_metrics": {
            "roc_auc": val_roc_auc,
            "pr_auc": val_pr_auc,
            "optimal_threshold": optimal_t,
            "f1_score": best_val_f1,
            "precision": val_prec,
            "recall": val_rec,
            "accuracy": val_acc
        },
        "training_time_seconds": round(time.time() - start_time, 2)
    }
    with open(os.path.join(artifacts_dir, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=2)

    print("TRAINING AND ARTIFACT SERIALIZATION COMPLETED SUCCESSFULLY!")
    return rf_model, preprocessor, optimal_t


if __name__ == "__main__":
    train_random_forest()
