"""
models/linear_regression/src/train.py
Training and validation script for RoadSafe Delhi Linear Regression (Ridge Baseline).

Assigned Member: Anushree
Model: Ridge Regression (alpha=10.0, random_state=42)
Target: crash_density_per_km (Continuous Crash Rate per Kilometer)
"""

import os
import json
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from preprocessing import RoadSafeLinearPreprocessor


def train_linear_regression(
    data_dir=None,
    artifacts_dir=None,
    alpha=10.0,
    random_state=42,
    target_column="crash_density_per_km"
):
    start_time = time.time()

    # Configure project-relative paths
    if data_dir is None:
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data_final"))
    if artifacts_dir is None:
        artifacts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../artifacts"))

    os.makedirs(artifacts_dir, exist_ok=True)

    print("=================================================================")
    print("ROADSAFE DELHI — LINEAR REGRESSION (RIDGE BASELINE) TRAINING")
    print("=================================================================")
    print(f"Loading data partitions from: {data_dir}")

    train_path = os.path.join(data_dir, "train.parquet")
    val_path = os.path.join(data_dir, "validation.parquet")
    test_path = os.path.join(data_dir, "test.parquet")

    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)

    print(f"Partition sizes: Train={len(df_train):,}, Val={len(df_val):,}, Test={len(df_test):,}")
    print(f"Continuous Target Column: {target_column}")

    # Zero-leakage spatial split verification
    train_blocks = set(df_train["spatial_block"].unique())
    val_blocks = set(df_val["spatial_block"].unique())
    test_blocks = set(df_test["spatial_block"].unique())
    assert train_blocks.isdisjoint(val_blocks), "Spatial overlap between Train and Val!"
    assert train_blocks.isdisjoint(test_blocks), "Spatial overlap between Train and Test!"
    assert val_blocks.isdisjoint(test_blocks), "Spatial overlap between Val and Test!"

    # 1. Preprocessing (fitted strictly on training data)
    preprocessor = RoadSafeLinearPreprocessor()
    X_train = preprocessor.fit_transform(df_train)
    X_val = preprocessor.transform(df_val)
    X_test = preprocessor.transform(df_test)

    y_train = df_train[target_column].values.astype(float)
    y_val = df_val[target_column].values.astype(float)
    y_test = df_test[target_column].values.astype(float)

    print(f"Encoded feature dimension: {X_train.shape[1]} features")

    # 2. Train Ridge Regression Model
    print(f"\nTraining Ridge Regression (alpha={alpha}, random_state={random_state})...")
    lr_model = Ridge(alpha=alpha, random_state=random_state)
    lr_model.fit(X_train, y_train)
    train_duration = time.time() - start_time

    # 3. Validation Assessment
    y_val_pred = lr_model.predict(X_val)
    val_mae = float(mean_absolute_error(y_val, y_val_pred))
    val_rmse = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))
    val_r2 = float(r2_score(y_val, y_val_pred))

    print(f"Validation MAE:  {val_mae:.4f} crashes/km")
    print(f"Validation RMSE: {val_rmse:.4f} crashes/km")
    print(f"Validation R2:   {val_r2:.4f}")

    # 4. Serialize Model Artifacts
    model_save_path = os.path.join(artifacts_dir, "linear_regression_model.joblib")
    joblib.dump(lr_model, model_save_path)
    preprocessor.save(artifacts_dir)

    model_config_data = {
        "model_type": "RidgeRegression",
        "library": "scikit-learn",
        "assigned_member": "Anushree",
        "target": target_column,
        "hyperparameters": {
            "alpha": alpha,
            "random_state": random_state,
            "solver": "auto"
        },
        "training_duration_sec": round(train_duration, 4),
        "feature_count": X_train.shape[1],
        "validation_metrics": {
            "val_mae": round(val_mae, 4),
            "val_rmse": round(val_rmse, 4),
            "val_r2": round(val_r2, 4)
        }
    }
    with open(os.path.join(artifacts_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(model_config_data, f, indent=2)

    print(f"Saved Ridge model to: {model_save_path}")
    print(f"Saved preprocessing artifacts to: {artifacts_dir}")
    print("Linear Regression Training Completed Successfully!")

    return lr_model, preprocessor, model_config_data


if __name__ == "__main__":
    train_linear_regression()
