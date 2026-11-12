"""
models/ann/src/train.py
Training execution module for RoadSafe Delhi ANN.
Executes training strictly on the training partition, monitors validation performance,
applies early stopping, learning rate scheduling, and saves model checkpoints.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd

# Configure backend and cache before importing keras
os.environ['KERAS_BACKEND'] = 'torch'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
cache_dir = os.path.join(project_root, ".cache")
os.environ['KERAS_HOME'] = os.path.join(cache_dir, "keras")
os.environ['MPLCONFIGDIR'] = os.path.join(cache_dir, "matplotlib")
os.makedirs(os.environ['KERAS_HOME'], exist_ok=True)
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import keras
import torch

from preprocessing import RoadSafePreprocessor
from ann_model import build_ann_model


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_ann(
    data_dir=None,
    artifacts_dir=None,
    target_column="is_crash_segment",
    epochs=40,
    batch_size=64,
    learning_rate=0.001
):
    set_seed(42)

    if data_dir is None:
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data_final"))
    if artifacts_dir is None:
        artifacts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../artifacts"))
    os.makedirs(artifacts_dir, exist_ok=True)

    print("=" * 60)
    print("ROADSAFE DELHI — ARTIFICIAL NEURAL NETWORK TRAINING")
    print("=" * 60)

    # 1. Load Parquet Partitions
    print(f"Loading data partitions from: {data_dir}")
    df_train = pd.read_parquet(os.path.join(data_dir, "train.parquet"))
    df_val = pd.read_parquet(os.path.join(data_dir, "validation.parquet"))
    df_test = pd.read_parquet(os.path.join(data_dir, "test.parquet"))

    print(f"Partition sizes: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
    print(f"Target column: {target_column}")

    # 2. Preprocess features strictly with training parameters
    preprocessor = RoadSafePreprocessor(os.path.join(data_dir, "feature_contract.json"))
    X_train = preprocessor.fit_transform(df_train)
    X_val = preprocessor.transform(df_val)
    X_test = preprocessor.transform(df_test)

    y_train = df_train[target_column].values.astype(np.float32)
    y_val = df_val[target_column].values.astype(np.float32)
    y_test = df_test[target_column].values.astype(np.float32)

    print(f"Input feature count: {X_train.shape[1]}")
    print(f"Train positive class: {int(y_train.sum())} / {len(y_train)} ({y_train.mean()*100:.2f}%)")
    print(f"Val positive class:   {int(y_val.sum())} / {len(y_val)} ({y_val.mean()*100:.2f}%)")
    print(f"Test positive class:  {int(y_test.sum())} / {len(y_test)} ({y_test.mean()*100:.2f}%)")

    # Save preprocessing objects
    preprocessor.save(artifacts_dir)
    print(f"Saved preprocessing objects to {artifacts_dir}")

    # 3. Class Weight calculation for imbalance
    neg_count = float((y_train == 0).sum())
    pos_count = float((y_train == 1).sum())
    class_weights = {0: 1.0, 1: neg_count / max(1.0, pos_count)}
    print(f"Calculated class weighting: {class_weights}")

    # 4. Build Model
    model = build_ann_model(input_dim=X_train.shape[1], learning_rate=learning_rate)
    model.summary()

    # 5. Callbacks
    model_save_path = os.path.join(artifacts_dir, "ann_model.keras")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_pr_auc",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1
        )
    ]

    # 6. Fit Model
    print("\nStarting ANN training...")
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    training_duration_sec = time.time() - start_time
    print(f"\nTraining completed in {training_duration_sec:.2f} seconds.")

    # Save training history
    history_dict = {k: [float(val) for val in v] for k, v in history.history.items()}
    history_dict["training_duration_sec"] = training_duration_sec
    history_dict["epochs_completed"] = len(history.history["loss"])
    with open(os.path.join(artifacts_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2)
    print(f"Saved training history to {os.path.join(artifacts_dir, 'training_history.json')}")

    # Explicitly save model if not already saved by checkpoint
    model.save(model_save_path)
    print(f"Saved best ANN model to {model_save_path}")

    return model, preprocessor, history_dict, (X_train, y_train, X_val, y_val, X_test, y_test, df_test)


if __name__ == "__main__":
    train_ann()

