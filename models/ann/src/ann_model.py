"""
models/ann/src/ann_model.py
Defines the Artificial Neural Network architecture for RoadSafe Delhi.
Architecture:
  - Input Layer: matches approved input feature dimension (23 features)
  - Dense Layer 1: 64 neurons, ReLU activation
  - Dropout: 0.2
  - Dense Layer 2: 32 neurons, ReLU activation
  - Dropout: 0.2
  - Output Layer: 1 neuron, Sigmoid activation (calibrated hazard probability)
"""

import os
# Configure backend and cache before importing keras
os.environ['KERAS_BACKEND'] = 'torch'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
cache_dir = os.path.join(project_root, ".cache")
os.environ['KERAS_HOME'] = os.path.join(cache_dir, "keras")
os.environ['MPLCONFIGDIR'] = os.path.join(cache_dir, "matplotlib")
os.makedirs(os.environ['KERAS_HOME'], exist_ok=True)
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import keras


def build_ann_model(input_dim, learning_rate=0.001):
    """
    Constructs and compiles the Feedforward Artificial Neural Network.
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,), name="road_features_input"),
        keras.layers.Dense(64, activation="relu", name="dense_layer_1"),
        keras.layers.Dropout(0.2, name="dropout_1"),
        keras.layers.Dense(32, activation="relu", name="dense_layer_2"),
        keras.layers.Dropout(0.2, name="dropout_2"),
        keras.layers.Dense(1, activation="sigmoid", name="risk_probability_output")
    ], name="RoadSafe_Delhi_ANN")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.BinaryCrossentropy()

    metrics = [
        "accuracy",
        keras.metrics.AUC(name="roc_auc"),
        keras.metrics.AUC(curve="PR", name="pr_auc")
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

