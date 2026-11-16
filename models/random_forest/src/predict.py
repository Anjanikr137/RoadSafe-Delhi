"""
models/random_forest/src/predict.py
Standalone inference module for RoadSafe Delhi Random Forest Classifier.

Assigned Member: Nakshtr
Model: Random Forest Classifier
Target: is_crash_segment
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import RoadSafeRFPreprocessor


class RandomForestRoadRiskPredictor:
    """
    Production inference class for RoadSafe Delhi Random Forest model.
    Loads serialized artifacts and executes zero-leakage scoring.
    """

    def __init__(self, artifacts_dir=None):
        if artifacts_dir is None:
            artifacts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../artifacts"))
        self.artifacts_dir = artifacts_dir

        # Load trained model
        model_path = os.path.join(artifacts_dir, "random_forest_model.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact not found: {model_path}")
        self.model = joblib.load(model_path)

        # Load preprocessor
        self.preprocessor = RoadSafeRFPreprocessor()
        self.preprocessor.load(artifacts_dir)

        # Load frozen threshold
        thresh_path = os.path.join(artifacts_dir, "threshold.json")
        with open(thresh_path, "r") as f:
            thresh_data = json.load(f)
        self.threshold = float(thresh_data["optimal_threshold"])

        print(f"Loaded Random Forest artifacts from: {artifacts_dir}")
        print(f"Frozen Validation Threshold: {self.threshold:.6f}")

    def predict_proba(self, df):
        """
        Computes crash occurrence probabilities for road segments.
        """
        X = self.preprocessor.transform(df)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, df, threshold=None):
        """
        Predicts binary hazard classification (0 or 1).
        """
        t = self.threshold if threshold is None else threshold
        probs = self.predict_proba(df)
        return (probs >= t).astype(int)

    def predict_detailed(self, df):
        """
        Generates detailed inference DataFrame matching project schema.
        """
        probs = self.predict_proba(df)
        preds = (probs >= self.threshold).astype(int)

        out = pd.DataFrame()
        if "osm_id" in df.columns:
            out["osm_id"] = df["osm_id"].values
        if "is_crash_segment" in df.columns:
            out["is_crash_segment"] = df["is_crash_segment"].values
        out["predicted_probability"] = probs
        out["predicted_class"] = preds
        return out


if __name__ == "__main__":
    # Smoke test on test partition
    test_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../data_final/test.parquet")
    )
    if os.path.exists(test_path):
        df_test = pd.read_parquet(test_path)
        predictor = RandomForestRoadRiskPredictor()
        sample_preds = predictor.predict_detailed(df_test.head(5))
        print("\nSample Detailed Predictions (First 5 records):")
        print(sample_preds)
