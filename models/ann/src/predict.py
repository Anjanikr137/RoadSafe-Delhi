"""
models/ann/src/predict.py
Standalone reusable prediction script for RoadSafe Delhi ANN.
Loads the trained model, fitted preprocessing transformers, and decision threshold
to score new road segments without retraining.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

# Configure backend and cache before importing keras
os.environ['KERAS_BACKEND'] = 'torch'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
cache_dir = os.path.join(project_root, ".cache")
os.environ['KERAS_HOME'] = os.path.join(cache_dir, "keras")
os.environ['MPLCONFIGDIR'] = os.path.join(cache_dir, "matplotlib")
os.makedirs(os.environ['KERAS_HOME'], exist_ok=True)
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import keras


class RoadRiskPredictor:
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.artifacts_dir = os.path.join(model_dir, "artifacts")

        # Load artifacts
        print(f"Loading ANN artifacts from: {self.artifacts_dir}")
        self.model = keras.models.load_model(os.path.join(self.artifacts_dir, "ann_model.keras"))
        self.scaler = joblib.load(os.path.join(self.artifacts_dir, "scaler.joblib"))
        self.encoder = joblib.load(os.path.join(self.artifacts_dir, "encoder.joblib"))

        with open(os.path.join(self.artifacts_dir, "feature_list.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
            self.num_cols = meta["numerical_features"]
            self.cat_cols = meta["categorical_features"]
            self.feature_names = meta["final_feature_names"]

        with open(os.path.join(self.artifacts_dir, "threshold.json"), "r", encoding="utf-8") as f:
            thresh_meta = json.load(f)
            self.threshold = float(thresh_meta.get("optimal_threshold", 0.50))
        print(f"Loaded ANN model with {len(self.feature_names)} features. Decision threshold: {self.threshold:.4f}")

    def predict(self, df):
        """Scores a DataFrame of road segment features."""
        # Validate required columns
        missing = [c for c in self.num_cols + self.cat_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Input data is missing required predictor columns: {missing}")

        X_num = self.scaler.transform(df[self.num_cols])
        X_cat = self.encoder.transform(df[self.cat_cols])
        X = np.hstack([X_num, X_cat])

        probabilities = self.model.predict(X).flatten()
        predictions = (probabilities >= self.threshold).astype(int)

        results = df.copy()
        results["predicted_risk_probability"] = np.round(probabilities, 5)
        results["predicted_high_risk"] = predictions
        results["decision_threshold"] = self.threshold
        return results


def main():
    if len(sys.argv) > 1:
        input_csv = sys.argv[1]
        output_csv = sys.argv[2] if len(sys.argv) > 2 else "predicted_road_risks.csv"
        df = pd.read_csv(input_csv)
    else:
        # Default test run on sample from test partition
        test_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data_final/test.parquet"))
        df = pd.read_parquet(test_path).head(10)
        output_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "../outputs/sample_predictions.csv"))

    predictor = RoadRiskPredictor()
    results = predictor.predict(df)
    results.to_csv(output_csv, index=False)
    print(f"\nPrediction completed for {len(results)} records. Saved to: {output_csv}")
    print("\nSample Output:")
    display_cols = ['osm_id', 'type', 'length_m', 'predicted_risk_probability', 'predicted_high_risk']
    avail_cols = [c for c in display_cols if c in results.columns]
    print(results[avail_cols].to_string(index=False))


if __name__ == "__main__":
    main()

