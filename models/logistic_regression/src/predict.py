"""
models/logistic_regression/src/predict.py
Standalone reusable prediction script for RoadSafe Delhi Logistic Regression.
Loads the trained model, fitted preprocessing transformers, and decision threshold
to score new road segments without retraining.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd


class LogisticRoadRiskPredictor:
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.artifacts_dir = os.path.join(model_dir, "artifacts")

        # Load artifacts
        print(f"Loading Logistic Regression artifacts from: {self.artifacts_dir}")
        self.model = joblib.load(os.path.join(self.artifacts_dir, "logistic_regression_model.joblib"))
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
        print(f"Loaded Logistic Regression model with {len(self.feature_names)} features. Decision threshold: {self.threshold:.4f}")

    def predict(self, df):
        """Scores a DataFrame of road segment features."""
        # Validate required columns
        missing = [c for c in self.num_cols + self.cat_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Input data is missing required predictor columns: {missing}")

        X_num = self.scaler.transform(df[self.num_cols])
        X_cat = self.encoder.transform(df[self.cat_cols])
        X = np.hstack([X_num, X_cat])

        probabilities = self.model.predict_proba(X)[:, 1]
        predictions = (probabilities >= self.threshold).astype(int)

        results = df.copy()
        results["predicted_risk_probability"] = np.round(probabilities, 5)
        results["predicted_crash_segment"] = predictions
        results["decision_threshold"] = self.threshold
        return results


def main():
    if len(sys.argv) > 1:
        input_csv = sys.argv[1]
        output_csv = sys.argv[2] if len(sys.argv) > 2 else "predicted_road_risks_lr.csv"
        df = pd.read_csv(input_csv)
    else:
        # Default test run on sample from test partition
        test_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data_final/test.parquet"))
        df = pd.read_parquet(test_path).head(10)
        output_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "../outputs/sample_predictions.csv"))

    predictor = LogisticRoadRiskPredictor()
    results = predictor.predict(df)
    results.to_csv(output_csv, index=False)
    print(f"\nPrediction completed for {len(results)} records. Saved to: {output_csv}")
    print("\nSample Output:")
    display_cols = ['osm_id', 'type', 'length_m', 'predicted_risk_probability', 'predicted_crash_segment']
    avail_cols = [c for c in display_cols if c in results.columns]
    print(results[avail_cols].to_string(index=False))


if __name__ == "__main__":
    main()

