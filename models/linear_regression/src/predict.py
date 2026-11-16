"""
models/linear_regression/src/predict.py
Standalone inference module for RoadSafe Delhi Linear Regression (Ridge Baseline).

Assigned Member: Anushree
Model: Ridge Regression (alpha=10.0, random_state=42)
Target: crash_density_per_km (Continuous Crash Rate per Kilometer)

Usage:
    from predict import LinearRoadRiskPredictor
    predictor = LinearRoadRiskPredictor()
    results = predictor.predict(df_roads)
"""

import os
import joblib
import pandas as pd
import numpy as np

from preprocessing import RoadSafeLinearPreprocessor


class LinearRoadRiskPredictor:
    def __init__(self, artifacts_dir=None):
        if artifacts_dir is None:
            artifacts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../artifacts"))

        self.artifacts_dir = artifacts_dir
        self.model_path = os.path.join(artifacts_dir, "linear_regression_model.joblib")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model artifact not found at: {self.model_path}")

        print(f"Loading Linear Regression artifacts from: {artifacts_dir}")
        self.model = joblib.load(self.model_path)
        self.preprocessor = RoadSafeLinearPreprocessor().load(artifacts_dir)
        print(f"Loaded Ridge model with {len(self.preprocessor.feature_names)} features.")

    def predict(self, df):
        """Generates continuous crash density predictions for input road segments."""
        X = self.preprocessor.transform(df)
        preds = self.model.predict(X)

        output_df = pd.DataFrame()
        if "osm_id" in df.columns:
            output_df["osm_id"] = df["osm_id"].values
        if "type" in df.columns:
            output_df["type"] = df["type"].values
        if "length_m" in df.columns:
            output_df["length_m"] = df["length_m"].values

        output_df["predicted_crash_density_per_km"] = np.round(preds, 6)

        if "crash_density_per_km" in df.columns:
            output_df["actual_crash_density_per_km"] = np.round(df["crash_density_per_km"].values, 6)
            output_df["residual_error"] = np.round(df["crash_density_per_km"].values - preds, 6)

        return output_df


def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data_final"))
    test_path = os.path.join(data_dir, "test.parquet")

    if not os.path.exists(test_path):
        print(f"Test data not found at {test_path}")
        return

    df_test = pd.read_parquet(test_path)
    predictor = LinearRoadRiskPredictor()

    sample = df_test.head(10)
    results = predictor.predict(sample)

    outputs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../outputs"))
    os.makedirs(outputs_dir, exist_ok=True)
    sample_out_path = os.path.join(outputs_dir, "sample_predictions.csv")
    results.to_csv(sample_out_path, index=False)

    print(f"\nSample Prediction Results (first 10 records saved to {sample_out_path}):")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
