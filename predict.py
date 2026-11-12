"""
predict.py
Root entrypoint to generate RoadSafe Delhi risk predictions using the trained ANN model.
Usage:
  python predict.py [input_features.csv] [output_predictions.csv]
"""

import os
import sys
import pandas as pd
import importlib.util

ann_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "models", "ann", "src"))
sys.path.insert(0, ann_src_dir)

spec = importlib.util.spec_from_file_location("ann_predict_module", os.path.join(ann_src_dir, "predict.py"))
ann_predict_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ann_predict_mod)
RoadRiskPredictor = ann_predict_mod.RoadRiskPredictor


def main():
    predictor = RoadRiskPredictor(model_dir=os.path.abspath("models/ann"))

    if len(sys.argv) > 1:
        input_csv = sys.argv[1]
        output_csv = sys.argv[2] if len(sys.argv) > 2 else "predicted_road_risks.csv"
        df = pd.read_csv(input_csv)
    else:
        # Default test run on sample from test partition
        test_path = os.path.abspath("data_final/test.parquet")
        df = pd.read_parquet(test_path).head(10)
        output_csv = os.path.abspath("models/ann/outputs/sample_predictions.csv")

    results = predictor.predict(df)
    results.to_csv(output_csv, index=False)
    print(f"\nPrediction completed for {len(results)} records. Saved to: {output_csv}")
    print("\nSample Output:")
    display_cols = ['osm_id', 'type', 'length_m', 'predicted_risk_probability', 'predicted_high_risk']
    avail_cols = [c for c in display_cols if c in results.columns]
    print(results[avail_cols].to_string(index=False))


if __name__ == "__main__":
    main()
