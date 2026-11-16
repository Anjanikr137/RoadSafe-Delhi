"""
models/random_forest/src/preprocessing.py
Zero-leakage data preprocessing pipeline for RoadSafe Delhi Random Forest Classifier.

Assigned Member: Nakshtr
Model: Random Forest Classifier
Target: is_crash_segment (Binary: 1 if segment has >=1 historical crashes, 0 otherwise)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class RoadSafeRFPreprocessor:
    """
    Zero-leakage preprocessor for RoadSafe Delhi Random Forest.
    Fits OneHotEncoder exclusively on training data.
    Random Forest does NOT require feature scaling (no StandardScaler).
    """

    def __init__(self, contract_path=None):
        if contract_path is None:
            contract_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../../../data_final/feature_contract.json")
            )
        self.contract_path = contract_path
        self._load_contract()
        self.encoder = None
        self.feature_names = None

    def _load_contract(self):
        with open(self.contract_path, "r") as f:
            contract = json.load(f)
        self.num_cols = contract["approved_numerical_predictors"]
        self.cat_cols = contract["approved_categorical_predictors"]
        self.forbidden_cols = set(contract["quarantined_leakage_columns"]).union(
            set(contract["target_definitions"].keys())
        )

    def _assert_zero_leakage(self, df):
        used_cols = set(self.num_cols + self.cat_cols)
        leaked = used_cols.intersection(self.forbidden_cols)
        assert len(leaked) == 0, f"FATAL: Quarantined leakage features detected in pipeline: {leaked}"

    def fit(self, df_train):
        self._assert_zero_leakage(df_train)
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.encoder.fit(df_train[self.cat_cols])
        encoded_cat_names = list(self.encoder.get_feature_names_out(self.cat_cols))
        self.feature_names = self.num_cols + encoded_cat_names
        return self

    def transform(self, df):
        self._assert_zero_leakage(df)
        if self.encoder is None:
            raise RuntimeError("Preprocessor must be fitted before calling transform().")
        X_num = df[self.num_cols].values
        X_cat = self.encoder.transform(df[self.cat_cols])
        return np.hstack([X_num, X_cat])

    def fit_transform(self, df_train):
        return self.fit(df_train).transform(df_train)

    def save(self, artifacts_dir):
        os.makedirs(artifacts_dir, exist_ok=True)
        joblib.dump(self.encoder, os.path.join(artifacts_dir, "encoder.joblib"))
        schema = {
            "features": self.feature_names,
            "n_features": len(self.feature_names),
            "numerical_features": self.num_cols,
            "categorical_features": self.cat_cols,
            "scaler_applied": False,
            "scaling_note": "Random Forest is scale-invariant; StandardScaler intentionally omitted."
        }
        with open(os.path.join(artifacts_dir, "feature_list.json"), "w") as f:
            json.dump(schema, f, indent=2)

    def load(self, artifacts_dir):
        encoder_path = os.path.join(artifacts_dir, "encoder.joblib")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Missing encoder artifact: {encoder_path}")
        self.encoder = joblib.load(encoder_path)
        encoded_cat_names = list(self.encoder.get_feature_names_out(self.cat_cols))
        self.feature_names = self.num_cols + encoded_cat_names
        return self
