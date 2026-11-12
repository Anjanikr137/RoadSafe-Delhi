"""
models/ann/src/preprocessing.py
Data preprocessing module for RoadSafe Delhi ANN.
Ensures strict zero-leakage:
  - StandardScaler fitted ONLY on training partition.
  - OneHotEncoder fitted ONLY on training partition.
  - Validation and test partitions transformed using the frozen training transformers.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class RoadSafePreprocessor:
    def __init__(self, contract_path=None):
        if contract_path is None:
            # Default relative path to data_final/feature_contract.json
            contract_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data_final/feature_contract.json"))
            if not os.path.exists(contract_path):
                contract_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", "data_final", "feature_contract.json"))
        
        with open(contract_path, "r", encoding="utf-8") as f:
            self.contract = json.load(f)

        self.num_cols = self.contract["approved_numerical_predictors"]
        self.cat_cols = self.contract["approved_categorical_predictors"]
        self.all_approved = self.contract["approved_predictors"]
        self.target_definitions = self.contract["target_definitions"]
        self.quarantined = self.contract["quarantined_leakage_columns"]

        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.feature_names = []
        self.is_fitted = False

    def verify_no_leakage(self, df):
        """Enforces zero-leakage check on candidate feature dataframe."""
        cols = set(df.columns)
        forbidden = set(self.quarantined).union(set(self.target_definitions.keys()))
        leaks = cols.intersection(forbidden)
        assert len(leaks) == 0, f"FATAL TARGET LEAKAGE: Columns {leaks} detected in feature matrix!"

    def fit(self, df_train):
        """Fits transformers ONLY on training partition."""
        # Programmatic check
        features_only = df_train[self.num_cols + self.cat_cols]
        self.verify_no_leakage(features_only)

        self.scaler.fit(df_train[self.num_cols])
        self.encoder.fit(df_train[self.cat_cols])

        cat_names = self.encoder.get_feature_names_out(self.cat_cols).tolist()
        self.feature_names = list(self.num_cols) + cat_names
        self.is_fitted = True
        return self

    def transform(self, df):
        """Transforms a dataset using pre-fitted training transformers."""
        assert self.is_fitted, "Preprocessor must be fitted on training data before transform!"
        features_only = df[self.num_cols + self.cat_cols]
        self.verify_no_leakage(features_only)

        num_scaled = self.scaler.transform(df[self.num_cols])
        cat_encoded = self.encoder.transform(df[self.cat_cols])

        X = np.hstack([num_scaled, cat_encoded])
        return X

    def fit_transform(self, df_train):
        self.fit(df_train)
        return self.transform(df_train)

    def save(self, artifacts_dir):
        os.makedirs(artifacts_dir, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(artifacts_dir, "scaler.joblib"))
        joblib.dump(self.encoder, os.path.join(artifacts_dir, "encoder.joblib"))
        with open(os.path.join(artifacts_dir, "feature_list.json"), "w", encoding="utf-8") as f:
            json.dump({
                "numerical_features": self.num_cols,
                "categorical_features": self.cat_cols,
                "final_feature_names": self.feature_names,
                "feature_count": len(self.feature_names)
            }, f, indent=2)

    def load(self, artifacts_dir):
        self.scaler = joblib.load(os.path.join(artifacts_dir, "scaler.joblib"))
        self.encoder = joblib.load(os.path.join(artifacts_dir, "encoder.joblib"))
        with open(os.path.join(artifacts_dir, "feature_list.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
            self.num_cols = meta["numerical_features"]
            self.cat_cols = meta["categorical_features"]
            self.feature_names = meta["final_feature_names"]
        self.is_fitted = True
        return self

