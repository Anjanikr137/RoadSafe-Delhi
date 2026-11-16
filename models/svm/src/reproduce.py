
import joblib
import pandas as pd
import numpy as np

model = joblib.load("svm_outputs/svm_model.joblib")
X_test = pd.read_csv("svm_outputs/test_features.csv")

predictions = model.predict(X_test)

np.savetxt(
    "svm_outputs/reproduced_predictions.csv",
    predictions,
    fmt="%d",
    delimiter=","
)

print("Independent prediction completed!")
print("Total predictions:", len(predictions))
