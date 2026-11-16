
# RoadSafe Delhi - SVM Evaluation Report

## Selected Model

- Kernel: rbf
- C: 0.1
- Validation F1: 0.7754
- Validation PR-AUC: 0.8393

## Test Dataset

- Total test samples: 3686
- Positive crash samples: 113
- Negative crash samples: 3573

## Test Performance

Accuracy: 0.9826
Precision: 0.6601
Recall: 0.8938
F1 Score: 0.7594
ROC-AUC: 0.9703
PR-AUC: 0.8563

## Confusion Matrix

```text
[[3521   52]
 [  12  101]]
```

## Classification Report

```text
              precision    recall  f1-score   support

           0       1.00      0.99      0.99      3573
           1       0.66      0.89      0.76       113

    accuracy                           0.98      3686
   macro avg       0.83      0.94      0.88      3686
weighted avg       0.99      0.98      0.98      3686

```

## Conclusion

The selected RBF SVM was evaluated on the
provided independent test partition.

The reported metrics describe performance on
this dataset and should not be interpreted as
a guarantee of future crash prediction.
