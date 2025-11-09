import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# ---------------------------------------------------
# Step 1

def evaluate_holdout(model, X_te, y_te, log_to_mlflow=False) :
  """Full eval: Metrics, curves, report."""
  y_pred = model.predict(X_te)
  y_proba = model.predict_proba(X_te)[:, 1]  # Pos prob
  
  # Core metrics
  acc = accuracy_score(y_te, y_pred)  # From sklearn.metrics
  f1 = f1_score(y_te, y_pred)
  auc = roc_auc_score(y_te, y_proba)
  
  print(f"Accuracy: {acc:.3f}")
  print(f"F1-Score: {f1:.3f}")
  print(f"ROC-AUC: {auc:.3f}")
  print("\nDetailed Report:\n", classification_report(y_te, y_pred))
  
  # Log to MLflow if flagged
  if log_to_mlflow:
    import mlflow
    with mlflow.start_run(run_name="holdout-eval"):
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_f1", f1)
        mlflow.log_metric("test_auc", auc)
  
  # ROC Curve Plot
  fpr, tpr, _ = roc_curve(y_te, y_proba)
  plt.figure(figsize=(8, 6))
  plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
  plt.plot([0, 1], [0, 1], 'k--')  # Diagonal
  plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
  plt.title('ROC Curve (Test Set)')
  plt.legend()
  plt.show()
  
  # Precision-Recall Curve (good for imbalanced data)
  prec, rec, _ = precision_recall_curve(y_te, y_proba)
  plt.figure(figsize=(8, 6))
  plt.plot(rec, prec, label='PR Curve')
  plt.xlabel('Recall'); plt.ylabel('Precision')
  plt.title('Precision-Recall Curve (Test Set)')
  plt.show()
  
  return {'acc': acc, 'f1': f1, 'auc': auc, 'y_pred': y_pred, 'y_proba': y_proba}


if __name__ == "__main__":
  X_test = pd.read_csv("data/proceseed/test_features.csv")
  y_test =pd.read_csv("data/proceseed/test_labels.csv")
  model = joblib.load("data/models/logreg_tuned.pkl")
  print(f"Loaded: X_test {X_test.shape}, y_test {y_test.shape}")
  
  metrics = evaluate_holdout(model, X_test, y_test, log_to_mlflow=True)  # Set False if no MLflow
  print("Metrics dict:", metrics)