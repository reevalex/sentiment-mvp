import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score

# ---------------------------------------------------
# Step 1

def train_baseline(X_tr, y_tr, X_te, y_te, log_to_mlflow=False, run_name="sentiment_baseline"):
  model = LogisticRegression(random_state=123, max_iter=1000)
  
  if log_to_mlflow:
    mlflow.start_run(run_name=run_name)
    mlflow.log_param("model_type", "logistic_regression")
    mlflow.log_param("max_features", X_tr.shape[1])
    mlflow.log_param("train_sampels", len(X_tr))
  
  model.fit(X_tr, y_tr)
  
  if log_to_mlflow:
    mlflow.sklearn.log_model(model, "trained_model") #artifact
  
  if X_te is not None and y_te is not None:
    y_pred = model.predict(X_te)
    y_pred_proba = model.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_te, y_pred)
    print(f"Test Accuracy: {acc:.3f}")
    print(f"\nClassification Report:\n{classification_report(y_te, y_pred)}")

    if log_to_mlflow:    
      mlflow.log_metric("test_accuracy", acc)
      mlflow.log_metric("test_precision", classification_report(y_te, y_pred, output_dict=True)["1"]["precision"])
      mlflow.log_metric("test_recall", classification_report(y_te, y_pred, output_dict=True)["1"]["recall"])
      
      cm = confusion_matrix(y_te, y_pred)
      fig, ax = plt.subplots(figsize=(6, 4))
      sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
      ax.set_title("Confusoin Matrix (Test)")
      ax.set_ylabel("True")
      ax.set_xlabel("Predicted")
      mlflow.log_figure(fig, "confusion_matrix.png")
      plt.close(fig)
    
    if log_to_mlflow:
      mlflow.end_run()

    return model, y_pred, y_pred_proba, acc

  if log_to_mlflow:
    mlflow.end_run()

  return model, None, None, None

def cross_validate(model, X, y, cv_folds=5, log_to_mlflow=False):
  if log_to_mlflow:
    mlflow.start_run(run_name="cross-validation")
    mlflow.log_param("cv_folds", cv_folds)
  
  scores = cross_val_score(model, X, y, cv=cv_folds, scoring="accuracy")
  print(f"CV Scores: {scores}")
  print(f"Mean CV Acc: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
  
  if log_to_mlflow:
    mlflow.log_metric("cv_mean_accuracy", scores.mean())
    mlflow.log_metric("cv_std_accuracy", scores.std())
    mlflow.end_run()
  
  return scores.mean()
  
def tune_hyperparameters(X_tr, y_tr, k=3, log_to_mlflow=False):
  param_grid = {'C': [0.1, 1, 10]}
  
  if log_to_mlflow:
    mlflow.start_run(run_name="tuning-parameters")
    mlflow.log_param("param_grid", param_grid)
    
  grid = GridSearchCV(LogisticRegression(random_state=123, max_iter=1000), 
                      param_grid=param_grid, cv=k, scoring="accuracy") 
  grid.fit(X_tr, y_tr)
  print(f"Best params: {grid.best_params_}")
  print(f"Best CV Score: {grid.best_score_:.3f}")
  
  if log_to_mlflow:
    mlflow.log_param("best_C", grid.best_params_['C'])
    mlflow.log_metric("best_cv_score", grid.best_score_)
    mlflow.sklearn.log_model(grid.best_estimator_, "tuned_model")
    mlflow.end_run()
  
  return grid.best_estimator_
  

if __name__ == "__main__":
  X_train = pd.read_csv("data/proceseed/train_features.csv")
  X_test = pd.read_csv("data/proceseed/test_features.csv")
  y_train = pd.read_csv("data/proceseed/train_labels.csv").squeeze()
  y_test = pd.read_csv("data/proceseed/test_labels.csv").squeeze()
  
  model, y_pred, y_pred_proba, test_acc = train_baseline(X_train, y_train, X_test, y_test, log_to_mlflow=True)
  joblib.dump(model, "data/models/logreg_baseline.pkl")
  print("Model saved to ./data/models/logreg_baseline.pkl")
  
  cv_acc = cross_validate(model, X_train, y_train, log_to_mlflow=True)
  best_model = tune_hyperparameters(X_train, y_train, k=3, log_to_mlflow=True)
  
  joblib.dump(best_model, "data/models/logreg_tuned.pkl")
  print("Tuned Model saved to ./data/models/logreg_tuned.pkl")
  
  y_pred_tuned = best_model.predict(X_test)
  tuned_acc = accuracy_score(y_test, y_pred_tuned)
  print(f"Tuned Test Acc: {tuned_acc:.3f} (vs baseline {test_acc:.3f})")
  
  with mlflow.start_run(run_name="tuned_eval"):
    mlflow.log_metric("tuned_test_acc", tuned_acc)
    
  log_df = pd.DataFrame({
    "model_type": ["baseline", "tuned"],
    "cv_acc": [cv_acc, None],
    "test_acc": [test_acc, tuned_acc],
    "best_params": [None, str(best_model.get_params()['C'])]
  })
  log_df.to_csv("data/models/training_log.csv", index=False)
  print("Metrics also logged to data/models/training_log.csv")
  


