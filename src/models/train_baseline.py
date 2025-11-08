import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------
# Step 1

if __name__ == "__main__":
  X_train = pd.read_csv("data/proceseed/train_features.csv")
  X_test = pd.read_csv("data/proceseed/test_features.csv")
  y_train = pd.read_csv("data/proceseed/train_labels.csv").squeeze()  
  y_test = pd.read_csv("data/proceseed/test_labels.csv").squeeze()


